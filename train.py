import argparse
import json
import logging
import os

import gymnasium as gym
import minari
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from rich.console import Console
from rich.logging import RichHandler

from hyperparameters import hyperparameters
from utils.data_sampler import Data_Sampler
from utils.Stoping import EarlyStopping


def train_agent(dataset, state_dim, action_dim, max_action, device, output_dir, args):
    # 加载数据缓冲区（使用 Minari）
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    logger.info("Loaded buffer")

    if args.algo == "ql":
        import agents.ql_diffusion

        Agent = agents.ql_diffusion.Diffusion_QL
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=args.discount,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
        )
    elif args.algo == "bc":
        import agents.bc_diffusion

        Agent = agents.bc_diffusion.Diffusion_BC
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=args.discount,
            tau=args.tau,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            lr=args.lr,
        )
    elif args.algo == "all":
        import agents.td3bc_diffusion

        Agent = agents.td3bc_diffusion.Diffusion_TD3BC
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=args.discount,
            tau=args.tau,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            lr=args.lr,
        )

    early_stop = False
    stop_check = EarlyStopping(tolerance=1, min_delta=0.0)

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.0
    logger.info("Training Start")

    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(
            data_sampler,
            iterations=iterations,
            batch_size=args.batch_size,
        )
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # 日志输出：用 logger.info 或 logger.info 打印关键信息
        logger.info(f"Train step: {training_iters}")
        logger.info(f"Trained Epochs: {curr_epoch}")
        logger.info(f"BC Loss: {np.mean(loss_metric['bc_loss'])}")
        logger.info(f"QL Loss: {np.mean(loss_metric['ql_loss'])}")
        logger.info(f"Actor Loss: {np.mean(loss_metric['actor_loss'])}")
        logger.info(f"Critic Loss: {np.mean(loss_metric['critic_loss'])}")

        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(
            agent, dataset.env_spec.id, args.seed, eval_episodes=args.eval_episodes
        )
        evaluations.append(
            [
                eval_res,
                eval_res_std,
                eval_norm_res,
                eval_norm_res_std,
                np.mean(loss_metric["bc_loss"]),
                np.mean(loss_metric["ql_loss"]),
                np.mean(loss_metric["actor_loss"]),
                np.mean(loss_metric["critic_loss"]),
                curr_epoch,
            ]
        )
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.info(f"Average Episodic Reward: {eval_res}")
        logger.info(f"Average Episodic N-Reward: {eval_norm_res}")

        bc_loss = np.mean(loss_metric["bc_loss"])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

        if args.record:
            record(
                dataset.env_spec.id,
                agent,
                os.path.join(results_dir, "video"),
            )

    # 模型选择：online 或 offline
    scores = np.array(evaluations)
    if args.ms == "online":
        best_id = np.argmax(scores[:, 2])
        best_res = {
            "model selection": args.ms,
            "epoch": scores[best_id, -1],
            "best normalized score avg": scores[best_id, 2],
            "best normalized score std": scores[best_id, 3],
            "best raw score avg": scores[best_id, 0],
            "best raw score std": scores[best_id, 1],
        }
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), "w") as f:
            f.write(json.dumps(best_res))
    elif args.ms == "offline":
        bc_loss = scores[:, 4]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {
            "model selection": args.ms,
            "epoch": scores[where_k][0][-1],
            "best normalized score avg": scores[where_k][0][2],
            "best normalized score std": scores[where_k][0][3],
            "best raw score avg": scores[where_k][0][0],
            "best raw score std": scores[where_k][0][1],
        }

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), "w") as f:
            f.write(json.dumps(best_res))


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


# 评估策略：在固定种子下运行若干个 episode 并返回平均 reward
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    state, _ = eval_env.reset(seed=seed + 100)
    eval_env.action_space.seed(seed + 100)
    eval_env.observation_space.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.0
        state, _ = eval_env.reset()
        while True:
            action = policy.sample_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            traj_return += reward
            if terminated or truncated:
                break
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    # 解包，尝试获取原始环境
    base_env = unwrap_env(eval_env)

    if hasattr(base_env, "get_normalized_score"):
        normalized_scores = [base_env.get_normalized_score(s) for s in scores]
        avg_norm_score = base_env.get_normalized_score(avg_reward)
        std_norm_score = np.std(normalized_scores)
    else:
        normalized_scores = scores  # 不归一化
        avg_norm_score = avg_reward
        std_norm_score = std_reward

    logger.info(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}"
    )
    return avg_reward, std_reward, avg_norm_score, std_norm_score


def record(env_name, policy, output_dir, num_eval_episodes=4):
    if not os.path.exists(os.path.join(output_dir, "video")):
        os.makedirs(os.path.join(output_dir, "video"))
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=output_dir,
        name_prefix="eval",
        episode_trigger=lambda _: True,
    )
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    for episode_num in range(num_eval_episodes):
        obs, __ = env.reset()
        episode_over = False
        while not episode_over:
            action = policy.sample_action(np.array(obs))
            obs, _, terminated, truncated, __ = env.step(action)

            episode_over = terminated or truncated
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### 实验设置 ###
    parser.add_argument("--exp", default="exp_1", type=str)  # 实验ID
    parser.add_argument("--device", default=0, type=int)  # 设备编号：0，1，...
    parser.add_argument(
        "--env_name", default="walker2d-medium-expert-v2", type=str
    )  # 环境名称
    parser.add_argument("--dir", default="results", type=str)  # 日志保存目录
    parser.add_argument(
        "--seed", default=0, type=int
    )  # 设置 Gym、PyTorch 和 Numpy 的随机种子
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### 优化设置 ###
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--save_best_model", action="store_true")

    ### RL 参数 ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion 设置 ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default="vp", type=str)
    ### 算法选择 ###
    parser.add_argument("--algo", default="bc", type=str)  # ['bc', 'ql']
    parser.add_argument(
        "--ms", default="offline", type=str, help="['online', 'offline']"
    )
    parser.add_argument("--record", action="store_true", default=True)
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f"{args.dir}"

    args.num_epochs = hyperparameters[args.env_name]["num_epochs"]
    args.eval_freq = hyperparameters[args.env_name]["eval_freq"]
    args.eval_episodes = 10 if "v2" in args.env_name else 100

    args.lr = hyperparameters[args.env_name]["lr"]
    args.eta = hyperparameters[args.env_name]["eta"]
    args.max_q_backup = hyperparameters[args.env_name]["max_q_backup"]
    args.reward_tune = hyperparameters[args.env_name]["reward_tune"]
    args.gn = hyperparameters[args.env_name]["gn"]
    args.top_k = hyperparameters[args.env_name]["top_k"]

    # 设置日志输出目录
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay:
        file_name += "|lr_decay"
    file_name += f"|ms-{args.ms}"

    if args.ms == "offline":
        file_name += f"|k-{args.top_k}"
    file_name += f"|{args.seed}"

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    variant = vars(args)
    variant.update(version="Diffusion-Policies-RL")

    # 日志输出到文件（纯文本，不带颜色）
    file_handler = logging.FileHandler(os.path.join(results_dir, "log.txt"), mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # RichHandler 输出到终端（带颜色）
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setLevel(logging.INFO)

    # Rich logger 初始化，使用 RichHandler 替代默认格式
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler, file_handler],
    )
    logger = logging.getLogger("rich")
    console = Console()

    logger.info(f"Saving location: {results_dir}")

    # === 使用 Minari 加载数据集 ===
    dataset = minari.load_dataset(args.env_name, download=True)
    env_name = dataset.env_spec.id  # 获取对应的 Gymnasium 环境名
    env = gym.make(env_name)

    # === 设置种子 ===
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # === 获取维度信息 ===
    assert isinstance(env.observation_space, gym.spaces.Box), (
        "Only supports Box observation space"
    )
    assert env.observation_space.shape is not None, "Observation space shape is None"
    state_dim = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Box), (
        "Only supports Box action space"
    )
    assert env.action_space.shape is not None, "Action space shape is None"
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # === 日志和参数记录 ===
    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    logger.info(f"Env: {env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    # === 开始训练 ===
    train_agent(
        dataset, state_dim, action_dim, max_action, args.device, results_dir, args
    )
