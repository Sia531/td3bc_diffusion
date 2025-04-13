import numpy as np
import torch


class Data_Sampler(object):
    def __init__(self, dataset, device, reward_tune="no"):
        self.device = device

        all_obs, all_actions, all_next_obs, all_rewards, all_dones = [], [], [], [], []

        for episode in dataset.iterate_episodes():
            obs = episode.observations
            actions = episode.actions
            rewards = episode.rewards
            dones = episode.terminations

            # 构造 next_obs（t+1）
            next_obs = obs[1:]
            obs = obs[:-1]
            actions = actions[:-1]
            rewards = rewards[:-1]
            dones = dones[:-1]

            all_obs.append(obs)
            all_actions.append(actions)
            all_next_obs.append(next_obs)
            all_rewards.append(rewards)
            all_dones.append(dones)

        obs = np.concatenate(all_obs, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        next_obs = np.concatenate(all_next_obs, axis=0)
        rewards = np.concatenate(all_rewards, axis=0).reshape(-1, 1)
        dones = np.concatenate(all_dones, axis=0).reshape(-1, 1)

        self.state = torch.from_numpy(obs).float()
        self.action = torch.from_numpy(actions).float()
        self.next_state = torch.from_numpy(next_obs).float()
        self.not_done = 1.0 - torch.from_numpy(dones).float()

        if reward_tune == "normalize":
            rewards = (rewards - rewards.mean()) / rewards.std()
        elif reward_tune == "iql_antmaze":
            rewards = rewards - 1.0
        elif reward_tune == "iql_locomotion":
            rewards = self.iql_normalize(rewards, self.not_done)
        elif reward_tune == "cql_antmaze":
            rewards = (rewards - 0.5) * 4.0
        elif reward_tune == "antmaze":
            rewards = (rewards - 0.25) * 2.0
        self.reward = torch.from_numpy(rewards).float()

        # 修复维度不一致导致的越界采样
        min_len = min(
            self.state.shape[0],
            self.action.shape[0],
            self.next_state.shape[0],
            self.reward.shape[0],
            self.not_done.shape[0],
        )

        self.state = self.state[:min_len]
        self.action = self.action[:min_len]
        self.next_state = self.next_state[:min_len]
        self.reward = self.reward[:min_len]
        self.not_done = self.not_done[:min_len]

        self.size = min_len  # 更新 size

        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))
        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device),
        )

    def iql_normalize(self, reward, not_done):
        trajs_rt = []
        episode_return = 0.0
        for i in range(len(reward)):
            episode_return += reward[i]
            if not not_done[i]:
                trajs_rt.append(episode_return)
                episode_return = 0.0
        rt_max, rt_min = (
            torch.max(torch.tensor(trajs_rt)),
            torch.min(torch.tensor(trajs_rt)),
        )
        reward /= rt_max - rt_min
        reward *= 1000.0
        return reward
