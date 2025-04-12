import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.diffusion import Diffusion
from agents.model import MLP


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))


class Diffusion_TD3BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        beta_schedule="linear",
        n_timesteps=100,
        lr=3e-4,
        alpha=2.5,
    ):
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(
            state_dim, action_dim, self.model, max_action, beta_schedule, n_timesteps
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Q networks
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device
        self.alpha = alpha  # weighting between BC loss and Q value

    def train(self, replay_buffer, iterations, batch_size=256, log_writer=None):
        metric = {"bc_loss": [], "actor_loss": [], "critic_loss": []}

        for _ in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size
            )

            with torch.no_grad():
                next_action = self.actor.sample(next_state)
                target_q1 = self.q1_target(next_state, next_action)
                target_q2 = self.q2_target(next_state, next_action)
                target_q = reward + not_done * self.discount * torch.min(
                    target_q1, target_q2
                )

            # Q loss
            current_q1 = self.q1(state, action)
            current_q2 = self.q2(state, action)
            q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            # Actor loss = TD3-BC
            pi = self.actor.sample(state)
            q_val = self.q1(state, pi)

            bc_loss = self.actor.loss(
                action, state
            )  # diffusion-based behavior cloning loss
            actor_loss = self.alpha * bc_loss - q_val.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak update
            for param, target_param in zip(
                self.q1.parameters(), self.q1_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.q2.parameters(), self.q2_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            metric["bc_loss"].append(bc_loss.item())
            metric["critic_loss"].append(q_loss.item())
            metric["actor_loss"].append(actor_loss.item())

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{id}.pth")
            torch.save(self.q1.state_dict(), f"{dir}/q1_{id}.pth")
            torch.save(self.q2.state_dict(), f"{dir}/q2_{id}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pth")
            torch.save(self.q1.state_dict(), f"{dir}/q1.pth")
            torch.save(self.q2.state_dict(), f"{dir}/q2.pth")

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{id}.pth"))
            self.q1.load_state_dict(torch.load(f"{dir}/q1_{id}.pth"))
            self.q2.load_state_dict(torch.load(f"{dir}/q2_{id}.pth"))
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pth"))
            self.q1.load_state_dict(torch.load(f"{dir}/q1.pth"))
            self.q2.load_state_dict(torch.load(f"{dir}/q2.pth"))
