"""
SecFedRL: A2C + Generalized Advantage Estimation (GAE)
Local RL training for each agent in the P2P federated setting.

Hyperparameters (Table 2 of paper):
  gamma     = 0.99   (discount factor)
  gae_lambda= 0.95   (GAE smoothing parameter)
  c_v       = 0.5    (critic loss weight)
  c_e       = 0.01   (entropy coefficient)
  lr        = 3e-4   (Adam learning rate)
  T_local   = 128-512 steps depending on environment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)
        # For continuous action spaces (Pendulum-v1) add log-std parameter
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        value = self.critic_head(h).squeeze(-1)
        logits = self.actor_head(h)
        return logits, value

    def get_dist(self, x: torch.Tensor):
        logits, value = self.forward(x)
        if self.continuous:
            std = self.log_std.exp().expand_as(logits)
            dist = torch.distributions.Normal(logits, std)
        else:
            dist = torch.distributions.Categorical(logits=logits)
        return dist, value


class A2CAgent:
    """
    Single agent running A2C with GAE.

    Implements Eq. (1) from the paper:
      L(θ) = L_π(θ) + c_v · L_V(θ) − c_e · H(π(·|s))
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        continuous: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        c_v: float = 0.5,
        c_e: float = 0.01,
        t_local: int = 256,
        hidden: int = 128,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.c_v = c_v
        self.c_e = c_e
        self.t_local = t_local
        self.continuous = continuous
        self.device = torch.device(device)

        self.net = ActorCritic(obs_dim, act_dim, hidden, continuous).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Replay buffer for virtual trajectory generation (§4.4)
        self.replay_states: List[np.ndarray] = []
        self.replay_buffer_size = 2000

    def collect_trajectory(self, env) -> Tuple[float, int, List[np.ndarray]]:
        """
        Run one episode (up to t_local steps), collect (s, a, r, v, logp).
        Returns (episode_reward, episode_steps, states_visited).
        """
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0

        for _ in range(self.t_local):
            obs_t = torch.FloatTensor(obs).to(self.device)
            dist, val = self.net.get_dist(obs_t.unsqueeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1) if self.continuous else dist.log_prob(action)

            act_np = action.cpu().numpy()[0] if self.continuous else int(action.item())
            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            values.append(val.item())
            log_probs.append(log_prob)
            dones.append(done)

            ep_reward += reward
            step += 1
            obs = next_obs

            # Store state in replay buffer for virtual trajectories
            self.replay_states.append(obs.copy())
            if len(self.replay_states) > self.replay_buffer_size:
                self.replay_states.pop(0)

            if done:
                obs, _ = env.reset()

        # Bootstrap value for last state
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(self.device)
            _, last_val = self.net.get_dist(obs_t.unsqueeze(0))
            last_val = last_val.item()

        advantages = self._compute_gae(rewards, values, dones, last_val)
        returns = [adv + val for adv, val in zip(advantages, values)]

        loss = self._update(states, actions, log_probs, advantages, returns)

        return ep_reward, step, states

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
    ) -> List[float]:
        """Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0.0
        next_value = last_value

        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + self.gamma * next_value * (1 - float(d)) - v
            gae = delta + self.gamma * self.gae_lambda * (1 - float(d)) * gae
            advantages.insert(0, gae)
            next_value = v

        return advantages

    def _update(self, states, actions, log_probs_old, advantages, returns) -> float:
        """One A2C gradient step — Eq. (1)."""
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dist, values_pred = self.net.get_dist(states_t)

        if self.continuous:
            actions_t = torch.stack(actions).to(self.device)
            log_probs = dist.log_prob(actions_t).sum(-1)
        else:
            actions_t = torch.stack(actions).to(self.device)
            log_probs = dist.log_prob(actions_t)

        # Policy gradient loss
        l_pi = -(log_probs * advantages_t).mean()
        # Value loss
        l_v = ((values_pred - returns_t) ** 2).mean()
        # Entropy bonus
        entropy = dist.entropy().mean() if not self.continuous else dist.entropy().sum(-1).mean()

        loss = l_pi + self.c_v * l_v - self.c_e * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.optimizer.step()

        return loss.item()

    def sample_replay_states(self, n: int) -> np.ndarray:
        """Sample n states from the replay buffer for virtual trajectory evaluation."""
        if len(self.replay_states) < n:
            # Pad with zero states if buffer not yet full
            pad = [np.zeros_like(self.replay_states[0])] * (n - len(self.replay_states))
            return np.array(pad + self.replay_states)
        idx = np.random.choice(len(self.replay_states), n, replace=False)
        return np.array([self.replay_states[i] for i in idx])

    def get_state_dict(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
