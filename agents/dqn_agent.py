"""Simple DQN agent for Minesweeper (tabular replacement baseline)."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Action = Tuple[int, int]
Transition = Tuple[np.ndarray, Action, float, np.ndarray, bool]


class DQN(nn.Module):
    """Small MLP that maps board state -> Q-value for each board cell."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Basic DQN agent with replay buffer, epsilon-greedy, and target network."""

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 20000,
        target_update_every: int = 200,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize DQN components."""
        self.rows = rows
        self.cols = cols
        self.num_actions = rows * cols
        self.state_dim = rows * cols

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = DQN(self.state_dim, self.num_actions).to(self.device)
        self.target_net = DQN(self.state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory: Deque[Transition] = deque(maxlen=memory_size)
        self.learn_steps = 0

    def _flatten_observation(self, observation: np.ndarray) -> np.ndarray:
        """Flatten board observation into 1D float array."""
        return observation.astype(np.float32).reshape(-1)

    def _action_to_index(self, action: Action) -> int:
        row, col = action
        return row * self.cols + col

    def _index_to_action(self, index: int) -> Action:
        row = index // self.cols
        col = index % self.cols
        return int(row), int(col)

    def get_valid_actions(self, observation: np.ndarray) -> List[Action]:
        """Return actions for currently hidden cells only."""
        hidden_positions = np.argwhere(observation == -1)
        return [(int(r), int(c)) for r, c in hidden_positions]

    def select_action(self, observation: np.ndarray, epsilon: float | None = None) -> Action:
        """Select an action with epsilon-greedy policy over valid actions."""
        valid_actions = self.get_valid_actions(observation)
        if not valid_actions:
            raise ValueError("No valid actions available.")

        eps = self.epsilon if epsilon is None else epsilon

        if self.rng.random() < eps:
            return valid_actions[int(self.rng.integers(0, len(valid_actions)))]

        state = torch.tensor(
            self._flatten_observation(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze(0).cpu().numpy()

        # Mask out invalid actions so they are never selected greedily.
        valid_indices = [self._action_to_index(a) for a in valid_actions]
        masked_q = np.full_like(q_values, fill_value=-1e9)
        masked_q[valid_indices] = q_values[valid_indices]

        max_q = np.max(masked_q)
        best_indices = np.where(masked_q == max_q)[0]
        chosen_index = int(best_indices[int(self.rng.integers(0, len(best_indices)))])
        return self._index_to_action(chosen_index)

    def remember(
        self,
        observation: np.ndarray,
        action: Action,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition in replay memory."""
        self.memory.append(
            (
                self._flatten_observation(observation).copy(),
                action,
                float(reward),
                self._flatten_observation(next_observation).copy(),
                bool(done),
            )
        )

    def train_step(self) -> float | None:
        """Sample a random batch from replay and update the policy network."""
        if len(self.memory) < self.batch_size:
            return None

        indices = self.rng.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[int(i)] for i in indices]

        states = torch.tensor(
            np.stack([item[0] for item in batch]), dtype=torch.float32, device=self.device
        )
        action_indices = torch.tensor(
            [self._action_to_index(item[1]) for item in batch], dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            np.stack([item[3] for item in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=self.device)

        q_pred = self.policy_net(states).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            valid_next_mask = next_states == -1.0
            masked_next_q = next_q_all.masked_fill(~valid_next_mask, -1e9)
            has_valid_actions = valid_next_mask.any(dim=1)
            next_max_q = torch.where(
                has_valid_actions,
                masked_next_q.max(dim=1).values,
                torch.zeros_like(rewards),
            )
            targets = rewards + self.gamma * next_max_q * (1.0 - dones)

        loss = self.loss_fn(q_pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def update(
        self,
        observation: np.ndarray,
        action: Action,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> float | None:
        """Store transition and run one gradient update (if replay is ready)."""
        self.remember(observation, action, reward, next_observation, done)
        return self.train_step()

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
