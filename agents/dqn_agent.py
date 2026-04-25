"""Simple DQN agent for Minesweeper (tabular replacement baseline)."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Action = Tuple[int, int]
Transition = Tuple[np.ndarray, Action, float, np.ndarray, bool]


class MLPDQN(nn.Module):
    """Small MLP that maps flattened state -> Q-value for each board cell."""

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


class CNNDQN(nn.Module):
    """Small CNN that maps 2-channel board state -> Q-value for each board cell."""

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)  # [B, 1, rows, cols]
        return out.view(x.shape[0], self.rows * self.cols)


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
        model_type: Literal["mlp", "cnn"] = "mlp",
    ) -> None:
        """Initialize DQN components."""
        self.rows = rows
        self.cols = cols
        self.num_actions = rows * cols
        self.state_dim = 2 * rows * cols
        self.model_type = model_type.lower()
        if self.model_type not in {"mlp", "cnn"}:
            raise ValueError("model_type must be 'mlp' or 'cnn'")

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

        if self.model_type == "mlp":
            self.policy_net: nn.Module = MLPDQN(self.state_dim, self.num_actions).to(self.device)
            self.target_net: nn.Module = MLPDQN(self.state_dim, self.num_actions).to(self.device)
        else:
            self.policy_net = CNNDQN(self.rows, self.cols).to(self.device)
            self.target_net = CNNDQN(self.rows, self.cols).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory: Deque[Transition] = deque(maxlen=memory_size)
        self.learn_steps = 0

    def _encode_observation(self, observation: np.ndarray) -> np.ndarray:
        """Build a 2-channel board state.

        Channel 1 (hidden mask):
        - 1.0 where cell is hidden (observation == -1)
        - 0.0 where cell is revealed

        Channel 2 (revealed normalized values):
        - observation / 8.0 for revealed cells (0.0 .. 1.0)
        - 0.0 for hidden cells
        """
        obs = observation.astype(np.float32)
        hidden_mask = (obs == -1.0).astype(np.float32)
        revealed_values = np.where(obs >= 0.0, obs / 8.0, 0.0).astype(np.float32)
        return np.stack([hidden_mask, revealed_values], axis=0)

    def _network_forward(self, network: nn.Module, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass helper for either MLP or CNN model type."""
        if self.model_type == "mlp":
            return network(state_batch.view(state_batch.shape[0], -1))
        return network(state_batch)

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
            self._encode_observation(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self._network_forward(self.policy_net, state).squeeze(0).cpu().numpy()

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
                self._encode_observation(observation).copy(),
                action,
                float(reward),
                self._encode_observation(next_observation).copy(),
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

        q_pred = self._network_forward(self.policy_net, states).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = self._network_forward(self.target_net, next_states)
            # Valid next actions come from channel 1 (hidden mask).
            valid_next_mask = next_states[:, 0, :, :].reshape(next_states.shape[0], -1) > 0.5
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
