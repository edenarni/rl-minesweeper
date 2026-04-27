"""Simple DQN agent for Minesweeper (tabular replacement baseline)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


Action = Tuple[int, int]
Transition = Tuple[np.ndarray, Action, float, np.ndarray, bool]


@dataclass
class ReplayBatch:
    """Container for replay samples plus sampling metadata."""

    indices: np.ndarray
    transitions: list[Transition]
    weights: np.ndarray


class UniformReplayBuffer:
    """Simple uniform replay buffer."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> ReplayBatch:
        indices = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        transitions = [self.memory[int(i)] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return ReplayBatch(indices=np.asarray(indices, dtype=np.int64), transitions=transitions, weights=weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        # Uniform replay ignores priority updates.
        _ = indices, priorities

    def __len__(self) -> int:
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Prioritized replay buffer using simple arrays and O(n) sampling."""

    def __init__(
        self,
        capacity: int,
        rng: np.random.Generator,
        alpha: float,
        beta_start: float,
        beta_end: float,
        priority_epsilon: float,
    ) -> None:
        self.capacity = capacity
        self.rng = rng
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.priority_epsilon = priority_epsilon

        self.memory: list[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, transition: Transition) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> ReplayBatch:
        current_size = len(self.memory)
        scaled_priorities = self.priorities[:current_size] ** self.alpha
        probability_sum = float(np.sum(scaled_priorities))
        if probability_sum <= 0.0:
            probabilities = np.full(current_size, 1.0 / current_size, dtype=np.float32)
        else:
            probabilities = scaled_priorities / probability_sum

        indices = self.rng.choice(current_size, size=batch_size, replace=False, p=probabilities)
        transitions = [self.memory[int(i)] for i in indices]

        sample_probs = probabilities[indices]
        weights = (current_size * sample_probs) ** (-beta)
        weights = weights / np.max(weights)
        return ReplayBatch(
            indices=np.asarray(indices, dtype=np.int64),
            transitions=transitions,
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            adjusted = float(abs(priority) + self.priority_epsilon)
            self.priorities[int(idx)] = adjusted
            self.max_priority = max(self.max_priority, adjusted)

    def __len__(self) -> int:
        return len(self.memory)


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


class DeepCNNDQN(nn.Module):
    """Slightly deeper CNN variant with one extra spatial convolution."""

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
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
        model_type: Literal["mlp", "cnn", "cnn_deep"] = "mlp",
        replay_type: Literal["uniform", "prioritized"] = "uniform",
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        priority_epsilon: float = 1e-5,
    ) -> None:
        """Initialize DQN components."""
        self.rows = rows
        self.cols = cols
        self.num_actions = rows * cols
        self.state_dim = 2 * rows * cols
        self.model_type = model_type.lower()
        if self.model_type not in {"mlp", "cnn", "cnn_deep"}:
            raise ValueError("model_type must be 'mlp', 'cnn', or 'cnn_deep'")
        self.replay_type = replay_type.lower()
        if self.replay_type not in {"uniform", "prioritized"}:
            raise ValueError("replay_type must be 'uniform' or 'prioritized'")

        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_every = target_update_every
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.priority_epsilon = priority_epsilon

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
        elif self.model_type == "cnn":
            self.policy_net = CNNDQN(self.rows, self.cols).to(self.device)
            self.target_net = CNNDQN(self.rows, self.cols).to(self.device)
        else:
            self.policy_net = DeepCNNDQN(self.rows, self.cols).to(self.device)
            self.target_net = DeepCNNDQN(self.rows, self.cols).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_loss_fn = nn.MSELoss(reduction="none")
        if self.replay_type == "prioritized":
            self.memory: UniformReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
                capacity=memory_size,
                rng=self.rng,
                alpha=alpha,
                beta_start=beta_start,
                beta_end=beta_end,
                priority_epsilon=priority_epsilon,
            )
        else:
            self.memory = UniformReplayBuffer(capacity=memory_size, rng=self.rng)
        self.learn_steps = 0

    def _checkpoint_config(self) -> dict[str, Any]:
        """Return the model/training settings needed to rebuild this agent."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "memory_size": self.memory_size,
            "target_update_every": self.target_update_every,
            "model_type": self.model_type,
            "replay_type": self.replay_type,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "priority_epsilon": self.priority_epsilon,
        }

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Save model weights and config so the agent can be reused without retraining."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self._checkpoint_config(),
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
                "metadata": metadata or {},
            },
            checkpoint_path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str | None = None) -> "DQNAgent":
        """Load a saved DQN agent checkpoint."""
        checkpoint = torch.load(Path(path), map_location=device or "cpu")
        config = dict(checkpoint["config"])
        config["device"] = device

        agent = cls(**config)
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["policy_state_dict"]))
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = float(checkpoint.get("epsilon", config.get("epsilon", 0.0)))
        agent.learn_steps = int(checkpoint.get("learn_steps", 0))
        agent.policy_net.eval()
        agent.target_net.eval()
        return agent

    def _current_beta(self) -> float:
        """Linearly anneal importance-sampling beta during training."""
        if self.replay_type != "prioritized":
            return 1.0
        anneal_fraction = min(1.0, self.learn_steps / max(1, self.memory_size))
        return self.beta_start + anneal_fraction * (self.beta_end - self.beta_start)

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
        transition = (
            self._encode_observation(observation).copy(),
            action,
            float(reward),
            self._encode_observation(next_observation).copy(),
            bool(done),
        )
        self.memory.add(transition)

    def train_step(self) -> float | None:
        """Sample a random batch from replay and update the policy network."""
        if len(self.memory) < self.batch_size:
            return None

        if self.replay_type == "prioritized":
            replay_batch = self.memory.sample(self.batch_size, beta=self._current_beta())
        else:
            replay_batch = self.memory.sample(self.batch_size)
        batch = replay_batch.transitions

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
        weights = torch.tensor(replay_batch.weights, dtype=torch.float32, device=self.device)

        q_pred = self._network_forward(self.policy_net, states).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Valid next actions come from channel 1 (hidden mask).
            valid_next_mask = next_states[:, 0, :, :].reshape(next_states.shape[0], -1) > 0.5
            has_valid_actions = valid_next_mask.any(dim=1)

            # Double DQN:
            # 1. policy_net chooses the best valid next action
            # 2. target_net evaluates that chosen action
            next_q_policy = self._network_forward(self.policy_net, next_states)
            masked_policy_q = next_q_policy.masked_fill(~valid_next_mask, -1e9)
            next_action_indices = masked_policy_q.argmax(dim=1)

            next_q_target = self._network_forward(self.target_net, next_states)
            next_chosen_q = next_q_target.gather(1, next_action_indices.unsqueeze(1)).squeeze(1)
            next_max_q = torch.where(has_valid_actions, next_chosen_q, torch.zeros_like(rewards))

            targets = rewards + self.gamma * next_max_q * (1.0 - dones)
            td_errors = targets - q_pred.detach()

        per_sample_loss = self.replay_loss_fn(q_pred, targets)
        loss = torch.mean(per_sample_loss * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.replay_type == "prioritized":
            self.memory.update_priorities(replay_batch.indices, td_errors.abs().cpu().numpy())

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
