"""Simple tabular Q-learning agent for Minesweeper."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


StateKey = Tuple[int, ...]
Action = Tuple[int, int]


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy action selection."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ) -> None:
        """Initialize Q-learning hyperparameters and storage.

        Args:
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Initial exploration rate.
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Multiplicative epsilon decay after each episode.
            seed: Optional random seed for action sampling.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-table: maps (state_key, action) -> q_value
        self.q_values: Dict[Tuple[StateKey, Action], float] = {}

    def observation_to_state_key(self, observation: np.ndarray) -> StateKey:
        """Convert a board observation into a hashable state key."""
        return tuple(int(x) for x in observation.flatten())

    def get_available_actions(self, observation: np.ndarray) -> List[Action]:
        """Return unrevealed cells as currently valid actions."""
        hidden_positions = np.argwhere(observation == -1)
        return [(int(r), int(c)) for r, c in hidden_positions]

    def get_q_value(self, state_key: StateKey, action: Action) -> float:
        """Read Q(state, action), defaulting to 0.0 for unseen pairs."""
        return self.q_values.get((state_key, action), 0.0)

    def select_action(self, observation: np.ndarray) -> Action:
        """Choose an action via epsilon-greedy exploration."""
        actions = self.get_available_actions(observation)
        if not actions:
            raise ValueError("No valid actions available.")

        if self.rng.random() < self.epsilon:
            return actions[int(self.rng.integers(0, len(actions)))]

        state_key = self.observation_to_state_key(observation)
        q_values = np.array([self.get_q_value(state_key, action) for action in actions])
        max_q = np.max(q_values)

        # Break ties randomly to avoid always picking the first max action.
        best_indices = np.where(q_values == max_q)[0]
        chosen_index = int(best_indices[int(self.rng.integers(0, len(best_indices)))])
        return actions[chosen_index]

    def update(
        self,
        observation: np.ndarray,
        action: Action,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Apply one tabular Q-learning update."""
        state_key = self.observation_to_state_key(observation)
        next_state_key = self.observation_to_state_key(next_observation)

        current_q = self.get_q_value(state_key, action)

        if done:
            target = reward
        else:
            next_actions = self.get_available_actions(next_observation)
            if next_actions:
                next_max_q = max(self.get_q_value(next_state_key, a) for a in next_actions)
            else:
                next_max_q = 0.0
            target = reward + self.gamma * next_max_q

        updated_q = current_q + self.alpha * (target - current_q)
        self.q_values[(state_key, action)] = updated_q

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
