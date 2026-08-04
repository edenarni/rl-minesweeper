"""Random baseline agent for Minesweeper."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class RandomAgent:
    """A simple baseline agent that picks a random hidden cell."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the agent's RNG.

        Args:
            seed: Optional seed for reproducible random actions.
        """
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray) -> Tuple[int, int]:
        """Select a random action from currently hidden cells.

        Args:
            observation: Visible board where -1 means hidden.

        Returns:
            (row, col) action tuple.
        """
        candidates: List[Tuple[int, int]] = list(map(tuple, np.argwhere(observation == -1)))
        if not candidates:
            raise ValueError("No valid actions available: no hidden cells remain.")

        index = int(self.rng.integers(0, len(candidates)))
        row, col = candidates[index]
        return int(row), int(col)
