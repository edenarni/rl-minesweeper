"""Minesweeper environment for a future RL agent.

This module intentionally keeps the API simple:
- reset() starts a new game
- step(action) applies one action and returns (observation, reward, done, info)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Literal, Tuple

import numpy as np

RewardMode = Literal["classic", "progress", "frontier"]


class MinesweeperEnv:
    """A small Minesweeper environment with NumPy boards."""

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        num_mines: int = 3,
        seed: int | None = None,
        reward_mode: RewardMode = "classic",
        frontier_bonus: float = 0.5,
    ) -> None:
        """Initialize environment configuration and random generator.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            num_mines: Number of mines to place on the hidden board.
            seed: Optional random seed for deterministic behavior.
            reward_mode: "classic" keeps the original rewards; "progress" rewards
                the number of newly revealed safe cells; "frontier" keeps classic
                rewards and adds a small bonus for safe moves next to revealed
                numbered cells.
            frontier_bonus: Extra reward for safe frontier moves when using
                reward_mode="frontier".
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        if num_mines < 1:
            raise ValueError("num_mines must be >= 1")
        if num_mines >= rows * cols:
            raise ValueError("num_mines must be less than rows * cols")

        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.seed = seed
        if reward_mode not in {"classic", "progress", "frontier"}:
            raise ValueError("reward_mode must be 'classic', 'progress', or 'frontier'")
        self.reward_mode = reward_mode
        self.frontier_bonus = frontier_bonus
        self.rng = np.random.default_rng(seed)

        # Hidden board: -1 = mine, 0-8 = count of neighboring mines.
        self.hidden_board = np.zeros((self.rows, self.cols), dtype=int)
        # Visible board: -1 = hidden/unknown, 0-8 = revealed values.
        self.visible_board = np.full((self.rows, self.cols), -1, dtype=int)
        self.done = False
        self.lost = False
        self.last_mine_hit: Tuple[int, int] | None = None

    def reset(self) -> np.ndarray:
        """Start a new game and return the initial observation."""
        self.hidden_board = np.zeros((self.rows, self.cols), dtype=int)
        self._place_mines()
        self._calculate_numbers()

        self.visible_board = np.full((self.rows, self.cols), -1, dtype=int)
        self.done = False
        self.lost = False
        self.last_mine_hit = None
        return self.visible_board.copy()

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict[str, str | int]]:
        """Apply one action (row, col) and return (observation, reward, done, info)."""
        if self.done:
            return (
                self.visible_board.copy(),
                0.0,
                True,
                {"message": "game_already_ended"},
            )

        row, col = action
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return (
                self.visible_board.copy(),
                -5.0,
                self.done,
                {"error": "invalid_action"},
            )

        if self.visible_board[row, col] != -1:
            return (
                self.visible_board.copy(),
                -2.0,
                self.done,
                {"error": "already_revealed"},
            )

        if self.hidden_board[row, col] == -1:
            self._reveal_cell(row, col)
            self.done = True
            self.lost = True
            self.last_mine_hit = (row, col)
            return (
                self.visible_board.copy(),
                -10.0,
                True,
                {"result": "loss"},
            )

        was_frontier_action = self._is_frontier_action(row, col)
        revealed_before = np.count_nonzero(self.visible_board != -1)
        self._reveal_cell(row, col)
        if self.hidden_board[row, col] == 0:
            self._reveal_empty_area(row, col)
        revealed_after = np.count_nonzero(self.visible_board != -1)
        newly_revealed = revealed_after - revealed_before

        if self.reward_mode == "progress":
            reward = 0.5 * newly_revealed
        else:
            reward = 1.0
            if self.reward_mode == "frontier" and was_frontier_action:
                reward += self.frontier_bonus
        info: Dict[str, str | int] = {"result": "continue", "newly_revealed": int(newly_revealed)}
        if self.reward_mode == "frontier":
            info["frontier_action"] = int(was_frontier_action)

        if self._check_win():
            self.done = True
            if self.reward_mode == "progress":
                reward += 25.0
            else:
                reward += 20.0  # +1 safe move and +20 win bonus = +21 total
            info["result"] = "win"

        return self.visible_board.copy(), reward, self.done, info

    def _place_mines(self) -> None:
        """Randomly place mines on the hidden board."""
        flat_indices = self.rng.choice(self.rows * self.cols, size=self.num_mines, replace=False)
        mine_rows, mine_cols = np.unravel_index(flat_indices, (self.rows, self.cols))
        self.hidden_board[mine_rows, mine_cols] = -1

    def _calculate_numbers(self) -> None:
        """Fill non-mine cells with the count of neighboring mines."""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.hidden_board[row, col] == -1:
                    continue
                mine_count = 0
                for nr, nc in self._get_neighbors(row, col):
                    if self.hidden_board[nr, nc] == -1:
                        mine_count += 1
                self.hidden_board[row, col] = mine_count

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Return valid neighboring coordinates (8-directional)."""
        neighbors: List[Tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = row + dr
                nc = col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def _is_frontier_action(self, row: int, col: int) -> bool:
        """Return True if a hidden cell touches at least one revealed number.

        This is calculated before revealing the selected cell. It rewards moves
        near existing information, but only after the move is known to be safe.
        """
        if self.visible_board[row, col] != -1:
            return False
        for nr, nc in self._get_neighbors(row, col):
            if self.visible_board[nr, nc] > 0:
                return True
        return False

    def _reveal_cell(self, row: int, col: int) -> None:
        """Reveal a single cell on the visible board."""
        self.visible_board[row, col] = self.hidden_board[row, col]

    def _reveal_empty_area(self, start_row: int, start_col: int) -> None:
        """Reveal connected zero-cells and their border numbers using BFS."""
        queue: deque[Tuple[int, int]] = deque()
        queue.append((start_row, start_col))
        visited = {(start_row, start_col)}

        while queue:
            row, col = queue.popleft()
            for nr, nc in self._get_neighbors(row, col):
                if (nr, nc) in visited:
                    continue
                visited.add((nr, nc))

                # Never auto-reveal mines from flood fill.
                if self.hidden_board[nr, nc] == -1:
                    continue

                if self.visible_board[nr, nc] == -1:
                    self._reveal_cell(nr, nc)

                # If neighbor is also empty, continue expanding.
                if self.hidden_board[nr, nc] == 0:
                    queue.append((nr, nc))

    def _check_win(self) -> bool:
        """Return True when all non-mine cells are revealed."""
        total_cells = self.rows * self.cols
        hidden_cells = np.count_nonzero(self.visible_board == -1)
        return hidden_cells == self.num_mines and total_cells > self.num_mines

    def render(self) -> None:
        """Print the current visible board in a human-friendly format."""
        rows_to_print: List[str] = []
        for row in range(self.rows):
            symbols: List[str] = []
            for col in range(self.cols):
                cell = self.visible_board[row, col]
                if self.lost and self.last_mine_hit == (row, col):
                    symbols.append("*")
                elif cell == -1:
                    symbols.append("?")
                elif cell == 0:
                    symbols.append(".")
                else:
                    symbols.append(str(cell))
            rows_to_print.append(" ".join(symbols))

        print("\n".join(rows_to_print))

    def render_hidden(self) -> None:
        """Print the hidden board for debugging.

        This is for human debugging only. The agent should never use hidden_board
        as its observation.
        """
        rows_to_print: List[str] = []
        for row in range(self.rows):
            symbols: List[str] = []
            for col in range(self.cols):
                cell = self.hidden_board[row, col]
                if cell == -1:
                    symbols.append("*")
                elif cell == 0:
                    symbols.append(".")
                else:
                    symbols.append(str(cell))
            rows_to_print.append(" ".join(symbols))

        print("\n".join(rows_to_print))
