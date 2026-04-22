"""Small demo script for the Minesweeper environment."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from minesweeper_env import MinesweeperEnv


def choose_random_unrevealed(env: MinesweeperEnv, rng: np.random.Generator) -> Tuple[int, int]:
    """Pick one random hidden cell from the visible board."""
    candidates: List[Tuple[int, int]] = list(map(tuple, np.argwhere(env.visible_board == -1)))
    index = rng.integers(0, len(candidates))
    row, col = candidates[int(index)]
    return int(row), int(col)


def main() -> None:
    """Run a random-play demo until the game ends."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    env.reset()
    rng = np.random.default_rng(123)

    print("Hidden board (debug only):")
    env.render_hidden()
    print()

    print("Visible board after reset:")
    env.render()
    print()

    done = False
    step_index = 0
    while not done:
        step_index += 1
        action = choose_random_unrevealed(env, rng)
        _, reward, done, info = env.step(action)
        print(f"Step {step_index}")
        print(f"chosen action: {action}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"info: {info}")
        print("visible board:")
        env.render()
        print()


if __name__ == "__main__":
    main()
