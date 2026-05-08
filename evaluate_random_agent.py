"""Evaluate the RandomAgent baseline on the Minesweeper environment."""

from __future__ import annotations

import argparse
from agents.random_agent import RandomAgent
from minesweeper_env import MinesweeperEnv, RewardMode


def evaluate_random_agent(
    rows: int = 5,
    cols: int = 5,
    num_mines: int = 3,
    num_games: int = 1000,
    reward_mode: RewardMode = "classic",
    frontier_bonus: float = 0.5,
) -> None:
    """Run many episodes and print baseline performance metrics."""
    env = MinesweeperEnv(
        rows=rows,
        cols=cols,
        num_mines=num_mines,
        seed=42,
        reward_mode=reward_mode,
        frontier_bonus=frontier_bonus,
    )
    agent = RandomAgent(seed=123)

    wins = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(num_games):
        observation = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        final_info = {}

        while not done:
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            final_info = info

        if final_info.get("result") == "win":
            wins += 1

        total_reward += episode_reward
        total_steps += episode_steps

    win_rate = wins / num_games
    average_reward = total_reward / num_games
    average_steps = total_steps / num_games

    print(f"Board: {rows}x{cols}, mines={num_mines}, reward={reward_mode}, frontier_bonus={frontier_bonus}")
    print(f"Games played: {num_games}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average reward: {average_reward:.3f}")
    print(f"Average steps: {average_steps:.3f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for baseline evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate RandomAgent on Minesweeper.")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--num-mines", type=int, default=3)
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--reward-mode", choices=["classic", "progress", "frontier"], default="classic")
    parser.add_argument("--frontier-bonus", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_random_agent(
        rows=args.rows,
        cols=args.cols,
        num_mines=args.num_mines,
        num_games=args.num_games,
        reward_mode=args.reward_mode,
        frontier_bonus=args.frontier_bonus,
    )
