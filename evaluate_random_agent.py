"""Evaluate the RandomAgent baseline on the Minesweeper environment."""

from __future__ import annotations

from agents.random_agent import RandomAgent
from minesweeper_env import MinesweeperEnv


def evaluate_random_agent(num_games: int = 1000) -> None:
    """Run many episodes and print baseline performance metrics."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
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

    print(f"Games played: {num_games}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average reward: {average_reward:.3f}")
    print(f"Average steps: {average_steps:.3f}")


if __name__ == "__main__":
    evaluate_random_agent(num_games=1000)
