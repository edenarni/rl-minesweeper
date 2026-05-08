"""Train and evaluate a simple tabular Q-learning agent on Minesweeper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from minesweeper_env import MinesweeperEnv, RewardMode


class AgentProtocol(Protocol):
    """Minimal protocol shared by evaluation agents."""

    def select_action(self, observation: np.ndarray) -> tuple[int, int]:
        """Select action from current observation."""


@dataclass
class Metrics:
    """Simple container for aggregate evaluation metrics."""

    win_rate: float
    average_reward: float
    average_steps: float


def run_episode_training(env: MinesweeperEnv, agent: QLearningAgent) -> tuple[float, int, bool]:
    """Run one training episode and update Q-values after each step."""
    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    won = False

    while not done:
        action = agent.select_action(observation)
        next_observation, reward, done, info = env.step(action)
        agent.update(observation, action, reward, next_observation, done)

        observation = next_observation
        total_reward += reward
        steps += 1

        if done and info.get("result") == "win":
            won = True

    return total_reward, steps, won


def evaluate_agent(env: MinesweeperEnv, agent: AgentProtocol, num_games: int) -> Metrics:
    """Evaluate an agent without training updates."""
    wins = 0
    rewards = 0.0
    steps_total = 0

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

        rewards += episode_reward
        steps_total += episode_steps

    return Metrics(
        win_rate=wins / num_games,
        average_reward=rewards / num_games,
        average_steps=steps_total / num_games,
    )


def train_q_learning(
    rows: int = 5,
    cols: int = 5,
    num_mines: int = 3,
    num_episodes: int = 5000,
    progress_every: int = 500,
    reward_mode: RewardMode = "classic",
    frontier_bonus: float = 0.5,
) -> None:
    """Train tabular Q-learning and compare against a random baseline."""
    env = MinesweeperEnv(
        rows=rows,
        cols=cols,
        num_mines=num_mines,
        seed=42,
        reward_mode=reward_mode,
        frontier_bonus=frontier_bonus,
    )
    q_agent = QLearningAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        seed=123,
    )

    recent_rewards: list[float] = []
    recent_steps: list[int] = []
    recent_wins: list[int] = []

    for episode in range(1, num_episodes + 1):
        episode_reward, episode_steps, won = run_episode_training(env, q_agent)
        q_agent.decay_epsilon()

        recent_rewards.append(episode_reward)
        recent_steps.append(episode_steps)
        recent_wins.append(1 if won else 0)

        if episode % progress_every == 0:
            window_reward = float(np.mean(recent_rewards[-progress_every:]))
            window_steps = float(np.mean(recent_steps[-progress_every:]))
            window_win_rate = float(np.mean(recent_wins[-progress_every:]))
            print(
                f"Episode {episode}/{num_episodes} | "
                f"epsilon={q_agent.epsilon:.3f} | "
                f"win_rate={window_win_rate:.2%} | "
                f"avg_reward={window_reward:.3f} | "
                f"avg_steps={window_steps:.3f}"
            )

    # Evaluate trained Q agent greedily (no exploration).
    eval_games = 1000
    saved_epsilon = q_agent.epsilon
    q_agent.epsilon = 0.0
    q_metrics = evaluate_agent(env, q_agent, num_games=eval_games)
    q_agent.epsilon = saved_epsilon

    random_agent = RandomAgent(seed=456)
    random_metrics = evaluate_agent(env, random_agent, num_games=eval_games)

    print("\n=== Final Evaluation (1000 games) ===")
    print(f"Board: {rows}x{cols}, mines={num_mines}, reward={reward_mode}, frontier_bonus={frontier_bonus}")
    print("Evaluation setting:")
    print("  QLearningAgent epsilon: 0.0 (greedy policy)")

    print("\nRandomAgent baseline:")
    print(f"  Win rate: {random_metrics.win_rate:.2%}")
    print(f"  Average reward: {random_metrics.average_reward:.3f}")
    print(f"  Average steps: {random_metrics.average_steps:.3f}")

    print("\nQLearningAgent (after training):")
    print(f"  Win rate: {q_metrics.win_rate:.2%}")
    print(f"  Average reward: {q_metrics.average_reward:.3f}")
    print(f"  Average steps: {q_metrics.average_steps:.3f}")

    print("\nQLearningAgent minus RandomAgent:")
    print(f"  Win rate delta: {(q_metrics.win_rate - random_metrics.win_rate):.2%}")
    print(f"  Average reward delta: {(q_metrics.average_reward - random_metrics.average_reward):.3f}")
    print(f"  Average steps delta: {(q_metrics.average_steps - random_metrics.average_steps):.3f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Q-learning training."""
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on Minesweeper.")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--num-mines", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--reward-mode", choices=["classic", "progress", "frontier"], default="classic")
    parser.add_argument("--frontier-bonus", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_q_learning(
        rows=args.rows,
        cols=args.cols,
        num_mines=args.num_mines,
        num_episodes=args.num_episodes,
        progress_every=args.progress_every,
        reward_mode=args.reward_mode,
        frontier_bonus=args.frontier_bonus,
    )
