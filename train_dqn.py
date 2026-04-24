"""Train a simple DQN agent on Minesweeper and compare against baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from minesweeper_env import MinesweeperEnv

try:
    from agents.q_learning_agent import QLearningAgent

    HAS_Q_LEARNING = True
except ImportError:
    HAS_Q_LEARNING = False


class AgentProtocol(Protocol):
    """Minimal protocol for evaluation agents."""

    def select_action(self, observation: np.ndarray) -> tuple[int, int]:
        """Select action from current observation."""


@dataclass
class Metrics:
    """Container for evaluation metrics."""

    win_rate: float
    average_reward: float
    average_steps: float


def evaluate_agent(
    env: MinesweeperEnv,
    agent: AgentProtocol,
    num_games: int,
    epsilon_override: float | None = None,
) -> Metrics:
    """Run evaluation episodes and compute aggregate metrics."""
    wins = 0
    rewards_total = 0.0
    steps_total = 0

    for _ in range(num_games):
        observation = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        final_info = {}

        while not done:
            if epsilon_override is not None and hasattr(agent, "select_action"):
                # DQN supports epsilon override; RandomAgent/QLearningAgent ignore this path.
                try:
                    action = agent.select_action(observation, epsilon=epsilon_override)  # type: ignore[misc]
                except TypeError:
                    action = agent.select_action(observation)
            else:
                action = agent.select_action(observation)

            observation, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            final_info = info

        if final_info.get("result") == "win":
            wins += 1

        rewards_total += episode_reward
        steps_total += episode_steps

    return Metrics(
        win_rate=wins / num_games,
        average_reward=rewards_total / num_games,
        average_steps=steps_total / num_games,
    )


def train_qlearning_baseline(env: MinesweeperEnv, num_episodes: int = 5000) -> Metrics | None:
    """Train and evaluate tabular Q-learning baseline if available."""
    if not HAS_Q_LEARNING:
        return None

    q_agent = QLearningAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        seed=321,
    )

    for _ in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = q_agent.select_action(observation)
            next_observation, reward, done, _ = env.step(action)
            q_agent.update(observation, action, reward, next_observation, done)
            observation = next_observation
        q_agent.decay_epsilon()

    saved_epsilon = q_agent.epsilon
    q_agent.epsilon = 0.0
    metrics = evaluate_agent(env, q_agent, num_games=1000)
    q_agent.epsilon = saved_epsilon
    return metrics


def train_dqn(
    num_episodes: int = 5000,
    progress_every: int = 500,
    qlearning_baseline_episodes: int | None = None,
) -> None:
    """Train DQN on Minesweeper and compare to available baselines."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    dqn_agent = DQNAgent(
        rows=5,
        cols=5,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=20000,
        target_update_every=200,
        seed=123,
    )

    recent_rewards: list[float] = []
    recent_steps: list[int] = []
    recent_wins: list[int] = []
    recent_losses: list[float] = []

    for episode in range(1, num_episodes + 1):
        observation = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        won = False

        while not done:
            action = dqn_agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            loss = dqn_agent.update(observation, action, reward, next_observation, done)
            observation = next_observation

            episode_reward += reward
            episode_steps += 1
            if loss is not None:
                recent_losses.append(loss)

            if done and info.get("result") == "win":
                won = True

        dqn_agent.decay_epsilon()
        recent_rewards.append(episode_reward)
        recent_steps.append(episode_steps)
        recent_wins.append(1 if won else 0)

        if episode % progress_every == 0:
            window_reward = float(np.mean(recent_rewards[-progress_every:]))
            window_steps = float(np.mean(recent_steps[-progress_every:]))
            window_win_rate = float(np.mean(recent_wins[-progress_every:]))
            if recent_losses:
                window_loss = float(np.mean(recent_losses[-progress_every:]))
            else:
                window_loss = float("nan")
            print(
                f"Episode {episode}/{num_episodes} | "
                f"epsilon={dqn_agent.epsilon:.3f} | "
                f"win_rate={window_win_rate:.2%} | "
                f"avg_reward={window_reward:.3f} | "
                f"avg_steps={window_steps:.3f} | "
                f"avg_loss={window_loss:.4f}"
            )

    eval_games = 1000
    dqn_metrics = evaluate_agent(env, dqn_agent, num_games=eval_games, epsilon_override=0.0)
    random_metrics = evaluate_agent(env, RandomAgent(seed=456), num_games=eval_games)
    baseline_episodes = num_episodes if qlearning_baseline_episodes is None else qlearning_baseline_episodes
    qlearning_metrics = train_qlearning_baseline(env, num_episodes=baseline_episodes)

    print("\n=== Final Evaluation (1000 games) ===")
    print("Evaluation setting:")
    print("  DQNAgent epsilon: 0.0 (greedy policy)")

    print("\nRandomAgent baseline:")
    print(f"  Win rate: {random_metrics.win_rate:.2%}")
    print(f"  Average reward: {random_metrics.average_reward:.3f}")
    print(f"  Average steps: {random_metrics.average_steps:.3f}")

    if qlearning_metrics is not None:
        print("\nQLearningAgent baseline (trained then evaluated with epsilon=0.0):")
        print(f"  Win rate: {qlearning_metrics.win_rate:.2%}")
        print(f"  Average reward: {qlearning_metrics.average_reward:.3f}")
        print(f"  Average steps: {qlearning_metrics.average_steps:.3f}")

    print("\nDQNAgent (after training):")
    print(f"  Win rate: {dqn_metrics.win_rate:.2%}")
    print(f"  Average reward: {dqn_metrics.average_reward:.3f}")
    print(f"  Average steps: {dqn_metrics.average_steps:.3f}")

    print("\nDQNAgent minus RandomAgent:")
    print(f"  Win rate delta: {(dqn_metrics.win_rate - random_metrics.win_rate):.2%}")
    print(f"  Average reward delta: {(dqn_metrics.average_reward - random_metrics.average_reward):.3f}")
    print(f"  Average steps delta: {(dqn_metrics.average_steps - random_metrics.average_steps):.3f}")

    if qlearning_metrics is not None:
        print("\nDQNAgent minus QLearningAgent:")
        print(f"  Win rate delta: {(dqn_metrics.win_rate - qlearning_metrics.win_rate):.2%}")
        print(
            f"  Average reward delta: {(dqn_metrics.average_reward - qlearning_metrics.average_reward):.3f}"
        )
        print(f"  Average steps delta: {(dqn_metrics.average_steps - qlearning_metrics.average_steps):.3f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train DQN on Minesweeper.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5000,
        help="Number of DQN training episodes (default: 5000).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print training progress every N episodes (default: 500).",
    )
    parser.add_argument(
        "--qlearning-baseline-episodes",
        type=int,
        default=None,
        help="Optional episodes for QLearning baseline training. Default: same as --num-episodes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(
        num_episodes=args.num_episodes,
        progress_every=args.progress_every,
        qlearning_baseline_episodes=args.qlearning_baseline_episodes,
    )
