"""Multi-seed comparison of MLP DQN vs CNN DQN on Minesweeper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Literal

import numpy as np

from agents.dqn_agent import DQNAgent
from minesweeper_env import MinesweeperEnv


@dataclass
class Metrics:
    """Evaluation metrics for one trained model."""

    win_rate: float
    average_reward: float
    average_steps: float


def train_one_seed(
    model_type: Literal["mlp", "cnn"],
    seed: int,
    num_episodes: int,
    progress_every: int,
) -> DQNAgent:
    """Train one DQN agent for a specific seed."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=seed)
    agent = DQNAgent(
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
        seed=seed,
        model_type=model_type,
    )

    recent_rewards: list[float] = []
    recent_wins: list[int] = []

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        won = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            if done and info.get("result") == "win":
                won = True

        agent.decay_epsilon()
        recent_rewards.append(episode_reward)
        recent_wins.append(1 if won else 0)

        if progress_every > 0 and episode % progress_every == 0:
            window_reward = float(np.mean(recent_rewards[-progress_every:]))
            window_win_rate = float(np.mean(recent_wins[-progress_every:]))
            print(
                f"  seed={seed} episode {episode}/{num_episodes} "
                f"| epsilon={agent.epsilon:.3f} | win_rate={window_win_rate:.2%} "
                f"| avg_reward={window_reward:.3f}"
            )

    return agent


def evaluate_one_seed(agent: DQNAgent, seed: int, eval_games: int) -> Metrics:
    """Evaluate one trained DQN agent with greedy policy (epsilon=0)."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=seed + 10_000)
    wins = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(eval_games):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        final_info = {}

        while not done:
            action = agent.select_action(obs, epsilon=0.0)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            final_info = info

        if final_info.get("result") == "win":
            wins += 1
        total_reward += episode_reward
        total_steps += episode_steps

    return Metrics(
        win_rate=wins / eval_games,
        average_reward=total_reward / eval_games,
        average_steps=total_steps / eval_games,
    )


def summarize(metrics: list[Metrics]) -> Metrics:
    """Compute mean metrics across seeds."""
    return Metrics(
        win_rate=mean(m.win_rate for m in metrics),
        average_reward=mean(m.average_reward for m in metrics),
        average_steps=mean(m.average_steps for m in metrics),
    )


def summarize_std(metrics: list[Metrics]) -> Metrics:
    """Compute population std metrics across seeds."""
    return Metrics(
        win_rate=pstdev(m.win_rate for m in metrics),
        average_reward=pstdev(m.average_reward for m in metrics),
        average_steps=pstdev(m.average_steps for m in metrics),
    )


def run_multi_seed(
    model_type: Literal["mlp", "cnn"],
    seeds: list[int],
    num_episodes: int,
    eval_games: int,
    progress_every: int,
) -> list[Metrics]:
    """Train+evaluate one model type across many seeds."""
    print(f"\n=== {model_type.upper()} DQN | seeds={seeds} ===")
    per_seed_metrics: list[Metrics] = []

    for seed in seeds:
        print(f"\nTraining {model_type.upper()} seed={seed}")
        agent = train_one_seed(model_type, seed, num_episodes, progress_every)
        metrics = evaluate_one_seed(agent, seed=seed, eval_games=eval_games)
        per_seed_metrics.append(metrics)
        print(
            f"Evaluation seed={seed} | win_rate={metrics.win_rate:.2%} "
            f"| avg_reward={metrics.average_reward:.3f} | avg_steps={metrics.average_steps:.3f}"
        )

    agg_mean = summarize(per_seed_metrics)
    agg_std = summarize_std(per_seed_metrics)
    print(f"\n{model_type.upper()} summary across {len(seeds)} seeds:")
    print(
        f"  Mean   | win_rate={agg_mean.win_rate:.2%} | avg_reward={agg_mean.average_reward:.3f} "
        f"| avg_steps={agg_mean.average_steps:.3f}"
    )
    print(
        f"  Stddev | win_rate={agg_std.win_rate:.2%} | avg_reward={agg_std.average_reward:.3f} "
        f"| avg_steps={agg_std.average_steps:.3f}"
    )
    return per_seed_metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI args for multi-seed comparison."""
    parser = argparse.ArgumentParser(description="Compare MLP DQN vs CNN DQN over multiple seeds.")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33, 44, 55],
        help="List of seeds to evaluate (default: 11 22 33 44 55).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mlp_results = run_multi_seed(
        model_type="mlp",
        seeds=args.seeds,
        num_episodes=args.num_episodes,
        eval_games=args.eval_games,
        progress_every=args.progress_every,
    )
    cnn_results = run_multi_seed(
        model_type="cnn",
        seeds=args.seeds,
        num_episodes=args.num_episodes,
        eval_games=args.eval_games,
        progress_every=args.progress_every,
    )

    mlp_mean = summarize(mlp_results)
    mlp_std = summarize_std(mlp_results)
    cnn_mean = summarize(cnn_results)
    cnn_std = summarize_std(cnn_results)

    print("\n=== Final Comparison (Mean ± Std across seeds) ===")
    print(
        f"MLP DQN | win_rate={mlp_mean.win_rate:.2%}±{mlp_std.win_rate:.2%} "
        f"| avg_reward={mlp_mean.average_reward:.3f}±{mlp_std.average_reward:.3f} "
        f"| avg_steps={mlp_mean.average_steps:.3f}±{mlp_std.average_steps:.3f}"
    )
    print(
        f"CNN DQN | win_rate={cnn_mean.win_rate:.2%}±{cnn_std.win_rate:.2%} "
        f"| avg_reward={cnn_mean.average_reward:.3f}±{cnn_std.average_reward:.3f} "
        f"| avg_steps={cnn_mean.average_steps:.3f}±{cnn_std.average_steps:.3f}"
    )


if __name__ == "__main__":
    main()
