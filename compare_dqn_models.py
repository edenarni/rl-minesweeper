"""Multi-seed comparison of MLP DQN vs CNN DQN on Minesweeper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class MethodConfig:
    """Configuration for one replay-method experiment."""

    name: str
    replay_type: Literal["uniform", "prioritized"]


def train_one_seed(
    model_type: Literal["mlp", "cnn", "cnn_deep"],
    replay_type: Literal["uniform", "prioritized"],
    reward_mode: Literal["classic", "progress"],
    seed: int,
    num_episodes: int,
    progress_every: int,
    epsilon_min: float,
    epsilon_decay: float,
    alpha: float,
    beta_start: float,
    beta_end: float,
    priority_epsilon: float,
) -> DQNAgent:
    """Train one DQN agent for a specific seed."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=seed, reward_mode=reward_mode)
    agent = DQNAgent(
        rows=5,
        cols=5,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=64,
        memory_size=20000,
        target_update_every=200,
        seed=seed,
        model_type=model_type,
        replay_type=replay_type,
        alpha=alpha,
        beta_start=beta_start,
        beta_end=beta_end,
        priority_epsilon=priority_epsilon,
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


def evaluate_one_seed(
    agent: DQNAgent,
    seed: int,
    eval_games: int,
    reward_mode: Literal["classic", "progress"],
) -> Metrics:
    """Evaluate one trained DQN agent with greedy policy (epsilon=0)."""
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=seed + 10_000, reward_mode=reward_mode)
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
    model_type: Literal["mlp", "cnn", "cnn_deep"],
    replay_type: Literal["uniform", "prioritized"],
    reward_mode: Literal["classic", "progress"],
    seeds: list[int],
    num_episodes: int,
    eval_games: int,
    progress_every: int,
    epsilon_min: float,
    epsilon_decay: float,
    alpha: float,
    beta_start: float,
    beta_end: float,
    priority_epsilon: float,
    save_model_path: Path | None = None,
) -> list[Metrics]:
    """Train+evaluate one model type across many seeds."""
    print(f"\n=== {model_type.upper()} DQN ({replay_type}, reward={reward_mode}) | seeds={seeds} ===")
    per_seed_metrics: list[Metrics] = []

    for seed in seeds:
        print(f"\nTraining {model_type.upper()} seed={seed} | replay={replay_type}")
        agent = train_one_seed(
            model_type,
            replay_type,
            reward_mode,
            seed,
            num_episodes,
            progress_every,
            epsilon_min,
            epsilon_decay,
            alpha,
            beta_start,
            beta_end,
            priority_epsilon,
        )
        metrics = evaluate_one_seed(agent, seed=seed, eval_games=eval_games, reward_mode=reward_mode)
        per_seed_metrics.append(metrics)
        print(
            f"Evaluation seed={seed} | win_rate={metrics.win_rate:.2%} "
            f"| avg_reward={metrics.average_reward:.3f} | avg_steps={metrics.average_steps:.3f}"
        )
        if save_model_path is not None:
            agent.epsilon = 0.0
            agent.save_checkpoint(
                save_model_path,
                metadata={
                    "rows": 5,
                    "cols": 5,
                    "num_mines": 3,
                    "seed": seed,
                    "num_episodes": num_episodes,
                    "eval_games": eval_games,
                    "model_type": model_type,
                    "replay_type": replay_type,
                    "reward_mode": reward_mode,
                    "epsilon_min": epsilon_min,
                    "epsilon_decay": epsilon_decay,
                    "win_rate": metrics.win_rate,
                    "average_reward": metrics.average_reward,
                    "average_steps": metrics.average_steps,
                },
            )
            print(f"Saved model checkpoint to {save_model_path}")

    agg_mean = summarize(per_seed_metrics)
    agg_std = summarize_std(per_seed_metrics)
    print(f"\n{model_type.upper()} ({replay_type}, reward={reward_mode}) summary across {len(seeds)} seeds:")
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
    parser = argparse.ArgumentParser(description="Evaluate DQN variants over one or more seeds.")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--priority-epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--save-model-path",
        type=Path,
        default=None,
        help="Optional checkpoint path. Requires exactly one model, replay mode, reward mode, and seed.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mlp", "cnn", "cnn_deep"],
        default=["cnn"],
        help="Model types to evaluate (default: cnn).",
    )
    parser.add_argument(
        "--replay-modes",
        nargs="+",
        choices=["uniform", "prioritized"],
        default=["uniform"],
        help="Replay modes to evaluate (default: uniform).",
    )
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        choices=["classic", "progress"],
        default=["classic"],
        help="Reward modes to evaluate (default: classic).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[55],
        help="List of seeds to evaluate (default: 55).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_model_path is not None:
        if (
            len(args.models) != 1
            or len(args.replay_modes) != 1
            or len(args.reward_modes) != 1
            or len(args.seeds) != 1
        ):
            raise SystemExit("--save-model-path requires exactly one model, replay mode, reward mode, and seed.")

    results_by_method: dict[str, list[Metrics]] = {}

    for model_type in args.models:
        for replay_type in args.replay_modes:
            for reward_mode in args.reward_modes:
                key = f"{model_type}:{replay_type}:{reward_mode}"
                results_by_method[key] = run_multi_seed(
                    model_type=model_type,
                    replay_type=replay_type,
                    reward_mode=reward_mode,
                    seeds=args.seeds,
                    num_episodes=args.num_episodes,
                    eval_games=args.eval_games,
                    progress_every=args.progress_every,
                    epsilon_min=args.epsilon_min,
                    epsilon_decay=args.epsilon_decay,
                    alpha=args.alpha,
                    beta_start=args.beta_start,
                    beta_end=args.beta_end,
                    priority_epsilon=args.priority_epsilon,
                    save_model_path=args.save_model_path,
                )

    print("\n=== Final Summary (Mean ± Std across seeds) ===")
    for model_type in args.models:
        for replay_type in args.replay_modes:
            for reward_mode in args.reward_modes:
                method_key = f"{model_type}:{replay_type}:{reward_mode}"
                model_results = results_by_method[method_key]
                model_mean = summarize(model_results)
                model_std = summarize_std(model_results)
                print(
                    f"{model_type.upper()} DQN ({replay_type}, reward={reward_mode}) | "
                    f"win_rate={model_mean.win_rate:.2%}±{model_std.win_rate:.2%} "
                    f"| avg_reward={model_mean.average_reward:.3f}±{model_std.average_reward:.3f} "
                    f"| avg_steps={model_mean.average_steps:.3f}±{model_std.average_steps:.3f}"
                )


if __name__ == "__main__":
    main()
