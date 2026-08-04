"""Train a simple DQN agent on Minesweeper and compare against baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from agents.dqn_agent import DQNAgent, resolve_torch_device
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


def save_loss_graph(losses: list[float], output_path: str) -> str:
    """Save loss graph to file.

    Uses matplotlib when available, otherwise writes a simple SVG without extra
    dependencies. Returns the actual output path used.
    """
    valid_losses = np.array(losses, dtype=np.float32)
    episode_axis = np.arange(1, len(valid_losses) + 1)
    valid_mask = ~np.isnan(valid_losses)

    try:
        import matplotlib.pyplot as plt  # Local import so script works without matplotlib installed.

        plt.figure(figsize=(10, 4))
        plt.plot(episode_axis[valid_mask], valid_losses[valid_mask], label="Episode avg loss", alpha=0.35)

        if np.count_nonzero(valid_mask) >= 100:
            clean_losses = valid_losses[valid_mask]
            smooth = np.convolve(clean_losses, np.ones(100) / 100.0, mode="valid")
            smooth_axis = episode_axis[valid_mask][99:]
            plt.plot(smooth_axis, smooth, label="Moving avg loss (100)", linewidth=2.0)

        plt.title("DQN Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    except ModuleNotFoundError:
        # Fallback: create a basic SVG line chart so plotting never blocks training.
        if output_path.lower().endswith(".png"):
            output_path = output_path[:-4] + ".svg"
        elif not output_path.lower().endswith(".svg"):
            output_path = output_path + ".svg"

        clean_x = episode_axis[valid_mask]
        clean_y = valid_losses[valid_mask]
        if len(clean_x) == 0:
            clean_x = np.array([0, 1], dtype=np.float32)
            clean_y = np.array([0.0, 0.0], dtype=np.float32)

        width, height = 1000, 420
        pad_left, pad_right, pad_top, pad_bottom = 70, 30, 30, 50
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom

        x_min, x_max = float(np.min(clean_x)), float(np.max(clean_x))
        y_min, y_max = float(np.min(clean_y)), float(np.max(clean_y))
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0

        def to_px(x: float, y: float) -> tuple[float, float]:
            x_norm = (x - x_min) / (x_max - x_min)
            y_norm = (y - y_min) / (y_max - y_min)
            px = pad_left + x_norm * plot_w
            py = pad_top + (1.0 - y_norm) * plot_h
            return px, py

        points = [to_px(float(x), float(y)) for x, y in zip(clean_x, clean_y)]
        polyline_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  <text x="{pad_left}" y="20" font-family="Arial, sans-serif" font-size="18" fill="#111">DQN Training Loss</text>
  <line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" y2="{pad_top + plot_h}" stroke="#333" stroke-width="1.5"/>
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}" stroke="#333" stroke-width="1.5"/>
  <polyline fill="none" stroke="#0066cc" stroke-width="2" points="{polyline_points}"/>
  <text x="{pad_left + plot_w / 2:.0f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">Episode</text>
  <text x="18" y="{pad_top + plot_h / 2:.0f}" transform="rotate(-90 18 {pad_top + plot_h / 2:.0f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">Loss</text>
  <text x="{pad_left}" y="{height - 30}" font-family="Arial, sans-serif" font-size="12" fill="#555">x: {x_min:.0f} .. {x_max:.0f}</text>
  <text x="{pad_left + 180}" y="{height - 30}" font-family="Arial, sans-serif" font-size="12" fill="#555">y: {y_min:.4f} .. {y_max:.4f}</text>
</svg>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg)
        return output_path


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
    rows: int = 5,
    cols: int = 5,
    num_mines: int = 3,
    num_episodes: int = 5000,
    progress_every: int = 500,
    qlearning_baseline_episodes: int | None = None,
    loss_plot_path: str = "dqn_loss_curve.png",
    model_type: Literal["mlp", "cnn", "cnn_deep"] = "mlp",
    reward_mode: Literal["classic", "progress"] = "classic",
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    device: str | None = None,
) -> None:
    """Train DQN on Minesweeper and compare to available baselines."""
    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines, seed=42, reward_mode=reward_mode)
    resolved_device = resolve_torch_device(device)
    dqn_agent = DQNAgent(
        rows=rows,
        cols=cols,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=64,
        memory_size=20000,
        target_update_every=200,
        seed=123,
        model_type=model_type,
        device=resolved_device.type,
    )

    print(f"Using torch device: {resolved_device}")

    recent_rewards: list[float] = []
    recent_steps: list[int] = []
    recent_wins: list[int] = []
    recent_losses: list[float] = []
    avg_loss_per_episode: list[float] = []

    for episode in range(1, num_episodes + 1):
        observation = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        won = False
        episode_losses: list[float] = []

        while not done:
            action = dqn_agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            loss = dqn_agent.update(observation, action, reward, next_observation, done)
            observation = next_observation

            episode_reward += reward
            episode_steps += 1
            if loss is not None:
                recent_losses.append(loss)
                episode_losses.append(loss)

            if done and info.get("result") == "win":
                won = True

        dqn_agent.decay_epsilon()
        recent_rewards.append(episode_reward)
        recent_steps.append(episode_steps)
        recent_wins.append(1 if won else 0)
        avg_loss_per_episode.append(float(np.mean(episode_losses)) if episode_losses else float("nan"))

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
    print(f"Board: {rows}x{cols}, mines={num_mines}, reward={reward_mode}, model={model_type}")
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

    actual_plot_path = save_loss_graph(avg_loss_per_episode, loss_plot_path)
    print(f"\nSaved loss graph to: {actual_plot_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train DQN on Minesweeper.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5000,
        help="Number of DQN training episodes (default: 5000).",
    )
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--num-mines", type=int, default=3)
    parser.add_argument("--model-type", choices=["mlp", "cnn", "cnn_deep"], default="mlp")
    parser.add_argument("--reward-mode", choices=["classic", "progress"], default="classic")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
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
    parser.add_argument(
        "--loss-plot-path",
        type=str,
        default="dqn_loss_curve.png",
        help="Path for saving the training loss graph (default: dqn_loss_curve.png).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(
        rows=args.rows,
        cols=args.cols,
        num_mines=args.num_mines,
        num_episodes=args.num_episodes,
        progress_every=args.progress_every,
        qlearning_baseline_episodes=args.qlearning_baseline_episodes,
        loss_plot_path=args.loss_plot_path,
        model_type=args.model_type,
        reward_mode=args.reward_mode,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        device=args.device,
    )
