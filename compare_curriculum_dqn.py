"""Compare direct DQN training against staged curriculum training.

The curriculum transfers CNN weights between board sizes, but starts each stage
with a fresh replay buffer because replay states have board-specific shapes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from minesweeper_env import MinesweeperEnv, RewardMode


@dataclass
class StageConfig:
    """Training settings for one curriculum stage."""

    rows: int
    cols: int
    num_mines: int
    episodes: int
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    learning_rate: float
    memory_size: int


@dataclass
class Metrics:
    """Evaluation metrics for one trained model."""

    win_rate: float
    average_reward: float
    average_steps: float


def run_training_loop(
    agent: DQNAgent,
    env: MinesweeperEnv,
    episodes: int,
    progress_every: int,
) -> DQNAgent:
    """Train an agent in-place for a fixed number of episodes."""
    agent.policy_net.train()
    agent.target_net.eval()

    recent_rewards: list[float] = []
    recent_wins: list[int] = []

    for episode in range(1, episodes + 1):
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
            window = min(progress_every, len(recent_rewards))
            print(
                f"    episode {episode}/{episodes} | epsilon={agent.epsilon:.3f} "
                f"| win_rate={float(np.mean(recent_wins[-window:])):.2%} "
                f"| avg_reward={float(np.mean(recent_rewards[-window:])):.3f}"
            )

    return agent


def parse_stage(value: str) -> StageConfig:
    """Parse a curriculum stage.

    Preferred format:
    ROWSxCOLSxMINES:EPISODES:EPS_START:EPS_MIN:EPS_DECAY:LR:MEMORY_SIZE

    The older ROWSxCOLSxMINES:EPISODES:EPS_MIN:EPS_DECAY:MEMORY_SIZE
    format is still accepted and uses epsilon_start=1.0 and lr=0.001.
    """
    try:
        parts = value.split(":")
        if len(parts) == 5:
            board_part, episodes, epsilon_min, epsilon_decay, memory_size = parts
            epsilon_start = 1.0
            learning_rate = 1e-3
        elif len(parts) == 7:
            board_part, episodes, epsilon_start, epsilon_min, epsilon_decay, learning_rate, memory_size = parts
        else:
            raise ValueError

        rows, cols, num_mines = board_part.lower().split("x")
        return StageConfig(
            rows=int(rows),
            cols=int(cols),
            num_mines=int(num_mines),
            episodes=int(episodes),
            epsilon_start=float(epsilon_start),
            epsilon_min=float(epsilon_min),
            epsilon_decay=float(epsilon_decay),
            learning_rate=float(learning_rate),
            memory_size=int(memory_size),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "stage must use ROWSxCOLSxMINES:EPISODES:EPS_START:EPS_MIN:EPS_DECAY:LR:MEMORY_SIZE"
        ) from exc


def transfer_policy_weights(source_agent: DQNAgent, target_agent: DQNAgent) -> None:
    """Copy compatible CNN weights from source to target.

    CNN convolution weights are independent of board size, so they can transfer
    from 5x5 to 8x8 or 10x10. If the target uses an extra frontier channel,
    the first convolution has one extra input channel; in that case we copy the
    shared channels and leave the extra channel at its random initialization.
    """
    source_state = source_agent.policy_net.state_dict()
    target_state = target_agent.policy_net.state_dict()
    updated_state = {}

    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            updated_state[name] = target_tensor
            continue

        if source_tensor.shape == target_tensor.shape:
            updated_state[name] = source_tensor
            continue

        if source_tensor.ndim == target_tensor.ndim:
            copied = target_tensor.clone()
            slices = tuple(slice(0, min(src, dst)) for src, dst in zip(source_tensor.shape, target_tensor.shape))
            copied[slices] = source_tensor[slices]
            updated_state[name] = copied
        else:
            updated_state[name] = target_tensor

    target_agent.policy_net.load_state_dict(updated_state)
    target_agent.target_net.load_state_dict(updated_state)


def set_agent_learning_rate(agent: DQNAgent, learning_rate: float) -> None:
    """Update the optimizer learning rate for resumed or staged fine-tuning."""
    agent.lr = learning_rate
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = learning_rate


def train_stage(
    stage: StageConfig,
    seed: int,
    reward_mode: RewardMode,
    frontier_bonus: float,
    model_type: Literal["cnn", "cnn_deep"],
    replay_type: Literal["uniform", "prioritized"],
    use_frontier_channel: bool,
    progress_every: int,
    previous_agent: DQNAgent | None = None,
) -> DQNAgent:
    """Train one stage, optionally initializing from the previous stage CNN weights."""
    env = MinesweeperEnv(
        rows=stage.rows,
        cols=stage.cols,
        num_mines=stage.num_mines,
        seed=seed,
        reward_mode=reward_mode,
        frontier_bonus=frontier_bonus,
    )
    agent = DQNAgent(
        rows=stage.rows,
        cols=stage.cols,
        lr=stage.learning_rate,
        gamma=0.99,
        epsilon=stage.epsilon_start,
        epsilon_min=stage.epsilon_min,
        epsilon_decay=stage.epsilon_decay,
        batch_size=64,
        memory_size=stage.memory_size,
        target_update_every=200,
        seed=seed,
        model_type=model_type,
        replay_type=replay_type,
        use_frontier_channel=use_frontier_channel,
    )

    if previous_agent is not None:
        # CNN weights are independent of board dimensions, so they can transfer.
        # Replay memory is intentionally not transferred across different shapes.
        transfer_policy_weights(previous_agent, agent)

    return run_training_loop(agent=agent, env=env, episodes=stage.episodes, progress_every=progress_every)


def load_resume_context(checkpoint_path: Path) -> tuple[DQNAgent, StageConfig, int, RewardMode, float]:
    """Load a checkpoint and infer the board setup needed to continue training it."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    final_board = metadata.get("final_board", {})
    config = checkpoint.get("config", {})

    rows = int(final_board.get("rows", metadata.get("rows", config.get("rows"))))
    cols = int(final_board.get("cols", metadata.get("cols", config.get("cols"))))
    num_mines = int(final_board.get("num_mines", metadata.get("num_mines")))
    reward_mode = metadata.get("reward_mode", "classic")
    frontier_bonus = float(metadata.get("frontier_bonus", 0.5))
    seed = int(metadata.get("seed", 55))

    stage = StageConfig(
        rows=rows,
        cols=cols,
        num_mines=num_mines,
        episodes=0,
        epsilon_start=float(config.get("epsilon", config["epsilon_min"])),
        epsilon_min=float(config["epsilon_min"]),
        epsilon_decay=float(config["epsilon_decay"]),
        learning_rate=float(config.get("lr", 1e-3)),
        memory_size=int(config["memory_size"]),
    )
    return DQNAgent.load_checkpoint(checkpoint_path), stage, seed, reward_mode, frontier_bonus


def resume_training_from_checkpoint(
    checkpoint_path: Path,
    additional_episodes: int,
    eval_games: int,
    progress_every: int,
    epsilon_start: float | None,
    epsilon_min: float | None,
    epsilon_decay: float | None,
    learning_rate: float | None,
    reward_mode_override: RewardMode | None,
    frontier_bonus_override: float | None,
) -> tuple[Metrics, DQNAgent, StageConfig, int, RewardMode, float]:
    """Continue training an existing checkpoint on the same board."""
    agent, stage, seed, checkpoint_reward_mode, checkpoint_frontier_bonus = load_resume_context(checkpoint_path)
    reward_mode = reward_mode_override or checkpoint_reward_mode
    frontier_bonus = checkpoint_frontier_bonus if frontier_bonus_override is None else frontier_bonus_override
    env = MinesweeperEnv(
        rows=stage.rows,
        cols=stage.cols,
        num_mines=stage.num_mines,
        seed=seed,
        reward_mode=reward_mode,
        frontier_bonus=frontier_bonus,
    )

    print(
        f"\nResume checkpoint={checkpoint_path} | board={stage.rows}x{stage.cols} | "
        f"mines={stage.num_mines} | reward={reward_mode} | frontier_bonus={frontier_bonus} | seed={seed}"
    )
    print(
        f"  Starting from epsilon={agent.epsilon:.3f}, learn_steps={agent.learn_steps}, "
        f"continuing for {additional_episodes} episodes"
    )
    if epsilon_start is not None:
        agent.epsilon = epsilon_start
        print(f"  Reset training epsilon to {agent.epsilon:.3f}")
    if epsilon_min is not None:
        agent.epsilon_min = epsilon_min
        stage.epsilon_min = epsilon_min
        print(f"  Set epsilon_min to {epsilon_min:g}")
    if epsilon_decay is not None:
        agent.epsilon_decay = epsilon_decay
        stage.epsilon_decay = epsilon_decay
        print(f"  Set epsilon_decay to {epsilon_decay:g}")
    if learning_rate is not None:
        set_agent_learning_rate(agent, learning_rate)
        stage.learning_rate = learning_rate
        print(f"  Set learning rate to {learning_rate:g}")

    trained_agent = run_training_loop(
        agent=agent,
        env=env,
        episodes=additional_episodes,
        progress_every=progress_every,
    )
    metrics = evaluate_agent(
        trained_agent,
        stage,
        seed=seed,
        eval_games=eval_games,
        reward_mode=reward_mode,
        frontier_bonus=frontier_bonus,
    )
    print(
        f"  Evaluation | win_rate={metrics.win_rate:.2%} "
        f"| avg_reward={metrics.average_reward:.3f} | avg_steps={metrics.average_steps:.3f}"
    )
    return metrics, trained_agent, stage, seed, reward_mode, frontier_bonus


def evaluate_agent(
    agent: DQNAgent,
    stage: StageConfig,
    seed: int,
    eval_games: int,
    reward_mode: RewardMode,
    frontier_bonus: float,
) -> Metrics:
    """Evaluate the trained final-stage agent with epsilon=0."""
    env = MinesweeperEnv(
        rows=stage.rows,
        cols=stage.cols,
        num_mines=stage.num_mines,
        seed=seed + 10_000,
        reward_mode=reward_mode,
        frontier_bonus=frontier_bonus,
    )
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


def train_mode(
    mode: str,
    stages: list[StageConfig],
    seed: int,
    eval_games: int,
    progress_every: int,
    reward_mode: RewardMode,
    frontier_bonus: float,
    model_type: Literal["cnn", "cnn_deep"],
    replay_type: Literal["uniform", "prioritized"],
    initial_checkpoint: Path | None,
) -> tuple[Metrics, DQNAgent]:
    """Train one comparison mode and return final-stage evaluation metrics."""
    if mode == "direct":
        final_stage = stages[-1]
        active_stages = [
            StageConfig(
                rows=final_stage.rows,
                cols=final_stage.cols,
                num_mines=final_stage.num_mines,
                episodes=sum(stage.episodes for stage in stages),
                epsilon_start=final_stage.epsilon_start,
                epsilon_min=final_stage.epsilon_min,
                epsilon_decay=final_stage.epsilon_decay,
                learning_rate=final_stage.learning_rate,
                memory_size=final_stage.memory_size,
            )
        ]
        use_frontier_channel = False
    elif mode == "curriculum":
        active_stages = stages
        use_frontier_channel = False
    elif mode == "curriculum_frontier":
        active_stages = stages
        use_frontier_channel = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"\nMode={mode} | seed={seed} | frontier={use_frontier_channel}")
    agent: DQNAgent | None = None
    if initial_checkpoint is not None and mode in {"curriculum", "curriculum_frontier"}:
        agent = DQNAgent.load_checkpoint(initial_checkpoint)
        print(f"  Loaded initial checkpoint: {initial_checkpoint}")

    for index, stage in enumerate(active_stages, start=1):
        print(
            f"  Stage {index}/{len(active_stages)}: {stage.rows}x{stage.cols}, "
            f"mines={stage.num_mines}, episodes={stage.episodes}, "
            f"epsilon_start={stage.epsilon_start}, epsilon_min={stage.epsilon_min}, "
            f"epsilon_decay={stage.epsilon_decay}, lr={stage.learning_rate:g}, "
            f"memory_size={stage.memory_size}"
        )
        agent = train_stage(
            stage=stage,
            seed=seed,
            reward_mode=reward_mode,
            frontier_bonus=frontier_bonus,
            model_type=model_type,
            replay_type=replay_type,
            use_frontier_channel=use_frontier_channel,
            progress_every=progress_every,
            previous_agent=agent,
        )

    assert agent is not None
    final_stage = active_stages[-1]
    metrics = evaluate_agent(
        agent,
        final_stage,
        seed=seed,
        eval_games=eval_games,
        reward_mode=reward_mode,
        frontier_bonus=frontier_bonus,
    )
    print(
        f"  Evaluation | win_rate={metrics.win_rate:.2%} "
        f"| avg_reward={metrics.average_reward:.3f} | avg_steps={metrics.average_steps:.3f}"
    )
    return metrics, agent


def summarize(metrics: list[Metrics]) -> tuple[Metrics, Metrics]:
    """Return mean and population stddev for a list of metrics."""
    avg = Metrics(
        win_rate=mean(m.win_rate for m in metrics),
        average_reward=mean(m.average_reward for m in metrics),
        average_steps=mean(m.average_steps for m in metrics),
    )
    std = Metrics(
        win_rate=pstdev(m.win_rate for m in metrics),
        average_reward=pstdev(m.average_reward for m in metrics),
        average_steps=pstdev(m.average_steps for m in metrics),
    )
    return avg, std


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare direct vs curriculum DQN training.")
    parser.add_argument(
        "--stages",
        type=parse_stage,
        nargs="+",
        default=[
            StageConfig(5, 5, 3, 1000, 1.0, 0.001, 0.999, 1e-3, 20000),
            StageConfig(8, 8, 10, 2000, 1.0, 0.05, 0.9995, 1e-3, 50000),
            StageConfig(10, 10, 15, 3000, 1.0, 0.10, 0.9997, 1e-3, 100000),
        ],
        help=(
            "One or more ROWSxCOLSxMINES:EPISODES:EPS_START:EPS_MIN:EPS_DECAY:LR:MEMORY_SIZE "
            "stages. Older ROWSxCOLSxMINES:EPISODES:EPS_MIN:EPS_DECAY:MEMORY_SIZE stages "
            "still work with EPS_START=1.0 and LR=0.001."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["direct", "curriculum", "curriculum_frontier"],
        default=["direct", "curriculum", "curriculum_frontier"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[55])
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--reward-mode", choices=["classic", "progress", "frontier"], default="classic")
    parser.add_argument(
        "--frontier-bonus",
        type=float,
        default=None,
        help="Frontier reward bonus. Defaults to 0.5 for new runs; resumed checkpoints keep their saved bonus unless set.",
    )
    parser.add_argument("--model-type", choices=["cnn", "cnn_deep"], default="cnn_deep")
    parser.add_argument("--replay-type", choices=["uniform", "prioritized"], default="uniform")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize curriculum modes before the first stage.",
    )
    parser.add_argument(
        "--save-model-path",
        type=Path,
        default=None,
        help="Optional path for saving the final trained agent. Requires exactly one mode and one seed.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Continue training an existing checkpoint on the same board instead of running mode comparisons.",
    )
    parser.add_argument(
        "--resume-episodes",
        type=int,
        default=0,
        help="Additional episodes to train when using --resume-checkpoint.",
    )
    parser.add_argument(
        "--resume-epsilon-start",
        type=float,
        default=None,
        help="Optional epsilon value to reset to before continued checkpoint training.",
    )
    parser.add_argument(
        "--resume-epsilon-min",
        type=float,
        default=None,
        help="Optional epsilon minimum to use before continued checkpoint training.",
    )
    parser.add_argument(
        "--resume-epsilon-decay",
        type=float,
        default=None,
        help="Optional epsilon decay to use before continued checkpoint training.",
    )
    parser.add_argument(
        "--resume-learning-rate",
        type=float,
        default=None,
        help="Optional learning rate to use before continued checkpoint training.",
    )
    parser.add_argument(
        "--resume-reward-mode",
        choices=["classic", "progress", "frontier"],
        default=None,
        help="Optional reward mode override for continued checkpoint training.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all requested comparison modes."""
    args = parse_args()
    frontier_bonus = 0.5 if args.frontier_bonus is None else args.frontier_bonus
    if args.resume_checkpoint is not None:
        if args.resume_episodes <= 0:
            raise ValueError("--resume-episodes must be > 0 when using --resume-checkpoint.")
        if len(args.seeds) != 1:
            raise ValueError("--resume-checkpoint requires exactly one seed for evaluation consistency.")

        metrics, agent, stage, seed, reward_mode, frontier_bonus = resume_training_from_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            additional_episodes=args.resume_episodes,
            eval_games=args.eval_games,
            progress_every=args.progress_every,
            epsilon_start=args.resume_epsilon_start,
            epsilon_min=args.resume_epsilon_min,
            epsilon_decay=args.resume_epsilon_decay,
            learning_rate=args.resume_learning_rate,
            reward_mode_override=args.resume_reward_mode,
            frontier_bonus_override=args.frontier_bonus,
        )
        if args.save_model_path is not None:
            agent.save_checkpoint(
                args.save_model_path,
                metadata={
                    "script": "compare_curriculum_dqn.py",
                    "mode": "resume",
                    "seed": seed,
                    "final_board": {
                        "rows": stage.rows,
                        "cols": stage.cols,
                        "num_mines": stage.num_mines,
                    },
                    "eval_games": args.eval_games,
                    "reward_mode": reward_mode,
                    "frontier_bonus": frontier_bonus,
                    "model_type": agent.model_type,
                    "replay_type": agent.replay_type,
                    "resumed_from": str(args.resume_checkpoint),
                    "resume_episodes": args.resume_episodes,
                    "resume_epsilon_start": args.resume_epsilon_start,
                    "resume_epsilon_min": args.resume_epsilon_min,
                    "resume_epsilon_decay": args.resume_epsilon_decay,
                    "resume_learning_rate": args.resume_learning_rate,
                    "resume_reward_mode": args.resume_reward_mode,
                    "metrics": metrics.__dict__,
                },
            )
            print(f"Saved resumed agent checkpoint to: {args.save_model_path}")
        return

    if args.save_model_path is not None and (len(args.modes) != 1 or len(args.seeds) != 1):
        raise ValueError("--save-model-path requires exactly one mode and exactly one seed.")

    results: dict[str, list[Metrics]] = {mode: [] for mode in args.modes}

    for mode in args.modes:
        for seed in args.seeds:
            metrics, agent = train_mode(
                mode=mode,
                stages=args.stages,
                seed=seed,
                eval_games=args.eval_games,
                progress_every=args.progress_every,
                reward_mode=args.reward_mode,
                model_type=args.model_type,
                replay_type=args.replay_type,
                initial_checkpoint=args.initial_checkpoint,
                frontier_bonus=frontier_bonus,
            )
            results[mode].append(metrics)
            if args.save_model_path is not None:
                final_stage = args.stages[-1]
                agent.save_checkpoint(
                    args.save_model_path,
                    metadata={
                        "script": "compare_curriculum_dqn.py",
                        "mode": mode,
                        "seed": seed,
                        "stages": [stage.__dict__ for stage in args.stages],
                        "final_board": {
                            "rows": final_stage.rows,
                            "cols": final_stage.cols,
                            "num_mines": final_stage.num_mines,
                        },
                        "eval_games": args.eval_games,
                        "reward_mode": args.reward_mode,
                        "frontier_bonus": frontier_bonus,
                        "model_type": args.model_type,
                        "replay_type": args.replay_type,
                        "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
                        "metrics": metrics.__dict__,
                    },
                )
                print(f"Saved final agent checkpoint to: {args.save_model_path}")

    print("\n=== Curriculum Comparison Summary ===")
    for mode in args.modes:
        avg, std = summarize(results[mode])
        print(
            f"{mode} | win_rate={avg.win_rate:.2%}±{std.win_rate:.2%} "
            f"| avg_reward={avg.average_reward:.3f}±{std.average_reward:.3f} "
            f"| avg_steps={avg.average_steps:.3f}±{std.average_steps:.3f}"
        )


if __name__ == "__main__":
    main()
