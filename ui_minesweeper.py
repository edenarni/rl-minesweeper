"""Simple UI to watch an agent play Minesweeper one move at a time.

Usage:
    python3 ui_minesweeper.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext
from tkinter import ttk
from typing import Protocol

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from minesweeper_env import MinesweeperEnv


class AgentProtocol(Protocol):
    """Protocol for step-play agents used by the UI."""

    def select_action(self, observation: np.ndarray) -> tuple[int, int]:
        """Pick one action for current observation."""


class MinesweeperUI:
    """Tkinter app that replays one agent action per button click."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RL Minesweeper - Step Viewer")

        self.rows = 5
        self.cols = 5
        self.num_mines = 3
        self.dqn_seed = 55
        self.dqn_epsilon_min = 0.001
        self.dqn_epsilon_decay = 0.999
        self.reward_mode = "classic"
        self.frontier_bonus = 0.5
        self.rows_var = tk.StringVar(value=str(self.rows))
        self.cols_var = tk.StringVar(value=str(self.cols))
        self.num_mines_var = tk.StringVar(value=str(self.num_mines))
        self.reward_mode_var = tk.StringVar(value=self.reward_mode)
        self.frontier_bonus_var = tk.StringVar(value=str(self.frontier_bonus))
        self.project_dir = Path(__file__).resolve().parent
        self.models_dir = self.project_dir / "models"
        self.best_model_path = self.models_dir / "best_cnn_deep_5x5_seed55.pt"
        self.checkpoint_paths: dict[str, Path] = {}
        self.env = MinesweeperEnv(
            rows=self.rows,
            cols=self.cols,
            num_mines=self.num_mines,
            seed=42,
            reward_mode=self.reward_mode,
            frontier_bonus=self.frontier_bonus,
        )
        self.observation = self.env.reset()

        self.agent: AgentProtocol = RandomAgent(seed=123)
        self.episode_steps = 0
        self.episode_reward = 0.0

        self._build_ui()
        self._refresh_board()
        self._update_model_info()
        self._set_status("Ready. Click 'Prepare Agent' then 'Next Move'.")

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Agent:").pack(side=tk.LEFT)
        self.agent_var = tk.StringVar(value="Saved DQN Checkpoint")
        self.agent_menu = ttk.Combobox(
            top,
            textvariable=self.agent_var,
            values=["Saved DQN Checkpoint", "Random", "Q-Learning", "DQN-MLP", "DQN-CNN", "DQN-CNN-DEEP"],
            state="readonly",
            width=20,
        )
        self.agent_menu.pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(top, text="Train Episodes:").pack(side=tk.LEFT)
        self.train_episodes_var = tk.StringVar(value="10000")
        self.train_entry = ttk.Entry(top, textvariable=self.train_episodes_var, width=8)
        self.train_entry.pack(side=tk.LEFT, padx=(6, 12))

        self.prepare_button = ttk.Button(top, text="Prepare Agent", command=self.prepare_agent)
        self.prepare_button.pack(side=tk.LEFT, padx=(0, 6))

        self.reset_button = ttk.Button(top, text="New Game", command=self.new_game)
        self.reset_button.pack(side=tk.LEFT, padx=(0, 6))

        self.next_button = ttk.Button(top, text="Next Move", command=self.next_move)
        self.next_button.pack(side=tk.LEFT)

        checkpoint_frame = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        checkpoint_frame.pack(fill=tk.X)

        ttk.Label(checkpoint_frame, text="Saved model:").pack(side=tk.LEFT)
        self.checkpoint_var = tk.StringVar(value="")
        self.checkpoint_menu = ttk.Combobox(
            checkpoint_frame,
            textvariable=self.checkpoint_var,
            values=[],
            state="readonly",
            width=54,
        )
        self.checkpoint_menu.pack(side=tk.LEFT, padx=(6, 6))

        self.refresh_models_button = ttk.Button(
            checkpoint_frame,
            text="Refresh Models",
            command=self._refresh_checkpoint_list,
        )
        self.refresh_models_button.pack(side=tk.LEFT)
        self._refresh_checkpoint_list()

        board_settings_frame = ttk.Frame(self.root, padding=(10, 6, 10, 0))
        board_settings_frame.pack(fill=tk.X)

        ttk.Label(board_settings_frame, text="Rows:").pack(side=tk.LEFT)
        self.rows_entry = ttk.Entry(board_settings_frame, textvariable=self.rows_var, width=5)
        self.rows_entry.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Label(board_settings_frame, text="Cols:").pack(side=tk.LEFT)
        self.cols_entry = ttk.Entry(board_settings_frame, textvariable=self.cols_var, width=5)
        self.cols_entry.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Label(board_settings_frame, text="Mines:").pack(side=tk.LEFT)
        self.num_mines_entry = ttk.Entry(board_settings_frame, textvariable=self.num_mines_var, width=5)
        self.num_mines_entry.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Label(board_settings_frame, text="Reward:").pack(side=tk.LEFT)
        self.reward_mode_menu = ttk.Combobox(
            board_settings_frame,
            textvariable=self.reward_mode_var,
            values=["classic", "progress", "frontier"],
            state="readonly",
            width=10,
        )
        self.reward_mode_menu.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Label(board_settings_frame, text="Frontier Bonus:").pack(side=tk.LEFT)
        self.frontier_bonus_entry = ttk.Entry(
            board_settings_frame,
            textvariable=self.frontier_bonus_var,
            width=6,
        )
        self.frontier_bonus_entry.pack(side=tk.LEFT, padx=(4, 10))

        self.apply_board_button = ttk.Button(
            board_settings_frame,
            text="Apply Board",
            command=self.apply_board_settings,
        )
        self.apply_board_button.pack(side=tk.LEFT)

        self.board_frame = ttk.Frame(self.root, padding=10)
        self.board_frame.pack()
        self.cell_labels: list[list[tk.Label]] = []
        self._build_board_grid()

        info_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        info_frame.pack(fill=tk.X)

        self.model_info_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.model_info_var).pack(anchor="w")

        self.summary_var = tk.StringVar(value="steps=0 | total_reward=0.0")
        ttk.Label(info_frame, textvariable=self.summary_var).pack(anchor="w")

        self.progress_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.progress_var).pack(anchor="w", pady=(4, 0))

        self.progress_bar = ttk.Progressbar(
            info_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            length=420,
        )
        self.progress_bar.pack(anchor="w", pady=(4, 0))

        self.status_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.status_var, wraplength=620).pack(anchor="w", pady=(4, 0))

        ttk.Label(info_frame, text="Preparation log:").pack(anchor="w", pady=(8, 0))
        self.log_text = scrolledtext.ScrolledText(
            info_frame,
            width=78,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Courier", 10),
        )
        self.log_text.pack(fill=tk.X, pady=(4, 0))

    def _build_board_grid(self) -> None:
        """Create board cells for the current board size."""
        for child in self.board_frame.winfo_children():
            child.destroy()

        self.cell_labels = []
        for r in range(self.rows):
            row_labels: list[tk.Label] = []
            for c in range(self.cols):
                label = tk.Label(
                    self.board_frame,
                    text="?",
                    width=3,
                    height=1,
                    font=("Helvetica", 12, "bold"),
                    relief=tk.RIDGE,
                    bd=1,
                    bg="#d9d9d9",
                )
                label.grid(row=r, column=c, padx=1, pady=1)
                row_labels.append(label)
            self.cell_labels.append(row_labels)

    def _refresh_checkpoint_list(self) -> None:
        """Reload available saved PyTorch checkpoints from the models directory."""
        self.models_dir.mkdir(exist_ok=True)
        checkpoint_files = sorted(self.models_dir.glob("*.pt"))
        self.checkpoint_paths = {path.name: path for path in checkpoint_files}
        names = list(self.checkpoint_paths)
        self.checkpoint_menu["values"] = names

        if not names:
            self.checkpoint_var.set("")
            return

        preferred_name = self.best_model_path.name
        current_name = self.checkpoint_var.get()
        if current_name in self.checkpoint_paths:
            self.checkpoint_var.set(current_name)
        elif preferred_name in self.checkpoint_paths:
            self.checkpoint_var.set(preferred_name)
        else:
            self.checkpoint_var.set(names[0])

    def _selected_checkpoint_path(self) -> Path:
        """Return the checkpoint selected in the UI."""
        selected = self.checkpoint_var.get()
        if selected not in self.checkpoint_paths:
            raise FileNotFoundError("No saved checkpoint is selected. Put .pt files under the models directory.")
        return self.checkpoint_paths[selected]

    def _read_checkpoint_details(self, checkpoint_path: Path) -> tuple[int, int, int, str, float, dict, dict]:
        """Read board/reward metadata without creating the agent yet."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        metadata = checkpoint.get("metadata", {})
        final_board = metadata.get("final_board", {})

        rows = int(final_board.get("rows", metadata.get("rows", config.get("rows"))))
        cols = int(final_board.get("cols", metadata.get("cols", config.get("cols"))))
        num_mines = int(final_board.get("num_mines", metadata.get("num_mines", self.num_mines)))
        reward_mode = str(metadata.get("reward_mode", self.reward_mode))
        frontier_bonus = float(metadata.get("frontier_bonus", self.frontier_bonus))
        return rows, cols, num_mines, reward_mode, frontier_bonus, config, metadata

    def _sync_board_controls(self) -> None:
        """Keep the editable board controls in sync with the active environment."""
        self.rows_var.set(str(self.rows))
        self.cols_var.set(str(self.cols))
        self.num_mines_var.set(str(self.num_mines))
        self.reward_mode_var.set(self.reward_mode)
        self.frontier_bonus_var.set(str(self.frontier_bonus))

    def _reset_environment_for_model(
        self,
        rows: int,
        cols: int,
        num_mines: int,
        reward_mode: str,
        frontier_bonus: float,
    ) -> None:
        """Resize the UI board and reset the environment for a loaded checkpoint."""
        board_changed = rows != self.rows or cols != self.cols
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.reward_mode = reward_mode
        self.frontier_bonus = frontier_bonus
        self.env = MinesweeperEnv(
            rows=self.rows,
            cols=self.cols,
            num_mines=self.num_mines,
            seed=42,
            reward_mode=self.reward_mode,
            frontier_bonus=self.frontier_bonus,
        )
        self.observation = self.env.reset()
        self.episode_steps = 0
        self.episode_reward = 0.0
        if board_changed:
            self._build_board_grid()
        self._sync_board_controls()
        self._refresh_board()
        self._update_summary()

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _append_log(self, message: str) -> None:
        """Append one line to the on-screen preparation log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update()

    def _clear_log(self) -> None:
        """Clear previous preparation logs."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _set_progress(self, current: int, total: int, label: str) -> None:
        """Update progress widgets during agent preparation."""
        if total <= 0:
            self.progress_bar["value"] = 0
            self.progress_var.set("")
            return

        percent = (current / total) * 100.0
        self.progress_bar["value"] = percent
        self.progress_var.set(f"{label}: {current}/{total} episodes ({percent:.1f}%)")
        self.root.update()

    def _clear_progress(self) -> None:
        """Reset progress widgets after preparation completes."""
        self.progress_bar["value"] = 0
        self.progress_var.set("")

    def _refresh_board(self) -> None:
        board = self.env.visible_board
        for r in range(self.rows):
            for c in range(self.cols):
                value = int(board[r, c])
                label = self.cell_labels[r][c]

                if self.env.lost and self.env.last_mine_hit == (r, c):
                    text = "*"
                    bg = "#ff6b6b"
                    fg = "black"
                elif value == -1:
                    text = "?"
                    bg = "#d9d9d9"
                    fg = "black"
                elif value == 0:
                    text = "."
                    bg = "#ffffff"
                    fg = "#444444"
                else:
                    text = str(value)
                    bg = "#ffffff"
                    fg = "#1f4e79"

                label.config(text=text, bg=bg, fg=fg)

    def _update_summary(self) -> None:
        self.summary_var.set(
            f"steps={self.episode_steps} | total_reward={self.episode_reward:.2f} | done={self.env.done}"
        )

    def _update_model_info(self) -> None:
        """Show the current board setup used by the UI."""
        self.model_info_var.set(
            f"board={self.rows}x{self.cols} | mines={self.num_mines} "
            f"| reward={self.reward_mode} | frontier_bonus={self.frontier_bonus}"
        )

    def _read_board_controls(self) -> tuple[int, int, int, str, float]:
        """Read and validate board settings from the UI controls."""
        rows = int(self.rows_var.get())
        cols = int(self.cols_var.get())
        num_mines = int(self.num_mines_var.get())
        reward_mode = self.reward_mode_var.get()
        frontier_bonus = float(self.frontier_bonus_var.get())
        if reward_mode not in {"classic", "progress", "frontier"}:
            raise ValueError("reward mode must be 'classic', 'progress', or 'frontier'")
        # Let MinesweeperEnv enforce the core board validation rules.
        MinesweeperEnv(
            rows=rows,
            cols=cols,
            num_mines=num_mines,
            reward_mode=reward_mode,
            frontier_bonus=frontier_bonus,
        )
        return rows, cols, num_mines, reward_mode, frontier_bonus

    def apply_board_settings(self) -> None:
        """Apply manually entered board settings and invalidate any prepared DQN."""
        try:
            rows, cols, num_mines, reward_mode, frontier_bonus = self._read_board_controls()
        except ValueError as exc:
            self._set_status(f"Invalid board settings: {exc}")
            return

        changed = (
            rows != self.rows
            or cols != self.cols
            or num_mines != self.num_mines
            or reward_mode != self.reward_mode
            or frontier_bonus != self.frontier_bonus
        )
        self._reset_environment_for_model(
            rows=rows,
            cols=cols,
            num_mines=num_mines,
            reward_mode=reward_mode,
            frontier_bonus=frontier_bonus,
        )
        if changed:
            self.agent = RandomAgent(seed=123)
            self._set_status(
                "Applied board settings. Prepared agent was reset to Random; prepare/load a matching agent."
            )
        else:
            self._set_status("Board settings already match the active game.")
        self._update_model_info()

    def new_game(self) -> None:
        try:
            rows, cols, num_mines, reward_mode, frontier_bonus = self._read_board_controls()
            if (
                rows != self.rows
                or cols != self.cols
                or num_mines != self.num_mines
                or reward_mode != self.reward_mode
                or frontier_bonus != self.frontier_bonus
            ):
                self._reset_environment_for_model(
                    rows=rows,
                    cols=cols,
                    num_mines=num_mines,
                    reward_mode=reward_mode,
                    frontier_bonus=frontier_bonus,
                )
                self.agent = RandomAgent(seed=123)
                self._update_model_info()
                self._set_status(
                    "Started a new game with updated board settings. Prepared agent was reset to Random."
                )
                return
        except ValueError as exc:
            self._set_status(f"Invalid board settings: {exc}")
            return

        self.observation = self.env.reset()
        self.episode_steps = 0
        self.episode_reward = 0.0
        self._refresh_board()
        self._update_summary()
        self._set_status("Started a new game.")

    def prepare_agent(self) -> None:
        agent_name = self.agent_var.get()
        episodes = self._parse_episodes()
        self._clear_log()

        self.prepare_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.apply_board_button.config(state=tk.DISABLED)
        self.refresh_models_button.config(state=tk.DISABLED)
        self.root.update()

        try:
            if agent_name not in {"Saved DQN Checkpoint", "Saved Best DQN"}:
                rows, cols, num_mines, reward_mode, frontier_bonus = self._read_board_controls()
                if (
                    rows != self.rows
                    or cols != self.cols
                    or num_mines != self.num_mines
                    or reward_mode != self.reward_mode
                    or frontier_bonus != self.frontier_bonus
                ):
                    self._reset_environment_for_model(
                        rows=rows,
                        cols=cols,
                        num_mines=num_mines,
                        reward_mode=reward_mode,
                        frontier_bonus=frontier_bonus,
                    )
                    self._update_model_info()
                self._append_log(
                    f"Using board setup: {self.rows}x{self.cols}, "
                    f"mines={self.num_mines}, reward={self.reward_mode}, "
                    f"frontier_bonus={self.frontier_bonus}"
                )

            if agent_name in {"Saved DQN Checkpoint", "Saved Best DQN"}:
                self.agent = self._load_saved_dqn_checkpoint()
                self._clear_progress()
                self._append_log("Loaded saved DQN checkpoint. No training needed.")
                self._set_status("Loaded saved DQN checkpoint. Click 'Next Move' to play.")
            elif agent_name == "Random":
                self.agent = RandomAgent(seed=123)
                self._clear_progress()
                self._append_log("Prepared Random agent instantly (no training needed).")
                self._set_status("Prepared Random agent.")
            elif agent_name == "Q-Learning":
                self._append_log(f"Starting Q-Learning training for {episodes} episodes.")
                self._set_status(f"Training Q-Learning for {episodes} episodes...")
                self.root.update()
                self.agent = self._train_qlearning_agent(episodes)
                self._append_log("Q-Learning training finished. Switched to greedy policy (epsilon=0).")
                self._set_status(f"Q-Learning ready (trained {episodes} episodes, epsilon=0).")
            elif agent_name == "DQN-MLP":
                self._append_log(f"Starting DQN-MLP training for {episodes} episodes.")
                self._set_status(f"Training DQN-MLP for {episodes} episodes...")
                self.root.update()
                self.agent = self._train_dqn_agent(episodes, model_type="mlp")
                self._append_log("DQN-MLP training finished. Switched to greedy policy (epsilon=0).")
                self._set_status(f"DQN-MLP ready (trained {episodes} episodes, epsilon=0).")
            elif agent_name == "DQN-CNN":
                self._append_log(f"Starting DQN-CNN training for {episodes} episodes.")
                self._set_status(f"Training DQN-CNN for {episodes} episodes...")
                self.root.update()
                self.agent = self._train_dqn_agent(episodes, model_type="cnn")
                self._append_log("DQN-CNN training finished. Switched to greedy policy (epsilon=0).")
                self._set_status(f"DQN-CNN ready (trained {episodes} episodes, epsilon=0).")
            elif agent_name == "DQN-CNN-DEEP":
                self._append_log(f"Starting DQN-CNN-DEEP training for {episodes} episodes.")
                self._set_status(f"Training DQN-CNN-DEEP for {episodes} episodes...")
                self.root.update()
                self.agent = self._train_dqn_agent(episodes, model_type="cnn_deep")
                if isinstance(self.agent, DQNAgent):
                    self._save_current_dqn_checkpoint(self.agent, episodes)
                self._append_log("DQN-CNN-DEEP training finished. Switched to greedy policy (epsilon=0).")
                self._set_status(f"DQN-CNN-DEEP ready (trained {episodes} episodes, epsilon=0).")
            else:
                self._set_status(f"Unknown agent type: {agent_name}")
        except Exception as exc:  # Keep UI robust for beginner usage.
            self._clear_progress()
            self._append_log(f"Preparation failed: {exc}")
            self._set_status(f"Failed to prepare agent: {exc}")
        finally:
            self.prepare_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.apply_board_button.config(state=tk.NORMAL)
            self.refresh_models_button.config(state=tk.NORMAL)

    def next_move(self) -> None:
        if self.env.done:
            self._set_status("Game already ended. Click 'New Game' to start another episode.")
            return

        action = self.agent.select_action(self.observation)
        next_obs, reward, done, info = self.env.step(action)
        self.observation = next_obs
        self.episode_steps += 1
        self.episode_reward += reward

        self._refresh_board()
        self._update_summary()

        self._set_status(
            f"action={action}, reward={reward:.2f}, done={done}, info={info}"
        )

    def _parse_episodes(self) -> int:
        try:
            value = int(self.train_episodes_var.get())
        except ValueError:
            return 5000
        return max(1, value)

    def _load_saved_dqn_checkpoint(self) -> DQNAgent:
        """Load the selected DQN checkpoint and resize the UI to match it."""
        self._refresh_checkpoint_list()
        checkpoint_path = self._selected_checkpoint_path()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}. "
                "Create it with compare_dqn_models.py --save-model-path first."
            )

        rows, cols, num_mines, reward_mode, frontier_bonus, config, metadata = self._read_checkpoint_details(
            checkpoint_path
        )
        agent = DQNAgent.load_checkpoint(checkpoint_path)
        if agent.rows != rows or agent.cols != cols:
            raise ValueError(
                f"Checkpoint config board is {agent.rows}x{agent.cols}, "
                f"but checkpoint metadata says {rows}x{cols}."
            )
        agent.epsilon = 0.0
        self._reset_environment_for_model(
            rows=rows,
            cols=cols,
            num_mines=num_mines,
            reward_mode=reward_mode,
            frontier_bonus=frontier_bonus,
        )
        self._update_model_info()

        metrics = metadata.get("metrics", metadata)
        win_rate = metrics.get("win_rate")
        avg_reward = metrics.get("average_reward")
        avg_steps = metrics.get("average_steps")
        metric_text = ""
        if win_rate is not None and avg_reward is not None and avg_steps is not None:
            metric_text = (
                f", saved_eval_win_rate={float(win_rate):.2%}, "
                f"avg_reward={float(avg_reward):.3f}, avg_steps={float(avg_steps):.3f}"
            )

        self._append_log(f"Checkpoint path: {checkpoint_path}")
        self._append_log(
            f"Loaded setup: model_type={agent.model_type}, replay={agent.replay_type}, reward={self.reward_mode}, "
            f"board={self.rows}x{self.cols}, mines={self.num_mines}, "
            f"frontier_bonus={self.frontier_bonus}, "
            f"epsilon=0 for UI playback, lr={config.get('lr', 'unknown')}, "
            f"frontier_channel={config.get('use_frontier_channel', False)}{metric_text}"
        )
        return agent

    def _save_current_dqn_checkpoint(self, agent: DQNAgent, episodes: int) -> None:
        """Save the latest UI-trained deep CNN checkpoint for later playback."""
        checkpoint_path = (
            self.models_dir
            / f"ui_{agent.model_type}_{self.rows}x{self.cols}_{self.num_mines}m_seed{self.dqn_seed}_{episodes}eps.pt"
        )
        agent.save_checkpoint(
            checkpoint_path,
            metadata={
                "rows": self.rows,
                "cols": self.cols,
                "num_mines": self.num_mines,
                "seed": self.dqn_seed,
                "num_episodes": episodes,
                "model_type": agent.model_type,
                "replay_type": agent.replay_type,
                "reward_mode": self.reward_mode,
                "frontier_bonus": self.frontier_bonus,
                "epsilon_min": self.dqn_epsilon_min,
                "epsilon_decay": self.dqn_epsilon_decay,
                "source": "ui_minesweeper.py",
            },
        )
        self._append_log(f"Saved DQN checkpoint to: {checkpoint_path}")
        self._refresh_checkpoint_list()

    def _train_qlearning_agent(self, episodes: int) -> QLearningAgent:
        agent = QLearningAgent(
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.999,
            seed=123,
        )
        env = MinesweeperEnv(
            rows=self.rows,
            cols=self.cols,
            num_mines=self.num_mines,
            seed=123,
            reward_mode=self.reward_mode,
            frontier_bonus=self.frontier_bonus,
        )
        log_every = max(1, episodes // 20)
        recent_rewards: list[float] = []
        recent_wins: list[int] = []

        for episode in range(1, episodes + 1):
            obs = env.reset()
            done = False
            total_reward = 0.0
            won = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward
                if done and info.get("result") == "win":
                    won = True
            agent.decay_epsilon()
            recent_rewards.append(total_reward)
            recent_wins.append(1 if won else 0)
            self._set_progress(episode, episodes, "Training Q-Learning")
            if episode == 1 or episode % log_every == 0 or episode == episodes:
                window_size = min(100, len(recent_rewards))
                avg_reward = float(np.mean(recent_rewards[-window_size:]))
                win_rate = float(np.mean(recent_wins[-window_size:])) * 100.0
                self._append_log(
                    f"Q-Learning episode {episode}/{episodes} | epsilon={agent.epsilon:.3f} "
                    f"| episode_reward={total_reward:.2f} | last{window_size}_avg_reward={avg_reward:.2f} "
                    f"| last{window_size}_win_rate={win_rate:.1f}%"
                )

        # Use greedy policy in UI playback.
        agent.epsilon = 0.0
        self._clear_progress()
        return agent

    def _train_dqn_agent(self, episodes: int, model_type: str) -> DQNAgent:
        agent = DQNAgent(
            rows=self.rows,
            cols=self.cols,
            lr=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=self.dqn_epsilon_min,
            epsilon_decay=self.dqn_epsilon_decay,
            batch_size=64,
            memory_size=20000,
            target_update_every=200,
            seed=self.dqn_seed,
            model_type=model_type,
            replay_type="uniform",
        )
        env = MinesweeperEnv(
            rows=self.rows,
            cols=self.cols,
            num_mines=self.num_mines,
            seed=self.dqn_seed,
            reward_mode=self.reward_mode,
            frontier_bonus=self.frontier_bonus,
        )

        model_labels = {
            "mlp": "Training DQN-MLP",
            "cnn": "Training DQN-CNN",
            "cnn_deep": "Training DQN-CNN-DEEP",
        }
        model_label = model_labels[model_type]
        self._append_log(
            "DQN setup: "
            f"board={self.rows}x{self.cols}, mines={self.num_mines}, "
            f"model_type={model_type}, replay=uniform, reward={self.reward_mode}, "
            f"frontier_bonus={self.frontier_bonus}, seed={self.dqn_seed}, "
            f"epsilon_min={self.dqn_epsilon_min}, epsilon_decay={self.dqn_epsilon_decay}"
        )
        log_every = max(1, episodes // 20)
        recent_rewards: list[float] = []
        recent_wins: list[int] = []
        recent_losses: list[float] = []

        for episode in range(1, episodes + 1):
            obs = env.reset()
            done = False
            total_reward = 0.0
            episode_losses: list[float] = []
            won = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                loss = agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward
                if loss is not None:
                    episode_losses.append(loss)
                if done and info.get("result") == "win":
                    won = True
            agent.decay_epsilon()
            recent_rewards.append(total_reward)
            recent_wins.append(1 if won else 0)
            if episode_losses:
                recent_losses.append(float(np.mean(episode_losses)))
            self._set_progress(episode, episodes, model_label)
            if episode == 1 or episode % log_every == 0 or episode == episodes:
                window_size = min(100, len(recent_rewards))
                avg_reward = float(np.mean(recent_rewards[-window_size:]))
                win_rate = float(np.mean(recent_wins[-window_size:])) * 100.0
                if recent_losses:
                    loss_window = min(100, len(recent_losses))
                    avg_loss = float(np.mean(recent_losses[-loss_window:]))
                    loss_text = f"last{loss_window}_avg_loss={avg_loss:.4f}"
                else:
                    loss_text = "avg_loss=n/a"
                self._append_log(
                    f"{model_label} episode {episode}/{episodes} | epsilon={agent.epsilon:.3f} "
                    f"| episode_reward={total_reward:.2f} | last{window_size}_avg_reward={avg_reward:.2f} "
                    f"| last{window_size}_win_rate={win_rate:.1f}% | {loss_text}"
                )

        # Use greedy policy in UI playback.
        agent.epsilon = 0.0
        self._clear_progress()
        return agent


def main() -> None:
    root = tk.Tk()
    app = MinesweeperUI(root)
    app._update_summary()
    root.mainloop()


if __name__ == "__main__":
    main()
