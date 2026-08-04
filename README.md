# RL Minesweeper

A small Python reinforcement-learning project built around a custom Minesweeper environment.

The repo includes:
- A NumPy-based Minesweeper environment with several reward schemes
- Three agent types: random, tabular Q-learning, and Deep Q-Network (DQN)
- Training and evaluation scripts for baselines and DQN variants
- Experiment scripts for replay-mode and curriculum-learning comparisons
- A Tkinter UI for stepping through agent moves

## Project Structure

```text
.
├── agents/
│   ├── dqn_agent.py
│   ├── q_learning_agent.py
│   └── random_agent.py
├── models/                     # saved checkpoints
├── plots/                      # generated plots
├── compare_curriculum_dqn.py   # direct vs curriculum DQN training
├── compare_dqn_models.py       # compare DQN architectures/replay modes
├── evaluate_random_agent.py    # random baseline evaluation
├── main.py                     # simple random-play demo
├── minesweeper_env.py          # core environment
├── test_dqn_agent.py
├── test_env.py
├── train_dqn.py
├── train_q_learning.py
└── ui_minesweeper.py
```

## Environment

`MinesweeperEnv` exposes a simple RL-style API:
- `reset() -> observation`
- `step((row, col)) -> observation, reward, done, info`

Board conventions:
- Hidden cells in observations are `-1`
- Revealed safe cells are `0..8`
- Mines are stored only in the hidden board

Supported reward modes:
- `classic`: fixed reward for safe moves, penalty for invalid/repeated moves, large loss penalty, win bonus
- `progress`: rewards the number of newly revealed safe cells
- `frontier`: classic rewards plus an extra bonus for safe moves adjacent to revealed numbered cells

## Agents

### RandomAgent
Chooses a random unrevealed cell.

### QLearningAgent
Tabular baseline trained directly on environment transitions.

### DQNAgent
PyTorch implementation with:
- `mlp`, `cnn`, and `cnn_deep` model variants
- uniform and prioritized replay support
- optional frontier input channel for CNN-based models
- checkpoint save/load support

## Requirements

The codebase uses:
- Python 3
- `numpy`
- `torch`
- `tkinter` for the UI
- `pytest` for tests

Optional:
- `matplotlib` for PNG loss plots in DQN training

If `matplotlib` is not installed, `train_dqn.py` falls back to writing a simple SVG loss plot.

## Quick Start

Run the simple random-play demo:

```bash
python3 main.py
```

Evaluate the random baseline:

```bash
python3 evaluate_random_agent.py
```

Train tabular Q-learning:

```bash
python3 train_q_learning.py --num-episodes 5000
```

Train a DQN:

```bash
python3 train_dqn.py --num-episodes 5000 --model-type cnn
```

Open the step-by-step UI:

```bash
python3 ui_minesweeper.py
```

Run tests:

```bash
pytest
```

## Main Scripts

### `train_q_learning.py`
Trains a tabular Q-learning agent and compares it against the random baseline.

Example:

```bash
python3 train_q_learning.py \
  --rows 5 \
  --cols 5 \
  --num-mines 3 \
  --num-episodes 5000 \
  --reward-mode classic
```

### `train_dqn.py`
Trains a DQN agent, evaluates it greedily, compares it to random and optionally to Q-learning, and saves a loss plot.

Useful flags:
- `--model-type {mlp,cnn,cnn_deep}`
- `--reward-mode {classic,progress,frontier}`
- `--frontier-bonus FLOAT`
- `--frontier-channel`
- `--epsilon-min FLOAT`
- `--epsilon-decay FLOAT`
- `--memory-size INT`
- `--loss-plot-path PATH`

Example:

```bash
python3 train_dqn.py \
  --rows 8 \
  --cols 8 \
  --num-mines 10 \
  --num-episodes 10000 \
  --model-type cnn_deep \
  --reward-mode frontier \
  --frontier-bonus 0.5 \
  --frontier-channel
```

### `compare_dqn_models.py`
Runs multi-seed comparisons across DQN architectures, replay modes, and reward modes.

Useful flags:
- `--models mlp cnn cnn_deep`
- `--replay-modes uniform prioritized`
- `--reward-modes classic progress frontier`
- `--seeds 55 56 57`
- `--save-model-path PATH`

Example:

```bash
python3 compare_dqn_models.py \
  --rows 8 \
  --cols 8 \
  --num-mines 10 \
  --num-episodes 10000 \
  --models cnn cnn_deep \
  --replay-modes uniform prioritized \
  --reward-modes classic frontier \
  --seeds 55 56 57
```

### `compare_curriculum_dqn.py`
Compares direct DQN training against staged curriculum training. The curriculum transfers compatible CNN weights between board sizes while starting each stage with a fresh replay buffer.

This script is intended for CNN-based experiments and supports checkpoint resume flows.

### `ui_minesweeper.py`
Tkinter viewer for:
- preparing agents
- loading saved DQN checkpoints
- changing board settings
- stepping through one move at a time

## Experiment Process

The project was developed as an iterative experiment loop rather than a single training run.

Typical process:
- Start from a simple baseline and verify the environment with tests.
- Compare baseline agents first: `RandomAgent`, then `QLearningAgent`, then DQN.
- Change one major variable at a time:
  - state encoding
  - model architecture (`mlp`, `cnn`, `cnn_deep`)
  - replay method (`uniform` vs `prioritized`)
  - reward mode (`classic`, `progress`, `frontier`)
  - exploration schedule (`epsilon_min`, `epsilon_decay`)
- Run focused experiments on one main seed for fast iteration.
- Re-run promising setups on multiple seeds before treating them as stronger evidence.
- Save the best checkpoints to `models/` so they can be reused in the UI without retraining.

The main experiment history is tracked in `IMPROVEMENTS_REPORT.md`, including commands, per-run metrics, and conclusions after each change.

## Results

The main results recorded so far are:
- Separating the DQN input into a hidden-mask channel plus normalized revealed-value channel was a clear improvement over the earlier single-channel encoding.
- CNN-based DQN outperformed the MLP DQN consistently on win rate and reward in multi-seed tests on the smaller board.
- Switching from MSE loss to Huber loss made results worse in this project, so MSE remained the preferred loss.
- Prioritized replay was useful as an experiment option, but uniform replay remained stronger than the tested prioritized-replay settings.
- Reward shaping based on newly revealed cells (`progress` mode) improved focused results enough to justify deeper follow-up experiments.
- Lowering the exploration floor and using slower epsilon decay improved the strongest focused `cnn_deep` runs.
- Curriculum-style training was added for larger-board experiments so compatible CNN weights could transfer between stages.

Representative reported numbers:
- On the `5x5` board, multi-seed comparison showed CNN DQN outperforming MLP DQN:
  - MLP DQN: `46.22% +- 1.03%` win rate
  - CNN DQN: `57.96% +- 2.16%` win rate
- A stronger focused `5x5` `cnn_deep` run reached:
  - `69.20%` win rate
  - `16.085` average reward
  - `5.633` average steps
- On focused `8x8` curriculum/frontier experiments, the repo now contains saved checkpoints from longer training runs for continued evaluation and playback.

## Conclusions

The current conclusions from the recorded experiments are:
- Spatial structure matters for Minesweeper, so CNN-based models are a better fit than a plain MLP baseline.
- State representation matters almost as much as architecture; explicitly separating hidden/revealed information made learning substantially easier.
- Not every standard RL upgrade helped here. Huber loss and the tested prioritized replay settings both underperformed simpler defaults.
- Exploration tuning has a large effect on final Minesweeper performance because late random moves are especially costly.
- The project is past the proof-of-concept stage: it now has reproducible experiments, saved checkpoints, a UI for replay, and a documented path for continuing larger-board training.

## Tests

Current tests cover:
- environment initialization and reset behavior
- invalid and repeated actions
- win/loss conditions
- frontier reward behavior
- DQN frontier-channel encoding

Run:

```bash
pytest test_env.py test_dqn_agent.py
```

## Notes

- The repo contains saved model checkpoints in `models/` from prior experiments.
- Plot files in the repo root and `plots/` are generated artifacts from training runs.
- The UI defaults to loading a saved 5x5 CNN-deep checkpoint when available.
