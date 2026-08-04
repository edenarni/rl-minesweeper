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
