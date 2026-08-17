# Experiments Overview

This page is the GitHub-friendly index for the full experiment history in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

If you want the detailed commands, progress snapshots, and exact per-run notes,
use the full log. If you want the portfolio version, start here.

## What This Project Explored

- State representation for partially observed board state
- MLP vs CNN vs deeper CNN DQN architectures
- Uniform vs prioritized replay
- Reward shaping with `classic`, `progress`, and `frontier` rewards
- Exploration schedule tuning
- Curriculum learning across board sizes
- Checkpoint reuse for UI playback and continued training

## Main Takeaways

| Area | Conclusion |
| --- | --- |
| State encoding | A 2-channel representation separating hidden cells from revealed values was a major improvement over a single mixed scalar input. |
| Architecture | CNN-based DQNs consistently outperformed the MLP baseline on Minesweeper. |
| Loss function | Huber loss underperformed the original MSE setup in this project. |
| Replay strategy | Prioritized replay was useful to test, but uniform replay remained stronger in the recorded runs. |
| Reward shaping | `progress` and later `frontier` reward shaping produced stronger focused results than the original reward setup. |
| Exploration | Lower late-training exploration improved strong focused runs because random endgame clicks are especially costly in Minesweeper. |
| Training strategy | Curriculum-style training made larger-board experiments more practical and supported checkpoint transfer between stages. |

## Representative Results

| Experiment | Setup | Result |
| --- | --- | --- |
| Multi-seed architecture comparison | `5x5`, 3 mines, 5 seeds | CNN DQN: `57.96% ± 2.16%` win rate vs MLP DQN: `46.22% ± 1.03%` |
| Best focused `5x5` classic run | `cnn_deep`, seed `55` | `69.20%` win rate, `16.085` average reward, `5.633` average steps |
| Best documented `8x8` frontier run | `cnn_deep`, seed `55`, frontier reward | `49.10%` win rate, `23.974` average reward, `13.967` average steps |

## Experiment Timeline

| Entries | Theme | Summary |
| --- | --- | --- |
| 001-003 | Baseline and input encoding | Established the first DQN baseline, tested normalization-only input, then moved to the stronger 2-channel encoding. |
| 004-007 | Core DQN comparisons | Compared MLP vs CNN, tested Huber loss, tuned exploration floor, and evaluated prioritized replay. |
| 008-010 | Focused `5x5` improvements | Tested progress reward shaping, improved `cnn_deep` exploration, and saved the best `5x5` checkpoint for the UI. |
| 011-015 | Larger-board transition | Moved from `5x5` to larger boards, generalized scripts, and introduced curriculum-based training. |
| 016-019 | Frontier fine-tuning | Used saved curriculum checkpoints as starting points for frontier-reward experiments on `8x8`. |

## Files To Read

- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): full chronological log with commands and metrics
- [README.md](README.md): portfolio summary and project overview
- [models/README.md](models/README.md): explanation of which checkpoints are intentionally kept in the repo
