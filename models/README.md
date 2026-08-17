# Saved Checkpoints

This directory intentionally keeps a small set of checkpoints for UI demos and
quick evaluation without retraining:

- `best_cnn_deep_5x5_seed55.pt`: strongest small-board demo checkpoint used by the UI by default
- `best_cnn_deep_8x8_10m_progress_seed55.pt`: compact `8x8` progress-reward checkpoint
- `best_cnn_deep_8x8_10m_curriculum_seed55.pt`: compact `8x8` curriculum checkpoint
- `best_cnn_deep_10x10_15m_seed55.pt`: larger-board demo checkpoint

Long experimental checkpoints and regenerated artifacts are intentionally left
out of the portfolio repo to keep the repository focused and lightweight.
