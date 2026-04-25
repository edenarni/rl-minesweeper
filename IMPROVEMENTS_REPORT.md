# Improvements Report

This file is an ongoing experiment log for model/environment changes.
After each improvement, add one new entry with the exact training command and final metrics.

## Standard Evaluation Setup
- Environment: `MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)`
- DQN training: `python3 train_dqn.py --num-episodes 5000 --progress-every 500`
- Final evaluation: 1000 games with DQN epsilon set to `0.0`
- Baselines in report:
  - `RandomAgent` (1000 eval games)
  - `QLearningAgent` (trained, then evaluated with epsilon `0.0`)

## Metrics Legend
- Win rate: fraction of games won
- Average reward: average total reward per episode
- Average steps: average number of steps per episode

## Entry 001 - Initial DQN Baseline
- Date: 2026-04-24
- Change: Initial DQN implementation (`agents/dqn_agent.py`, `train_dqn.py`)
- Hypothesis: DQN should beat RandomAgent and tabular QLearningAgent on this 5x5 setting.
- Result: See the comparison plot below.
- Notes: This result came from the first full DQN run after implementation.

## Entry 002 - Input Scaling for Revealed Cells
- Date: 2026-04-24
- Change: Normalize DQN inputs in `_flatten_observation`:
  - unknown stays `-1.0`
  - revealed `0..8` is scaled to `0.0..1.0` by dividing by `8.0`
- Hypothesis: More consistent input scale may improve stability/learning quality.
- Command:
  - `python3 train_dqn.py --num-episodes 5000 --progress-every 500`
- Progress snapshots:
  - Episode 500: win `11.80%`, avg_reward `-3.406`, avg_steps `3.936`
  - Episode 2500: win `20.60%`, avg_reward `-0.030`, avg_steps `4.584`
  - Episode 5000: win `28.00%`, avg_reward `2.338`, avg_steps `4.658`
- Final Result: See the comparison plot below.
- Outcome: **Degraded vs Entry 001** (clear in the plot). This change was reverted in the next experiment.

## Entry 003 - 2-Channel State Input (Hidden Mask + Revealed Values)
- Date: 2026-04-24
- Change:
  - Reverted single-channel normalization-only input.
  - Switched DQN state to 2 channels flattened and concatenated:
    - Channel 1: hidden mask (`1.0` hidden, `0.0` revealed)
    - Channel 2: revealed normalized values (`0..8` mapped to `0.0..1.0`, hidden=`0.0`)
  - Updated target-mask logic to read valid actions from hidden-mask channel.
- Hypothesis: Separating "visibility" signal from "number value" signal should be more learnable than mixing both in one scalar.
- Command:
  - `python3 train_dqn.py --num-episodes 5000 --progress-every 500 --loss-plot-path dqn_loss_curve_entry003.png`
- Progress snapshots:
  - Episode 500: win `9.00%`, avg_reward `-4.288`, avg_steps `3.922`
  - Episode 2500: win `28.20%`, avg_reward `2.518`, avg_steps `4.776`
  - Episode 5000: win `42.40%`, avg_reward `7.040`, avg_steps `4.896`
- Final Result: See the comparison plot below.
- Outcome: Strong improvement. Better than Entry 001 and much better than Entry 002. Keep this as the new best configuration.

## Result Plot
![Encoding Comparison Plot](/Users/edenar/Desktop/rl-minesweeper/plots/encoding_comparison.svg)

## Entry 004 - Reliable 5-Seed Evaluation + CNN DQN Variant
- Date: 2026-04-25
- Change:
  - Added reliable multi-seed evaluation for DQN.
  - Added CNN-based DQN variant while keeping the same 2-channel input:
    - Channel 1: hidden mask
    - Channel 2: revealed normalized values
  - Preserved replay buffer, target network, epsilon-greedy, and valid-action masking.
- Hypothesis: CNN should use board structure better than MLP and improve win rate/reward.
- Command:
  - `python3 compare_dqn_models.py --num-episodes 5000 --eval-games 1000 --seeds 11 22 33 44 55 --progress-every 1000`
- Per-seed results (MLP):
  - Seed 11: win `44.50%`, reward `7.829`, steps `5.034`
  - Seed 22: win `45.80%`, reward `8.310`, steps `5.112`
  - Seed 33: win `46.70%`, reward `8.122`, steps `4.645`
  - Seed 44: win `46.50%`, reward `8.155`, steps `4.740`
  - Seed 55: win `47.60%`, reward `8.731`, steps `4.975`
- Per-seed results (CNN):
  - Seed 11: win `59.60%`, reward `13.176`, steps `5.700`
  - Seed 22: win `56.80%`, reward `12.064`, steps `5.456`
  - Seed 33: win `58.90%`, reward `12.703`, steps `5.444`
  - Seed 44: win `54.30%`, reward `11.329`, steps `5.496`
  - Seed 55: win `60.20%`, reward `13.262`, steps `5.600`
- Mean ± std across seeds:
  - MLP DQN: win `46.22% ± 1.03%`, reward `8.229 ± 0.295`, steps `4.901 ± 0.178`
  - CNN DQN: win `57.96% ± 2.16%`, reward `12.507 ± 0.727`, steps `5.539 ± 0.097`
- Outcome: CNN clearly outperformed MLP on win rate and reward across all 5 seeds. New best model is CNN DQN.

## Template For Next Entries
Copy this block for each new improvement:

```text
## Entry XXX - <Short title>
- Date: YYYY-MM-DD
- Change:
- Hypothesis:
- Command:
- Progress snapshots:
- Final Result:
  - RandomAgent: win `..`, reward `..`, steps `..`
  - QLearningAgent: win `..`, reward `..`, steps `..`
  - DQNAgent: win `..`, reward `..`, steps `..`
- Delta vs RandomAgent:
- Delta vs QLearningAgent:
- Outcome:
```
