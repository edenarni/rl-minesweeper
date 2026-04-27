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

## Current Default Workflow
- Main model to improve: `CNN_DEEP DQN`
- Main seed for fast iteration: `55`
- Default focused evaluation command:
  - `python3 compare_dqn_models.py --models cnn_deep --replay-modes uniform --reward-modes classic --seeds 55 --num-episodes 10000 --eval-games 1000 --progress-every 500 --epsilon-min 0.001 --epsilon-decay 0.999`
- Best saved 5x5 checkpoint:
  - `models/best_cnn_deep_5x5_seed55.pt`
- Multi-seed comparisons should only be repeated when a change looks promising enough to justify a broader validation pass.

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

## Entry 005 - Huber Loss (SmoothL1Loss)
- Date: 2026-04-25
- Change:
  - Replaced DQN loss from `MSELoss` to `SmoothL1Loss` (Huber loss).
  - Kept the same 2-channel input, replay buffer, target network, epsilon schedule, and model architectures.
- Hypothesis: Huber loss might make training more robust to large TD-error spikes and improve stability.
- Command:
  - `python3 compare_dqn_models.py --num-episodes 5000 --eval-games 1000 --seeds 11 22 33 44 55 --progress-every 1000`
- Per-seed results (MLP with Huber):
  - Seed 11: win `34.90%`, reward `4.594`, steps `4.775`
  - Seed 22: win `35.40%`, reward `4.727`, steps `4.753`
  - Seed 33: win `27.90%`, reward `1.935`, steps `4.286`
  - Seed 44: win `33.10%`, reward `3.858`, steps `4.597`
  - Seed 55: win `33.30%`, reward `4.050`, steps `4.727`
- Per-seed results (CNN with Huber):
  - Seed 11: win `50.10%`, reward `9.703`, steps `5.172`
  - Seed 22: win `53.40%`, reward `10.987`, steps `5.433`
  - Seed 33: win `49.50%`, reward `9.584`, steps `5.239`
  - Seed 44: win `53.10%`, reward `10.603`, steps `5.142`
  - Seed 55: win `49.40%`, reward `9.503`, steps `5.189`
- Mean ± std across seeds:
  - MLP DQN: win `32.92% ± 2.66%`, reward `3.833 ± 1.003`, steps `4.628 ± 0.182`
  - CNN DQN: win `51.10% ± 1.77%`, reward `10.076 ± 0.603`, steps `5.235 ± 0.104`
- Outcome:
  - Huber loss degraded both architectures relative to Entry 004.
  - Previous MSE-based CNN remained better: `57.96%` win rate vs `51.10%` with Huber.
  - Recommendation: revert Huber loss and keep MSE for now.
  - Status: reverted after this experiment.

## Entry 006 - CNN Seed 55: `epsilon_min=0.05` vs `0.1`
- Date: 2026-04-25
- Change:
  - Compared the current best setup with only one change:
    - `epsilon_min=0.05`
    - `epsilon_min=0.1`
  - Focused only on the current main target:
    - `CNN DQN`
    - seed `55`
- Hypothesis: A slightly higher exploration floor (`0.1`) might help the CNN keep discovering useful states later in training.
- Commands:
  - `python3 compare_dqn_models.py --models cnn --seeds 55 --num-episodes 5000 --eval-games 1000 --progress-every 1000 --epsilon-min 0.05`
  - `python3 compare_dqn_models.py --models cnn --seeds 55 --num-episodes 5000 --eval-games 1000 --progress-every 1000 --epsilon-min 0.1`
- Result:
  - `epsilon_min=0.05`: win `60.20%`, reward `13.262`, steps `5.600`
  - `epsilon_min=0.1`: win `59.10%`, reward `12.990`, steps `5.669`
- Outcome:
  - `0.05` stayed slightly better than `0.1` on the current focused evaluation.
  - Keep `epsilon_min=0.05` as the default for now.

## Entry 007 - Prioritized Experience Replay
- Date: 2026-04-25
- Change:
  - Added prioritized replay as an optional replay mode.
  - Kept the current CNN Double DQN architecture unchanged.
  - Kept the environment unchanged.
  - Kept the epsilon schedule unchanged.
  - Priorities are updated from absolute TD error plus `priority_epsilon`.
  - Importance-sampling weights are applied to the per-sample MSE loss.
- Default PER command:
  - `python3 compare_dqn_models.py --models cnn --replay-modes uniform prioritized --seeds 55 --num-episodes 5000 --eval-games 1000 --progress-every 1000`
- Default PER result:
  - Uniform replay: win `64.00%`, reward `14.405`, steps `5.565`
  - Prioritized replay (`alpha=0.6`, `beta_start=0.4`): win `57.30%`, reward `12.425`, steps `5.662`
- Tuned PER command:
  - `python3 compare_dqn_models.py --models cnn --replay-modes prioritized --seeds 55 --num-episodes 5000 --eval-games 1000 --progress-every 1000 --alpha 0.4 --beta-start 0.6 --beta-end 1.0 --priority-epsilon 1e-5`
- Tuned PER result:
  - Prioritized replay (`alpha=0.4`, `beta_start=0.6`): win `61.60%`, reward `13.757`, steps `5.661`
- Outcome:
  - Gentler PER improved over default PER.
  - Uniform replay still remains the best current seed-55 result: win `64.00%`, reward `14.405`.
  - Recommendation: keep PER available as an experiment option, but keep uniform replay as the default.

## Entry 008 - Reward Shaping: Newly Revealed Cells
- Date: 2026-04-26
- Change:
  - Kept existing penalties unchanged:
    - invalid action: `-5`
    - already revealed action: `-2`
    - mine/loss: `-10`
  - Added `reward_mode=progress`:
    - safe move reward: `0.5 * newly_revealed`
    - win reward: safe move reward plus `25`
  - Kept architecture, replay method, epsilon schedule, and environment rules unchanged.
- Command:
  - `python3 compare_dqn_models.py --models cnn --replay-modes uniform --reward-modes classic progress --seeds 55 --num-episodes 10000 --eval-games 1000 --progress-every 500`
- Result:
  - Classic reward: win `58.70%`, reward `12.907`, steps `5.710`
  - Progress reward: win `62.40%`, reward `20.297`, steps `5.459`
- Outcome:
  - On focused seed `55`, progress reward improved win rate and average steps.
  - Average reward is not directly comparable because the reward scale changed.
  - This is promising, but still needs a 3-seed validation before treating it as the new default.

## Entry 009 - Lower Exploration Floor for CNN_DEEP
- Date: 2026-04-26
- Change:
  - Kept the current best focused architecture and setup:
    - `CNN_DEEP DQN`
    - uniform replay
    - classic reward
    - seed `55`
  - Changed only the exploration schedule:
    - `epsilon_min=0.001`
    - `epsilon_decay=0.999`
- Hypothesis:
  - Minesweeper is highly sensitive to random bad clicks, so lowering the late-training random-action floor may improve final policy quality.
  - Slower decay keeps exploration high for longer before settling near-greedy behavior.
- Command:
  - `python3 compare_dqn_models.py --models cnn_deep --replay-modes uniform --reward-modes classic --seeds 55 --num-episodes 10000 --eval-games 1000 --progress-every 500 --epsilon-min 0.001 --epsilon-decay 0.999`
- Progress snapshots:
  - Episode 500: win `10.80%`, reward `-3.670`, epsilon `0.606`
  - Episode 3000: win `46.60%`, reward `8.172`, epsilon `0.050`
  - Episode 6000: win `65.80%`, reward `15.068`, epsilon `0.002`
  - Episode 8500: win `71.00%`, reward `17.118`, epsilon `0.001`
  - Episode 10000: win `69.20%`, reward `15.972`, epsilon `0.001`
- Final Result:
  - CNN_DEEP DQN: win `69.20%`, reward `16.085`, steps `5.633`
- Comparison to previous best focused classic-reward result:
  - Previous best: win `64.30%`, reward `14.425`, steps `5.492`
  - New result: win `69.20%`, reward `16.085`, steps `5.633`
- Outcome:
  - Lowering `epsilon_min` from the earlier `0.05` floor to `0.001` improved the focused seed-55 result.
  - This is now the best known focused configuration.
  - Because this was only one seed, the next validation step should be a 3-seed run before treating it as robust.

## Entry 010 - Save Best 5x5 CNN_DEEP Checkpoint for UI Playback
- Date: 2026-04-26
- Change:
  - Added DQN checkpoint save/load support.
  - Saved the current best focused 5x5 model weights to:
    - `models/best_cnn_deep_5x5_seed55.pt`
  - Added `Saved Best DQN` to the UI agent selector.
  - UI can now load the saved checkpoint with `epsilon=0.0` and play without retraining.
- Command:
  - `python3 compare_dqn_models.py --models cnn_deep --replay-modes uniform --reward-modes classic --seeds 55 --num-episodes 10000 --eval-games 1000 --progress-every 500 --epsilon-min 0.001 --epsilon-decay 0.999 --save-model-path models/best_cnn_deep_5x5_seed55.pt`
- Final Result:
  - CNN_DEEP DQN: win `69.20%`, reward `16.085`, steps `5.633`
- Checkpoint verification:
  - Loaded checkpoint successfully as `cnn_deep`, board `5x5`, `epsilon=0.0`
- Outcome:
  - The UI no longer needs to retrain the best model before playback.
  - Use `Saved Best DQN` in `python3 ui_minesweeper.py` to watch the saved model play.

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
