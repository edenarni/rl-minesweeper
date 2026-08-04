"""Basic tests for the Minesweeper environment."""

from __future__ import annotations

import numpy as np

from minesweeper_env import MinesweeperEnv


def test_hidden_board_shape_is_correct() -> None:
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=1)
    env.reset()
    assert env.hidden_board.shape == (5, 5)


def test_visible_board_shape_is_correct() -> None:
    env = MinesweeperEnv(rows=4, cols=6, num_mines=5, seed=2)
    obs = env.reset()
    assert obs.shape == (4, 6)
    assert env.visible_board.shape == (4, 6)


def test_initial_visible_board_all_unknown() -> None:
    env = MinesweeperEnv(rows=4, cols=6, num_mines=5, seed=2)
    obs = env.reset()
    assert np.all(obs == -1)


def test_hidden_board_contains_exactly_num_mines() -> None:
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=3)
    env.reset()
    mine_count = np.count_nonzero(env.hidden_board == -1)
    assert mine_count == 3


def test_calculate_numbers_center_mine_3x3() -> None:
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=10)
    env.hidden_board = np.zeros((3, 3), dtype=int)
    env.hidden_board[1, 1] = -1

    env._calculate_numbers()

    expected = np.array(
        [
            [1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ]
    )
    assert np.array_equal(env.hidden_board, expected)


def test_reset_creates_new_valid_game_state() -> None:
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=11)
    env.reset()
    env.done = True
    env.visible_board[0, 0] = 4

    obs = env.reset()

    assert env.done is False
    assert np.all(obs == -1)
    assert np.all(env.visible_board == -1)
    assert np.count_nonzero(env.hidden_board == -1) == env.num_mines


def test_invalid_action_returns_penalty_and_error() -> None:
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=4)
    env.reset()
    _, reward, done, info = env.step((-1, 0))
    assert reward < 0
    assert done is False
    assert info.get("error") == "invalid_action"


def test_already_revealed_returns_penalty_and_error() -> None:
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=5)
    env.reset()

    # Find one safe cell to reveal first.
    safe_positions = np.argwhere(env.hidden_board != -1)
    row, col = tuple(safe_positions[0])
    env.step((int(row), int(col)))

    _, reward, done, info = env.step((int(row), int(col)))
    assert reward < 0
    assert done is False
    assert info.get("error") == "already_revealed"


def test_clicking_mine_ends_game_with_loss() -> None:
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=6)
    env.reset()

    mine_positions = np.argwhere(env.hidden_board == -1)
    row, col = tuple(mine_positions[0])
    _, reward, done, info = env.step((int(row), int(col)))

    assert reward == -10
    assert done is True
    assert info.get("result") == "loss"


def test_revealing_all_safe_cells_wins_game() -> None:
    env = MinesweeperEnv(rows=3, cols=3, num_mines=1, seed=7)
    env.reset()

    safe_positions = np.argwhere(env.hidden_board != -1)
    done = False
    final_reward = 0.0
    final_info = {}
    for row, col in safe_positions:
        _, reward, done, info = env.step((int(row), int(col)))
        final_reward = reward
        final_info = info
        if done:
            break

    assert done is True
    assert final_reward == 21
    assert final_info.get("result") == "win"


def test_render_hidden_runs_without_crashing(capsys) -> None:
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=12)
    env.reset()

    env.render_hidden()
    captured = capsys.readouterr()
    assert captured.out.strip() != ""
