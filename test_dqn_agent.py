"""Tests for DQN state encoding helpers."""

from __future__ import annotations

import numpy as np

from agents.dqn_agent import DQNAgent


def test_frontier_channel_marks_hidden_cells_next_to_revealed_numbers() -> None:
    agent = DQNAgent(rows=3, cols=3, model_type="cnn", use_frontier_channel=True, seed=1)
    observation = np.full((3, 3), -1, dtype=int)
    observation[1, 1] = 1

    encoded = agent._encode_observation(observation)
    frontier = encoded[2]

    expected = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert encoded.shape == (3, 3, 3)
    assert np.array_equal(frontier, expected)
