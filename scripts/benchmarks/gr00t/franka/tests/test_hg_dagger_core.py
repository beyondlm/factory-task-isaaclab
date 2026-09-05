# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from hg_dagger_core import (  # noqa: E402
    InterventionGate,
    as_action_vector,
    contiguous_true_segments,
    valid_horizon_anchors,
)
from merge_lerobot_dagger_datasets import (  # noqa: E402
    choose_dagger_episodes,
    validate_feature_compatibility,
)


def test_intervention_gate_defers_early_release() -> None:
    gate = InterventionGate(minimum_steps=3)
    assert gate.toggle().activated
    assert gate.active
    assert not gate.toggle().released

    assert gate.complete_step().active
    assert gate.complete_step().active
    transition = gate.complete_step()
    assert transition.released
    assert not gate.active


def test_intervention_gate_releases_immediately_after_minimum() -> None:
    gate = InterventionGate(minimum_steps=2)
    gate.toggle()
    gate.complete_step()
    gate.complete_step()
    transition = gate.toggle()
    assert transition.released
    assert not gate.active


def test_contiguous_true_segments_filters_short_runs() -> None:
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert contiguous_true_segments(mask, minimum_length=2) == [(1, 3), (4, 7)]
    assert contiguous_true_segments(mask, minimum_length=3) == [(4, 7)]


def test_valid_horizon_anchors_require_complete_future_window() -> None:
    mask = np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    np.testing.assert_array_equal(
        valid_horizon_anchors(mask, horizon=3),
        np.array([1, 1, 0, 0, 0, 1, 0, 0], dtype=bool),
    )


def test_action_vector_validation() -> None:
    result = as_action_vector(np.arange(8))
    assert result.dtype == np.float32
    with pytest.raises(ValueError, match="8 values"):
        as_action_vector(np.arange(7))
    with pytest.raises(ValueError, match="non-finite"):
        as_action_vector(np.array([0.0] * 7 + [np.nan]))


def test_dagger_mix_selection_targets_frame_ratio() -> None:
    episodes = [
        {"episode_index": 0, "length": 100},
        {"episode_index": 1, "length": 100},
        {"episode_index": 2, "length": 100},
    ]
    selected = choose_dagger_episodes(
        episodes,
        base_frames=800,
        target_fraction=0.2,
        allow_repeat=False,
    )
    assert [row["episode_index"] for row in selected] == [0, 1]


def test_dagger_mix_can_include_every_unique_episode() -> None:
    episodes = [
        {"episode_index": 0, "length": 100},
        {"episode_index": 1, "length": 100},
        {"episode_index": 2, "length": 100},
    ]
    selected = choose_dagger_episodes(
        episodes,
        base_frames=800,
        target_fraction=0.1,
        allow_repeat=False,
        all_unique=True,
    )
    assert [row["episode_index"] for row in selected] == [0, 1, 2]


def test_dagger_mix_can_repeat_when_explicitly_enabled() -> None:
    episodes = [{"episode_index": 0, "length": 50}]
    selected = choose_dagger_episodes(
        episodes,
        base_frames=800,
        target_fraction=0.2,
        allow_repeat=True,
    )
    assert sum(row["length"] for row in selected) == 200


def test_merge_rejects_missing_dagger_videos() -> None:
    scalar = {"dtype": "float32", "shape": [8]}
    video = {"dtype": "video", "shape": [480, 640, 3]}
    base_info = {"features": {"observation.state": scalar, "observation.images.wrist": video}}
    dagger_info = {"features": {"observation.state": scalar}}
    with pytest.raises(ValueError, match="video features differ"):
        validate_feature_compatibility(base_info, dagger_info)


def test_merge_rejects_incompatible_base_feature() -> None:
    base_info = {"features": {"observation.state": {"dtype": "float32", "shape": [8]}}}
    dagger_info = {"features": {"observation.state": {"dtype": "float32", "shape": [7]}}}
    with pytest.raises(ValueError, match="incompatible shape"):
        validate_feature_compatibility(base_info, dagger_info)
