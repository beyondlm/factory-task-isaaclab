# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

GROOT_BENCHMARK_DIR = Path(__file__).parents[2]
sys.path.insert(0, str(GROOT_BENCHMARK_DIR))

from dagger.data_contract import validate_transition_arrays  # noqa: E402
from dagger.evaluation import stable_inference_seed, summarize_paired_outcomes  # noqa: E402
from dagger.task_spec import GripperCommandSpec, VLADAggerTaskSpec  # noqa: E402


def _arrays(action_dim: int = 6, state_dim: int = 4, steps: int = 8) -> dict[str, np.ndarray]:
    policy = np.arange(steps * action_dim, dtype=np.float32).reshape(steps, action_dim)
    expert = policy + 100
    state = np.arange(steps * state_dim, dtype=np.float32).reshape(steps, state_dim)
    mask = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    executed = np.where(mask[:, None], expert, policy)
    return {
        "observation_state": state - 1,
        "achieved_state": state + 1,
        "policy_action": policy,
        "expert_action": expert,
        "executed_action": executed,
        "intervention_mask": mask,
        "policy_action_valid": ~mask,
        "frame_index": np.arange(steps),
    }


def test_generic_contract_accepts_non_franka_action_dimension() -> None:
    result = validate_transition_arrays(
        _arrays(),
        state_dim=4,
        action_dim=6,
        action_horizon=4,
        minimum_segment_length=4,
    )
    assert result.steps == 8
    assert result.intervention_segments == 1
    assert result.valid_anchors == 3


def test_generic_contract_rejects_misaligned_execution() -> None:
    arrays = _arrays()
    arrays["executed_action"][3, 0] = -1
    with pytest.raises(ValueError, match="expert_action"):
        validate_transition_arrays(
            arrays, state_dim=4, action_dim=6, action_horizon=4, minimum_segment_length=4
        )


def test_task_spec_round_trip() -> None:
    spec = VLADAggerTaskSpec(
        name="customer_task",
        isaaclab_task="Customer-Task-v0",
        policy_type="joint_space",
        state_dim=4,
        action_dim=6,
        action_horizon=16,
        minimum_intervention_steps=32,
        observation_keys=("robot_state",),
        action_keys=("robot_action",),
        camera_names=("front",),
        language_instruction="Place the part.",
        embodiment_tag="CUSTOMER_ROBOT",
        gripper=GripperCommandSpec(action_index=5, open_target=1.0),
    )
    assert VLADAggerTaskSpec.from_dict(spec.to_dict()) == spec


def test_stable_seed_matches_franka_protocol() -> None:
    expected = int.from_bytes(
        hashlib.sha256(b"franka_gr00t_policy_noise_v1:11:2:7").digest()[:8], "big"
    ) & ((1 << 63) - 1)
    assert stable_inference_seed("franka_gr00t_policy_noise_v1", 11, 2, 7) == expected


def test_paired_summary_does_not_hide_regressions() -> None:
    result = summarize_paired_outcomes(
        [True, False, True, False],
        [True, True, False, True],
    )
    assert (result.improved, result.regressed, result.unchanged) == (2, 1, 1)
    assert result.delta == 0.25
