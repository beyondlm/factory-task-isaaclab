# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference VLA DAgger task contract for Franka box sorting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from dagger.task_spec import GripperCommandSpec, VLADAggerTaskSpec  # noqa: E402


FRANKA_SORTING_DAGGER_TASK = VLADAggerTaskSpec(
    name="franka_sorting_joint_space_h32",
    isaaclab_task="Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0",
    policy_type="joint_space",
    state_dim=8,
    action_dim=8,
    action_horizon=32,
    minimum_intervention_steps=64,
    observation_keys=("franka_joint_pos", "franka_gripper_width"),
    action_keys=("franka_joint_pos", "franka_gripper_width"),
    camera_names=("wrist_camera", "table_camera"),
    language_instruction=(
        "Pick up the labeled box and place it into the blue bin. "
        "Pick up the unlabeled box and place it into the black bin."
    ),
    embodiment_tag="NEW_EMBODIMENT",
    gripper=GripperCommandSpec(action_index=7),
    success_metric_version="bin_local_box_footprint_v1",
)


if __name__ == "__main__":
    print(json.dumps(FRANKA_SORTING_DAGGER_TASK.to_dict(), indent=2, sort_keys=True))
