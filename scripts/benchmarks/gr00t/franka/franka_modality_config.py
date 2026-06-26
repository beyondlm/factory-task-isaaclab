# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.7 modality config for Franka pick-and-place task-space demos."""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig


def _positive_int_from_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


VIDEO_DELTA_INDICES = [0]
STATE_DELTA_INDICES = [0]
ACTION_HORIZON = _positive_int_from_env("FRANKA_GROOT_ACTION_HORIZON", 32)
ACTION_DELTA_INDICES = list(range(ACTION_HORIZON))


franka_config = {
    "video": ModalityConfig(
        delta_indices=VIDEO_DELTA_INDICES,
        modality_keys=["wrist_camera", "table_camera"],
    ),
    "state": ModalityConfig(
        delta_indices=STATE_DELTA_INDICES,
        modality_keys=[
            "franka_eef_pos",
            "franka_eef_quat",
            "franka_gripper_width",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=ACTION_DELTA_INDICES,
        modality_keys=[
            "franka_eef_delta_pos",
            "franka_eef_delta_rot",
            "franka_gripper_cmd",
        ],
        action_configs=[
            # The HDF5 actions are already relative teleop deltas, so do not ask GR00T
            # to compute another relative transform from absolute action targets.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=STATE_DELTA_INDICES,
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(franka_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
