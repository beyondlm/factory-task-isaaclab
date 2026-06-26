# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.7 modality config for Franka pick-and-place joint-space demos."""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig


def _positive_int_from_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


ACTION_HORIZON = _positive_int_from_env("FRANKA_GROOT_ACTION_HORIZON", 32)


franka_joint_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist_camera", "table_camera"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "franka_joint_pos",
            "franka_gripper_width",
        ],
        sin_cos_embedding_keys=["franka_joint_pos"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "franka_joint_pos",
            "franka_gripper_width",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
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
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(franka_joint_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
