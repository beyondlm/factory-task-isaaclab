# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T data config for Franka pick-and-place relative task-space datasets.

Copy or merge this class into Isaac-GR00T's ``gr00t/experiment/data_config.py``
and add the map entry shown at the bottom of this file.
"""

import os

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoColorJitter, VideoCrop, VideoResize, VideoToNumpy, VideoToTensor
from gr00t.model.transforms import GR00TTransform


def _positive_int_from_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


ACTION_HORIZON = _positive_int_from_env("FRANKA_GROOT_ACTION_HORIZON", 32)


class FrankaPickPlaceRelativeTaskSpaceDataConfig:
    video_keys = [
        "video.wrist_camera",
        "video.table_camera",
    ]
    state_keys = [
        "state.franka_eef_pos",
        "state.franka_eef_quat",
        "state.franka_gripper_width",
    ]
    action_keys = [
        "action.franka_eef_delta_pos",
        "action.franka_eef_delta_rot",
        "action.franka_gripper_cmd",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(ACTION_HORIZON))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.franka_eef_pos": "min_max",
                    "state.franka_gripper_width": "min_max",
                },
                target_rotations={
                    "state.franka_eef_quat": "rotation_6d",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.franka_eef_delta_pos": "min_max",
                    "action.franka_eef_delta_rot": "min_max",
                    "action.franka_gripper_cmd": "min_max",
                },
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP_ENTRY = {
    "franka_pick_place_relative_task_space": FrankaPickPlaceRelativeTaskSpaceDataConfig(),
}
