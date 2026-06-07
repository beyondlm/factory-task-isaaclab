# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .joint_pos_env_cfg import JointPositionPickPlaceFrankaEnvCfg, SORTING_BELT_POS, SORTING_BELT_SCALE, _franka_robot_cfg


FRANKA_TABLE_CAM_LOCAL_POS = (-0.2, 0.04208489388267458, 1.25000000223517418)
FRANKA_TABLE_CAM_LOCAL_ROT_XYZW = (0.39627450956965327, -0.3970864283360251, -0.5835610660438553, 0.587150205394181)


def _env_float(name: str, default: float) -> float:
    """Read a positive float tuning value from the environment."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _belt_scaled_pos(local_pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert an unscaled belt-USD local position to env-local coordinates."""
    return tuple(origin + value * scale for origin, value, scale in zip(SORTING_BELT_POS, local_pos, SORTING_BELT_SCALE))


@configclass
class IKRelPickPlaceFrankaEnvCfg(JointPositionPickPlaceFrankaEnvCfg):
    """Relative-pose differential-IK Franka Panda pick-and-place environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = _franka_robot_cfg()
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]
        ik_action_scale = _env_float("FRANKA_IK_ACTION_SCALE", 1.0)
        keyboard_pos_sensitivity = _env_float("FRANKA_KEYBOARD_POS_SENSITIVITY", 0.1)
        keyboard_rot_sensitivity = _env_float("FRANKA_KEYBOARD_ROT_SENSITIVITY", 0.1)
        spacemouse_pos_sensitivity = _env_float("FRANKA_SPACEMOUSE_POS_SENSITIVITY", 0.2)
        spacemouse_rot_sensitivity = _env_float("FRANKA_SPACEMOUSE_ROT_SENSITIVITY", 0.2)

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint[1-7]"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=self.use_relative_mode,
                ik_method="dls",
            ),
            scale=ik_action_scale,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=keyboard_pos_sensitivity,
                    rot_sensitivity=keyboard_rot_sensitivity,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=spacemouse_pos_sensitivity,
                    rot_sensitivity=spacemouse_rot_sensitivity,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class IKRelReplayCameraPlaceAOntoBEnvCfg(IKRelPickPlaceFrankaEnvCfg):
    """Replay version with camera support for video recording."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_fingertip_centered/wrist_camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.11398, -0.00619, -0.1167),
                # CameraCfg uses xyzw order; USD copied orientations are usually wxyz.
                rot=(0.70952, 0.67137, 0.12810, 0.17158),
                convention="opengl",
            ),
        )

        self.scene.table_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=8.0,
                focus_distance=400.0,
                horizontal_aperture=20.954999923706055,
                vertical_aperture=15.290800094604492,
                clipping_range=(0.01, 10000000.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=_belt_scaled_pos(FRANKA_TABLE_CAM_LOCAL_POS),
                rot=FRANKA_TABLE_CAM_LOCAL_ROT_XYZW,
                convention="opengl",
            ),
        )

        self.num_rerenders_on_reset = 3
        self.image_obs_list = ["wrist_camera", "table_camera"]


@configclass
class JointPositionReplayCameraPickPlaceFrankaEnvCfg(JointPositionPickPlaceFrankaEnvCfg):
    """Joint-position Franka environment with the same cameras used by GR00T replay data."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_fingertip_centered/wrist_camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.11398, -0.00619, -0.1167),
                rot=(0.70952, 0.67137, 0.12810, 0.17158),
                convention="opengl",
            ),
        )

        self.scene.table_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_camera",
            update_period=0.0333,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=8.0,
                focus_distance=400.0,
                horizontal_aperture=20.954999923706055,
                vertical_aperture=15.290800094604492,
                clipping_range=(0.01, 10000000.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=_belt_scaled_pos(FRANKA_TABLE_CAM_LOCAL_POS),
                rot=FRANKA_TABLE_CAM_LOCAL_ROT_XYZW,
                convention="opengl",
            ),
        )

        self.num_rerenders_on_reset = 3
        self.image_obs_list = ["wrist_camera", "table_camera"]


IKRelPlaceAOntoBEnvCfg = IKRelPickPlaceFrankaEnvCfg
