# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def object_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """Return an object's pose in the robot base frame."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    pos_object_world = wp.to_torch(object.data.root_pos_w)
    quat_object_world = wp.to_torch(object.data.root_quat_w)
    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)

    pos_object_base, quat_object_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_object_world, quat_object_world
    )

    if return_key == "pos":
        return pos_object_base
    if return_key == "quat":
        return quat_object_base
    return torch.cat((pos_object_base, quat_object_base), dim=1)


def a_and_b_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object A/B poses plus relative vectors in the robot base frame."""
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)

    object_a_pos_w = wp.to_torch(object_a.data.root_pos_w)
    object_a_quat_w = wp.to_torch(object_a.data.root_quat_w)
    object_b_pos_w = wp.to_torch(object_b.data.root_pos_w)
    object_b_quat_w = wp.to_torch(object_b.data.root_quat_w)

    pos_object_a_base, quat_object_a_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, object_a_pos_w, object_a_quat_w
    )
    pos_object_b_base, quat_object_b_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, object_b_pos_w, object_b_quat_w
    )

    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :]
    gripper_to_object_a = object_a_pos_w - ee_pos_w
    gripper_to_object_b = object_b_pos_w - ee_pos_w
    object_a_to_object_b = object_a_pos_w - object_b_pos_w

    return torch.cat(
        (
            pos_object_a_base,
            quat_object_a_base,
            pos_object_b_base,
            quat_object_b_base,
            gripper_to_object_a,
            gripper_to_object_b,
            object_a_to_object_b,
        ),
        dim=1,
    )


def a_and_b_abs_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object A/B poses and absolute end-effector pose in the robot base frame."""
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)

    object_a_pos_w = wp.to_torch(object_a.data.root_pos_w)
    object_a_quat_w = wp.to_torch(object_a.data.root_quat_w)
    object_b_pos_w = wp.to_torch(object_b.data.root_pos_w)
    object_b_quat_w = wp.to_torch(object_b.data.root_quat_w)

    pos_object_a_base, quat_object_a_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, object_a_pos_w, object_a_quat_w
    )
    pos_object_b_base, quat_object_b_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, object_b_pos_w, object_b_quat_w
    )

    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :]
    ee_quat_w = wp.to_torch(ee_frame.data.target_quat_w)[:, 0, :]
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return torch.cat(
        (
            pos_object_a_base,
            quat_object_a_base,
            pos_object_b_base,
            quat_object_b_base,
            ee_pos_base,
            ee_quat_base,
        ),
        dim=1,
    )


def _contact_force_ok(env: ManagerBasedRLEnv, sensor_name: str, force_threshold: float) -> torch.Tensor | None:
    if sensor_name not in env.scene.keys() or env.scene[sensor_name] is None:
        return None

    forces_w = env.scene[sensor_name].data.net_forces_w
    if forces_w is None:
        return None

    forces = wp.to_torch(forces_w)
    force_norm = torch.linalg.vector_norm(forces, dim=-1).reshape(forces.shape[0], -1)
    return torch.mean(force_norm, dim=1) > force_threshold


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Check if an object is grasped by the configured robot gripper."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = wp.to_torch(object.data.root_pos_w)
    end_effector_pos = wp.to_torch(ee_frame.data.target_pos_w)[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    grasped = pose_diff < diff_threshold

    contact_ok = _contact_force_ok(env, "contact_grasp", force_threshold)
    object_contact_ok = _contact_force_ok(env, f"contact_grasp_{object_cfg.name}", force_threshold)
    if contact_ok is not None:
        grasped = torch.logical_and(grasped, contact_ok)
    elif object_contact_ok is not None:
        grasped = torch.logical_and(grasped, object_contact_ok)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        if len(gripper_joint_ids) >= 2:
            joint_pos = wp.to_torch(robot.data.joint_pos)
            open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
            finger_1_closed = torch.abs(joint_pos[:, gripper_joint_ids[0]] - open_val) > env.cfg.gripper_threshold
            finger_2_closed = torch.abs(joint_pos[:, gripper_joint_ids[1]] - open_val) > env.cfg.gripper_threshold
            grasped = torch.logical_and(grasped, torch.logical_and(finger_1_closed, finger_2_closed))

    return grasped
