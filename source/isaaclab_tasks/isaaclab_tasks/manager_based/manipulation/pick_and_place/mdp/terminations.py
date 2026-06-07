# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _as_torch(value) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)


def _gripper_is_open(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor | None:
    cfg = getattr(env, "cfg", getattr(getattr(env, "unwrapped", None), "cfg", None))
    if cfg is None or not hasattr(cfg, "gripper_joint_names"):
        return None

    robot = env.scene[robot_cfg.name]
    gripper_joint_ids, _ = robot.find_joints(cfg.gripper_joint_names)
    if len(gripper_joint_ids) < 2 or not hasattr(cfg, "gripper_open_val") or not hasattr(cfg, "gripper_threshold"):
        return None

    joint_pos = _as_torch(robot.data.joint_pos)
    open_val = torch.tensor(cfg.gripper_open_val, dtype=torch.float32, device=env.device)
    finger_1_open = torch.abs(torch.abs(joint_pos[:, gripper_joint_ids[0]]) - open_val) < cfg.gripper_threshold
    finger_2_open = torch.abs(torch.abs(joint_pos[:, gripper_joint_ids[1]]) - open_val) < cfg.gripper_threshold
    return torch.logical_and(finger_1_open, finger_2_open)


def object_a_is_onto_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.08,
    force_threshold: float = 0.3,
    require_gripper_open: bool = True,
) -> torch.Tensor:
    """Check whether object A is placed on object B."""
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]

    object_a_pos = _as_torch(object_a.data.root_pos_w)
    object_b_pos = _as_torch(object_b.data.root_pos_w)

    xy_dist = torch.linalg.vector_norm(object_a_pos[:, :2] - object_b_pos[:, :2], dim=1)
    height_diff = torch.abs(object_a_pos[:, 2] - object_b_pos[:, 2])
    success = torch.logical_and(xy_dist < xy_threshold, height_diff < height_threshold)

    if "contact_object" in env.scene.keys() and env.scene["contact_object"] is not None:
        forces_w = env.scene["contact_object"].data.net_forces_w
        if forces_w is not None:
            forces = wp.to_torch(forces_w)
            force_norm = torch.linalg.vector_norm(forces, dim=-1).reshape(forces.shape[0], -1)
            success = torch.logical_and(success, torch.mean(force_norm, dim=1) > force_threshold)

    if require_gripper_open:
        gripper_open = _gripper_is_open(env, robot_cfg)
        if gripper_open is not None:
            success = torch.logical_and(success, gripper_open)

    return success


def object_a_is_into_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.3,
    height_diff: float = 0.0,
    min_height_diff: float | None = None,
    max_height_diff: float | None = None,
    force_threshold: float = 0.3,
    require_gripper_open: bool = True,
    max_linear_velocity: float | None = None,
    max_angular_velocity: float | None = None,
) -> torch.Tensor:
    """Check whether object A is placed into object B."""
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]

    object_a_pos = _as_torch(object_a.data.root_pos_w)
    object_b_pos = _as_torch(object_b.data.root_pos_w)
    pos_diff = object_a_pos - object_b_pos

    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_diff = pos_diff[:, 2]
    z_dist = torch.abs(z_diff)
    success = torch.logical_and(xy_dist < xy_threshold, torch.abs(z_dist - height_diff) < height_threshold)

    if min_height_diff is not None:
        success = torch.logical_and(success, z_diff > min_height_diff)

    if max_height_diff is not None:
        success = torch.logical_and(success, z_diff < max_height_diff)

    if "contact_object" in env.scene.keys() and env.scene["contact_object"] is not None:
        forces_w = env.scene["contact_object"].data.net_forces_w
        if forces_w is not None:
            forces = wp.to_torch(forces_w)
            force_norm = torch.linalg.vector_norm(forces, dim=-1).reshape(forces.shape[0], -1)
            success = torch.logical_and(success, torch.mean(force_norm, dim=1) > force_threshold)

    if require_gripper_open:
        gripper_open = _gripper_is_open(env, robot_cfg)
        if gripper_open is not None:
            success = torch.logical_and(success, gripper_open)

    if max_linear_velocity is not None:
        lin_vel = torch.linalg.vector_norm(_as_torch(object_a.data.root_lin_vel_w), dim=1)
        success = torch.logical_and(success, lin_vel < max_linear_velocity)

    if max_angular_velocity is not None:
        ang_vel = torch.linalg.vector_norm(_as_torch(object_a.data.root_ang_vel_w), dim=1)
        success = torch.logical_and(success, ang_vel < max_angular_velocity)

    return success


def both_boxes_placed_a_into_c_b_into_d(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    object_c_cfg: SceneEntityCfg = SceneEntityCfg("object_c"),
    object_d_cfg: SceneEntityCfg = SceneEntityCfg("object_d"),
    xy_threshold: float = 0.08,
    height_threshold: float = 0.3,
    height_diff: float = 0.0,
    min_height_diff: float | None = None,
    max_height_diff: float | None = None,
    force_threshold: float = 0.3,
    require_gripper_open: bool = True,
    max_linear_velocity: float | None = None,
    max_angular_velocity: float | None = None,
) -> torch.Tensor:
    """Success when object A is in C and object B is in D."""
    success_ac = object_a_is_into_b(
        env,
        robot_cfg=robot_cfg,
        object_a_cfg=object_a_cfg,
        object_b_cfg=object_c_cfg,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
        height_diff=height_diff,
        min_height_diff=min_height_diff,
        max_height_diff=max_height_diff,
        force_threshold=force_threshold,
        require_gripper_open=require_gripper_open,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=max_angular_velocity,
    )
    success_bd = object_a_is_into_b(
        env,
        robot_cfg=robot_cfg,
        object_a_cfg=object_b_cfg,
        object_b_cfg=object_d_cfg,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
        height_diff=height_diff,
        min_height_diff=min_height_diff,
        max_height_diff=max_height_diff,
        force_threshold=force_threshold,
        require_gripper_open=require_gripper_open,
        max_linear_velocity=max_linear_velocity,
        max_angular_velocity=max_angular_velocity,
    )
    return torch.logical_and(success_ac, success_bd)
