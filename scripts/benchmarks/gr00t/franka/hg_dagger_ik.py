# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differential-IK adapter for SpaceMouse control in a joint-action environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.controllers import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import math as math_utils

FRANKA_ARM_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float32,
)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def to_torch(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    return torch.as_tensor(value)


def joint_id_list(joint_ids: Any) -> list[int]:
    return np.asarray(to_numpy(joint_ids), dtype=np.int64).reshape(-1).tolist()


class SpaceMouseJointIK:
    """Convert relative SpaceMouse commands to safe Franka joint targets."""

    def __init__(self, env, max_joint_step: float):
        if max_joint_step <= 0.0:
            raise ValueError(f"max_joint_step must be > 0, got {max_joint_step}")
        self.env = env
        self.robot = env.scene["robot"]
        arm_joint_ids, _ = self.robot.find_joints([f"panda_joint{i}" for i in range(1, 8)])
        body_ids, body_names = self.robot.find_bodies(["panda_hand"])
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one panda_hand body, found {body_names}")

        self.arm_joint_ids = joint_id_list(arm_joint_ids)
        self.body_idx = int(np.asarray(to_numpy(body_ids)).reshape(-1)[0])
        self.jacobian_body_idx = self.body_idx - 1 if self.robot.is_fixed_base else self.body_idx
        self.max_joint_step = float(max_joint_step)
        # Match the task's ``ee_frame`` offset from panda_hand to the control point.
        self.offset_pos = torch.tensor([[0.0, 0.0, 0.1034]], device=env.device)
        self.offset_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device)
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            num_envs=1,
            device=env.device,
        )

    def _frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = to_torch(self.robot.data.body_pos_w)[:, self.body_idx]
        ee_quat_w = to_torch(self.robot.data.body_quat_w)[:, self.body_idx]
        root_pos_w = to_torch(self.robot.data.root_pos_w)
        root_quat_w = to_torch(self.robot.data.root_quat_w)
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return math_utils.combine_frame_transforms(ee_pos_b, ee_quat_b, self.offset_pos, self.offset_quat)

    def _frame_jacobian(self, ee_quat_b: torch.Tensor) -> torch.Tensor:
        jacobian = to_torch(self.robot.root_view.get_jacobians())[
            :, self.jacobian_body_idx, :, self.arm_joint_ids
        ].clone()
        base_quat = to_torch(self.robot.data.root_quat_w)
        base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(base_quat))
        jacobian[:, :3, :] = torch.bmm(base_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot, jacobian[:, 3:, :])
        # The configured offset is expressed in panda_hand coordinates. Rotate
        # it into the base frame before shifting the geometric Jacobian.
        offset_b = math_utils.quat_apply(ee_quat_b, self.offset_pos)
        jacobian[:, :3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_b), jacobian[:, 3:, :])
        return jacobian

    def command(self, teleop_action: torch.Tensor) -> torch.Tensor:
        command = teleop_action.to(device=self.env.device, dtype=torch.float32).reshape(1, -1)
        if command.shape[1] != 7:
            raise ValueError(f"Expected 7D SpaceMouse command, got {tuple(command.shape)}")

        ee_pos, ee_quat = self._frame_pose()
        joint_pos = to_torch(self.robot.data.joint_pos)[:, self.arm_joint_ids]
        self.controller.set_command(command[:, :6], ee_pos, ee_quat)
        joint_target = self.controller.compute(ee_pos, ee_quat, self._frame_jacobian(ee_quat), joint_pos)
        joint_target = torch.clamp(
            joint_target,
            min=joint_pos - self.max_joint_step,
            max=joint_pos + self.max_joint_step,
        )
        limits = to_torch(self.robot.data.soft_joint_pos_limits)[:, self.arm_joint_ids]
        joint_target = torch.clamp(joint_target, min=limits[..., 0], max=limits[..., 1])
        return torch.cat([joint_target, command[:, 6:7]], dim=1)
