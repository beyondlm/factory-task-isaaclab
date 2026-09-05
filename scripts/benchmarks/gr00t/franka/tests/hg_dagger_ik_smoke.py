# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""L2 headless smoke test for Franka HG-DAgger differential IK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Pick-Place-Franka-Joint-Position-v0",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.num_envs = 1
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import import_packages  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from hg_dagger_ik import FRANKA_ARM_JOINT_LIMITS, SpaceMouseJointIK, to_torch  # noqa: E402


def main() -> None:
    import_packages(
        "isaaclab_tasks",
        ["utils", ".mdp", "pick_place", "articulated", "assembly", "libero", "robotwin", "stack"],
    )
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    try:
        env.reset()
        controller = SpaceMouseJointIK(env, max_joint_step=0.12)
        before = to_torch(env.scene["robot"].data.joint_pos)[:, controller.arm_joint_ids].clone()
        command = torch.zeros(7, dtype=torch.float32, device=env.device)
        command[-1] = 1.0
        target = controller.command(command)
        assert target.shape == (1, 8)
        assert torch.isfinite(target).all()
        assert torch.max(torch.abs(target[:, :7] - before)) <= 0.120001
        limits = torch.as_tensor(FRANKA_ARM_JOINT_LIMITS, device=env.device)
        assert torch.all(target[:, :7] >= limits[:, 0])
        assert torch.all(target[:, :7] <= limits[:, 1])
        env.step(target)
        print("HG-DAgger IK L2 smoke test passed")
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
