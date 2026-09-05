# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative contract for adapting HG-DAgger to a VLA task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GripperCommandSpec:
    """Map a recorded binary gripper command to the policy target space."""

    action_index: int
    close_command: float = -1.0
    open_command: float = 1.0
    close_target: float = 0.0
    open_target: float = 0.08

    def validate(self, action_dim: int) -> None:
        if not 0 <= self.action_index < action_dim:
            raise ValueError(f"gripper action_index must be in [0, {action_dim}), got {self.action_index}")
        if self.close_command == self.open_command:
            raise ValueError("close_command and open_command must differ")
        if self.close_target >= self.open_target:
            raise ValueError("close_target must be less than open_target")


@dataclass(frozen=True)
class VLADAggerTaskSpec:
    """Minimum reproducible information required by the DAgger workflow.

    Environment-specific observation extraction, teleoperation, action execution,
    and success evaluation remain adapter callbacks. This dataclass freezes the
    scalar and schema choices that must stay identical across collection,
    conversion, training, and evaluation.
    """

    name: str
    isaaclab_task: str
    policy_type: str
    state_dim: int
    action_dim: int
    action_horizon: int
    minimum_intervention_steps: int
    observation_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    camera_names: tuple[str, ...]
    language_instruction: str
    embodiment_tag: str
    state_history_frames: int = 1
    video_history_frames: int = 1
    gripper: GripperCommandSpec | None = None
    success_metric_version: str = "task_success_v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for field_name in ("name", "isaaclab_task", "policy_type", "language_instruction", "embodiment_tag"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.state_dim < 1:
            raise ValueError("state_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        if self.minimum_intervention_steps < self.action_horizon:
            raise ValueError("minimum_intervention_steps must cover at least one complete action horizon")
        if self.state_history_frames < 1 or self.video_history_frames < 1:
            raise ValueError("history frame counts must be >= 1")
        for field_name in ("observation_keys", "action_keys", "camera_names"):
            values = getattr(self, field_name)
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain non-empty names")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} contains duplicates")
        if self.gripper is not None:
            self.gripper.validate(self.action_dim)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VLADAggerTaskSpec:
        payload = dict(value)
        for key in ("observation_keys", "action_keys", "camera_names"):
            if key in payload:
                payload[key] = tuple(payload[key])
        if payload.get("gripper") is not None and not isinstance(payload["gripper"], GripperCommandSpec):
            payload["gripper"] = GripperCommandSpec(**payload["gripper"])
        spec = cls(**payload)
        spec.validate()
        return spec
