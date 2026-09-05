# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder adapter for Franka HG-DAgger metadata.

The environment's standard ``ActionStateRecorderManagerCfg`` remains
responsible for replay-compatible initial state, actions, observations, and
post-step simulation state. This adapter appends DAgger-specific tensors to the
same in-memory episode before it is exported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from hg_dagger_core import as_action_vector

OUTCOME_CODES = {
    "success": 1,
    "timeout": 2,
    "human_abort": 3,
    "policy_failure": 4,
}


@dataclass
class EpisodeSummary:
    """Counters accumulated for one collected episode."""

    frame_count: int = 0
    intervention_steps: int = 0
    intervention_segments: int = 0
    valid_policy_actions: int = 0

    @property
    def intervention_ratio(self) -> float:
        return self.intervention_steps / max(self.frame_count, 1)


class HGDAggerRecorder:
    """Append aligned DAgger fields through an IsaacLab recorder manager."""

    def __init__(self, recorder_manager, device: str, *, state_dim: int = 8, action_dim: int = 8):
        if state_dim < 1 or action_dim < 1:
            raise ValueError("state_dim and action_dim must be positive")
        self.recorder_manager = recorder_manager
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.summary = EpisodeSummary()
        self._was_intervening = False

    def reset(self) -> None:
        self.summary = EpisodeSummary()
        self._was_intervening = False

    def _tensor(self, value, *, dtype=torch.float32) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype, device=self.device)
        return tensor.reshape(1, *tensor.shape)

    def record_pre_step(
        self,
        *,
        observation: np.ndarray,
        policy_action: np.ndarray,
        expert_action: np.ndarray,
        executed_action: np.ndarray,
        intervention: bool,
        policy_action_valid: bool,
        inference_id: int,
        chunk_index: int,
        frame_index: int,
    ) -> None:
        """Record values defined at the state immediately before ``env.step``."""

        observation = as_action_vector(observation, size=self.state_dim, name="observation")
        policy_action = as_action_vector(policy_action, size=self.action_dim, name="policy_action")
        expert_action = as_action_vector(expert_action, size=self.action_dim, name="expert_action")
        executed_action = as_action_vector(executed_action, size=self.action_dim, name="executed_action")

        fields = {
            "dagger/observation_joint_state": self._tensor(observation),
            "dagger/policy_action": self._tensor(policy_action),
            "dagger/expert_action": self._tensor(expert_action),
            "dagger/executed_action": self._tensor(executed_action),
            "dagger/intervention_mask": self._tensor(intervention, dtype=torch.bool),
            "dagger/policy_action_valid": self._tensor(policy_action_valid, dtype=torch.bool),
            "dagger/inference_id": self._tensor(inference_id, dtype=torch.int64),
            "dagger/chunk_index": self._tensor(chunk_index, dtype=torch.int64),
            "dagger/frame_index": self._tensor(frame_index, dtype=torch.int64),
        }
        for key, value in fields.items():
            self.recorder_manager.add_to_episodes(key, value)

        self.summary.frame_count += 1
        self.summary.intervention_steps += int(intervention)
        self.summary.valid_policy_actions += int(policy_action_valid)
        if intervention and not self._was_intervening:
            self.summary.intervention_segments += 1
        self._was_intervening = intervention

    def record_post_step(self, achieved_joint_state: np.ndarray) -> None:
        """Record the achieved state after the action was simulated."""

        achieved = as_action_vector(achieved_joint_state, size=self.state_dim, name="achieved_joint_state")
        self.recorder_manager.add_to_episodes("dagger/achieved_joint_state", self._tensor(achieved))

    def record_shadow_query(
        self,
        *,
        observation: np.ndarray,
        policy_action: np.ndarray,
        policy_action_valid: bool,
        query_frame: int,
    ) -> None:
        """Record a counterfactual policy query keyed to its source frame."""

        observation = as_action_vector(observation, size=self.state_dim, name="shadow_observation")
        policy_action = as_action_vector(policy_action, size=self.action_dim, name="shadow_policy_action")
        self.recorder_manager.add_to_episodes("dagger/shadow/observation_joint_state", self._tensor(observation))
        self.recorder_manager.add_to_episodes("dagger/shadow/policy_action", self._tensor(policy_action))
        self.recorder_manager.add_to_episodes(
            "dagger/shadow/policy_action_valid",
            self._tensor(policy_action_valid, dtype=torch.bool),
        )
        self.recorder_manager.add_to_episodes("dagger/shadow/query_frame", self._tensor(query_frame, dtype=torch.int64))
        self.summary.valid_policy_actions += int(policy_action_valid)

    def finish_episode(self, *, success: bool, outcome: str, seed: int, policy_checkpoint_id: str) -> None:
        """Store one-element episode summary datasets before export."""

        if outcome not in OUTCOME_CODES:
            raise ValueError(f"Unknown outcome {outcome!r}; expected one of {sorted(OUTCOME_CODES)}")
        checkpoint_bytes = np.frombuffer(policy_checkpoint_id.encode("utf-8"), dtype=np.uint8)
        if checkpoint_bytes.size == 0:
            raise ValueError("policy_checkpoint_id must not be empty")
        summary_fields = {
            "dagger/episode/success": self._tensor(success, dtype=torch.bool),
            "dagger/episode/outcome_code": self._tensor(OUTCOME_CODES[outcome], dtype=torch.int64),
            "dagger/episode/seed": self._tensor(seed, dtype=torch.int64),
            "dagger/episode/frame_count": self._tensor(self.summary.frame_count, dtype=torch.int64),
            "dagger/episode/intervention_steps": self._tensor(self.summary.intervention_steps, dtype=torch.int64),
            "dagger/episode/intervention_segments": self._tensor(self.summary.intervention_segments, dtype=torch.int64),
            "dagger/episode/intervention_ratio": self._tensor(self.summary.intervention_ratio),
            "dagger/episode/valid_policy_actions": self._tensor(self.summary.valid_policy_actions, dtype=torch.int64),
            "dagger/episode/policy_checkpoint_id_utf8": self._tensor(checkpoint_bytes, dtype=torch.uint8),
        }
        for key, value in summary_fields.items():
            self.recorder_manager.add_to_episodes(key, value)
