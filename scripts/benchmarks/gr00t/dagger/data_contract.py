# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure NumPy validation for aligned HG-DAgger transition arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


STATE_FIELDS = ("observation_state", "achieved_state")
ACTION_FIELDS = ("policy_action", "expert_action", "executed_action")
VECTOR_FIELDS = (*STATE_FIELDS, *ACTION_FIELDS)


@dataclass(frozen=True)
class EpisodeAudit:
    """Structural counts from one validated DAgger rollout."""

    steps: int
    intervention_steps: int
    intervention_segments: int
    valid_anchors: int


def _segments(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(end))
        for start, end in edges.reshape(-1, 2)
        if end - start >= minimum_length
    ]


def complete_human_horizon_mask(intervention_mask: np.ndarray, horizon: int) -> np.ndarray:
    """Return anchors whose complete future action chunk is human-controlled."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    mask = np.asarray(intervention_mask, dtype=bool).reshape(-1)
    result = np.zeros(mask.shape, dtype=bool)
    if len(mask) >= horizon:
        counts = np.convolve(mask.astype(np.int32), np.ones(horizon, dtype=np.int32), mode="valid")
        result[: len(counts)] = counts == horizon
    return result


def validate_transition_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    action_dim: int,
    state_dim: int | None = None,
    action_horizon: int,
    minimum_segment_length: int,
) -> EpisodeAudit:
    """Validate policy/human/action alignment without depending on HDF5 or IsaacLab."""

    if action_dim < 1:
        raise ValueError("action_dim must be >= 1")
    state_dim = action_dim if state_dim is None else state_dim
    if state_dim < 1:
        raise ValueError("state_dim must be >= 1")
    if minimum_segment_length < action_horizon:
        raise ValueError("minimum_segment_length must be at least action_horizon")
    required = (*VECTOR_FIELDS, "intervention_mask", "policy_action_valid", "frame_index")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing transition fields: {', '.join(missing)}")

    mask = np.asarray(arrays["intervention_mask"], dtype=bool).reshape(-1)
    steps = len(mask)
    if steps < 1:
        raise ValueError("rollout must contain at least one transition")
    vectors: dict[str, np.ndarray] = {}
    field_dimensions = {**{name: state_dim for name in STATE_FIELDS}, **{name: action_dim for name in ACTION_FIELDS}}
    for name, dimension in field_dimensions.items():
        value = np.asarray(arrays[name])
        if value.shape != (steps, dimension):
            raise ValueError(f"{name} must have shape {(steps, dimension)}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
        vectors[name] = value

    policy_valid = np.asarray(arrays["policy_action_valid"], dtype=bool).reshape(-1)
    if policy_valid.shape != mask.shape:
        raise ValueError("policy_action_valid must align with intervention_mask")
    if np.any(policy_valid & mask):
        raise ValueError("policy actions cannot be marked valid during intervention")
    if not np.array_equal(np.asarray(arrays["frame_index"]).reshape(-1), np.arange(steps)):
        raise ValueError("frame_index must be contiguous from zero")
    if not np.array_equal(vectors["executed_action"][mask], vectors["expert_action"][mask]):
        raise ValueError("executed_action must equal expert_action during intervention")
    if not np.array_equal(vectors["executed_action"][~mask], vectors["policy_action"][~mask]):
        raise ValueError("executed_action must equal policy_action outside intervention")

    segments = _segments(mask, minimum_segment_length)
    anchors = complete_human_horizon_mask(mask, action_horizon)
    return EpisodeAudit(
        steps=steps,
        intervention_steps=int(mask.sum()),
        intervention_segments=len(segments),
        valid_anchors=int(anchors.sum()),
    )
