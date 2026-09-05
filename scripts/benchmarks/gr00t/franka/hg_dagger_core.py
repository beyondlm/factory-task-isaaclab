# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure helpers shared by Franka HG-DAgger collection and conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterventionTransition:
    """State transition produced by an intervention gate update."""

    active: bool
    activated: bool = False
    released: bool = False


class InterventionGate:
    """Toggle gate with a minimum recovery length.

    A release requested before ``minimum_steps`` is deferred until the minimum
    is reached. This prevents accidental one-frame corrections while still
    allowing the operator to request release early.
    """

    def __init__(self, minimum_steps: int = 64):
        if minimum_steps < 1:
            raise ValueError(f"minimum_steps must be >= 1, got {minimum_steps}")
        self.minimum_steps = minimum_steps
        self.reset()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def steps(self) -> int:
        return self._steps

    def reset(self) -> None:
        self._active = False
        self._steps = 0
        self._release_requested = False

    def toggle(self) -> InterventionTransition:
        if not self._active:
            self._active = True
            self._steps = 0
            self._release_requested = False
            return InterventionTransition(active=True, activated=True)

        if self._steps >= self.minimum_steps:
            self._active = False
            self._release_requested = False
            return InterventionTransition(active=False, released=True)

        self._release_requested = True
        return InterventionTransition(active=True)

    def complete_step(self) -> InterventionTransition:
        if not self._active:
            return InterventionTransition(active=False)

        self._steps += 1
        if self._release_requested and self._steps >= self.minimum_steps:
            self._active = False
            self._release_requested = False
            return InterventionTransition(active=False, released=True)
        return InterventionTransition(active=True)


def contiguous_true_segments(mask: np.ndarray, minimum_length: int = 1) -> list[tuple[int, int]]:
    """Return half-open ``[start, end)`` ranges of contiguous true values."""

    if minimum_length < 1:
        raise ValueError(f"minimum_length must be >= 1, got {minimum_length}")
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []

    padded = np.concatenate(([False], values, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    segments = [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]
    return [(start, end) for start, end in segments if end - start >= minimum_length]


def valid_horizon_anchors(mask: np.ndarray, horizon: int) -> np.ndarray:
    """Mark frames whose complete future horizon is human-controlled."""

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    values = np.asarray(mask, dtype=bool).reshape(-1)
    anchors = np.zeros_like(values, dtype=bool)
    if values.size < horizon:
        return anchors

    window_counts = np.convolve(values.astype(np.int32), np.ones(horizon, dtype=np.int32), mode="valid")
    anchors[: window_counts.size] = window_counts == horizon
    return anchors


def as_action_vector(value: np.ndarray, *, size: int = 8, name: str = "action") -> np.ndarray:
    """Normalize one action to a finite float32 vector."""

    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must contain {size} values, got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector
