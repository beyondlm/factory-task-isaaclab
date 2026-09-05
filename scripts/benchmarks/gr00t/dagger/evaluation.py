# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic pairing and outcome summaries for closed-loop VLA evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


def stable_inference_seed(namespace: str, episode_seed: int, repeat_index: int, inference_index: int) -> int:
    """Derive a stable cross-process seed without Python's randomized ``hash``."""

    if not namespace or ":" in namespace:
        raise ValueError("namespace must be non-empty and must not contain ':'")
    if min(episode_seed, repeat_index, inference_index) < 0:
        raise ValueError("seed components must be non-negative")
    payload = f"{namespace}:{episode_seed}:{repeat_index}:{inference_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class PairedOutcomeSummary:
    """Counts for a fixed set of paired binary rollout outcomes."""

    pairs: int
    baseline_successes: int
    treatment_successes: int
    improved: int
    regressed: int
    unchanged: int

    @property
    def delta(self) -> float:
        return (self.treatment_successes - self.baseline_successes) / self.pairs

    @property
    def resolution(self) -> float:
        return 1.0 / self.pairs


def summarize_paired_outcomes(
    baseline: Iterable[bool],
    treatment: Iterable[bool],
) -> PairedOutcomeSummary:
    """Summarize paired outcomes in input order; do not silently truncate."""

    baseline_values = tuple(bool(value) for value in baseline)
    treatment_values = tuple(bool(value) for value in treatment)
    if not baseline_values or len(baseline_values) != len(treatment_values):
        raise ValueError("baseline and treatment must have the same non-zero number of outcomes")
    improved = sum(not base and treated for base, treated in zip(baseline_values, treatment_values, strict=True))
    regressed = sum(base and not treated for base, treated in zip(baseline_values, treatment_values, strict=True))
    pairs = len(baseline_values)
    return PairedOutcomeSummary(
        pairs=pairs,
        baseline_successes=sum(baseline_values),
        treatment_successes=sum(treatment_values),
        improved=improved,
        regressed=regressed,
        unchanged=pairs - improved - regressed,
    )
