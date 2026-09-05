# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate alignment and usable H32 windows in an HG-DAgger HDF5 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from hg_dagger_core import contiguous_true_segments, valid_horizon_anchors

STEP_FIELDS = (
    "observation_joint_state",
    "achieved_joint_state",
    "policy_action",
    "expert_action",
    "executed_action",
    "intervention_mask",
    "policy_action_valid",
    "inference_id",
    "chunk_index",
    "frame_index",
)

EPISODE_FIELDS = (
    "success",
    "outcome_code",
    "seed",
    "frame_count",
    "intervention_steps",
    "intervention_segments",
    "intervention_ratio",
    "valid_policy_actions",
)

SHADOW_FIELDS = (
    "observation_joint_state",
    "policy_action",
    "policy_action_valid",
    "query_frame",
)


def validate_episode(episode: h5py.Group, *, horizon: int, minimum_segment_length: int) -> dict[str, int]:
    if "actions" not in episode or "dagger" not in episode:
        raise ValueError(f"{episode.name}: missing actions or dagger group")
    step_count = len(episode["actions"])
    recorded_actions = episode["actions"][()]
    if recorded_actions.shape != (step_count, 8) or not np.all(np.isfinite(recorded_actions)):
        raise ValueError(f"{episode.name}: actions has invalid shape or non-finite values")
    for field in STEP_FIELDS:
        if field not in episode["dagger"]:
            raise ValueError(f"{episode.name}: missing dagger/{field}")
        if len(episode["dagger"][field]) != step_count:
            raise ValueError(
                f"{episode.name}: dagger/{field} has {len(episode['dagger'][field])} rows, expected {step_count}"
            )

    for field in (
        "observation_joint_state",
        "achieved_joint_state",
        "policy_action",
        "expert_action",
        "executed_action",
    ):
        value = episode["dagger"][field][()]
        if value.shape != (step_count, 8):
            raise ValueError(f"{episode.name}: dagger/{field} has invalid shape {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{episode.name}: dagger/{field} contains non-finite values")

    if "episode" not in episode["dagger"]:
        raise ValueError(f"{episode.name}: missing dagger/episode group")
    for field in EPISODE_FIELDS:
        if field not in episode["dagger/episode"]:
            raise ValueError(f"{episode.name}: missing dagger/episode/{field}")
        if np.asarray(episode["dagger/episode"][field][()]).size != 1:
            raise ValueError(f"{episode.name}: dagger/episode/{field} must contain one value")
    if "policy_checkpoint_id_utf8" not in episode["dagger/episode"]:
        raise ValueError(f"{episode.name}: missing dagger/episode/policy_checkpoint_id_utf8")
    checkpoint_bytes = np.asarray(episode["dagger/episode/policy_checkpoint_id_utf8"][()], dtype=np.uint8).reshape(-1)
    try:
        checkpoint_id = checkpoint_bytes.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{episode.name}: policy checkpoint id is not valid UTF-8") from exc
    if not checkpoint_id:
        raise ValueError(f"{episode.name}: policy checkpoint id is empty")
    recorded_frames = int(np.asarray(episode["dagger/episode/frame_count"][()]).reshape(-1)[0])
    if recorded_frames != step_count:
        raise ValueError(f"{episode.name}: episode frame_count is {recorded_frames}, expected {step_count}")

    if "shadow" in episode["dagger"]:
        shadow = episode["dagger/shadow"]
        for field in SHADOW_FIELDS:
            if field not in shadow:
                raise ValueError(f"{episode.name}: missing dagger/shadow/{field}")
        shadow_count = len(shadow["query_frame"])
        if any(len(shadow[field]) != shadow_count for field in SHADOW_FIELDS):
            raise ValueError(f"{episode.name}: shadow query fields have inconsistent lengths")
        query_frames = shadow["query_frame"][()].reshape(-1)
        if np.any(query_frames < 0) or np.any(query_frames >= step_count):
            raise ValueError(f"{episode.name}: shadow query_frame is outside the episode")

    mask = episode["dagger/intervention_mask"][()].astype(bool).reshape(-1)
    policy_valid = episode["dagger/policy_action_valid"][()].astype(bool).reshape(-1)
    executed = episode["dagger/executed_action"][()]
    policy = episode["dagger/policy_action"][()]
    expert = episode["dagger/expert_action"][()]
    if not np.allclose(recorded_actions, executed, rtol=0.0, atol=0.0):
        raise ValueError(f"{episode.name}: standard actions and dagger executed actions differ")
    if np.any(policy_valid & mask):
        raise ValueError(f"{episode.name}: policy action marked valid during intervention")
    if not np.allclose(executed[mask], expert[mask], rtol=0.0, atol=0.0):
        raise ValueError(f"{episode.name}: executed and expert actions differ during intervention")
    if not np.allclose(executed[~mask], policy[~mask], rtol=0.0, atol=0.0):
        raise ValueError(f"{episode.name}: executed and policy actions differ outside intervention")
    frame_index = episode["dagger/frame_index"][()].reshape(-1)
    if not np.array_equal(frame_index, np.arange(step_count)):
        raise ValueError(f"{episode.name}: frame_index is not contiguous from zero")

    segments = contiguous_true_segments(mask, minimum_segment_length)
    anchors = valid_horizon_anchors(mask, horizon)
    return {
        "steps": step_count,
        "intervention_steps": int(mask.sum()),
        "segments": len(segments),
        "valid_anchors": int(anchors.sum()),
    }


def validate(path: Path, *, horizon: int, minimum_segment_length: int) -> None:
    totals = {"episodes": 0, "steps": 0, "intervention_steps": 0, "segments": 0, "valid_anchors": 0}
    with h5py.File(path, "r") as stream:
        if "data" not in stream:
            raise ValueError("Missing HDF5 data group")
        for name in sorted(stream["data"]):
            result = validate_episode(
                stream["data"][name],
                horizon=horizon,
                minimum_segment_length=minimum_segment_length,
            )
            totals["episodes"] += 1
            for key, value in result.items():
                totals[key] += value

    print(
        "HG-DAgger dataset valid: "
        f"{totals['episodes']} episodes, {totals['steps']} steps, "
        f"{totals['intervention_steps']} intervention steps, "
        f"{totals['segments']} recovery segments, {totals['valid_anchors']} H{horizon} anchors"
    )
    if totals["valid_anchors"] == 0:
        raise RuntimeError("Dataset is structurally valid but contains no usable action-horizon anchors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--minimum-segment-length", type=int, default=64)
    args = parser.parse_args()
    validate(
        args.dataset_file.resolve(),
        horizon=args.action_horizon,
        minimum_segment_length=args.minimum_segment_length,
    )


if __name__ == "__main__":
    main()
