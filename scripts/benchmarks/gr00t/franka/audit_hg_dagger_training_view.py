# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit a converted LeRobot HG-DAgger recovery training view.

This tool cross-checks every converted recovery episode against the complete
source HDF5 rollout. It verifies strict human-only action-horizon anchors and
reports distribution and handoff diagnostics without changing either dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

import convert_hdf5_to_lerobot_joint_space as base
from hg_dagger_core import contiguous_true_segments, valid_horizon_anchors


REQUIRED_COLUMNS = (
    "observation.state",
    "action",
    "action.policy",
    "action.expert",
    "action.executed_command",
    "annotation.human.action.intervention",
    "annotation.policy.action.valid",
    "annotation.human.action.valid",
    "episode_index",
    "index",
)

SOURCE_COLUMN_MAP = {
    "observation.state": "dagger/observation_joint_state",
    "action": "dagger/achieved_joint_state",
    "action.policy": "dagger/policy_action",
    "action.expert": "dagger/expert_action",
    "action.executed_command": "dagger/executed_action",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-file", type=Path, required=True)
    parser.add_argument("--lerobot-data-dir", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--minimum-segment-length", type=int, default=64)
    parser.add_argument("--xy-threshold", type=float, default=0.08)
    parser.add_argument("--min-height-diff", type=float, default=-0.01)
    parser.add_argument("--max-height-diff", type=float, default=0.035)
    parser.add_argument(
        "--allow-failed-source",
        action="store_true",
        help=(
            "Audit recovery segments from source rollouts whose final task result is failure. "
            "By default these are rejected to protect success-only training views."
        ),
    )
    parser.add_argument(
        "--allow-truncated-video-tail",
        action="store_true",
        help="Allow a source-final recovery to omit the last transition when no matching video frame exists.",
    )
    parser.add_argument("--binary-gripper-command-target", action="store_true")
    parser.add_argument("--gripper-close-width", type=float, default=0.0)
    parser.add_argument("--gripper-open-width", type=float, default=0.08)
    parser.add_argument(
        "--bin-half-extents",
        type=float,
        nargs=2,
        metavar=("HALF_X", "HALF_Y"),
        default=(0.10018354, 0.12499835),
        help="Bin local-frame outer half extents [m] from the USD asset.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.action_horizon < 1:
        parser.error("--action-horizon must be positive")
    if args.minimum_segment_length < args.action_horizon:
        parser.error("--minimum-segment-length must be at least --action-horizon")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def hdf5_value(group: h5py.Group, key: str) -> np.ndarray:
    value: h5py.Group | h5py.Dataset = group
    for part in key.split("/"):
        value = value[part]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"Expected HDF5 dataset at {group.name}/{key}")
    return value[()]


def parquet_path(dataset_dir: Path, episode_index: int) -> Path:
    matches = list(dataset_dir.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(f"Expected one parquet for episode {episode_index}, found {len(matches)}")
    return matches[0]


def dataframe_array(dataframe: pd.DataFrame, column: str) -> np.ndarray:
    values = dataframe[column].to_numpy()
    if values.dtype == object:
        return np.stack(values)
    return values


def source_success(trajectory: h5py.Group) -> bool:
    value = np.asarray(hdf5_value(trajectory, "dagger/episode/success")).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{trajectory.name}: expected one success value")
    return bool(value[0])


def pose_position(trajectory: h5py.Group, name: str) -> np.ndarray:
    pose = hdf5_value(trajectory, f"obs/{name}_pose")
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"{trajectory.name}: obs/{name}_pose has shape {pose.shape}")
    return pose[:, :3]


def pose_quaternion_xyzw(trajectory: h5py.Group, name: str) -> np.ndarray:
    return hdf5_value(trajectory, f"obs/{name}_pose")[:, 3:]


def z_is_inside(z_difference: np.ndarray, *, minimum: float, maximum: float) -> np.ndarray:
    return np.logical_and(z_difference > minimum, z_difference < maximum)


def radial_geometry(
    object_position: np.ndarray,
    bin_position: np.ndarray,
    *,
    xy_threshold: float,
    min_height_diff: float,
    max_height_diff: float,
) -> np.ndarray:
    difference = object_position - bin_position
    xy_distance = np.linalg.norm(difference[:, :2], axis=1)
    return np.logical_and(
        xy_distance < xy_threshold,
        z_is_inside(difference[:, 2], minimum=min_height_diff, maximum=max_height_diff),
    )


def quaternion_yaw_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rectangular_geometry(
    object_position: np.ndarray,
    bin_position: np.ndarray,
    bin_quaternion_xyzw: np.ndarray,
    *,
    half_extents: tuple[float, float],
    min_height_diff: float,
    max_height_diff: float,
) -> tuple[np.ndarray, np.ndarray]:
    difference = object_position - bin_position
    yaw = quaternion_yaw_xyzw(bin_quaternion_xyzw)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    local_x = cosine * difference[:, 0] + sine * difference[:, 1]
    local_y = -sine * difference[:, 0] + cosine * difference[:, 1]
    margin_x = half_extents[0] - np.abs(local_x)
    margin_y = half_extents[1] - np.abs(local_y)
    margin = np.minimum(margin_x, margin_y)
    inside_xy = np.logical_and(margin_x > 0.0, margin_y > 0.0)
    inside = np.logical_and(
        inside_xy,
        z_is_inside(difference[:, 2], minimum=min_height_diff, maximum=max_height_diff),
    )
    return inside, margin


def phase_name(a_done: bool, b_done: bool) -> str:
    if a_done and b_done:
        return "both_done"
    if a_done:
        return "a_done"
    if b_done:
        return "b_done"
    return "none_done"


def layout_name(trajectory: h5py.Group) -> str:
    a_y = float(hdf5_value(trajectory, "obs/a_pose")[0, 1])
    b_y = float(hdf5_value(trajectory, "obs/b_pose")[0, 1])
    c_y = float(hdf5_value(trajectory, "obs/c_pose")[0, 1])
    d_y = float(hdf5_value(trajectory, "obs/d_pose")[0, 1])
    return "same_side" if (a_y - b_y) * (c_y - d_y) > 0.0 else "cross_side"


def first_true_offset(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if len(indices) == 0 else int(indices[0])


def next_segment_start(mask: np.ndarray, end: int, minimum_length: int) -> int:
    later_segments = contiguous_true_segments(mask[end:], minimum_length)
    return len(mask) if not later_segments else end + later_segments[0][0]


def scalar_bool_column(dataframe: pd.DataFrame, column: str) -> np.ndarray:
    return dataframe_array(dataframe, column).astype(bool).reshape(-1)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.lerobot_data_dir.resolve()
    metadata = read_jsonl(dataset_dir / "meta/episodes.jsonl")
    metadata.sort(key=lambda item: int(item["episode_index"]))
    expected_indices = list(range(len(metadata)))
    actual_indices = [int(item["episode_index"]) for item in metadata]
    if actual_indices != expected_indices:
        raise ValueError("LeRobot episode metadata is not contiguous from zero")

    segment_records: list[dict[str, Any]] = []
    global_indices: list[np.ndarray] = []
    source_counts: Counter[str] = Counter()
    phase_layout_counts: Counter[tuple[str, str]] = Counter()
    phase_layout_anchors: Counter[tuple[str, str]] = Counter()

    with h5py.File(args.hdf5_file.resolve(), "r") as source:
        for item in metadata:
            episode_index = int(item["episode_index"])
            source_episode = str(item["source_episode"])
            trajectory = source[f"data/{source_episode}"]
            source_range = tuple(int(value) for value in item["source_range"])
            recovery_range = tuple(int(value) for value in item["recovery_range"])
            context_start, source_end = source_range
            recovery_start, recovery_end = recovery_range
            if source_end != recovery_end or context_start > recovery_start:
                raise ValueError(f"episode {episode_index}: inconsistent source/recovery ranges")
            mask = hdf5_value(trajectory, "dagger/intervention_mask").astype(bool).reshape(-1)
            source_segments = contiguous_true_segments(mask, args.minimum_segment_length)
            source_segment = next(
                ((start, end) for start, end in source_segments if start == recovery_start),
                None,
            )
            if source_segment is None:
                raise ValueError(
                    f"episode {episode_index}: recovery range is not a source intervention segment"
                )
            truncated_source_tail_frames = source_segment[1] - recovery_end
            if truncated_source_tail_frames != 0 and not (
                args.allow_truncated_video_tail
                and truncated_source_tail_frames == 1
                and source_segment[1] == len(mask)
            ):
                raise ValueError(
                    f"episode {episode_index}: recovery end differs from source segment by "
                    f"{truncated_source_tail_frames} frames"
                )
            is_source_success = source_success(trajectory)
            if not is_source_success and not args.allow_failed_source:
                raise ValueError(f"episode {episode_index}: success-only view contains failed source episode")

            dataframe = pd.read_parquet(parquet_path(dataset_dir, episode_index))
            missing = [column for column in REQUIRED_COLUMNS if column not in dataframe]
            if missing:
                raise ValueError(f"episode {episode_index}: missing columns {missing}")
            expected_length = source_end - context_start
            if len(dataframe) != expected_length or len(dataframe) != int(item["length"]):
                raise ValueError(f"episode {episode_index}: metadata/parquet length mismatch")
            if not np.all(dataframe["episode_index"].to_numpy() == episode_index):
                raise ValueError(f"episode {episode_index}: incorrect episode_index column")

            source_slice = slice(context_start, source_end)
            for column, source_key in SOURCE_COLUMN_MAP.items():
                converted = dataframe_array(dataframe, column)
                expected = hdf5_value(trajectory, source_key)[source_slice]
                if column == "action" and args.binary_gripper_command_target:
                    expected = expected.copy()
                    executed = hdf5_value(trajectory, "dagger/executed_action")[source_slice]
                    expected[:, -1] = base.binary_gripper_command_to_width(
                        executed[:, -1],
                        close_width=args.gripper_close_width,
                        open_width=args.gripper_open_width,
                    )
                if not np.all(np.isfinite(converted)):
                    raise ValueError(f"episode {episode_index}: {column} contains non-finite values")
                if not np.allclose(converted, expected, rtol=0.0, atol=0.0):
                    raise ValueError(f"episode {episode_index}: {column} differs from source HDF5")

            intervention = scalar_bool_column(dataframe, "annotation.human.action.intervention")
            source_intervention = mask[source_slice]
            if not np.array_equal(intervention, source_intervention):
                raise ValueError(f"episode {episode_index}: intervention mask differs from source")
            valid = scalar_bool_column(dataframe, "annotation.human.action.valid")
            expected_valid = valid_horizon_anchors(intervention, args.action_horizon)
            if not np.array_equal(valid, expected_valid):
                raise ValueError(f"episode {episode_index}: strict-H{args.action_horizon} mask is incorrect")
            policy_valid = scalar_bool_column(dataframe, "annotation.policy.action.valid")
            if np.any(np.logical_and(policy_valid, intervention)):
                raise ValueError(f"episode {episode_index}: policy action is valid during intervention")
            global_indices.append(dataframe["index"].to_numpy(dtype=np.int64))

            a_position = pose_position(trajectory, "a")
            b_position = pose_position(trajectory, "b")
            c_position = pose_position(trajectory, "c")
            d_position = pose_position(trajectory, "d")
            c_quaternion = pose_quaternion_xyzw(trajectory, "c")
            d_quaternion = pose_quaternion_xyzw(trajectory, "d")
            a_radial = radial_geometry(
                a_position,
                c_position,
                xy_threshold=args.xy_threshold,
                min_height_diff=args.min_height_diff,
                max_height_diff=args.max_height_diff,
            )
            b_radial = radial_geometry(
                b_position,
                d_position,
                xy_threshold=args.xy_threshold,
                min_height_diff=args.min_height_diff,
                max_height_diff=args.max_height_diff,
            )
            a_rectangular, a_margin = rectangular_geometry(
                a_position,
                c_position,
                c_quaternion,
                half_extents=tuple(args.bin_half_extents),
                min_height_diff=args.min_height_diff,
                max_height_diff=args.max_height_diff,
            )
            b_rectangular, b_margin = rectangular_geometry(
                b_position,
                d_position,
                d_quaternion,
                half_extents=tuple(args.bin_half_extents),
                min_height_diff=args.min_height_diff,
                max_height_diff=args.max_height_diff,
            )

            start_index = recovery_start
            end_index = recovery_end - 1
            phase = phase_name(bool(a_radial[start_index]), bool(b_radial[start_index]))
            rectangular_phase = phase_name(
                bool(a_rectangular[start_index]), bool(b_rectangular[start_index])
            )
            layout = layout_name(trajectory)
            anchors = int(valid.sum())
            source_counts[source_episode] += 1
            phase_layout_counts[(layout, phase)] += 1
            phase_layout_anchors[(layout, phase)] += anchors

            next_start = next_segment_start(mask, recovery_end, args.minimum_segment_length)
            handoff_slice = slice(recovery_end, next_start)
            radial_complete_after_handoff = np.logical_and(a_radial[handoff_slice], b_radial[handoff_slice])
            rectangular_complete_after_handoff = np.logical_and(
                a_rectangular[handoff_slice], b_rectangular[handoff_slice]
            )
            start_radial_count = int(a_radial[start_index]) + int(b_radial[start_index])
            future_radial_count = a_radial[recovery_end:].astype(np.int8) + b_radial[recovery_end:].astype(np.int8)
            new_radial_placement = future_radial_count > start_radial_count

            eef_position = hdf5_value(trajectory, "obs/eef_pos")[recovery_start:recovery_end]
            eef_step = np.diff(eef_position, axis=0)
            eef_path_length = float(np.linalg.norm(eef_step, axis=1).sum())
            object_a_displacement = float(
                np.linalg.norm(a_position[recovery_start:recovery_end] - a_position[recovery_start], axis=1).max()
            )
            object_b_displacement = float(
                np.linalg.norm(b_position[recovery_start:recovery_end] - b_position[recovery_start], axis=1).max()
            )
            gripper = hdf5_value(trajectory, "obs/gripper_pos")[recovery_start:recovery_end]
            gripper_range = float(np.ptp(gripper, axis=0).max())

            segment_records.append(
                {
                    "episode_index": episode_index,
                    "source_episode": source_episode,
                    "source_success": is_source_success,
                    "truncated_source_tail_frames": truncated_source_tail_frames,
                    "recovery_range": [recovery_start, recovery_end],
                    "length": len(dataframe),
                    "valid_h32_anchors": anchors,
                    "layout": layout,
                    "phase_radial": phase,
                    "phase_rectangular": rectangular_phase,
                    "task_complete_at_start_radial": bool(a_radial[start_index] and b_radial[start_index]),
                    "task_complete_at_start_rectangular": bool(
                        a_rectangular[start_index] and b_rectangular[start_index]
                    ),
                    "correct_count_start_radial": start_radial_count,
                    "correct_count_end_radial": int(a_radial[end_index]) + int(b_radial[end_index]),
                    "correct_count_start_rectangular": int(a_rectangular[start_index])
                    + int(b_rectangular[start_index]),
                    "correct_count_end_rectangular": int(a_rectangular[end_index])
                    + int(b_rectangular[end_index]),
                    "min_rectangular_margin_at_start_m": float(
                        min(a_margin[start_index], b_margin[start_index])
                    ),
                    "first_new_radial_placement_after_handoff_frames": first_true_offset(
                        new_radial_placement
                    ),
                    "radial_task_complete_before_next_takeover": bool(
                        radial_complete_after_handoff.any()
                    ),
                    "rectangular_task_complete_before_next_takeover": bool(
                        rectangular_complete_after_handoff.any()
                    ),
                    "eef_path_length_m": eef_path_length,
                    "object_a_max_displacement_m": object_a_displacement,
                    "object_b_max_displacement_m": object_b_displacement,
                    "gripper_range_m": gripper_range,
                }
            )

    concatenated_indices = np.concatenate(global_indices)
    if not np.array_equal(concatenated_indices, np.arange(len(concatenated_indices))):
        raise ValueError("LeRobot global index column is not contiguous")

    total_frames = sum(record["length"] for record in segment_records)
    total_anchors = sum(record["valid_h32_anchors"] for record in segment_records)
    anchor_shares = [record["valid_h32_anchors"] / total_anchors for record in segment_records]
    episode_anchor_counts: dict[str, int] = defaultdict(int)
    for record in segment_records:
        episode_anchor_counts[record["source_episode"]] += record["valid_h32_anchors"]

    phase_layout_table = []
    for layout in ("cross_side", "same_side"):
        for phase in ("none_done", "a_done", "b_done", "both_done"):
            count = phase_layout_counts[(layout, phase)]
            if count:
                phase_layout_table.append(
                    {
                        "layout": layout,
                        "phase_radial": phase,
                        "segments": count,
                        "valid_h32_anchors": phase_layout_anchors[(layout, phase)],
                    }
                )

    summary = {
        "status": "valid",
        "action_horizon": args.action_horizon,
        "segments": len(segment_records),
        "source_episodes": len(source_counts),
        "successful_source_segments": sum(record["source_success"] for record in segment_records),
        "failed_source_segments": sum(not record["source_success"] for record in segment_records),
        "successful_source_episodes": len(
            {record["source_episode"] for record in segment_records if record["source_success"]}
        ),
        "failed_source_episodes": len(
            {record["source_episode"] for record in segment_records if not record["source_success"]}
        ),
        "video_tail_truncated_segments": sum(
            record["truncated_source_tail_frames"] > 0 for record in segment_records
        ),
        "frames": total_frames,
        "strict_h32_anchors": total_anchors,
        "segments_per_source_episode": dict(sorted(Counter(source_counts.values()).items())),
        "max_segment_anchor_share": max(anchor_shares),
        "max_source_episode_anchor_share": max(episode_anchor_counts.values()) / total_anchors,
        "task_complete_at_start_radial": sum(
            record["task_complete_at_start_radial"] for record in segment_records
        ),
        "task_complete_at_start_rectangular": sum(
            record["task_complete_at_start_rectangular"] for record in segment_records
        ),
        "correct_count_changed_during_human_radial": sum(
            record["correct_count_start_radial"] != record["correct_count_end_radial"]
            for record in segment_records
        ),
        "correct_count_changed_during_human_rectangular": sum(
            record["correct_count_start_rectangular"] != record["correct_count_end_rectangular"]
            for record in segment_records
        ),
        "radial_task_complete_before_next_takeover": sum(
            record["radial_task_complete_before_next_takeover"] for record in segment_records
        ),
        "rectangular_task_complete_before_next_takeover": sum(
            record["rectangular_task_complete_before_next_takeover"] for record in segment_records
        ),
        "phase_layout": phase_layout_table,
        "motion_quantiles": {
            key: {
                "minimum": float(np.min(values)),
                "p10": float(np.quantile(values, 0.10)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
                "maximum": float(np.max(values)),
            }
            for key in (
                "eef_path_length_m",
                "object_a_max_displacement_m",
                "object_b_max_displacement_m",
                "gripper_range_m",
            )
            for values in [[record[key] for record in segment_records]]
        },
    }
    return {"summary": summary, "segments": segment_records}


def main() -> None:
    args = parse_args()
    result = audit(args)
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
