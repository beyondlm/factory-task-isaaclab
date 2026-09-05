# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relabel a LeRobot dataset with recorded binary Franka gripper commands.

The arm action, observations, videos, episode order, and validity masks are
preserved. Only the final element of action is replaced:

    -1 (close) -> gripper_close_width
    +1 (open)  -> gripper_open_width

Commands can come from an existing vector column (HG-DAgger recovery data) or
from one or more original Isaac Lab HDF5 files. HDF5 episodes are matched to
LeRobot episodes by the complete float32 arm-action trajectory, not by episode
number or length alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import convert_hdf5_to_lerobot_joint_space as base
import h5py
import numpy as np
import pandas as pd

EXCLUDED_META_FILES = {
    "stats.json",
    "relative_stats.json",
    "baseline_comparison.json",
    "gripper_relabel.json",
}


@dataclass(frozen=True)
class Hdf5CommandSource:
    path: Path
    episode: str
    arm_action: np.ndarray
    gripper_command: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=episode_index // int(info["chunks_size"]),
        episode_index=episode_index,
    )


def arm_digest(arm_action: np.ndarray) -> str:
    value = np.ascontiguousarray(arm_action, dtype=np.float32)
    return hashlib.sha256(value.tobytes()).hexdigest()


def build_hdf5_command_index(paths: list[Path]) -> dict[tuple[int, str], list[Hdf5CommandSource]]:
    index: dict[tuple[int, str], list[Hdf5CommandSource]] = {}
    for path in paths:
        with h5py.File(path, "r") as source:
            for episode in base.sorted_demo_ids(source["data"]):
                trajectory = source["data"][episode]
                joint_position = base.read_nested(trajectory, base.HDF5_JOINT_POSITION_KEY)[()].astype(np.float32)
                recorded_action = trajectory["actions"][()].astype(np.float32)
                if recorded_action.ndim != 2 or len(recorded_action) != len(joint_position):
                    raise ValueError(
                        f"{path}:{episode} has misaligned actions={recorded_action.shape}, "
                        f"joint_position={joint_position.shape}"
                    )
                arm_action = np.ascontiguousarray(joint_position[1:, :7], dtype=np.float32)
                command = np.ascontiguousarray(recorded_action[:-1, -1], dtype=np.float32)
                row = Hdf5CommandSource(path, episode, arm_action, command)
                index.setdefault((len(arm_action), arm_digest(arm_action)), []).append(row)
    return index


def command_from_hdf5(
    arm_action: np.ndarray,
    index: dict[tuple[int, str], list[Hdf5CommandSource]],
) -> tuple[np.ndarray, Hdf5CommandSource]:
    candidates = index.get((len(arm_action), arm_digest(arm_action)), [])
    exact = [row for row in candidates if np.array_equal(row.arm_action, arm_action)]
    if len(exact) != 1:
        matches = [(str(row.path), row.episode) for row in exact]
        raise ValueError(f"Expected one exact HDF5 trajectory match, got {matches}")
    return exact[0].gripper_command, exact[0]


def hardlink_tree(source_root: Path, output_root: Path) -> int:
    if not source_root.exists():
        return 0
    count = 0
    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        destination = output_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if not source.is_file():
            raise ValueError(f"Unsupported video tree entry: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        count += 1
    return count


def relabel(
    source_root: Path,
    output_root: Path,
    *,
    command_column: str | None,
    hdf5_paths: list[Path],
    close_width: float,
    open_width: float,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if bool(command_column) == bool(hdf5_paths):
        raise ValueError("Specify exactly one of --command-column or --hdf5-file")

    info = load_json(source_root / "meta/info.json")
    total_episodes = int(info["total_episodes"])
    hdf5_index = build_hdf5_command_index(hdf5_paths) if hdf5_paths else {}
    provenance: list[dict[str, Any]] = []
    total_frames = 0
    close_frames = 0
    open_frames = 0

    try:
        for episode_index in range(total_episodes):
            source_path = episode_path(source_root, info, episode_index)
            dataframe = pd.read_parquet(source_path)
            action = np.stack(dataframe["action"].to_numpy()).astype(np.float32)
            if action.ndim != 2 or action.shape[1] != 8:
                raise ValueError(f"{source_path}: expected action shape (N, 8), got {action.shape}")

            source_record: dict[str, Any]
            if command_column:
                if command_column not in dataframe:
                    raise ValueError(f"{source_path}: missing command column {command_column!r}")
                command_vector = np.stack(dataframe[command_column].to_numpy()).astype(np.float32)
                if command_vector.shape != action.shape:
                    raise ValueError(
                        f"{source_path}: command column shape {command_vector.shape} != action shape {action.shape}"
                    )
                command = command_vector[:, -1]
                source_record = {"command_column": command_column}
            else:
                command, match = command_from_hdf5(action[:, :7], hdf5_index)
                source_record = {
                    "hdf5_file": str(match.path),
                    "hdf5_episode": match.episode,
                }

            target = base.binary_gripper_command_to_width(
                command,
                close_width=close_width,
                open_width=open_width,
            )
            action[:, -1] = target
            dataframe["action"] = list(action)

            output_path = episode_path(output_root, info, episode_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_parquet(output_path)

            total_frames += len(dataframe)
            close_frames += int(np.count_nonzero(target == close_width))
            open_frames += int(np.count_nonzero(target == open_width))
            provenance.append({"episode_index": episode_index, **source_record})

        meta_source = source_root / "meta"
        meta_output = output_root / "meta"
        meta_output.mkdir(parents=True, exist_ok=True)
        for source in meta_source.iterdir():
            if source.name in EXCLUDED_META_FILES or not source.is_file():
                continue
            shutil.copy2(source, meta_output / source.name)

        if total_frames != int(info["total_frames"]):
            raise ValueError(f"Relabeled frame count {total_frames} != source metadata {info['total_frames']}")
        output_info = dict(info)
        output_info["splits"] = {"train": f"0:{total_episodes}"}
        dump_json(output_info, meta_output / "info.json")

        video_files = hardlink_tree(source_root / "videos", output_root / "videos")
        dump_json(
            {
                "source_dataset": str(source_root),
                "target_semantics": "arm=unchanged; gripper=recorded_binary_command_width",
                "gripper_close_width": close_width,
                "gripper_open_width": open_width,
                "total_episodes": total_episodes,
                "total_frames": total_frames,
                "close_frames": close_frames,
                "open_frames": open_frames,
                "video_storage": "hardlink",
                "video_files": video_files,
                "statistics_copied": False,
                "episode_command_sources": provenance,
            },
            meta_output / "gripper_relabel.json",
        )
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise

    print(
        f"Relabeled {total_episodes} episodes / {total_frames} frames: "
        f"close={close_frames}, open={open_frames}, videos={video_files}"
    )
    print(f"Output: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--command-column", type=str, default=None)
    parser.add_argument("--hdf5-file", type=Path, action="append", default=[])
    parser.add_argument("--gripper-close-width", type=float, default=0.0)
    parser.add_argument("--gripper-open-width", type=float, default=0.08)
    args = parser.parse_args()
    relabel(
        args.source_dataset.expanduser().resolve(),
        args.output_dataset.expanduser().resolve(),
        command_column=args.command_column,
        hdf5_paths=[path.expanduser().resolve() for path in args.hdf5_file],
        close_width=args.gripper_close_width,
        open_width=args.gripper_open_width,
    )


if __name__ == "__main__":
    main()
