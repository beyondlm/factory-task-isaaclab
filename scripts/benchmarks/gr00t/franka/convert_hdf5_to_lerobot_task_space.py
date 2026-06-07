# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert Franka pick-and-place IK-relative HDF5 demos to GR00T-LeRobot v2.

This converter is scoped to datasets recorded from:
    Isaac-Pick-Place-Franka-IK-Rel-v0

The recorded HDF5 actions are relative SE(3) teleop commands:
    [dx, dy, dz, d_rx, d_ry, d_rz, gripper_cmd]

The LeRobot state is absolute task-space:
    [eef_pos(3), eef_quat(4), gripper_width(1)]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).parent.resolve()

LEROBOT_KEY = {
    "state": "observation.state",
    "action": "action",
    "video": {
        "observation.images.wrist_camera": ("wrist_cam", "wrist_camera"),
        "observation.images.table_camera": ("table_cam", "table_camera"),
    },
    "task": "annotation.human.action.task_description",
    "valid": "annotation.human.action.valid",
}

STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qw",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "gripper_width",
]

ACTION_NAMES = [
    "eef_delta_x",
    "eef_delta_y",
    "eef_delta_z",
    "eef_delta_rx",
    "eef_delta_ry",
    "eef_delta_rz",
    "gripper_cmd",
]


@dataclass
class Config:
    hdf5_file_path: Path = Path("datasets/dataset_sorting_105.hdf5")
    lerobot_data_dir: Path | None = None
    video_dir: Path | None = None
    modality_template_path: Path = SCRIPT_DIR / "modality_task_space.json"
    chunks_size: int = 1000
    fps: int = 30
    total_episodes: int = 0
    robot_type: str = "franka_pick_place_relative_task_space"
    task_index: int = 0
    task_description: str = (
        "Pick up the labeled box and place it into the blue bin. "
        "Pick up the unlabeled box and place it into the black bin."
    )
    overwrite: bool = False
    require_videos: bool = False
    only_success: bool = True

    data_path: str = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    video_path: str = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    def resolve(self) -> "Config":
        self.hdf5_file_path = self.hdf5_file_path.expanduser().resolve()
        if self.lerobot_data_dir is None:
            self.lerobot_data_dir = self.hdf5_file_path.with_suffix("") / "lerobot_task_space"
        else:
            self.lerobot_data_dir = self.lerobot_data_dir.expanduser().resolve()

        if self.video_dir is not None:
            self.video_dir = self.video_dir.expanduser().resolve()
        return self


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-file-path", type=Path, default=Config.hdf5_file_path)
    parser.add_argument("--lerobot-data-dir", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--modality-template-path", type=Path, default=Config.modality_template_path)
    parser.add_argument("--chunks-size", type=int, default=Config.chunks_size)
    parser.add_argument("--fps", type=int, default=Config.fps)
    parser.add_argument("--total-episodes", type=int, default=Config.total_episodes, help="0 means all episodes.")
    parser.add_argument("--robot-type", type=str, default=Config.robot_type)
    parser.add_argument("--task-index", type=int, default=Config.task_index)
    parser.add_argument("--task-description", type=str, default=Config.task_description)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-videos", action="store_true")
    parser.add_argument("--include-failed", action="store_true", help="Include HDF5 demos with success=False.")
    args = parser.parse_args()
    return Config(
        hdf5_file_path=args.hdf5_file_path,
        lerobot_data_dir=args.lerobot_data_dir,
        video_dir=args.video_dir,
        modality_template_path=args.modality_template_path,
        chunks_size=args.chunks_size,
        fps=args.fps,
        total_episodes=args.total_episodes,
        robot_type=args.robot_type,
        task_index=args.task_index,
        task_description=args.task_description,
        overwrite=args.overwrite,
        require_videos=args.require_videos,
        only_success=not args.include_failed,
    ).resolve()


def dump_json(data: Any, path: Path, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
        f.write("\n")


def dump_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sorted_demo_ids(hdf5_data: h5py.Group) -> list[str]:
    return sorted([k for k in hdf5_data.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))


def gripper_width(gripper_pos: np.ndarray) -> np.ndarray:
    return np.abs(gripper_pos[:, 0] - gripper_pos[:, 1]).reshape(-1, 1)


def convert_trajectory_to_df(
    trajectory: h5py.Group,
    episode_index: int,
    index_start: int,
    config: Config,
) -> tuple[pd.DataFrame, int]:
    eef_pos = trajectory["obs"]["eef_pos"][()].astype(np.float32)
    eef_quat = trajectory["obs"]["eef_quat"][()].astype(np.float32)
    gripper_pos = trajectory["obs"]["gripper_pos"][()].astype(np.float32)
    actions = trajectory["actions"][()].astype(np.float32)

    length = min(len(eef_pos), len(eef_quat), len(gripper_pos), len(actions)) - 1
    if length <= 0:
        raise ValueError(f"Episode {episode_index} is too short: {length + 1} samples")

    state = np.concatenate([eef_pos[:length], eef_quat[:length], gripper_width(gripper_pos[:length])], axis=1)
    action = actions[:length]
    if action.shape[1] != 7:
        raise ValueError(f"Expected Franka IK-relative action dim 7, got {action.shape[1]}")

    data: dict[str, Any] = {
        LEROBOT_KEY["state"]: [row for row in state],
        LEROBOT_KEY["action"]: [row for row in action],
        "timestamp": np.arange(length, dtype=np.float64) * (1.0 / config.fps),
        LEROBOT_KEY["task"]: np.full(length, config.task_index, dtype=np.int64),
        LEROBOT_KEY["valid"]: np.ones(length, dtype=bool),
        "episode_index": np.full(length, episode_index, dtype=np.int64),
        "task_index": np.full(length, config.task_index, dtype=np.int64),
        "index": np.arange(length, dtype=np.int64) + index_start,
    }

    reward = np.zeros(length, dtype=np.float64)
    reward[-1] = 1.0
    done = np.zeros(length, dtype=bool)
    done[-1] = True
    data["next.reward"] = reward
    data["next.done"] = done

    return pd.DataFrame(data), length


def get_video_metadata(video_path: Path) -> dict[str, Any] | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        probe_data = json.loads(subprocess.check_output(cmd).decode("utf-8"))
        stream = probe_data["streams"][0]
        num, den = map(int, stream["r_frame_rate"].split("/"))
    except Exception as exc:
        print(f"Warning: could not read video metadata for {video_path}: {exc}")
        return None

    return {
        "dtype": "video",
        "shape": [int(stream["height"]), int(stream["width"]), 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": num / den,
            "video.codec": stream["codec_name"],
            "video.pix_fmt": stream["pix_fmt"],
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def find_video(video_dir: Path, trajectory_id: str, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        for name in (f"{trajectory_id}_{suffix}.mp4", f"{trajectory_id}_{suffix}_rgb.mp4"):
            candidate = video_dir / name
            if candidate.exists():
                return candidate
    return None


def copy_episode_videos(config: Config, trajectory_id: str, episode_index: int) -> dict[str, Path]:
    copied_paths: dict[str, Path] = {}
    if config.video_dir is None or not config.video_dir.exists():
        return copied_paths

    episode_chunk = episode_index // config.chunks_size
    for video_key, suffixes in LEROBOT_KEY["video"].items():
        source = find_video(config.video_dir, trajectory_id, suffixes)
        if source is None:
            message = f"Video missing for {trajectory_id}: tried suffixes {suffixes} in {config.video_dir}"
            if config.require_videos:
                raise FileNotFoundError(message)
            print(f"Warning: {message}")
            continue

        relpath = config.video_path.format(
            episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index
        )
        destination = config.lerobot_data_dir / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_paths[video_key] = destination
    return copied_paths


def feature_info_for_dataframe(step_data: pd.DataFrame, video_paths: dict[str, Path]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for video_key, video_path in video_paths.items():
        metadata = get_video_metadata(video_path)
        if metadata is not None:
            features[video_key] = metadata

    for column in step_data.columns:
        column_data = np.stack(step_data[column], axis=0)
        if column_data.ndim == 1:
            shape = (1,)
        else:
            shape = column_data.shape[1:]
        features[column] = {"dtype": column_data.dtype.name, "shape": list(shape)}

    features[LEROBOT_KEY["state"]]["names"] = STATE_NAMES
    features[LEROBOT_KEY["action"]]["names"] = ACTION_NAMES
    return features


def write_modality(config: Config, use_videos: bool) -> None:
    modality = load_json(config.modality_template_path)
    if not use_videos:
        modality.pop("video", None)
    dump_json(modality, config.lerobot_data_dir / "meta" / "modality.json", indent=4)


def generate_info(
    config: Config,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    example_data: pd.DataFrame,
    example_video_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "codebase_version": "v2.0",
        "robot_type": config.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": (total_episodes + config.chunks_size - 1) // config.chunks_size,
        "chunks_size": config.chunks_size,
        "fps": config.fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": config.data_path,
        "video_path": config.video_path if total_videos else None,
        "features": feature_info_for_dataframe(example_data, example_video_paths),
    }


def convert(config: Config) -> None:
    if not config.hdf5_file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {config.hdf5_file_path}")
    if not config.modality_template_path.exists():
        raise FileNotFoundError(f"Modality template not found: {config.modality_template_path}")
    if config.require_videos and (config.video_dir is None or not config.video_dir.exists()):
        raise FileNotFoundError(f"--require-videos set but video dir is missing: {config.video_dir}")

    if config.lerobot_data_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite to replace: {config.lerobot_data_dir}")
        shutil.rmtree(config.lerobot_data_dir)

    config.lerobot_data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = config.lerobot_data_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_videos = 0
    episodes_info: list[dict[str, Any]] = []
    example_data: pd.DataFrame | None = None
    example_video_paths: dict[str, Path] = {}

    with h5py.File(config.hdf5_file_path, "r") as hdf5_file:
        hdf5_data = hdf5_file["data"]
        trajectory_ids = sorted_demo_ids(hdf5_data)
        if config.total_episodes > 0:
            trajectory_ids = trajectory_ids[: config.total_episodes]

        print(f"Converting {len(trajectory_ids)} HDF5 episodes from {config.hdf5_file_path}")
        for trajectory_id in tqdm(trajectory_ids):
            trajectory = hdf5_data[trajectory_id]
            if config.only_success and not bool(trajectory.attrs.get("success", False)):
                continue

            episode_index = len(episodes_info)
            dataframe, length = convert_trajectory_to_df(
                trajectory=trajectory, episode_index=episode_index, index_start=total_frames, config=config
            )

            episode_chunk = episode_index // config.chunks_size
            relpath = config.data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
            save_path = config.lerobot_data_dir / relpath
            save_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_parquet(save_path)

            video_paths = copy_episode_videos(config, trajectory_id, episode_index)
            total_videos += len(video_paths)
            if video_paths and not example_video_paths:
                example_video_paths = video_paths

            total_frames += length
            episodes_info.append(
                {
                    "episode_index": episode_index,
                    "tasks": [config.task_description],
                    "length": length,
                }
            )

            if example_data is None:
                example_data = dataframe

    if example_data is None:
        raise RuntimeError("No episodes were converted.")

    dump_jsonl([{"task_index": config.task_index, "task": config.task_description}], meta_dir / "tasks.jsonl")
    dump_jsonl(episodes_info, meta_dir / "episodes.jsonl")
    write_modality(config, use_videos=bool(total_videos))
    dump_json(
        generate_info(
            config=config,
            total_episodes=len(episodes_info),
            total_frames=total_frames,
            total_videos=total_videos,
            example_data=example_data,
            example_video_paths=example_video_paths,
        ),
        meta_dir / "info.json",
        indent=4,
    )

    print("\nConversion completed:")
    print(f"Total episodes processed: {len(episodes_info)}")
    print(f"Total frames: {total_frames}")
    print(f"Total videos copied: {total_videos}")
    print(f"Output directory: {config.lerobot_data_dir}")


if __name__ == "__main__":
    convert(parse_args())
