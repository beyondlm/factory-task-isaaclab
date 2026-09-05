# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert Franka HG-DAgger recovery segments to a LeRobot v2 dataset.

The source HDF5 keeps complete rollouts. This converter emits one LeRobot
episode per contiguous human recovery segment and marks only anchors with a
complete future action horizon as valid.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import convert_hdf5_to_lerobot_joint_space as base
import h5py
import numpy as np
import pandas as pd
from hg_dagger_core import contiguous_true_segments, valid_horizon_anchors
from tqdm import tqdm


@dataclass
class Config(base.Config):
    minimum_segment_length: int = 64
    action_horizon: int = 32
    state_history_frames: int = 1
    include_failed: bool = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-file-path", type=Path, required=True)
    parser.add_argument("--lerobot-data-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--modality-template-path", type=Path, default=base.Config.modality_template_path)
    parser.add_argument("--chunks-size", type=int, default=base.Config.chunks_size)
    parser.add_argument("--fps", type=int, default=base.Config.fps)
    parser.add_argument("--minimum-segment-length", type=int, default=64)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument(
        "--state-history-frames",
        type=int,
        default=1,
        help="Must remain 1 for the standard GR00T N1.7 SFT path; policy-context loss masking is not wired.",
    )
    parser.add_argument("--task-description", default=base.Config.task_description)
    parser.add_argument("--require-videos", action="store_true")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--binary-gripper-command-target",
        action="store_true",
        help=(
            "Keep arm targets as next achieved joint state, but supervise the gripper from "
            "the executed binary command."
        ),
    )
    parser.add_argument("--gripper-close-width", type=float, default=base.Config.gripper_close_width)
    parser.add_argument("--gripper-open-width", type=float, default=base.Config.gripper_open_width)
    args = parser.parse_args()
    if args.minimum_segment_length < args.action_horizon:
        raise ValueError("--minimum-segment-length must be >= --action-horizon")
    if args.state_history_frames != 1:
        raise ValueError(
            "--state-history-frames must be 1: the standard GR00T N1.7 loader does not consume "
            "annotation.human.action.valid as an anchor-level loss mask"
        )
    return Config(
        hdf5_file_path=args.hdf5_file_path,
        lerobot_data_dir=args.lerobot_data_dir,
        video_dir=args.video_dir,
        modality_template_path=args.modality_template_path,
        chunks_size=args.chunks_size,
        fps=args.fps,
        task_description=args.task_description,
        overwrite=args.overwrite,
        require_videos=args.require_videos,
        only_success=args.success_only,
        include_failed=not args.success_only,
        minimum_segment_length=args.minimum_segment_length,
        action_horizon=args.action_horizon,
        state_history_frames=args.state_history_frames,
        binary_gripper_command_target=args.binary_gripper_command_target,
        gripper_close_width=args.gripper_close_width,
        gripper_open_width=args.gripper_open_width,
    ).resolve()


def _dataset(trajectory: h5py.Group, key: str) -> np.ndarray:
    value: h5py.Group | h5py.Dataset = trajectory
    for part in key.split("/"):
        value = value[part]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"Expected HDF5 dataset at {key}")
    return value[()]


def episode_success(trajectory: h5py.Group) -> bool:
    """Read the online success label from HG-DAgger or legacy trajectory data."""
    if "dagger" in trajectory and "episode" in trajectory["dagger"] and "success" in trajectory["dagger/episode"]:
        value = np.asarray(_dataset(trajectory, "dagger/episode/success")).reshape(-1)
        if value.size != 1:
            raise ValueError(f"Expected one online success value, received shape {value.shape}")
        return bool(value[0])
    return bool(trajectory.attrs.get("success", False))


def clip_segment_to_available_frames(
    start: int,
    end: int,
    *,
    available_frames: int,
    minimum_length: int,
) -> tuple[int, int] | None:
    """Clip a recovery segment to frames with matching observations/videos."""
    clipped_end = min(end, available_frames)
    if clipped_end - start < minimum_length:
        return None
    return start, clipped_end


def segment_dataframe(
    trajectory: h5py.Group,
    start: int,
    end: int,
    *,
    episode_index: int,
    index_start: int,
    config: Config,
) -> pd.DataFrame:
    if config.state_history_frames != 1:
        raise ValueError(
            "Policy-context rows require an explicit GR00T anchor/loss-mask integration; "
            "the standard SFT path supports state_history_frames=1 only"
        )
    context_start = start
    data_slice = slice(context_start, end)
    state = _dataset(trajectory, "dagger/observation_joint_state")[data_slice].astype(np.float32)
    action = _dataset(trajectory, "dagger/achieved_joint_state")[data_slice].astype(np.float32).copy()
    policy_action = _dataset(trajectory, "dagger/policy_action")[data_slice].astype(np.float32)
    expert_action = _dataset(trajectory, "dagger/expert_action")[data_slice].astype(np.float32)
    executed_action = _dataset(trajectory, "dagger/executed_action")[data_slice].astype(np.float32)
    policy_valid = _dataset(trajectory, "dagger/policy_action_valid")[data_slice].astype(bool).reshape(-1)
    intervention = _dataset(trajectory, "dagger/intervention_mask")[data_slice].astype(bool).reshape(-1)

    if config.binary_gripper_command_target:
        action[:, -1] = base.binary_gripper_command_to_width(
            executed_action[:, -1],
            close_width=config.gripper_close_width,
            open_width=config.gripper_open_width,
        )

    length = end - context_start
    for name, value in {
        "state": state,
        "action": action,
        "policy_action": policy_action,
        "expert_action": expert_action,
        "executed_action": executed_action,
    }.items():
        if value.shape != (length, 8):
            raise ValueError(f"{name} has shape {value.shape}; expected {(length, 8)}")

    valid = valid_horizon_anchors(intervention, config.action_horizon)
    success = episode_success(trajectory)
    reward = np.zeros(length, dtype=np.float64)
    reward[-1] = float(success)
    done = np.zeros(length, dtype=bool)
    done[-1] = True

    return pd.DataFrame(
        {
            base.LEROBOT_KEY["state"]: list(state),
            base.LEROBOT_KEY["action"]: list(action),
            "action.policy": list(policy_action),
            "action.expert": list(expert_action),
            "action.executed_command": list(executed_action),
            "annotation.human.action.intervention": intervention,
            "annotation.policy.action.valid": policy_valid,
            base.LEROBOT_KEY["task"]: np.full(length, config.task_index, dtype=np.int64),
            base.LEROBOT_KEY["valid"]: valid,
            "timestamp": np.arange(length, dtype=np.float64) / config.fps,
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "task_index": np.full(length, config.task_index, dtype=np.int64),
            "index": np.arange(length, dtype=np.int64) + index_start,
            "next.reward": reward,
            "next.done": done,
        }
    )


def trim_episode_videos(
    config: Config,
    trajectory_id: str,
    episode_index: int,
    start: int,
    length: int,
) -> dict[str, Path]:
    if config.video_dir is None:
        return {}

    output: dict[str, Path] = {}
    episode_chunk = episode_index // config.chunks_size
    for video_key, suffixes in base.LEROBOT_KEY["video"].items():
        source = base.find_video(config.video_dir, trajectory_id, suffixes)
        if source is None:
            if config.require_videos:
                raise FileNotFoundError(f"No video for {trajectory_id}, key {video_key}")
            continue
        destination = config.lerobot_data_dir / config.video_path.format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-an",
                "-vf",
                f"trim=start_frame={start}:end_frame={start + length},setpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(destination),
            ],
            check=True,
        )
        output[video_key] = destination
    return output


def convert(config: Config) -> None:
    if config.lerobot_data_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {config.lerobot_data_dir}")
        import shutil

        shutil.rmtree(config.lerobot_data_dir)
    config.lerobot_data_dir.mkdir(parents=True)

    total_frames = 0
    total_videos = 0
    episodes: list[dict[str, Any]] = []
    example_data = None
    example_videos: dict[str, Path] = {}

    with h5py.File(config.hdf5_file_path, "r") as source:
        for trajectory_id in tqdm(base.sorted_demo_ids(source["data"])):
            trajectory = source["data"][trajectory_id]
            source_success = episode_success(trajectory)
            if config.only_success and not source_success:
                continue
            mask = _dataset(trajectory, "dagger/intervention_mask").astype(bool).reshape(-1)
            # Replay/render produces one observation video frame per transition target:
            # an episode with N recorded actions therefore has N - 1 aligned frames.
            available_frames = len(mask) if config.video_dir is None else max(len(mask) - 1, 0)
            for raw_start, raw_end in contiguous_true_segments(mask, config.minimum_segment_length):
                segment = clip_segment_to_available_frames(
                    raw_start,
                    raw_end,
                    available_frames=available_frames,
                    minimum_length=config.minimum_segment_length,
                )
                if segment is None:
                    continue
                start, end = segment
                episode_index = len(episodes)
                context_start = start
                dataframe = segment_dataframe(
                    trajectory,
                    start,
                    end,
                    episode_index=episode_index,
                    index_start=total_frames,
                    config=config,
                )
                episode_chunk = episode_index // config.chunks_size
                data_path = config.lerobot_data_dir / config.data_path.format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                )
                data_path.parent.mkdir(parents=True, exist_ok=True)
                dataframe.to_parquet(data_path)
                videos = trim_episode_videos(config, trajectory_id, episode_index, context_start, end - context_start)
                total_videos += len(videos)
                if videos and not example_videos:
                    example_videos = videos
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [config.task_description],
                        "length": len(dataframe),
                        "source_episode": trajectory_id,
                        "source_range": [context_start, end],
                        "recovery_range": [start, end],
                        "source_success": source_success,
                    }
                )
                total_frames += len(dataframe)
                if example_data is None:
                    example_data = dataframe

    if example_data is None:
        raise RuntimeError("No intervention segments met the configured minimum length.")

    meta_dir = config.lerobot_data_dir / "meta"
    base.dump_jsonl([{"task_index": 0, "task": config.task_description}], meta_dir / "tasks.jsonl")
    base.dump_jsonl(episodes, meta_dir / "episodes.jsonl")
    base.write_modality(config, use_videos=bool(total_videos))
    base.dump_json(
        base.generate_info(
            config,
            total_episodes=len(episodes),
            total_frames=total_frames,
            total_videos=total_videos,
            example_data=example_data,
            example_video_paths=example_videos,
        ),
        meta_dir / "info.json",
        indent=4,
    )
    print(f"Converted {len(episodes)} recovery segments / {total_frames} frames to {config.lerobot_data_dir}")


if __name__ == "__main__":
    convert(parse_args())
