# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay Franka pick-and-place HDF5 demos and render camera videos.

The generated MP4 names are compatible with
``convert_hdf5_to_lerobot_task_space.py --video-dir``:

    demo_0_wrist_camera.mp4
    demo_0_table_camera.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Only 1 is supported for deterministic video output.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0",
    help="Camera replay task. Use the Replay-Camera Franka IK-relative task.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="datasets/dataset_sorting_105.hdf5",
    help="HDF5 dataset file to replay.",
)
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="Episode indices to replay. Empty means all episodes.",
)
parser.add_argument(
    "--camera_view_list",
    type=str,
    nargs="+",
    default=["wrist_camera", "table_camera"],
    help="Camera sensors or aliases. Supported aliases: wrist, wrist_cam, wrist_camera, table, table_cam, table_camera.",
)
parser.add_argument("--video", action="store_true", default=False, help="Write MP4 videos during replay.")
parser.add_argument(
    "--video-output-dir",
    type=Path,
    default=Path("datasets/dataset_sorting_105/generated_videos"),
    help="Directory where replay videos are written.",
)
parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
parser.add_argument("--save_depth", action="store_true", default=False, help="Also save depth visualization videos.")
parser.add_argument("--depth_max", type=float, default=2.0, help="Depth value mapped to white in depth videos.")
parser.add_argument(
    "--validate_success_rate",
    action="store_true",
    default=False,
    help="Evaluate the task success term after replaying each episode.",
)
parser.add_argument(
    "--failure-output-file",
    type=Path,
    default=Path("datasets/dataset_sorting_105/generated_videos/failure.jsonl"),
    help="JSONL file where replay failure episode ids are written.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("This Franka video replay script currently supports --num_envs 1 only.")

if args_cli.video or len(args_cli.camera_view_list) > 0:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402

from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


CAMERA_ALIASES = {
    "wrist": ("wrist_camera", "wrist_camera"),
    "wrist_cam": ("wrist_camera", "wrist_cam"),
    "wrist_camera": ("wrist_camera", "wrist_camera"),
    "table": ("table_camera", "table_camera"),
    "table_cam": ("table_camera", "table_cam"),
    "table_camera": ("table_camera", "table_camera"),
}


class VideoWriterSet:
    """Lazy-open RGB/depth MP4 writers for one replayed episode."""

    def __init__(self, output_dir: Path, demo_id: int, fps: float):
        self.output_dir = output_dir
        self.demo_id = demo_id
        self.fps = fps
        self._writers: dict[tuple[str, str], cv2.VideoWriter] = {}

    def write_rgb(self, camera_suffix: str, frame_rgb: np.ndarray) -> None:
        frame_rgb = normalize_rgb(frame_rgb)
        writer = self._writer(camera_suffix, "rgb", frame_rgb.shape[1], frame_rgb.shape[0])
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def write_depth(self, camera_suffix: str, frame_depth: np.ndarray, depth_max: float) -> None:
        depth = np.asarray(frame_depth).squeeze()
        depth = np.nan_to_num(depth, nan=0.0, posinf=depth_max, neginf=0.0)
        depth_u8 = np.clip(depth / depth_max, 0.0, 1.0)
        depth_u8 = (depth_u8 * 255.0).astype(np.uint8)
        depth_rgb = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
        writer = self._writer(camera_suffix, "depth", depth_rgb.shape[1], depth_rgb.shape[0])
        writer.write(depth_rgb)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

    def _writer(self, camera_suffix: str, stream_name: str, width: int, height: int) -> cv2.VideoWriter:
        key = (camera_suffix, stream_name)
        if key in self._writers:
            return self._writers[key]

        suffix = camera_suffix if stream_name == "rgb" else f"{camera_suffix}_{stream_name}"
        path = self.output_dir / f"demo_{self.demo_id}_{suffix}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        self._writers[key] = writer
        return writer


def normalize_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert camera output to uint8 RGB with 3 channels."""
    frame = np.asarray(frame)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        max_value = 1.0 if np.nanmax(frame) <= 1.0 else 255.0
        frame = np.clip(frame, 0.0, max_value) / max_value * 255.0
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def resolve_camera_views(view_names: list[str]) -> list[tuple[str, str]]:
    cameras = []
    for view in view_names:
        if view in CAMERA_ALIASES:
            cameras.append(CAMERA_ALIASES[view])
        elif view.endswith("_cam"):
            cameras.append((view, view))
        else:
            cameras.append((f"{view}_cam", f"{view}_cam"))
    return cameras


def sorted_episode_names(dataset_file_handler: HDF5DatasetFileHandler) -> list[str]:
    def episode_index(name: str) -> int:
        return int(name.split("_")[-1])

    return sorted(dataset_file_handler.get_episode_names(), key=episode_index)


def write_failure_file(path: Path, task: str, failed_demo_ids: list[int]) -> None:
    if not failed_demo_ids:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({task: sorted(failed_demo_ids)}) + "\n")


def capture_cameras(env, writers: VideoWriterSet, camera_views: list[tuple[str, str]]) -> None:
    for sensor_name, camera_suffix in camera_views:
        if sensor_name not in env.scene.sensors:
            available = ", ".join(sorted(env.scene.sensors.keys()))
            raise KeyError(f"Camera sensor '{sensor_name}' was not found. Available sensors: {available}")

        sensor = env.scene.sensors[sensor_name]
        rgb = sensor.data.output["rgb"].detach().cpu().numpy()[0]
        writers.write_rgb(camera_suffix, rgb)

        if args_cli.save_depth:
            depth = sensor.data.output["distance_to_image_plane"].detach().cpu().numpy()[0]
            writers.write_depth(camera_suffix, depth, args_cli.depth_max)


def main() -> None:
    dataset_path = Path(args_cli.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"The dataset file does not exist: {dataset_path}")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(str(dataset_path))
    episode_names = sorted_episode_names(dataset_file_handler)
    episode_count = len(episode_names)
    if episode_count == 0:
        raise RuntimeError("No episodes found in the dataset.")

    episode_indices = args_cli.select_episodes or list(range(episode_count))
    camera_views = resolve_camera_views(args_cli.camera_view_list)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    success_term = None
    if args_cli.validate_success_rate and hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    env_cfg.recorders = {}
    env_cfg.terminations = {}

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    replayed_episode_count = 0
    successful_episode_count = 0
    failed_demo_ids: list[int] = []

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for episode_index in episode_indices:
            if episode_index >= episode_count:
                print(f"Skipping episode {episode_index}: dataset only has {episode_count} episodes.")
                continue

            episode_name = episode_names[episode_index]
            episode_data = dataset_file_handler.load_episode(episode_name, env.device)
            initial_state = episode_data.get_initial_state()
            env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
            for _ in range(getattr(env_cfg, "num_rerenders_on_reset", 0)):
                env.sim.render()

            action_count = len(episode_data.data["actions"])
            video_frame_count = max(action_count - 1, 0)
            writers = VideoWriterSet(args_cli.video_output_dir, episode_index, args_cli.fps)

            print(
                f"{replayed_episode_count + 1:4}: Replaying {episode_name} "
                f"({action_count} actions, {video_frame_count} video frames)"
            )

            try:
                for action_index in range(action_count):
                    action = episode_data.get_next_action()
                    if action is None:
                        break

                    if args_cli.video and action_index < video_frame_count:
                        capture_cameras(env, writers, camera_views)

                    actions = torch.zeros(env.action_space.shape, device=env.device)
                    actions[0] = action
                    env.step(actions)
            finally:
                writers.close()

            replayed_episode_count += 1
            if success_term is None:
                continue

            is_success = bool(success_term.func(env, **success_term.params)[0])
            if is_success:
                successful_episode_count += 1
                print(f"Successfully replayed {successful_episode_count} episodes out of {replayed_episode_count} demos.")
            else:
                failed_demo_ids.append(episode_index)
                print(f"Replay failed success check for episode {episode_index}.")

    write_failure_file(args_cli.failure_output_file, args_cli.task, failed_demo_ids)

    print(f"Finished replaying {replayed_episode_count} episodes.")
    if success_term is not None and replayed_episode_count > 0:
        print(f"Replay success rate: {successful_episode_count}/{replayed_episode_count}")
        if failed_demo_ids:
            print(f"Failed demo IDs: {sorted(failed_demo_ids)}")
            print(f"Failure record written to: {args_cli.failure_output_file}")
    if args_cli.video:
        print(f"Videos written to: {args_cli.video_output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
