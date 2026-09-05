# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Annotate replay videos with HG-DAgger human-intervention timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse annotation inputs and outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", type=Path, required=True, help="HG-DAgger HDF5 dataset.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing replay MP4 files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotated MP4 files. Defaults to <video-dir>/annotated.",
    )
    parser.add_argument(
        "--select-episodes",
        type=int,
        nargs="+",
        default=[],
        help="Episode indices to annotate. Empty annotates every replayed episode.",
    )
    return parser.parse_args()


def intervention_intervals(mask: np.ndarray) -> list[dict[str, int]]:
    """Return contiguous human-intervention intervals as frame ranges."""
    intervals = []
    start = None
    for frame_index, active in enumerate(mask):
        if active and start is None:
            start = frame_index
        elif not active and start is not None:
            intervals.append({"start_frame": start, "end_frame_exclusive": frame_index})
            start = None
    if start is not None:
        intervals.append({"start_frame": start, "end_frame_exclusive": len(mask)})
    return intervals


def draw_annotation(frame: np.ndarray, human_takeover: bool) -> np.ndarray:
    """Draw the policy or human-control state on a BGR video frame."""
    color = (0, 0, 255) if human_takeover else (0, 160, 0)
    label = "HUMAN TAKEOVER" if human_takeover else "POLICY"
    thickness = max(3, min(frame.shape[0], frame.shape[1]) // 160)
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, thickness)
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def annotate_video(video_path: Path, output_path: Path, mask: np.ndarray) -> int:
    """Write an annotated copy of one replay video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != len(mask):
        raise ValueError(
            f"Frame count mismatch for {video_path}: video has {frame_count}, intervention mask has {len(mask)}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    try:
        for frame_index, human_takeover in enumerate(mask):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
            writer.write(draw_annotation(frame, bool(human_takeover)))
    finally:
        capture.release()
        writer.release()
    return frame_count


def main() -> None:
    """Annotate all selected replay episodes and write a timeline manifest."""
    args = parse_args()
    if not args.dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {args.dataset_file}")
    if not args.video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {args.video_dir}")

    output_dir = args.output_dir or args.video_dir / "annotated"
    with h5py.File(args.dataset_file, "r") as dataset:
        data_group = dataset["data"]
        available_episodes = sorted(int(name.removeprefix("demo_")) for name in data_group if name.startswith("demo_"))
        episode_indices = args.select_episodes or available_episodes
        manifest: dict[str, dict[str, int | list[dict[str, int]]]] = {}

        for episode_index in episode_indices:
            episode_name = f"demo_{episode_index}"
            if episode_name not in data_group:
                raise KeyError(f"Episode {episode_name} is not present in {args.dataset_file}")

            action_mask = np.asarray(data_group[episode_name]["dagger"]["intervention_mask"]).reshape(-1).astype(bool)
            video_mask = action_mask[:-1]
            video_paths = sorted(args.video_dir.glob(f"demo_{episode_index}_*.mp4"))
            if not video_paths:
                raise FileNotFoundError(f"No replay videos found for {episode_name} in {args.video_dir}")

            intervention_frame_count = int(video_mask.sum())
            annotation_prefix = (
                f"human_takeover_{intervention_frame_count:05d}_frames_" if intervention_frame_count else "policy_only_"
            )
            for video_path in video_paths:
                frame_count = annotate_video(video_path, output_dir / f"{annotation_prefix}{video_path.name}", video_mask)
                print(
                    f"Annotated {video_path.name}: {frame_count} frames, "
                    f"{intervention_frame_count} human-takeover frames."
                )

            manifest[episode_name] = {
                "total_video_frames": len(video_mask),
                "human_takeover_frames": intervention_frame_count,
                "human_takeover_intervals": intervention_intervals(video_mask),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "intervention_timeline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote annotated videos to: {output_dir}")
    print(f"Wrote intervention timeline to: {manifest_path}")


if __name__ == "__main__":
    main()
