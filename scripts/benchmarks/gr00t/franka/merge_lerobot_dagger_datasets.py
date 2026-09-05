# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge base and HG-DAgger LeRobot v2 datasets at a frame-level mix ratio."""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import cycle
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def dump_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def source_episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    return root / info["data_path"].format(
        episode_chunk=episode_index // info["chunks_size"],
        episode_index=episode_index,
    )


def validate_feature_compatibility(base_info: dict[str, Any], dagger_info: dict[str, Any]) -> None:
    """Reject mixed datasets that cannot satisfy the base training schema."""

    base_features = base_info["features"]
    dagger_features = dagger_info["features"]
    base_videos = {key for key, feature in base_features.items() if feature.get("dtype") == "video"}
    dagger_videos = {key for key, feature in dagger_features.items() if feature.get("dtype") == "video"}
    if base_videos != dagger_videos:
        raise ValueError(
            "Base and DAgger video features differ; every merged episode must provide "
            f"the same cameras (base={sorted(base_videos)}, dagger={sorted(dagger_videos)})"
        )

    for key, base_feature in base_features.items():
        if base_feature.get("dtype") == "video":
            dagger_feature = dagger_features[key]
            if dagger_feature.get("shape") != base_feature.get("shape"):
                raise ValueError(
                    f"Video feature {key!r} has incompatible shape: "
                    f"base={base_feature.get('shape')!r}, dagger={dagger_feature.get('shape')!r}"
                )
            continue
        dagger_feature = dagger_features.get(key)
        if dagger_feature is None:
            raise ValueError(f"DAgger dataset is missing base feature {key!r}")
        for field in ("dtype", "shape"):
            if dagger_feature.get(field) != base_feature.get(field):
                raise ValueError(
                    f"Feature {key!r} has incompatible {field}: "
                    f"base={base_feature.get(field)!r}, dagger={dagger_feature.get(field)!r}"
                )


def choose_dagger_episodes(
    episodes: list[dict[str, Any]],
    *,
    base_frames: int,
    target_fraction: float,
    allow_repeat: bool,
    all_unique: bool = False,
) -> list[dict[str, Any]]:
    if not episodes:
        return []
    if all_unique:
        return list(episodes)
    if target_fraction <= 0.0:
        return []
    target_frames = int(round(base_frames * target_fraction / (1.0 - target_fraction)))
    selected: list[dict[str, Any]] = []
    selected_frames = 0
    for row in episodes:
        selected.append(row)
        selected_frames += int(row["length"])
        if selected_frames >= target_frames:
            return selected
    if not allow_repeat:
        return selected

    for row in cycle(episodes):
        selected.append(row)
        selected_frames += int(row["length"])
        if selected_frames >= target_frames:
            break
    return selected


def copy_video_features(
    source_root: Path,
    source_info: dict[str, Any],
    source_episode: int,
    output_root: Path,
    output_info: dict[str, Any],
    output_episode: int,
) -> int:
    video_path = source_info.get("video_path")
    if not video_path:
        return 0
    count = 0
    for key, feature in source_info["features"].items():
        if feature.get("dtype") != "video":
            continue
        source = source_root / video_path.format(
            episode_chunk=source_episode // source_info["chunks_size"],
            video_key=key,
            episode_index=source_episode,
        )
        if not source.exists():
            raise FileNotFoundError(f"Missing source video: {source}")
        destination = output_root / output_info["video_path"].format(
            episode_chunk=output_episode // output_info["chunks_size"],
            video_key=key,
            episode_index=output_episode,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        count += 1
    return count


def merge(
    base_root: Path,
    dagger_root: Path,
    output_root: Path,
    *,
    target_fraction: float,
    allow_repeat: bool,
    overwrite: bool,
    all_unique_dagger: bool = False,
) -> None:
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be between 0 and 1")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)

    base_info = load_json(base_root / "meta/info.json")
    dagger_info = load_json(dagger_root / "meta/info.json")
    validate_feature_compatibility(base_info, dagger_info)
    base_tasks = load_jsonl(base_root / "meta/tasks.jsonl")
    dagger_tasks = load_jsonl(dagger_root / "meta/tasks.jsonl")
    if dagger_tasks != base_tasks:
        raise ValueError(
            "Base and DAgger task metadata differ; explicit task remapping is required "
            "before these datasets can be merged"
        )
    base_episodes = load_jsonl(base_root / "meta/episodes.jsonl")
    dagger_episodes = load_jsonl(dagger_root / "meta/episodes.jsonl")
    selected_dagger = choose_dagger_episodes(
        dagger_episodes,
        base_frames=sum(int(row["length"]) for row in base_episodes),
        target_fraction=target_fraction,
        allow_repeat=allow_repeat,
        all_unique=all_unique_dagger,
    )

    output_root.mkdir(parents=True)
    output_episodes: list[dict[str, Any]] = []
    total_frames = 0
    total_videos = 0
    base_columns = {key for key, feature in base_info["features"].items() if feature.get("dtype") != "video"}
    sources = [
        ("base", base_root, base_info, base_episodes),
        ("dagger", dagger_root, dagger_info, selected_dagger),
    ]

    for source_kind, source_root, source_info, episodes in sources:
        for source_row in episodes:
            source_episode = int(source_row["episode_index"])
            dataframe = pd.read_parquet(source_episode_path(source_root, source_info, source_episode))
            missing = base_columns - set(dataframe.columns)
            if missing:
                raise ValueError(f"{source_kind} episode {source_episode} misses columns: {sorted(missing)}")
            dataframe = dataframe[[column for column in dataframe.columns if column in base_columns]].copy()

            output_episode = len(output_episodes)
            dataframe["episode_index"] = output_episode
            dataframe["index"] = range(total_frames, total_frames + len(dataframe))
            output_path = output_root / base_info["data_path"].format(
                episode_chunk=output_episode // base_info["chunks_size"],
                episode_index=output_episode,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_parquet(output_path)
            total_videos += copy_video_features(
                source_root,
                source_info,
                source_episode,
                output_root,
                base_info,
                output_episode,
            )
            output_episodes.append(
                {
                    "episode_index": output_episode,
                    "tasks": source_row["tasks"],
                    "length": len(dataframe),
                    "source": source_kind,
                    "source_episode": source_episode,
                }
            )
            total_frames += len(dataframe)

    dagger_frames = sum(row["length"] for row in output_episodes if row["source"] == "dagger")
    output_info = dict(base_info)
    output_info.update(
        {
            "total_episodes": len(output_episodes),
            "total_frames": total_frames,
            "total_videos": total_videos,
            "total_chunks": (len(output_episodes) + base_info["chunks_size"] - 1) // base_info["chunks_size"],
            "splits": {"train": f"0:{len(output_episodes)}"},
        }
    )

    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_root / "meta/tasks.jsonl", output_root / "meta/tasks.jsonl")
    shutil.copy2(base_root / "meta/modality.json", output_root / "meta/modality.json")
    dump_jsonl(output_episodes, output_root / "meta/episodes.jsonl")
    dump_json(output_info, output_root / "meta/info.json")
    dump_json(
        {
            "requested_dagger_frame_fraction": target_fraction,
            "actual_dagger_frame_fraction": dagger_frames / max(total_frames, 1),
            "base_frames": total_frames - dagger_frames,
            "dagger_frames": dagger_frames,
            "dagger_episode_repetition_enabled": allow_repeat,
            "dagger_selection_mode": "all_unique" if all_unique_dagger else "target_fraction",
        },
        output_root / "meta/dagger_mix.json",
    )
    print(
        f"Merged {len(output_episodes)} episodes / {total_frames} frames; "
        f"DAgger fraction={dagger_frames / max(total_frames, 1):.2%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--dagger-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--target-dagger-fraction", type=float, default=0.2)
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument(
        "--all-unique-dagger",
        action="store_true",
        help="Include every unique DAgger episode once and ignore the target fraction.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.all_unique_dagger and args.allow_repeat:
        parser.error("--all-unique-dagger cannot be combined with --allow-repeat")
    merge(
        args.base_dataset.resolve(),
        args.dagger_dataset.resolve(),
        args.output_dataset.resolve(),
        target_fraction=args.target_dagger_fraction,
        allow_repeat=args.allow_repeat,
        overwrite=args.overwrite,
        all_unique_dagger=args.all_unique_dagger,
    )


if __name__ == "__main__":
    main()
