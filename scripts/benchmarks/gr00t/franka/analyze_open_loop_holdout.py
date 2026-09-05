# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one GR00T checkpoint on a source-disjoint recovery holdout."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--traj-ids", type=int, nargs="+", required=True)
    parser.add_argument("--primary-traj-ids", type=int, nargs="+", required=True)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--gripper-threshold", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    return parser.parse_args()


def inference_seed(seed: int, traj_id: int, step: int) -> int:
    payload = f"franka_gr00t_open_loop_holdout_v1:{seed}:{traj_id}:{step}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def extract_columns(trajectory: Any, columns: list[str]) -> np.ndarray:
    arrays = {column: np.vstack([value for value in trajectory[column]]) for column in columns}
    return np.concatenate([arrays[column] for column in columns], axis=-1)


def trajectory_metrics(
    state: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    traj_id: int,
    gripper_threshold: float,
) -> dict[str, Any]:
    del state
    error = prediction - ground_truth
    ground_truth_close = ground_truth[:, 7] < gripper_threshold
    prediction_close = prediction[:, 7] < gripper_threshold
    true_positive = int(np.count_nonzero(ground_truth_close & prediction_close))
    true_negative = int(np.count_nonzero(~ground_truth_close & ~prediction_close))
    false_positive = int(np.count_nonzero(~ground_truth_close & prediction_close))
    false_negative = int(np.count_nonzero(ground_truth_close & ~prediction_close))
    return {
        "traj_id": traj_id,
        "steps": int(len(error)),
        "all_absolute_error_sum": float(np.abs(error).sum()),
        "all_squared_error_sum": float((error**2).sum()),
        "all_element_count": int(error.size),
        "joint_absolute_error_sum": float(np.abs(error[:, :7]).sum()),
        "joint_squared_error_sum": float((error[:, :7] ** 2).sum()),
        "joint_element_count": int(error[:, :7].size),
        "gripper_absolute_error_sum": float(np.abs(error[:, 7]).sum()),
        "gripper_squared_error_sum": float((error[:, 7] ** 2).sum()),
        "gripper_element_count": int(len(error)),
        "all_mae": float(np.mean(np.abs(error))),
        "all_mse": float(np.mean(error**2)),
        "joint_mae": float(np.mean(np.abs(error[:, :7]))),
        "joint_mse": float(np.mean(error[:, :7] ** 2)),
        "gripper_mae": float(np.mean(np.abs(error[:, 7]))),
        "gripper_mse": float(np.mean(error[:, 7] ** 2)),
        "gripper_binary_agreement": float(np.mean(ground_truth_close == prediction_close)),
        "gripper_confusion": {
            "true_close": true_positive,
            "true_open": true_negative,
            "false_close": false_positive,
            "missed_close": false_negative,
        },
        "per_dim_mae": np.mean(np.abs(error), axis=0).tolist(),
        "per_dim_mse": np.mean(error**2, axis=0).tolist(),
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot aggregate an empty trajectory group")
    all_count = sum(int(item["all_element_count"]) for item in items)
    joint_count = sum(int(item["joint_element_count"]) for item in items)
    gripper_count = sum(int(item["gripper_element_count"]) for item in items)
    confusion = {
        key: sum(int(item["gripper_confusion"][key]) for item in items)
        for key in ("true_close", "true_open", "false_close", "missed_close")
    }
    return {
        "trajectories": len(items),
        "traj_ids": [int(item["traj_id"]) for item in items],
        "steps": sum(int(item["steps"]) for item in items),
        "weighted_all_mae": sum(float(item["all_absolute_error_sum"]) for item in items) / all_count,
        "weighted_all_mse": sum(float(item["all_squared_error_sum"]) for item in items) / all_count,
        "weighted_joint_mae": sum(float(item["joint_absolute_error_sum"]) for item in items) / joint_count,
        "weighted_joint_mse": sum(float(item["joint_squared_error_sum"]) for item in items) / joint_count,
        "weighted_gripper_mae": sum(float(item["gripper_absolute_error_sum"]) for item in items)
        / gripper_count,
        "weighted_gripper_mse": sum(float(item["gripper_squared_error_sum"]) for item in items)
        / gripper_count,
        "gripper_binary_agreement": (confusion["true_close"] + confusion["true_open"])
        / gripper_count,
        "gripper_confusion": confusion,
        "unweighted_trajectory_all_mae": float(np.mean([item["all_mae"] for item in items])),
        "unweighted_trajectory_joint_mae": float(np.mean([item["joint_mae"] for item in items])),
        "unweighted_trajectory_gripper_mae": float(np.mean([item["gripper_mae"] for item in items])),
    }


def save_plot(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    traj_id: int,
    action_horizon: int,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    dimensions = ground_truth.shape[1]
    figure, axes = plt.subplots(dimensions, 1, figsize=(11, 2.8 * dimensions), squeeze=False)
    figure.suptitle(f"Holdout trajectory {traj_id}: ground truth vs prediction")
    for dimension, axis in enumerate(axes[:, 0]):
        axis.plot(ground_truth[:, dimension], label="ground truth", linewidth=2.0)
        axis.plot(prediction[:, dimension], label="prediction", linewidth=1.25)
        for step in range(0, len(ground_truth), action_horizon):
            axis.axvline(step, color="red", alpha=0.2, linestyle="--")
        axis.set_title(f"Action {dimension}")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.action_horizon < 1:
        raise ValueError("--action-horizon must be positive")
    if not set(args.primary_traj_ids).issubset(args.traj_ids):
        raise ValueError("--primary-traj-ids must be a subset of --traj-ids")
    if len(set(args.traj_ids)) != len(args.traj_ids):
        raise ValueError("--traj-ids contains duplicates")

    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    embodiment_tag = EmbodimentTag.resolve("NEW_EMBODIMENT")
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    action_keys = ["franka_joint_pos", "franka_gripper_width"]
    state_keys = loader.modality_configs["state"].modality_keys
    input_modalities = deepcopy(loader.modality_configs)
    input_modalities.pop("action")
    results: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for traj_id in args.traj_ids:
            if traj_id < 0 or traj_id >= len(loader):
                raise ValueError(f"trajectory {traj_id} is outside dataset length {len(loader)}")
            trajectory = loader[traj_id]
            valid_steps = len(trajectory) - args.action_horizon + 1
            if valid_steps < 1:
                raise ValueError(f"trajectory {traj_id} is shorter than action horizon")
            predicted_rows: list[np.ndarray] = []
            for step in range(0, valid_steps, args.action_horizon):
                data_point = extract_step_data(trajectory, step, input_modalities, embodiment_tag)
                observation: dict[str, Any] = {}
                for key, value in data_point.states.items():
                    observation[f"state.{key}"] = value
                for key, value in data_point.images.items():
                    observation[f"video.{key}"] = np.array(value)
                for language_key in loader.modality_configs["language"].modality_keys:
                    observation[language_key] = data_point.text
                parsed_observation = parse_observation_gr00t(observation, loader.modality_configs)
                action, _ = policy.get_action(
                    parsed_observation,
                    options={"inference_seed": inference_seed(args.seed, traj_id, step)},
                )
                action_chunk = parse_action_gr00t(action)
                for offset in range(args.action_horizon):
                    predicted_rows.append(
                        np.concatenate(
                            [
                                np.atleast_1d(action_chunk[f"action.{key}"][offset])
                                for key in action_keys
                            ],
                            axis=0,
                        )
                    )
            prediction = np.asarray(predicted_rows)[:valid_steps]
            ground_truth = extract_columns(
                trajectory, [f"action.{key}" for key in action_keys]
            )[:valid_steps]
            state = extract_columns(trajectory, [f"state.{key}" for key in state_keys])[:valid_steps]
            if ground_truth.shape != prediction.shape:
                raise ValueError(
                    f"trajectory {traj_id}: ground-truth shape {ground_truth.shape} != prediction {prediction.shape}"
                )
            metrics = trajectory_metrics(
                state, ground_truth, prediction, traj_id, args.gripper_threshold
            )
            results.append(metrics)
            np.savez_compressed(
                args.output_dir / f"traj_{traj_id}.npz",
                state=state,
                ground_truth=ground_truth,
                prediction=prediction,
            )
            save_plot(
                ground_truth,
                prediction,
                traj_id,
                args.action_horizon,
                args.output_dir / f"traj_{traj_id}.jpeg",
            )
            print(
                f"traj={traj_id} steps={valid_steps} joint_mae={metrics['joint_mae']:.6f} "
                f"gripper_mae={metrics['gripper_mae']:.6f} "
                f"gripper_agreement={metrics['gripper_binary_agreement']:.2%}",
                flush=True,
            )

    primary_ids = set(args.primary_traj_ids)
    primary = [item for item in results if item["traj_id"] in primary_ids]
    report = {
        "schema_version": 1,
        "checkpoint_id": args.checkpoint_id,
        "model_path": str(Path(args.model_path).resolve()),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "action_horizon": args.action_horizon,
        "gripper_threshold": args.gripper_threshold,
        "seed": args.seed,
        "primary": aggregate(primary),
        "all_failed_source_sensitivity": aggregate(results),
        "per_trajectory": results,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"primary": report["primary"], "all": report["all_failed_source_sensitivity"]}, indent=2))


if __name__ == "__main__":
    main()
