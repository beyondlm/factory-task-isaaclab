# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare two source-disjoint recovery holdout reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def metric_delta(control: dict[str, Any], treatment: dict[str, Any], key: str) -> dict[str, float]:
    control_value = float(control[key])
    treatment_value = float(treatment[key])
    relative = None if control_value == 0.0 else (treatment_value - control_value) / control_value
    return {
        "control": control_value,
        "treatment": treatment_value,
        "absolute_delta": treatment_value - control_value,
        "relative_delta": relative,
    }


def group_comparison(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    if control["traj_ids"] != treatment["traj_ids"]:
        raise ValueError("control and treatment trajectory lists differ")
    keys = (
        "weighted_all_mae",
        "weighted_joint_mae",
        "weighted_gripper_mae",
        "gripper_binary_agreement",
    )
    return {
        "trajectories": control["trajectories"],
        "traj_ids": control["traj_ids"],
        "steps": control["steps"],
        "metrics": {key: metric_delta(control, treatment, key) for key in keys},
        "control_gripper_confusion": control["gripper_confusion"],
        "treatment_gripper_confusion": treatment["gripper_confusion"],
    }


def main() -> None:
    args = parse_args()
    control = json.loads(args.control.read_text())
    treatment = json.loads(args.treatment.read_text())
    if control["dataset_path"] != treatment["dataset_path"]:
        raise ValueError("control and treatment dataset paths differ")
    if control["seed"] != treatment["seed"]:
        raise ValueError("control and treatment inference seeds differ")
    if control["action_horizon"] != treatment["action_horizon"]:
        raise ValueError("control and treatment action horizons differ")
    report = {
        "schema_version": 1,
        "control_checkpoint_id": control["checkpoint_id"],
        "treatment_checkpoint_id": treatment["checkpoint_id"],
        "dataset_path": control["dataset_path"],
        "primary": group_comparison(control["primary"], treatment["primary"]),
        "all_failed_source_sensitivity": group_comparison(
            control["all_failed_source_sensitivity"], treatment["all_failed_source_sensitivity"]
        ),
    }
    control_per_traj = {int(item["traj_id"]): item for item in control["per_trajectory"]}
    treatment_per_traj = {int(item["traj_id"]): item for item in treatment["per_trajectory"]}
    if control_per_traj.keys() != treatment_per_traj.keys():
        raise ValueError("control and treatment per-trajectory sets differ")
    report["per_trajectory"] = [
        {
            "traj_id": traj_id,
            "joint_mae": metric_delta(control_per_traj[traj_id], treatment_per_traj[traj_id], "joint_mae"),
            "gripper_mae": metric_delta(
                control_per_traj[traj_id], treatment_per_traj[traj_id], "gripper_mae"
            ),
            "gripper_binary_agreement": metric_delta(
                control_per_traj[traj_id], treatment_per_traj[traj_id], "gripper_binary_agreement"
            ),
        }
        for traj_id in sorted(control_per_traj)
    ]

    lines = [
        "# Source-disjoint recovery holdout",
        "",
        f"- Control: `{report['control_checkpoint_id']}`",
        f"- Treatment: `{report['treatment_checkpoint_id']}`",
        "- Primary subset: six predeclared high-confidence local recoveries.",
        "- Sensitivity subset: all 20 recoveries from nine failed source episodes.",
        "- Lower MAE is better; higher gripper binary agreement is better.",
        "- This is an offline imitation diagnostic, not a closed-loop SR result.",
        "",
        "| Group | Metric | Control | Treatment | Absolute delta | Relative delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group_name in ("primary", "all_failed_source_sensitivity"):
        group = report[group_name]
        for key, item in group["metrics"].items():
            relative = "n/a" if item["relative_delta"] is None else f"{100 * item['relative_delta']:+.2f}%"
            lines.append(
                f"| {group_name} | {key} | {item['control']:.8f} | {item['treatment']:.8f} | "
                f"{item['absolute_delta']:+.8f} | {relative} |"
            )
    lines.extend(
        [
            "",
            "## Per trajectory",
            "",
            "| Traj | Joint MAE control→treatment | Gripper MAE control→treatment | Agreement control→treatment |",
            "|---:|---:|---:|---:|",
        ]
    )
    for item in report["per_trajectory"]:
        lines.append(
            f"| {item['traj_id']} | {item['joint_mae']['control']:.6f}→{item['joint_mae']['treatment']:.6f} | "
            f"{item['gripper_mae']['control']:.6f}→{item['gripper_mae']['treatment']:.6f} | "
            f"{100 * item['gripper_binary_agreement']['control']:.2f}%→"
            f"{100 * item['gripper_binary_agreement']['treatment']:.2f}% |"
        )
    lines.append("")
    markdown = "\n".join(lines)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
