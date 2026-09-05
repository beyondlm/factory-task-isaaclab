#!/usr/bin/env python3
"""Read-only preflight checks for the checked-in VLA DAgger workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_REPO_FILES = (
    "docs/vla_dagger_guide.md",
    "docs/vla_dagger_reference.md",
    "scripts/benchmarks/gr00t/dagger/task_spec.py",
    "scripts/benchmarks/gr00t/dagger/data_contract.py",
    "scripts/benchmarks/gr00t/franka/hg_dagger_core.py",
    "scripts/benchmarks/gr00t/franka/hg_dagger_recorder.py",
    "scripts/benchmarks/gr00t/franka/validate_hg_dagger_dataset.py",
    "scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_dagger_joint_space.py",
    "scripts/benchmarks/gr00t/franka/merge_lerobot_dagger_datasets.py",
    "scripts/benchmarks/gr00t/franka/analyze_paired_closed_loop.py",
    "patches/isaac-gr00t-n1.7-crn-inference-seed.patch",
)

REQUIRED_SPEC_FIELDS = (
    "name",
    "isaaclab_task",
    "policy_type",
    "state_dim",
    "action_dim",
    "action_horizon",
    "minimum_intervention_steps",
    "observation_keys",
    "action_keys",
    "camera_names",
    "language_instruction",
    "embodiment_tag",
    "success_metric_version",
)


def validate_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_SPEC_FIELDS if field not in spec]
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
        return errors
    state_dim = spec["state_dim"]
    action_dim = spec["action_dim"]
    horizon = spec["action_horizon"]
    minimum = spec["minimum_intervention_steps"]
    if not isinstance(state_dim, int) or state_dim < 1:
        errors.append("state_dim must be a positive integer")
    if not isinstance(action_dim, int) or action_dim < 1:
        errors.append("action_dim must be a positive integer")
    if not isinstance(horizon, int) or horizon < 1:
        errors.append("action_horizon must be a positive integer")
    if not isinstance(minimum, int) or not isinstance(horizon, int) or minimum < horizon:
        errors.append("minimum_intervention_steps must be at least action_horizon")
    for field in ("observation_keys", "action_keys", "camera_names"):
        value = spec[field]
        if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
            errors.append(f"{field} must be a non-empty list of strings")
        elif len(value) != len(set(value)):
            errors.append(f"{field} contains duplicates")
    gripper = spec.get("gripper")
    if gripper is not None:
        index = gripper.get("action_index") if isinstance(gripper, dict) else None
        if not isinstance(index, int) or not isinstance(action_dim, int) or not 0 <= index < action_dim:
            errors.append("gripper.action_index is outside action_dim")
        close_target = gripper.get("close_target") if isinstance(gripper, dict) else None
        open_target = gripper.get("open_target") if isinstance(gripper, dict) else None
        if not isinstance(close_target, (int, float)) or not isinstance(open_target, (int, float)):
            errors.append("gripper close/open targets must be numeric")
        elif close_target >= open_target:
            errors.append("gripper.close_target must be less than gripper.open_target")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    result: dict[str, Any] = {
        "repo": str(repo),
        "repo_files": {},
        "task_spec": None,
        "ok": True,
    }
    for relative in REQUIRED_REPO_FILES:
        exists = (repo / relative).is_file()
        result["repo_files"][relative] = exists
        result["ok"] = result["ok"] and exists

    if args.task_spec is not None:
        spec_path = args.task_spec.resolve()
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        errors = validate_spec(spec)
        result["task_spec"] = {"path": str(spec_path), "errors": errors}
        result["ok"] = result["ok"] and not errors

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
