# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that two same-seed closed-loop evaluation runs are exactly reproducible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    return parser.parse_args()


def episode(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    episodes = [record for record in records if record.get("record_type") == "episode"]
    summaries = [record for record in records if record.get("record_type") == "run_summary"]
    if len(episodes) != 1 or len(summaries) != 1:
        raise ValueError(f"{path}: expected exactly one episode and one summary")
    if summaries[0].get("interrupted") or summaries[0].get("completed_experiments") != 1:
        raise ValueError(f"{path}: smoke run is incomplete")
    return episodes[0]


def trace_projection(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "inference_index": item["inference_index"],
            "inference_seed": item["inference_seed"],
            "observation_sha256": item["observation_sha256"],
            "raw_action_sha256": item["raw_action_sha256"],
            "scene": item.get("scene"),
        }
        for item in record["inference_trace"]
    ]


def main() -> None:
    args = parse_args()
    first = episode(args.first)
    second = episode(args.second)
    checks = {
        "seed": first["seed"] == second["seed"],
        "initial_scene_signature": first["initial_scene_signature"]
        == second["initial_scene_signature"],
        "strict_success": first["strict_success"] == second["strict_success"],
        "containment_success": first["containment_success"] == second["containment_success"],
        "termination_reason": first["termination_reason"] == second["termination_reason"],
        "env_steps": first["env_steps"] == second["env_steps"],
        "inference_calls": first["inference_calls"] == second["inference_calls"],
        "inference_trace": trace_projection(first) == trace_projection(second),
        "final_scene": first["final_scene"] == second["final_scene"],
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"determinism smoke failed: {failed}")


if __name__ == "__main__":
    main()
