#!/usr/bin/env python3
"""Render a concise Markdown summary from analyze_paired_closed_loop.py JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def summarize(report: dict[str, Any], metric_name: str) -> str:
    metrics = report.get("metrics", {})
    if metric_name not in metrics:
        raise ValueError(f"metric {metric_name!r} not found; available: {sorted(metrics)}")
    metric = metrics[metric_name]
    validation = report.get("validation", {})
    validation_ok = all(
        validation.get(key) is True
        for key in (
            "complete_runs",
            "matching_common_prefix_inference_seeds",
            "matching_initial_scene_signatures",
            "matching_seed_sets",
        )
    )
    control = int(metric["control_successes"])
    treatment = int(metric["treatment_successes"])
    pairs = int(metric["pairs"])
    lines = [
        f"# Paired DAgger summary — {metric_name}",
        "",
        f"Integrity checks: **{'PASS' if validation_ok else 'FAIL'}**",
        "",
        "| Arm | Successes | SR |",
        "| --- | ---: | ---: |",
        f"| Base | {control}/{pairs} | {percentage(float(metric['control_rate']))} |",
        f"| DAgger | {treatment}/{pairs} | {percentage(float(metric['treatment_rate']))} |",
        "",
        f"Net DAgger change: **{100.0 * float(metric['delta']):+.2f} pp**",
        "",
        "| Paired outcome | Count |",
        "| --- | ---: |",
        f"| Improved | {int(metric['treatment_only'])} |",
        f"| Regressed | {int(metric['baseline_only'])} |",
        f"| Unchanged | {int(metric['both_success']) + int(metric['both_failure'])} |",
        f"| Discordance | {percentage(float(metric['discordance_rate']))} |",
        "",
        f"Resolution: one paired rollout = {100.0 / pairs:.3f} pp.",
        f"Zero difference-variance scenes: {int(metric['zero_difference_variance_scenes'])}/{int(metric['scenes'])}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--metric", default="containment")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rendered = summarize(report, args.metric)
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
