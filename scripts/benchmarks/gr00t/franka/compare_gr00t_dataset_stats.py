# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare GR00T dataset statistics before finetuning.

The report is intentionally warning-first: DAgger recovery data should change
the data distribution. It blocks only structural problems that make
normalization unsafe, such as missing features, incompatible shapes, non-finite
values, or near-constant numeric features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_STATISTICS = ("mean", "std", "min", "max", "q01", "q99")


def load_statistics(dataset: Path) -> dict[str, dict[str, list[float]]]:
    """Load a dataset's GR00T normalization statistics."""
    stats_path = dataset / "meta" / "stats.json"
    with stats_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def compare_statistics(
    baseline: dict[str, dict[str, list[float]]],
    candidate: dict[str, dict[str, list[float]]],
    *,
    mean_z_warning: float,
    std_ratio_warning: float,
    range_ratio_warning: float,
) -> dict[str, Any]:
    """Return structural errors, distribution warnings, and per-feature metrics."""
    errors: list[str] = []
    warnings: list[str] = []
    features: dict[str, dict[str, float]] = {}

    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    for key in sorted(baseline_keys - candidate_keys):
        errors.append(f"Candidate is missing baseline feature: {key}")
    for key in sorted(candidate_keys - baseline_keys):
        warnings.append(f"Candidate has extra feature not present in baseline: {key}")

    for key in sorted(baseline_keys & candidate_keys):
        base_feature = baseline[key]
        candidate_feature = candidate[key]
        missing_statistics = [
            name
            for name in REQUIRED_STATISTICS
            if name not in base_feature or name not in candidate_feature
        ]
        if missing_statistics:
            errors.append(f"{key}: missing statistics {missing_statistics}")
            continue

        base = {name: np.asarray(base_feature[name], dtype=np.float64) for name in REQUIRED_STATISTICS}
        mixed = {name: np.asarray(candidate_feature[name], dtype=np.float64) for name in REQUIRED_STATISTICS}
        shapes = {value.shape for value in [*base.values(), *mixed.values()]}
        if len(shapes) != 1:
            errors.append(f"{key}: baseline and candidate statistic shapes differ")
            continue
        if not all(np.isfinite(value).all() for value in [*base.values(), *mixed.values()]):
            errors.append(f"{key}: statistics contain NaN or Inf")
            continue
        if np.any(base["std"] <= 1e-8) or np.any(mixed["std"] <= 1e-8):
            errors.append(f"{key}: contains a near-zero standard deviation")
            continue
        if np.any(base["min"] > base["max"]) or np.any(mixed["min"] > mixed["max"]):
            errors.append(f"{key}: has min values larger than max values")
            continue

        mean_z = np.abs(mixed["mean"] - base["mean"]) / base["std"]
        std_ratio = mixed["std"] / base["std"]
        base_range = np.maximum(base["max"] - base["min"], 1e-8)
        range_ratio = (mixed["max"] - mixed["min"]) / base_range
        feature_metrics = {
            "max_mean_shift_std": float(mean_z.max()),
            "min_std_ratio": float(std_ratio.min()),
            "max_std_ratio": float(std_ratio.max()),
            "max_range_ratio": float(range_ratio.max()),
        }
        features[key] = feature_metrics

        if feature_metrics["max_mean_shift_std"] > mean_z_warning:
            warnings.append(
                f"{key}: maximum mean shift is {feature_metrics['max_mean_shift_std']:.2f} baseline standard deviations"
            )
        if (
            feature_metrics["min_std_ratio"] < 1.0 / std_ratio_warning
            or feature_metrics["max_std_ratio"] > std_ratio_warning
        ):
            warnings.append(
                f"{key}: standard-deviation ratio is "
                f"[{feature_metrics['min_std_ratio']:.2f}, {feature_metrics['max_std_ratio']:.2f}]"
            )
        if feature_metrics["max_range_ratio"] > range_ratio_warning:
            warnings.append(
                f"{key}: value range expands by {feature_metrics['max_range_ratio']:.2f}x over baseline"
            )

    return {"errors": errors, "warnings": warnings, "features": features}


def main() -> None:
    """Compare baseline and candidate datasets and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dataset", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--mean-z-warning", type=float, default=2.0)
    parser.add_argument("--std-ratio-warning", type=float, default=2.0)
    parser.add_argument("--range-ratio-warning", type=float, default=2.0)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    report = compare_statistics(
        load_statistics(args.baseline_dataset),
        load_statistics(args.candidate_dataset),
        mean_z_warning=args.mean_z_warning,
        std_ratio_warning=args.std_ratio_warning,
        range_ratio_warning=args.range_ratio_warning,
    )
    report.update(
        {
            "baseline_dataset": str(args.baseline_dataset.resolve()),
            "candidate_dataset": str(args.candidate_dataset.resolve()),
        }
    )
    report_path = args.report_file or args.candidate_dataset / "meta" / "baseline_comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for key, metrics in report["features"].items():
        print(
            f"{key}: mean_shift={metrics['max_mean_shift_std']:.2f} std, "
            f"std_ratio=[{metrics['min_std_ratio']:.2f}, {metrics['max_std_ratio']:.2f}], "
            f"range_ratio={metrics['max_range_ratio']:.2f}"
        )
    for message in report["warnings"]:
        print(f"WARNING: {message}")
    for message in report["errors"]:
        print(f"ERROR: {message}")
    print(f"Wrote comparison report to: {report_path}")

    if report["errors"] or (args.fail_on_warning and report["warnings"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
