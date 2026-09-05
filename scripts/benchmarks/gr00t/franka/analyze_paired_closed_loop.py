# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze repeated, common-random-number closed-loop policy evaluations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    starts = [record for record in records if record.get("record_type") == "run_start"]
    summaries = [record for record in records if record.get("record_type") == "run_summary"]
    episodes = [record for record in records if record.get("record_type") == "episode"]
    if len(starts) != 1 or len(summaries) != 1:
        raise ValueError(f"{path}: expected one run_start and one run_summary")
    start = starts[0]
    summary = summaries[0]
    if summary.get("interrupted") or summary.get("completed_experiments") != start.get("requested_experiments"):
        raise ValueError(f"{path}: run is incomplete or interrupted")
    by_seed = {int(record["seed"]): record for record in episodes}
    if len(by_seed) != len(episodes):
        raise ValueError(f"{path}: duplicate episode seed")
    if len(by_seed) != int(summary["completed_experiments"]):
        raise ValueError(f"{path}: episode count disagrees with summary")
    return start, by_seed, summary


def load_arm(directory: Path) -> dict[int, dict[str, Any]]:
    paths = sorted(directory.glob("repeat_*.jsonl"))
    if not paths:
        raise ValueError(f"{directory}: no repeat_*.jsonl files")
    repeats: dict[int, dict[str, Any]] = {}
    for path in paths:
        start, episodes, summary = load_jsonl(path)
        protocol = start.get("policy_noise_protocol") or {}
        repeat_index = protocol.get("repeat_index")
        if repeat_index is None:
            raise ValueError(f"{path}: missing policy-noise repeat index")
        repeat_index = int(repeat_index)
        if repeat_index in repeats:
            raise ValueError(f"{directory}: duplicate repeat {repeat_index}")
        repeats[repeat_index] = {
            "path": str(path.resolve()),
            "start": start,
            "episodes": episodes,
            "summary": summary,
        }
    expected = list(range(len(repeats)))
    if sorted(repeats) != expected:
        raise ValueError(f"{directory}: repeats must be contiguous from zero; got {sorted(repeats)}")
    return repeats


def layout_name(episode: dict[str, Any]) -> str:
    objects = episode["initial_scene"]["objects"]
    a_y = float(objects["object_a"]["position_env"][1])
    b_y = float(objects["object_b"]["position_env"][1])
    c_y = float(objects["object_c"]["position_env"][1])
    d_y = float(objects["object_d"]["position_env"][1])
    return "same_side" if (a_y - b_y) * (c_y - d_y) > 0.0 else "cross_side"


def validate_pair(
    control: dict[int, dict[str, Any]], treatment: dict[int, dict[str, Any]]
) -> tuple[list[int], list[int]]:
    if sorted(control) != sorted(treatment):
        raise ValueError("control and treatment repeat indices differ")
    repeat_indices = sorted(control)
    reference_seeds: list[int] | None = None
    reference_signatures: dict[int, str] | None = None
    for repeat_index in repeat_indices:
        control_run = control[repeat_index]
        treatment_run = treatment[repeat_index]
        control_seeds = sorted(control_run["episodes"])
        treatment_seeds = sorted(treatment_run["episodes"])
        if control_seeds != treatment_seeds:
            raise ValueError(f"repeat {repeat_index}: arm seed sets differ")
        signatures = {
            seed: control_run["episodes"][seed]["initial_scene_signature"] for seed in control_seeds
        }
        for seed in control_seeds:
            control_episode = control_run["episodes"][seed]
            treatment_episode = treatment_run["episodes"][seed]
            if treatment_episode["initial_scene_signature"] != signatures[seed]:
                raise ValueError(f"repeat {repeat_index}, seed {seed}: scene signatures differ between arms")
            control_trace = control_episode.get("inference_trace", [])
            treatment_trace = treatment_episode.get("inference_trace", [])
            common_length = min(len(control_trace), len(treatment_trace))
            for inference_index in range(common_length):
                control_seed = control_trace[inference_index].get("inference_seed")
                treatment_seed = treatment_trace[inference_index].get("inference_seed")
                if control_seed != treatment_seed:
                    raise ValueError(
                        f"repeat {repeat_index}, seed {seed}, inference {inference_index}: CRN seeds differ"
                    )
        if reference_seeds is None:
            reference_seeds = control_seeds
            reference_signatures = signatures
        else:
            if control_seeds != reference_seeds:
                raise ValueError(f"repeat {repeat_index}: seed set differs from repeat zero")
            if signatures != reference_signatures:
                raise ValueError(f"repeat {repeat_index}: scene signatures differ from repeat zero")
    assert reference_seeds is not None
    return repeat_indices, reference_seeds


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def exact_sign_test_p(baseline_only: int, treatment_only: int) -> float:
    discordant = baseline_only + treatment_only
    if discordant == 0:
        return 1.0
    lower = min(baseline_only, treatment_only)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def episode_success(episode: dict[str, Any], metric: str) -> bool:
    key = "containment_success" if metric == "containment" else "strict_success"
    return bool(episode[key])


def failure_bucket(episode: dict[str, Any], metric: str) -> str:
    """Return an unambiguous bucket for a failed metric."""
    bucket = str(episode.get("failure_bucket", "unknown"))
    if metric == "containment" and bucket == "success":
        return "strict_success_but_final_containment_failure"
    return bucket


def analyze_metric(
    control: dict[int, dict[str, Any]],
    treatment: dict[int, dict[str, Any]],
    repeat_indices: list[int],
    seeds: list[int],
    metric: str,
) -> dict[str, Any]:
    repeats = len(repeat_indices)
    scenes = len(seeds)
    per_scene: list[dict[str, Any]] = []
    baseline_only = 0
    treatment_only = 0
    both_success = 0
    both_failure = 0
    control_failures: Counter[str] = Counter()
    treatment_failures: Counter[str] = Counter()
    layout_totals: dict[str, Counter[str]] = {
        "cross_side": Counter(),
        "same_side": Counter(),
    }
    inference_calls: list[int] = []

    for seed in seeds:
        control_values: list[int] = []
        treatment_values: list[int] = []
        differences: list[int] = []
        seed_b = 0
        seed_c = 0
        layout: str | None = None
        for repeat_index in repeat_indices:
            control_episode = control[repeat_index]["episodes"][seed]
            treatment_episode = treatment[repeat_index]["episodes"][seed]
            current_layout = layout_name(control_episode)
            if layout is not None and current_layout != layout:
                raise ValueError(f"seed {seed}: layout differs between repeats")
            layout = current_layout
            control_success = int(episode_success(control_episode, metric))
            treatment_success = int(episode_success(treatment_episode, metric))
            control_values.append(control_success)
            treatment_values.append(treatment_success)
            difference = treatment_success - control_success
            differences.append(difference)
            inference_calls.extend(
                [int(control_episode["inference_calls"]), int(treatment_episode["inference_calls"])]
            )
            if control_success and treatment_success:
                both_success += 1
                layout_totals[layout]["both_success"] += 1
            elif not control_success and not treatment_success:
                both_failure += 1
                layout_totals[layout]["both_failure"] += 1
            elif control_success:
                baseline_only += 1
                seed_b += 1
                layout_totals[layout]["baseline_only"] += 1
            else:
                treatment_only += 1
                seed_c += 1
                layout_totals[layout]["treatment_only"] += 1
            if not control_success:
                control_failures[failure_bucket(control_episode, metric)] += 1
            if not treatment_success:
                treatment_failures[failure_bucket(treatment_episode, metric)] += 1
        assert layout is not None
        control_k = sum(control_values)
        treatment_k = sum(treatment_values)
        control_u = (
            0.0
            if repeats < 2
            else 2.0 * control_k * (repeats - control_k) / (repeats * (repeats - 1))
        )
        treatment_u = (
            0.0
            if repeats < 2
            else 2.0 * treatment_k * (repeats - treatment_k) / (repeats * (repeats - 1))
        )
        per_scene.append(
            {
                "seed": seed,
                "layout": layout,
                "control_successes": control_k,
                "treatment_successes": treatment_k,
                "control_rate": control_k / repeats,
                "treatment_rate": treatment_k / repeats,
                "difference": sum(differences) / repeats,
                "differences_by_repeat": differences,
                "baseline_only_pairs": seed_b,
                "treatment_only_pairs": seed_c,
                "control_self_disagreement_u": control_u,
                "treatment_self_disagreement_u": treatment_u,
                "difference_sample_variance": sample_variance([float(value) for value in differences]),
            }
        )

    total_pairs = scenes * repeats
    control_successes = sum(item["control_successes"] for item in per_scene)
    treatment_successes = sum(item["treatment_successes"] for item in per_scene)
    differences = [float(item["difference"]) for item in per_scene]
    estimate = sum(differences) / scenes
    variance = sum(float(item["difference_sample_variance"]) / repeats for item in per_scene) / scenes**2
    standard_error = math.sqrt(variance)
    half_width = 1.959963984540054 * standard_error
    zero_variance_scenes = sum(item["difference_sample_variance"] == 0.0 for item in per_scene)
    for layout, counts in layout_totals.items():
        counts["pairs"] = sum(counts[key] for key in ("both_success", "both_failure", "baseline_only", "treatment_only"))
        counts["control_successes"] = counts["both_success"] + counts["baseline_only"]
        counts["treatment_successes"] = counts["both_success"] + counts["treatment_only"]

    return {
        "metric": metric,
        "scenes": scenes,
        "repeats": repeats,
        "pairs": total_pairs,
        "control_successes": control_successes,
        "treatment_successes": treatment_successes,
        "control_rate": control_successes / total_pairs,
        "treatment_rate": treatment_successes / total_pairs,
        "delta": estimate,
        "both_success": both_success,
        "both_failure": both_failure,
        "baseline_only": baseline_only,
        "treatment_only": treatment_only,
        "discordance_rate": (baseline_only + treatment_only) / total_pairs,
        "exact_sign_test_p_sensitivity": exact_sign_test_p(baseline_only, treatment_only),
        "control_self_disagreement_u_mean": sum(item["control_self_disagreement_u"] for item in per_scene)
        / scenes,
        "treatment_self_disagreement_u_mean": sum(
            item["treatment_self_disagreement_u"] for item in per_scene
        )
        / scenes,
        "zero_difference_variance_scenes": zero_variance_scenes,
        "approximate_within_scene_wald_diagnostic": {
            "standard_error": standard_error,
            "lower": estimate - half_width,
            "upper": estimate + half_width,
            "warning": (
                "Approximate diagnostic only; Bernoulli/Wald coverage can be poor when paired "
                "discordance is sparse. A zero-width interval does not prove equality."
            ),
        },
        "mean_inference_calls": sum(inference_calls) / len(inference_calls),
        "layout": {layout: dict(counts) for layout, counts in layout_totals.items()},
        "control_failure_buckets": dict(sorted(control_failures.items())),
        "treatment_failure_buckets": dict(sorted(treatment_failures.items())),
        "per_scene": per_scene,
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired closed-loop evaluation",
        "",
        f"- Control: `{report['control_checkpoint_id']}`",
        f"- Treatment: `{report['treatment_checkpoint_id']}`",
        f"- Scene/repeat protocol: {report['scenes']} fixed scenes × {report['repeats']} CRN repeats",
        "- Primary metric: containment; strict evaluator success is secondary.",
        "",
        "## Summary",
        "",
        "| Metric | Control | Treatment | Delta | b | c | Discordance | Sign-test p (sensitivity) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in ("containment", "strict"):
        result = report["metrics"][metric]
        lines.append(
            f"| {metric} | {result['control_successes']}/{result['pairs']} "
            f"({percent(result['control_rate'])}) | {result['treatment_successes']}/{result['pairs']} "
            f"({percent(result['treatment_rate'])}) | {percent(result['delta'])} | "
            f"{result['baseline_only']} | {result['treatment_only']} | "
            f"{percent(result['discordance_rate'])} | "
            f"{result['exact_sign_test_p_sensitivity']:.6g} |"
        )
    lines.extend(["", "## Containment diagnostics", ""])
    primary = report["metrics"]["containment"]
    wald = primary["approximate_within_scene_wald_diagnostic"]
    lines.extend(
        [
            f"- Control self-disagreement Ū: {primary['control_self_disagreement_u_mean']:.6f}",
            f"- Treatment self-disagreement Ū: {primary['treatment_self_disagreement_u_mean']:.6f}",
            f"- Zero within-scene difference variance: {primary['zero_difference_variance_scenes']}/{primary['scenes']} scenes",
            f"- Mean inference calls: {primary['mean_inference_calls']:.2f}",
            f"- Approximate within-scene Wald diagnostic: [{percent(wald['lower'])}, {percent(wald['upper'])}]",
            f"- Warning: {wald['warning']}",
            f"- Resolution: one paired rollout flip changes the aggregate estimate by {percent(1.0 / primary['pairs'])}.",
            "",
            "### Layout",
            "",
            "| Layout | Pairs | Control | Treatment | Delta | b | c |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layout in ("cross_side", "same_side"):
        item = primary["layout"][layout]
        control_rate = item["control_successes"] / item["pairs"] if item["pairs"] else 0.0
        treatment_rate = item["treatment_successes"] / item["pairs"] if item["pairs"] else 0.0
        lines.append(
            f"| {layout} | {item['pairs']} | {item['control_successes']} ({percent(control_rate)}) | "
            f"{item['treatment_successes']} ({percent(treatment_rate)}) | "
            f"{percent(treatment_rate - control_rate)} | {item['baseline_only']} | {item['treatment_only']} |"
        )
    lines.extend(
        [
            "",
            "### Failure buckets",
            "",
            f"- Control: `{json.dumps(primary['control_failure_buckets'], sort_keys=True)}`",
            f"- Treatment: `{json.dumps(primary['treatment_failure_buckets'], sort_keys=True)}`",
            "",
            "### Per-scene containment",
            "",
            "| Seed | Layout | Control k/R | Treatment k/R | D_s | b | c |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in primary["per_scene"]:
        lines.append(
            f"| {item['seed']} | {item['layout']} | {item['control_successes']}/{primary['repeats']} | "
            f"{item['treatment_successes']}/{primary['repeats']} | {item['difference']:+.3f} | "
            f"{item['baseline_only_pairs']} | {item['treatment_only_pairs']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    control = load_arm(args.control_dir)
    treatment = load_arm(args.treatment_dir)
    repeat_indices, seeds = validate_pair(control, treatment)
    first_repeat = repeat_indices[0]
    report = {
        "schema_version": 1,
        "control_dir": str(args.control_dir.resolve()),
        "treatment_dir": str(args.treatment_dir.resolve()),
        "control_checkpoint_id": control[first_repeat]["start"].get("policy_checkpoint_id"),
        "treatment_checkpoint_id": treatment[first_repeat]["start"].get("policy_checkpoint_id"),
        "repeat_indices": repeat_indices,
        "seeds": seeds,
        "scenes": len(seeds),
        "repeats": len(repeat_indices),
        "validation": {
            "complete_runs": True,
            "matching_seed_sets": True,
            "matching_initial_scene_signatures": True,
            "matching_common_prefix_inference_seeds": True,
        },
        "metrics": {
            metric: analyze_metric(control, treatment, repeat_indices, seeds, metric)
            for metric in ("containment", "strict")
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(render_markdown(report))
    print(render_markdown(report))


if __name__ == "__main__":
    main()
