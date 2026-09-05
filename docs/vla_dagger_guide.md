# VLA DAgger customer guide

This guide turns a working IsaacLab + GR00T VLA task into a human-gated DAgger loop:

```text
base policy → fixed-scene evaluation → human intervention collection
            → data validation/conversion → base+recovery SFT
            → paired closed-loop evaluation → next targeted collection round
```

The checked-in Franka implementation is the executable reference. The robot-independent task, data, and evaluation
contracts live under `scripts/benchmarks/gr00t/dagger/`. Codex users can invoke the repo-local `$vla-dagger` Skill to
inspect another task and produce an adapted checklist or commands.

## Use the repo-local Codex Skill

Open this repository as the Codex workspace and confirm that `.agents/skills/vla-dagger/SKILL.md` is present. Invoke
the Skill explicitly with `$vla-dagger`; no copy into a personal skill directory is required.

Start with a read-only onboarding pass:

```text
Use $vla-dagger to inspect this repository for my IsaacLab + GR00T task <task-id>.
Do not modify files or launch training yet. Identify the task contract, existing base
checkpoint and dataset path, action/state semantics, cameras, horizons, teleoperation
adapter, success metric, and the first blocking DAgger readiness gate. Then give me an
adaptation plan with exact source files and validation commands.
```

Provide the task ID and known base checkpoint/dataset paths. When they cannot be inferred from the repository, also
provide the robot/action representation, observation and action dimensions, camera keys, action/history horizons,
teleoperation device, and success condition. The Skill must report unknown semantics as blockers rather than guess.

After reviewing the plan, authorize the implementation explicitly:

```text
Use $vla-dagger to implement the approved DAgger adaptation. Preserve complete source
rollouts, build a human-only LeRobot recovery view, add structural validation, and stop
before long training or bulk collection. Run the local unit and CLI smoke tests.
```

For an existing collection, use a focused audit request:

```text
Use $vla-dagger to audit <dataset.hdf5> for action alignment, intervention boundaries,
complete H32 human anchors, gripper semantics, video synchronization, and LeRobot
conversion readiness. Do not train; return the first failing gate and exact evidence.
```

For evaluation setup, request frozen scenes and paired flow-policy noise explicitly:

```text
Use $vla-dagger to prepare paired closed-loop evaluation for <base-checkpoint> and
<dagger-checkpoint>. Reuse the frozen scene list, apply stable inference seeds, save
failure videos, and report automatic plus manually adjudicated success rates.
```

The Skill reads this Guide and only the relevant files under `.agents/skills/vla-dagger/references/`. Its expected
handoff states what is already valid, the first blocking gate, files or commands changed, validations performed, and
the next safe action. It does not launch long training, bulk collection, destructive dataset overwrite, or large
evaluation unless the customer explicitly requests it.

## Validated result

On the Franka two-box sorting task, the final fixed-scene, common-random-number evaluation used 50 scenes × 3 policy
noise repeats and manual review of every automatic failure:

| Route | Base | Base + DAgger SFT | Gain |
| --- | ---: | ---: | ---: |
| Continuous gripper representation | 120/150 = 80.00% | 132/150 = 88.00% | **+8.00 pp** |
| Direct binary gripper representation | 123/150 = 82.00% | 120/150 = 80.00% | −2.00 pp |

See the [system reference](vla_dagger_reference.md) for the architecture, data contract, and interpretation. The
matrix proves the end-to-end DAgger stage can add value; it also shows that changing action representation is not
automatically an improvement. Keep the working base representation fixed during a first DAgger iteration.

## 1. Freeze the task contract

Before collection, record one immutable task specification. Start from
[`franka_dagger_task.py`](../scripts/benchmarks/gr00t/franka/franka_dagger_task.py) and replace:

- IsaacLab task ID and policy type;
- policy observation/action keys plus independent state and action dimensions;
- camera names, frame rate, action horizon, and state/video history;
- teleoperation device and expert-action construction;
- gripper command semantics, including the action index and open/close targets;
- language instruction and embodiment tag;
- automatic task-success definition.

The reusable dataclass is
[`task_spec.py`](../scripts/benchmarks/gr00t/dagger/task_spec.py). A task spec rejects duplicate or empty keys, an
invalid gripper index, and intervention windows shorter than the action horizon.

Four callbacks remain task-specific and must be tested in simulation:

1. `observation(env) -> policy observation`;
2. `policy_action(model_output) -> environment action`;
3. `expert_action(teleop_input, env) -> environment action`;
4. `success(env) -> bool` plus any diagnostic failure buckets.

Do not collect until one manually controlled episode can be recorded, replayed, converted, and trained through the
complete path.

## 2. Lock action and time semantics

For every transition, write down and test this equation:

```text
state[t] -- executed_action[t] --> achieved_state[t+1]
```

The HDF5 must keep all three action views:

- `policy_action[t]`: what the policy proposed;
- `expert_action[t]`: what the human proposed;
- `executed_action[t]`: the command actually sent to the environment.

During takeover, `executed_action == expert_action`; outside takeover, `executed_action == policy_action`. Never label
policy failure actions as expert targets. Preserve the complete rollout in HDF5 and create the training view during
conversion.

The Franka reference uses an 8-D joint-space vector. Its arm target is the next achieved joint state. The gripper can
use either:

- continuous achieved width; or
- the recorded binary command mapped as close = `0.0`, open = `0.08`.

If a customer task has the true controller command, prefer a command target whose inference decoder sends the same
physical quantity. Whatever representation is chosen, rebuild the base and recovery datasets consistently. Mixing a
continuous base label with a binary DAgger label creates contradictory supervision.

## 3. Establish the base result first

Freeze a scene list before evaluating any DAgger checkpoint. The Franka reference uses seeds 11–60. Record:

- checkpoint ID and normalization-statistics hash;
- task/assets version;
- action/history horizons and gripper decode threshold;
- scene seed, initial-scene signature, result, failure type, and failure video;
- policy-side inference seed for every action request.

For flow policies, apply the minimal GR00T N1.7 seed patch:

```bash
cd "$GROOT"
git apply --check "$OVERLAY/patches/isaac-gr00t-n1.7-crn-inference-seed.patch"
git apply "$OVERLAY/patches/isaac-gr00t-n1.7-crn-inference-seed.patch"
```

If `git apply --check` reports that the patch is already applied, inspect the two target files instead of applying it
again. The patch forwards `options` through `Gr00tPolicy.get_action` and seeds only the initial flow-noise tensor.

Generate a frozen seed list once:

```bash
python - <<'PY'
import json
from pathlib import Path

Path("eval_seeds_11_60.json").write_text(json.dumps(list(range(11, 61))) + "\n")
PY
```

Run at least three policy-noise repeats for both base and candidate. Use the same repeat index on both arms. The
Franka client derives a stable seed from `(episode_seed, repeat_index, inference_index)` with SHA-256; it never uses
Python's process-randomized `hash()`.

## 4. Integrate human-gated collection

The reference implementation is split by responsibility:

| File | Responsibility |
| --- | --- |
| [`hg_dagger_core.py`](../scripts/benchmarks/gr00t/franka/hg_dagger_core.py) | takeover gate, contiguous segments, complete-horizon masks |
| [`hg_dagger_ik.py`](../scripts/benchmarks/gr00t/franka/hg_dagger_ik.py) | Franka SpaceMouse-to-joint IK adapter |
| [`hg_dagger_recorder.py`](../scripts/benchmarks/gr00t/franka/hg_dagger_recorder.py) | aligned per-transition and per-episode HDF5 metadata |
| [`gr00t_inference_client_franka.py`](../scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py) | policy execution, takeover loop, collection, JSONL eval, failure videos |

For another robot, keep the gate and data contract, then replace the Franka IK/action extraction and task success
logic. The generic array validator in
[`data_contract.py`](../scripts/benchmarks/gr00t/dagger/data_contract.py) supports arbitrary action dimensions.

Franka collection example:

```bash
cd "$ISAACLAB"
export FRANKA_GROOT_ACTION_HORIZON=32
export FRANKA_GROOT_STATE_DELTA_INDICES=0

./isaaclab.sh -p scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py \
  --policy-type joint_space \
  --task Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0 \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --hg-dagger \
  --policy-checkpoint-id "$BASE_CKPT_ID" \
  --baseline-dataset-id "$BASE_DATASET_ID" \
  --asset-version "$ASSET_VERSION" \
  --dataset-file datasets/customer_hg_dagger.hdf5 \
  --num-total-experiments 50 \
  --num-feedback-actions 32 \
  --minimum-intervention-steps 64 \
  --max-episode-steps 1200
```

Collection requires a visible Kit window and SpaceMouse. In the Franka adapter, `B` toggles takeover/release and `R`
aborts an episode. A release requested too early is deferred until the minimum intervention length.

Operator rules:

- intervene as soon as failure is likely, not after the state is unrecoverable;
- finish one coherent local recovery, including approach, settle, close, and lift/retry when applicable;
- after the decisive action, retain human control for at least one complete future action horizon;
- avoid long idle sections, abrupt wrist flips, collision-driven shortcuts, and ambiguous handoff states;
- retain complete source rollouts, including policy context and final outcome;
- do not discard a locally valid recovery only because the full episode later failed.

## 5. Run data-readiness gates

### Gate A — structural alignment

```bash
python scripts/benchmarks/gr00t/franka/validate_hg_dagger_dataset.py \
  --dataset-file datasets/customer_hg_dagger.hdf5 \
  --action-horizon 32 \
  --minimum-segment-length 64
```

This verifies transition counts, finite vectors, policy/expert/executed-action equality rules, contiguous frame indices,
episode metadata, and usable fully human-controlled action-horizon anchors.

### Gate B — replay and video synchronization

```bash
./isaaclab.sh -p scripts/benchmarks/gr00t/franka/replay_demos_with_camera.py \
  --task Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0 \
  --dataset_file datasets/customer_hg_dagger.hdf5 \
  --video \
  --video-output-dir datasets/customer_hg_dagger/replay_videos \
  --validate_success_rate \
  --failure-output-file datasets/customer_hg_dagger/replay_failures.jsonl

python scripts/benchmarks/gr00t/franka/annotate_hg_dagger_videos.py \
  --dataset-file datasets/customer_hg_dagger.hdf5 \
  --video-dir datasets/customer_hg_dagger/replay_videos
```

Watch a sample spanning short/long interventions, episode tails, successful sources, and failed sources. Reject data
with camera/action offset, incomplete recovery, harmful human motion, recording corruption, or the wrong task label.

### Gate C — action semantics

Independently check:

- gripper command distribution and command-to-target mapping;
- delay between close command and achieved gripper state;
- per-joint residual between executed command and achieved next state;
- model output distribution relative to the inference decode threshold.

A threshold sweep diagnoses a continuous output; it does not repair incorrect training labels.

## 6. Convert complete rollouts into a recovery training view

The converter extracts each contiguous human takeover as a LeRobot episode. With the standard no-padding GR00T N1.7
loader, the episode boundary ensures that only anchors with a complete future H32 human-action window are sampled.
`annotation.human.action.valid` is retained for audit, but the standard loader does **not** consume it as an
anchor-level loss mask. Therefore this reference path requires `--state-history-frames 1` and does not prepend policy
context. Adding context requires a separately tested loader/loss-mask integration.

See [LeRobot recovery training-view format](vla_dagger_reference.md#lerobot-recovery-training-view-format) for the
directory layout, Parquet fields, episode provenance, H32 anchor example, and the schema projection performed during
base-plus-recovery merging.

Continuous Franka representation:

```bash
python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_dagger_joint_space.py \
  --hdf5-file-path datasets/customer_hg_dagger.hdf5 \
  --video-dir datasets/customer_hg_dagger/replay_videos \
  --lerobot-data-dir datasets/customer_hg_dagger/lerobot_recovery \
  --minimum-segment-length 64 \
  --action-horizon 32 \
  --state-history-frames 1 \
  --require-videos
```

Binary Franka gripper targets add:

```text
--binary-gripper-command-target --gripper-close-width 0.0 --gripper-open-width 0.08
```

The converter includes locally valid segments from failed source episodes by default. `--success-only` is a
conservative filter, but it may discard useful recovery. Decide from segment quality, not only final episode outcome.

Audit the exact training view:

```bash
python scripts/benchmarks/gr00t/franka/audit_hg_dagger_training_view.py \
  --hdf5-file datasets/customer_hg_dagger.hdf5 \
  --lerobot-data-dir datasets/customer_hg_dagger/lerobot_recovery \
  --action-horizon 32 \
  --minimum-segment-length 64 \
  --allow-failed-source \
  --output-json datasets/customer_hg_dagger/training_view_audit.json
```

`audit_hg_dagger_training_view.py` contains Franka sorting geometry diagnostics. Another task should keep the alignment
checks and replace the phase/layout/containment calculations.

## 7. Aggregate with the base dataset

First use every unique recovery episode once. Do not oversample by default.

```bash
python scripts/benchmarks/gr00t/franka/merge_lerobot_dagger_datasets.py \
  --base-dataset "$BASE_LEROBOT" \
  --dagger-dataset datasets/customer_hg_dagger/lerobot_recovery \
  --output-dataset datasets/customer_base_plus_dagger \
  --all-unique-dagger
```

`--all-unique-dagger` ignores the requested target fraction and includes every accepted recovery episode once.
Read `meta/dagger_mix.json` for the actual frame fraction and selection mode. For a ratio-limited experiment, omit
this flag and set `--target-dagger-fraction`; add `--allow-repeat` only when repetition is the declared variable.

If a few long segments dominate the mix, report both frame/anchor fraction and independent source-episode counts.
Only test source/segment-balanced sampling after the natural unique-data recipe has a measured result.

## 8. Pin normalization, then train

For incremental SFT with unchanged action semantics, keep the parent normalization. Copy the base dataset's matching
`meta/stats.json` and action-horizon-specific `relative_stats.json` into the mix, or otherwise configure GR00T to load
the exact parent statistics. Verify the effective files and hashes before training and inference.

Recompute statistics only when deliberately changing the action representation, and then rebuild **both** base and
recovery datasets in that representation. Never let GR00T silently normalize the two arms with different statistics.

Useful comparison:

```bash
python scripts/benchmarks/gr00t/franka/compare_gr00t_dataset_stats.py \
  --baseline-dataset "$BASE_LEROBOT" \
  --candidate-dataset datasets/customer_base_plus_dagger \
  --report-file datasets/customer_base_plus_dagger/meta/stats_comparison.json
```

GR00T N1.7 SFT template:

```bash
cd "$GROOT"
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=32
export FRANKA_GROOT_STATE_DELTA_INDICES=0

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_CKPT" \
  --dataset-path "$MIX_DATASET" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --modality-config-path "$MODALITY_CONFIG" \
  --num-gpus 1 \
  --output-dir "$OUTPUT_DIR" \
  --save-steps 1000 \
  --max-steps 5000 \
  --global-batch-size 256 \
  --dataloader-num-workers 4
```

The tested 5k checkpoint is a reference, not a universal optimum. Save intermediate checkpoints (for example 1k,
2k, and 5k) and predeclare the primary one before closed-loop evaluation. When only model weights were saved, use a
fresh optimizer with the same LR/warmup settings for compared runs; an optimizer state is not required for customer
DAgger SFT.

## 9. Evaluate base and DAgger checkpoints identically

Start one GR00T server per checkpoint. For each repeat index `0, 1, 2`, run the same frozen seed file:

```bash
./isaaclab.sh -p scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py \
  --policy-type joint_space \
  --task Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0 \
  --server-host 127.0.0.1 \
  --server-port 5555 \
  --num-total-experiments 50 \
  --num-feedback-actions 32 \
  --episode-seeds-file eval_seeds_11_60.json \
  --policy-noise-repeat-index 0 \
  --verify-policy-action-determinism \
  --policy-checkpoint-id "$CHECKPOINT_ID" \
  --baseline-dataset-id "$BASE_DATASET_ID" \
  --asset-version "$ASSET_VERSION" \
  --eval-results-file eval/repeat_0.jsonl \
  --overwrite-eval-results \
  --failure-video-dir eval/failures/repeat_0 \
  --headless
```

Repeat with indices 1 and 2 for both models. Use the same gripper decode threshold within an action representation.

Analyze the automatic paired results:

```bash
python scripts/benchmarks/gr00t/franka/verify_eval_determinism.py \
  eval/smoke/base_first.jsonl eval/smoke/base_second.jsonl

python scripts/benchmarks/gr00t/franka/verify_eval_determinism.py \
  eval/smoke/dagger_first.jsonl eval/smoke/dagger_second.jsonl

python scripts/benchmarks/gr00t/franka/analyze_paired_closed_loop.py \
  --control-dir eval/base \
  --treatment-dir eval/dagger \
  --json-output eval/paired.json \
  --markdown-output eval/paired.md
```

Then manually review **every automatic failure video**. Mix the videos across checkpoints/repeats and hide their
source while judging. Report the automatic metric and manually adjudicated SR together; never silently replace one.
For a new task, implement a geometry-based success metric that checks the real task semantics rather than relying on
a convenient center-distance threshold.

Always report:

- base and DAgger SR by repeat and total;
- net percentage-point gain;
- paired improved / regressed / unchanged counts;
- failures by type;
- scene-signature and common inference-seed mismatch counts;
- manual corrections to automatic failure labels.

## 10. Decide the next iteration

If SR rises and critical failure types do not regress, retain the new checkpoint and collect the next round from its
remaining failures.

If SR is flat but many scenes flip, DAgger is changing behavior without net improvement. Inspect improved versus
regressed failure types, label consistency, normalization, and segment/source concentration before increasing the
recovery sampling ratio.

If SR falls, stop adding steps. Compare earlier checkpoints, verify parent stats and action semantics, and inspect
whether recovery targets conflict with base behavior.

Collect new data only after current data has passed the gates and a closed-loop result identifies a genuine coverage
gap. New collection should target the current policy's failures, use multiple independent source episodes per failure
mode, and retain the old recovery set as part of dataset aggregation.

## Common failure modes

- **No usable H32 anchors:** intervention segments are shorter than the horizon or release occurs too early.
- **Video is one frame shorter:** replay has one observation per transition target; the converter clips only a valid
  tail and rejects an incomplete segment.
- **Policy learns to stay open:** gripper action target used achieved width while inference interpreted it as a binary
  command, or close timing is shifted.
- **DAgger hurts base behavior:** recovery labels conflict with the base representation, parent normalization was not
  preserved, or a few source episodes dominate the mix.
- **Paired results are not paired:** scene signatures differ, GR00T ignores `inference_seed`, or Python `hash()` was
  used across processes.
- **Automatic SR disagrees with video:** success geometry does not encode containment, correct target, release, or
  final stability. Fix and freeze the metric before evaluating candidates.

## Local validation

Pure data tests:

```bash
python -m pytest \
  scripts/benchmarks/gr00t/franka/tests/test_hg_dagger_core.py \
  scripts/benchmarks/gr00t/franka/tests/test_dagger_converter.py \
  scripts/benchmarks/gr00t/franka/tests/test_dagger_generic.py -q
```

IsaacLab Franka IK smoke test:

```bash
./isaaclab.sh -p scripts/benchmarks/gr00t/franka/tests/hg_dagger_ik_smoke.py \
  --task Isaac-Pick-Place-Franka-Joint-Position-v0 \
  --headless
```
