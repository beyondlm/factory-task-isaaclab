# VLA DAgger system reference

This document describes the validated architecture and transferable engineering conclusions for human-gated DAgger
with IsaacLab and GR00T N1.7. It is a system reference, not an experiment diary.

## Reference system

The reference implementation uses a Franka joint-space VLA policy on a two-object sorting task:

- two RGB observations: wrist and table cameras;
- 8-D robot vector: seven arm joints plus gripper width;
- action horizon 32;
- current-frame video and robot state;
- SpaceMouse expert control with a keyboard takeover gate;
- complete source rollouts in IsaacLab HDF5;
- recovery-segment training views in LeRobot v2;
- GR00T N1.7 supervised fine-tuning from the base checkpoint;
- fixed-scene, repeated, paired closed-loop evaluation.

The robot-independent configuration is represented by
[`VLADAggerTaskSpec`](../scripts/benchmarks/gr00t/dagger/task_spec.py). The executable Franka adapter is
[`franka_dagger_task.py`](../scripts/benchmarks/gr00t/franka/franka_dagger_task.py).

## System architecture

```text
                         ┌──────────────────────────┐
                         │ Base VLA checkpoint      │
                         └────────────┬─────────────┘
                                      │ policy action chunks
                                      ▼
┌──────────────┐  takeover   ┌──────────────────────────┐
│ Human expert │────────────▶│ IsaacLab execution loop  │
└──────────────┘             └────────────┬─────────────┘
                                         │ complete aligned rollout
                                         ▼
                              ┌──────────────────────────┐
                              │ HDF5 source dataset      │
                              │ policy/expert/executed   │
                              │ intervention + outcome   │
                              └────────────┬─────────────┘
                                           │ validate/replay/audit
                                           ▼
                              ┌──────────────────────────┐
                              │ LeRobot recovery view    │
                              │ human-only future masks  │
                              └────────────┬─────────────┘
                                           │ unique-data aggregation
                                           ▼
                              ┌──────────────────────────┐
                              │ Base + recovery dataset  │
                              └────────────┬─────────────┘
                                           │ SFT from base weights
                                           ▼
                              ┌──────────────────────────┐
                              │ DAgger checkpoint        │
                              └────────────┬─────────────┘
                                           │ paired closed-loop eval
                                           └──────────▶ next iteration
```

## Transition contract

At simulation step `t`:

```text
observation_state[t]
policy_action[t]
expert_action[t]
intervention_mask[t]
executed_action[t] = expert_action[t] if intervention else policy_action[t]
achieved_state[t+1] = env.step(executed_action[t])
```

The recorder writes these fields before and after the same environment step. The validator rejects:

- mismatched row counts or non-finite vectors;
- non-contiguous frame indices;
- policy actions marked valid during takeover;
- executed actions that disagree with the active controller;
- missing episode provenance;
- recovery segments without at least one complete future action horizon.

The source rollout is the immutable record. Filtering and loss eligibility are training-view operations, not reasons
to destroy source data.

## Human-only action-horizon supervision

For an action horizon `H`, anchor `t` is an expert target only when every transition in `[t, t + H)` is
human-controlled. A takeover of length `L` therefore contributes `max(0, L - H + 1)` valid anchors.

The checked-in standard SFT path enforces this with human-only episode boundaries and GR00T's no-padding horizon
sampler. `annotation.human.action.valid` is audit metadata, not an automatically consumed loss mask in GR00T N1.7.
Policy context may be retained in the source HDF5, but adding it to a training episode requires an explicit, tested
loader/loss-mask integration so it cannot silently become expert loss.

## LeRobot recovery training-view format

The reference pipeline deliberately keeps three different artifacts. They have different purposes and must not be
treated as interchangeable:

| Artifact | Content | Purpose |
| --- | --- | --- |
| Source IsaacLab HDF5 | Complete policy rollout, human takeovers, executed commands, achieved next states, timing, and final outcome | Immutable replay and audit record |
| Recovery LeRobot v2 dataset | One episode per contiguous human takeover; only takeover rows and their aligned videos | Auditable expert-only SFT source |
| Merged LeRobot v2 dataset | Baseline episodes plus selected recovery episodes, projected onto the baseline feature schema | Actual GR00T SFT input |

The HDF5 remains the source of truth. Its aligned transition fields include
`dagger/observation_joint_state`, `dagger/policy_action`, `dagger/expert_action`,
`dagger/executed_action`, `dagger/intervention_mask`, `dagger/policy_action_valid`,
`dagger/achieved_joint_state`, chunk/frame indices, and episode provenance. During intervention,
`executed_action == expert_action`; outside intervention, `executed_action == policy_action`.

### Directory layout

The converter writes the following LeRobot v2 layout:

```text
lerobot_recovery/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── modality.json
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.wrist_camera/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.table_camera/
            ├── episode_000000.mp4
            └── ...
```

`meta/info.json` records the path templates, FPS, counts, splits, and feature dtype/shape metadata.
`meta/tasks.jsonl` maps `task_index` to the language instruction. `meta/modality.json` provides the GR00T modality
mapping. Videos are stored as MP4 files rather than embedded in Parquet.

### Recovery Parquet schema

The checked-in Franka converter writes these columns for every row in a human takeover. Vector sizes are task-specific;
the Franka reference uses eight values: seven arm joints plus gripper width.

| Column | Reference dtype/shape | Meaning | Used as the standard training target? |
| --- | --- | --- | --- |
| `observation.state` | `float32[state_dim]` | Robot state immediately before the executed action | Yes, observation |
| `action` | `float32[action_dim]` | Next achieved joint state; optional binary gripper element is derived from the executed human command | Yes, action target |
| `action.policy` | `float32[action_dim]` | Policy proposal at the same transition | No, audit only |
| `action.expert` | `float32[action_dim]` | Human proposal at the same transition | No, audit only |
| `action.executed_command` | `float32[action_dim]` | Command actually passed to `env.step` | No, audit only |
| `annotation.human.action.intervention` | `bool` | Whether the human controlled this transition; true for every row in this recovery episode | No, audit only |
| `annotation.policy.action.valid` | `bool` | Whether the recorded policy proposal was valid | No, audit only |
| `annotation.human.action.valid` | `bool` | Whether this row can start a complete future human-controlled action chunk | Audit only; GR00T N1.7 does not consume it as a loss mask |
| `annotation.human.action.task_description` | `int64` | Task/instruction index | Yes, task conditioning |
| `timestamp` | `float64` | Time from the beginning of this recovery episode | Dataset bookkeeping |
| `episode_index`, `task_index`, `index` | `int64` | Local episode, task, and global row indices | Dataset bookkeeping |
| `next.reward`, `next.done` | `float64`, `bool` | Terminal metadata for the recovery episode | Dataset bookkeeping |

The `action` target is intentionally not a blind copy of `action.expert`. In the Franka joint-space recipe, arm targets
use the achieved state after the human command, matching the baseline representation. When binary gripper targets are
requested, only the gripper element is replaced from `action.executed_command` using close = `0.0` and open = `0.08`.
Any customer task must keep baseline and recovery action semantics identical.

### Episode metadata and horizon rule

Each line of `meta/episodes.jsonl` preserves the source mapping. A representative record is:

```json
{"episode_index": 0, "tasks": ["<instruction>"], "length": 64, "source_episode": "demo_0007", "source_range": [120, 184], "recovery_range": [120, 184], "source_success": false}
```

`source_success: false` does not make the local recovery invalid. It means only that the complete source rollout later
failed; the takeover remains usable when its local human action is coherent and passes audit.

For horizon `H`, a recovery episode of `L` rows contributes `max(0, L - H + 1)` training anchors. For example, a
64-step takeover stored as 64 Parquet rows contributes 33 valid H32 anchors. The final 31 rows have
`annotation.human.action.valid = false` because they cannot start a full H32 future. The standard GR00T no-padding
sampler reaches the same result from the episode boundary and samples only anchors 0 through 32; it does not rely on
the annotation as a loss mask.

Only manual-takeover rows enter the recovery LeRobot episodes: the converter uses `context_start = recovery_start`.
Pre-takeover policy context and post-handoff policy behavior remain available in the complete HDF5 for replay and
diagnosis, but neither is copied into the standard SFT episode and no policy failure action becomes an expert target.

### Final base-plus-recovery dataset

The merge step first verifies that recovery data supplies every baseline feature with the same dtype and shape and that
both datasets use the same cameras and task metadata. It then projects every episode onto the baseline's non-video
feature columns. Consequently, recovery-only audit columns such as `action.policy`, `action.expert`,
`action.executed_command`, and intervention markers normally remain in `lerobot_recovery` but are not copied into the
actual training dataset.

The merged `meta/episodes.jsonl` adds `source: "base"` or `source: "dagger"` plus the original episode index.
`meta/dagger_mix.json` records requested and actual recovery frame fractions, frame counts, whether repetition was
enabled, and the selection mode. The merge utility does not define a new normalizer; the training recipe must apply the
chosen parent or rebuilt statistics explicitly.

The executable references are:

- [`convert_hdf5_to_lerobot_dagger_joint_space.py`](../scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_dagger_joint_space.py) — extract human-only recovery episodes;
- [`audit_hg_dagger_training_view.py`](../scripts/benchmarks/gr00t/franka/audit_hg_dagger_training_view.py) — compare the training view with the source HDF5;
- [`merge_lerobot_dagger_datasets.py`](../scripts/benchmarks/gr00t/franka/merge_lerobot_dagger_datasets.py) — create the baseline-compatible SFT dataset.

## Action-representation reference matrix

The complete DAgger stage was validated with 50 fixed scenes, three policy-noise repeats, common inference noise,
automatic success checks, and review of all automatic failures:

| Representation | Close threshold | Base SR | Base + DAgger SFT SR | Net |
| --- | ---: | ---: | ---: | ---: |
| Continuous gripper width | `0.065 m` | 120/150 = 80.00% | 132/150 = 88.00% | **+8.00 pp** |
| Direct binary target | `0.04 m` | 123/150 = 82.00% | 120/150 = 80.00% | −2.00 pp |

Transferable conclusions:

1. The end-to-end DAgger system can provide a material closed-loop gain.
2. DAgger data is not sufficient by itself; its label semantics must match the base action representation.
3. Binary gripper targets are a representation option, not a universal improvement.
4. For the first DAgger iteration, keep the working base action representation and normalization fixed.
5. Evaluate improved and regressed pairs, not only aggregate SR.

The reference Continuous comparison changed 17 paired rollouts from failure to success and regressed 5, producing a
net +12/150. The binary comparison improved 10 and regressed 13, producing a net −3/150. This is why equal-looking
aggregate behavior can still hide substantial capability transfer.

## Gripper semantics

The reference converter supports two internally consistent targets:

- continuous: next achieved gripper width;
- binary command target: close command → `0.0`, open command → `0.08`.

The model action head remains continuous in both cases. Binary labels create two target modes but do not turn the
action head into a classifier; predictions between the two targets remain possible and still require a frozen decode
threshold.

Before training, measure:

- binary command agreement;
- close-command to achieved-width delay;
- predicted gripper distribution;
- early-close and no-close failures under the chosen threshold.

Do not repair a label-space mismatch only by adjusting the inference threshold.

## Data selection and aggregation

The reference HDF5 keeps full policy and intervention rollouts. Conversion extracts contiguous recovery segments.
Final episode failure does not automatically invalidate an earlier local recovery; segment quality determines whether
it can be used.

The validated first-pass aggregation used every selected recovery segment once and did not repeat episodes to force a
target ratio. Report both:

- frame/anchor exposure; and
- independent source-episode coverage.

Oversampling a small number of sources increases exposure but not state diversity. Sampling changes should follow a
measured natural-data result.

## Normalization contract

For incremental DAgger SFT with unchanged action semantics:

- start from the base model weights;
- preserve the base normalization statistics;
- verify the effective statistics hash in each inference checkpoint;
- use a fresh optimizer if only model weights were saved.

When changing action representation, rebuild base and recovery labels consistently and regenerate statistics for that
new representation. Do not let base and DAgger arms use independently drifting normalizers in a comparison.

## Evaluation contract

The fixed-scene evaluation uses two layers of pairing:

1. the same initial scene seed and verified initial-scene signature;
2. the same SHA-256-derived flow-noise seed for each `(scene, repeat, inference index)`.

The client records a JSONL run manifest, per-episode result, inference trace, automatic success metrics, failure bucket,
and optional failure video. The paired analyzer refuses incomplete runs, duplicate seeds, mismatched scenes, or
mismatched common inference seeds.

Automatic task success must represent the real task. For placement, a robust metric includes correct target identity,
bin-local object-footprint containment, height, release, and final stability. All automatic failures should be reviewed
blindly; report automatic and adjudicated SR together.

## Failure taxonomy for the next collection round

Use failure buckets that identify a recoverable state, not only the final outcome. The reference task uses categories
such as:

- missed grasp without retry;
- object pushed away by gripper contact;
- wrist/arm pose collapse after the first subtask;
- wrong target bin;
- hover or delayed commit;
- timeout after partial completion.

New collection targets the current checkpoint's remaining failure distribution while preserving older recovery data.
Use multiple independent source episodes for each recurring state; do not count overlapping H32 anchors as independent
coverage.

## Reference implementation map

| Layer | Files |
| --- | --- |
| Generic contract | `scripts/benchmarks/gr00t/dagger/{task_spec,data_contract,evaluation}.py` |
| Takeover and masks | `scripts/benchmarks/gr00t/franka/hg_dagger_core.py` |
| Expert controller | `scripts/benchmarks/gr00t/franka/hg_dagger_ik.py` |
| HDF5 recorder | `scripts/benchmarks/gr00t/franka/hg_dagger_recorder.py` |
| Collection/evaluation client | `scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py` |
| Structural validation | `scripts/benchmarks/gr00t/franka/validate_hg_dagger_dataset.py` |
| Replay/annotation | `replay_demos_with_camera.py`, `annotate_hg_dagger_videos.py` |
| Training-view conversion | `convert_hdf5_to_lerobot_dagger_joint_space.py` |
| Training-view audit | `audit_hg_dagger_training_view.py` |
| Dataset aggregation | `merge_lerobot_dagger_datasets.py` |
| Closed-loop integrity/analysis | `verify_eval_determinism.py`, `analyze_paired_closed_loop.py` |
| GR00T flow-noise hook | `patches/isaac-gr00t-n1.7-crn-inference-seed.patch` |

For commands and customer adaptation steps, use the [VLA DAgger customer guide](vla_dagger_guide.md).
