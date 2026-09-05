---
name: vla-dagger
description: Adapt, audit, train, and evaluate human-gated DAgger for IsaacLab and GR00T VLA tasks. Use when onboarding a new robot task, validating intervention HDF5 or LeRobot data, fixing action or gripper semantics, preparing base-plus-recovery SFT, running deterministic paired closed-loop evaluation, analyzing DAgger gains and regressions, or deciding whether existing recovery data can be reused before new collection.
---

# VLA DAgger

Build the smallest reproducible DAgger loop around the customer's existing VLA task. Preserve the working base
policy and source data, change one material variable at a time, and make every training claim pass closed-loop review.

## Start here

1. Locate the repository root and read `docs/vla_dagger_guide.md` plus `docs/vla_dagger_reference.md` when present.
2. Inspect the current task registration, policy client, modality config, base dataset converter, success metric, and
   latest evaluated checkpoint. Do not assume the Franka keys or 8-D action space apply.
3. Classify the request: new-task integration, collection, data audit/conversion, SFT preparation, paired evaluation,
   failure diagnosis, or reuse-versus-recollect decision.
4. Read only the matching reference files listed below, then produce or execute a gated plan.

Run `scripts/preflight.py` when a task contract JSON is available. It performs read-only structural checks.

## Non-negotiable rules

- Keep complete source rollouts. Generate recovery clips and loss masks during conversion.
- Record policy, expert, and executed actions at the same transition, plus the achieved next state.
- Require `executed == expert` during intervention and `executed == policy` otherwise.
- Make an expert action valid only when its complete future action horizon is human-controlled.
- Do not assume `annotation.human.action.valid` changes GR00T N1.7 loss; use human-only episode boundaries unless an
  explicit loader/loss-mask hook has been implemented and tested.
- Freeze action semantics, gripper decode, history/action horizons, and normalization before a comparison.
- Rebuild both base and recovery labels if the action representation changes.
- Use every unique recovery source once before considering oversampling.
- Use stable SHA-256-derived inference seeds for paired flow-policy evaluation; never Python `hash()`.
- Review every automatic failure video and report automatic plus adjudicated SR.
- Do not infer that offline action fit guarantees closed-loop gain.
- Do not launch long training, bulk collection, destructive dataset overwrite, or large evaluation unless the user asked.

## Workflow

### 1. Freeze the task contract

Read `references/task-contract.md`. Create a `VLADAggerTaskSpec`-style record containing task ID, policy keys, action
dimension, camera names, horizon/history, gripper semantics, instruction, embodiment, and success metric. Identify the
four task callbacks: observation, policy decode, expert action, and success evaluation.

Stop if state/action time semantics or gripper command meaning are unknown.

### 2. Establish a reproducible base

Freeze scene seeds before candidate evaluation. Record checkpoint ID, task/assets version, normalizer hash, action
horizon, decode threshold, initial-scene signature, outcome, and failure video. For stochastic flow policies, verify
that one repeated request with the same observation and inference seed returns the same raw action chunk.

### 3. Collect complete interventions

Adapt the checked-in Franka gate, recorder, and client instead of rewriting the protocol. Retain policy context and
episode outcome. Require a coherent recovery long enough to expose at least one full future action chunk. Treat final
episode success as metadata, not a substitute for local recovery quality.

### 4. Pass data-readiness gates

Read `references/data-quality-gates.md`. Validate alignment first, replay and inspect video second, inspect action
semantics third, and only then convert. A structural failure blocks training. Preserve rejected raw data and document
why its training view was excluded.

### 5. Build the training view

Extract contiguous intervention segments into human-only training episodes. Aggregate them with the base dataset
without repetition on the first run. Report frame/anchor exposure and independent source coverage. Keep policy context
in source HDF5 only unless the active training loader demonstrably masks it out of expert loss. Before changing the
converter or merge path, read the `LeRobot recovery training-view format` section in `docs/vla_dagger_reference.md`.

### 6. Pin normalization and train

Read `references/training-and-normalization.md`. For unchanged action semantics, warm-start from base weights and keep
the parent normalizer. If only model weights exist, a fresh optimizer is valid; make LR and warmup explicit. Save
intermediate checkpoints, but predeclare the primary evaluation checkpoint.

### 7. Run paired closed-loop evaluation

Read `references/paired-evaluation.md`. Evaluate base and DAgger models on identical frozen scenes and repeat indices.
Require zero scene-signature and common-inference-seed mismatches. Report SR, percentage-point delta, improved,
regressed, unchanged, and failure-type changes. Run `scripts/summarize_eval.py` on the analyzer JSON for a compact
Markdown table.

### 8. Decide reuse versus recollection

Prefer using validated existing recovery data before collecting more. Recollect only when closed-loop failures reveal
a state/phase/layout gap, existing targets conflict with the current policy, or two reproducible training attempts show
no useful signal. Retain older valid recovery data and add current-policy failures as a new aggregation round.

## Reference routing

- `references/task-contract.md`: new robot/task integration and action semantics.
- `references/data-quality-gates.md`: HDF5, video, mask, source-quality, and conversion audits.
- `references/training-and-normalization.md`: base/recovery mixing, stats, optimizer, and checkpoint choices.
- `references/paired-evaluation.md`: CRN protocol, failure review, metrics, and reporting.
- `references/franka-reference.md`: validated implementation map and reference outcome matrix.
- `docs/vla_dagger_reference.md#lerobot-recovery-training-view-format`: exact human-only LeRobot v2 directory,
  Parquet, provenance, horizon-anchor, and merge schemas.

## Expected output

Return a concise status with:

- what is already valid;
- the first blocking gate, if any;
- exact files or commands changed/generated;
- tests performed and their results;
- the next safe action;
- an explicit statement when current data is insufficient, including the evidence and targeted recollection rule.
