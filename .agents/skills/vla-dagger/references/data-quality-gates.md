# Data-quality gates

Run gates in order. Do not use training to diagnose a failed recording contract.

## Gate A — source integrity

Require:

- complete HDF5 close and readable episode groups;
- non-empty policy checkpoint ID and episode seed;
- finite policy, expert, executed, observation, and achieved vectors;
- equal transition counts across all step fields;
- contiguous frame indices;
- complete initial-state data for deterministic replay.

## Gate B — controller consistency

For each transition:

```text
intervention=True  => executed_action == expert_action
intervention=False => executed_action == policy_action
intervention=True  => policy_action_valid=False
```

Check dimensions and equality before any clipping, normalization, or conversion.

## Gate C — synchronized replay

Replay the source actions with cameras enabled. Check a sample from episode starts, intervention boundaries, long
takeovers, source-final takeovers, successful outcomes, and failed outcomes. Reject camera/state/action offset,
truncated video, abrupt jumps, or recording mistakes.

## Gate D — local recovery quality

Classify each candidate segment before looking at candidate-model results:

- usable: human action resolves the local failure and hands off in a nominal state;
- incomplete: intervention ends before a coherent recovery or complete future horizon;
- harmful: collision, unstable wrist motion, wrong task action, or worse handoff state;
- ambiguous: insufficient visual/state evidence.

Final episode failure alone does not make a locally usable recovery invalid.

## Gate E — training-view masks

For horizon `H`, valid anchor `t` requires human control for all actions in `[t, t+H)`. A segment shorter than `H`
has no valid anchors. The standard GR00T N1.7 loader does not use `annotation.human.action.valid` as an anchor-level
loss mask, so build human-only training episodes and rely on no-padding horizon sampling. Context frames may enter the
training view only after an explicit loader/loss-mask hook is implemented and tested. Never pad an expert target across
a human/policy boundary.

Report:

- source episodes and recovery segments;
- intervention frames and valid anchors;
- segment length distribution;
- sources contributing the most anchors;
- phase/layout/failure-mode coverage;
- failed-source segments accepted/rejected and why.

## Gate F — action semantics

Check gripper command-to-target mapping, command/achieved delay, per-joint command-versus-achieved residuals, and output
distribution relative to the inference decoder. A decode threshold is a diagnostic/contract parameter, not a repair
for incorrect training labels.

## Gate G — dataset compatibility

Base and recovery LeRobot datasets must share camera keys/shapes, policy keys, task metadata, action order/dimension,
FPS, modality config, and action/history horizons. Refuse a merge on mismatch.
