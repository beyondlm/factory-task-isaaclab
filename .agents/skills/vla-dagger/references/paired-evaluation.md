# Paired closed-loop evaluation

## Fixed inputs

Freeze before candidate results:

- scene-seed list;
- task/assets version;
- action and history horizons;
- policy checkpoint and normalizer hashes;
- gripper decoder/threshold;
- automatic success metric;
- repeat count and primary checkpoint.

## Common random numbers

For each policy request, derive:

```text
inference_seed = SHA256(namespace, episode_seed, repeat_index, inference_index)
```

Both models receive the same seed at the same scene/repeat/inference index. Patch GR00T N1.7 with
`patches/isaac-gr00t-n1.7-crn-inference-seed.patch` so `options.inference_seed` controls the initial flow-noise tensor.

Smoke-test the first action request twice and require bit-identical raw actions. Record every inference seed in JSONL.
Never use Python `hash()` for cross-process seeds.

## Integrity checks

Reject or rerun when:

- a run is incomplete/interrupted;
- seed sets or repeat indices differ;
- initial-scene signatures mismatch;
- common inference seeds mismatch;
- checkpoint IDs or normalizer hashes are missing;
- the automatic success metric changed between arms.

## Primary report

For each metric, report:

- base and DAgger successes by repeat and total;
- absolute percentage-point delta;
- improved, regressed, and unchanged paired rollouts;
- discordance rate and failure buckets;
- all per-scene differences;
- automatic and manually adjudicated outcomes.

Do not claim strict equivalence merely because a small repeated evaluation observes no discordance. Treat approximate
Wald intervals from sparse paired Bernoulli results as diagnostics, not universal proof.

## Failure adjudication

Review every automatic failure. Randomize videos across model and repeat, hide source identity, and judge the final
task semantics rather than trajectory smoothness. Keep an auditable correction list and report automatic plus corrected
SR side by side.

Prefer an automatic geometry metric that encodes correct target, local-frame footprint containment, height, release,
and final stability. Freeze it before candidate evaluation.

## Interpretation

- Positive net SR with no critical regression: retain checkpoint and collect its remaining failures.
- Flat SR with many flips: capability transfer; inspect improved/regressed failure types and data concentration.
- Negative result: stop adding steps; verify label semantics, parent stats, checkpoint sensitivity, and conflicts.
- Offline imitation improvement alone: useful screening evidence, never sufficient for deployment.
