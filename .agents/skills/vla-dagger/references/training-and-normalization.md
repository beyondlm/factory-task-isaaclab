# Training and normalization

## First-pass recipe

1. Start from the evaluated base model weights.
2. Aggregate the base dataset with every accepted recovery source once.
3. Keep the base action representation, modality config, and normalizer unchanged.
4. Use a fresh optimizer when the parent checkpoint has only model weights.
5. Save intermediate checkpoints; predeclare the primary evaluation checkpoint.
6. Evaluate closed loop before changing sampling, history, horizon, or representation.

This isolates the operational DAgger stage: additional recovery data plus its SFT.

## Normalization decision table

| Situation | Statistics |
| --- | --- |
| Incremental SFT, unchanged representation | Freeze the parent/base statistics |
| Base and recovery built together from scratch | Generate one shared set after conversion |
| Continuous-to-binary or other action-space change | Rebuild both datasets and regenerate one shared set |
| Comparing two data mixtures | Use the same normalizer unless normalization is the variable under test |

Dump the effective normalizer before training and compare hashes. Verify the output checkpoint carries the intended
statistics before inference. Missing or silently regenerated stats invalidate the comparison.

## Dataset exposure

Begin with natural unique-data exposure using `--all-unique-dagger`. Report actual frame fraction and selection mode
from `meta/dagger_mix.json`, plus independent source count. Do not use `--allow-repeat` in the first pass. If a measured result suggests dilution, test one sampling
change at a time:

1. source/segment balancing;
2. modest recovery exposure change;
3. step-level masks/weights;
4. critic-based weighting only when reward/Q infrastructure is justified.

Oversampling cannot create missing state coverage.

## Training budget

The Franka reference evaluated a 5k SFT checkpoint, but step count does not transfer directly across dataset sizes.
Save early checkpoints such as 1k and 2k. Use open-loop fit only as a forgetting/label gate; select conclusions from
closed-loop behavior, not minimum validation loss.

## Optimizer state

Model-only checkpoints cannot resume Adam moments or scheduler position. That is acceptable for a new DAgger SFT
stage: initialize a fresh optimizer, use explicit warmup/LR, and apply identical settings to any matched arms. Preserve
full trainer state in future runs only when true interruption/resume behavior is required.
