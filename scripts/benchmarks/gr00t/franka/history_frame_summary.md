# Franka GR00T History-Frame Summary

## Meaning

History frame means the model sees an older camera observation together with the current camera observation.

For this setup:

```text
video.delta_indices = [-16, 0]
```

Each model input contains:

```text
t-16 video frame
t current video frame
```

This gives GR00T temporal context from one action chunk earlier. The test below was run only for the Franka
joint-space policy.

## What Changes

For the history-frame experiment, only video history was changed:

```text
video.delta_indices = [-16, 0]
state.delta_indices = [0]
action.delta_indices = list(range(16))
language.delta_indices = [0]
```

State remains current-only. Action output remains a 16-step chunk.

## Current Decision

Joint-space history-frame training with `video.delta_indices = [-16, 0]` was tested and closed-loop SR dropped to about
35%. The likely issue is that the model sees historical video but only current joint state:

```text
video: [t-16, t]
state: [t]
```

For joint-space control this can make the old robot pose in the image conflict with the current joint state. The
joint-space modality config was therefore reverted to single-frame video:

```text
video.delta_indices = [0]
state.delta_indices = [0]
action.delta_indices = list(range(16))
```

Keep the `[-16, 0]` result as an archived joint-space experiment. Possible future joint-space tests include shorter
history `[-4, 0]`, `[-8, 0]`, or matching robot-state history.

## Retraining Requirement

Retraining is required.

The model input shape changes from one video frame to two video frames. A checkpoint trained with
`video.delta_indices = [0]` should not be used as a history-frame model.

## Dataset Requirement

The LeRobot dataset does not need to be regenerated only for history frames.

The videos already contain all frames. The history behavior is controlled by the GR00T modality config, not by rewriting
the dataset.

Use the joint-space dataset:

```text
lerobot_joint_space
```

## Required Files

Joint-space config:

```text
scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py
```

GR00T training padding patch:

```text
Isaac-GR00T/gr00t/experiment/launch_finetune.py
```

The padding patch is needed because `-16` is a negative delta index. Early frames in each episode must be padded instead
of indexing incorrectly.

## Training Sample Layout

For each training sample at time `t`, GR00T loads:

```text
video:  [t-16, t]
state:  [t]
action: [t, t+1, ..., t+15]
```

For joint-space training:

```text
franka_joint_pos
franka_gripper_width
```

are current-frame state only.

This setup is video history only, not robot-state history.

## Inference Requirement

For history-frame checkpoints, run the IsaacLab GR00T client with:

```text
--video-history-frames 2
```

Joint-space inference uses:

```text
--policy-type joint_space
--task Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0
--video-history-frames 2
```

For reverted single-frame joint-space checkpoints, use:

```text
--policy-type joint_space
--task Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0
--video-history-frames 1
```

or omit `--video-history-frames`, because the joint-space client default is single-frame.

## Training Speed Notes

At the start of training, this log is normal:

```text
Rank 0, Worker X: Wait for shard ...
Rank 0, Worker X: Caching shard...
```

This means the dataloader workers are warming the dataset shard cache. Early iteration time can be slower. Wait until
around `200-500` steps before judging final speed.

History-frame training is slower than single-frame training because video input changes from:

```text
[0] -> 1 frame
```

to:

```text
[-16, 0] -> 2 frames
```

With two cameras, this changes video loading/processing from 2 images per sample to 4 images per sample.

If the progress bar shows:

```text
2.7s/it
```

that means seconds per iteration, not milliseconds.

The warning below is not fatal:

```text
Could not estimate the number of tokens of the input, floating-point operations will not be computed
```

Training still runs normally; only FLOPs reporting is skipped.

To check whether training is GPU-bound or dataloader-bound:

```bash
watch -n 1 nvidia-smi
```

If GPU utilization is high, keep training. If GPU utilization is often low and iteration time remains slow after cache
warmup, restart with more dataloader workers:

```bash
--dataloader-num-workers 8
```

The original command used:

```bash
--dataloader-num-workers 4
```
