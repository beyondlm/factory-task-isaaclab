# Updating GR00T Action Horizon From 16 To 32

This project now uses a dynamic GR00T action chunk for Franka closed-loop evaluation. The default is 32 steps, and it
can be changed with `FRANKA_GROOT_ACTION_HORIZON`. The change is an action horizon change, not a robot
action-dimension change: each step still contains the same action fields, but GR00T predicts a configurable number of
future steps.

## Code Changes

Set one action horizon before generating stats, training, serving, and evaluation:

```bash
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON
```

The Franka modality/data configs read the env var and build action indices dynamically:

```python
ACTION_HORIZON = _positive_int_from_env("FRANKA_GROOT_ACTION_HORIZON", 32)
delta_indices = list(range(ACTION_HORIZON))
```

or, for the older data-config class:

```python
action_indices = list(range(ACTION_HORIZON))
```

The overlay applies this in:

- `scripts/benchmarks/gr00t/franka/franka_modality_config.py`
- `scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py`
- `scripts/benchmarks/gr00t/franka/data_config.py`

The IsaacLab closed-loop client default was also updated:

```bash
--num-feedback-actions "$ACTION_HORIZON"
```

## Required Regeneration

After changing the action horizon, regenerate GR00T dataset statistics with the same modality config that will be used for training:

```bash
cd "$GROOT"

export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON

CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_modality_config.py"

uv run python gr00t/data/stats.py \
  --dataset-path "$ISAACLAB/datasets/dataset_sorting_105/lerobot_task_space" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG"
```

Use `franka_joint_modality_config.py` for joint-space datasets.

## Training And Evaluation

Train or fine-tune with the same action horizon used for statistics. Do not mix a checkpoint trained with 16-step
actions with a 32-step config unless the checkpoint was explicitly resumed with compatible model/action heads.

Open-loop evaluation should use:

```bash
--action-horizon "$ACTION_HORIZON"
```

Closed-loop IsaacLab evaluation should use:

```bash
--num-feedback-actions "$ACTION_HORIZON"
```

If latency or stop-and-go behavior is still high, keep the model horizon at 32 and tune the client-side number of executed feedback actions separately instead of retraining back to 16.
