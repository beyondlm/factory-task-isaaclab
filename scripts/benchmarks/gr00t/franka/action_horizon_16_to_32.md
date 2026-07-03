# Updating GR00T Action Horizon From 16 To 32

The active Franka GR00T IsaacLab workflow uses a 32-step action chunk by default. This is an action horizon change, not a robot action-dimension change: each predicted step still contains the same action fields, but GR00T predicts 32 future steps instead of 16.

## Active Code Defaults

The active IsaacLab files read `FRANKA_GROOT_ACTION_HORIZON` and default to 32:

```bash
export FRANKA_GROOT_ACTION_HORIZON=32
```

This controls:

- `scripts/benchmarks/gr00t/franka/franka_modality_config.py`
- `scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py`
- `scripts/benchmarks/gr00t/franka/data_config.py`
- `scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py`

The effective modality action indices are:

```python
delta_indices=list(range(FRANKA_GROOT_ACTION_HORIZON))
```

## Required Regeneration

After changing the action horizon, regenerate GR00T dataset statistics with the same modality config that will be used for training:

```bash
cd "$GROOT"
export FRANKA_GROOT_ACTION_HORIZON=32

CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_modality_config.py"

uv run python gr00t/data/stats.py \
  --dataset-path "$ISAACLAB/datasets/dataset_sorting_105/lerobot_task_space" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG"
```

Use `franka_joint_modality_config.py` for joint-space datasets.

## Training And Evaluation

Train or fine-tune with the same exported horizon:

```bash
export FRANKA_GROOT_ACTION_HORIZON=32
```

Do not mix a checkpoint trained with 16-step actions with a 32-step config unless the checkpoint was explicitly trained or resumed with compatible action heads and regenerated stats.

Open-loop evaluation should use:

```bash
--action-horizon "$FRANKA_GROOT_ACTION_HORIZON"
```

Closed-loop IsaacLab evaluation should use:

```bash
--num-feedback-actions "$FRANKA_GROOT_ACTION_HORIZON"
```

If latency or stop-and-go behavior is still high, keep the model horizon at 32 and tune how many client-side feedback actions are executed per inference cycle separately.
