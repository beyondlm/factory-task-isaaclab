# Franka GR00T State-History H32 Local Eval Commands

This is the local reference for the joint-space state-history experiment:

```text
state.delta_indices = [-2, -1, 0]
action_horizon = 32
```

Use the same state history and action horizon for stats, training, open-loop eval, and closed-loop inference.

Set these paths for your machine before running the commands:

```bash
export WORKSPACE_ROOT=/path/to/workspace
export ISAACLAB_ROOT=$WORKSPACE_ROOT/projects/isaaclab_3_beta/IsaacLab
export GROOT_ROOT=$WORKSPACE_ROOT/Isaac-GR00T
export FRANKA_SORTING_ASSET_DIR=/path/to/franka_sorting_assets
```

## Implementation Details

State history is controlled by environment variables instead of hard-coding a separate config for every experiment:

```bash
export FRANKA_GROOT_ACTION_HORIZON=32
export FRANKA_GROOT_STATE_DELTA_INDICES=-2,-1,0
```

The Franka modality configs read these variables at import time:

```text
scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py
scripts/benchmarks/gr00t/franka/franka_modality_config.py
scripts/benchmarks/gr00t/franka/data_config.py
```

For this experiment, the joint-space config resolves to:

```text
video.delta_indices = [0]
state.delta_indices = [-2, -1, 0]
action.delta_indices = list(range(32))
language.delta_indices = [0]
```

`video.delta_indices` intentionally stays `[0]`, so this is state history only. It does not reuse the older
history-frame experiment that used temporal camera frames.

The GR00T finetune launcher also needs the local state-history patch:

```text
<Isaac-GR00T>/gr00t/experiment/launch_finetune.py
<Isaac-GR00T>/gr00t/model/gr00t_n1d7/setup.py
```

Those changes do two things:

```text
1. Enable data.allow_padding=True when negative delta_indices are present.
2. Set model.state_history_length from len(state.delta_indices), so [-2, -1, 0] becomes 3.
```

Without this model-side change, training fails at:

```text
assert action_input.state.shape[1] == self.config.state_history_length
```

Closed-loop inference is implemented in:

```text
scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py
```

The client uses `StateHistoryBuffer` to append the current policy state after reset and after every `env.step`.
For `--state-history-frames 3`, early rollout steps are padded with the first state until three frames are available.
The state-history frame count must match the training config:

```bash
--state-history-frames 3
```

## Checkpoint

Expected local checkpoint copied back from Brev:

```bash
export RUN_NAME=franka_joint_201_statehist3_h32_gr00t_bs256_20000
export CKPT=$WORKSPACE_ROOT/tmp_gr00t/brev_checkpoints/$RUN_NAME/checkpoint-20000
```

## Dataset Paths

Use this dataset when evaluating the model trained on the 201 replay-success joint-space dataset:

```bash
export DATASET=$ISAACLAB_ROOT/datasets/dataset_sorting_201_20260612_replay_success/lerobot_joint_space
```

Optional 210 replay-success joint-space dataset:

```bash
export DATASET=$ISAACLAB_ROOT/datasets/dataset_sorting_210_20260612_replay_success/lerobot_joint_space
```

## Open-Loop Eval

```bash
export RUN_NAME=franka_joint_201_statehist3_h32_gr00t_bs256_20000
export DATASET=$ISAACLAB_ROOT/datasets/dataset_sorting_201_20260612_replay_success/lerobot_joint_space
export CKPT=$WORKSPACE_ROOT/tmp_gr00t/brev_checkpoints/$RUN_NAME/checkpoint-20000
export OUT=$WORKSPACE_ROOT/tmp_gr00t/open_loop_franka_joint_statehist3_h32_20000_traj0.jpeg

export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON
export FRANKA_GROOT_STATE_DELTA_INDICES=-2,-1,0

cd "$GROOT_ROOT"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/open_loop_eval.py \
  --model-path "$CKPT" \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 \
  --action-horizon "$ACTION_HORIZON" \
  --steps 400 \
  --modality-keys franka_joint_pos franka_gripper_width \
  --save-plot-path "$OUT"
```

## Copy Saved JPEG To Downloads

```bash
mkdir -p ~/Downloads

cp "$OUT" ~/Downloads/

ls -lh ~/Downloads/open_loop_franka_joint_statehist3_h32_20000_traj0.jpeg
```

## Closed-Loop Test

Start the GR00T server:

```bash
export RUN_NAME=franka_joint_201_statehist3_h32_gr00t_bs256_20000
export CKPT=$WORKSPACE_ROOT/tmp_gr00t/brev_checkpoints/$RUN_NAME/checkpoint-20000

export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON
export FRANKA_GROOT_STATE_DELTA_INDICES=-2,-1,0

cd "$GROOT_ROOT"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Run the IsaacLab client in another terminal:

```bash
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON
export FRANKA_GROOT_STATE_DELTA_INDICES=-2,-1,0

cd "$ISAACLAB_ROOT"
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate isaaclab3_beta

./isaaclab.sh -p scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py \
  --policy-type joint_space \
  --task Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0 \
  --server-host localhost \
  --server-port 5555 \
  --language-instruction "Pick up the labeled box and place it into the blue bin. Pick up the unlabeled box and place it into the black bin." \
  --num-total-experiments 20 \
  --max-inference-steps 62 \
  --num-feedback-actions "$ACTION_HORIZON" \
  --video-history-frames 1 \
  --state-history-frames 3
```
