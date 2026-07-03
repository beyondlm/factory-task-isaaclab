# Brev Franka GR00T Current-Frame / Dynamic Action-Horizon Training Commands

This file collects the copy and training commands for the Franka GR00T policies on Brev.

The task-space and joint-space configs use the current camera frame only:

```text
video.delta_indices = [0]
```

The action output horizon defaults to 32 and can be changed at runtime:

```text
FRANKA_GROOT_ACTION_HORIZON=32
action.delta_indices = list(range(FRANKA_GROOT_ACTION_HORIZON))
```

Use the same `FRANKA_GROOT_ACTION_HORIZON` value for stats, training, open-loop evaluation, and closed-loop client
feedback actions. The LeRobot dataset does not need to be regenerated for a new action horizon. GR00T reads the future
action chunk from the existing per-frame actions.

## Local To Brev Copy

Run on the local host:

```bash
export BREV_TARGET=agi-test
export LOCAL_WORKSPACE=/path/to/workspace
export LOCAL_ISAACLAB=$LOCAL_WORKSPACE/projects/isaaclab_3_beta/IsaacLab
export LOCAL_CHECKPOINT_ROOT=$LOCAL_WORKSPACE/tmp_gr00t/brev_checkpoints
export LOCAL_GROOT=$LOCAL_WORKSPACE/Isaac-GR00T
export BREV_GROOT=/home/ubuntu/workspace/Isaac-GR00T
export BREV_CONFIG_DIR=$BREV_GROOT/examples/franka

brev exec "$BREV_TARGET" "mkdir -p $BREV_CONFIG_DIR"

brev copy \
  "$LOCAL_ISAACLAB/scripts/benchmarks/gr00t/franka/franka_modality_config.py" \
  "$BREV_TARGET:$BREV_CONFIG_DIR/franka_modality_config.py"

brev copy \
  "$LOCAL_ISAACLAB/scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py" \
  "$BREV_TARGET:$BREV_CONFIG_DIR/franka_joint_modality_config.py"

brev copy \
  "$LOCAL_ISAACLAB/scripts/benchmarks/gr00t/franka/data_config.py" \
  "$BREV_TARGET:$BREV_CONFIG_DIR/data_config.py"

brev copy \
  "$LOCAL_GROOT/gr00t/experiment/launch_finetune.py" \
  "$BREV_TARGET:$BREV_GROOT/gr00t/experiment/launch_finetune.py"
```

`gr00t_inference_client_franka.py` is an IsaacLab-side client and is not needed on Brev unless IsaacLab evaluation is
also being run there.

Verify the copied files on Brev:

```bash
brev exec "$BREV_TARGET" "grep -n 'FRANKA_GROOT_ACTION_HORIZON' $BREV_CONFIG_DIR/franka_modality_config.py && \
grep -n 'FRANKA_GROOT_ACTION_HORIZON' $BREV_CONFIG_DIR/franka_joint_modality_config.py && \
grep -n 'FRANKA_GROOT_ACTION_HORIZON' $BREV_CONFIG_DIR/data_config.py && \
grep -n 'allow_padding=True' $BREV_GROOT/gr00t/experiment/launch_finetune.py && \
echo brev_franka_dynamic_horizon_files_ok"
```

## Task-Space Dataset Copy

Run on the local host. Skip this if the dataset already exists at `BREV_DATASET`.

```bash
export BREV_TARGET=agi-test
export LOCAL_DATASET=$LOCAL_ISAACLAB/datasets/dataset_sorting_201_20260612_replay_success/lerobot_task_space
export BREV_DATASET=/home/ubuntu/workspace/data/franka_sorting_201_20260612_replay_success/lerobot_task_space

brev exec "$BREV_TARGET" "mkdir -p $(dirname $BREV_DATASET)"
brev copy "$LOCAL_DATASET/" "$BREV_TARGET:$BREV_DATASET/"

brev exec "$BREV_TARGET" "test -f $BREV_DATASET/meta/info.json && echo task_dataset_ok"
```

## Task-Space Training On Brev

Enter Brev:

```bash
brev shell agi-test
```

Run inside Brev:

```bash
export GROOT=/home/ubuntu/workspace/Isaac-GR00T
export DATASET=/home/ubuntu/workspace/data/franka_sorting_201_20260612_replay_success/lerobot_task_space
export CONFIG=/home/ubuntu/workspace/Isaac-GR00T/examples/franka/franka_modality_config.py
export CHECKPOINT_ROOT=/home/ubuntu/workspace/checkpoints
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON

cd "$GROOT"

NO_ALBUMENTATIONS_UPDATE=1 \
uv run python gr00t/data/stats.py \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG" \
  --num-gpus 1 \
  --output-dir "$CHECKPOINT_ROOT/franka_eef_201_h${ACTION_HORIZON}_gr00t_bs256_20000" \
  --save-total-limit 3 \
  --save-steps 5000 \
  --max-steps 20000 \
  --global-batch-size 256 \
  --dataloader-num-workers 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

## Copy Task-Space Checkpoint Back To Host

Run on the local host after training finishes:

```bash
export BREV_TARGET=agi-test
export ACTION_HORIZON=32
export BREV_CKPT=/home/ubuntu/workspace/checkpoints/franka_eef_201_h${ACTION_HORIZON}_gr00t_bs256_20000
export LOCAL_CKPT=$LOCAL_CHECKPOINT_ROOT/franka_eef_201_h${ACTION_HORIZON}_gr00t_bs256_20000

mkdir -p "$(dirname "$LOCAL_CKPT")"
brev copy "$BREV_TARGET:$BREV_CKPT/" "$LOCAL_CKPT/"
```

## Host Task-Space Inference After Brev Training

Start the GR00T server on the host with the copied checkpoint:

```bash
export GROOT=$LOCAL_GROOT
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON
export CKPT=$LOCAL_CHECKPOINT_ROOT/franka_eef_201_h${ACTION_HORIZON}_gr00t_bs256_20000/checkpoint-20000

cd "$GROOT"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Run the host IsaacLab client:

```bash
export ISAACLAB=$LOCAL_ISAACLAB
export FRANKA_SORTING_ASSET_DIR=/path/to/franka_sorting_assets
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON

cd "$ISAACLAB"
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate isaaclab3_beta

./isaaclab.sh -p scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py \
  --policy-type task_space \
  --task Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0 \
  --server-host localhost \
  --server-port 5555 \
  --language-instruction "Pick up the labeled box and place it into the blue bin. Pick up the unlabeled box and place it into the black bin." \
  --num-total-experiments 10 \
  --max-inference-steps 62 \
  --num-feedback-actions "$ACTION_HORIZON" \
  --video-history-frames 1
```

## Joint-Space Dataset Copy

Run on the local host. Skip this if the dataset already exists at `BREV_DATASET`.

```bash
export BREV_TARGET=agi-test
export LOCAL_DATASET=$LOCAL_ISAACLAB/datasets/dataset_sorting_201_20260612_replay_success/lerobot_joint_space
export BREV_DATASET=/home/ubuntu/workspace/data/franka_sorting_201_20260612_replay_success/lerobot_joint_space

brev exec "$BREV_TARGET" "mkdir -p $(dirname $BREV_DATASET)"
brev copy "$LOCAL_DATASET/" "$BREV_TARGET:$BREV_DATASET/"

brev exec "$BREV_TARGET" "test -f $BREV_DATASET/meta/info.json && echo joint_dataset_ok"
```

## Joint-Space Training On Brev

The current joint-space config uses single-frame video and a runtime action horizon:

```text
video.delta_indices = [0]
state.delta_indices = [0]
action.delta_indices = list(range(FRANKA_GROOT_ACTION_HORIZON))
```

Enter Brev:

```bash
brev shell agi-test
```

Run inside Brev:

```bash
export GROOT=/home/ubuntu/workspace/Isaac-GR00T
export DATASET=/home/ubuntu/workspace/data/franka_sorting_201_20260612_replay_success/lerobot_joint_space
export CONFIG=/home/ubuntu/workspace/Isaac-GR00T/examples/franka/franka_joint_modality_config.py
export CHECKPOINT_ROOT=/home/ubuntu/workspace/checkpoints
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON

cd "$GROOT"

NO_ALBUMENTATIONS_UPDATE=1 \
uv run python gr00t/data/stats.py \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG" \
  --num-gpus 1 \
  --output-dir "$CHECKPOINT_ROOT/franka_joint_201_h${ACTION_HORIZON}_gr00t_bs256_20000" \
  --save-total-limit 3 \
  --save-steps 5000 \
  --max-steps 20000 \
  --global-batch-size 256 \
  --dataloader-num-workers 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

## Copy Joint-Space Checkpoint Back To Host

Run on the local host after training finishes:

```bash
export BREV_TARGET=agi-test
export ACTION_HORIZON=32
export BREV_CKPT=/home/ubuntu/workspace/checkpoints/franka_joint_201_h${ACTION_HORIZON}_gr00t_bs256_20000
export LOCAL_CKPT=$LOCAL_CHECKPOINT_ROOT/franka_joint_201_h${ACTION_HORIZON}_gr00t_bs256_20000

mkdir -p "$(dirname "$LOCAL_CKPT")"
brev copy "$BREV_TARGET:$BREV_CKPT/" "$LOCAL_CKPT/"
```

## Host Joint-Space Inference After Brev Training

Start the joint-space GR00T server on the host:

```bash
export GROOT=$LOCAL_GROOT
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON
export CKPT=$LOCAL_CHECKPOINT_ROOT/franka_joint_201_h${ACTION_HORIZON}_gr00t_bs256_20000/checkpoint-20000

cd "$GROOT"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Run the host IsaacLab joint-space client:

```bash
export ISAACLAB=$LOCAL_ISAACLAB
export FRANKA_SORTING_ASSET_DIR=/path/to/franka_sorting_assets
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON

cd "$ISAACLAB"
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate isaaclab3_beta

./isaaclab.sh -p scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py \
  --policy-type joint_space \
  --task Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0 \
  --server-host localhost \
  --server-port 5555 \
  --language-instruction "Pick up the labeled box and place it into the blue bin. Pick up the unlabeled box and place it into the black bin." \
  --num-total-experiments 10 \
  --max-inference-steps 62 \
  --num-feedback-actions "$ACTION_HORIZON" \
  --video-history-frames 1
```
