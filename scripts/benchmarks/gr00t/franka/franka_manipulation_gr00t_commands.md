# Franka Manipulation GR00T Commands

Set customer-specific paths first:

```bash
export ISAACLAB=/path/to/IsaacLab
export GROOT=/path/to/Isaac-GR00T
export LEROBOT=/path/to/lerobot
export DATA_ROOT=/path/to/data
export CHECKPOINT_ROOT=/path/to/checkpoints
export LOCAL_GROOT_WORKDIR=/path/to/local/gr00t_workdir
export FRANKA_SORTING_ASSET_DIR=/path/to/franka_sorting_assets
```

Run from the IsaacLab root:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta
```

Record 100 drop-and-settle demos:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --viz kit \
  --dataset_file ./datasets/franka_sorting_drop_settle_100.hdf5 \
  --num_demos 100 \
  --teleop_device spacemouse \
  --num_success_steps 30
```

Record 100 longer completed-action demos:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --viz kit \
  --dataset_file ./datasets/franka_sorting_completed_tail_100.hdf5 \
  --num_demos 100 \
  --teleop_device spacemouse \
  --num_success_steps 60
```

Generate replay camera videos from the recorded HDF5:

```bash
./isaaclab.sh -p scripts/benchmarks/gr00t/franka/replay_demos_with_camera.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0 \
  --dataset_file datasets/dataset_sorting_105.hdf5 \
  --video \
  --camera_view_list wrist_camera table_camera \
  --video-output-dir datasets/dataset_sorting_105/generated_videos \
  --validate_success_rate \
  --viz none
```

Convert HDF5 plus replay videos to GR00T-LeRobot v2:

```bash
python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_task_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5 \
  --video-dir datasets/dataset_sorting_105/generated_videos \
  --require-videos \
  --overwrite
```

## Teleop HDF5 To LeRobot Representations

The Franka demos are collected with `Isaac-Pick-Place-Franka-IK-Rel-v0`, so the original teleop command in the HDF5 is EEF/task-space relative:

```text
actions = [dx, dy, dz, d_rx, d_ry, d_rz, gripper_cmd]
```

The same HDF5 also records the robot state trajectory, so it can be converted into different LeRobot action/state spaces:

```text
Task-space relative EEF LeRobot:
  state  <- obs/eef_pos + obs/eef_quat + obs/gripper_pos
  action <- actions

Joint-space absolute LeRobot:
  state[t]  <- states/articulation/robot/joint_position[t]
  action[t] <- states/articulation/robot/joint_position[t + 1]
```

For Franka joint-space conversion, the two finger joints are compressed to one `gripper_width` value:

```text
[panda_joint1..panda_joint7, gripper_width]
```

Important GR00T config detail: `ActionRepresentation.ABSOLUTE` means the model processor should use the stored action vector as a direct target. It does not necessarily mean the original teleop command was absolute. Our current task-space converter stores raw relative EEF deltas and marks them as absolute in the GR00T config so GR00T does not convert them again. For the joint-space config, the arm action can use `RELATIVE` because the parquet stores absolute next-joint targets and GR00T can compute relative joint deltas during preprocessing.

Convert the same HDF5 plus replay videos to joint-space GR00T-LeRobot v2:

```bash
python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_joint_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5 \
  --video-dir datasets/dataset_sorting_105/generated_videos \
  --require-videos \
  --overwrite
```

Optional one-episode video smoke test:

```bash
./isaaclab.sh -p scripts/benchmarks/gr00t/franka/replay_demos_with_camera.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0 \
  --dataset_file datasets/dataset_sorting_105.hdf5 \
  --select_episodes 0 \
  --video \
  --camera_view_list wrist_camera table_camera \
  --video-output-dir /tmp/franka_replay_video_test \
  --viz none
```

## GR00T Training Reference

Task-space dataset sanity check:

```text
dataset size: 510M
parquet files: 105
mp4 files: 210
episodes: 105
frames: 60432
videos: 210
fps: 30
robot_type: franka_pick_place_relative_task_space
```

20k-step task-space run command:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_task_space"
CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_modality_config.py"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG" \
  --num-gpus 1 \
  --output-dir "$CHECKPOINT_ROOT/franka_gr00t_bs256_20000" \
  --save-total-limit 3 \
  --save-steps 5000 \
  --max-steps 20000 \
  --global-batch-size 256 \
  --dataloader-num-workers 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

Notes:

- `--color-jitter-params` must use space-separated key/value pairs, not `brightness=0.3` syntax.
- If Hugging Face access fails, verify both `nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B`.

## Joint-Space Conversion

Convert the same HDF5 plus replay videos to joint-space GR00T-LeRobot v2:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta

python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_joint_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5 \
  --video-dir datasets/dataset_sorting_105/generated_videos \
  --require-videos \
  --overwrite
```

Output:

```text
datasets/dataset_sorting_105/lerobot_joint_space
```

Generate missing GR00T dataset statistics:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_joint_space"
CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py"

NO_ALBUMENTATIONS_UPDATE=1 \
uv run python gr00t/data/stats.py \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG"
```

## Joint-Space Training

Dataset/config:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_joint_space"
CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py"
```

Start or rerun 20k-step joint-space training:

```bash
NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG" \
  --num-gpus 1 \
  --output-dir "$CHECKPOINT_ROOT/franka_joint_gr00t_bs256_20000" \
  --save-total-limit 3 \
  --save-steps 5000 \
  --max-steps 20000 \
  --global-batch-size 256 \
  --dataloader-num-workers 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

True resume from checkpoint 10000 to 20000:

```bash
SRC="$CHECKPOINT_ROOT/franka_joint_gr00t_bs256_20000/checkpoint-10000"
OUT="$CHECKPOINT_ROOT/franka_joint_gr00t_resume_10000_to_20000"

mkdir -p "$OUT"
cp -al "$SRC" "$OUT/checkpoint-10000"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG" \
  --num-gpus 1 \
  --output-dir "$OUT" \
  --save-total-limit 3 \
  --save-steps 5000 \
  --max-steps 20000 \
  --global-batch-size 256 \
  --dataloader-num-workers 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

`--base-model-path checkpoint-10000` only loads weights and starts a fresh optimizer from step 0. True resume needs `checkpoint-10000` under `--output-dir`.

## Open-Loop Evaluation

Task-space checkpoint:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_task_space"
CKPT="$CHECKPOINT_ROOT/franka_gr00t_bs256_20000/checkpoint-10000"
OUT="$LOCAL_GROOT_WORKDIR/open_loop_franka_eef_10000_traj0.jpeg"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/open_loop_eval.py \
  --model-path "$CKPT" \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 \
  --action-horizon 16 \
  --steps 400 \
  --modality-keys franka_eef_delta_pos franka_eef_delta_rot franka_gripper_cmd \
  --save-plot-path "$OUT"
```

Joint-space checkpoint:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_joint_space"
CKPT="$CHECKPOINT_ROOT/franka_joint_gr00t_bs256_20000/checkpoint-10000"
OUT="$LOCAL_GROOT_WORKDIR/open_loop_franka_joint_10000_traj0.jpeg"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/open_loop_eval.py \
  --model-path "$CKPT" \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 \
  --action-horizon 16 \
  --steps 400 \
  --modality-keys franka_joint_pos franka_gripper_width \
  --save-plot-path "$OUT"
```

## Closed-Loop Evaluation

Start GR00T server for task-space checkpoint:

```bash
cd "$GROOT"

CKPT="$CHECKPOINT_ROOT/franka_gr00t_bs256_20000/checkpoint-10000"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Run task-space client:

```bash
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
  --num-feedback-actions 16
```

Start GR00T server for joint-space checkpoint:

```bash
cd "$GROOT"

CKPT="$CHECKPOINT_ROOT/franka_joint_gr00t_bs256_20000/checkpoint-10000"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$CKPT" \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Run joint-space client:

```bash
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
  --num-total-experiments 20 \
  --max-inference-steps 62 \
  --num-feedback-actions 8 \
  --debug \
  --pause-on-error
```

Add `--headless` to the client command if GUI is not needed.
