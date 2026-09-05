# Franka Factory Sorting GR00T Benchmark And Commands

This guide is the Franka benchmark and command reference for the IsaacLab 3 Beta factory sorting overlay.
It covers demo replay, LeRobot v2 conversion, GR00T training data preparation, open-loop evaluation, and closed-loop
GR00T server/client tests.

The Franka policy pipeline is based on
[NVIDIA Isaac-GR00T N1.7 release](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release).

## Path Variables

Set customer-specific paths first:

```bash
export WORKSPACE=/path/to/workspace
export ISAACLAB=/path/to/IsaacLab
export GROOT=/path/to/Isaac-GR00T
export LEROBOT=/path/to/lerobot
export DATA_ROOT=/path/to/data
export CHECKPOINT_ROOT=/path/to/checkpoints
export LOCAL_GROOT_WORKDIR=/path/to/local/gr00t_workdir
export FRANKA_SORTING_ASSET_DIR=/path/to/franka_sorting_assets
export JOINT_H32_BASELINE_CKPT=/path/to/joint_h32_baseline/checkpoint-20000
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON
```

`FRANKA_SORTING_ASSET_DIR` should point to the USD asset root for the factory Franka, belt, boxes, and bins.

## Benchmark Scope

Main teleoperation task:

```text
Isaac-Pick-Place-Franka-IK-Rel-v0
```

Replay camera task:

```text
Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0
```

Joint-position replay camera task:

```text
Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0
```

Sorting target mapping:

```text
object_a = box_3_label -> blue bin  = sorting_bin_blue
object_b = box_4_no    -> black bin = black_sorting_bin
```

Historical closed-loop benchmark summary for the previous 16-step action horizon:

The 10k rows use 105 episodes; the 20k rows use 201 episodes.


| Policy action space             | GR00T N1.7 modality config                      | Training dataset | Training setup | Batch size | Closed-loop SR | Failure notes                                                                                                                                                                                                             |
| ------------------------------- | ----------------------------------------------- | ---------------- | -------------- | ---------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EEF / IK-relative               | [EEF config](franka_modality_config.py)         | 105 episodes     | 10k steps      | 256        | 65%            | 20 trials: 1: OOD pick failure, 7 times.                                                                                                                                                                                  |
| EEF / IK-relative               | [EEF config](franka_modality_config.py)         | 201 episodes     | 20k steps      | 256        | 100%           | 20 trials: no failures.                                                                                                                                                                                                   |
| Joint space                     | [Joint config](franka_joint_modality_config.py) | 105 episodes     | 10k steps      | 256        | 30%            | 20 trials: 1: mixed pick/place/OOD failures, 14 times.                                                                                                                                                                    |
| Joint space                     | [Joint config](franka_joint_modality_config.py) | 201 episodes     | 20k steps      | 256        | 50%            | 20 trials: 1: OOD, 3 times. 2: near box, but no gripper close, 7 times.                                                                                                                                                   |
| Joint space + action horizon 32 | [Joint config](franka_joint_modality_config.py) | 201 episodes     | 20k steps      | 256        | 65%            | 20 trials, 13 successes / 7 failures: 1: grasp hesitation above the box; after placing the first box, the pose for the second box becomes abnormal, 6 times. 2: perception failure; gripper closes above the box, 1 time. |


For new reportable numbers, rerun the closed-loop client with a fixed checkpoint, fixed seed/task setup, and
`--num-total-experiments` set to the desired trial count.

For the 16-to-32 action-horizon migration, see [action_horizon_16_to_32.md](action_horizon_16_to_32.md).

## Task Setup And Success Criteria

Main task files:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/joint_pos_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/ik_rel_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/mdp/terminations.py
```

Scene and control details:

- Franka is positioned closer to the blue/black bins for easier manipulation.
- Belt/table scene scale is controlled by `SORTING_SCENE_SCALE` and `SORTING_BELT_SCALE`.
- Proxy belt/table colliders are sized from the same scene scale.
- `table_camera` position derives from `_belt_scaled_pos(FRANKA_TABLE_CAM_LOCAL_POS)`, so changing
`SORTING_BELT_SCALE` moves the camera consistently.
- Replay cameras use `CameraCfg.OffsetCfg(..., convention="opengl")`.
- Wrist camera quaternion is stored in IsaacLab `CameraCfg` order: `(x, y, z, w)`.
- Viewport camera starts closer to the Franka/bin workspace.

Current Franka teleop/control defaults:

```text
FRANKA_IK_ACTION_SCALE=1.0
FRANKA_SPACEMOUSE_POS_SENSITIVITY=0.2
FRANKA_SPACEMOUSE_ROT_SENSITIVITY=0.2
FRANKA_KEYBOARD_POS_SENSITIVITY=0.1
FRANKA_KEYBOARD_ROT_SENSITIVITY=0.1
```

The success logic waits for a real release/drop into the bins:

- `box_3_label` must be in the blue bin.
- `box_4_no` must be in the black bin.
- Success checks both XY distance and a tight bin height threshold.
- The box root must be low in the bin, with signed height difference gated to roughly `[-0.01, 0.035]` m.
- `require_gripper_open=True` prevents ending while still holding the object.
- Both boxes must be dynamically settled with low linear and angular velocity.

Relevant function:

```text
both_boxes_placed_a_into_c_b_into_d(...)
```



## Required Local Setup

Install the teleoperation submodule before running `teleop_se3_agent.py`:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta
./isaaclab.sh -i teleop
```

Verify:

```bash
python -c "import isaaclab_teleop; print(isaaclab_teleop.__file__)"
```

Useful conda envs:

```text
isaaclab3_beta  -> IsaacLab replay, recording, conversion smoke tests
lerobot_v040    -> LeRobot v0.4.0 target environment for later GR00T/LeRobot work
```



## LeRobot V2 Setup

This pipeline writes GR00T-compatible LeRobot v2 datasets. Use the LeRobot package version that matches the GR00T data
pipeline. The current setup uses the Hugging Face LeRobot repo pinned to `v0.4.0`, with conda env name
`lerobot_v040`.

Create the environment:

```bash
conda create -n lerobot_v040 python=3.10 -y
conda activate lerobot_v040
python -m pip install --upgrade pip
```

Clone or update LeRobot:

```bash
mkdir -p "$WORKSPACE/projects"
cd "$WORKSPACE/projects"

git clone https://github.com/huggingface/lerobot.git
cd lerobot
git fetch --tags
git checkout v0.4.0
```

Install LeRobot and conversion dependencies:

```bash
conda activate lerobot_v040
cd "$LEROBOT"

python -m pip install -e .
python -m pip install h5py pandas pyarrow tqdm "imageio[ffmpeg]"
```

Verify:

```bash
conda activate lerobot_v040
cd "$LEROBOT"

git describe --tags --always
python -c "import lerobot; print(lerobot.__file__)"
python -c "import h5py, pandas, pyarrow, tqdm; print('LeRobot v2 conversion deps OK')"
```

Expected tag:

```text
v0.4.0
```

The Franka converters are self-contained and can run from `isaaclab3_beta`, but keeping `lerobot_v040` available is
useful for checking LeRobot data and later GR00T data transfer workflows.

## Teleoperation

Keyboard teleop:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --viz kit
```

SpaceMouse teleop:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device spacemouse \
  --viz kit
```

Faster SpaceMouse run:

```bash
FRANKA_SPACEMOUSE_POS_SENSITIVITY=0.3 \
FRANKA_SPACEMOUSE_ROT_SENSITIVITY=0.3 \
FRANKA_IK_ACTION_SCALE=1.0 \
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device spacemouse \
  --viz kit
```



## Demo Recording

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

`record_demos.py` creates the HDF5 with write mode, so reusing the same `--dataset_file` overwrites the old file
instead of appending.

Dataset used in the examples:

```text
datasets/dataset_sorting_105.hdf5
```

Recorded HDF5 action format:

```text
actions: [dx, dy, dz, d_rx, d_ry, d_rz, gripper_cmd]
```

Replay low-dimensional demos:

```bash
./isaaclab.sh -p scripts/tools/replay_demos.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --dataset_file ./datasets/dataset_sorting_105.hdf5 \
  --validate_success_rate \
  --viz kit
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

Generated LeRobot v2 layout:

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/modality.json
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.wrist_camera/episode_000000.mp4
videos/chunk-000/observation.images.table_camera/episode_000000.mp4
```



## Dataset Statistics

GR00T has two different statistics locations:

```text
dataset/meta/stats.json
dataset/meta/relative_stats.json
checkpoint/statistics.json
```

`meta/stats.json` and `meta/relative_stats.json` are dataset-side files used by training and open-loop dataset loading.
`checkpoint/statistics.json` is checkpoint-side normalization used for inference. A checkpoint can be valid even if the
local dataset copy is missing the two meta files.

Task-space stats:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_task_space"
CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_modality_config.py"

NO_ALBUMENTATIONS_UPDATE=1 \
uv run python gr00t/data/stats.py \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG"
```

Joint-space stats:

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

Verify:

```bash
ls -lh "$DATASET/meta/stats.json" "$DATASET/meta/relative_stats.json"
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
  --output-dir "$CHECKPOINT_ROOT/franka_gr00t_h${ACTION_HORIZON}_bs256_20000" \
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
- Add this data config entry to Isaac-GR00T before training:

```python
"franka_pick_place_relative_task_space": FrankaPickPlaceRelativeTaskSpaceDataConfig(),
```



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

Joint-space sanity check:

```text
dataset size: about 510M
parquet files: 105
mp4 files: 210
episodes: 105
frames: 58857
videos: 210
fps: 30
robot_type: franka_pick_place_joint_space
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
  --output-dir "$CHECKPOINT_ROOT/franka_joint_gr00t_h${ACTION_HORIZON}_bs256_20000" \
  --save-total-limit 3 \
  --save-steps 5000 \
  --max-steps 20000 \
  --global-batch-size 256 \
  --dataloader-num-workers 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

True resume from checkpoint 10000 to 20000:

```bash
SRC="$CHECKPOINT_ROOT/franka_joint_gr00t_h${ACTION_HORIZON}_bs256_20000/checkpoint-10000"
OUT="$CHECKPOINT_ROOT/franka_joint_gr00t_h${ACTION_HORIZON}_resume_10000_to_20000"

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

## Checkpoint Copy And Verification

Inference-only checkpoint files:

```text
config.json
embodiment_id.json
processor_config.json
statistics.json
model.safetensors.index.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
```

Training resume additionally needs:

```text
optimizer.pt
scheduler.pt
rng_state.pth
trainer_state.json
training_args.bin
```

If `safetensors_rust.SafetensorError: incomplete metadata` appears, the safetensors shard is truncated and must be
copied again.

## Open-Loop Evaluation

Task-space checkpoint:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_task_space"
CKPT="$CHECKPOINT_ROOT/franka_gr00t_h${ACTION_HORIZON}_bs256_20000/checkpoint-10000"
OUT="$LOCAL_GROOT_WORKDIR/open_loop_franka_eef_10000_traj0.jpeg"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/eval/open_loop_eval.py \
  --model-path "$CKPT" \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 \
  --action-horizon "$ACTION_HORIZON" \
  --steps 400 \
  --modality-keys franka_eef_delta_pos franka_eef_delta_rot franka_gripper_cmd \
  --save-plot-path "$OUT"
```

The 5k-step EEF/task-space GR00T N1.7 checkpoint tracks the IK-relative delta trajectory in open-loop evaluation:

Franka EEF/task-space open-loop trajectory

Joint-space checkpoint:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_joint_space"
CKPT="$CHECKPOINT_ROOT/franka_joint_gr00t_h${ACTION_HORIZON}_bs256_20000/checkpoint-20000"
OUT="$LOCAL_GROOT_WORKDIR/open_loop_franka_joint_20000_traj0.jpeg"

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

The 20k-step joint-space GR00T N1.7 checkpoint tracks the held-out demonstration trajectory closely in open-loop
evaluation:

Franka joint-space open-loop trajectory

## Closed-Loop Evaluation

Start GR00T server for task-space checkpoint:

```bash
cd "$GROOT"

CKPT="$CHECKPOINT_ROOT/franka_gr00t_h${ACTION_HORIZON}_bs256_20000/checkpoint-10000"

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
  --num-feedback-actions "$ACTION_HORIZON"
```

Start GR00T server for joint-space checkpoint:

```bash
cd "$GROOT"

export CKPT="$JOINT_H32_BASELINE_CKPT"

test -f "$CKPT/config.json"
test -f "$CKPT/processor_config.json"
test -f "$CKPT/model.safetensors.index.json"

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
  --num-feedback-actions "$ACTION_HORIZON"
```

Add `--headless` to the client command if GUI is not needed.

## Joint-Space Closed-Loop Notes

Franka joint-space closed-loop support is implemented in:

```text
scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/ik_rel_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/__init__.py
```

Registered task:

```text
Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0
```

The client sends joint-space observation/action keys:

```text
franka_joint_pos
franka_gripper_width
```

If debug output shows EEF action keys, the GR00T server is using the wrong checkpoint. The joint-space client sends
decoded GR00T joint targets directly to the IsaacLab joint-position action term.

## Human-Gated DAgger

The reusable DAgger workflow is maintained as a separate system guide so collection, conversion, normalization,
training, and evaluation use one authoritative recipe:

- [VLA DAgger customer guide](../../../../docs/vla_dagger_guide.md)
- [VLA DAgger system reference](../../../../docs/vla_dagger_reference.md)
- [Codex VLA DAgger Skill](../../../../.agents/skills/vla-dagger/SKILL.md)

The guide keeps customer paths and dataset identifiers parameterized. It also explains which parts of the Franka
implementation are reusable contracts and which must be replaced for another robot or task.

## Troubleshooting

- `ModuleNotFoundError: No module named 'isaaclab_teleop'` means the teleop extension is not installed in the active
conda environment. Run `./isaaclab.sh -i teleop`.
- `module 'omni' has no attribute 'appwindow'` happens when keyboard teleoperation is launched without Kit GUI
support. Add `--viz kit`.
- Long waits such as `Waiting for RtPso async group async compilation` are RTX shader compilation during Kit startup,
especially when cameras/rendering are enabled.
- HDF5 replay/conversion fails on truncated files. A canceled recording may leave a partial HDF5, so prefer a
completed dataset file for replay and conversion.
