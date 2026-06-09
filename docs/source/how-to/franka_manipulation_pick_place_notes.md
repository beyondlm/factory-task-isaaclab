# Franka Manipulation Pick-and-Place Pipeline Notes

This is the single maintained markdown note for the Franka pick-and-place pipeline in this IsaacLab checkout. It combines the previous command notes and pipeline optimization summary.

Command-only quick reference:

```text
scripts/benchmarks/gr00t/franka/franka_manipulation_gr00t_commands.md
```

## Scope

Primary task:

```text
Isaac-Pick-Place-Franka-IK-Rel-v0
```

Replay camera task:

```text
Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0
```

Object/bin mapping:

```text
object_a = box_3_label -> blue bin  = sorting_bin_blue
object_b = box_4_no    -> black bin = black_sorting_bin
```

Main files:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/joint_pos_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/ik_rel_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/mdp/terminations.py
```

## Path Variables

The commands below use customer-provided paths. Set them for your machine before running the snippets:

```bash
export WORKSPACE=/path/to/workspace
export ISAACLAB=/path/to/IsaacLab
export GROOT=/path/to/Isaac-GR00T
export LEROBOT=/path/to/lerobot
export DATA_ROOT=/path/to/data
export CHECKPOINT_ROOT=/path/to/checkpoints
export LOCAL_GROOT_WORKDIR=/path/to/local/gr00t_workdir
export FRANKA_SORTING_ASSET_DIR=/path/to/franka_sorting_assets
```

`FRANKA_SORTING_ASSET_DIR` should contain the factory Franka, belt, box, and bin USD files used by
`joint_pos_env_cfg.py`.

## Optimized Task Setup

Scene and control changes:

- Franka was moved toward the blue/black bins for easier manipulation.
- Belt/table scene scale is controlled by `SORTING_SCENE_SCALE` and `SORTING_BELT_SCALE`.
- Proxy belt/table colliders are sized from the same scene scale.
- `table_camera` position now derives from `_belt_scaled_pos(FRANKA_TABLE_CAM_LOCAL_POS)`, so changing `SORTING_BELT_SCALE` moves the camera consistently.
- Replay cameras use `CameraCfg.OffsetCfg(..., convention="opengl")`.
- Wrist camera quaternion is stored in IsaacLab `CameraCfg` order: `(x, y, z, w)`.
- Viewport camera starts closer to the Franka/bin workspace.
- Ground setup was checked so the task uses one normal ground plane setup.

Current Franka teleop/control defaults:

```text
FRANKA_IK_ACTION_SCALE=1.0
FRANKA_SPACEMOUSE_POS_SENSITIVITY=0.2
FRANKA_SPACEMOUSE_ROT_SENSITIVITY=0.2
FRANKA_KEYBOARD_POS_SENSITIVITY=0.1
FRANKA_KEYBOARD_ROT_SENSITIVITY=0.1
```

Franka actuator tracking was also made more responsive in `joint_pos_env_cfg.py` by increasing shoulder/forearm stiffness, damping, effort limits, and velocity limits.

## Success Criteria

The success logic was tightened so the task does not finish while a box is only hovering above a bin:

- `box_3_label` must be in the blue bin.
- `box_4_no` must be in the black bin.
- Success checks both XY distance and a tight bin height threshold.
- The box root must be low in the bin: signed height difference is gated to roughly `[-0.01, 0.035]` m.
- `require_gripper_open=True` prevents ending while still holding the object.
- Success also requires both boxes to be dynamically settled: low linear and angular velocity.
- The task waits for a real release/drop into the bin.

Relevant function:

```text
both_boxes_placed_a_into_c_b_into_d(...)
```

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

Slower SpaceMouse run:

```bash
FRANKA_SPACEMOUSE_POS_SENSITIVITY=0.1 \
FRANKA_SPACEMOUSE_ROT_SENSITIVITY=0.1 \
FRANKA_IK_ACTION_SCALE=0.75 \
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device spacemouse \
  --viz kit
```

Camera-enabled teleop:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --enable_cameras \
  --viz kit
```

## HDF5 Recording

Current record command:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta

./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --viz kit \
  --dataset_file ./datasets/dataset_sorting_105.hdf5 \
  --num_demos 105 \
  --teleop_device spacemouse \
  --num_success_steps 20
```

Recommended two-batch recording for training:

```bash
# Batch 1: 100 episodes ending only after both boxes drop and settle in the blue/black bins.
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --viz kit \
  --dataset_file ./datasets/franka_sorting_drop_settle_100.hdf5 \
  --num_demos 100 \
  --teleop_device spacemouse \
  --num_success_steps 30

# Batch 2: 100 episodes with a longer completed-action tail after both boxes are settled.
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --viz kit \
  --dataset_file ./datasets/franka_sorting_completed_tail_100.hdf5 \
  --num_demos 100 \
  --teleop_device spacemouse \
  --num_success_steps 60
```

`record_demos.py` creates the HDF5 with write mode, so reusing the same `--dataset_file` overwrites the old file instead of appending.

Recorded HDF5 action format:

```text
actions: [dx, dy, dz, d_rx, d_ry, d_rz, gripper_cmd]
```

Dataset used for conversion:

```text
datasets/dataset_sorting_105.hdf5
```

Observed dataset stats:

```text
episodes: 105
converted LeRobot frames: 60432
```

Replay low-dimensional demos:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta

./isaaclab.sh -p scripts/tools/replay_demos.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-v0 \
  --dataset_file ./datasets/dataset_sorting_105.hdf5 \
  --validate_success_rate \
  --viz kit
```

## Replay Video Generation

New replay script:

```text
scripts/benchmarks/gr00t/franka/replay_demos_with_camera.py
```

Generate full replay videos:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta

./isaaclab.sh -p scripts/benchmarks/gr00t/franka/replay_demos_with_camera.py \
  --task Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0 \
  --dataset_file datasets/dataset_sorting_105.hdf5 \
  --video \
  --camera_view_list wrist_camera table_camera \
  --video-output-dir datasets/dataset_sorting_105/generated_videos \
  --validate_success_rate \
  --viz none
```

Expected video names:

```text
datasets/dataset_sorting_105/generated_videos/demo_0_wrist_camera.mp4
datasets/dataset_sorting_105/generated_videos/demo_0_table_camera.mp4
```

One-episode smoke test:

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

Verified smoke-test output:

```text
/tmp/franka_replay_video_test/demo_0_wrist_camera.mp4
/tmp/franka_replay_video_test/demo_0_table_camera.mp4
```

Both test videos were `640x480`, `30 fps`, and `654` frames, matching the converted `demo_0` parquet frame count.

## LeRobot V2 Conversion

New converter:

```text
scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_task_space.py
```

Use task-space conversion because this dataset was recorded from IK-relative control.

State:

```text
observation.state = [eef_pos(3), eef_quat(4), gripper_width(1)]
shape = 8
```

Action:

```text
action = [dx, dy, dz, d_rx, d_ry, d_rz, gripper_cmd]
shape = 7
```

Convert without videos:

```bash
python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_task_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5 \
  --overwrite
```

Convert with replay videos:

```bash
python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_task_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5 \
  --video-dir datasets/dataset_sorting_105/generated_videos \
  --require-videos \
  --overwrite
```

Default output:

```text
datasets/dataset_sorting_105/lerobot_task_space
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

## GR00T Training Hooks

New files:

```text
scripts/benchmarks/gr00t/franka/modality_task_space.json
scripts/benchmarks/gr00t/franka/data_config.py
scripts/benchmarks/gr00t/franka/franka_manipulation_gr00t_commands.md
scripts/benchmarks/gr00t/franka/README.md
```

GR00T data config name:

```text
franka_pick_place_relative_task_space
```

Add the data config entry to Isaac-GR00T before training:

```python
"franka_pick_place_relative_task_space": FrankaPickPlaceRelativeTaskSpaceDataConfig(),
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
mkdir -p $WORKSPACE/projects
cd "$WORKSPACE"/projects

git clone https://github.com/huggingface/lerobot.git
cd lerobot
git fetch --tags
git checkout v0.4.0
```

Install LeRobot and the data conversion dependencies:

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

Notes:

- LeRobot v2 is the dataset layout written under `datasets/dataset_sorting_105/lerobot_task_space` and
  `datasets/dataset_sorting_105/lerobot_joint_space`.
- The Franka converters are self-contained and can run from `isaaclab3_beta`, but keeping `lerobot_v040` available is
  useful for checking LeRobot data and later GR00T data transfer workflows.
- Keep video files as MP4 in the LeRobot v2 dataset. GIFs are only for README/demo visualization.

The converter is self-contained and does not import IsaacLab, but it needs:

```text
h5py
pandas
pyarrow
tqdm
ffprobe/ffmpeg for video metadata/copy validation
```

## Runtime Findings

- `ModuleNotFoundError: No module named 'isaaclab_teleop'` means the `teleop` submodule is present in `source/isaaclab_teleop/` but not installed in the active conda environment.
- `module 'omni' has no attribute 'appwindow'` happens when keyboard teleoperation is launched without Kit GUI support. Add `--viz kit` so `omni.appwindow` is available.
- Long waits such as `Waiting for RtPso async group async compilation` are RTX shader compilation during Kit startup, especially when cameras/rendering are enabled. First launch can take several minutes.
- Stale Isaac Sim / Kit processes can consume GPU and CPU resources. Check and close old Kit processes before testing new commands.
- HDF5 replay/conversion fails on truncated files. A canceled recording may leave a partial HDF5, so prefer a completed dataset file for replay and conversion.

## Joint-Space LeRobot Conversion

The same HDF5 teleop dataset can also be converted to joint-space LeRobot v2.

Converter:

```text
scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_joint_space.py
```

State/action layout:

```text
state  = [panda_joint1..panda_joint7, gripper_width]
action = [next panda_joint1..panda_joint7, next gripper_width]
shape  = 8
```

Convert with replay videos:

```bash
cd "$ISAACLAB"
conda activate isaaclab3_beta

python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_joint_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5 \
  --video-dir datasets/dataset_sorting_105/generated_videos \
  --require-videos \
  --overwrite
```

Default output:

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

## Dataset Statistics

GR00T has two different statistics locations:

```text
dataset/meta/stats.json
dataset/meta/relative_stats.json
checkpoint/statistics.json
```

`meta/stats.json` and `meta/relative_stats.json` are dataset-side files used by training and open-loop dataset loading. `checkpoint/statistics.json` is checkpoint-side normalization used for inference. A checkpoint can be valid even if the local dataset copy is missing the two meta files.

The training dataset factory can generate the dataset-side files automatically on the training machine. For open-loop on another host, generate them locally or copy them from the training host.

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

Regenerate stats after changing action horizon, action keys, or modality config.

## Isaac-GR00T Franka Config Files

Local IsaacLab-maintained configs:

```text
scripts/benchmarks/gr00t/franka/franka_modality_config.py
scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py
```

## GR00T Training

Task-space 20k-step command:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_task_space"
CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_modality_config.py"

NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES=0 \
uv run python gr00t/experiment/launch_finetune.py \
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

Joint-space 20k-step command:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_joint_space"
CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py"

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

True resume from `checkpoint-10000` to `checkpoint-20000`:

```bash
cd "$GROOT"

DATASET="$ISAACLAB/datasets/dataset_sorting_105/lerobot_joint_space"
CONFIG="$ISAACLAB/scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py"
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

Expected resume log:

```text
Resuming from checkpoint .../checkpoint-10000
```

Important distinction:

```text
--base-model-path checkpoint-10000
```

loads only model weights and starts a fresh optimizer/training schedule from step 0. True resume requires a checkpoint directory inside `--output-dir` so Hugging Face can load `trainer_state.json`, `optimizer.pt`, `scheduler.pt`, and RNG state.

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

Expected joint checkpoint 10000 inference file sizes:

```text
model-00001-of-00003.safetensors  4986649584
model-00002-of-00003.safetensors  4970792616
model-00003-of-00003.safetensors  2618758696
statistics.json                      4321792
processor_config.json                  26640
```

If `safetensors_rust.SafetensorError: incomplete metadata` appears, the safetensors shard is truncated and must be copied again.

## Open-Loop Evaluation

Task-space checkpoint open loop:

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

Joint-space checkpoint open loop:

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

Server for task-space checkpoint:

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

Client for task-space checkpoint:

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

Server for joint-space checkpoint:

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

Client for joint-space checkpoint:

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

Use `--headless` on the client if GUI is not needed.

## Joint-Space Closed-Loop Notes

Franka joint-space closed-loop support was added in:

```text
scripts/benchmarks/gr00t/franka/gr00t_inference_client_franka.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/ik_rel_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/__init__.py
```

Registered task:

```text
Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0
```

The client sends joint-space observation keys:

```text
franka_joint_pos
franka_gripper_width
```

Expected joint checkpoint action keys:

```text
franka_joint_pos
franka_gripper_width
```

If debug output shows EEF action keys, the GR00T server is using the wrong checkpoint.

The joint-space client sends decoded GR00T joint targets directly to the IsaacLab joint-position action term.

The joint-space policy may need more training than task-space EEF because it must learn wrist orientation and IK-like behavior from the joint trajectory.
