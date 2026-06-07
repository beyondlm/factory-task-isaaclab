# Franka Pick-and-Place GR00T-LeRobot Conversion

This folder converts HDF5 demos recorded from `Isaac-Pick-Place-Franka-IK-Rel-v0` to the GR00T-LeRobot v2 layout.

The current runnable commands are also collected in `franka_manipulation_gr00t_commands.md`.

Use task-space conversion for this dataset. The recorded action is a relative IK command:

```text
[dx, dy, dz, d_rx, d_ry, d_rz, gripper_cmd]
```

Generate replay videos for the HDF5 dataset:

```bash
cd /home/npnsa/workspace/jeff/projects/isaaclab_3_beta/IsaacLab
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

The generated videos are named for the converter:

```text
datasets/dataset_sorting_105/generated_videos/demo_0_wrist_camera.mp4
datasets/dataset_sorting_105/generated_videos/demo_0_table_camera.mp4
```

Convert the low-dimensional HDF5 dataset without videos:

```bash
python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_task_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5
```

Default output:

```text
datasets/dataset_sorting_105/lerobot_task_space
```

Convert the HDF5 dataset with replay videos:

```bash
python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_task_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5 \
  --video-dir datasets/dataset_sorting_105/generated_videos \
  --require-videos \
  --overwrite
```

Convert the same HDF5 dataset to joint-space LeRobot data:

```bash
python scripts/benchmarks/gr00t/franka/convert_hdf5_to_lerobot_joint_space.py \
  --hdf5-file-path datasets/dataset_sorting_105.hdf5 \
  --video-dir datasets/dataset_sorting_105/generated_videos \
  --require-videos \
  --overwrite
```

Default joint-space output:

```text
datasets/dataset_sorting_105/lerobot_joint_space
```

The joint-space converter stores:

```text
state[t]  = [panda_joint1..panda_joint7, gripper_width] at t
action[t] = [panda_joint1..panda_joint7, gripper_width] at t + 1
```

For GR00T training, merge `data_config.py` into Isaac-GR00T's `gr00t/experiment/data_config.py` and add:

```python
"franka_pick_place_relative_task_space": FrankaPickPlaceRelativeTaskSpaceDataConfig(),
```
