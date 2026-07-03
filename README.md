# Factory Task IsaacLab Overlay

This repo is intended for IsaacLab factory-task development and GR00T N-series policy evaluation. It provides task
overlays, LeRobot conversion utilities, and both open-loop and closed-loop evaluation workflows for GR00T N.x models.

Current validated GR00T version: **GR00T N1.7**.

This repository is an IsaacLab 3 Beta overlay focused on factory robot learning scenarios.

Install IsaacLab 3 Beta first:
[IsaacLab v3.0.0-beta](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-beta)

Then sync this overlay into the IsaacLab root while preserving the same relative file hierarchy, including the README
and docs media:

```bash
OVERLAY=/path/to/factory-task-isaaclab
ISAACLAB=/path/to/IsaacLab

rsync -av \
  "$OVERLAY/README.md" \
  "$OVERLAY/docs" \
  "$OVERLAY/scripts" \
  "$OVERLAY/source" \
  "$ISAACLAB"/
```

Set the Franka sorting USD asset root before running the task:

```bash
export FRANKA_SORTING_ASSET_DIR=/path/to/franka_sorting_assets
```

The asset directory should contain the factory Franka, belt, box, and bin USD files referenced by the Franka task config.

viewport for visualization

![Viewport for visualization](docs/source/_static/how-to/penetrate.gif)

training viewport : table(static) and wrist

![Training viewport: table(static) and wrist](docs/source/_static/how-to/head-wrist.gif)

## Focus

This repo targets factory-style manipulation workflows:

- pick-and-place
- bin sorting
- object transfer
- gripper-based manipulation
- closed-loop policy evaluation
- dexterous-hand factory tasks to be added later

The first implemented task family is Franka box sorting with GR00T N1.7/LeRobot data conversion and closed-loop
evaluation.

GR00T baseline:
[NVIDIA Isaac-GR00T N1.7 release](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.7-release)

## Franka GR00T Action Horizon

The Franka GR00T modality configs use a runtime action horizon:

```bash
export ACTION_HORIZON=32
export FRANKA_GROOT_ACTION_HORIZON=$ACTION_HORIZON
```

The default is 32 when `FRANKA_GROOT_ACTION_HORIZON` is not set. Use the same value for dataset statistics,
training, open-loop evaluation, and closed-loop client feedback actions. The current task-space and joint-space configs
use the current camera frame only, with `video.delta_indices = [0]`.

## Franka GR00T State History

The current joint-space state-history experiment keeps the camera input current-only and adds short robot-state history:

```text
video.delta_indices = [0]
state.delta_indices = [-2, -1, 0]
action.delta_indices = list(range(32))
```

This gives GR00T the two previous joint states plus the current joint state while keeping the action horizon at 32.
Detailed local eval commands are tracked in the
[state-history H32 command reference](scripts/benchmarks/gr00t/franka/state_history_h32_local_eval_commands.md).

Franka sorting closed-loop benchmark summary:

The 10k rows use 105 episodes; the 20k rows use 201 episodes.

| Policy action space | GR00T N1.7 modality config | GR00T N1.7 action representation | Training dataset | Training setup | Batch size | Closed-loop SR | Failure notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EEF / IK-relative | [EEF config](scripts/benchmarks/gr00t/franka/franka_modality_config.py) | `franka_eef_delta_pos`, `franka_eef_delta_rot`, `franka_gripper_cmd` as raw continuous actions | 105 episodes | 10k steps | 256 | 65% | 20 trials:<br />1: OOD pick failure, 7 times. |
| EEF / IK-relative | [EEF config](scripts/benchmarks/gr00t/franka/franka_modality_config.py) | `franka_eef_delta_pos`, `franka_eef_delta_rot`, `franka_gripper_cmd` as raw continuous actions | 201 episodes | 20k steps | 256 | 100% | 20 trials: no failures. |
| Joint space | [Joint config](scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py) | `franka_joint_pos`, `franka_gripper_width` | 105 episodes | 10k steps | 256 | 30% | 20 trials:<br />1: mixed pick/place/OOD failures, 14 times. |
| Joint space | [Joint config](scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py) | `franka_joint_pos`, `franka_gripper_width` | 201 episodes | 20k steps | 256 | 50% | 20 trials:<br />1: OOD, 3 times.<br />2: near box, but no gripper close, 7 times. |
| Joint space + video history | [Joint config](scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py) with archived `video.delta_indices = [-16, 0]` | `franka_joint_pos`, `franka_gripper_width`; current-state only, two video frames | 201 episodes | 20k steps | 256 | 35% | 20 trials. History-frame video hurt joint-space closed-loop SR, likely because historical robot images conflicted with current-only joint state. |
| Joint space + action horizon 32 | [Joint config](scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py) | `franka_joint_pos`, `franka_gripper_width` | 201 episodes | 20k steps | 256 | 65% | 20 trials, 13 successes / 7 failures:<br />1: grasp hesitation above the box, then after placing the first box the pose for reaching the second box becomes abnormal, 6 times.<br />2: perception failure, gripper closes above the box, 1 time. |
| Joint space + action horizon 32 + state history | [Joint config](scripts/benchmarks/gr00t/franka/franka_joint_modality_config.py) | `franka_joint_pos`, `franka_gripper_width` | 201 episodes | 20k steps, action horizon 32, state history `[-2, -1, 0]` | 256 | 60% | 20 trials:<br />1: gripper hesitates above angled boxes, 3 times.<br />2: remaining failures need finer categorization. |

For the EEF policy, `ABSOLUTE` in the GR00T modality config does not mean the robot executes absolute EEF poses. It means GR00T should learn the recorded IK-relative delta vector directly as a normal continuous action, while IsaacLab's IK-relative controller applies that delta during rollout.

## Future Plan

- ☑ Add another 100 Franka sorting episodes with broader box pose and task-progress coverage.
- ☑ Evaluate GR00T history-frame training with temporal camera context: joint-space 201-episode 20k-step checkpoint reached 35% SR, so joint-space was reverted to single-frame video ([history-frame summary](scripts/benchmarks/gr00t/franka/history_frame_summary.md)).
- ☑ Set GR00T action horizon from 16 to 32 for Franka training, stats generation, open-loop evaluation, and closed-loop feedback execution ([16-to-32 action horizon summary](scripts/benchmarks/gr00t/franka/action_horizon_16_to_32.md)).
- ☑ Continue joint-space state-history experiments. The 20k-step `[-2, -1, 0]` state-history run improved SR to 60%, but still shows gripper hesitation above boxes at some object angles ([implementation and eval notes](scripts/benchmarks/gr00t/franka/state_history_h32_local_eval_commands.md)).
- ☐ Add DAgger evaluation to reduce closed-loop drift and collect correction data.

## Main Links

Franka benchmark details and command reference:

[Franka GR00T command guide](scripts/benchmarks/gr00t/franka/franka_manipulation_gr00t_commands.md)

Action chunk migration note:

[Update GR00T action horizon from 16 to 32](scripts/benchmarks/gr00t/franka/action_horizon_16_to_32.md)

Franka joint-space state-history H32 eval commands:

[State-history H32 command reference](scripts/benchmarks/gr00t/franka/state_history_h32_local_eval_commands.md)

## Included Overlay Paths

```text
docs/source/_static/how-to/penetrate.gif
docs/source/_static/how-to/head-wrist.gif
docs/source/_static/how-to/franka_eef_open_loop_5000_traj0.jpeg
docs/source/_static/how-to/franka_joint_open_loop_20000_traj0.jpeg
scripts/benchmarks/gr00t/franka/
scripts/benchmarks/gr00t/franka/state_history_h32_local_eval_commands.md
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/mdp/
```
