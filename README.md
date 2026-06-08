# Factory Task IsaacLab Overlay

This repository is an IsaacLab 3 Beta overlay focused on factory robot learning scenarios.

Install IsaacLab 3 Beta first:
[IsaacLab v3.0.0-beta](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-beta)

Then copy this overlay into the IsaacLab root while preserving the same relative file hierarchy:

```bash
OVERLAY=/path/to/factory-task-isaaclab
ISAACLAB=/path/to/IsaacLab

rsync -av "$OVERLAY"/ "$ISAACLAB"/
```

Set the Franka sorting USD asset root before running the task:

```bash
export FRANKA_SORTING_ASSET_DIR=/path/to/franka_sorting_assets
```

The asset directory should contain the factory Franka, belt, box, and bin USD files referenced by the Franka task config.

![Franka sorting closed-loop demo](docs/source/_static/how-to/franka_sorting_closed_loop.gif)

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

Current EEF/task-space closed-loop benchmark:

```text
Franka sorting EEF policy: 65% SR
Model: GR00T N1.7
Training: 20k steps, global batch size 256
```

## Main Links

Franka benchmark details:

[Franka GR00T benchmark README](scripts/benchmarks/gr00t/franka/README.md)

Franka command reference:

[Franka GR00T command reference](scripts/benchmarks/gr00t/franka/franka_manipulation_gr00t_commands.md)

Full Franka pipeline notes:

[Franka manipulation pick-and-place guide](docs/source/how-to/franka_manipulation_pick_place_notes.md)

## Included Overlay Paths

```text
docs/source/how-to/franka_manipulation_pick_place_notes.md
docs/source/_static/how-to/franka_sorting_closed_loop.gif
scripts/benchmarks/gr00t/franka/
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/mdp/
```
