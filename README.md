# Factory Task Robot Learning Workspace

This repository is based on IsaacLab 3 Beta and is maintained as a factory-task robot learning workspace on top of
Isaac Lab and Isaac Sim.

The focus is industrial manipulation: pick-and-place, bin sorting, object transfer, and contact-rich manipulation with
parallel-jaw grippers and dexterous hands. The codebase is used to build simulation tasks, collect teleoperation demos,
convert datasets, train policies, and evaluate closed-loop behavior in factory-style scenes.

![Franka sorting closed-loop demo](docs/source/_static/how-to/franka_sorting_closed_loop.gif)

## Install As An IsaacLab 3 Beta Overlay

Install IsaacLab 3 Beta first:

[IsaacLab v3.0.0-beta](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-beta)

This repository is intended to be copied into an existing IsaacLab 3 Beta checkout while preserving the same relative
file hierarchy. After IsaacLab 3 Beta is installed and working, copy this overlay into the IsaacLab root:

```bash
OVERLAY=/path/to/franka-factory-task-isaaclab
ISAACLAB=/path/to/IsaacLab

rsync -av "$OVERLAY"/ "$ISAACLAB"/
```

The paths in this repository are kept identical to the IsaacLab tree so the task configs, benchmark scripts, docs, and
assets land in the correct locations.

## Focus Areas

- **Gripper manipulation**: Franka pick-and-place, box sorting, object placement, and bin-targeted tasks.
- **Factory sorting workflows**: teleoperation, HDF5 recording, replay validation, camera video generation, and success checks.
- **GR00T and LeRobot data pipelines**: converting IsaacLab HDF5 demos into LeRobot v2 task-space and joint-space datasets.
- **Policy training and evaluation**: open-loop trajectory checks, GR00T server/client inference, and closed-loop simulation tests.
- **Dexterous hand manipulation**: contact-rich manipulation tasks for factory operations beyond simple gripper control.

## Main Task Family

Current active task work is under:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/
```

Franka pick-and-place/sorting tasks:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_and_place/config/franka/
```

GR00T benchmark tools:

```text
scripts/benchmarks/gr00t/franka/
```

## Franka Sorting Pipeline

The Franka sorting pipeline supports:

- SpaceMouse and keyboard teleoperation.
- HDF5 demo recording.
- Replay with wrist and table cameras.
- Conversion to LeRobot v2 task-space data.
- Conversion to LeRobot v2 joint-space data.
- GR00T fine-tuning.
- Open-loop plot evaluation.
- Closed-loop GR00T server/client evaluation.

Detailed notes:

[Franka manipulation pick-and-place guide](docs/source/how-to/franka_manipulation_pick_place_notes.md)

Command quick reference:

[Franka GR00T command reference](scripts/benchmarks/gr00t/franka/franka_manipulation_gr00t_commands.md)

## Current Franka Tasks

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
box_3_label -> blue bin
box_4_no    -> black bin
```

## Development Goal

The goal is to build reusable factory automation pipelines that connect:

```text
simulation task -> teleoperation demo -> replay video -> LeRobot dataset -> GR00T training -> closed-loop evaluation
```

The repository should stay focused on practical manipulation workflows and the assets, task configs, conversion scripts,
and evaluation scripts needed to iterate on those workflows.
