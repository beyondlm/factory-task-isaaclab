# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import convert_hdf5_to_lerobot_joint_space as base  # noqa: E402
from convert_hdf5_to_lerobot_dagger_joint_space import (  # noqa: E402
    Config,
    clip_segment_to_available_frames,
    episode_success,
    segment_dataframe,
)
from validate_hg_dagger_dataset import validate_episode  # noqa: E402


def test_episode_success_reads_hg_dagger_summary(tmp_path: Path) -> None:
    source_path = tmp_path / "success.hdf5"
    with h5py.File(source_path, "w") as stream:
        successful = stream.create_group("demo_success")
        successful.attrs["success"] = False
        successful.create_group("dagger").create_group("episode").create_dataset("success", data=np.array([True]))

        failed = stream.create_group("demo_failed")
        failed.attrs["success"] = True
        failed.create_group("dagger").create_group("episode").create_dataset("success", data=np.array([False]))

        assert episode_success(successful)
        assert not episode_success(failed)


def test_tail_segment_is_clipped_to_available_video_frames() -> None:
    assert clip_segment_to_available_frames(
        967,
        1200,
        available_frames=1199,
        minimum_length=64,
    ) == (967, 1199)
    assert clip_segment_to_available_frames(
        100,
        200,
        available_frames=1199,
        minimum_length=64,
    ) == (100, 200)
    assert (
        clip_segment_to_available_frames(
            1150,
            1200,
            available_frames=1199,
            minimum_length=64,
        )
        is None
    )


def test_segment_dataframe_preserves_alignment(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.hdf5"
    with h5py.File(source_path, "w") as stream:
        trajectory = stream.create_group("demo_0")
        dagger = trajectory.create_group("dagger")
        values = np.arange(80, dtype=np.float32).reshape(10, 8)
        dagger.create_dataset("observation_joint_state", data=values)
        dagger.create_dataset("achieved_joint_state", data=values + 1)
        dagger.create_dataset("policy_action", data=values + 2)
        dagger.create_dataset("expert_action", data=values + 3)
        dagger.create_dataset("executed_action", data=values + 4)
        dagger.create_dataset("policy_action_valid", data=np.ones(10, dtype=bool))
        dagger.create_dataset("intervention_mask", data=np.ones(10, dtype=bool))
        trajectory.attrs["success"] = True

        config = Config(action_horizon=4, minimum_segment_length=4)
        dataframe = segment_dataframe(
            trajectory,
            2,
            8,
            episode_index=3,
            index_start=10,
            config=config,
        )

    assert len(dataframe) == 6
    np.testing.assert_array_equal(dataframe.iloc[0]["observation.state"], values[2])
    np.testing.assert_array_equal(dataframe.iloc[0]["action"], values[2] + 1)
    np.testing.assert_array_equal(
        dataframe["annotation.human.action.valid"].to_numpy(),
        np.array([1, 1, 1, 0, 0, 0], dtype=bool),
    )
    assert dataframe["episode_index"].unique().tolist() == [3]
    assert dataframe["index"].tolist() == list(range(10, 16))
    assert dataframe.iloc[-1]["next.reward"] == 1.0
    assert bool(dataframe.iloc[-1]["next.done"])


def test_segment_dataframe_uses_executed_binary_gripper_target(tmp_path: Path) -> None:
    source_path = tmp_path / "binary_gripper.hdf5"
    with h5py.File(source_path, "w") as stream:
        trajectory = stream.create_group("demo_0")
        dagger = trajectory.create_group("dagger")
        values = np.arange(48, dtype=np.float32).reshape(6, 8)
        achieved = values + 1
        achieved[:, -1] = 0.06
        executed = values + 4
        executed[:, -1] = np.array([-1, -1, 1, 1, -1, 1], dtype=np.float32)
        dagger.create_dataset("observation_joint_state", data=values)
        dagger.create_dataset("achieved_joint_state", data=achieved)
        dagger.create_dataset("policy_action", data=values + 2)
        dagger.create_dataset("expert_action", data=values + 3)
        dagger.create_dataset("executed_action", data=executed)
        dagger.create_dataset("policy_action_valid", data=np.ones(6, dtype=bool))
        dagger.create_dataset("intervention_mask", data=np.ones(6, dtype=bool))
        trajectory.attrs["success"] = True

        dataframe = segment_dataframe(
            trajectory,
            0,
            6,
            episode_index=0,
            index_start=0,
            config=Config(
                action_horizon=4,
                minimum_segment_length=4,
                binary_gripper_command_target=True,
            ),
        )

    np.testing.assert_array_equal(np.stack(dataframe["action"].to_numpy())[:, :7], achieved[:, :7])
    np.testing.assert_array_equal(
        np.stack(dataframe["action"].to_numpy())[:, -1],
        np.array([0.0, 0.0, 0.08, 0.08, 0.0, 0.08], dtype=np.float32),
    )


def test_binary_gripper_command_to_width_rejects_nonbinary_values() -> None:
    with np.testing.assert_raises_regex(ValueError, "binary gripper commands"):
        base.binary_gripper_command_to_width(np.array([-1.0, 0.0, 1.0], dtype=np.float32))


def test_segment_dataframe_rejects_unmasked_policy_context(tmp_path: Path) -> None:
    source_path = tmp_path / "history.hdf5"
    with h5py.File(source_path, "w") as stream:
        trajectory = stream.create_group("demo_0")
        dagger = trajectory.create_group("dagger")
        values = np.arange(80, dtype=np.float32).reshape(10, 8)
        for name in (
            "observation_joint_state",
            "achieved_joint_state",
            "policy_action",
            "expert_action",
            "executed_action",
        ):
            dagger.create_dataset(name, data=values)
        intervention = np.zeros(10, dtype=bool)
        intervention[4:10] = True
        dagger.create_dataset("intervention_mask", data=intervention)
        dagger.create_dataset("policy_action_valid", data=np.zeros(10, dtype=bool))

        with np.testing.assert_raises_regex(ValueError, "anchor/loss-mask integration"):
            segment_dataframe(
                trajectory,
                4,
                10,
                episode_index=0,
                index_start=0,
                config=Config(action_horizon=4, minimum_segment_length=4, state_history_frames=3),
            )

def test_hdf5_alignment_validation(tmp_path: Path) -> None:
    source_path = tmp_path / "aligned.hdf5"
    length = 8
    with h5py.File(source_path, "w") as stream:
        episode = stream.create_group("demo_0")
        episode.create_dataset("actions", data=np.zeros((length, 8), dtype=np.float32))
        dagger = episode.create_group("dagger")
        for name in (
            "observation_joint_state",
            "achieved_joint_state",
            "policy_action",
            "expert_action",
            "executed_action",
        ):
            dagger.create_dataset(name, data=np.zeros((length, 8), dtype=np.float32))
        dagger.create_dataset("intervention_mask", data=np.ones(length, dtype=bool))
        dagger.create_dataset("policy_action_valid", data=np.zeros(length, dtype=bool))
        dagger.create_dataset("inference_id", data=np.zeros(length, dtype=np.int64))
        dagger.create_dataset("chunk_index", data=-np.ones(length, dtype=np.int64))
        dagger.create_dataset("frame_index", data=np.arange(length, dtype=np.int64))
        episode_summary = dagger.create_group("episode")
        summary_values = {
            "success": True,
            "outcome_code": 1,
            "seed": 11,
            "frame_count": length,
            "intervention_steps": length,
            "intervention_segments": 1,
            "intervention_ratio": 1.0,
            "valid_policy_actions": 0,
        }
        for name, value in summary_values.items():
            episode_summary.create_dataset(name, data=np.asarray([value]))
        episode_summary.create_dataset(
            "policy_checkpoint_id_utf8",
            data=np.frombuffer(b"checkpoint-1", dtype=np.uint8).reshape(1, -1),
        )
        result = validate_episode(episode, horizon=4, minimum_segment_length=6)

    assert result == {
        "steps": 8,
        "intervention_steps": 8,
        "segments": 1,
        "valid_anchors": 5,
    }
