# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop GR00T client for Franka pick-and-place GR00T policies.

Run this script in the IsaacLab environment. It connects to a GR00T N1.7
PolicyServer running in the Isaac-GR00T environment and executes the returned
action chunks in either the Franka IK-relative or joint-position simulation task.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import traceback
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import msgpack
import numpy as np
import torch
import zmq

from isaaclab.app import AppLauncher


DEFAULT_EEF_TASK = "Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0"
DEFAULT_JOINT_TASK = "Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0"
FRANKA_ARM_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float32,
)


def _positive_int_from_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _int_list_from_env(name: str, default: list[int]) -> list[int]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return list(default)
    values = [int(value.strip()) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer index")
    return values


DEFAULT_NUM_FEEDBACK_ACTIONS = _positive_int_from_env("FRANKA_GROOT_ACTION_HORIZON", 32)
DEFAULT_STATE_HISTORY_FRAMES = len(_int_list_from_env("FRANKA_GROOT_STATE_DELTA_INDICES", [0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_EEF_TASK)
    parser.add_argument("--policy-type", choices=["task_space", "eef", "joint_space"], default="task_space")
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--server-api-token", default=None)
    parser.add_argument(
        "--language-instruction",
        default=(
            "Pick up the labeled box and place it into the blue bin. "
            "Pick up the unlabeled box and place it into the black bin."
        ),
    )
    parser.add_argument("--num-total-experiments", type=int, default=10)
    parser.add_argument("--max-inference-steps", type=int, default=62)
    parser.add_argument("--num-feedback-actions", type=int, default=DEFAULT_NUM_FEEDBACK_ACTIONS)
    parser.add_argument("--video-history-frames", type=int, default=None)
    parser.add_argument("--state-history-frames", type=int, default=DEFAULT_STATE_HISTORY_FRAMES)
    parser.add_argument("--camera-names", nargs="+", default=["wrist_camera", "table_camera"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-pos-delta", type=float, default=0.2)
    parser.add_argument("--max-rot-delta", type=float, default=0.2)
    parser.add_argument("--joint-gripper-open-threshold", type=float, default=0.04)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--pause-on-error", action="store_true")
    args = parser.parse_args()
    if args.policy_type == "joint_space" and args.task == DEFAULT_EEF_TASK:
        args.task = DEFAULT_JOINT_TASK
    if args.video_history_frames is None:
        args.video_history_frames = 1
    if args.video_history_frames < 1:
        raise ValueError("--video-history-frames must be >= 1")
    if args.state_history_frames < 1:
        raise ValueError("--state-history-frames must be >= 1")
    if args.num_feedback_actions < 1:
        raise ValueError("--num-feedback-actions must be >= 1")
    return args


args_cli = parse_args()

app_launcher = AppLauncher(
    headless=args_cli.headless,
    enable_cameras=True,
    num_envs=1,
    visualizer="none" if args_cli.headless else "kit",
    visualizer_explicit=True,
    experience="isaaclab.python.rendering.kit",
)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils import import_packages  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


_BLACKLIST_PKGS = [
    "utils",
    ".mdp",
    "pick_place",
    "articulated",
    "assembly",
    "libero",
    "robotwin",
    "stack",
]
import_packages("isaaclab_tasks", _BLACKLIST_PKGS)


class MsgSerializer:
    """Message serializer compatible with Isaac-GR00T's PolicyServer."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class Gr00tPolicyClient:
    host: str = "localhost"
    port: int = 5555
    api_token: str | None = None
    timeout_ms: int = 60000

    def __post_init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        self.socket.send(MsgSerializer.to_bytes(request))
        response = MsgSerializer.from_bytes(self.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except Exception:
            return False

    def get_action(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        response = self.call_endpoint("get_action", {"observation": observation, "options": None})
        if isinstance(response, list) and len(response) == 2:
            return response[0], response[1]
        if isinstance(response, tuple) and len(response) == 2:
            return response
        raise RuntimeError(f"Unexpected get_action response: {type(response)}")


def normalize_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert camera output to uint8 RGB with 3 channels."""
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        max_value = 1.0 if np.nanmax(frame) <= 1.0 else 255.0
        frame = np.clip(frame, 0.0, max_value) / max_value * 255.0
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def capture_camera_frames(env, camera_names: list[str]) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        if camera_name not in env.scene.sensors:
            available = ", ".join(sorted(env.scene.sensors.keys()))
            raise KeyError(f"Camera sensor '{camera_name}' was not found. Available sensors: {available}")
        sensor = env.scene.sensors[camera_name]
        frame = sensor.data.output["rgb"].detach().cpu().numpy()[0]
        frames[camera_name] = normalize_rgb(frame)
    return frames


class CameraHistoryBuffer:
    def __init__(self, history_frames: int):
        self.history_frames = history_frames
        self._history: dict[str, list[np.ndarray]] = {}

    def reset(self) -> None:
        self._history.clear()

    def observation(self, env, camera_names: list[str]) -> dict[str, np.ndarray]:
        current_frames = capture_camera_frames(env, camera_names)
        videos: dict[str, np.ndarray] = {}
        keep_history = max(self.history_frames - 1, 0)

        for camera_name, current_frame in current_frames.items():
            previous_frames = self._history.get(camera_name, [])
            frames = (previous_frames + [current_frame])[-self.history_frames :]
            if len(frames) < self.history_frames:
                frames = [frames[0]] * (self.history_frames - len(frames)) + frames
            videos[camera_name] = np.stack(frames, axis=0)[None, ...]

            if keep_history:
                self._history[camera_name] = (previous_frames + [current_frame])[-keep_history:]
            else:
                self._history[camera_name] = []

        return videos


class StateHistoryBuffer:
    def __init__(self, history_frames: int):
        self.history_frames = history_frames
        self._history: list[dict[str, np.ndarray]] = []

    def reset(self) -> None:
        self._history.clear()

    def append(self, state: dict[str, np.ndarray]) -> None:
        copied_state = {key: np.asarray(value, dtype=np.float32).copy() for key, value in state.items()}
        self._history.append(copied_state)
        self._history = self._history[-self.history_frames :]

    def observation(self) -> dict[str, np.ndarray]:
        if not self._history:
            raise RuntimeError("State history is empty; append the reset state before requesting an observation.")

        frames = self._history[-self.history_frames :]
        if len(frames) < self.history_frames:
            frames = [frames[0]] * (self.history_frames - len(frames)) + frames

        return {
            key: np.stack([frame[key] for frame in frames], axis=1)
            for key in frames[-1]
        }


def gripper_width_from_obs(gripper_pos: np.ndarray) -> np.ndarray:
    if gripper_pos.shape[-1] >= 2:
        width = np.abs(gripper_pos[:, 0] - gripper_pos[:, 1])
    else:
        width = gripper_pos[:, 0]
    return width.reshape(-1, 1).astype(np.float32)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def joint_id_list(joint_ids: Any) -> list[int]:
    return np.asarray(to_numpy(joint_ids), dtype=np.int64).reshape(-1).tolist()


def franka_joint_state(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.scene["robot"]
    arm_joint_ids, _ = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)])
    finger_joint_ids, _ = robot.find_joints(["panda_finger_joint[1-2]"])
    joint_pos = to_numpy(robot.data.joint_pos).astype(np.float32)
    arm_joint_pos = joint_pos[:, joint_id_list(arm_joint_ids)]
    gripper_width = joint_pos[:, joint_id_list(finger_joint_ids)].sum(axis=1, keepdims=True)
    return arm_joint_pos, gripper_width.astype(np.float32)


def task_space_policy_state(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    eef_pos = obs["policy"]["eef_pos"].detach().cpu().numpy().astype(np.float32)
    eef_quat = obs["policy"]["eef_quat"].detach().cpu().numpy().astype(np.float32)
    gripper_pos = obs["policy"]["gripper_pos"].detach().cpu().numpy().astype(np.float32)
    return {
        "franka_eef_pos": eef_pos,
        "franka_eef_quat": eef_quat,
        "franka_gripper_width": gripper_width_from_obs(gripper_pos),
    }


def joint_space_policy_state(env) -> dict[str, np.ndarray]:
    arm_joint_pos, gripper_width = franka_joint_state(env)
    return {
        "franka_joint_pos": arm_joint_pos,
        "franka_gripper_width": gripper_width,
    }


def current_policy_state(env, obs: dict[str, Any], args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.policy_type == "joint_space":
        return joint_space_policy_state(env)
    return task_space_policy_state(obs)


def task_space_policy_observation(
    env,
    args: argparse.Namespace,
    camera_history: CameraHistoryBuffer,
    state_history: StateHistoryBuffer,
) -> dict[str, Any]:
    return {
        "video": camera_history.observation(env, list(args.camera_names)),
        "state": state_history.observation(),
        "language": {
            "annotation.human.action.task_description": [[args.language_instruction]],
        },
    }


def joint_space_policy_observation(
    env,
    args: argparse.Namespace,
    camera_history: CameraHistoryBuffer,
    state_history: StateHistoryBuffer,
) -> dict[str, Any]:
    return {
        "video": camera_history.observation(env, list(args.camera_names)),
        "state": state_history.observation(),
        "language": {
            "annotation.human.action.task_description": [[args.language_instruction]],
        },
    }


def policy_observation(
    env,
    args: argparse.Namespace,
    camera_history: CameraHistoryBuffer,
    state_history: StateHistoryBuffer,
) -> dict[str, Any]:
    if args.policy_type == "joint_space":
        return joint_space_policy_observation(env, args, camera_history, state_history)
    return task_space_policy_observation(env, args, camera_history, state_history)


def action_value(action_dict: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key in action_dict:
        value = action_dict[key]
    elif f"action.{key}" in action_dict:
        value = action_dict[f"action.{key}"]
    else:
        raise KeyError(f"Missing action key '{key}'. Available keys: {sorted(action_dict.keys())}")

    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    elif value.ndim == 1:
        value = value.reshape(-1, 1)
    return value


def parse_task_space_action(action_dict: dict[str, np.ndarray], args: argparse.Namespace) -> torch.Tensor:
    pos = action_value(action_dict, "franka_eef_delta_pos")
    rot = action_value(action_dict, "franka_eef_delta_rot")
    gripper = action_value(action_dict, "franka_gripper_cmd")
    if gripper.shape[1] != 1:
        gripper = gripper.reshape(gripper.shape[0], 1)

    action = np.concatenate([pos, rot, gripper], axis=1).astype(np.float32)
    action[:, :3] = np.clip(action[:, :3], -args.max_pos_delta, args.max_pos_delta)
    action[:, 3:6] = np.clip(action[:, 3:6], -args.max_rot_delta, args.max_rot_delta)

    if not args.no_binarize_gripper:
        action[:, 6] = np.where(action[:, 6] >= 0.0, 1.0, -1.0)

    return torch.from_numpy(action[: args.num_feedback_actions])


def parse_joint_space_action(action_dict: dict[str, np.ndarray], args: argparse.Namespace) -> torch.Tensor:
    arm_joint_pos = action_value(action_dict, "franka_joint_pos")
    gripper_width = action_value(action_dict, "franka_gripper_width")
    if gripper_width.shape[1] != 1:
        gripper_width = gripper_width.reshape(gripper_width.shape[0], 1)

    action_horizon = min(arm_joint_pos.shape[0], gripper_width.shape[0], args.num_feedback_actions)
    arm_joint_pos = np.clip(arm_joint_pos[:action_horizon, :7], FRANKA_ARM_JOINT_LIMITS[:, 0], FRANKA_ARM_JOINT_LIMITS[:, 1])
    if args.no_binarize_gripper:
        gripper = gripper_width[:action_horizon]
    else:
        gripper = np.where(gripper_width[:action_horizon] >= args.joint_gripper_open_threshold, 1.0, -1.0)
    action = np.concatenate([arm_joint_pos, gripper], axis=1).astype(np.float32)
    return torch.from_numpy(action)


def parse_franka_action(action_dict: dict[str, np.ndarray], args: argparse.Namespace) -> torch.Tensor:
    if args.policy_type == "joint_space":
        return parse_joint_space_action(action_dict, args)
    return parse_task_space_action(action_dict, args)


def make_env(args: argparse.Namespace):
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.recorders = {}

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        print("No success termination term was found. Success rate will not be computed.")

    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    return env, success_term


def run_closed_loop(args: argparse.Namespace) -> None:
    client = Gr00tPolicyClient(
        host=args.server_host,
        port=args.server_port,
        api_token=args.server_api_token,
    )
    if not client.ping():
        raise RuntimeError(f"Cannot connect to GR00T server at {args.server_host}:{args.server_port}")
    print(f"Connected to GR00T server at {args.server_host}:{args.server_port}")

    env, success_term = make_env(args)
    camera_history = CameraHistoryBuffer(args.video_history_frames)
    state_history = StateHistoryBuffer(args.state_history_frames)
    successful_experiments = 0

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for exp_idx in range(args.num_total_experiments):
            print(f"\nStarting experiment {exp_idx + 1}/{args.num_total_experiments}")
            obs, _ = env.reset()
            camera_history.reset()
            state_history.reset()
            state_history.append(current_policy_state(env, obs, args))
            for _ in range(getattr(env.unwrapped.cfg, "num_rerenders_on_reset", 0)):
                env.sim.render()

            experiment_success = False
            frame_count = 0

            for inference_idx in range(args.max_inference_steps):
                observation = policy_observation(env, args, camera_history, state_history)
                action_dict, _ = client.get_action(observation)
                if args.debug:
                    print(f"[DEBUG] action_keys={sorted(action_dict.keys())}")
                    print(
                        "[DEBUG] video_shapes="
                        f"{ {key: tuple(value.shape) for key, value in observation['video'].items()} }"
                    )
                    print(
                        "[DEBUG] state_shapes="
                        f"{ {key: tuple(value.shape) for key, value in observation['state'].items()} }"
                    )
                action_chunk = parse_franka_action(action_dict, args).to(device=env.device)

                if args.debug:
                    print(
                        f"[DEBUG] inference={inference_idx} action_shape={tuple(action_chunk.shape)} "
                        f"first_action={action_chunk[0].cpu().numpy()}"
                    )

                for action in action_chunk:
                    obs, _, _, _, _ = env.step(action.reshape(1, -1))
                    state_history.append(current_policy_state(env, obs, args))
                    frame_count += 1

                    if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                        experiment_success = True
                        break

                if experiment_success:
                    successful_experiments += 1
                    print(
                        f"Experiment {exp_idx + 1} successful after {frame_count} env steps. "
                        f"Current SR: {successful_experiments}/{exp_idx + 1}"
                    )
                    break

            if not experiment_success:
                print(f"Experiment {exp_idx + 1} failed. Current SR: {successful_experiments}/{exp_idx + 1}")

    print("\nEvaluation Results:")
    print(f"Total experiments: {args.num_total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Success rate: {successful_experiments / max(args.num_total_experiments, 1) * 100.0:.2f}%")

    env.close()


if __name__ == "__main__":
    try:
        run_closed_loop(args_cli)
    except Exception:
        traceback.print_exc()
        if args_cli.pause_on_error and not args_cli.headless:
            input("Press Enter to close SimulationApp...")
        raise
    finally:
        simulation_app.close()
