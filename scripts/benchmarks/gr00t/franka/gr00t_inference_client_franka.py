# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import zmq

from isaaclab.app import AppLauncher

DEFAULT_EEF_TASK = "Isaac-Pick-Place-Franka-IK-Rel-Replay-Camera-v0"
DEFAULT_JOINT_TASK = "Isaac-Pick-Place-Franka-Joint-Position-Replay-Camera-v0"
SORTING_BOX_LOCAL_MIN_M = (-0.018910, -0.029798, -0.011343)
SORTING_BOX_LOCAL_MAX_M = (0.018677, 0.029969, 0.010947)
# Derived from the vertical inner-wall faces in the versioned sorting-bin USD meshes.
SORTING_BIN_INNER_HALF_EXTENTS_XY_M = (0.0960, 0.1194)


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


def derive_policy_noise_seed(episode_seed: int, repeat_index: int, inference_index: int) -> int:
    """Derive a stable cross-process seed for one policy inference call."""
    payload = f"franka_gr00t_policy_noise_v1:{episode_seed}:{repeat_index}:{inference_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


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
    parser.add_argument(
        "--policy-noise-repeat-index",
        type=int,
        default=None,
        help=(
            "Enable deterministic policy flow noise for paired evaluation. "
            "Use the same non-negative repeat index for both checkpoints."
        ),
    )
    parser.add_argument(
        "--verify-policy-action-determinism",
        action="store_true",
        help="Repeat the first policy request in each episode and require bit-identical raw actions.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-pos-delta", type=float, default=0.2)
    parser.add_argument("--max-rot-delta", type=float, default=0.2)
    parser.add_argument("--joint-gripper-open-threshold", type=float, default=0.04)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--pause-on-error", action="store_true")
    parser.add_argument("--hg-dagger", action="store_true", help="Collect human-gated joint-space corrections.")
    parser.add_argument("--dataset-file", type=Path, default=Path("datasets/franka_hg_dagger_round1.hdf5"))
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--teleop-device", choices=["spacemouse"], default="spacemouse")
    parser.add_argument("--intervention-key", default="B")
    parser.add_argument("--reset-key", default="R")
    parser.add_argument("--minimum-intervention-steps", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--max-joint-step", type=float, default=0.12)
    parser.add_argument("--spacemouse-pos-sensitivity", type=float, default=0.2)
    parser.add_argument("--spacemouse-rot-sensitivity", type=float, default=0.2)
    parser.add_argument("--policy-checkpoint-id", default=None)
    parser.add_argument("--baseline-dataset-id", default=None)
    parser.add_argument("--asset-version", default=None)
    parser.add_argument(
        "--episode-seeds-file",
        type=Path,
        default=None,
        help="JSON or text file containing exactly one deterministic seed per episode.",
    )
    parser.add_argument(
        "--eval-results-file",
        type=Path,
        default=None,
        help="Append-safe JSONL output for run, per-episode, and summary records.",
    )
    parser.add_argument("--overwrite-eval-results", action="store_true")
    parser.add_argument(
        "--failure-video-dir",
        type=Path,
        default=None,
        help="Optionally encode inference-point camera frames for failed episodes.",
    )
    parser.add_argument("--failure-video-fps", type=float, default=5.0)
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
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be >= 1")
    if args.num_total_experiments < 1:
        raise ValueError("--num-total-experiments must be >= 1")
    if args.max_inference_steps < 1:
        raise ValueError("--max-inference-steps must be >= 1")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0")
    if args.policy_noise_repeat_index is not None and args.policy_noise_repeat_index < 0:
        raise ValueError("--policy-noise-repeat-index must be >= 0")
    if args.verify_policy_action_determinism and args.policy_noise_repeat_index is None:
        raise ValueError("--verify-policy-action-determinism requires --policy-noise-repeat-index")
    if args.failure_video_fps <= 0:
        raise ValueError("--failure-video-fps must be > 0")
    if args.hg_dagger:
        if args.minimum_intervention_steps < args.num_feedback_actions:
            raise ValueError("--minimum-intervention-steps must be at least one action horizon")
        if args.policy_type != "joint_space":
            raise ValueError("--hg-dagger currently supports only --policy-type joint_space")
        if args.headless:
            raise ValueError("--hg-dagger requires a visible Kit window for the keyboard intervention gate")
        if not args.policy_checkpoint_id:
            raise ValueError("--hg-dagger requires --policy-checkpoint-id for reproducibility")
    elif args.eval_results_file is not None and not args.policy_checkpoint_id:
        raise ValueError("--eval-results-file requires --policy-checkpoint-id for reproducibility")
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
import warp as wp  # noqa: E402
from hg_dagger_core import InterventionGate  # noqa: E402
from hg_dagger_ik import FRANKA_ARM_JOINT_LIMITS, SpaceMouseJointIK  # noqa: E402
from hg_dagger_recorder import HGDAggerRecorder  # noqa: E402

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402

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

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action",
            {"observation": observation, "options": options},
        )
        if isinstance(response, list) and len(response) == 2:
            return response[0], response[1]
        if isinstance(response, tuple) and len(response) == 2:
            return response
        raise RuntimeError(f"Unexpected get_action response: {type(response)}")

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


@dataclass(frozen=True)
class ShadowPolicyResult:
    """Result of one takeover-time counterfactual query."""

    query_frame: int
    observation_state: np.ndarray
    action_dict: dict[str, np.ndarray] | None
    error: str | None = None


class AsyncShadowPolicy:
    """Run counterfactual takeover queries without blocking teleoperation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="franka-shadow-policy")
        self.future: Future | None = None
        self.query_frame = -1
        self.query_observation_state: np.ndarray | None = None

    def submit(self, observation: dict[str, Any], frame: int, observation_state: np.ndarray) -> bool:
        # A completed result must be consumed before its metadata can be
        # replaced by another query.
        if self.future is not None:
            return False

        def query():
            client = Gr00tPolicyClient(
                host=self.args.server_host,
                port=self.args.server_port,
                api_token=self.args.server_api_token,
            )
            try:
                return client.get_action(observation)
            finally:
                client.close()

        self.query_frame = frame
        self.query_observation_state = observation_state.copy()
        self.future = self.executor.submit(query)
        return True

    def poll(self) -> ShadowPolicyResult | None:
        if self.future is None or not self.future.done():
            return None
        return self._consume()

    def wait(self) -> ShadowPolicyResult | None:
        if self.future is None:
            return None
        return self._consume()

    def _consume(self) -> ShadowPolicyResult:
        future = self.future
        query_frame = self.query_frame
        observation_state = self.query_observation_state
        self.future = None
        if future is None or observation_state is None:
            raise RuntimeError("Shadow policy result metadata is incomplete")
        try:
            action_dict, _ = future.result()
        except Exception as exc:
            return ShadowPolicyResult(query_frame, observation_state, None, repr(exc))
        return ShadowPolicyResult(query_frame, observation_state, action_dict)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


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

        return {key: np.stack([frame[key] for frame in frames], axis=1) for key in frames[-1]}


def gripper_width_from_obs(gripper_pos: np.ndarray) -> np.ndarray:
    if gripper_pos.shape[-1] >= 2:
        width = np.abs(gripper_pos[:, 0] - gripper_pos[:, 1])
    else:
        width = gripper_pos[:, 0]
    return width.reshape(-1, 1).astype(np.float32)


def to_numpy(value: Any) -> np.ndarray:
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        try:
            value = wp.to_torch(value)
        except (AttributeError, RuntimeError, TypeError):
            pass
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_dict_sha256(action_dict: dict[str, np.ndarray]) -> str:
    """Hash a policy action dictionary without depending on key order."""
    digest = hashlib.sha256()
    for key in sorted(action_dict):
        array = np.ascontiguousarray(action_dict[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_episode_seeds(args: argparse.Namespace) -> tuple[list[int], dict[str, Any]]:
    if args.episode_seeds_file is None:
        seeds = [args.seed + episode_index for episode_index in range(args.num_total_experiments)]
        metadata = {
            "strategy": "base_seed_plus_episode_index",
            "base_seed": args.seed,
            "seeds_file": None,
            "seeds_file_sha256": None,
        }
    else:
        seeds_path = args.episode_seeds_file.expanduser().resolve()
        raw_text = seeds_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = [token for token in raw_text.replace(",", " ").split() if token]
        if isinstance(payload, dict):
            payload = payload.get("episode_seeds")
        if not isinstance(payload, list):
            raise ValueError("--episode-seeds-file must contain a JSON list, an episode_seeds list, or integers")
        seeds = [int(seed) for seed in payload]
        metadata = {
            "strategy": "explicit_seed_file",
            "base_seed": args.seed,
            "seeds_file": str(seeds_path),
            "seeds_file_sha256": file_sha256(seeds_path),
        }

    if len(seeds) != args.num_total_experiments:
        raise ValueError(
            f"Expected exactly {args.num_total_experiments} episode seeds, got {len(seeds)}"
        )
    if any(seed < 0 or seed > 2**31 - 1 for seed in seeds):
        raise ValueError("Episode seeds must be in [0, 2**31 - 1]")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Episode seeds must be unique")
    metadata["episode_seeds"] = seeds
    return seeds, metadata


class JsonlWriter:
    """Durable one-record-per-line evaluation log."""

    def __init__(self, path: Path | None, *, overwrite: bool):
        self.path = path.expanduser().resolve() if path is not None else None
        self.stream = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if overwrite else "x"
            self.stream = self.path.open(mode, encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self.stream is None:
            return
        record = {"timestamp_utc": utc_now(), **record}
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def combine_camera_frames(video_observation: dict[str, np.ndarray], camera_names: list[str]) -> np.ndarray:
    frames = [normalize_rgb(np.asarray(video_observation[name][0, -1])) for name in camera_names]
    max_height = max(frame.shape[0] for frame in frames)
    padded_frames = []
    for frame in frames:
        if frame.shape[0] == max_height:
            padded_frames.append(frame)
            continue
        padded = np.zeros((max_height, frame.shape[1], 3), dtype=np.uint8)
        padded[: frame.shape[0]] = frame
        padded_frames.append(padded)
    return np.concatenate(padded_frames, axis=1)


def write_failure_video(
    frames: list[np.ndarray],
    *,
    output_dir: Path | None,
    results_stem: str,
    policy_checkpoint_id: str,
    episode_index: int,
    seed: int,
    fps: float,
) -> tuple[str | None, str | None]:
    if output_dir is None or not frames:
        return None, None

    output_dir = output_dir.expanduser().resolve() / results_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_checkpoint = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in policy_checkpoint_id
    )
    output_path = output_dir / (
        f"{safe_checkpoint}_episode_{episode_index:03d}_seed_{seed}_failure.mp4"
    )
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(output_path, frames, fps=fps, macro_block_size=1)
    except Exception as exc:
        return None, repr(exc)
    return str(output_path), None


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


def joint_state_vector(env) -> np.ndarray:
    """Return the current 8D joint-space policy state for the first environment."""

    arm_joint_pos, gripper_width = franka_joint_state(env)
    return np.concatenate([arm_joint_pos[0], gripper_width[0]], axis=0).astype(np.float32)



def configured_entity_names(success_term) -> dict[str, str]:
    names = {
        "robot": "robot",
        "object_a": "object_a",
        "object_b": "object_b",
        "object_c": "object_c",
        "object_d": "object_d",
        "ee_frame": "ee_frame",
    }
    if success_term is None:
        return names
    for key in ("robot", "object_a", "object_b", "object_c", "object_d"):
        entity_cfg = success_term.params.get(f"{key}_cfg")
        if entity_cfg is not None:
            names[key] = entity_cfg.name
    return names


def placement_thresholds(success_term, env) -> dict[str, float | bool | None]:
    params = success_term.params if success_term is not None else {}
    env_cfg = env.unwrapped.cfg
    return {
        "xy_threshold": float(params.get("xy_threshold", 0.08)),
        "height_threshold": float(params.get("height_threshold", 0.04)),
        "height_diff": float(params.get("height_diff", 0.0)),
        "min_height_diff": (
            None if params.get("min_height_diff") is None else float(params["min_height_diff"])
        ),
        "max_height_diff": (
            None if params.get("max_height_diff") is None else float(params["max_height_diff"])
        ),
        "max_linear_velocity": (
            None if params.get("max_linear_velocity") is None else float(params["max_linear_velocity"])
        ),
        "max_angular_velocity": (
            None if params.get("max_angular_velocity") is None else float(params["max_angular_velocity"])
        ),
        "require_gripper_open": bool(params.get("require_gripper_open", True)),
        "gripper_open_val": float(getattr(env_cfg, "gripper_open_val", 0.04)),
        "gripper_threshold": float(getattr(env_cfg, "gripper_threshold", 0.005)),
    }


def first_env_vector(value: Any) -> np.ndarray:
    array = np.asarray(to_numpy(value), dtype=np.float64)
    return array[0]


def scene_snapshot(env, entity_names: dict[str, str]) -> dict[str, Any]:
    env_origin = first_env_vector(env.scene.env_origins)
    objects = {}
    for label in ("object_a", "object_b", "object_c", "object_d"):
        asset = env.scene[entity_names[label]]
        position_world = first_env_vector(asset.data.root_pos_w)
        quaternion_xyzw = first_env_vector(asset.data.root_quat_w)
        objects[label] = {
            "position_env": (position_world - env_origin).tolist(),
            "quaternion_xyzw": quaternion_xyzw.tolist(),
            # Kept for backward compatibility with schema v2 results; values are xyzw.
            "quaternion_wxyz": quaternion_xyzw.tolist(),
            "linear_velocity_world": first_env_vector(asset.data.root_lin_vel_w).tolist(),
            "angular_velocity_world": first_env_vector(asset.data.root_ang_vel_w).tolist(),
        }

    arm_joint_pos, gripper_width = franka_joint_state(env)
    robot = env.scene[entity_names["robot"]]
    finger_joint_ids, _ = robot.find_joints(["panda_finger_joint[1-2]"])
    joint_pos = to_numpy(robot.data.joint_pos).astype(np.float64)
    finger_joint_pos = joint_pos[:, joint_id_list(finger_joint_ids)]
    snapshot: dict[str, Any] = {
        "objects": objects,
        "robot": {
            "arm_joint_position": arm_joint_pos[0].astype(np.float64).tolist(),
            "finger_joint_position": finger_joint_pos[0].tolist(),
            "gripper_width": float(gripper_width[0, 0]),
        },
    }
    if entity_names["ee_frame"] in env.scene.keys():
        ee_frame = env.scene[entity_names["ee_frame"]]
        eef_position_world = np.asarray(to_numpy(ee_frame.data.target_pos_w), dtype=np.float64)[0, 0]
        eef_quaternion = np.asarray(to_numpy(ee_frame.data.target_quat_w), dtype=np.float64)[0, 0]
        snapshot["end_effector"] = {
            "position_env": (eef_position_world - env_origin).tolist(),
            "quaternion_xyzw": eef_quaternion.tolist(),
            # Kept for backward compatibility with schema v2 results; values are xyzw.
            "quaternion_wxyz": eef_quaternion.tolist(),
        }
    else:
        snapshot["end_effector"] = None
    return snapshot


def scene_signature(snapshot: dict[str, Any]) -> str:
    signature_payload = {
        "objects": {
            name: {
                "position_env": np.round(state["position_env"], 7).tolist(),
                "quaternion_wxyz": np.round(state["quaternion_wxyz"], 7).tolist(),
            }
            for name, state in snapshot["objects"].items()
        },
        "robot": {
            "arm_joint_position": np.round(snapshot["robot"]["arm_joint_position"], 7).tolist(),
            "gripper_width": round(snapshot["robot"]["gripper_width"], 7),
        },
    }
    canonical = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def object_in_bin_geometry(
    snapshot: dict[str, Any],
    object_name: str,
    bin_name: str,
    thresholds: dict[str, float | None],
) -> bool:
    object_position = np.asarray(snapshot["objects"][object_name]["position_env"])
    bin_position = np.asarray(snapshot["objects"][bin_name]["position_env"])
    difference = object_position - bin_position
    xy_ok = np.linalg.norm(difference[:2]) < thresholds["xy_threshold"]
    z_difference = float(difference[2])
    z_ok = abs(abs(z_difference) - thresholds["height_diff"]) < thresholds["height_threshold"]
    if thresholds["min_height_diff"] is not None:
        z_ok = z_ok and z_difference > thresholds["min_height_diff"]
    if thresholds["max_height_diff"] is not None:
        z_ok = z_ok and z_difference < thresholds["max_height_diff"]
    return bool(xy_ok and z_ok)


def placement_geometry_flags(
    snapshot: dict[str, Any],
    thresholds: dict[str, float | bool | None],
) -> dict[str, bool]:
    return {
        "a_in_blue_c": object_in_bin_geometry(snapshot, "object_a", "object_c", thresholds),
        "b_in_black_d": object_in_bin_geometry(snapshot, "object_b", "object_d", thresholds),
        "a_in_wrong_black_d": object_in_bin_geometry(snapshot, "object_a", "object_d", thresholds),
        "b_in_wrong_blue_c": object_in_bin_geometry(snapshot, "object_b", "object_c", thresholds),
    }


def rotation_matrix_from_quaternion_xyzw(quaternion: list[float]) -> np.ndarray:
    """Convert an xyzw quaternion to a 3D rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    scale = 2.0 / float(x * x + y * y + z * z + w * w)
    matrix = np.empty((3, 3), dtype=np.float64)
    matrix[0] = (1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w))
    matrix[1] = (scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w))
    matrix[2] = (scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y))
    return matrix


def object_in_bin_containment(
    snapshot: dict[str, Any],
    object_name: str,
    bin_name: str,
    thresholds: dict[str, float | bool | None],
) -> bool:
    object_state = snapshot["objects"][object_name]
    bin_state = snapshot["objects"][bin_name]
    object_position = np.asarray(object_state["position_env"], dtype=np.float64)
    bin_position = np.asarray(bin_state["position_env"], dtype=np.float64)
    object_quaternion = object_state.get("quaternion_xyzw", object_state["quaternion_wxyz"])
    bin_quaternion = bin_state.get("quaternion_xyzw", bin_state["quaternion_wxyz"])
    box_min = np.asarray(SORTING_BOX_LOCAL_MIN_M, dtype=np.float64)
    box_max = np.asarray(SORTING_BOX_LOCAL_MAX_M, dtype=np.float64)
    box_corners = np.array(
        [
            [x, y, z]
            for x in (box_min[0], box_max[0])
            for y in (box_min[1], box_max[1])
            for z in (box_min[2], box_max[2])
        ],
        dtype=np.float64,
    )
    object_rotation = rotation_matrix_from_quaternion_xyzw(object_quaternion)
    bin_rotation = rotation_matrix_from_quaternion_xyzw(bin_quaternion)
    world_corners = box_corners @ object_rotation.T + object_position
    bin_local_corners = (world_corners - bin_position) @ bin_rotation
    max_abs_xy = np.max(np.abs(bin_local_corners[:, :2]), axis=0)
    inner_half_extents = np.asarray(SORTING_BIN_INNER_HALF_EXTENTS_XY_M)
    footprint_inside = bool(
        np.all(max_abs_xy <= inner_half_extents)
    )
    difference = object_position - bin_position
    z_difference = float(difference[2])
    height_ok = (
        abs(abs(z_difference) - float(thresholds["height_diff"]))
        < float(thresholds["height_threshold"])
    )
    if thresholds["min_height_diff"] is not None:
        height_ok = height_ok and z_difference > float(thresholds["min_height_diff"])
    if thresholds["max_height_diff"] is not None:
        height_ok = height_ok and z_difference < float(thresholds["max_height_diff"])
    linear_speed = float(np.linalg.norm(object_state["linear_velocity_world"]))
    angular_speed = float(np.linalg.norm(object_state["angular_velocity_world"]))
    stable = True
    max_linear_velocity = thresholds.get("max_linear_velocity")
    if max_linear_velocity is not None:
        stable = stable and linear_speed < float(max_linear_velocity)
    max_angular_velocity = thresholds.get("max_angular_velocity")
    if max_angular_velocity is not None:
        stable = stable and angular_speed < float(max_angular_velocity)
    return bool(footprint_inside and height_ok and stable)


def placement_containment(
    snapshot: dict[str, Any],
    thresholds: dict[str, float | bool | None],
) -> dict[str, Any]:
    """Evaluate final task success using bin-local object-footprint containment."""
    flags = {
        "a_in_blue_c": object_in_bin_containment(snapshot, "object_a", "object_c", thresholds),
        "b_in_black_d": object_in_bin_containment(snapshot, "object_b", "object_d", thresholds),
        "a_in_wrong_black_d": object_in_bin_containment(
            snapshot, "object_a", "object_d", thresholds
        ),
        "b_in_wrong_blue_c": object_in_bin_containment(
            snapshot, "object_b", "object_c", thresholds
        ),
    }
    if not thresholds.get("require_gripper_open", True):
        gripper_released = True
    else:
        open_value = float(thresholds.get("gripper_open_val", 0.04))
        tolerance = float(thresholds.get("gripper_threshold", 0.005))
        finger_positions = snapshot["robot"].get("finger_joint_position")
        if finger_positions is None:
            gripper_released = bool(
                snapshot["robot"]["gripper_width"] >= 2.0 * (open_value - tolerance)
            )
        else:
            finger_positions_array = np.asarray(finger_positions, dtype=np.float64)
            gripper_released = bool(
                np.all(np.abs(np.abs(finger_positions_array) - open_value) < tolerance)
            )
    success = bool(flags["a_in_blue_c"] and flags["b_in_black_d"] and gripper_released)
    return {
        "geometry_version": "bin_local_box_footprint_v1",
        "box_local_min_m": list(SORTING_BOX_LOCAL_MIN_M),
        "box_local_max_m": list(SORTING_BOX_LOCAL_MAX_M),
        "bin_inner_half_extents_xy_m": list(SORTING_BIN_INNER_HALF_EXTENTS_XY_M),
        "flags": flags,
        "gripper_released": gripper_released,
        "success": success,
    }


class EpisodeDiagnostics:
    """Accumulate auditable motion facts; the bucket is only a heuristic summary."""

    HEURISTIC_VERSION = 1

    def __init__(
        self,
        initial_snapshot: dict[str, Any],
        thresholds: dict[str, float | bool | None],
    ):
        self.initial_snapshot = initial_snapshot
        self.thresholds = thresholds
        self.max_displacement = {"object_a": 0.0, "object_b": 0.0}
        self.max_lift = {"object_a": 0.0, "object_b": 0.0}
        self.max_linear_speed = {"object_a": 0.0, "object_b": 0.0}
        self.min_eef_distance = {"object_a": float("inf"), "object_b": float("inf")}
        self.min_gripper_width_near = {"object_a": None, "object_b": None}
        self.ever_geometry = placement_geometry_flags(initial_snapshot, thresholds)
        self.update(initial_snapshot)

    def update(self, snapshot: dict[str, Any]) -> None:
        eef_state = snapshot.get("end_effector")
        eef_position = None if eef_state is None else np.asarray(eef_state["position_env"])
        gripper_width = snapshot["robot"]["gripper_width"]

        for object_name in ("object_a", "object_b"):
            initial_position = np.asarray(
                self.initial_snapshot["objects"][object_name]["position_env"]
            )
            state = snapshot["objects"][object_name]
            position = np.asarray(state["position_env"])
            self.max_displacement[object_name] = max(
                self.max_displacement[object_name],
                float(np.linalg.norm(position - initial_position)),
            )
            self.max_lift[object_name] = max(
                self.max_lift[object_name],
                float(position[2] - initial_position[2]),
            )
            self.max_linear_speed[object_name] = max(
                self.max_linear_speed[object_name],
                float(np.linalg.norm(state["linear_velocity_world"])),
            )
            if eef_position is not None:
                distance = float(np.linalg.norm(position - eef_position))
                self.min_eef_distance[object_name] = min(
                    self.min_eef_distance[object_name], distance
                )
                if distance < 0.15:
                    previous = self.min_gripper_width_near[object_name]
                    self.min_gripper_width_near[object_name] = (
                        gripper_width if previous is None else min(previous, gripper_width)
                    )

        current_geometry = placement_geometry_flags(snapshot, self.thresholds)
        self.ever_geometry = {
            key: self.ever_geometry[key] or value for key, value in current_geometry.items()
        }

    def result(
        self,
        *,
        final_snapshot: dict[str, Any],
        success: bool,
        drop_reason: str | None,
    ) -> dict[str, Any]:
        final_geometry = placement_geometry_flags(final_snapshot, self.thresholds)
        final_containment = placement_containment(final_snapshot, self.thresholds)
        failure_bucket = self.failure_bucket(
            success=success,
            drop_reason=drop_reason,
            final_geometry=final_geometry,
        )
        min_eef_distance = {
            key: None if not np.isfinite(value) else value
            for key, value in self.min_eef_distance.items()
        }
        return {
            "failure_bucket": failure_bucket,
            "failure_bucket_heuristic_version": self.HEURISTIC_VERSION,
            "drop_reason": drop_reason,
            "placement_thresholds": self.thresholds,
            "max_displacement_m": self.max_displacement,
            "max_lift_m": self.max_lift,
            "max_linear_speed_m_s": self.max_linear_speed,
            "min_eef_distance_m": min_eef_distance,
            "min_gripper_width_near_object_m": self.min_gripper_width_near,
            "ever_geometry": self.ever_geometry,
            "final_geometry": final_geometry,
            "final_containment": final_containment,
        }

    def failure_bucket(
        self,
        *,
        success: bool,
        drop_reason: str | None,
        final_geometry: dict[str, bool],
    ) -> str:
        if success:
            return "success"
        if drop_reason is not None:
            return "object_dropped"
        if (
            self.ever_geometry["a_in_wrong_black_d"]
            or self.ever_geometry["b_in_wrong_blue_c"]
        ):
            return "wrong_bin"
        if final_geometry["a_in_blue_c"] and final_geometry["b_in_black_d"]:
            return "release_or_settle"
        if final_geometry["a_in_blue_c"] or final_geometry["b_in_black_d"]:
            return "partial_completion"
        max_displacement = max(self.max_displacement.values())
        if max_displacement < 0.03:
            min_distance_values = [
                value for value in self.min_eef_distance.values() if np.isfinite(value)
            ]
            if min_distance_values and min(min_distance_values) < 0.15:
                return "hover_or_grasp_failure"
            return "no_object_approach"
        return "placement_incomplete"


def triggered_termination(env, termination_terms: dict[str, Any]) -> str | None:
    for name, term in termination_terms.items():
        if bool(term.func(env, **term.params)[0]):
            return name
    return None
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
    arm_joint_pos = np.clip(
        arm_joint_pos[:action_horizon, :7], FRANKA_ARM_JOINT_LIMITS[:, 0], FRANKA_ARM_JOINT_LIMITS[:, 1]
    )
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
    if args.hg_dagger:
        dataset_path = args.dataset_file.expanduser().resolve()
        if dataset_path.exists() and not args.overwrite_dataset:
            raise FileExistsError(f"Dataset already exists; pass --overwrite-dataset: {dataset_path}")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if dataset_path.exists():
            dataset_path.unlink()

        env_cfg.env_name = args.task
        env_cfg.observations.policy.concatenate_terms = False
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
        env_cfg.recorders.dataset_filename = dataset_path.stem
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.export_in_record_pre_reset = False
    else:
        env_cfg.recorders = {}

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        print("No success termination term was found. Success rate will not be computed.")

    drop_terms = {}
    for term_name in ("object_dropped", "object_b_dropped"):
        if hasattr(env_cfg.terminations, term_name):
            term = getattr(env_cfg.terminations, term_name)
            if term is not None:
                drop_terms[term_name] = term
    # The client owns every episode boundary; auto-reset could splice scenes.
    env_cfg.terminations = {}

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    return env, success_term, drop_terms


def write_collection_manifest(
    args: argparse.Namespace,
    *,
    successful_episodes: int,
    completed_episodes: int,
) -> None:
    """Write reproducibility metadata beside the collected HDF5."""

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    dataset_path = args.dataset_file.expanduser().resolve()
    manifest = {
        "schema_version": 1,
        "dataset_file": str(dataset_path),
        "task": args.task,
        "policy_type": args.policy_type,
        "policy_checkpoint_id": args.policy_checkpoint_id,
        "baseline_dataset_id": args.baseline_dataset_id,
        "asset_version": args.asset_version,
        "git_commit": git_commit,
        "seed": args.seed,
        "action_horizon": args.num_feedback_actions,
        "state_history_frames": args.state_history_frames,
        "video_history_frames": args.video_history_frames,
        "minimum_intervention_steps": args.minimum_intervention_steps,
        "max_episode_steps": args.max_episode_steps,
        "completed_episodes": completed_episodes,
        "successful_episodes": successful_episodes,
    }
    manifest_path = dataset_path.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def run_hg_dagger(args: argparse.Namespace) -> None:
    """Collect full joint-space episodes with human-gated recovery metadata."""

    client = Gr00tPolicyClient(
        host=args.server_host,
        port=args.server_port,
        api_token=args.server_api_token,
    )
    if not client.ping():
        raise RuntimeError(f"Cannot connect to GR00T server at {args.server_host}:{args.server_port}")

    env, success_term, _ = make_env(args)
    camera_history = CameraHistoryBuffer(args.video_history_frames)
    state_history = StateHistoryBuffer(args.state_history_frames)
    gate = InterventionGate(args.minimum_intervention_steps)
    ik = SpaceMouseJointIK(env, args.max_joint_step)
    recorder = HGDAggerRecorder(env.recorder_manager, env.device, state_dim=8, action_dim=8)
    shadow = AsyncShadowPolicy(args)

    def record_shadow_result(result: ShadowPolicyResult) -> None:
        action = np.zeros(8, dtype=np.float32)
        valid = False
        error = result.error
        if result.action_dict is not None:
            try:
                action_chunk = parse_joint_space_action(result.action_dict, args)
                if len(action_chunk) < 1:
                    raise ValueError("Shadow policy returned an empty action chunk")
                action = action_chunk[0].cpu().numpy()
                valid = bool(np.all(np.isfinite(action)))
                if not valid:
                    error = "Shadow policy returned a non-finite action"
            except Exception as exc:
                error = repr(exc)
        recorder.record_shadow_query(
            observation=result.observation_state,
            policy_action=action,
            policy_action_valid=valid,
            query_frame=result.query_frame,
        )
        if error is not None:
            print(f"Warning: shadow query at frame {result.query_frame} was invalid: {error}")

    spacemouse = Se3SpaceMouse(
        Se3SpaceMouseCfg(
            pos_sensitivity=args.spacemouse_pos_sensitivity,
            rot_sensitivity=args.spacemouse_rot_sensitivity,
            sim_device=env.device,
        )
    )
    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.01, rot_sensitivity=0.01, sim_device=env.device))
    toggle_requested = False
    reset_requested = False

    def request_toggle() -> None:
        nonlocal toggle_requested
        toggle_requested = True

    def request_reset() -> None:
        nonlocal reset_requested
        reset_requested = True

    keyboard.add_callback(args.intervention_key, request_toggle)
    keyboard.add_callback(args.reset_key, request_reset)

    successful_episodes = 0
    print(
        f"HG-DAgger ready. Press {args.intervention_key} to take over/release and "
        f"{args.reset_key} to discard and retry the current episode."
    )

    try:
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            episode_index = 0
            while episode_index < args.num_total_experiments:
                obs, _ = env.reset()
                camera_history.reset()
                state_history.reset()
                state_history.append(current_policy_state(env, obs, args))
                for _ in range(getattr(env.unwrapped.cfg, "num_rerenders_on_reset", 0)):
                    env.sim.render()
                gate.reset()
                recorder.reset()
                spacemouse.reset()
                toggle_requested = False
                reset_requested = False
                action_chunk: torch.Tensor | None = None
                chunk_index = 0
                inference_id = -1
                episode_success = False
                outcome = "policy_failure"

                for frame_index in range(args.max_episode_steps):
                    if reset_requested:
                        outcome = "human_abort"
                        break

                    shadow_result = shadow.poll()
                    if shadow_result is not None:
                        record_shadow_result(shadow_result)

                    transition = None
                    if toggle_requested:
                        toggle_requested = False
                        transition = gate.toggle()
                        if transition.activated or transition.released:
                            action_chunk = None
                            chunk_index = 0
                        if transition.activated:
                            shadow_observation = policy_observation(env, args, camera_history, state_history)
                            submitted = shadow.submit(
                                shadow_observation,
                                frame_index,
                                joint_state_vector(env),
                            )
                            if not submitted:
                                recorder.record_shadow_query(
                                    observation=joint_state_vector(env),
                                    policy_action=np.zeros(8, dtype=np.float32),
                                    policy_action_valid=False,
                                    query_frame=frame_index,
                                )
                                print(
                                    f"Warning: shadow query at frame {frame_index} was skipped "
                                    "because the previous query is still running"
                                )
                            print(f"Episode {episode_index}: human takeover at frame {frame_index}")
                        elif transition.released:
                            print(f"Episode {episode_index}: policy resumed at frame {frame_index}")

                    pre_step_state = joint_state_vector(env)
                    policy_action_valid = False
                    policy_action = np.zeros(8, dtype=np.float32)
                    expert_action = np.zeros(8, dtype=np.float32)

                    if gate.active:
                        expert_tensor = ik.command(spacemouse.advance())
                        executed_tensor = expert_tensor
                        expert_action = expert_tensor[0].detach().cpu().numpy()
                    else:
                        if action_chunk is None or chunk_index >= len(action_chunk):
                            observation = policy_observation(env, args, camera_history, state_history)
                            action_dict, _ = client.get_action(observation)
                            action_chunk = parse_joint_space_action(action_dict, args).to(env.device)
                            inference_id += 1
                            chunk_index = 0
                        executed_tensor = action_chunk[chunk_index].reshape(1, -1)
                        policy_action = executed_tensor[0].detach().cpu().numpy()
                        policy_action_valid = True

                    recorder.record_pre_step(
                        observation=pre_step_state,
                        policy_action=policy_action,
                        expert_action=expert_action,
                        executed_action=executed_tensor[0].detach().cpu().numpy(),
                        intervention=gate.active,
                        policy_action_valid=policy_action_valid,
                        inference_id=inference_id,
                        chunk_index=chunk_index if not gate.active else -1,
                        frame_index=frame_index,
                    )
                    obs, _, _, _, _ = env.step(executed_tensor)
                    recorder.record_post_step(joint_state_vector(env))
                    state_history.append(current_policy_state(env, obs, args))

                    if gate.active:
                        gate_transition = gate.complete_step()
                        if gate_transition.released:
                            action_chunk = None
                            chunk_index = 0
                            print(f"Episode {episode_index}: deferred release completed at frame {frame_index}")
                    else:
                        chunk_index += 1

                    if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                        episode_success = True
                        outcome = "success"
                        break
                else:
                    outcome = "timeout"

                shadow_result = shadow.wait()
                if shadow_result is not None:
                    record_shadow_result(shadow_result)
                if outcome == "human_abort":
                    env.recorder_manager.reset([0])
                    print(f"Episode {episode_index + 1}/{args.num_total_experiments}: discarded; retrying.")
                    continue

                recorder.finish_episode(
                    success=episode_success,
                    outcome=outcome,
                    seed=args.seed,
                    policy_checkpoint_id=args.policy_checkpoint_id,
                )
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([episode_success], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                env.recorder_manager.reset([0])
                successful_episodes += int(episode_success)
                print(
                    f"Episode {episode_index + 1}/{args.num_total_experiments}: {outcome}; "
                    f"intervention={recorder.summary.intervention_ratio:.1%}; "
                    f"SR={successful_episodes}/{episode_index + 1}"
                )
                episode_index += 1
    finally:
        shadow.close()
        client.close()
        completed_episodes = (
            env.recorder_manager.exported_successful_episode_count
            + env.recorder_manager.exported_failed_episode_count
        )
        env.close()
        write_collection_manifest(
            args,
            successful_episodes=successful_episodes,
            completed_episodes=completed_episodes,
        )


def run_closed_loop(args: argparse.Namespace) -> None:
    episode_seeds, seed_metadata = resolve_episode_seeds(args)
    writer = JsonlWriter(args.eval_results_file, overwrite=args.overwrite_eval_results)
    client = None
    env = None
    successful_experiments = 0
    containment_successful_experiments = 0
    completed_experiments = 0
    interrupted = False
    started_at = time.monotonic()
    started_utc = utc_now()
    run_id = hashlib.sha256(
        f"{started_utc}:{os.getpid()}:{args.policy_checkpoint_id}".encode()
    ).hexdigest()[:16]
    results_stem = writer.path.stem if writer.path is not None else f"closed_loop_{run_id}"

    try:
        client = Gr00tPolicyClient(
            host=args.server_host,
            port=args.server_port,
            api_token=args.server_api_token,
        )
        if not client.ping():
            raise RuntimeError(f"Cannot connect to GR00T server at {args.server_host}:{args.server_port}")
        print(f"Connected to GR00T server at {args.server_host}:{args.server_port}")

        writer.write(
            {
                "record_type": "run_start",
                "schema_version": 3,
                "run_id": run_id,
                "started_utc": started_utc,
                "task": args.task,
                "policy_type": args.policy_type,
                "policy_checkpoint_id": args.policy_checkpoint_id,
                "baseline_dataset_id": args.baseline_dataset_id,
                "asset_version": args.asset_version,
                "language_instruction": args.language_instruction,
                "server": {"host": args.server_host, "port": args.server_port},
                "headless": args.headless,
                "device": args.device,
                "requested_experiments": args.num_total_experiments,
                "max_inference_steps": args.max_inference_steps,
                "num_feedback_actions": args.num_feedback_actions,
                "video_history_frames": args.video_history_frames,
                "state_history_frames": args.state_history_frames,
                "camera_names": list(args.camera_names),
                "seed_protocol": seed_metadata,
                "policy_noise_protocol": (
                    None
                    if args.policy_noise_repeat_index is None
                    else {
                        "strategy": "sha256_episode_repeat_inference_v1",
                        "repeat_index": args.policy_noise_repeat_index,
                    }
                ),
                "verify_policy_action_determinism": args.verify_policy_action_determinism,
                "episode_boundary_protocol": "explicit_reset_no_env_auto_reset_v1",
                "scene_signature_protocol": "sha256_pose_and_robot_state_rounded_1e-7_v1",
                "failure_bucket_heuristic_version": EpisodeDiagnostics.HEURISTIC_VERSION,
                "scene_quaternion_order": "xyzw",
                "containment_geometry_version": "bin_local_box_footprint_v1",
                "failure_video_dir": (
                    None
                    if args.failure_video_dir is None
                    else str(args.failure_video_dir.expanduser().resolve())
                ),
                "failure_video_fps": args.failure_video_fps,
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": file_sha256(Path(__file__).resolve()),
                "git_commit": current_git_commit(),
            }
        )

        env, success_term, drop_terms = make_env(args)
        entity_names = configured_entity_names(success_term)
        thresholds = placement_thresholds(success_term, env)
        camera_history = CameraHistoryBuffer(args.video_history_frames)
        state_history = StateHistoryBuffer(args.state_history_frames)

        try:
            with torch.inference_mode():
                for episode_index, episode_seed in enumerate(episode_seeds):
                    print(
                        f"\nStarting experiment {episode_index + 1}/{args.num_total_experiments} "
                        f"with seed={episode_seed}"
                    )
                    torch.manual_seed(episode_seed)
                    np.random.seed(episode_seed)
                    env.seed(episode_seed)
                    obs, _ = env.reset()
                    camera_history.reset()
                    state_history.reset()
                    state_history.append(current_policy_state(env, obs, args))
                    for _ in range(getattr(env.unwrapped.cfg, "num_rerenders_on_reset", 0)):
                        env.sim.render()

                    initial_snapshot = scene_snapshot(env, entity_names)
                    initial_signature = scene_signature(initial_snapshot)
                    diagnostics = EpisodeDiagnostics(initial_snapshot, thresholds)
                    failure_video_frames: list[np.ndarray] = []
                    experiment_success = False
                    drop_reason = None
                    frame_count = 0
                    inference_count = 0
                    inference_trace: list[dict[str, Any]] = []
                    episode_started_at = time.monotonic()

                    for inference_index in range(args.max_inference_steps):
                        observation = policy_observation(env, args, camera_history, state_history)
                        if args.failure_video_dir is not None:
                            failure_video_frames.append(
                                combine_camera_frames(observation["video"], list(args.camera_names))
                            )
                        observation_sha256 = hashlib.sha256(
                            MsgSerializer.to_bytes(observation)
                        ).hexdigest()
                        observation_component_sha256 = {
                            key: hashlib.sha256(
                                MsgSerializer.to_bytes(observation[key])
                            ).hexdigest()
                            for key in sorted(observation)
                        }
                        inference_seed = None
                        policy_options = None
                        if args.policy_noise_repeat_index is not None:
                            inference_seed = derive_policy_noise_seed(
                                episode_seed=episode_seed,
                                repeat_index=args.policy_noise_repeat_index,
                                inference_index=inference_index,
                            )
                            policy_options = {"inference_seed": inference_seed}
                        action_dict, _ = client.get_action(
                            observation,
                            options=policy_options,
                        )
                        action_sha256 = action_dict_sha256(action_dict)
                        if args.verify_policy_action_determinism and inference_index == 0:
                            repeated_action_dict, _ = client.get_action(
                                observation,
                                options=policy_options,
                            )
                            repeated_sha256 = action_dict_sha256(repeated_action_dict)
                            if repeated_sha256 != action_sha256:
                                raise RuntimeError(
                                    "Policy returned different raw actions for identical observation "
                                    f"and inference_seed={inference_seed}: {action_sha256} != {repeated_sha256}"
                                )
                        inference_trace.append(
                            {
                                "inference_index": inference_index,
                                "inference_seed": inference_seed,
                                "observation_sha256": observation_sha256,
                                "observation_component_sha256": observation_component_sha256,
                                "raw_action_sha256": action_sha256,
                                "scene": scene_snapshot(env, entity_names),
                            }
                        )
                        inference_count += 1
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
                                f"[DEBUG] inference={inference_index} action_shape={tuple(action_chunk.shape)} "
                                f"first_action={action_chunk[0].cpu().numpy()}"
                            )

                        for action in action_chunk:
                            obs, _, _, _, _ = env.step(action.reshape(1, -1))
                            state_history.append(current_policy_state(env, obs, args))
                            frame_count += 1
                            diagnostics.update(scene_snapshot(env, entity_names))

                            if success_term is not None and bool(
                                success_term.func(env, **success_term.params)[0]
                            ):
                                experiment_success = True
                                break
                            drop_reason = triggered_termination(env, drop_terms)
                            if drop_reason is not None:
                                break

                        if experiment_success or drop_reason is not None:
                            break

                    final_snapshot = scene_snapshot(env, entity_names)
                    diagnostic_result = diagnostics.result(
                        final_snapshot=final_snapshot,
                        success=experiment_success,
                        drop_reason=drop_reason,
                    )
                    if experiment_success:
                        successful_experiments += 1
                    containment_success = bool(diagnostic_result["final_containment"]["success"])
                    if containment_success:
                        containment_successful_experiments += 1
                    completed_experiments += 1

                    failure_video_path = None
                    failure_video_error = None
                    if not experiment_success:
                        failure_video_path, failure_video_error = write_failure_video(
                            failure_video_frames,
                            output_dir=args.failure_video_dir,
                            results_stem=results_stem,
                            policy_checkpoint_id=args.policy_checkpoint_id or "unknown_checkpoint",
                            episode_index=episode_index,
                            seed=episode_seed,
                            fps=args.failure_video_fps,
                        )

                    termination_reason = (
                        "success"
                        if experiment_success
                        else drop_reason or "max_inference_steps"
                    )
                    writer.write(
                        {
                            "record_type": "episode",
                            "schema_version": 3,
                            "run_id": run_id,
                            "episode_index": episode_index,
                            "episode_number": episode_index + 1,
                            "seed": episode_seed,
                            "initial_scene_signature": initial_signature,
                            "success": experiment_success,
                            "strict_success": experiment_success,
                            "containment_success": containment_success,
                            "termination_reason": termination_reason,
                            "failure_bucket": diagnostic_result["failure_bucket"],
                            "env_steps": frame_count,
                            "inference_calls": inference_count,
                            "inference_trace": inference_trace,
                            "duration_seconds": time.monotonic() - episode_started_at,
                            "initial_scene": initial_snapshot,
                            "final_scene": final_snapshot,
                            "diagnostics": diagnostic_result,
                            "failure_video_path": failure_video_path,
                            "failure_video_error": failure_video_error,
                        }
                    )

                    outcome = "success" if experiment_success else diagnostic_result["failure_bucket"]
                    print(
                        f"Experiment {episode_index + 1}: {outcome}; seed={episode_seed}; "
                        f"signature={initial_signature[:12]}; steps={frame_count}; "
                        f"SR={successful_experiments}/{completed_experiments}"
                        f"; containment_SR={containment_successful_experiments}/{completed_experiments}"
                    )
        except KeyboardInterrupt:
            interrupted = True
            print("\nEvaluation interrupted; completed episode records were preserved.")
    except Exception as exc:
        writer.write(
            {
                "record_type": "run_error",
                "schema_version": 3,
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "completed_experiments": completed_experiments,
            }
        )
        raise
    finally:
        writer.write(
            {
                "record_type": "run_summary",
                "schema_version": 3,
                "run_id": run_id,
                "requested_experiments": args.num_total_experiments,
                "completed_experiments": completed_experiments,
                "successful_experiments": successful_experiments,
                "containment_successful_experiments": containment_successful_experiments,
                "containment_success_rate": (
                    None if completed_experiments == 0
                    else containment_successful_experiments / completed_experiments
                ),
                "success_rate": (
                    None
                    if completed_experiments == 0
                    else successful_experiments / completed_experiments
                ),
                "completed_episode_seeds": episode_seeds[:completed_experiments],
                "interrupted": interrupted,
                "duration_seconds": time.monotonic() - started_at,
            }
        )
        if env is not None:
            env.close()
        if client is not None:
            client.close()
        writer.close()

    print("\nEvaluation Results:")
    print(f"Completed experiments: {completed_experiments}/{args.num_total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Containment-successful experiments: {containment_successful_experiments}")
    success_rate = successful_experiments / max(completed_experiments, 1) * 100.0
    print(f"Success rate over completed experiments: {success_rate:.2f}%")
    containment_success_rate = containment_successful_experiments / max(completed_experiments, 1) * 100.0
    print(f"Containment success rate: {containment_success_rate:.2f}%")


if __name__ == "__main__":
    try:
        if args_cli.hg_dagger:
            run_hg_dagger(args_cli)
        else:
            run_closed_loop(args_cli)
    except Exception:
        traceback.print_exc()
        if args_cli.pause_on_error and not args_cli.headless:
            input("Press Enter to close SimulationApp...")
        raise
    finally:
        simulation_app.close()
