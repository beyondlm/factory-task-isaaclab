# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING
from pathlib import Path

import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.pick_and_place import mdp as pick_and_place_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events


SORTING_ASSET_DATA_DIR = Path("/home/npnsa/workspace/jeff/customer/jd")

SORTING_SCENE_SCALE = 0.6
SORTING_Z_OFFSET = 0.6
# Quaternion order is xyzw. Place the Franka beyond the bins and face it back toward the table.
SORTING_ROBOT_POS = (0.75, 0.0, SORTING_Z_OFFSET)
SORTING_ROBOT_YAW_180_QUAT = (0.0, 0.0, 1.0, 0.0)
SORTING_BELT_POS = (0.0, 0.0, 0.0)
SORTING_BELT_SCALE = (SORTING_SCENE_SCALE, SORTING_SCENE_SCALE, SORTING_SCENE_SCALE)
SORTING_BOX_SCALE = (1.0, 1.0, 1.0)
SORTING_CONTACT_OFFSET = 0.02
SORTING_REST_OFFSET = 0.0
SORTING_BELT_SURFACE_Z = 1.781 * SORTING_SCENE_SCALE
SORTING_BELT_PROXY_THICKNESS = 0.04
SORTING_BELT_PROXY_SIZE = (4.0 * SORTING_SCENE_SCALE, 0.9 * SORTING_SCENE_SCALE, SORTING_BELT_PROXY_THICKNESS)
SORTING_BELT_PROXY_POS = (
    -2.0 * SORTING_SCENE_SCALE,
    0.0,
    SORTING_BELT_SURFACE_Z - SORTING_BELT_PROXY_THICKNESS / 2.0,
)
SORTING_TABLE_SURFACE_Z = 0.994 * SORTING_SCENE_SCALE
SORTING_TABLE_PROXY_THICKNESS = 0.04
SORTING_TABLE_PROXY_SIZE = (0.8 * SORTING_SCENE_SCALE, 2.2 * SORTING_SCENE_SCALE, SORTING_TABLE_PROXY_THICKNESS)
SORTING_TABLE_PROXY_POS = (
    0.2 * SORTING_SCENE_SCALE,
    0.0,
    SORTING_TABLE_SURFACE_Z - SORTING_TABLE_PROXY_THICKNESS / 2.0,
)
SORTING_CONVEYOR_VELOCITY = (0.35, 0.0, 0.0)
SORTING_CONVEYOR_X_RANGE = (-4.0 * SORTING_SCENE_SCALE, 0.05)
SORTING_CONVEYOR_Y_RANGE = (-0.45 * SORTING_SCENE_SCALE, 0.45 * SORTING_SCENE_SCALE)
SORTING_CONVEYOR_Z_RANGE = (SORTING_BELT_SURFACE_Z - 0.05, SORTING_BELT_SURFACE_Z + 0.12)
SORTING_CONVEYOR_INTERVAL_S = 0.025
SORTING_OBJECT_DROP_HEIGHT_ABOVE_SURFACE = 0.05
SORTING_OBJECT_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
SORTING_OBJECT_X = -0.20
SORTING_OBJECT_LEFT_Y = -0.10
SORTING_OBJECT_RIGHT_Y = 0.10
SORTING_OBJECT_A_Y_RANGE = (SORTING_OBJECT_LEFT_Y, SORTING_OBJECT_LEFT_Y)
SORTING_OBJECT_B_Y_RANGE = (SORTING_OBJECT_RIGHT_Y, SORTING_OBJECT_RIGHT_Y)
SORTING_OBJECT_Z = SORTING_BELT_SURFACE_Z + SORTING_OBJECT_DROP_HEIGHT_ABOVE_SURFACE
SORTING_OBJECT_A_INIT_POS = (SORTING_OBJECT_X, SORTING_OBJECT_A_Y_RANGE[0], SORTING_OBJECT_Z)
SORTING_OBJECT_B_INIT_POS = (SORTING_OBJECT_X, SORTING_OBJECT_B_Y_RANGE[0], SORTING_OBJECT_Z)
SORTING_OBJECT_INIT_YAW_RANGE = (-3.141592653589793, 3.141592653589793)
SORTING_OBJECT_ORIENTATION_RANGE = {
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": SORTING_OBJECT_INIT_YAW_RANGE,
}
SORTING_VIEWPORT_CAMERA_PRIM_PATH = "/Camera"
SORTING_VIEWPORT_CAMERA_TRANSLATE = (1.65, -1.25, 1.20)
SORTING_VIEWPORT_CAMERA_LOOKAT = (0.25, 0.0, 0.85)
SORTING_VIEWPORT_CAMERA_ROTATE_XYZ = (68.00214, -0.0, 135.01662)
SORTING_VIEWPORT_CAMERA_SCALE = (1.0, 1.0, 1.0)
SORTING_OBJECT_A_POSE_RANGE = {
    "x": (SORTING_OBJECT_A_INIT_POS[0], SORTING_OBJECT_A_INIT_POS[0]),
    "y": SORTING_OBJECT_A_Y_RANGE,
    "z": (SORTING_OBJECT_A_INIT_POS[2], SORTING_OBJECT_A_INIT_POS[2]),
    **SORTING_OBJECT_ORIENTATION_RANGE,
}
SORTING_OBJECT_B_POSE_RANGE = {
    "x": (SORTING_OBJECT_B_INIT_POS[0], SORTING_OBJECT_B_INIT_POS[0]),
    "y": SORTING_OBJECT_B_Y_RANGE,
    "z": (SORTING_OBJECT_B_INIT_POS[2], SORTING_OBJECT_B_INIT_POS[2]),
    **SORTING_OBJECT_ORIENTATION_RANGE,
}
SORTING_BIN_BLUE_POS = (0.3, 0.4, SORTING_Z_OFFSET)
SORTING_BIN_BLACK_POS = (0.3, -0.4, SORTING_Z_OFFSET)
SORTING_YAW_NEG_90 = -1.5707963267948966
SORTING_BIN_YAW_NEG_90_QUAT = (0.0, 0.0, -0.7071067811865475, 0.7071067811865476)
DROP_HEIGHT = SORTING_Z_OFFSET - 0.25

SORTING_BELT_USD = SORTING_ASSET_DATA_DIR / "franka_belt/belt.usd"
SORTING_FRANKA_USD = SORTING_ASSET_DATA_DIR / "Collected_factory_franka_instanceable/factory_franka_instanceable_camera.usd"

OBJECT_A_NAME = os.getenv("OBJECT_A_NAME", "box_3_label")
OBJECT_B_NAME = os.getenv("OBJECT_B_NAME", "box_4_no")
OBJECT_C_NAME = os.getenv("OBJECT_C_NAME", "sorting_bin_blue")
OBJECT_D_NAME = os.getenv("OBJECT_D_NAME", "black_sorting_bin")

MDP_OBJECT_A_NAME = OBJECT_A_NAME
MDP_OBJECT_B_NAME = OBJECT_B_NAME
MDP_OBJECT_C_NAME = OBJECT_C_NAME
MDP_OBJECT_D_NAME = OBJECT_D_NAME

OBJECT_A_PRIM_PATH = "{ENV_REGEX_NS}/Object_A"
OBJECT_B_PRIM_PATH = "{ENV_REGEX_NS}/Object_B"
OBJECT_GRASP_PRIM_PATHS = [OBJECT_A_PRIM_PATH, OBJECT_B_PRIM_PATH]

SORTING_OBJECT_USD_REL_PATHS = {
    "box_3_label": Path("box_both/Collected_box_grap_label/box_grap_label.usd"),
    "box_4_no": Path("box_both/Collected_box_grap/box_grap.usd"),
    "sorting_bin_blue": Path("bins/bin_blue/sorting_bin_blue.usd"),
    "black_sorting_bin": Path("bins/bin_black/black_sorting_bin.usd"),
}


FRANKA_PICK_PLACE_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PICK_PLACE_CFG.spawn.usd_path = str(SORTING_FRANKA_USD)
FRANKA_PICK_PLACE_CFG.spawn.activate_contact_sensors = True
FRANKA_PICK_PLACE_CFG.spawn.rigid_props.disable_gravity = True
# Faster test tracking for teleop: the stock Franka forearm effort limit is conservative.
FRANKA_PICK_PLACE_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PICK_PLACE_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PICK_PLACE_CFG.actuators["panda_shoulder"].effort_limit_sim = 5200.0
FRANKA_PICK_PLACE_CFG.actuators["panda_shoulder"].velocity_limit_sim = 2.175
FRANKA_PICK_PLACE_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PICK_PLACE_CFG.actuators["panda_forearm"].damping = 80.0
FRANKA_PICK_PLACE_CFG.actuators["panda_forearm"].effort_limit_sim = 720.0
FRANKA_PICK_PLACE_CFG.actuators["panda_forearm"].velocity_limit_sim = 2.61
FRANKA_PICK_PLACE_CFG.actuators["panda_hand"].stiffness = 2000.0
FRANKA_PICK_PLACE_CFG.actuators["panda_hand"].damping = 100.0


def _object_usd_path(object_name: str) -> str:
    sorting_object_path = SORTING_OBJECT_USD_REL_PATHS.get(object_name)
    if sorting_object_path is not None:
        return str(SORTING_ASSET_DATA_DIR / sorting_object_path)

    direct_path = SORTING_ASSET_DATA_DIR / "Objects" / f"{object_name}.usd"
    nested_path = SORTING_ASSET_DATA_DIR / "Objects" / object_name / f"{object_name}.usd"
    if direct_path.exists():
        return str(direct_path)
    if nested_path.exists():
        return str(nested_path)
    return str(direct_path)


def _object_scale(object_name: str) -> tuple[float, float, float]:
    if object_name in (OBJECT_A_NAME, OBJECT_B_NAME):
        return SORTING_BOX_SCALE
    return (1.0, 1.0, 1.0)


def _franka_robot_cfg() -> ArticulationCfg:
    return FRANKA_PICK_PLACE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=SORTING_ROBOT_POS,
            rot=SORTING_ROBOT_YAW_180_QUAT,
            joint_pos=FRANKA_PICK_PLACE_CFG.init_state.joint_pos,
            joint_vel=FRANKA_PICK_PLACE_CFG.init_state.joint_vel,
        ),
    )


def convey_objects_on_belt(
    env,
    env_ids: torch.Tensor | None,
    asset_cfgs: list[SceneEntityCfg],
    velocity: tuple[float, float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """Drive objects along the proxy belt while they are on its top surface."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    conveyor_velocity = torch.tensor(velocity, device=env.device)
    for asset_cfg in asset_cfgs:
        asset = env.scene[asset_cfg.name]
        root_pos_e = wp.to_torch(asset.data.root_pos_w)[env_ids] - env.scene.env_origins[env_ids]
        on_belt = (
            (root_pos_e[:, 0] >= x_range[0])
            & (root_pos_e[:, 0] <= x_range[1])
            & (root_pos_e[:, 1] >= y_range[0])
            & (root_pos_e[:, 1] <= y_range[1])
            & (root_pos_e[:, 2] >= z_range[0])
            & (root_pos_e[:, 2] <= z_range[1])
        )
        if not torch.any(on_belt):
            continue

        active_env_ids = env_ids[on_belt]
        root_vel_w = wp.to_torch(asset.data.root_vel_w)[active_env_ids].clone()
        root_vel_w[:, 0] = conveyor_velocity[0]
        root_vel_w[:, 1] = conveyor_velocity[1]
        asset.write_root_velocity_to_sim_index(root_velocity=root_vel_w, env_ids=active_env_ids)


def reset_boxes_with_random_lanes(
    env,
    env_ids: torch.Tensor,
    object_a_cfg: SceneEntityCfg,
    object_b_cfg: SceneEntityCfg,
    x: float,
    left_y: float,
    right_y: float,
    z: float,
    yaw_range: tuple[float, float],
):
    """Reset A/B boxes on opposite belt lanes, randomly swapping left/right."""
    if env_ids is None or len(env_ids) == 0:
        return

    object_a = env.scene[object_a_cfg.name]
    object_b = env.scene[object_b_cfg.name]
    swap_lanes = torch.rand(len(env_ids), device=env.device) < 0.5

    y_a = torch.where(
        swap_lanes,
        torch.full((len(env_ids),), right_y, device=env.device),
        torch.full((len(env_ids),), left_y, device=env.device),
    )
    y_b = torch.where(
        swap_lanes,
        torch.full((len(env_ids),), left_y, device=env.device),
        torch.full((len(env_ids),), right_y, device=env.device),
    )

    x_values = torch.full((len(env_ids),), x, device=env.device)
    z_values = torch.full((len(env_ids),), z, device=env.device)
    roll_values = torch.zeros(len(env_ids), device=env.device)
    pitch_values = torch.zeros(len(env_ids), device=env.device)
    yaw_a = torch.empty(len(env_ids), device=env.device).uniform_(yaw_range[0], yaw_range[1])
    yaw_b = torch.empty(len(env_ids), device=env.device).uniform_(yaw_range[0], yaw_range[1])
    quat_a = math_utils.quat_from_euler_xyz(roll_values, pitch_values, yaw_a)
    quat_b = math_utils.quat_from_euler_xyz(roll_values, pitch_values, yaw_b)
    zero_vel = torch.zeros((len(env_ids), 6), device=env.device)

    object_a.write_root_pose_to_sim_index(
        root_pose=torch.cat(
            [torch.stack([x_values, y_a, z_values], dim=-1) + env.scene.env_origins[env_ids], quat_a], dim=-1
        ),
        env_ids=env_ids,
    )
    object_a.write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=env_ids)

    object_b.write_root_pose_to_sim_index(
        root_pose=torch.cat(
            [torch.stack([x_values, y_b, z_values], dim=-1) + env.scene.env_origins[env_ids], quat_b], dim=-1
        ),
        env_ids=env_ids,
    )
    object_b.write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=env_ids)


def set_viewport_camera(env, env_ids, eye: tuple[float, float, float], lookat: tuple[float, float, float]):
    """Reapply the viewer camera after the Kit visualizer is initialized."""
    del env_ids
    viewport_camera_controller = getattr(env, "viewport_camera_controller", None)
    if viewport_camera_controller is not None:
        viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
    else:
        env.sim.set_camera_view(eye=eye, target=lookat)


@configclass
class EventCfgFranka:
    """Configuration for reset-time randomization."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.04, 0.04],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    convey_objects = EventTerm(
        func=convey_objects_on_belt,
        mode="interval",
        interval_range_s=(SORTING_CONVEYOR_INTERVAL_S, SORTING_CONVEYOR_INTERVAL_S),
        params={
            "asset_cfgs": [
                SceneEntityCfg(OBJECT_A_NAME),
                SceneEntityCfg(OBJECT_B_NAME),
            ],
            "velocity": SORTING_CONVEYOR_VELOCITY,
            "x_range": SORTING_CONVEYOR_X_RANGE,
            "y_range": SORTING_CONVEYOR_Y_RANGE,
            "z_range": SORTING_CONVEYOR_Z_RANGE,
        },
    )

    init_boxes_pose = EventTerm(
        func=reset_boxes_with_random_lanes,
        mode="reset",
        params={
            "object_a_cfg": SceneEntityCfg(OBJECT_A_NAME),
            "object_b_cfg": SceneEntityCfg(OBJECT_B_NAME),
            "x": SORTING_OBJECT_X,
            "left_y": SORTING_OBJECT_LEFT_Y,
            "right_y": SORTING_OBJECT_RIGHT_Y,
            "z": SORTING_OBJECT_Z,
            "yaw_range": SORTING_OBJECT_INIT_YAW_RANGE,
        },
    )

    init_object_c_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (SORTING_BIN_BLUE_POS[0], SORTING_BIN_BLUE_POS[0]),
                "y": (SORTING_BIN_BLUE_POS[1], SORTING_BIN_BLUE_POS[1]),
                "z": (SORTING_Z_OFFSET, SORTING_Z_OFFSET),
                "yaw": (SORTING_YAW_NEG_90, SORTING_YAW_NEG_90),
            },
            "asset_cfgs": [SceneEntityCfg(OBJECT_C_NAME)],
        },
    )

    init_object_d_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (SORTING_BIN_BLACK_POS[0], SORTING_BIN_BLACK_POS[0]),
                "y": (SORTING_BIN_BLACK_POS[1], SORTING_BIN_BLACK_POS[1]),
                "z": (SORTING_Z_OFFSET, SORTING_Z_OFFSET),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (SORTING_YAW_NEG_90, SORTING_YAW_NEG_90),
            },
            "asset_cfgs": [SceneEntityCfg(OBJECT_D_NAME)],
        },
    )

    set_default_viewport_camera = EventTerm(
        func=set_viewport_camera,
        mode="reset",
        params={
            "eye": SORTING_VIEWPORT_CAMERA_TRANSLATE,
            "lookat": SORTING_VIEWPORT_CAMERA_LOOKAT,
        },
    )


@configclass
class FrankaPickPlaceSceneCfg(InteractiveSceneCfg):
    """Table scene for the Franka pick-and-place task."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    if SORTING_BELT_USD.exists():
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=SORTING_BELT_POS),
            spawn=UsdFileCfg(
                usd_path=str(SORTING_BELT_USD),
                scale=SORTING_BELT_SCALE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                ),
            ),
        )

        belt_proxy_collider = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Belt_Proxy_Collider",
            init_state=AssetBaseCfg.InitialStateCfg(pos=SORTING_BELT_PROXY_POS),
            spawn=sim_utils.CuboidCfg(
                size=SORTING_BELT_PROXY_SIZE,
                visible=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.8, 1.0),
                    opacity=0.25,
                ),
            ),
        )

        table_proxy_collider = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table_Proxy_Collider",
            init_state=AssetBaseCfg.InitialStateCfg(pos=SORTING_TABLE_PROXY_POS),
            spawn=sim_utils.CuboidCfg(
                size=SORTING_TABLE_PROXY_SIZE,
                visible=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
            ),
        )

    else:
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.0, 0.0, 0.707, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )

        plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
            spawn=GroundPlaneCfg(),
        )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: stack_mdp.JointPositionActionCfg = MISSING
    gripper_action: stack_mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Low-dimensional observations for policy learning and replay."""

        actions = ObsTerm(func=stack_mdp.last_action)
        joint_pos = ObsTerm(func=stack_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=stack_mdp.joint_vel_rel)
        object = ObsTerm(
            func=pick_and_place_mdp.a_and_b_abs_obs_in_base_frame,
            params={"object_a_cfg": SceneEntityCfg(MDP_OBJECT_A_NAME), "object_b_cfg": SceneEntityCfg(MDP_OBJECT_C_NAME)},
        )
        object_obs_ac = ObsTerm(
            func=pick_and_place_mdp.a_and_b_abs_obs_in_base_frame,
            params={"object_a_cfg": SceneEntityCfg(MDP_OBJECT_A_NAME), "object_b_cfg": SceneEntityCfg(MDP_OBJECT_C_NAME)},
        )
        object_obs_bd = ObsTerm(
            func=pick_and_place_mdp.a_and_b_abs_obs_in_base_frame,
            params={"object_a_cfg": SceneEntityCfg(MDP_OBJECT_B_NAME), "object_b_cfg": SceneEntityCfg(MDP_OBJECT_D_NAME)},
        )
        a_pose = ObsTerm(
            func=pick_and_place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg(MDP_OBJECT_A_NAME)},
        )
        b_pose = ObsTerm(
            func=pick_and_place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg(MDP_OBJECT_B_NAME)},
        )
        c_pose = ObsTerm(
            func=pick_and_place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg(MDP_OBJECT_C_NAME)},
        )
        d_pose = ObsTerm(
            func=pick_and_place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg(MDP_OBJECT_D_NAME)},
        )
        eef_pos = ObsTerm(func=stack_mdp.ee_frame_pose_in_base_frame, params={"return_key": "pos"})
        eef_quat = ObsTerm(func=stack_mdp.ee_frame_pose_in_base_frame, params={"return_key": "quat"})
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask observations used by data collection and evaluation flows."""

        grasp = ObsTerm(
            func=pick_and_place_mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg(MDP_OBJECT_A_NAME),
                "diff_threshold": 0.13,
                "force_threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=stack_mdp.time_out, time_out=True)

    object_dropped = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={"minimum_height": DROP_HEIGHT, "asset_cfg": SceneEntityCfg(MDP_OBJECT_A_NAME)},
    )

    object_b_dropped = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={"minimum_height": DROP_HEIGHT, "asset_cfg": SceneEntityCfg(MDP_OBJECT_B_NAME)},
    )

    success = DoneTerm(
        func=pick_and_place_mdp.both_boxes_placed_a_into_c_b_into_d,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg(MDP_OBJECT_A_NAME),
            "object_b_cfg": SceneEntityCfg(MDP_OBJECT_B_NAME),
            "object_c_cfg": SceneEntityCfg(MDP_OBJECT_C_NAME),
            "object_d_cfg": SceneEntityCfg(MDP_OBJECT_D_NAME),
            "xy_threshold": 0.08,
            # Require the box root to be low in the bin, not just hovering over the opening.
            "height_threshold": 0.04,
            "height_diff": 0.0,
            "min_height_diff": -0.01,
            "max_height_diff": 0.035,
            "force_threshold": 0.3,
            "require_gripper_open": True,
            # Prevent ending while a released box is still falling into the bin.
            "max_linear_velocity": 0.05,
            "max_angular_velocity": 0.75,
        },
    )


@configclass
class PickPlaceFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for placing object A onto object B with Franka Panda."""

    scene: FrankaPickPlaceSceneCfg = FrankaPickPlaceSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    events = None
    curriculum = None

    viewport_camera_prim_path: str = SORTING_VIEWPORT_CAMERA_PRIM_PATH
    viewport_camera_translate: tuple[float, float, float] = SORTING_VIEWPORT_CAMERA_TRANSLATE
    viewport_camera_rotate_xyz: tuple[float, float, float] = SORTING_VIEWPORT_CAMERA_ROTATE_XYZ
    viewport_camera_scale: tuple[float, float, float] = SORTING_VIEWPORT_CAMERA_SCALE

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 240
        self.sim.render_interval = 12
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            enable_ccd=True,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
        self.viewer.origin_type = "world"
        self.viewer.asset_name = None
        self.viewer.env_index = 0
        self.viewer.cam_prim_path = SORTING_VIEWPORT_CAMERA_PRIM_PATH
        self.viewer.eye = list(SORTING_VIEWPORT_CAMERA_TRANSLATE)
        self.viewer.lookat = list(SORTING_VIEWPORT_CAMERA_LOOKAT)


@configclass
class JointPositionPickPlaceFrankaEnvCfg(PickPlaceFrankaEnvCfg):
    """Joint-position Franka Panda pick-and-place environment."""

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfgFranka()

        self.scene.robot = _franka_robot_cfg()
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        self.actions.arm_action = stack_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint[1-7]"],
            scale=1.0,
            use_default_offset=False,
            offset=0.0,
        )

        self.actions.gripper_action = stack_mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint[1-2]"],
            open_command_expr={"panda_finger_joint[1-2]": 0.04},
            close_command_expr={"panda_finger_joint[1-2]": 0.0},
        )

        self.gripper_joint_names = ["panda_finger_joint[1-2]"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.046]),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.046]),
                ),
            ],
        )

        object_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=20.0,
            disable_gravity=False,
        )
        box_collision_properties = sim_utils.CollisionPropertiesCfg(
            contact_offset=SORTING_CONTACT_OFFSET,
            rest_offset=SORTING_REST_OFFSET,
        )
        bin_properties = RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
        bin_collision_properties = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        )

        object_specs = (
            (
                OBJECT_A_NAME,
                "{ENV_REGEX_NS}/Object_A",
                RigidObjectCfg.InitialStateCfg(pos=SORTING_OBJECT_A_INIT_POS, rot=SORTING_OBJECT_IDENTITY_QUAT),
            ),
            (
                OBJECT_B_NAME,
                "{ENV_REGEX_NS}/Object_B",
                RigidObjectCfg.InitialStateCfg(pos=SORTING_OBJECT_B_INIT_POS, rot=SORTING_OBJECT_IDENTITY_QUAT),
            ),
            (
                OBJECT_C_NAME,
                "{ENV_REGEX_NS}/Bin_Blue",
                RigidObjectCfg.InitialStateCfg(pos=SORTING_BIN_BLUE_POS, rot=SORTING_BIN_YAW_NEG_90_QUAT),
            ),
            (
                OBJECT_D_NAME,
                "{ENV_REGEX_NS}/Bin_Black",
                RigidObjectCfg.InitialStateCfg(pos=SORTING_BIN_BLACK_POS, rot=SORTING_BIN_YAW_NEG_90_QUAT),
            ),
        )

        for object_name, prim_path, init_state in object_specs:
            if object_name is None:
                raise ValueError("Sorting object names must not be None.")
            is_box_object = object_name in (OBJECT_A_NAME, OBJECT_B_NAME)
            is_bin_object = object_name in (OBJECT_C_NAME, OBJECT_D_NAME)

            setattr(
                self.scene,
                object_name,
                RigidObjectCfg(
                    prim_path=prim_path,
                    init_state=init_state,
                    spawn=UsdFileCfg(
                        usd_path=_object_usd_path(object_name),
                        activate_contact_sensors=True,
                        scale=_object_scale(object_name),
                        rigid_props=bin_properties if is_bin_object else object_properties,
                        collision_props=(
                            box_collision_properties
                            if is_box_object
                            else bin_collision_properties
                            if is_bin_object
                            else None
                        ),
                    ),
                ),
            )

        self.scene.contact_grasp = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
            update_period=0.0,
            history_length=6,
            debug_vis=False,
            filter_prim_paths_expr=OBJECT_GRASP_PRIM_PATHS,
        )


JointPositionPlaceAOntoBEnvCfg = JointPositionPickPlaceFrankaEnvCfg
