# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""LLC Environment Config — arXiv:2405.01792.

DirectRLEnv configuration for the low-level locomotion controller.
Based on AME-2 DirectEnvCfg pattern (Isaac Lab 0.46.x).

Robot: 轮足狗机器人v3 — 12 revolute (legs) + 4 continuous (wheels) = 16 DOF
  Legs:   fr/fl/rr/rl × [hip_joint, thigh_joint, calf_joint]  (revolute, position control)
  Wheels: fr/fl/rr/rl × [foot_joint]  (continuous, velocity control)

Action: 16D = 12 joint position offsets + 4 wheel velocity targets
Observation: proprioceptive (49D) + privileged (~120D) + height scan (~87D)
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.simulation_cfg import PhysxCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass


# ── Joint ordering (must match URDF traversal order) ─────────────────────
# Legs: 12 revolute joints
LEG_JOINT_NAMES = [
    "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint",
    "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint",
    "rr_hip_joint", "rr_thigh_joint", "rr_calf_joint",
    "rl_hip_joint", "rl_thigh_joint", "rl_calf_joint",
]
# Wheels: 4 continuous joints
WHEEL_JOINT_NAMES = [
    "fr_foot_joint", "fl_foot_joint", "rr_foot_joint", "rl_foot_joint",
]
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES  # 16 total

# ── Soft joint limits for KFE (knee/calf) joints ────────────────────────
# From URDF: calf joints have limits, apply 95% soft constraint
# Index in LEG_JOINT_NAMES: calf = [2, 5, 8, 11]
CALF_JOINT_INDICES = [2, 5, 8, 11]


# ── Terrain config (Miki et al. [16] style, for LLC training) ───────────
LLC_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,           # 10 difficulty levels
    num_cols=20,           # 20 terrain columns
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15,
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(-0.05, 0.05),
            noise_step=0.005,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.31,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.31,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "slopes": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(-0.08, 0.08),
            noise_step=0.01,
            border_width=0.25,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_height_max=0.05,
            stone_width_range=(0.25, 1.0),
            stone_distance_range=(0.05, 0.15),
            gaps_size_range=(0.02, 0.10),
            platform_width=2.0,
        ),
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.05, 0.15),
            num_obstacles=40,
            platform_width=2.0,
        ),
    },
)


# ── Robot articulation config ──────────────────────────────────────────
# NOTE: Replace prim_path with your USD asset path when running in Isaac Lab.
# This URDF needs to be converted to USD via Isaac Lab's URDF converter.
WHEELED_DOG_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        # TODO: Convert 轮足狗机器人v3.urdf to USD and set path here
        usd_path="assets/robots/anymal_d_wheeled/wheeled_dog.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),   # base height target from paper
        joint_pos={
            # Default stance (radians) — tune per robot kinematics
            "fr_hip_joint": 0.0,  "fl_hip_joint": 0.0,
            "rr_hip_joint": 0.0,  "rl_hip_joint": 0.0,
            "fr_thigh_joint": 0.8,  "fl_thigh_joint": 0.8,
            "rr_thigh_joint": -0.8, "rl_thigh_joint": -0.8,
            "fr_calf_joint": -1.5,  "fl_calf_joint": -1.5,
            "rr_calf_joint": 1.5,   "rl_calf_joint": 1.5,
            # Wheels: zero initial velocity
            "fr_foot_joint": 0.0, "fl_foot_joint": 0.0,
            "rr_foot_joint": 0.0, "rl_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Leg actuators: position-controlled (SEA model)
        "legs": sim_utils.DCMotorCfg(
            joint_names_expr=[".*hip_joint", ".*thigh_joint", ".*calf_joint"],
            saturation_effort=65.0,      # Nm, conservative
            effort_limit=65.0,
            velocity_limit=21.0,         # rad/s
            stiffness={".*hip_joint": 65.0, ".*thigh_joint": 95.0, ".*calf_joint": 120.0},
            damping={".*hip_joint": 5.0, ".*thigh_joint": 5.0, ".*calf_joint": 5.0},
        ),
        # Wheel actuators: velocity-controlled (direct drive)
        "wheels": sim_utils.DCMotorCfg(
            joint_names_expr=[".*foot_joint"],
            saturation_effort=10.0,      # Nm, wheel motors
            effort_limit=10.0,
            velocity_limit=40.0,         # rad/s (~4 m/s at r=0.1m)
            stiffness={".*foot_joint": 0.0},   # velocity mode: no position stiffness
            damping={".*foot_joint": 2.0},      # velocity tracking gain
        ),
    },
)


@configclass
class WheeledLLCEnvCfg(DirectRLEnvCfg):
    """LLC environment configuration for wheeled-legged locomotion.

    arXiv:2405.01792 — Low-Level Controller training.
    """

    # ── Simulation ──
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,                           # physics dt (4 substeps → 0.02s control dt)
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # ── Scene ──
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # ── Robot ──
    robot: ArticulationCfg = WHEELED_DOG_CFG

    # ── Sensors ──
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(3.0, 2.0),    # 3m front × 2m lateral
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # ── Terrain ──
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=LLC_TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # ── Environment parameters ──
    decimation: int = 4                     # physics steps per policy step (0.005 * 4 = 0.02s = 50Hz)
    episode_length_s: float = 10.0          # [stated] Table S2
    # 56D proprio + num_rays height scan. Actual dim set at runtime after height_scanner init.
    # proprio: lin_vel(3) + ang_vel(3) + gravity(3) + leg_pos(12) + leg_vel(12) + wheel_vel(4) + prev_act(16) + cmd(3) = 56
    observation_space: int = 56
    action_space: int = 16                  # 12 leg joints + 4 wheel velocities
    state_space: int = 0                    # not used in teacher training

    # ── Action scaling ──
    action_scale_legs: float = 0.25         # joint position offset scale (rad)
    action_scale_wheels: float = 10.0       # wheel velocity scale (rad/s)

    # ── Velocity command ranges (m/s, rad/s) ──
    cmd_vx_range: tuple[float, float] = (-2.5, 2.5)
    cmd_vy_range: tuple[float, float] = (-1.2, 1.2)
    cmd_yaw_range: tuple[float, float] = (-1.5, 1.5)
    cmd_zero_prob: float = 0.2              # probability of zero command

    # ── Base height target ──
    base_height_target: float = 0.55        # [stated] Eq. 18

    # ── Termination thresholds ──
    term_height_min: float = 0.25
    term_pitch_max: float = 1.2             # rad (~70°)
    term_roll_max: float = 1.2

    # ── Reward weights (arXiv:2405.01792, inferred from RSL conventions) ──
    # Positive = tracking / survival
    w_lin_vel: float = 1.0                  # Eq. 14
    w_ang_vel: float = 0.5                  # Eq. 15
    w_survival: float = 1.0                 # Eq. 25
    # Negative = penalties (weights are negative; reward functions return POSITIVE magnitudes)
    w_base_motion: float = -2.0             # Eq. 16
    w_orientation: float = -0.2             # Eq. 17
    w_base_height: float = -1.0             # Eq. 18
    w_torque: float = -1e-5                 # Eq. 19
    w_joint_vel_accel: float = -2.5e-7      # Eq. 20
    w_action_smooth: float = -0.01          # Eq. 21
    w_joint_constraint: float = -10.0       # Eq. 22-23
    w_body_contact: float = -1.0            # Eq. 24

    # ── Domain randomization ranges ──
    rand_mass_range: tuple[float, float] = (-0.15, 0.15)     # ±15% base mass
    rand_friction_range: tuple[float, float] = (0.4, 1.0)
    rand_restitution_range: tuple[float, float] = (0.0, 0.4)
    rand_push_force: float = 50.0           # N, random push at 1 Hz
    rand_imu_bias_std: float = 0.1
    rand_motor_strength_range: tuple[float, float] = (0.9, 1.1)  # ±10%

    # ── Height scan noise (student training) ──
    height_scan_noise_std: float = 0.02     # meters
    height_scan_dropout: float = 0.1        # 10% random dropout
