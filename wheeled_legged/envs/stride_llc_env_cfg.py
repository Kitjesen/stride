# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Stride LLC Environment Config — arXiv:2405.01792 on robot_lab infrastructure.

Extends the existing ThunderHistRoughEnvCfg to use paper-specific
reward weights and velocity command ranges while reusing the
THUNDER_V3_CFG robot, actuators, and action split.

Designed to run on the BSRL server where robot_lab is installed:
  /root/autodl-tmp/thunder2/robot_lab/

Usage on server:
    python -m isaaclab.app.runner \
        --task ThunderStride-Rough-v0 \
        --num_envs 4096 --max_iterations 30000
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

# Import from existing thunder-him config (must be on PYTHONPATH on server)
# Locally we provide a standalone version below
try:
    from config.thunder_hist.rough_env_cfg import (
        ThunderHistRoughEnvCfg,
        ThunderHistRoughRewardWeights,
        ThunderHistRoughCommandParams,
        ThunderHistRoughActuatorGains,
    )
    _HAS_THUNDER_HIM = True
except ImportError:
    _HAS_THUNDER_HIM = False


# ============================================================================
# Paper-specific reward weights (arXiv:2405.01792 Eq. 14-25)
# ============================================================================

@configclass
class StrideRewardWeights:
    """Reward weights aligned with arXiv:2405.01792.

    Replaces ThunderHistRoughRewardWeights with paper-faithful values.
    Key changes vs thunder-him baseline:
    - Wider velocity tracking with exp kernel (not L2)
    - Explicit base height target at 0.55m (paper) / 0.426m (our robot)
    - Survival reward always on
    - Body contact penalty for non-wheel parts
    """

    # ── Paper tracking rewards (Eq. 14-15) ──
    # These replace track_lin_vel_xy_exp and track_ang_vel_z_exp
    track_lin_vel_direction: float = 6.0       # Eq. 14: exp kernel tracking
    track_ang_vel_yaw: float = 3.0             # Eq. 15: exp kernel yaw tracking
    upward: float = 2.0                         # keep upright bonus

    # ── Paper posture penalties (Eq. 16-18) ──
    body_motion_penalty: float = -2.0           # Eq. 16: -(vz² + ωx² + ωy²)
    body_tilt_penalty: float = -1.0             # Eq. 17: orientation
    base_height_tolerance: float = -10.0        # Eq. 18: height regulation

    # ── Paper smoothness (Eq. 19-21) ──
    joint_torques_l2: float = -1e-5             # Eq. 19: torque penalty
    joint_torques_wheel_l2: float = 0.0
    joint_acc_l2: float = -1e-7                 # Eq. 20: velocity + acceleration
    joint_acc_wheel_l2: float = 0.0
    action_rate_l2: float = -0.03               # Eq. 21: action smoothness

    # ── Paper constraints (Eq. 22-24) ──
    joint_pos_limits: float = -2.0              # Eq. 22-23: soft knee limits
    undesired_contacts: float = -3.0            # Eq. 24: non-wheel body contacts
    contact_forces: float = -5e-4

    # ── Paper survival (Eq. 25) ──
    # Implicit: episode continues → accumulated reward

    # ── Additional from thunder-him (proven useful) ──
    stand_still: float = -5.0
    joint_pos_penalty: float = -1.0
    joint_mirror: float = -0.05
    feet_stumble: float = -5.0
    foot_impact_velocity: float = -0.6
    contact_force_threshold: float = -0.01
    joint_power: float = -2e-5

    # ── Zero (disabled) ──
    is_terminated: float = 0.0
    track_lin_vel_xy_exp: float = 0.0
    track_ang_vel_z_exp: float = 0.0
    lin_vel_z_l2: float = 0.0
    ang_vel_xy_l2: float = 0.0
    flat_orientation_l2: float = 0.0
    base_height_l2: float = 0.0
    body_lin_acc_l2: float = 0.0
    joint_vel_l2: float = 0.0
    joint_vel_wheel_l2: float = 0.0
    joint_vel_limits: float = 0.0
    wheel_vel_penalty: float = 0.0
    action_smoothness_l2: float = 0.0
    feet_air_time: float = 0.0
    feet_contact: float = 0.0
    feet_contact_without_cmd: float = 0.0
    feet_slide: float = 0.0
    feet_height: float = 0.0
    feet_height_body: float = 0.0
    feet_gait: float = 0.0


@configclass
class StrideCommandParams:
    """Velocity command ranges from arXiv:2405.01792.

    Paper uses wider range than thunder-him baseline:
    - vx: [-2.5, 2.5] m/s (paper) vs [-1.0, 1.0] (baseline)
    - vy: [-1.2, 1.2] m/s
    - ωz: [-1.5, 1.5] rad/s
    """
    lin_vel_x: tuple = (-2.5, 2.5)
    lin_vel_y: tuple = (-1.2, 1.2)
    ang_vel_z: tuple = (-1.5, 1.5)


@configclass
class StrideActuatorGains:
    """Actuator PD gains — same as thunder-him but with tuned wheel damping."""
    hip_stiffness: float = 70.0
    hip_damping: float = 15.0
    thigh_stiffness: float = 100.0
    thigh_damping: float = 15.0
    calf_stiffness: float = 120.0
    calf_damping: float = 20.0
    wheel_stiffness: float = 0.0      # velocity mode
    wheel_damping: float = 2.0        # increased from 1.0 for better tracking


if _HAS_THUNDER_HIM:
    @configclass
    class ThunderStrideRoughEnvCfg(ThunderHistRoughEnvCfg):
        """Thunder Stride (arXiv:2405.01792) environment on robot_lab.

        Inherits all scene/sensor/terrain from ThunderHistRoughEnvCfg,
        replaces reward weights and command ranges with paper values.
        """

        reward_weights: StrideRewardWeights = StrideRewardWeights()
        command_params: StrideCommandParams = StrideCommandParams()
        actuator_gains: StrideActuatorGains = StrideActuatorGains()

        def __post_init__(self):
            super().__post_init__()

            # Override base height target for our robot (shorter than ANYmal)
            self.rewards.base_height_tolerance.params["target_height"] = 0.426
            self.rewards.base_height_tolerance.params["tolerance"] = 0.05

            # Wider velocity ranges per paper
            c = self.command_params
            self.commands.base_velocity.ranges.lin_vel_x = c.lin_vel_x
            self.commands.base_velocity.ranges.lin_vel_y = c.lin_vel_y
            self.commands.base_velocity.ranges.ang_vel_z = c.ang_vel_z

            # Wheel velocity scale from paper
            self.actions.joint_vel.scale = 10.0

            # Terrain curriculum: unlock full range
            self.curriculum.terrain_levels.params["max_terrain_level"] = 10


# ============================================================================
# Standalone config (for use without robot_lab import)
# ============================================================================

# When robot_lab is not available (local development), the standalone
# DirectRLEnv config in llc_env_cfg.py provides equivalent functionality.
# Use ThunderStrideRoughEnvCfg on the server, llc_env_cfg.WheeledLLCEnvCfg locally.

STANDALONE_CONFIG_NOTE = """
Server training:
  cd /root/autodl-tmp/thunder2/
  python train_him.py --task ThunderStride-Rough --num_envs 4096

Local development (no Isaac Lab):
  Use wheeled_legged.envs.llc_env_cfg.WheeledLLCEnvCfg
  with wheeled_legged.envs.llc_env.WheeledLLCEnv
"""
