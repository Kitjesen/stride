# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""LLC Direct Environment — arXiv:2405.01792.

Low-level locomotion controller environment for wheeled-legged robots.
Based on AME-2 DirectRLEnv pattern (Isaac Lab 0.46.x).

Key features:
- 16D action: 12 leg joint positions + 4 wheel velocities
- Split actuator control: position for legs, velocity for wheels
- 11 reward terms (Eq. 14-25)
- Terrain curriculum
- Domain randomization
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .llc_env_cfg import WheeledLLCEnvCfg, CALF_JOINT_INDICES
from ..rewards import llc_rewards


class WheeledLLCEnv(DirectRLEnv):
    """Wheeled-legged locomotion environment.

    Actions [0:12] → leg joint position offsets (added to default pose)
    Actions [12:16] → wheel velocity targets (rad/s)
    """

    cfg: WheeledLLCEnvCfg

    def __init__(self, cfg: WheeledLLCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        n, dev = self.num_envs, self.device

        # ── Body indices for contact detection ──
        self._base_cs_id, _ = self._contact_sensor.find_bodies("base_link")
        self._thigh_cs_ids, _ = self._contact_sensor.find_bodies(".*thigh_Link")
        self._shank_cs_ids, _ = self._contact_sensor.find_bodies(".*shank_Link")
        self._foot_cs_ids, _ = self._contact_sensor.find_bodies(".*foot_Link")
        # Non-wheel contacts: base + thigh + shank (NOT foot/wheel)
        self._non_wheel_cs_ids = list(self._base_cs_id) + list(self._thigh_cs_ids) + list(self._shank_cs_ids)

        # ── Action buffers ──
        self._actions = torch.zeros(n, 16, device=dev)
        self._prev_actions = torch.zeros(n, 16, device=dev)
        self._prev_prev_actions = torch.zeros(n, 16, device=dev)

        # ── Velocity command buffer ──
        self._velocity_commands = torch.zeros(n, 3, device=dev)  # [vx, vy, yaw_rate]

        # ── Joint velocity history for acceleration computation ──
        self._prev_joint_vel = torch.zeros(n, 12, device=dev)

        # ── Soft joint limits for knee constraints (Eq. 22-23) ──
        joint_limits = self._robot.data.joint_pos_limits
        if joint_limits.dim() == 3:
            joint_limits = joint_limits[0]
        # Apply 95% soft limits on leg joints
        self._soft_joint_limits = joint_limits[:12].clone()
        self._soft_joint_limits[:, 0] *= 0.95  # lower
        self._soft_joint_limits[:, 1] *= 0.95  # upper

        # ── Terminated flag ──
        self._terminated = torch.zeros(n, dtype=torch.bool, device=dev)

        # ── Terrain curriculum state ──
        self.terrain_levels = torch.zeros(n, dtype=torch.long, device=dev)

        # ── Reward weight scaling by dt (stored separately to avoid mutating cfg) ──
        _dt = self.step_dt
        _weight_names = [
            "w_lin_vel", "w_ang_vel", "w_survival",
            "w_base_motion", "w_orientation", "w_base_height",
            "w_torque", "w_joint_vel_accel", "w_action_smooth",
            "w_joint_constraint", "w_body_contact",
        ]
        self._rw = {k: getattr(cfg, k) * _dt for k in _weight_names}

        # ── Episode reward sums for logging ──
        self._ep_sums = {
            "lin_vel": torch.zeros(n, device=dev),
            "ang_vel": torch.zeros(n, device=dev),
            "base_motion": torch.zeros(n, device=dev),
            "orientation": torch.zeros(n, device=dev),
            "base_height": torch.zeros(n, device=dev),
            "torque": torch.zeros(n, device=dev),
            "joint_vel_accel": torch.zeros(n, device=dev),
            "smoothness": torch.zeros(n, device=dev),
            "joint_constraint": torch.zeros(n, device=dev),
            "body_contact": torch.zeros(n, device=dev),
            "survival": torch.zeros(n, device=dev),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Scene Setup
    # ─────────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ─────────────────────────────────────────────────────────────────────
    # Action Processing
    # ─────────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_prev_actions = self._prev_actions.clone()
        self._prev_actions = self._actions.clone()
        self._actions = actions.clamp(-100.0, 100.0)

    def _apply_action(self) -> None:
        # Split actions: legs (position) vs wheels (velocity)
        leg_actions = self._actions[:, :12]
        wheel_actions = self._actions[:, 12:]

        # Leg joints: offset from default stance
        leg_targets = (
            self._robot.data.default_joint_pos[:, :12]
            + leg_actions * self.cfg.action_scale_legs
        )

        # Wheel joints: direct velocity target
        wheel_targets = wheel_actions * self.cfg.action_scale_wheels

        # Apply position targets to legs
        self._robot.set_joint_position_target(leg_targets, joint_ids=list(range(12)))
        # Apply velocity targets to wheels
        self._robot.set_joint_velocity_target(wheel_targets, joint_ids=list(range(12, 16)))

    # ─────────────────────────────────────────────────────────────────────
    # Observations
    # ─────────────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        # Height scan from RayCaster
        heights = (
            self._height_scanner.data.pos_w[:, 2:3]
            - self._height_scanner.data.ray_hits_w[..., 2]
        )  # (N, num_rays) relative height

        # Proprioceptive observations (56D)
        # Teacher uses ground-truth base velocity; student will estimate via RNN.
        obs_proprio = torch.cat([
            self._robot.data.root_lin_vel_b,                                    # 3  — body linear velocity (privileged for student)
            self._robot.data.root_ang_vel_b,                                    # 3  — body angular velocity (IMU gyro)
            self._robot.data.projected_gravity_b,                               # 3  — gravity in body frame (IMU derived)
            self._robot.data.joint_pos[:, :12] - self._robot.data.default_joint_pos[:, :12],  # 12 — leg joint pos offset
            self._robot.data.joint_vel[:, :12] * 0.05,                          # 12 — leg joint velocities (scaled)
            self._robot.data.joint_vel[:, 12:16] * 0.1,                         # 4  — wheel joint velocities (scaled)
            self._prev_actions,                                                  # 16 — previous actions
            self._velocity_commands,                                             # 3  — velocity command
        ], dim=-1)  # Total: 3+3+3+12+12+4+16+3 = 56D

        # Policy obs = proprio + height scan (teacher gets noiseless, student gets noisy)
        obs_policy = torch.cat([obs_proprio, heights], dim=-1)  # 56 + num_rays

        # Privileged observations (teacher only): contact ground truth
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, 0, self._foot_cs_ids, :]
        foot_contact = (torch.norm(contact_forces, dim=-1) > 1.0).float()  # (N, 4)

        obs_privileged = torch.cat([
            self._robot.data.root_lin_vel_w,                                    # 3  — world linear velocity
            self._robot.data.root_ang_vel_w,                                    # 3  — world angular velocity
            foot_contact,                                                        # 4  — binary contact states
            contact_forces.reshape(self.num_envs, -1),                           # 12 — 3D contact forces
        ], dim=-1)  # 22D privileged-only

        return {
            "policy": obs_policy,
            "teacher_privileged": torch.cat([obs_policy, obs_privileged], dim=-1),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Rewards
    # ─────────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        base_lin_vel_b = self._robot.data.root_lin_vel_b
        base_ang_vel_b = self._robot.data.root_ang_vel_b
        base_pos_w = self._robot.data.root_pos_w
        projected_gravity = self._robot.data.projected_gravity_b
        joint_torques = self._robot.data.applied_torque
        joint_vel = self._robot.data.joint_vel[:, :12]
        joint_pos = self._robot.data.joint_pos[:, :12]
        contact_forces = self._contact_sensor.data.net_forces_w_history[:, 0]

        # Compute each reward term
        r_lv = llc_rewards.linear_velocity_tracking(
            base_lin_vel_b, self._velocity_commands
        )
        r_av = llc_rewards.angular_velocity_tracking(
            base_ang_vel_b, self._velocity_commands
        )
        r_bm = llc_rewards.base_motion_penalty(base_lin_vel_b, base_ang_vel_b)
        r_ori = llc_rewards.orientation_penalty(projected_gravity)
        r_h = llc_rewards.base_height_penalty(base_pos_w, cfg.base_height_target)
        r_tau = llc_rewards.torque_penalty(joint_torques)
        r_jva = llc_rewards.joint_velocity_acceleration_penalty(
            joint_vel, self._prev_joint_vel, dt=self.step_dt
        )
        r_as = llc_rewards.action_smoothness_penalty(
            self._actions, self._prev_actions, self._prev_prev_actions
        )
        r_jc = llc_rewards.joint_constraint_penalty(
            joint_pos, self._soft_joint_limits
        )
        r_bc = llc_rewards.body_contact_penalty(
            contact_forces, self._non_wheel_cs_ids
        )
        r_surv = llc_rewards.survival_reward(self._terminated)

        # Weighted sum (using dt-scaled weights from self._rw)
        w = self._rw
        total = (
            w["w_lin_vel"] * r_lv
            + w["w_ang_vel"] * r_av
            + w["w_base_motion"] * r_bm
            + w["w_orientation"] * r_ori
            + w["w_base_height"] * r_h
            + w["w_torque"] * r_tau
            + w["w_joint_vel_accel"] * r_jva
            + w["w_action_smooth"] * r_as
            + w["w_joint_constraint"] * r_jc
            + w["w_body_contact"] * r_bc
            + w["w_survival"] * r_surv
        )

        # Update buffers
        self._prev_joint_vel = joint_vel.clone()

        # Log episode sums
        self._ep_sums["lin_vel"] += r_lv
        self._ep_sums["ang_vel"] += r_av
        self._ep_sums["base_motion"] += r_bm
        self._ep_sums["orientation"] += r_ori
        self._ep_sums["base_height"] += r_h
        self._ep_sums["torque"] += r_tau
        self._ep_sums["joint_vel_accel"] += r_jva
        self._ep_sums["smoothness"] += r_as
        self._ep_sums["joint_constraint"] += r_jc
        self._ep_sums["body_contact"] += r_bc
        self._ep_sums["survival"] += r_surv

        return total

    # ─────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_pos = self._robot.data.root_pos_w
        base_quat = self._robot.data.root_quat_w

        # Extract pitch and roll from quaternion
        # Simplified: use projected gravity to check orientation
        grav = self._robot.data.projected_gravity_b
        pitch = torch.atan2(grav[:, 0], grav[:, 2])
        roll = torch.atan2(grav[:, 1], grav[:, 2])

        height_fail = base_pos[:, 2] < self.cfg.term_height_min
        pitch_fail = torch.abs(pitch) > self.cfg.term_pitch_max
        roll_fail = torch.abs(roll) > self.cfg.term_roll_max

        self._terminated = height_fail | pitch_fail | roll_fail

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return self._terminated, time_out

    # ─────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)

        # Reset robot
        self._robot.reset(env_ids)

        # Randomize initial joint positions slightly
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, :12] += torch.randn_like(joint_pos[:, :12]) * 0.1
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Place robot on terrain
        if hasattr(self, '_terrain'):
            root_state = self._robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self._terrain.env_origins[env_ids]
            root_state[:, 2] += 0.1  # small offset above ground
            self._robot.write_root_state_to_sim(root_state, env_ids)

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        self._prev_joint_vel[env_ids] = 0.0

        # Resample velocity commands
        self._resample_commands(env_ids)

        # Reset episode sums
        for k in self._ep_sums:
            self._ep_sums[k][env_ids] = 0.0

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """Resample velocity commands for specified environments."""
        n = len(env_ids)
        dev = self.device

        # Random velocity commands
        vx = torch.empty(n, device=dev).uniform_(*self.cfg.cmd_vx_range)
        vy = torch.empty(n, device=dev).uniform_(*self.cfg.cmd_vy_range)
        yaw = torch.empty(n, device=dev).uniform_(*self.cfg.cmd_yaw_range)

        # Zero command with probability cmd_zero_prob
        zero_mask = torch.rand(n, device=dev) < self.cfg.cmd_zero_prob
        vx[zero_mask] = 0.0
        vy[zero_mask] = 0.0
        yaw[zero_mask] = 0.0

        self._velocity_commands[env_ids, 0] = vx
        self._velocity_commands[env_ids, 1] = vy
        self._velocity_commands[env_ids, 2] = yaw
