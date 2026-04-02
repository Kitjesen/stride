# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""LLC Reward Functions — arXiv:2405.01792 Eq. 14-25.

Low-level locomotion controller rewards for wheeled-legged robots.
11 reward terms covering velocity tracking, posture regulation,
smoothness, joint limits, contact penalties, and survival.

Convention: ALL functions return POSITIVE values.
The sign (reward vs penalty) is determined by the weight in the config.
This follows the RSL legged_gym convention.
"""

from __future__ import annotations

import torch


# ============================================================================
# Tracking Rewards (positive weights → maximize)
# ============================================================================


def linear_velocity_tracking(
    base_lin_vel_b: torch.Tensor,
    velocity_commands: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Linear velocity tracking reward — Eq. 14.

    Returns exp(-||v_cmd_xy - v_base_xy||^2 / sigma), always >= 0.

    When |v_cmd| < 0.05, rewards zero velocity (standing still).
    """
    cmd_xy = velocity_commands[:, :2]
    vel_xy = base_lin_vel_b[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=1)

    error_sq = torch.sum((cmd_xy - vel_xy) ** 2, dim=1)
    standing_reward = 2.0 * torch.exp(-2.0 * torch.sum(vel_xy**2, dim=1))
    tracking_reward = torch.exp(-error_sq / sigma) + torch.clamp(
        torch.sum(cmd_xy * vel_xy, dim=1), min=0.0
    )

    return torch.where(cmd_norm < 0.05, standing_reward, tracking_reward)


def angular_velocity_tracking(
    base_ang_vel_b: torch.Tensor,
    velocity_commands: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Angular velocity tracking reward — Eq. 15.

    Returns exp(-(w_cmd_z - w_base_z)^2 / sigma), always >= 0.
    """
    error_sq = (velocity_commands[:, 2] - base_ang_vel_b[:, 2]) ** 2
    return torch.exp(-error_sq / sigma)


# ============================================================================
# Posture / Regulation Penalties (negative weights → minimize)
# All return POSITIVE magnitudes. Weight sign handles negation.
# ============================================================================


def base_motion_penalty(
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
) -> torch.Tensor:
    """Base motion penalty — Eq. 16.

    Returns 1.25*vz^2 + 0.4*|wx| + 0.4*|wy| (positive magnitude).
    """
    vz_sq = base_lin_vel_b[:, 2] ** 2
    wx_abs = torch.abs(base_ang_vel_b[:, 0])
    wy_abs = torch.abs(base_ang_vel_b[:, 1])
    return 1.25 * vz_sq + 0.4 * wx_abs + 0.4 * wy_abs


def orientation_penalty(
    projected_gravity_b: torch.Tensor,
) -> torch.Tensor:
    """Orientation penalty — Eq. 17.

    Returns ||g_xy||^2 (positive magnitude).
    """
    return projected_gravity_b[:, 0] ** 2 + projected_gravity_b[:, 1] ** 2


def base_height_penalty(
    base_pos_w: torch.Tensor,
    target_height: float = 0.55,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Base height regulation — Eq. 18.

    Returns max(0, |h - target| - tolerance)^2 (positive magnitude).
    """
    height_error = torch.abs(base_pos_w[:, 2] - target_height) - tolerance
    return torch.clamp(height_error, min=0.0) ** 2


# ============================================================================
# Smoothness / Efficiency Penalties (negative weights → minimize)
# ============================================================================


def torque_penalty(
    joint_torques: torch.Tensor,
) -> torch.Tensor:
    """Torque penalty — Eq. 19.

    Returns sum(tau_i^2) for leg joints (positive magnitude).
    """
    return torch.sum(joint_torques[:, :12] ** 2, dim=1)


def joint_velocity_acceleration_penalty(
    joint_vel: torch.Tensor,
    joint_vel_prev: torch.Tensor,
    dt: float = 0.02,
    c_k: float = 1.0,
) -> torch.Tensor:
    """Joint velocity and acceleration smoothness — Eq. 20.

    Returns c_k * sum(dq^2 + 0.01*ddq^2) (positive magnitude).
    """
    leg_vel = joint_vel[:, :12]
    leg_vel_prev = joint_vel_prev[:, :12]
    joint_accel = (leg_vel - leg_vel_prev) / dt
    return c_k * torch.sum(leg_vel**2 + 0.01 * joint_accel**2, dim=1)


def action_smoothness_penalty(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    prev_prev_actions: torch.Tensor,
) -> torch.Tensor:
    """Action smoothness — Eq. 21.

    Returns sum((a_t - a_{t-1})^2 + (a_t - 2*a_{t-1} + a_{t-2})^2) (positive magnitude).
    """
    first_order = torch.sum((actions - prev_actions) ** 2, dim=1)
    second_order = torch.sum(
        (actions - 2.0 * prev_actions + prev_prev_actions) ** 2, dim=1
    )
    return first_order + second_order


# ============================================================================
# Constraint Penalties (negative weights → minimize)
# ============================================================================


def joint_constraint_penalty(
    joint_pos: torch.Tensor,
    joint_limits_soft: torch.Tensor,
) -> torch.Tensor:
    """Soft joint limit penalty — Eq. 22-23.

    Returns sum of squared violations (positive magnitude).
    """
    lower = joint_limits_soft[:, 0].unsqueeze(0)
    upper = joint_limits_soft[:, 1].unsqueeze(0)

    below = torch.clamp(lower - joint_pos, min=0.0)
    above = torch.clamp(joint_pos - upper, min=0.0)
    penalty = below**2 + above**2

    return torch.sum(penalty, dim=1)


def body_contact_penalty(
    contact_forces: torch.Tensor,
    non_wheel_body_ids: list[int],
    threshold: float = 1.0,
) -> torch.Tensor:
    """Body contact penalty — Eq. 24.

    Returns count of non-wheel body contacts (positive magnitude).
    """
    forces = contact_forces[:, non_wheel_body_ids, :]
    force_norms = torch.norm(forces, dim=-1)
    contacts = (force_norms > threshold).float()
    return torch.sum(contacts, dim=1)


# ============================================================================
# Survival Reward (positive weight → maximize)
# ============================================================================


def survival_reward(
    terminated: torch.Tensor,
) -> torch.Tensor:
    """Survival reward — Eq. 25.

    Returns 1.0 at every non-terminal step.
    """
    return (~terminated).float()
