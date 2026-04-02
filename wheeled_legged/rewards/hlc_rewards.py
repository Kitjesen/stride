# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""HLC Reward Functions — arXiv:2405.01792 Eq. 9-13.

High-level navigation controller rewards. 4 task-specific terms
plus pass-through of LLC regularization rewards weighted by w_l.
"""

from __future__ import annotations

import torch


def goal_reaching(
    robot_pos_xy: torch.Tensor,
    wp1: torch.Tensor,
    threshold: float = 0.75,
) -> torch.Tensor:
    """Goal reaching sparse reward — Eq. 9.

    r_goal = 1.0 if ||p_robot - wp1|| < threshold, else 0.0

    Args:
        robot_pos_xy: Robot XY position in world frame (N, 2).
        wp1: First waypoint position (N, 2).
        threshold: Distance threshold in meters.
    """
    dist = torch.norm(robot_pos_xy - wp1, dim=1)
    return (dist < threshold).float()


def dense_progress(
    robot_vel_w: torch.Tensor,
    robot_pos_xy: torch.Tensor,
    wp1: torch.Tensor,
    v_thres: float = 0.5,
) -> torch.Tensor:
    """Dense progress reward — Eq. 10.

    r_dense = clip(v · ê_wp1, 0, v_thres) / v_thres
    Only active when |e_wp1| >= threshold (not near goal).
    At close range (< 0.75m), returns 1.0.

    Args:
        robot_vel_w: Robot velocity in world frame (N, 3).
        robot_pos_xy: Robot XY position (N, 2).
        wp1: First waypoint (N, 2).
        v_thres: Velocity threshold (m/s).
    """
    e_wp1 = wp1 - robot_pos_xy                           # (N, 2)
    dist = torch.norm(e_wp1, dim=1, keepdim=True)        # (N, 1)
    e_hat = e_wp1 / (dist + 1e-6)                        # unit direction

    vel_xy = robot_vel_w[:, :2]                           # (N, 2)
    v_proj = torch.sum(vel_xy * e_hat, dim=1)             # (N,)

    # Dense reward: normalized projected velocity
    r = torch.clamp(v_proj, 0.0, v_thres) / v_thres

    # Near goal: full reward
    near = (dist.squeeze(-1) < 0.75)
    r = torch.where(near, torch.ones_like(r), r)

    return r


def exploration_penalty(
    robot_pos_xy: torch.Tensor,
    wp1: torch.Tensor,
    position_buffer: torch.Tensor,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Exploration bonus (penalty for revisiting) — Eq. 11-12.

    C(p, wp1, p_buf) = -n_buf^i   if |p - p_buf^i| < 1.0 and |p - wp1| >= 0.75
    Summed over all buffer entries near wp1.

    Encourages the robot to explore new areas rather than circling.

    Args:
        robot_pos_xy: Robot XY position (N, 2).
        wp1: First waypoint (N, 2).
        position_buffer: Position buffer observation (N, max_entries, 3) = [dx, dy, count].
        sigma: Gaussian kernel width for soft distance.
    """
    # Distance from robot to wp1
    dist_to_wp1 = torch.norm(robot_pos_xy - wp1, dim=1)  # (N,)

    # Only active when not yet near goal
    active = (dist_to_wp1 >= 0.75).float()                # (N,)

    # Buffer entries relative to robot (already in robot frame from to_obs)
    buf_xy = position_buffer[:, :, :2]                     # (N, 20, 2)
    buf_count = position_buffer[:, :, 2]                   # (N, 20)

    # Distance from each buffer entry to robot (entries are robot-relative, so distance = norm)
    buf_dist = torch.norm(buf_xy, dim=2)                   # (N, 20)

    # Soft proximity: Gaussian kernel
    proximity = torch.exp(-0.5 * (buf_dist / sigma) ** 2)  # (N, 20)

    # Weighted sum: visit_count * proximity
    revisit_cost = torch.sum(buf_count * proximity, dim=1)  # (N,)

    return -active * revisit_cost


def near_goal_stability(
    robot_vel_w: torch.Tensor,
    robot_pos_xy: torch.Tensor,
    wp1: torch.Tensor,
    threshold: float = 0.75,
    coeff: float = 2.0,
) -> torch.Tensor:
    """Near-goal stability reward — Eq. 13.

    r_stability = exp(-coeff * ||v||²)  if near goal, else 0.0

    Encourages the robot to slow down and stabilize when near the waypoint.

    Args:
        robot_vel_w: Robot velocity in world frame (N, 3).
        robot_pos_xy: Robot XY position (N, 2).
        wp1: First waypoint (N, 2).
        threshold: Distance threshold for activation.
        coeff: Exponential decay coefficient.
    """
    dist = torch.norm(robot_pos_xy - wp1, dim=1)
    vel_sq = torch.sum(robot_vel_w[:, :2] ** 2, dim=1)

    r = torch.exp(-coeff * vel_sq)
    return torch.where(dist < threshold, r, torch.zeros_like(r))


# ============================================================================
# HLC Reward Weights
# ============================================================================

HLC_REWARD_WEIGHTS = {
    "goal_reaching": 1.0,
    "dense_progress": 1.0,
    "exploration_penalty": 0.1,       # not stated in paper; tune empirically
    "near_goal_stability": 0.5,
    "llc_passthrough_weight": 0.2,    # w_l: not stated; start at 0.2
}
