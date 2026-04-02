# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Waypoint Manager — arXiv:2405.01792 "Anchor Pursuit" Algorithm.

Manages 2 waypoints (WP1, WP2) along pre-computed navigation paths.
Implements the "anchor pursuit" waypoint selection inspired by
pure pursuit path following.

Key behaviors:
- Waypoints are sampled along the Dijkstra-shortest-path on the nav graph
- Look-ahead distance: Uniform[5.0, 20.0] m
- WP1 advances when robot reaches within 0.75m
- Tracks waypoint history (2 previous WPs + 3 previous HLC commands)
"""

from __future__ import annotations

import torch


class WaypointManager:
    """GPU-batched waypoint manager for parallel environments.

    Each environment has:
    - A path (sequence of waypoint positions from Dijkstra)
    - Two active waypoints (WP1, WP2)
    - History of previous waypoints and HLC commands
    """

    def __init__(
        self,
        num_envs: int,
        goal_threshold: float = 0.75,
        lookahead_range: tuple[float, float] = (5.0, 20.0),
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.goal_threshold = goal_threshold
        self.lookahead_min, self.lookahead_max = lookahead_range
        self.device = device

        # Active waypoints in world frame (N, 2)
        self.wp1 = torch.zeros(num_envs, 2, device=device)
        self.wp2 = torch.zeros(num_envs, 2, device=device)

        # Waypoint history: 2 previous WPs (N, 2, 2)
        self.prev_wps = torch.zeros(num_envs, 2, 2, device=device)

        # HLC command history: 3 previous (vx, vy, wz) (N, 3, 3)
        self.prev_cmds = torch.zeros(num_envs, 3, 3, device=device)

        # Goal reached flag
        self.goal_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, wp1: torch.Tensor, wp2: torch.Tensor) -> None:
        """Reset waypoints for specified environments.

        Args:
            env_ids: Environment indices to reset.
            wp1: New WP1 positions (len(env_ids), 2).
            wp2: New WP2 positions (len(env_ids), 2).
        """
        self.wp1[env_ids] = wp1
        self.wp2[env_ids] = wp2
        self.prev_wps[env_ids] = 0.0
        self.prev_cmds[env_ids] = 0.0
        self.goal_reached[env_ids] = False

    def update(
        self,
        robot_pos_xy: torch.Tensor,
        hlc_cmd: torch.Tensor,
        new_wp2_fn=None,
    ) -> torch.Tensor:
        """Update waypoints based on robot position.

        When robot reaches WP1 (within goal_threshold):
        - WP1 ← WP2
        - WP2 ← new waypoint from path (via new_wp2_fn)
        - Records previous WP1 to history

        Args:
            robot_pos_xy: Current robot XY positions (N, 2).
            hlc_cmd: Current HLC velocity commands (N, 3).
            new_wp2_fn: Optional callable(env_ids) -> (new_wp2_positions).

        Returns:
            goal_reached: Boolean mask (N,) of envs that reached WP1.
        """
        # Update command history: shift and append
        self.prev_cmds = torch.roll(self.prev_cmds, 1, dims=1)
        self.prev_cmds[:, 0] = hlc_cmd

        # Check which envs reached WP1
        dist_to_wp1 = torch.norm(robot_pos_xy - self.wp1, dim=1)
        reached = dist_to_wp1 < self.goal_threshold

        if reached.any():
            reached_ids = reached.nonzero(as_tuple=False).squeeze(-1)

            # Shift waypoint history
            self.prev_wps[reached_ids, 1] = self.prev_wps[reached_ids, 0]
            self.prev_wps[reached_ids, 0] = self.wp1[reached_ids]

            # Advance: WP1 ← WP2
            self.wp1[reached_ids] = self.wp2[reached_ids]

            # Get new WP2 from path
            if new_wp2_fn is not None:
                new_wp2 = new_wp2_fn(reached_ids)
                self.wp2[reached_ids] = new_wp2

            self.goal_reached[reached_ids] = True

        return reached

    def get_obs(
        self,
        robot_pos_xy: torch.Tensor,
        robot_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Get waypoint observation in robot body frame.

        Returns 17D vector per env:
        - WP1 in robot frame (2)
        - WP2 in robot frame (2)
        - Previous WP1 history in robot frame (2×2 = 4)
        - Previous HLC commands (3×3 = 9)

        Args:
            robot_pos_xy: Robot XY in world (N, 2).
            robot_yaw: Robot yaw angle (N,).

        Returns:
            Observation tensor (N, 17).
        """
        cos_yaw = torch.cos(robot_yaw)
        sin_yaw = torch.sin(robot_yaw)

        def world_to_body(points_w: torch.Tensor) -> torch.Tensor:
            """Transform (N, 2) world coords to body frame."""
            dx = points_w[:, 0] - robot_pos_xy[:, 0]
            dy = points_w[:, 1] - robot_pos_xy[:, 1]
            x_b = cos_yaw * dx + sin_yaw * dy
            y_b = -sin_yaw * dx + cos_yaw * dy
            return torch.stack([x_b, y_b], dim=1)

        wp1_b = world_to_body(self.wp1)                      # (N, 2)
        wp2_b = world_to_body(self.wp2)                      # (N, 2)
        prev0_b = world_to_body(self.prev_wps[:, 0])         # (N, 2)
        prev1_b = world_to_body(self.prev_wps[:, 1])         # (N, 2)
        cmds = self.prev_cmds.reshape(self.num_envs, -1)     # (N, 9)

        return torch.cat([wp1_b, wp2_b, prev0_b, prev1_b, cmds], dim=1)  # (N, 17)

    @staticmethod
    def sample_waypoints_on_path(
        path_positions: torch.Tensor,
        start_idx: int = 0,
        lookahead_min: float = 5.0,
        lookahead_max: float = 20.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample 2 waypoints along a path at random lookahead distances.

        Args:
            path_positions: (num_nodes, 2) positions along the Dijkstra path.
            start_idx: Index of the start node on the path.
            lookahead_min: Minimum lookahead distance (m).
            lookahead_max: Maximum lookahead distance (m).

        Returns:
            (wp1, wp2) each of shape (2,).
        """
        # Compute cumulative path length
        diffs = path_positions[1:] - path_positions[:-1]
        seg_lengths = torch.norm(diffs, dim=1)
        cum_lengths = torch.zeros(len(path_positions))
        cum_lengths[1:] = torch.cumsum(seg_lengths, dim=0)
        total_length = cum_lengths[-1].item()

        # Sample lookahead distances
        d1 = torch.empty(1).uniform_(lookahead_min, lookahead_max).item()
        d2 = d1 + torch.empty(1).uniform_(lookahead_min, lookahead_max).item() * 0.5

        d1 = min(d1, total_length)
        d2 = min(d2, total_length)

        # If near end, duplicate last node
        if d1 >= total_length * 0.95:
            return path_positions[-1], path_positions[-1]

        def interpolate_at_distance(d: float) -> torch.Tensor:
            for i in range(len(cum_lengths) - 1):
                if cum_lengths[i + 1] >= d:
                    t = (d - cum_lengths[i]) / (cum_lengths[i + 1] - cum_lengths[i] + 1e-6)
                    return path_positions[i] + t * (path_positions[i + 1] - path_positions[i])
            return path_positions[-1]

        wp1 = interpolate_at_distance(d1)
        wp2 = interpolate_at_distance(d2)
        return wp1, wp2
