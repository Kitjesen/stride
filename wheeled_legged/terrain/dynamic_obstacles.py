# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Dynamic Obstacles — arXiv:2405.01792.

Manages moving box obstacles for HLC navigation training.
Boxes move at 0.1-0.5 m/s toward the robot, bouncing off terrain boundaries.

In Isaac Lab: obstacles are separate rigid body actors whose root states
are updated each HLC step (10Hz). During training, they appear as
height changes in the elevation map temporal stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DynamicObstacleConfig:
    """Configuration for dynamic obstacles."""
    num_obstacles_per_env: int = 5
    speed_range: tuple[float, float] = (0.1, 0.5)  # m/s
    box_size_range: tuple[float, float] = (0.3, 0.8)  # meters
    box_height_range: tuple[float, float] = (0.2, 0.6)  # meters
    spawn_radius: float = 8.0                          # meters from env origin
    boundary_radius: float = 10.0                      # bounce boundary


class DynamicObstacleManager:
    """GPU-batched dynamic obstacle manager.

    Manages positions and velocities of box obstacles for all parallel envs.
    Obstacles move toward the robot with random perturbations and bounce
    off boundaries.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: DynamicObstacleConfig | None = None,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.cfg = cfg or DynamicObstacleConfig()
        self.device = device
        self.n_obs = self.cfg.num_obstacles_per_env

        # Obstacle states: (num_envs, num_obstacles, 5) = [x, y, z, vx, vy]
        self.positions = torch.zeros(num_envs, self.n_obs, 3, device=device)
        self.velocities = torch.zeros(num_envs, self.n_obs, 2, device=device)
        self.sizes = torch.zeros(num_envs, self.n_obs, 3, device=device)  # [w, d, h]
        self.active = torch.ones(num_envs, self.n_obs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
        """Reset obstacles for specified environments.

        Args:
            env_ids: Environment indices to reset.
            env_origins: World origins of environments (len(env_ids), 3).
        """
        n = len(env_ids)
        cfg = self.cfg

        # Random positions around env origin
        angles = torch.rand(n, self.n_obs, device=self.device) * 2 * 3.14159
        radii = torch.rand(n, self.n_obs, device=self.device) * cfg.spawn_radius

        self.positions[env_ids, :, 0] = env_origins[:, 0:1] + radii * torch.cos(angles)
        self.positions[env_ids, :, 1] = env_origins[:, 1:2] + radii * torch.sin(angles)
        self.positions[env_ids, :, 2] = env_origins[:, 2:3].expand_as(
            self.positions[env_ids, :, 2]
        )

        # Random speeds
        speeds = torch.empty(n, self.n_obs, device=self.device).uniform_(*cfg.speed_range)
        dirs = torch.rand(n, self.n_obs, device=self.device) * 2 * 3.14159
        self.velocities[env_ids, :, 0] = speeds * torch.cos(dirs)
        self.velocities[env_ids, :, 1] = speeds * torch.sin(dirs)

        # Random sizes
        self.sizes[env_ids, :, 0] = torch.empty(n, self.n_obs, device=self.device).uniform_(*cfg.box_size_range)
        self.sizes[env_ids, :, 1] = self.sizes[env_ids, :, 0]  # square base
        self.sizes[env_ids, :, 2] = torch.empty(n, self.n_obs, device=self.device).uniform_(*cfg.box_height_range)

        self.active[env_ids] = True

    def step(
        self,
        dt: float,
        robot_pos_xy: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Update obstacle positions for one time step.

        Obstacles drift toward the robot with velocity perturbation.
        Bounce off boundary circle centered on env_origin.

        Args:
            dt: Time step (seconds). For HLC: 0.1s.
            robot_pos_xy: Robot XY positions (num_envs, 2).
            env_origins: Environment origins (num_envs, 3).
        """
        # Slight steering toward robot (adversarial)
        to_robot = robot_pos_xy.unsqueeze(1) - self.positions[:, :, :2]  # (N, n_obs, 2)
        to_robot_norm = torch.norm(to_robot, dim=2, keepdim=True).clamp(min=0.1)
        to_robot_dir = to_robot / to_robot_norm

        # Blend: 80% current direction + 20% toward robot
        speed = torch.norm(self.velocities, dim=2, keepdim=True).clamp(min=0.01)
        cur_dir = self.velocities / speed
        new_dir = 0.8 * cur_dir + 0.2 * to_robot_dir
        new_dir = new_dir / torch.norm(new_dir, dim=2, keepdim=True).clamp(min=1e-6)
        self.velocities = new_dir * speed

        # Update positions
        self.positions[:, :, 0] += self.velocities[:, :, 0] * dt
        self.positions[:, :, 1] += self.velocities[:, :, 1] * dt

        # Bounce off boundary
        offset = self.positions[:, :, :2] - env_origins[:, :2].unsqueeze(1)
        dist_from_center = torch.norm(offset, dim=2)
        outside = dist_from_center > self.cfg.boundary_radius

        if outside.any():
            # Reflect velocity
            normal = offset / dist_from_center.unsqueeze(2).clamp(min=1e-6)
            dot = (self.velocities * normal).sum(dim=2, keepdim=True)
            reflected = self.velocities - 2 * dot * normal
            self.velocities[outside] = reflected[outside]

            # Clamp position back inside boundary
            clamped = env_origins[:, :2].unsqueeze(1) + \
                normal * self.cfg.boundary_radius * 0.95
            self.positions[:, :, :2][outside] = clamped[outside]

    def get_obstacle_heightmap_contribution(
        self,
        query_points_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute height contribution of obstacles at query points.

        For each query point, returns the maximum obstacle height if inside
        any box, else 0. Used to augment the elevation map.

        Args:
            query_points_xy: (num_envs, num_points, 2) world XY coords.

        Returns:
            heights: (num_envs, num_points) additional height from obstacles.
        """
        N, P, _ = query_points_xy.shape

        # Expand for broadcasting: (N, P, 1, 2) vs (N, 1, n_obs, 2)
        qp = query_points_xy.unsqueeze(2)           # (N, P, 1, 2)
        op = self.positions[:, :, :2].unsqueeze(1)   # (N, 1, n_obs, 2)
        sz = self.sizes[:, :, :2].unsqueeze(1) / 2   # (N, 1, n_obs, 2) half-size

        # Check if query point is inside any box (axis-aligned)
        inside = ((qp - op).abs() < sz).all(dim=3)  # (N, P, n_obs)

        # Height of each box
        box_h = self.sizes[:, :, 2].unsqueeze(1)     # (N, 1, n_obs)
        box_h = box_h.expand_as(inside.float())

        # Max height over all obstacles at each query point
        contrib = (inside.float() * box_h).max(dim=2).values  # (N, P)

        return contrib
