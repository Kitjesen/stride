# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""HLC Environment Config — arXiv:2405.01792.

High-Level Controller environment configuration.
The HLC runs at 10Hz and outputs velocity commands to a frozen LLC.
The LLC runs internally at 50Hz (5 substeps per HLC step).

Terrain: WFC-generated navigation world with dynamic obstacles.
Observation: heightmap(3 temporal) + LLC hidden state + position buffer + waypoints.
Action: Beta distribution → [vx, vy, ωz] velocity commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..terrain.wfc_terrain import WFCConfig
from ..terrain.dynamic_obstacles import DynamicObstacleConfig


@dataclass
class HLCEnvCfg:
    """HLC environment configuration."""

    # ── Timing ──
    hlc_dt: float = 0.1                  # HLC control period (10 Hz)
    llc_dt: float = 0.02                 # LLC control period (50 Hz)
    llc_substeps: int = 5                # llc steps per hlc step (0.1/0.02)
    episode_length_s: float = 15.0       # [stated] Table S3
    num_envs: int = 1000                 # 150k batch / 150 steps

    # ── LLC checkpoint ──
    llc_checkpoint: str = ""             # path to frozen LLC student .pt
    llc_gru_hidden: int = 512            # must match LLC student GRU

    # ── WFC Terrain ──
    wfc: WFCConfig = field(default_factory=lambda: WFCConfig(
        grid_size=(8, 8),
        tile_size=2.5,
        floor_0_height=0.0,
        floor_1_height=0.3,
        stair_steps=4,
    ))

    # ── Dynamic Obstacles ──
    obstacles: DynamicObstacleConfig = field(default_factory=lambda: DynamicObstacleConfig(
        num_obstacles_per_env=5,
        speed_range=(0.1, 0.5),
        box_size_range=(0.3, 0.8),
        box_height_range=(0.2, 0.6),
    ))

    # ── Heightmap observation ──
    heightmap_h: int = 16                # grid rows (lateral)
    heightmap_w: int = 26                # grid cols (longitudinal)
    heightmap_resolution: float = 0.115  # meters per cell
    temporal_scans: int = 3              # current + 2 previous (t-0.1s, t-0.2s)

    # ── Position buffer ──
    pos_buffer_size: int = 20            # [stated] max entries
    pos_buffer_interval: float = 0.5     # [stated] meters between records

    # ── Waypoints ──
    lookahead_range: tuple[float, float] = (5.0, 20.0)  # [stated] meters
    goal_threshold: float = 0.75         # [stated] Eq. 9

    # ── Action space (Beta distribution bounds) ──
    vx_range: tuple[float, float] = (-1.0, 2.0)    # [stated] asymmetric, forward-biased
    vy_range: tuple[float, float] = (-0.75, 0.75)   # [stated]
    wz_range: tuple[float, float] = (-1.25, 1.25)   # [stated]

    # ── Reward weights ──
    w_goal_reaching: float = 1.0         # Eq. 9
    w_dense_progress: float = 1.0        # Eq. 10
    w_exploration: float = 0.1           # Eq. 11-12 (not stated, tuned)
    w_stability: float = 0.5             # Eq. 13
    w_llc_passthrough: float = 0.2       # w_l (not stated, start at 0.2)
