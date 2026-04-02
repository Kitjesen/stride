# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""WFC Terrain Generation — arXiv:2405.01792.

Procedural terrain using Wave Function Collapse (WFC) algorithm.
Generates navigation-friendly worlds with 3 tile types:
  - Floor_0 (z=0.0m): flat ground level
  - Floor_1 (z=+0.3m): elevated platform
  - Stair: transition between floor levels

Uses leggedrobotics/terrain-generator when available, falls back to
a lightweight built-in WFC implementation for development.

Reference: github.com/leggedrobotics/terrain-generator
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch
import numpy as np


class TileType(IntEnum):
    FLOOR_0 = 0
    FLOOR_1 = 1
    STAIR_N = 2   # stair facing north (connects floor_0 south to floor_1 north)
    STAIR_S = 3
    STAIR_E = 4
    STAIR_W = 5


# Direction offsets: N, E, S, W
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
DIR_NAMES = ["N", "E", "S", "W"]
DIR_OPPOSITE = [2, 3, 0, 1]  # N↔S, E↔W


@dataclass
class WFCConfig:
    """Configuration for WFC terrain generation."""
    grid_size: tuple[int, int] = (8, 8)       # number of tiles (NxN)
    tile_size: float = 2.5                      # meters per tile
    floor_0_height: float = 0.0                 # ground level
    floor_1_height: float = 0.3                 # elevated level
    stair_steps: int = 4                        # steps per stair tile
    max_retries: int = 10                       # WFC contradiction retries
    floor_ratio: float = 0.7                    # proportion of floor tiles
    seed: Optional[int] = None


# ── Adjacency rules ──────────────────────────────────────────────────────
# Each tile type defines which tiles can be adjacent in each direction.
# Stair tiles connect different floor levels; floor tiles connect to same level.

def _build_adjacency() -> dict[int, list[set[int]]]:
    """Build adjacency rules: adj[tile][direction] = set of allowed neighbors."""
    adj = {}
    floors = {TileType.FLOOR_0, TileType.FLOOR_1}
    stairs = {TileType.STAIR_N, TileType.STAIR_S, TileType.STAIR_E, TileType.STAIR_W}

    # Floor_0: connects to Floor_0 or stairs that start at level 0
    adj[TileType.FLOOR_0] = [
        {TileType.FLOOR_0, TileType.STAIR_S},  # N neighbor
        {TileType.FLOOR_0, TileType.STAIR_W},  # E neighbor
        {TileType.FLOOR_0, TileType.STAIR_N},  # S neighbor
        {TileType.FLOOR_0, TileType.STAIR_E},  # W neighbor
    ]

    # Floor_1: connects to Floor_1 or stairs that end at level 1
    adj[TileType.FLOOR_1] = [
        {TileType.FLOOR_1, TileType.STAIR_N},  # N neighbor
        {TileType.FLOOR_1, TileType.STAIR_E},  # E neighbor
        {TileType.FLOOR_1, TileType.STAIR_S},  # S neighbor
        {TileType.FLOOR_1, TileType.STAIR_W},  # W neighbor
    ]

    # Stair_N: south side = floor_0, north side = floor_1
    adj[TileType.STAIR_N] = [
        {TileType.FLOOR_1, TileType.STAIR_S},  # N: connects to level 1
        {TileType.FLOOR_0, TileType.FLOOR_1, TileType.STAIR_W, TileType.STAIR_E},  # E
        {TileType.FLOOR_0, TileType.STAIR_N},  # S: connects to level 0
        {TileType.FLOOR_0, TileType.FLOOR_1, TileType.STAIR_W, TileType.STAIR_E},  # W
    ]

    # Stair_S: north side = floor_0, south side = floor_1
    adj[TileType.STAIR_S] = [
        {TileType.FLOOR_0, TileType.STAIR_S},  # N
        {TileType.FLOOR_0, TileType.FLOOR_1, TileType.STAIR_W, TileType.STAIR_E},  # E
        {TileType.FLOOR_1, TileType.STAIR_N},  # S
        {TileType.FLOOR_0, TileType.FLOOR_1, TileType.STAIR_W, TileType.STAIR_E},  # W
    ]

    # Stair_E: west side = floor_0, east side = floor_1
    adj[TileType.STAIR_E] = [
        {TileType.FLOOR_0, TileType.FLOOR_1, TileType.STAIR_N, TileType.STAIR_S},  # N
        {TileType.FLOOR_1, TileType.STAIR_W},  # E
        {TileType.FLOOR_0, TileType.FLOOR_1, TileType.STAIR_N, TileType.STAIR_S},  # S
        {TileType.FLOOR_0, TileType.STAIR_E},  # W
    ]

    # Stair_W: east side = floor_0, west side = floor_1
    adj[TileType.STAIR_W] = [
        {TileType.FLOOR_0, TileType.FLOOR_1, TileType.STAIR_N, TileType.STAIR_S},  # N
        {TileType.FLOOR_0, TileType.STAIR_W},  # E
        {TileType.FLOOR_0, TileType.FLOOR_1, TileType.STAIR_N, TileType.STAIR_S},  # S
        {TileType.FLOOR_1, TileType.STAIR_E},  # W
    ]

    return adj


ADJACENCY = _build_adjacency()
ALL_TILES = set(TileType)


class WFCGrid:
    """Simple Tiled Model WFC for terrain generation."""

    def __init__(self, cfg: WFCConfig):
        self.cfg = cfg
        self.w, self.h = cfg.grid_size
        self.rng = random.Random(cfg.seed)

    def generate(self) -> np.ndarray:
        """Run WFC and return tile grid (w, h) of TileType values.

        Retries on contradiction up to cfg.max_retries times.
        """
        for attempt in range(self.cfg.max_retries):
            try:
                return self._run_wfc()
            except _ContradictionError:
                if self.cfg.seed is not None:
                    self.rng = random.Random(self.cfg.seed + attempt + 1)
                continue
        raise RuntimeError(f"WFC failed after {self.cfg.max_retries} retries")

    def _run_wfc(self) -> np.ndarray:
        # Initialize: each cell can be any tile
        possible: list[list[set[int]]] = [
            [set(ALL_TILES) for _ in range(self.h)] for _ in range(self.w)
        ]

        # Bias: prefer floor tiles (reduce stair density)
        weights = {t: 1.0 for t in ALL_TILES}
        weights[TileType.FLOOR_0] = self.cfg.floor_ratio * 3
        weights[TileType.FLOOR_1] = self.cfg.floor_ratio * 3

        collapsed = 0
        total = self.w * self.h

        while collapsed < total:
            # Find cell with minimum entropy (most constrained)
            min_entropy = float("inf")
            min_cell = None
            for x in range(self.w):
                for y in range(self.h):
                    n = len(possible[x][y])
                    if n <= 1:
                        continue
                    # Add slight noise to break ties
                    entropy = n + self.rng.random() * 0.1
                    if entropy < min_entropy:
                        min_entropy = entropy
                        min_cell = (x, y)

            if min_cell is None:
                break  # all collapsed

            x, y = min_cell
            # Collapse: weighted random choice
            opts = list(possible[x][y])
            w = [weights.get(t, 1.0) for t in opts]
            total_w = sum(w)
            w = [v / total_w for v in w]
            chosen = self.rng.choices(opts, weights=w, k=1)[0]
            possible[x][y] = {chosen}
            collapsed += 1

            # Propagate constraints
            self._propagate(possible, x, y)

        # Convert to numpy array
        grid = np.zeros((self.w, self.h), dtype=np.int32)
        for x in range(self.w):
            for y in range(self.h):
                if len(possible[x][y]) == 1:
                    grid[x, y] = next(iter(possible[x][y]))
                else:
                    raise _ContradictionError()
        return grid

    def _propagate(self, possible: list[list[set[int]]], sx: int, sy: int) -> None:
        """Constraint propagation via BFS."""
        queue = [(sx, sy)]
        visited = set()

        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            for d, (dx, dy) in enumerate(DIRECTIONS):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.w and 0 <= ny < self.h):
                    continue
                if len(possible[nx][ny]) <= 1:
                    continue

                # Compute allowed tiles for neighbor based on current cell's possibilities
                allowed = set()
                for tile in possible[cx][cy]:
                    allowed |= ADJACENCY[tile][d]

                # Intersect with neighbor's current possibilities
                new_possible = possible[nx][ny] & allowed
                if not new_possible:
                    raise _ContradictionError()

                if new_possible != possible[nx][ny]:
                    possible[nx][ny] = new_possible
                    queue.append((nx, ny))


class _ContradictionError(Exception):
    pass


def tile_grid_to_heightmap(
    grid: np.ndarray,
    cfg: WFCConfig,
    resolution: float = 0.1,
) -> np.ndarray:
    """Convert WFC tile grid to a heightmap array.

    Args:
        grid: (w, h) tile type indices from WFC.
        cfg: WFC configuration.
        resolution: meters per heightmap pixel.

    Returns:
        heightmap: (H, W) float array in meters.
    """
    w, h = grid.shape
    tile_px = int(cfg.tile_size / resolution)
    hm = np.zeros((w * tile_px, h * tile_px), dtype=np.float32)

    for tx in range(w):
        for ty in range(h):
            tile = TileType(grid[tx, ty])
            x0, y0 = tx * tile_px, ty * tile_px
            x1, y1 = x0 + tile_px, y0 + tile_px

            if tile == TileType.FLOOR_0:
                hm[x0:x1, y0:y1] = cfg.floor_0_height
            elif tile == TileType.FLOOR_1:
                hm[x0:x1, y0:y1] = cfg.floor_1_height
            else:
                # Stair tile: linear interpolation
                hm[x0:x1, y0:y1] = _make_stair_tile(
                    tile, tile_px, cfg.floor_0_height, cfg.floor_1_height, cfg.stair_steps
                )

    return hm


def _make_stair_tile(
    tile: TileType,
    size: int,
    h0: float,
    h1: float,
    steps: int,
) -> np.ndarray:
    """Generate a stair heightmap tile."""
    patch = np.zeros((size, size), dtype=np.float32)
    step_size = size // steps

    for s in range(steps):
        height = h0 + (h1 - h0) * (s + 1) / steps
        s0 = s * step_size
        s1 = min((s + 1) * step_size, size)

        if tile == TileType.STAIR_N:
            patch[s0:s1, :] = height
        elif tile == TileType.STAIR_S:
            patch[size - s1:size - s0, :] = height
        elif tile == TileType.STAIR_E:
            patch[:, s0:s1] = height
        elif tile == TileType.STAIR_W:
            patch[:, size - s1:size - s0] = height

    return patch


def generate_wfc_terrain(
    cfg: WFCConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate WFC terrain: returns (tile_grid, heightmap).

    Args:
        cfg: WFC configuration. Uses defaults if None.

    Returns:
        tile_grid: (w, h) int array of TileType values.
        heightmap: (H, W) float array in meters.
    """
    if cfg is None:
        cfg = WFCConfig()

    wfc = WFCGrid(cfg)
    grid = wfc.generate()
    hm = tile_grid_to_heightmap(grid, cfg)
    return grid, hm
