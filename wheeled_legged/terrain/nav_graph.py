# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Navigation Graph — arXiv:2405.01792.

Builds a navigation graph on top of WFC-generated terrain.
Nodes are tile centers; edges connect traversable adjacent tiles.
Dijkstra shortest path provides obstacle-free waypoint sequences.

Requires: networkx (pip install networkx)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None  # graceful fallback for environments without networkx

from .wfc_terrain import TileType, WFCConfig, DIRECTIONS


class NavGraph:
    """Navigation graph built from WFC tile grid.

    Nodes: traversable tile centers (all tiles are traversable).
    Edges: 4-connected neighbors with height-penalized weights.
    """

    TILE_HEIGHTS = {
        TileType.FLOOR_0: 0.0,
        TileType.FLOOR_1: 0.3,
        TileType.STAIR_N: 0.15,
        TileType.STAIR_S: 0.15,
        TileType.STAIR_E: 0.15,
        TileType.STAIR_W: 0.15,
    }

    def __init__(self, tile_grid: np.ndarray, cfg: WFCConfig):
        if nx is None:
            raise ImportError("networkx is required: pip install networkx")

        self.grid = tile_grid
        self.cfg = cfg
        self.w, self.h = tile_grid.shape
        self.tile_size = cfg.tile_size

        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build graph from tile grid."""
        for x in range(self.w):
            for y in range(self.h):
                pos = self._tile_center(x, y)
                tile_type = TileType(self.grid[x, y])
                self.graph.add_node(
                    (x, y),
                    pos=pos,
                    tile_type=tile_type,
                    height=self.TILE_HEIGHTS.get(tile_type, 0.0),
                )

        # Add edges (4-connected)
        for x in range(self.w):
            for y in range(self.h):
                for dx, dy in DIRECTIONS:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < self.w and 0 <= ny_ < self.h:
                        if not self.graph.has_edge((x, y), (nx_, ny_)):
                            weight = self._edge_weight((x, y), (nx_, ny_))
                            self.graph.add_edge((x, y), (nx_, ny_), weight=weight)

    def _tile_center(self, x: int, y: int) -> tuple[float, float]:
        """Get world position of tile center."""
        return (
            x * self.tile_size + self.tile_size / 2,
            y * self.tile_size + self.tile_size / 2,
        )

    def _edge_weight(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Compute edge weight: distance + height penalty."""
        pos_a = np.array(self._tile_center(*a))
        pos_b = np.array(self._tile_center(*b))
        dist = float(np.linalg.norm(pos_b - pos_a))

        h_a = self.TILE_HEIGHTS.get(TileType(self.grid[a[0], a[1]]), 0.0)
        h_b = self.TILE_HEIGHTS.get(TileType(self.grid[b[0], b[1]]), 0.0)
        height_penalty = 0.5 * abs(h_a - h_b)

        return dist + height_penalty

    def shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[float, float]]:
        """Find shortest path using Dijkstra.

        Args:
            start: (x, y) tile coordinates.
            goal: (x, y) tile coordinates.

        Returns:
            List of (world_x, world_y) positions along the path.
        """
        try:
            node_path = nx.dijkstra_path(self.graph, start, goal, weight="weight")
        except nx.NetworkXNoPath:
            return []

        return [self.graph.nodes[n]["pos"] for n in node_path]

    def path_length(self, path: list[tuple[float, float]]) -> float:
        """Compute total path length in meters."""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total += (dx**2 + dy**2) ** 0.5
        return total

    def sample_random_path(
        self,
        min_length: float = 5.0,
        max_attempts: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[list[tuple[float, float]], tuple[int, int], tuple[int, int]]:
        """Sample a random path with minimum length.

        Args:
            min_length: Minimum path length in meters.
            max_attempts: Maximum sampling attempts.
            rng: Numpy random generator.

        Returns:
            (path_positions, start_node, goal_node)
        """
        if rng is None:
            rng = np.random.default_rng()

        nodes = list(self.graph.nodes)

        for _ in range(max_attempts):
            start = nodes[rng.integers(len(nodes))]
            goal = nodes[rng.integers(len(nodes))]
            if start == goal:
                continue

            path = self.shortest_path(start, goal)
            if path and self.path_length(path) >= min_length:
                return path, start, goal

        # Fallback: return longest path found
        start = nodes[0]
        goal = nodes[-1]
        path = self.shortest_path(start, goal)
        return path, start, goal

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()
