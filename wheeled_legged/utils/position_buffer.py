# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Position Buffer — arXiv:2405.01792 Sec. 4 (HLC observation).

Tracks the last 20 visited positions at 0.5m intervals.
Each entry stores [x_rel, y_rel, visit_count] in robot-relative coords.
Processed by PointNet-like 1D CNN in the HLC network.
"""

from __future__ import annotations

import torch


class PositionBuffer:
    """GPU-batched position buffer for parallel environments.

    Maintains a ring buffer of visited positions per environment.
    Entries are recorded when the robot moves ≥ interval meters.
    Nearby positions increment visit_count instead of creating new entries.
    """

    def __init__(
        self,
        num_envs: int,
        max_entries: int = 20,
        interval: float = 0.5,
        merge_radius: float = 0.25,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.max_entries = max_entries
        self.interval = interval
        self.merge_radius = merge_radius
        self.device = device

        # Buffer: (num_envs, max_entries, 3) — [x_world, y_world, visit_count]
        self.buffer = torch.zeros(num_envs, max_entries, 3, device=device)
        # Number of valid entries per env
        self.counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Accumulated distance since last record
        self.accum_dist = torch.zeros(num_envs, device=device)
        # Last recorded position
        self.last_pos = torch.zeros(num_envs, 2, device=device)
        # Whether first position has been set
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset buffers for specified environments."""
        self.buffer[env_ids] = 0.0
        self.counts[env_ids] = 0
        self.accum_dist[env_ids] = 0.0
        self.last_pos[env_ids] = 0.0
        self.initialized[env_ids] = False

    def update(self, robot_pos_xy: torch.Tensor) -> None:
        """Update position buffer with current robot positions.

        Args:
            robot_pos_xy: Current robot XY positions in world frame (N, 2).
        """
        # Initialize on first call
        first = ~self.initialized
        if first.any():
            self.last_pos[first] = robot_pos_xy[first]
            self.initialized[first] = True

        # Accumulate distance
        displacement = torch.norm(robot_pos_xy - self.last_pos, dim=1)
        self.accum_dist += displacement
        self.last_pos = robot_pos_xy.clone()

        # Find envs that exceeded the recording interval
        record_mask = self.accum_dist >= self.interval
        if not record_mask.any():
            return

        record_ids = record_mask.nonzero(as_tuple=False).squeeze(-1)
        if record_ids.dim() == 0:
            record_ids = record_ids.unsqueeze(0)
        if len(record_ids) == 0:
            return
        self.accum_dist[record_ids] = 0.0

        # Vectorized update: compute distances to all existing entries
        pos_rec = robot_pos_xy[record_ids]                          # (R, 2)
        buf_rec = self.buffer[record_ids]                           # (R, max, 3)
        counts_rec = self.counts[record_ids]                        # (R,)

        # Distance from new position to all buffer entries
        buf_xy = buf_rec[:, :, :2]                                  # (R, max, 2)
        dists = torch.norm(buf_xy - pos_rec.unsqueeze(1), dim=2)    # (R, max)

        # Mask out empty slots (set distance to inf)
        slot_indices = torch.arange(self.max_entries, device=self.device).unsqueeze(0)  # (1, max)
        empty_mask = slot_indices >= counts_rec.unsqueeze(1)        # (R, max)
        dists = dists.masked_fill(empty_mask, float('inf'))

        # Find nearest entry per env
        min_dists, min_idxs = dists.min(dim=1)                     # (R,), (R,)

        # Case 1: merge (nearest entry within merge_radius AND buffer non-empty)
        can_merge = (min_dists < self.merge_radius) & (counts_rec > 0)
        if can_merge.any():
            merge_env_ids = record_ids[can_merge]
            merge_slot_ids = min_idxs[can_merge]
            self.buffer[merge_env_ids, merge_slot_ids, 2] += 1.0

        # Case 2: append new entry
        should_append = ~can_merge
        if should_append.any():
            app_env_ids = record_ids[should_append]
            app_counts = self.counts[app_env_ids]

            # Sub-case 2a: buffer not full
            not_full = app_counts < self.max_entries
            if not_full.any():
                nf_ids = app_env_ids[not_full]
                nf_slots = self.counts[nf_ids]
                # Scatter new positions into their slots
                for i in range(len(nf_ids)):
                    eid = nf_ids[i]
                    slot = nf_slots[i]
                    self.buffer[eid, slot, 0] = robot_pos_xy[eid, 0]
                    self.buffer[eid, slot, 1] = robot_pos_xy[eid, 1]
                    self.buffer[eid, slot, 2] = 1.0
                self.counts[nf_ids] = nf_slots + 1

            # Sub-case 2b: buffer full → ring shift
            is_full = ~not_full
            if is_full.any():
                full_ids = app_env_ids[is_full]
                self.buffer[full_ids, :-1] = self.buffer[full_ids, 1:].clone()
                self.buffer[full_ids, -1, 0] = robot_pos_xy[full_ids, 0]
                self.buffer[full_ids, -1, 1] = robot_pos_xy[full_ids, 1]
                self.buffer[full_ids, -1, 2] = 1.0

    def to_obs(self, robot_pos_xy: torch.Tensor) -> torch.Tensor:
        """Convert buffer to robot-relative observation.

        Args:
            robot_pos_xy: Current robot XY positions (N, 2).

        Returns:
            Observation tensor (N, max_entries, 3) with [dx, dy, visit_count].
            Zero-padded for entries not yet filled.
        """
        obs = self.buffer.clone()
        # Make positions robot-relative
        obs[:, :, 0] -= robot_pos_xy[:, 0:1]
        obs[:, :, 1] -= robot_pos_xy[:, 1:2]
        # Zero out unfilled entries (vectorized)
        slot_indices = torch.arange(self.max_entries, device=self.device).unsqueeze(0)
        empty_mask = slot_indices >= self.counts.unsqueeze(1)  # (N, max_entries)
        obs[empty_mask] = 0.0
        return obs
