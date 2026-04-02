# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""HLC 4-Branch Policy Network — arXiv:2405.01792.

The HLC processes 4 input modalities through parallel branches:
  1. Heightmap branch:  3-layer 2D CNN (3 temporal scans as channels)
  2. Position buffer:   1D CNN + MaxPool (PointNet-like, permutation invariant)
  3. LLC hidden state:  Plain MLP
  4. Waypoints + history: Plain MLP

Branches are concatenated and passed through a merge MLP that outputs
6 values (Sigmoid) parameterizing 3 Beta distributions for velocity commands.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeightmapBranch(nn.Module):
    """3-layer 2D CNN for elevation map processing.

    Input: (batch, 3, H, W) — 3 temporal scans stacked as channels.
    Output: (batch, 256)
    """

    def __init__(self, h: int = 16, w: int = 26):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Compute flattened size after convolutions
        h2 = (h + 1) // 2  # after stride=2
        w2 = (w + 1) // 2
        h3 = (h2 + 1) // 2
        w3 = (w2 + 1) // 2
        self.flat_size = 64 * h3 * w3

        self.fc = nn.Linear(self.flat_size, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        return F.relu(self.fc(x))


class PositionBufferBranch(nn.Module):
    """1D CNN + MaxPool (PointNet-like) for position history.

    Input: (batch, 20, 3) — [dx, dy, visit_count] per entry.
    Output: (batch, 128)

    Permutation invariant via max-pooling over entries.
    """

    def __init__(self, max_entries: int = 20, entry_dim: int = 3):
        super().__init__()
        # Per-point MLPs (Conv1d with kernel=1)
        self.conv1 = nn.Conv1d(entry_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.fc = nn.Linear(256, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, 20, 3) → (batch, 3, 20)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Max pool over points → (batch, 256)
        x = x.max(dim=2).values
        return F.relu(self.fc(x))


class MLPBranch(nn.Module):
    """Plain MLP branch for LLC hidden state or waypoints."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HLCPolicy(nn.Module):
    """HLC 4-branch navigation policy.

    Observation modalities:
    - heightmap: (batch, 3, 16, 26) — 3 temporal elevation scans
    - llc_hidden: (batch, H_llc) — LLC RNN hidden state
    - pos_buffer: (batch, 20, 3) — visited position history
    - waypoints: (batch, 17) — WP1, WP2, prev WPs, prev cmds

    Output: 6 values (Sigmoid) → 3 Beta distributions (vx, vy, ωz)
    """

    def __init__(
        self,
        llc_hidden_dim: int = 512,
        heightmap_h: int = 16,
        heightmap_w: int = 26,
    ):
        super().__init__()

        # ── 4 parallel branches ──
        self.heightmap_branch = HeightmapBranch(heightmap_h, heightmap_w)    # → 256
        self.pos_buffer_branch = PositionBufferBranch()                       # → 128
        self.llc_hidden_branch = MLPBranch(llc_hidden_dim, 128, 128)         # → 128
        self.waypoint_branch = MLPBranch(17, 64, 64)                          # → 64

        # ── Merge head: 256+128+128+64 = 576 → 6 ──
        merge_dim = 256 + 128 + 128 + 64
        self.merge = nn.Sequential(
            nn.Linear(merge_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        # Actor output: 6 values → Sigmoid → Beta params
        self.actor_head = nn.Linear(256, 6)
        # Critic output: scalar value
        self.critic_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable starting distribution."""
        nn.init.zeros_(self.actor_head.weight)
        nn.init.constant_(self.actor_head.bias, 0.5)  # Sigmoid(0.5)≈0.62 → moderate alpha/beta

    def forward(
        self,
        heightmap: torch.Tensor,
        llc_hidden: torch.Tensor,
        pos_buffer: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            beta_params: (batch, 2, 3) — Sigmoid-activated params for Beta distribution
            value: (batch, 1) — critic value estimate
        """
        h1 = self.heightmap_branch(heightmap)      # (batch, 256)
        h2 = self.pos_buffer_branch(pos_buffer)    # (batch, 128)
        h3 = self.llc_hidden_branch(llc_hidden)    # (batch, 128)
        h4 = self.waypoint_branch(waypoints)       # (batch, 64)

        merged = torch.cat([h1, h2, h3, h4], dim=1)  # (batch, 576)
        features = self.merge(merged)                   # (batch, 256)

        # Actor: raw logits → reshape to (batch, 2, 3)
        # NOTE: Do NOT apply Sigmoid here. BetaDistribution.update() applies its own Sigmoid.
        # If using HLCPolicy standalone (without BetaDistribution), call torch.sigmoid() externally.
        raw = self.actor_head(features)               # (batch, 6)
        beta_params = raw.reshape(-1, 2, 3)

        # Critic
        value = self.critic_head(features)             # (batch, 1)

        return beta_params, value

    def act(
        self,
        heightmap: torch.Tensor,
        llc_hidden: torch.Tensor,
        pos_buffer: torch.Tensor,
        waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic action (Beta mean) for inference."""
        raw_params, _ = self.forward(heightmap, llc_hidden, pos_buffer, waypoints)
        # Apply Sigmoid here since BetaDistribution is not used in inference path
        a1 = torch.sigmoid(raw_params[:, 0, :])  # (batch, 3)
        a2 = torch.sigmoid(raw_params[:, 1, :])  # (batch, 3)
        alpha = (a1 * a2).clamp(min=1e-4)
        beta = (a2 * (1.0 - a1)).clamp(min=1e-4)
        return alpha / (alpha + beta)  # Beta mean in (0, 1)
