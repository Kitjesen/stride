# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""LLC Student Network (GRU) — arXiv:2405.01792.

The student policy uses a GRU to extract temporal features from
noisy proprioceptive observations, replacing the teacher's access
to privileged information (ground-truth velocity, contact states).

Paper key design: raw IMU measurements (linear acceleration + angular velocity)
instead of state-estimator outputs. The GRU implicitly learns to estimate
base velocity and contact state from IMU temporal history.

Input proprio (53D):
  imu_gyro(3) + imu_accel(3) + leg_pos(12) + leg_vel(12)
  + wheel_vel(4) + prev_actions(16) + cmd(3)

Architecture:
  proprio_encoder: Linear(53, 64) -> ELU -> Linear(64, 32)
  height_encoder:  Linear(num_rays, 64) -> ELU -> Linear(64, 32)
  GRU:             input=64, hidden=512, 1 layer
  actor_head:      Linear(512, 256) -> ELU -> Linear(256, 128) -> ELU -> Linear(128, 16) -> tanh

Training: DAgger (teacher frozen, student drives sim, MSE loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LLCStudentPolicy(nn.Module):
    """GRU-based student policy for LLC.

    Input: noisy proprioceptive (53D) + noisy height scan (num_rays)
    Output: 16D actions (12 leg joint pos + 4 wheel vel)
    Hidden: GRU state carries implicit velocity/contact estimation.
    """

    def __init__(
        self,
        proprio_dim: int = 53,
        height_dim: int = 187,     # num_rays from RayCaster, tune per grid
        gru_hidden: int = 512,
        action_dim: int = 16,
    ):
        super().__init__()
        self.gru_hidden = gru_hidden

        # Proprioceptive encoder
        self.proprio_enc = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )

        # Height scan encoder
        self.height_enc = nn.Sequential(
            nn.Linear(height_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )

        # GRU: temporal fusion (key for estimating velocity from IMU history)
        self.gru = nn.GRU(
            input_size=64,     # 32 + 32 from encoders
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(gru_hidden, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m == self.actor[-2] else 1.0)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        proprio: torch.Tensor,
        height_scan: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            proprio: (batch, 53) noisy proprioceptive obs.
            height_scan: (batch, num_rays) noisy height scan.
            hidden: (1, batch, gru_hidden) GRU hidden state.

        Returns:
            actions: (batch, 16) action output.
            hidden: (1, batch, gru_hidden) updated GRU hidden state.
        """
        p = self.proprio_enc(proprio)        # (batch, 32)
        h = self.height_enc(height_scan)     # (batch, 32)
        features = torch.cat([p, h], dim=1)  # (batch, 64)

        # GRU expects (batch, seq=1, features)
        gru_in = features.unsqueeze(1)
        gru_out, hidden = self.gru(gru_in, hidden)
        gru_out = gru_out.squeeze(1)         # (batch, 512)

        actions = self.actor(gru_out)
        return actions, hidden

    def init_hidden(self, batch_size: int, device: str = "cuda") -> torch.Tensor:
        """Create zero-initialized GRU hidden state."""
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def get_hidden_state(self, hidden: torch.Tensor) -> torch.Tensor:
        """Extract hidden state vector for HLC observation.

        Returns: (batch, gru_hidden) — the LLC belief state that HLC uses
        instead of standard proprioception.
        """
        return hidden.squeeze(0)  # (batch, 512)
