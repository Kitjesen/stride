# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""HLC PPO Training Config — arXiv:2405.01792 Table S3.

High-Level Controller PPO hyperparameters.
HLC is trained with LLC frozen; outputs Beta-distributed velocity commands.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HLCPPOConfig:
    """HLC PPO hyperparameters from paper Table S3."""

    # ── PPO Algorithm ──
    gamma: float = 0.991                  # [stated] Table S3
    lam: float = 0.95                     # GAE lambda
    clip_param: float = 0.2              # [stated]
    entropy_coef: float = 0.001          # [stated]
    value_loss_coef: float = 0.5
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01             # [stated] adaptive LR

    # ── Training ──
    num_learning_epochs: int = 5         # [stated] Table S3
    num_mini_batches: int = 10           # [stated] Table S3
    max_iterations: int = 20_000
    save_interval: int = 500

    # ── Rollout ──
    episode_length: int = 150            # 15s / 0.1s = 150 steps
    batch_size: int = 150_000            # [stated] Table S3
    num_envs: int = 1000                 # 150k / 150 steps

    # ── Network ──
    llc_hidden_dim: int = 512            # must match LLC student GRU
    heightmap_h: int = 16
    heightmap_w: int = 26

    # ── Experiment ──
    experiment_name: str = "stride_hlc"
    log_interval: int = 10
