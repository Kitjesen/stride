# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Stride PPO Runner Config — arXiv:2405.01792 on robot_lab infrastructure.

Based on ThunderHistRoughPPORunnerCfg with paper-specific hyperparameters.

Paper Table S2:
  discount=0.99, KL target=0.01, clip=0.2, entropy=0.001,
  batch=500k, minibatches=20, epochs=4
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ThunderStridePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training for Stride LLC teacher on robot_lab.

    Key differences from ThunderHistRoughPPORunnerCfg:
    - num_steps_per_env: 24 (not 200) — matches paper batch structure
    - entropy: 0.001 (paper Table S2)
    - mini_batches: 20 (paper) vs 4 (thunder-him)
    - max_grad_norm: 1.0 (paper) vs 10.0 (thunder-him)
    """

    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    save_interval: int = 500
    experiment_name: str = "thunder_stride_rough"
    class_name: str = "OnPolicyRunner"

    # Observation groups: same structure as thunder-him
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["critic", "height_scan_group"],
    }

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,                   # [stated] Table S2
        entropy_coef=0.001,               # [stated] Table S2
        num_learning_epochs=4,            # [stated] Table S2
        num_mini_batches=20,              # [stated] Table S2
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,                       # [stated] Table S2
        lam=0.95,
        desired_kl=0.01,                  # [stated] Table S2
        max_grad_norm=1.0,
    )


@configclass
class ThunderStrideFlatPPORunnerCfg(ThunderStridePPORunnerCfg):
    """Flat terrain variant — for initial debugging."""
    experiment_name: str = "thunder_stride_flat"
    max_iterations: int = 10_000
