# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""LLC PPO Training Configuration — arXiv:2405.01792 Table S2.

Teacher PPO hyperparameters for the low-level locomotion controller.
Based on Rudin et al. legged_gym conventions + paper-stated values.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LLCTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO training for LLC teacher policy.

    Teacher receives privileged observations:
      - Proprioceptive: IMU(6) + joint angles(12) + joint vel(12) + prev actions(16) + cmd(3) = 49D
      - Privileged: base vel(3) + accel(3) + terrain normal(3) + contacts(4) + contact forces(12)
                    + terrain props(5) + gravity(3) + noiseless height scan(87) ≈ 120D
      - Total teacher obs: ~170D

    Output: 16D = 12 joint position targets + 4 wheel velocity targets

    From paper Table S2:
      discount=0.99, KL target=0.01, clip=0.2, entropy=0.001,
      episode=10s (500 steps @ 50Hz), dt=0.02, batch=500000,
      minibatches=20, epochs=4, learning_rate=adaptive
    """

    # --- Runner ---
    num_steps_per_env: int = 24           # steps per env per iteration
    max_iterations: int = 30_000          # ~12-18h on single RTX 3090
    save_interval: int = 500
    experiment_name: str = "wln_llc_teacher"
    empirical_normalization: bool = False

    # --- Actor-Critic ---
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # --- PPO Algorithm (Table S2) ---
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,                   # [stated]
        entropy_coef=0.001,               # [stated]
        num_learning_epochs=4,            # [stated]
        num_mini_batches=20,              # [stated]
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,                       # [stated]
        lam=0.95,
        desired_kl=0.01,                  # [stated]
        max_grad_norm=1.0,
    )


@configclass
class LLCStudentDaggerCfg:
    """DAgger student training configuration.

    Student: GRU(hidden=512) with noisy proprioceptive + height scan.
    Teacher: frozen MLP teacher policy.
    Loss: MSE between teacher and student actions.

    Not using rsl_rl DistillationRunner because DAgger (student-drives-sim)
    requires custom rollout logic. Implemented in train_llc_student.py.
    """
    teacher_checkpoint: str = ""          # path to teacher .pt
    max_iterations: int = 2_000
    batch_size: int = 500_000
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    gru_hidden_size: int = 512
    gru_num_layers: int = 1
    save_interval: int = 100
    experiment_name: str = "wln_llc_student"


# ============================================================================
# Reward Weights — arXiv:2405.01792 (inferred from RSL conventions)
# ============================================================================

LLC_REWARD_WEIGHTS = {
    "linear_velocity_tracking": 1.0,
    "angular_velocity_tracking": 0.5,
    "base_motion_penalty": -2.0,
    "orientation_penalty": -0.2,
    "base_height_penalty": -1.0,
    "torque_penalty": -1e-5,
    "joint_velocity_acceleration_penalty": -2.5e-7,
    "action_smoothness_penalty": -0.01,
    "joint_constraint_penalty": -10.0,
    "body_contact_penalty": -1.0,
    "survival_reward": 1.0,
}
