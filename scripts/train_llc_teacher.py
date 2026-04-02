# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Train LLC Teacher Policy — arXiv:2405.01792.

Usage:
    python scripts/train_llc_teacher.py --num_envs 4096 --max_iterations 30000

Requires Isaac Lab + rsl_rl to be installed.
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Isaac Lab bootstrap ──
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train LLC Teacher for Wheeled-Legged Robot")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
parser.add_argument("--experiment_name", type=str, default="wln_llc_teacher")
parser.add_argument("--seed", type=int, default=42)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Imports after Isaac Sim init ──
import torch
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from rsl_rl.runners import OnPolicyRunner

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wheeled_legged.envs.llc_env import WheeledLLCEnv
from wheeled_legged.envs.llc_env_cfg import WheeledLLCEnvCfg
from wheeled_legged.agents.llc_ppo_cfg import LLCTeacherPPORunnerCfg


def main():
    # ── Configure environment ──
    env_cfg = WheeledLLCEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # ── Create environment ──
    env = WheeledLLCEnv(cfg=env_cfg)

    # ── Configure runner ──
    runner_cfg = LLCTeacherPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg.experiment_name = args.experiment_name

    # ── Log directory ──
    log_root = os.path.join("logs", "wln_llc_teacher")
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    # ── Create runner ──
    runner = OnPolicyRunner(
        env=env,
        train_cfg=runner_cfg,
        log_dir=log_dir,
        device=env.device,
    )

    # ── Resume from checkpoint ──
    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        runner.load(args.resume)

    # ── Train ──
    print(f"[Train] Starting LLC teacher training")
    print(f"  Envs:       {args.num_envs}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  Log dir:    {log_dir}")
    print(f"  Device:     {env.device}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # ── Cleanup ──
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
