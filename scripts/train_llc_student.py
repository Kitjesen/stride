# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Train LLC Student via DAgger — arXiv:2405.01792.

DAgger distillation: teacher frozen, student drives sim, MSE loss.
The student GRU learns to estimate privileged info (velocity, contacts)
from IMU + encoder history.

Usage:
    python scripts/train_llc_student.py \
        --teacher_checkpoint logs/wln_llc_teacher/model_30000.pt \
        --num_envs 4096 --max_iterations 2000
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import torch.nn.functional as F

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wheeled_legged.networks.llc_student import LLCStudentPolicy
from wheeled_legged.agents.llc_ppo_cfg import LLCStudentDaggerCfg


def main():
    parser = argparse.ArgumentParser(description="Train LLC Student (DAgger)")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gru_hidden", type=int, default=512)
    parser.add_argument("--experiment_name", type=str, default="wln_llc_student")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== LLC Student DAgger Training ===")
    print(f"  Teacher: {args.teacher_checkpoint}")
    print(f"  Envs:    {args.num_envs}")
    print(f"  Iters:   {args.max_iterations}")
    print(f"  Device:  {device}")
    print()

    # ── Load teacher ──
    print("[1/4] Loading teacher checkpoint...")
    teacher_state = torch.load(args.teacher_checkpoint, map_location=device)
    # Teacher architecture depends on rsl_rl ActorCritic - loaded externally
    # For now, document the interface:
    print("  Teacher loaded. Must implement teacher.act(privileged_obs) -> actions")
    print("  NOTE: Full DAgger loop requires Isaac Lab environment.")
    print("        This script provides the training loop skeleton.")

    # ── Create student ──
    print("[2/4] Creating student network...")
    student = LLCStudentPolicy(
        proprio_dim=53,
        gru_hidden=args.gru_hidden,
    ).to(device)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"  Student params: {total_params:,}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)

    # ── DAgger training loop (skeleton) ──
    print("[3/4] DAgger training loop:")
    print("""
    DAgger pseudocode (requires Isaac Lab env):

    teacher.eval()  # frozen
    for iteration in range(max_iterations):
        env.reset()
        hidden = student.init_hidden(num_envs, device)

        for t in range(episode_steps):  # 500 steps @ 50Hz = 10s
            # Get observations
            proprio = env.get_noisy_proprio()          # (N, 53)
            height_scan = env.get_noisy_height_scan()  # (N, num_rays)
            privileged_obs = env.get_teacher_obs()     # (N, 53+25+num_rays)

            # Teacher labels (frozen, no grad)
            with torch.no_grad():
                teacher_actions = teacher.act(privileged_obs)

            # Student prediction (drives sim for on-policy coverage)
            student_actions, hidden = student(proprio, height_scan, hidden)
            env.step(student_actions)

        # MSE loss over collected trajectory
        loss = F.mse_loss(student_actions_all, teacher_actions_all)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if iteration % 100 == 0:
            print(f"  iter {iteration}: loss={loss.item():.6f}")
    """)

    # ── Save student ──
    log_dir = os.path.join("logs", args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "student_initial.pt")
    torch.save(student.state_dict(), save_path)
    print(f"[4/4] Student skeleton saved to {save_path}")
    print("  To complete: integrate with Isaac Lab env and run full DAgger loop.")


if __name__ == "__main__":
    main()
