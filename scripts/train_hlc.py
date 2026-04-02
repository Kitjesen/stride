# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Train HLC Navigation Controller — arXiv:2405.01792.

Hierarchical training: LLC is frozen, HLC outputs velocity commands.
HLC uses Beta distribution action space and 4-branch observation encoder.

Training loop:
  1. Generate WFC terrain + navigation graph
  2. Sample paths and waypoints via Dijkstra
  3. HLC observes: heightmap + LLC hidden + position buffer + waypoints
  4. HLC outputs: Beta(vx, vy, wz) velocity commands
  5. LLC executes 5 substeps per HLC step (50Hz / 10Hz)
  6. PPO update with goal + dense + exploration + stability rewards

Usage:
    python scripts/train_hlc.py \
        --llc_checkpoint logs/wln_llc_student/student_final.pt \
        --num_envs 1000 --max_iterations 20000
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wheeled_legged.networks.hlc_policy import HLCPolicy
from wheeled_legged.networks.beta_distribution import BetaDistribution, rescale_beta_actions
from wheeled_legged.networks.llc_student import LLCStudentPolicy
from wheeled_legged.rewards.hlc_rewards import (
    goal_reaching, dense_progress, exploration_penalty, near_goal_stability,
)
from wheeled_legged.utils.position_buffer import PositionBuffer
from wheeled_legged.utils.waypoint_manager import WaypointManager
from wheeled_legged.terrain.wfc_terrain import WFCConfig, generate_wfc_terrain
from wheeled_legged.terrain.nav_graph import NavGraph
from wheeled_legged.terrain.dynamic_obstacles import DynamicObstacleManager, DynamicObstacleConfig
from wheeled_legged.envs.hlc_env_cfg import HLCEnvCfg
from wheeled_legged.agents.hlc_ppo_cfg import HLCPPOConfig


def main():
    parser = argparse.ArgumentParser(description="Train HLC Navigation Controller")
    parser.add_argument("--llc_checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1000)
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--experiment_name", type=str, default="stride_hlc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = HLCEnvCfg(num_envs=args.num_envs)
    ppo_cfg = HLCPPOConfig(num_envs=args.num_envs, max_iterations=args.max_iterations)

    print("=== HLC Navigation Training ===")
    print(f"  LLC checkpoint: {args.llc_checkpoint}")
    print(f"  Envs: {args.num_envs}, Iters: {args.max_iterations}")
    print()

    # ── Step 1: Generate WFC terrain + nav graph ──
    print("[1/5] Generating WFC terrain...")
    tile_grid, heightmap = generate_wfc_terrain(env_cfg.wfc)
    print(f"  Tile grid: {tile_grid.shape}, Heightmap: {heightmap.shape}")

    nav = NavGraph(tile_grid, env_cfg.wfc)
    print(f"  Nav graph: {nav.num_nodes} nodes, {nav.num_edges} edges")

    # ── Step 2: Load frozen LLC ──
    print("[2/5] Loading frozen LLC student...")
    llc = LLCStudentPolicy(gru_hidden=env_cfg.llc_gru_hidden).to(device)
    llc.load_state_dict(torch.load(args.llc_checkpoint, map_location=device))
    llc.eval()
    for p in llc.parameters():
        p.requires_grad = False
    print(f"  LLC frozen ({sum(p.numel() for p in llc.parameters()):,} params)")

    # ── Step 3: Create HLC policy ──
    print("[3/5] Creating HLC policy...")
    hlc = HLCPolicy(
        llc_hidden_dim=env_cfg.llc_gru_hidden,
        heightmap_h=env_cfg.heightmap_h,
        heightmap_w=env_cfg.heightmap_w,
    ).to(device)
    beta_dist = BetaDistribution(output_dim=3)
    print(f"  HLC params: {sum(p.numel() for p in hlc.parameters()):,}")

    optimizer = torch.optim.Adam(hlc.parameters(), lr=ppo_cfg.learning_rate)

    # ── Step 4: Create utilities ──
    print("[4/5] Initializing buffers...")
    pos_buffer = PositionBuffer(
        args.num_envs, max_entries=env_cfg.pos_buffer_size,
        interval=env_cfg.pos_buffer_interval, device=device,
    )
    wp_manager = WaypointManager(
        args.num_envs, goal_threshold=env_cfg.goal_threshold,
        lookahead_range=env_cfg.lookahead_range, device=device,
    )
    obstacles = DynamicObstacleManager(args.num_envs, env_cfg.obstacles, device=device)

    # ── Step 5: Training loop (skeleton) ──
    print("[5/5] HLC PPO training loop:")
    print(f"""
    Training skeleton (requires Isaac Lab for full physics):

    for iteration in range({args.max_iterations}):
        # Reset environments
        for env_id in reset_ids:
            path, start, goal = nav.sample_random_path(min_length=5.0)
            wp1, wp2 = WaypointManager.sample_waypoints_on_path(path_tensor)
            wp_manager.reset(env_id, wp1, wp2)
            pos_buffer.reset(env_id)
            obstacles.reset(env_id, env_origins)

        llc_hidden = llc.init_hidden(num_envs, device)

        for hlc_step in range({ppo_cfg.episode_length}):
            # HLC observation
            heightmap_obs = get_temporal_heightmap()    # (N, 3, 16, 26)
            llc_hidden_obs = llc.get_hidden_state(llc_hidden)  # (N, 512)
            pos_obs = pos_buffer.to_obs(robot_pos)     # (N, 20, 3)
            wp_obs = wp_manager.get_obs(robot_pos, robot_yaw)  # (N, 17)

            # HLC action (Beta distribution)
            raw_params, value = hlc(heightmap_obs, llc_hidden_obs, pos_obs, wp_obs)
            beta_dist.update(raw_params)
            raw_action = beta_dist.sample()
            log_prob = beta_dist.log_prob(raw_action)
            vel_cmd = rescale_beta_actions(raw_action,
                [{env_cfg.vx_range[0]}, {env_cfg.vy_range[0]}, {env_cfg.wz_range[0]}],
                [{env_cfg.vx_range[1]}, {env_cfg.vy_range[1]}, {env_cfg.wz_range[1]}])

            # LLC inner loop (5 substeps at 50Hz)
            for _ in range(5):
                llc_actions, llc_hidden = llc(proprio, height_scan, llc_hidden)
                env.step_physics(llc_actions)

            # Rewards
            r_goal = goal_reaching(robot_pos, wp1)
            r_dense = dense_progress(robot_vel, robot_pos, wp1)
            r_exp = exploration_penalty(robot_pos, wp1, pos_obs)
            r_stab = near_goal_stability(robot_vel, robot_pos, wp1)
            reward = w_goal*r_goal + w_dense*r_dense + w_exp*r_exp + w_stab*r_stab + w_llc*llc_reward

            # Update buffers
            pos_buffer.update(robot_pos)
            wp_manager.update(robot_pos, vel_cmd)
            obstacles.step(0.1, robot_pos, env_origins)

        # PPO update (gamma={ppo_cfg.gamma}, clip={ppo_cfg.clip_param})
        advantages = compute_gae(rewards, values, gamma={ppo_cfg.gamma}, lam={ppo_cfg.lam})
        for epoch in range({ppo_cfg.num_learning_epochs}):
            for mb in minibatches({ppo_cfg.num_mini_batches}):
                ppo_loss = clip_surrogate + value_loss - entropy_bonus
                optimizer.step()
    """)

    # Save initial model
    log_dir = os.path.join("logs", args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(log_dir, "hlc_initial.pt")
    torch.save(hlc.state_dict(), save_path)
    print(f"  HLC skeleton saved to {save_path}")
    print("  To complete: integrate with Isaac Lab physics simulation.")


if __name__ == "__main__":
    main()
