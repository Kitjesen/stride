#!/usr/bin/env python3
"""Test contact forces on stair terrain with Thunder Hist model."""
import argparse, os, sys
from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

parser = argparse.ArgumentParser(description="Test contact forces on stairs")
parser.add_argument("--task", type=str, default="Hist")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--test_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robot_lab.tasks  # noqa: F401
import gymnasium as gym
import torch
import csv
from datetime import datetime
from tensordict import TensorDict
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from runners.him_on_policy_runner import HIMOnPolicyRunner

CHECKPOINT = "/home/bsrl/hongsenpang/RLbased/robot_lab/logs/rsl_rl/thunder_hist_rough/2026-03-19_17-22-06/model_40000.pt"


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.scene.terrain.max_init_terrain_level = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    history_len = 10
    try:
        hl = getattr(env_cfg.observations.policy, "history_length", None)
        if hl:
            history_len = hl
    except Exception:
        pass
    print(f"[INFO] history_length = {history_len}")

    train_cfg = {
        "runner": {
            "policy_class_name": "HIMActorCritic",
            "algorithm_class_name": "HIMPPO",
            "num_steps_per_env": 24,
            "save_interval": 50,
        },
        "algorithm": {
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.998,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": 0.01,
        },
        "policy": {
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
            "estimator_latent_dim": 16,
            "estimator_lr": 1e-3,
            "num_prototype": 32,
        },
        "history_len": history_len,
    }

    print(f"[INFO] Loading: {CHECKPOINT}")
    runner = HIMOnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(CHECKPOINT)
    raw_policy = runner.get_inference_policy(device=env.unwrapped.device)

    def policy(obs):
        if isinstance(obs, TensorDict):
            policy_obs = obs["policy"]
        else:
            policy_obs = obs
        if runner.need_reshape:
            from utils.observation_reshaper import reshape_isaac_to_him
            policy_obs = reshape_isaac_to_him(
                policy_obs, history_len=history_len, obs_dims=runner.policy_dims
            )
        return raw_policy(policy_obs)

    unwrapped = env.unwrapped
    contact_sensor = None
    for key, sensor in unwrapped.scene.sensors.items():
        if "contact" in key.lower():
            contact_sensor = sensor
            print(f"[INFO] Contact sensor: {key}")
            break
    if contact_sensor is None:
        print("[ERROR] No contact sensor!")
        env.close()
        simulation_app.close()
        return

    # Print all available body names to find correct foot names
    all_bodies = contact_sensor.body_names
    print(f"[INFO] All contact bodies: {all_bodies}")

    # Try multiple naming patterns
    foot_patterns = [
        ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
        ["fr_foot_Link", "fl_foot_Link", "rr_foot_Link", "rl_foot_Link"],
        ["FR_FOOT", "FL_FOOT", "RR_FOOT", "RL_FOOT"],
    ]
    foot_names = foot_patterns[0]
    foot_ids = []
    for patterns in foot_patterns:
        foot_ids = []
        for fn in patterns:
            try:
                ids, _ = contact_sensor.find_bodies(fn)
                foot_ids.append(ids[0] if len(ids) > 0 else -1)
            except Exception:
                foot_ids.append(-1)
        if all(fid >= 0 for fid in foot_ids):
            foot_names = patterns
            break
    # Fallback: use last 4 bodies
    if all(fid <= 0 for fid in foot_ids):
        n_bodies = len(all_bodies)
        foot_ids = list(range(max(0, n_bodies - 4), n_bodies))
        foot_names = [all_bodies[i] for i in foot_ids]
        print(f"[WARN] Using fallback foot IDs: {dict(zip(foot_names, foot_ids))}")
    else:
        print(f"[INFO] Foot IDs: {dict(zip(foot_names, foot_ids))}")

    # get_observations returns different types depending on wrapper
    obs_result = env.get_observations()
    if isinstance(obs_result, tuple):
        obs = obs_result[0]
    else:
        obs = obs_result
    max_forces = torch.zeros(4, device=unwrapped.device)
    sum_forces = torch.zeros(4, device=unwrapped.device)
    sum_fz = torch.zeros(4, device=unwrapped.device)
    impact_count = torch.zeros(4, device=unwrapped.device)
    step_count = 0
    rows = []

    print(f"\n[INFO] Recording {args_cli.test_steps} steps, {args_cli.num_envs} envs...")
    for step in range(args_cli.test_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        forces = contact_sensor.data.net_forces_w_history[:, 0]
        robot = unwrapped.scene["robot"]

        for fi, fid in enumerate(foot_ids):
            f = forces[:, fid]
            f_mag = torch.norm(f, dim=1)
            f_z = f[:, 2].abs()
            max_forces[fi] = max(max_forces[fi], f_mag.max())
            sum_forces[fi] += f_mag.mean()
            sum_fz[fi] += f_z.mean()
            impact_count[fi] += (f_mag > 100.0).float().mean()

        for fi, fid in enumerate(foot_ids):
            f = forces[0, fid]
            rows.append({
                "step": step,
                "foot": foot_names[fi][:2],
                "fx": f"{f[0].item():.2f}",
                "fy": f"{f[1].item():.2f}",
                "fz": f"{f[2].item():.2f}",
                "f_mag": f"{torch.norm(f).item():.2f}",
                "base_h": f"{robot.data.root_pos_w[0, 2].item():.4f}",
                "vx": f"{robot.data.root_lin_vel_b[0, 0].item():.3f}",
            })

        step_count += 1
        if step % 200 == 0:
            avg = sum_forces / max(step_count, 1)
            print(
                f"  Step {step:4d} | Avg(N) FR={avg[0]:.1f} FL={avg[1]:.1f}"
                f" RR={avg[2]:.1f} RL={avg[3]:.1f} |"
                f" Max(N) FR={max_forces[0]:.0f} FL={max_forces[1]:.0f}"
                f" RR={max_forces[2]:.0f} RL={max_forces[3]:.0f}"
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"logs/contact_force_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    avg = sum_forces / step_count
    avg_fz = sum_fz / step_count
    impact_pct = impact_count / step_count * 100
    robot_weight = 18.7 * 9.81
    static_per_foot = robot_weight / 4

    print(f"\n{'=' * 70}")
    print(f" CONTACT FORCE REPORT - Thunder Hist Rough (model_40000)")
    print(f"{'=' * 70}")
    print(f" Test: {args_cli.test_steps} steps x {args_cli.num_envs} envs on mixed terrain")
    print()
    print(f" {'Foot':<6} {'Avg|F|(N)':<12} {'Avg Fz(N)':<12} {'Peak(N)':<10} {'Impact>100N':<12} {'Peak/Static'}")
    print(f" {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 12} {'-' * 12}")
    for i, fn in enumerate(["FR", "FL", "RR", "RL"]):
        print(
            f" {fn:<6} {avg[i]:<12.1f} {avg_fz[i]:<12.1f} {max_forces[i]:<10.0f}"
            f" {impact_pct[i]:<12.1f}% {max_forces[i] / static_per_foot:<.1f}x"
        )
    print()
    print(f" Robot weight: {robot_weight:.0f}N | Static/foot: {static_per_foot:.0f}N")
    print(f" CSV: {csv_path}")
    print(f"{'=' * 70}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
