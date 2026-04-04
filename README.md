# Stride

Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots

Reproduction of [arXiv:2405.01792](https://arxiv.org/abs/2405.01792) (ETH Zurich RSL, Science Robotics 2024)

## Training Pipeline

```
Phase 1: LLC Teacher PPO     ← CURRENT (robot_lab, GPU 0)
    │   MLP teacher with privileged obs (10-frame history)
    │   16D action: 12 leg joints + 4 wheel velocities
    │   Rough terrain curriculum
    ▼
Phase 2: LLC Student DAgger
    │   GRU(512) student, noisy obs only
    │   Teacher frozen, student drives sim, MSE loss
    │   Output: student checkpoint + hidden state for HLC
    ▼
Phase 3: HLC Navigation PPO
    │   4-branch network (2D CNN + PointNet + 2×MLP)
    │   Beta distribution → [vx, vy, ωz] velocity commands
    │   WFC terrain + navigation graph + dynamic obstacles
    ▼
Phase 4: Evaluation & Deployment
        SPL metric, ablation, video, ONNX export
```

## Project Structure

```
stride/
├── scripts/
│   ├── train_llc_teacher.py          # Phase 1: LLC Teacher PPO
│   ├── train_llc_student.py          # Phase 2: LLC Student DAgger
│   ├── train_hlc.py                  # Phase 3: HLC Navigation
│   └── test_contact_force_stairs.py  # Contact force evaluation
├── wheeled_legged/
│   ├── networks/
│   │   ├── beta_distribution.py      # Beta dist for PPO (HLC)
│   │   ├── hlc_policy.py            # 4-branch HLC (892K params)
│   │   └── llc_student.py           # GRU student (1.07M params)
│   ├── rewards/
│   │   ├── llc_rewards.py           # 11 terms (Eq.14-25)
│   │   └── hlc_rewards.py           # 4 terms (Eq.9-13)
│   ├── terrain/
│   │   ├── wfc_terrain.py           # Wave Function Collapse
│   │   ├── nav_graph.py             # Dijkstra navigation graph
│   │   └── dynamic_obstacles.py     # Moving box obstacles
│   ├── utils/
│   │   ├── position_buffer.py       # 20-entry visit tracker
│   │   └── waypoint_manager.py      # Anchor pursuit
│   ├── envs/
│   │   ├── llc_env.py               # LLC DirectRLEnv
│   │   ├── llc_env_cfg.py           # Standalone config
│   │   ├── stride_llc_env_cfg.py    # robot_lab compatible
│   │   └── hlc_env_cfg.py           # HLC environment
│   └── agents/
│       ├── llc_ppo_cfg.py           # LLC PPO (Table S2)
│       ├── stride_ppo_cfg.py        # Server PPO config
│       └── hlc_ppo_cfg.py           # HLC PPO (Table S3)
└── deploy_to_server.sh
```

## Server Training

### BSRL 8×RTX 3090

```bash
ssh -p 12346 bsrl@fe91fae6a6756695.natapp.cc

# Stride (robot_lab framework)
cd /home/bsrl/hongsenpang/RLbased/robot_lab
CUDA_VISIBLE_DEVICES=0 python -u scripts/reinforcement_learning/rsl_rl/train_him.py \
    --task Stride --headless --num_envs 4096 --max_iterations 40000

# Check progress
tail -50 logs/stride_robotlab.log
```

### Config location on server

```
# Stride task config (extends ThunderHistRoughEnvCfg):
source/robot_lab/robot_lab/tasks/.../thunder_hist/stride_env_cfg.py

# Registered as task "Stride" in thunder_hist/__init__.py
```

## Paper Observation Structure

### LLC Shared Proprioception (53D)
| Signal | Dim | Source | Note |
|--------|-----|--------|------|
| angular_velocity | 3 | IMU gyroscope | |
| **linear_acceleration** | **3** | **IMU accelerometer** | **specific force, NOT projected gravity** |
| leg_joint_pos | 12 | encoders | offset from default |
| leg_joint_vel | 12 | encoders | scaled ×0.05 |
| wheel_joint_vel | 4 | wheel encoders | scaled ×0.1 |
| prev_actions | 16 | self | |
| velocity_command | 3 | HLC / external | |

Paper key design: *"we directly used IMU measurements consisting of linear acceleration
and angular velocity"* — skips state estimator entirely, GRU learns implicit estimation.

### LLC Teacher (privileged additions)
| Signal | Dim | Source |
|--------|-----|--------|
| height_scan (noiseless) | ~187 | RayCaster |
| base_lin_vel | 3 | ground truth |
| gravity_vector | 3 | ground truth orientation |
| foot_contacts | 4 | contact sensor |
| contact_forces | 12 | contact sensor (3D×4) |
| terrain_normals | 12 | terrain mesh (3D×4 feet) |
| terrain_properties | 5 | domain randomization state |

### LLC Student (deployment)
Shared proprio (53D) + noisy height scan (~187D). No privileged signals.
GRU(hidden=512) estimates velocity/contact from IMU temporal history.

### HLC Navigator
| Branch | Input | Processing |
|--------|-------|------------|
| Heightmap | 3×16×26 (temporal) | 3-layer 2D CNN |
| LLC hidden | 512 | MLP |
| Position buffer | 20×3 | 1D CNN + MaxPool |
| Waypoints | 17 | MLP |

Output: Beta distribution → vx∈[-1,2], vy∈[-0.75,0.75], ωz∈[-1.25,1.25]

## Key Decisions

1. **robot_lab > legged_lab** for terrain curriculum (success-rate based, no demotion bug)
2. **10-frame history** observation (not 5-frame) for richer temporal context
3. **Conservative velocity range** (±1.5) for from-scratch training, expand later
4. **Soft landing rewards**: feet_impact_vel=-2.0, contact_forces=-0.01 (tested, reduces GRF)

## References

- [Paper](https://arxiv.org/abs/2405.01792) | [Project Page](https://junja94.github.io/learning_robust_autonomous_navigation_and_locomotion_for_wheeled_legged_robots/)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | [terrain-generator](https://github.com/leggedrobotics/terrain-generator)
