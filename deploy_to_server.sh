#!/usr/bin/env bash
# Deploy B2W wheeled-legged navigation code to BSRL training server.
#
# Usage:
#   bash deploy_to_server.sh
#
# Server: AutoDL RTX 5090 / BSRL 8×RTX 3090
# Target: /root/autodl-tmp/thunder2/

set -euo pipefail

# ── Server config ──
SERVER_HOST="${BSRL_HOST:-}"
SERVER_PORT="${BSRL_PORT:-12346}"
SERVER_USER="${BSRL_USER:-bsrl}"
REMOTE_BASE="/root/autodl-tmp/thunder2"

if [ -z "$SERVER_HOST" ]; then
    echo "Error: Set BSRL_HOST env var (e.g., fe91fae6a6756695.natapp.cc)"
    echo "  export BSRL_HOST=fe91fae6a6756695.natapp.cc"
    echo "  export BSRL_PORT=12346"
    echo "  export BSRL_USER=bsrl"
    exit 1
fi

SSH_CMD="ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
SCP_CMD="scp -P $SERVER_PORT"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== B2W Deploy ==="
echo "  Local:  $LOCAL_DIR"
echo "  Remote: $SERVER_USER@$SERVER_HOST:$REMOTE_BASE"
echo ""

# ── Step 1: Create remote directory ──
echo "[1/4] Creating remote directories..."
$SSH_CMD "mkdir -p $REMOTE_BASE/b2w_nav/{envs,networks,rewards,utils,agents,scripts}"

# ── Step 2: Upload core modules ──
echo "[2/4] Uploading core modules..."

# Networks
$SCP_CMD "$LOCAL_DIR/wheeled_legged/networks/beta_distribution.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/networks/"
$SCP_CMD "$LOCAL_DIR/wheeled_legged/networks/hlc_policy.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/networks/"

# Rewards
$SCP_CMD "$LOCAL_DIR/wheeled_legged/rewards/llc_rewards.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/rewards/"
$SCP_CMD "$LOCAL_DIR/wheeled_legged/rewards/hlc_rewards.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/rewards/"

# Utils
$SCP_CMD "$LOCAL_DIR/wheeled_legged/utils/position_buffer.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/utils/"
$SCP_CMD "$LOCAL_DIR/wheeled_legged/utils/waypoint_manager.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/utils/"

# Env configs (robot_lab compatible)
$SCP_CMD "$LOCAL_DIR/wheeled_legged/envs/b2w_llc_env_cfg.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/envs/"
$SCP_CMD "$LOCAL_DIR/wheeled_legged/agents/b2w_ppo_cfg.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/agents/"

# __init__.py files
for d in "" envs networks rewards utils agents; do
    $SCP_CMD "$LOCAL_DIR/wheeled_legged/${d:+$d/}__init__.py" \
        "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/${d:+$d/}" 2>/dev/null || true
done

# ── Step 3: Upload training scripts ──
echo "[3/4] Uploading training scripts..."
$SCP_CMD "$LOCAL_DIR/scripts/train_llc_teacher.py" \
    "$SERVER_USER@$SERVER_HOST:$REMOTE_BASE/b2w_nav/scripts/"

# ── Step 4: Verify ──
echo "[4/4] Verifying deployment..."
$SSH_CMD "ls -la $REMOTE_BASE/b2w_nav/networks/ $REMOTE_BASE/b2w_nav/rewards/ $REMOTE_BASE/b2w_nav/utils/"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "To train on server:"
echo "  $SSH_CMD"
echo "  cd $REMOTE_BASE"
echo "  python -m b2w_nav.scripts.train_llc_teacher --num_envs 4096"
