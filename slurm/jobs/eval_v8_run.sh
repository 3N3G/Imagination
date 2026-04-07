#!/usr/bin/env bash
# Evaluate top v7 models — same grid as v6 eval but with --arch-v2.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results/v8"
NUM_EPISODES=10

# 3 models × 3 modes = 9 tasks
TAGS=(  "lr1e4_w1024" "lr1e4_w1024" "lr1e4_w1024"
        "lr1e4_beta5" "lr1e4_beta5" "lr1e4_beta5"
        "lr1e4_ow10"  "lr1e4_ow10"  "lr1e4_ow10")
MODES=( "gemini"      "adversarial" "die"
        "gemini"      "adversarial" "die"
        "gemini"      "adversarial" "die")
WIDTHS=("1024" "1024" "1024"
        "512"  "512"  "512"
        "512"  "512"  "512")

TAG=${TAGS[$ID]}
MODE=${MODES[$ID]}
WIDTH=${WIDTHS[$ID]}

CKPT_DIR="${CKPT_BASE}/bcawr_v8_${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_${MODE}"

echo "=== Eval v8 [${ID}]: ${TAG} mode=${MODE} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --layer-width "${WIDTH}" \
    --embedding-mode "${MODE}" \
    --num-episodes ${NUM_EPISODES} \
    --arch-v2 \
    --wandb-name "v8_${TAG}_${MODE}" \
    --no-video
