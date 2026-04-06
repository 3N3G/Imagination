#!/usr/bin/env bash
# Evaluate top v6 models with full imagination pipeline + ablations.
# Array job: each task evaluates one (model, embedding_mode) combination.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results/v6"
NUM_EPISODES=10

# --- Config grid ---
# 3 models × 3 modes = 9 tasks
#  0-2: lr1e4_w1024 × (gemini, adversarial, die)
#  3-5: lr1e4_beta5 × (gemini, adversarial, die)
#  6-8: lr1e4_ow10  × (gemini, adversarial, die)

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

CKPT_DIR="${CKPT_BASE}/bcawr_v6_${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_${MODE}"

echo "=== Eval v6 [${ID}]: ${TAG} mode=${MODE} ==="
echo "  Checkpoint: ${CKPT_DIR}"
echo "  Output: ${OUT_DIR}"
echo "  Width: ${WIDTH}, Episodes: ${NUM_EPISODES}"
echo ""

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --layer-width "${WIDTH}" \
    --embedding-mode "${MODE}" \
    --num-episodes ${NUM_EPISODES} \
    --wandb-name "v6_${TAG}_${MODE}" \
    --no-video
