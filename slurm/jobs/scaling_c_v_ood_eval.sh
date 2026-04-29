#!/usr/bin/env bash
# OOD-steering pilot: try to elicit a never-unlocked achievement (eat_bat,
# enter_gnomish_mines, learn_fireball, etc.) using a tailored prompt.
#
# Reads VARIANT_TAG (default xxhighb) and runs the explore_ood_v1 prompt
# on that policy at n=15 (cheap pilot).

set -euo pipefail

VARIANT_TAG="${VARIANT_TAG:-xxhighb}"
NUM_EPISODES="${NUM_EPISODES:-15}"
MODE="explore_ood_v1"

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v3_pporn_1e8_grounded_${VARIANT_TAG}/freezenone"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"
EVAL_BASE="/data/user_data/geney/eval_results_temp/psf_v3_pporn_1e8_grounded_${VARIANT_TAG}_steer_score"
PROMPT="/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt"

if [ ! -f "${CKPT}" ]; then echo "ERROR: missing ${CKPT}" >&2; exit 1; fi

OUT_DIR="${EVAL_BASE}/${MODE}_${NUM_EPISODES}ep"
mkdir -p "${OUT_DIR}"

echo "=== OOD steering pilot — variant=${VARIANT_TAG}, mode=${MODE}, n=${NUM_EPISODES} ==="
echo "  CKPT: ${CKPT}"
echo "  OUT:  ${OUT_DIR}"

python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "${PROMPT}" \
    --embedding-mode "${MODE}" --num-episodes "${NUM_EPISODES}" \
    --output-dir "${OUT_DIR}" \
    --wandb-name "eval_psf_v3_pporn_1e8_grounded_${VARIANT_TAG}_${MODE}_${NUM_EPISODES}ep"

echo "=== DONE ==="
