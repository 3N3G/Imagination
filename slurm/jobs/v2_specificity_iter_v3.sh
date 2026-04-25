#!/usr/bin/env bash
# Iteration of two weak specificity cells with v3 prompts:
#   0: target_avoid_stone_v3 (removes stone-tier upgrade rule entirely → no
#      contradiction with avoid-stone goal; removes stone-tunnel sleep hint)
#   1: survive_long_v3      (drops the strict "base" criteria; just keeps
#      intrinsics topped up + NOOP when safe + run from enemies)

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

VARIANTS=(
    "target_avoid_stone_v3"   # 0
    "survive_long_v3"         # 1
)

MODE="${VARIANTS[$ID]}"

TAG="grounded_predonly_top2M"
TRACK_KEY="grounded_predonly_top2M"

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_specificity_iter"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi

echo "=== specificity_iter ID=${ID} TRACK=${TRACK_KEY} MODE=${MODE} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode "${MODE}" --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${MODE}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_specificity_iter_${MODE}_30ep"

echo "=== DONE ID=${ID} ==="
