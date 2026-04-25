#!/usr/bin/env bash
# Score-maximizing prompt iteration v1 — achievement_max_v1.
# Balanced enumeration prompt nudging the 6 "headroom" achievements:
#   wake_up, collect_iron, collect_sapling, place_plant, eat_plant,
#   enter_dungeon — while keeping the baseline's existing routine.

set -euo pipefail

MODE="${MODE:-achievement_max_v1}"

TAG="grounded_predonly_top2M"
TRACK_KEY="grounded_predonly_top2M"

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_steer_score"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi

echo "=== steer_score MODE=${MODE} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode "${MODE}" --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${MODE}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_steer_score_${MODE}_30ep"

echo "=== DONE MODE=${MODE} ==="
