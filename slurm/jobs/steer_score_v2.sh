#!/usr/bin/env bash
# Score-max v2: target_descend_v2 base (proven 17.23 vs baseline 14.66)
# + small "one-shot opportunistic milestone" sub-section (sapling, place_plant,
# eat_plant, place_torch). Goal: keep the descent cascade and add the
# headroom milestones that v1 captured (place_plant +17pp, eat_plant +7pp).

set -euo pipefail

MODE="${MODE:-achievement_max_v2}"

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
