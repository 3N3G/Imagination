#!/usr/bin/env bash
# Score-max v2_thresh6 single-cell eval on C_grounded_2M (freezenone).
# Probes: raise survive thresholds 4 -> 6, add canonical maintenance ordering
# (drink fastest, food next, energy slowest) per Craftax intrinsic decay rates
# verified in craftax/craftax/game_logic.py lines 1880/1893/1907.
# Outputs to /data/user_data/geney/eval_results_temp/ as quota-workaround.

set -euo pipefail

CKPT_KIND="freezenone"
MODE="achievement_max_v2_thresh6"

TAG="grounded_predonly_top2M"
TRACK_KEY="grounded_predonly_top2M"

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
DIR_TAG="${TRACK_KEY}_steer_score"

EVAL_BASE="/data/user_data/geney/eval_results_temp/psf_v2_cadence5_${DIR_TAG}"
CKPT="${CKPT_BASE}/final.pth"
STATS_DEFAULT="${CKPT_BASE}/hidden_state_stats.npz"
STATS_FALLBACK="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone/hidden_state_stats.npz"
STATS="${STATS_DEFAULT}"
[ -f "${STATS}" ] || STATS="${STATS_FALLBACK}"

if [ ! -f "${CKPT}" ]; then echo "ERROR: missing ${CKPT}" >&2; exit 1; fi

echo "=== score_max v2_thresh6 CKPT=${CKPT_KIND} MODE=${MODE} ==="
echo "  -> output: ${EVAL_BASE}/${MODE}_30ep"
mkdir -p "${EVAL_BASE}/${MODE}_30ep"
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode "${MODE}" --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${MODE}_30ep" \
    --wandb-name "eval_${DIR_TAG}_${MODE}_30ep_userdata"

echo "=== DONE v2_thresh6 ==="
