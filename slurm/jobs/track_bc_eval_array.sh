#!/usr/bin/env bash
# Track B/C: 50-ep online eval with real (natural) Gemini forecasts at inference.
# For Track C (grounded): inference uses the CONCISE prompt (deploy-realistic).
# For Track B (thinking): inference uses the THINKING prompt (same as training).
# Both use --extract-prediction-only to match the predonly training embedding.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}
case "${ID}" in
    0)
        TAG="think"
        TEMPLATE="/home/geney/Imagination/configs/training/templates/predict_only_thinking_prompt.txt"
        THINK_FLAG="--gemini-thinking-budget 512"
        ;;
    1)
        TAG="grounded"
        # Grounded eval must use natural forecasts (no future visible) since deploy
        # can't peek at oracle. Fall back to the CONCISE prompt at eval time.
        TEMPLATE="/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt"
        THINK_FLAG=""
        ;;
    *) echo "ERROR: unknown array ID ${ID}" >&2; exit 2 ;;
esac

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}_predonly_top2M/freezenone"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"
OUT_DIR="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TAG}_predonly_top2M/freezenone_50ep"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: ${CKPT} not found" >&2; exit 1
fi

echo "=== Track ${TAG}: predonly-top2M freezenone 50-ep eval ==="
echo "  Checkpoint: ${CKPT}"
echo "  Template:   ${TEMPLATE}"

python -m eval.eval_online \
    --checkpoint "${CKPT}" \
    --hidden-stats "${STATS}" \
    --embed-backend gemini_embed \
    --hidden-dim 3072 \
    --extract-prediction-only \
    --num-episodes 50 \
    --output-dir "${OUT_DIR}" \
    --wandb-name "eval_track_${TAG}_predonly_top2M_freezenone_50ep"

echo "=== DONE eval task ${ID} (${TAG}) ==="
