#!/usr/bin/env bash
# Synthetic embedding arithmetic sweep.
#
# For each (track, direction) pair, run α ∈ {-2, -1, +1, +2} (4 jobs each;
# α=0 is the regular eval, already on file). 30-ep evals to reduce wall
# time and Gemini cost.
#
# Tracks × directions × alphas:
#   - A_full   × a_full_die_v2          × {-2, -1, +1, +2}      = 4 jobs
#   - C_grnd   × c_grounded_die_v2      × {-2, -1, +1, +2}      = 4 jobs
#   - C_grnd   × c_grounded_avoid_anim  × {-2, -1, +1, +2}      = 4 jobs
# Total = 12 array tasks (0..11).

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

ALPHAS=("-2.0" "-1.0" "1.0" "2.0")
N_ALPHA=${#ALPHAS[@]}

# 3 cells × 4 alphas = 12
CELL=$(( ID / N_ALPHA ))
A_IDX=$(( ID % N_ALPHA ))
ALPHA="${ALPHAS[$A_IDX]}"
ALPHA_TAG="${ALPHA//./_}"   # -2.0 -> -2_0

DIR_ROOT="/home/geney/Imagination/probe_results/embed_directions"

case "${CELL}" in
    0)
        TAG="predonly"
        TRACK_KEY="predonly_full"
        DIR_NAME="a_full_die_v2"
        ;;
    1)
        TAG="grounded_predonly_top2M"
        TRACK_KEY="grounded_predonly_top2M"
        DIR_NAME="c_grounded_die_v2"
        ;;
    2)
        TAG="grounded_predonly_top2M"
        TRACK_KEY="grounded_predonly_top2M"
        DIR_NAME="c_grounded_avoid_animals_v2"
        ;;
    *) echo "ERROR unknown cell ${CELL}" >&2; exit 2 ;;
esac

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"
DIR_PATH="${DIR_ROOT}/${DIR_NAME}.npy"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi
if [ ! -f "${DIR_PATH}" ]; then echo "ERROR: ${DIR_PATH}" >&2; exit 1; fi

EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_embed_arith"
RUN_TAG="${DIR_NAME}_alpha${ALPHA_TAG}"

echo "=== embed_arith ID=${ID}  TRACK=${TRACK_KEY}  DIR=${DIR_NAME}  ALPHA=${ALPHA} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode embed_arith \
    --embed-arith-direction "${DIR_PATH}" \
    --embed-arith-alpha "${ALPHA}" \
    --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${RUN_TAG}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_${RUN_TAG}_30ep"

echo "=== DONE ID=${ID} ==="
