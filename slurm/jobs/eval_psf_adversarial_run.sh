#!/usr/bin/env bash
# Content-sensitivity probe: does the PSF freeze_obs_bcawr return depend on the
# CONTENT of Gemini's imagination narrative? Re-run 50-ep online eval with
# harmful prompts (die, adversarial) for each encoder and compare to the
# Apr-16 baseline returns.
#
# Array index -> (encoder × mode):
#   0: qwen3gen   × die
#   1: qwen3gen   × adversarial
#   2: qwen3emb   × die
#   3: qwen3emb   × adversarial
#   4: gemini_emb × die
#   5: gemini_emb × adversarial
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

ENCODERS=(qwen3gen qwen3gen qwen3emb qwen3emb gemini_emb gemini_emb)
MODES=(die adversarial die adversarial die adversarial)
BACKENDS=(qwen3_gen qwen3_gen qwen3_embed qwen3_embed gemini_embed gemini_embed)
HIDDEN_DIMS=(4096 4096 4096 4096 3072 3072)

ENCODER=${ENCODERS[$ID]}
MODE=${MODES[$ID]}
BACKEND=${BACKENDS[$ID]}
HIDDEN_DIM=${HIDDEN_DIMS[$ID]}

if [ "${ENCODER}" = "qwen3gen" ]; then
    TAG="psf_freeze_obs_bcawr"
else
    TAG="psf_freeze_obs_bcawr_${ENCODER}"
fi

CKPT_DIR="/data/group_data/rl/geney/checkpoints/${TAG}"
EVAL_DIR="/data/group_data/rl/geney/eval_results/${TAG}_${MODE}"

echo "===================================================================="
echo "[$ID] ${TAG} × ${MODE}  backend=${BACKEND}  dim=${HIDDEN_DIM}"
echo "  Baseline (gemini mode, 50 ep):"
echo "    qwen3gen=17.58 ± 3.56,  qwen3emb=16.60 ± 4.63,  gemini_emb=14.96 ± 5.11"
echo "===================================================================="

EXTRA_ARGS=()
if [ "${BACKEND}" != "qwen3_gen" ]; then
    EXTRA_ARGS+=(--embed-backend "${BACKEND}" --hidden-dim "${HIDDEN_DIM}")
fi

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embedding-mode "${MODE}" \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name "eval_${TAG}_${MODE}" \
    "${EXTRA_ARGS[@]}"
