#!/usr/bin/env bash
set -euo pipefail
ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results/indist"
NUM_EPISODES=10

TAGS=(   "pure_awr_oracle_w512" "pure_awr_psf_w512" "bc_top250_psf" "bc_top1000_psf" "bc_golden_ow001" "bc_top250_highow")
WIDTHS=( 512                     512                  1024            1024             1024               1024)
# pure AWR uses ActorCriticAug (--no-layernorm), BC+AWR uses V2
ARCHS=(  "--no-layernorm"        "--no-layernorm"     "--arch-v2"     "--arch-v2"      "--arch-v2"        "--arch-v2")

TAG=${TAGS[$ID]}
W=${WIDTHS[$ID]}
ARCH=${ARCHS[$ID]}
CKPT_DIR="${CKPT_BASE}/${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_gemini"

echo "=== Eval indist [${ID}]: ${TAG} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --layer-width ${W} \
    --embedding-mode gemini \
    --num-episodes ${NUM_EPISODES} \
    ${ARCH} \
    --wandb-name "${TAG}_gemini" \
    --no-video
