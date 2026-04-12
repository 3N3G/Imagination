#!/usr/bin/env bash
# Causal test: does the freeze_obs_bcawr policy's return depend on the CONTENT
# of Gemini's imagination narrative? Run 50-ep online eval with harmful prompts
# (die-seeking, adversarial-worst-play) and compare to baseline 16.80 ± 3.49.
# Array 0=die, 1=adversarial.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

MODES=("die" "adversarial")
MODE=${MODES[$ID]}

CKPT_DIR="/data/group_data/rl/geney/checkpoints/freeze_obs_bcawr"
EVAL_DIR="/data/group_data/rl/geney/eval_results/freeze_obs_bcawr_${MODE}"

echo "=== Eval freeze_obs_bcawr with --embedding-mode ${MODE} ==="
echo "  Checkpoint: ${CKPT_DIR}/final.pth"
echo "  Baseline (gemini mode): 16.80 ± 3.49"
echo ""

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embedding-mode "${MODE}" \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name "eval_freeze_obs_bcawr_${MODE}"

echo ""
echo "======================================================================"
echo "ALL DONE — freeze_obs_bcawr ${MODE}"
echo "======================================================================"
