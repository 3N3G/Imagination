#!/usr/bin/env bash
# Validate v2_awr_aug + v2_awr_bc_aug — needs --arch-v2 flag
set -euo pipefail

REG_DATA="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
GOLD_DATA="/data/group_data/rl/geney/oracle_pipeline/final_trajectories"
CKPT_ROOT="/data/group_data/rl/geney/checkpoints"

for tag in v2_awr_aug v2_awr_bc_aug; do
    CKPT_DIR="${CKPT_ROOT}/${tag}"
    echo "======================================================================"
    echo "=== $tag — Held-out training (offset=126 max-files=8) ==="
    echo "======================================================================"
    python -m eval.validate_awr --arch-v2 \
        --checkpoint "${CKPT_DIR}/final.pth" \
        --data-dir "${REG_DATA}" \
        --file-offset 126 --max-files 8 \
        --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
        --dropout 0.0
    echo ""
    echo "=== $tag — Oracle (golden) (offset=0 max-files=1) ==="
    python -m eval.validate_awr --arch-v2 \
        --checkpoint "${CKPT_DIR}/final.pth" \
        --data-dir "${GOLD_DATA}" \
        --file-offset 0 --max-files 1 \
        --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
        --dropout 0.0
done
echo ""
echo "ALL DONE"
