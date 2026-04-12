#!/usr/bin/env bash
# Run validate_awr on all 6 freeze/V2 checkpoints: regular held-out + golden
set -euo pipefail

REG_DATA="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
GOLD_DATA="/data/group_data/rl/geney/oracle_pipeline/final_trajectories"
CKPT_ROOT="/data/group_data/rl/geney/checkpoints"

for tag in freeze_obs_bc freeze_obs_bcawr freeze_all_bc freeze_all_bcawr v2_awr_aug v2_awr_bc_aug; do
    CKPT_DIR="${CKPT_ROOT}/${tag}"
    if [ ! -f "${CKPT_DIR}/final.pth" ]; then
        echo "=== $tag: MISSING ==="; continue
    fi
    echo "======================================================================"
    echo "=== $tag — Held-out training (offset=126 max-files=8) ==="
    echo "======================================================================"
    python -m eval.validate_awr \
        --checkpoint "${CKPT_DIR}/final.pth" \
        --data-dir "${REG_DATA}" \
        --file-offset 126 --max-files 8 \
        --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
        --dropout 0.0 || echo "validation failed for $tag held-out"
    echo ""
    echo "=== $tag — Oracle (golden) (offset=0 max-files=1) ==="
    python -m eval.validate_awr \
        --checkpoint "${CKPT_DIR}/final.pth" \
        --data-dir "${GOLD_DATA}" \
        --file-offset 0 --max-files 1 \
        --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
        --dropout 0.0 || echo "validation failed for $tag golden"
done
echo ""
echo "ALL DONE"
