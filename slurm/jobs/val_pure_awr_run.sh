#!/usr/bin/env bash
set -euo pipefail

echo "=== Validating pure_awr_oracle_w512 ==="
python -m eval.validate_awr \
    --checkpoint /data/group_data/rl/geney/checkpoints/pure_awr_oracle_w512/final.pth \
    --data-dir /data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories \
    --file-offset 126 --max-files 4 \
    --layer-width 512 --no-layernorm

echo ""
echo "=== Validating pure_awr_psf_w512 ==="
python -m eval.validate_awr \
    --checkpoint /data/group_data/rl/geney/checkpoints/pure_awr_psf_w512/final.pth \
    --data-dir /data/group_data/rl/geney/predict_state_full/final_trajectories \
    --file-offset 126 --max-files 4 \
    --layer-width 512 --no-layernorm
