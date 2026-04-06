#!/usr/bin/env bash
# Validate all v6 checkpoints on held-out test trajectory (predict-only).
set -euo pipefail

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
TEST_DIR="/data/group_data/rl/geney/oracle_pipeline/test_final"

echo "=== Validating v6 models on held-out test trajectory ==="
echo "  Test data: ${TEST_DIR}"
echo ""

for dir in ${CKPT_BASE}/bcawr_v6_*; do
    [ -d "$dir" ] || continue
    ckpt="${dir}/final.pth"
    stats="${dir}/hidden_state_stats.npz"
    meta="${dir}/training_metadata.json"

    [ -f "$ckpt" ] || { echo "SKIP $dir (no final.pth)"; continue; }
    [ -f "$stats" ] || { echo "SKIP $dir (no hidden_state_stats.npz)"; continue; }

    WIDTH=512
    DROPOUT=0.0
    if [ -f "$meta" ]; then
        WIDTH=$(python3 -c "import json; print(json.load(open('$meta')).get('layer_width', 512))")
        DROPOUT=$(python3 -c "import json; print(json.load(open('$meta')).get('dropout', 0.0))")
    fi

    echo "--- $(basename $dir) (width=${WIDTH}, dropout=${DROPOUT}) ---"
    python -m eval.validate_awr \
        --checkpoint "$ckpt" \
        --hidden-stats "$stats" \
        --data-dir "${TEST_DIR}" \
        --data-glob "trajectories_*.npz" \
        --file-offset 0 \
        --max-files 1 \
        --layer-width "$WIDTH" \
        --dropout "$DROPOUT" \
        --batch-size 2048
    echo ""
done

echo "=== All v6 validations complete ==="
