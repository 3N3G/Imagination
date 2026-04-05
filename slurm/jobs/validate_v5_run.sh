#!/usr/bin/env bash
# Helper: validate all v4+v5 checkpoints on re-labelled golden trajectory.
set -euo pipefail

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
GOLDEN_DIR="/data/group_data/rl/geney/oracle_pipeline/final_oracle_mode"

echo "=== Validating on re-labelled golden trajectory ==="
echo "  Golden data: ${GOLDEN_DIR}"
echo ""

for dir in ${CKPT_BASE}/bcawr_v5_* ${CKPT_BASE}/bcawr_v4_*; do
    [ -d "$dir" ] || continue
    ckpt="${dir}/final.pth"
    stats="${dir}/hidden_state_stats.npz"
    meta="${dir}/training_metadata.json"

    [ -f "$ckpt" ] || { echo "SKIP $dir (no final.pth)"; continue; }
    [ -f "$stats" ] || { echo "SKIP $dir (no hidden_state_stats.npz)"; continue; }

    # Extract architecture params from metadata
    WIDTH=512
    DROPOUT=0.0
    NOLN=false
    if [ -f "$meta" ]; then
        WIDTH=$(python3 -c "import json; print(json.load(open('$meta')).get('layer_width', 512))")
        DROPOUT=$(python3 -c "import json; print(json.load(open('$meta')).get('dropout', 0.0))")
        NOLN=$(python3 -c "import json; print('true' if json.load(open('$meta')).get('no_layernorm', False) else 'false')")
    fi

    NOLN_FLAG=""
    if [ "$NOLN" = "true" ]; then
        NOLN_FLAG="--no-layernorm"
    fi

    echo "--- $(basename $dir) (width=${WIDTH}, dropout=${DROPOUT}, noln=${NOLN}) ---"
    python -m eval.validate_awr \
        --checkpoint "$ckpt" \
        --hidden-stats "$stats" \
        --data-dir "${GOLDEN_DIR}" \
        --data-glob "trajectories_*.npz" \
        --file-offset 0 \
        --max-files 1 \
        --layer-width "$WIDTH" \
        --dropout "$DROPOUT" \
        --batch-size 2048 \
        ${NOLN_FLAG}
    echo ""
done

echo "=== All validations complete ==="
