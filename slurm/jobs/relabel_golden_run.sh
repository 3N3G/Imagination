#!/usr/bin/env bash
# Helper script run by SLURM: re-label golden trajectory (Phases 4-6).
set -euo pipefail

ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
FILTERED_DIR="${ORACLE_BASE}/filtered_trajectories"
GEMINI_DIR="${ORACLE_BASE}/gemini_labels_oracle_mode"
EMBED_DIR="${ORACLE_BASE}/embeddings_oracle_mode"
FINAL_DIR="${ORACLE_BASE}/final_oracle_mode"

echo "=== Phase 4: Gemini re-label (oracle mode, gemini-2.5-flash) ==="
python -m pipeline.gemini_label \
    --filtered-dir "$FILTERED_DIR" \
    --output-dir "$GEMINI_DIR"

echo ""
echo "=== Phase 5: Qwen3-8B embedding ==="
python -m pipeline.embed \
    --gemini-dir "$GEMINI_DIR" \
    --output-dir "$EMBED_DIR" \
    --batch-size 32

echo ""
echo "=== Phase 6: Merge ==="
python -m pipeline.merge \
    --filtered-dir "$FILTERED_DIR" \
    --gemini-dir "$GEMINI_DIR" \
    --embed-dir "$EMBED_DIR" \
    --output-dir "$FINAL_DIR"

echo ""
echo "=== Verify output ==="
python3 -c "
import numpy as np
d = np.load('${FINAL_DIR}/trajectories_000000.npz', allow_pickle=True)
print('Keys:', sorted(d.files))
for k in sorted(d.files):
    print(f'  {k}: shape={d[k].shape}, dtype={d[k].dtype}')
h = d['hidden_state']
print(f'Hidden state: min={h.min():.2f}, max={h.max():.2f}, mean={h.mean():.4f}')
print(f'Non-zero vectors: {(np.abs(h).sum(axis=1) > 0).sum()}/{len(h)}')
"
echo "Done!"
