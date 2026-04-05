#!/usr/bin/env bash
# Phase 5+6: Embed + Merge for predict-state-only labels.
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FILTERED_DIR="${DATA_BASE}/filtered_trajectories"
GEMINI_DIR="${DATA_BASE}/gemini_labels_psf"
EMBED_DIR="${DATA_BASE}/embeddings_psf"
FINAL_DIR="${DATA_BASE}/final_trajectories_psf"

echo "=== Phase 5: Qwen3-8B embedding ==="
echo "  Input:  ${GEMINI_DIR}"
echo "  Output: ${EMBED_DIR}"
echo ""

python -m pipeline.embed \
    --gemini-dir "${GEMINI_DIR}" \
    --output-dir "${EMBED_DIR}" \
    --batch-size 32

echo ""
echo "=== Phase 6: Merge ==="
echo "  Output: ${FINAL_DIR}"
echo ""

python -m pipeline.merge \
    --filtered-dir "${FILTERED_DIR}" \
    --gemini-dir "${GEMINI_DIR}" \
    --embed-dir "${EMBED_DIR}" \
    --output-dir "${FINAL_DIR}" \
    --max-files 158

echo ""
echo "=== Verify first file ==="
python3 -c "
import numpy as np
d = np.load('${FINAL_DIR}/trajectories_000000.npz', allow_pickle=True)
print('Keys:', sorted(d.files))
for k in sorted(d.files):
    print(f'  {k}: shape={d[k].shape}, dtype={d[k].dtype}')
h = d['hidden_state']
print(f'Hidden state: min={h.min():.2f}, max={h.max():.2f}, mean={h.mean():.4f}')
nz = (np.abs(h).sum(axis=1) > 0).sum()
print(f'Non-zero vectors: {nz}/{len(h)} ({nz/len(h)*100:.1f}%)')
"
echo "Done!"
