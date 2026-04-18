#!/usr/bin/env bash
# v2 embed + merge for PSF shards (gemini_emb backend, 3072-d).
# Reads gemini_labels_psf_v2_3flash/, writes:
#   embeddings_psf_v2_gemini_emb/
#   final_trajectories_psf_v2_gemini_emb/
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FILTERED_DIR="${DATA_BASE}/filtered_trajectories"
GEMINI_DIR="${DATA_BASE}/gemini_labels_psf_v2_3flash"
EMBED_DIR="${DATA_BASE}/embeddings_psf_v2_gemini_emb"
FINAL_DIR="${DATA_BASE}/final_trajectories_psf_v2_gemini_emb"

if [ ! -d "${GEMINI_DIR}" ]; then
    echo "ERROR: ${GEMINI_DIR} not found — run v2_relabel_psf_3flash_run.sh first." >&2
    exit 1
fi

echo "=== Phase 5: gemini_embed embedding (3072-d) ==="
python -m pipeline.embed \
    --gemini-dir "${GEMINI_DIR}" \
    --output-dir "${EMBED_DIR}" \
    --backend gemini_embed \
    --output-dim 3072 \
    --batch-size 16

echo ""
echo "=== Phase 6: merge ==="
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
h = d['hidden_state']
print(f'hidden: shape={h.shape} dtype={h.dtype}')
print(f'  min={h.min():.3f} max={h.max():.3f} mean={h.mean():.4f}')
nz = (np.abs(h).sum(axis=1) > 0).sum()
print(f'  non-zero rows: {nz}/{len(h)} ({nz/len(h)*100:.1f}%)')
assert h.shape[1] == 3072
print('OK')
"
