#!/usr/bin/env bash
# Inner script run inside the Phase 4 SLURM job.
# Runs pipeline.embed (gemini_embed 3072-d) then pipeline.merge to
# produce the final SCALING_C v3 training data.
set -euo pipefail

DATA_BASE="/data/user_data/geney/scaling_c_data"
FILTERED_DIR="${DATA_BASE}/filtered_trajectories_psf_v3_pporn_1e8_top4M"
GEMINI_DIR="${DATA_BASE}/gemini_labels_psf_v3_cadence5_grounded_3flash"
EMBED_DIR="${DATA_BASE}/embeddings_psf_v3_cadence5_grounded_predonly_gemini_emb"
FINAL_DIR="${DATA_BASE}/final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M"

if [ ! -d "${GEMINI_DIR}" ]; then
    echo "ERROR: ${GEMINI_DIR} not found — Phase 3 not complete" >&2
    exit 1
fi
if [ ! -d "${FILTERED_DIR}" ]; then
    echo "ERROR: ${FILTERED_DIR} not found — Phase 2 not complete" >&2
    exit 1
fi

n_label_files=$(ls "${GEMINI_DIR}"/*.jsonl 2>/dev/null | wc -l)
echo "Found ${n_label_files} jsonl label files in ${GEMINI_DIR}"

echo "=== Phase 4a: gemini_embed embedding (3072-d) ==="
python -m pipeline.embed \
    --gemini-dir "${GEMINI_DIR}" \
    --output-dir "${EMBED_DIR}" \
    --backend gemini_embed \
    --output-dim 3072 \
    --batch-size 16

echo
echo "=== Phase 4b: merge ==="
python -m pipeline.merge \
    --filtered-dir "${FILTERED_DIR}" \
    --gemini-dir "${GEMINI_DIR}" \
    --embed-dir "${EMBED_DIR}" \
    --output-dir "${FINAL_DIR}"

echo
echo "=== Verify ==="
python3 - <<PY
import numpy as np
from pathlib import Path
files = sorted(Path("${FINAL_DIR}").glob("trajectories_*.npz"))
print(f"final files: {len(files)}")
if files:
    d = np.load(files[0], allow_pickle=True)
    h = d["hidden_state"]
    print(f"  hidden shape={h.shape} dtype={h.dtype}")
    print(f"  range=[{h.min():.3f}, {h.max():.3f}]  mean={h.mean():.4f}")
    nz = (np.abs(h).sum(axis=1) > 0).sum()
    print(f"  non-zero hidden rows: {nz}/{len(h)} ({nz/len(h)*100:.1f}%)")
PY
