#!/usr/bin/env bash
# SCALING_C Phase 4 — embed (Gemini 3072-d) + merge for the new
# top-4M PPO-RNN-derived data.
#
# Reads:
#   gemini_labels_psf_v3_cadence5_grounded_3flash/  (Phase 3 output)
#   filtered_trajectories_psf_v3_pporn_1e8_top4M/   (Phase 2 output)
# Writes:
#   embeddings_psf_v3_cadence5_grounded_predonly_gemini_emb/
#   final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M/
#
# Embed step needs GPU (Gemini API + a CUDA-friendly batch dim, or
# the embed backend supports CPU). Use small GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
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

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job "scaling_c_phase4" \
    --gpu A100_80GB \
    --mem 64G \
    --time 6:00:00 \
    "$@" \
    -- bash -c "
set -euo pipefail
echo '=== Phase 4a: gemini_embed embedding (3072-d) ==='
PYTHONPATH=. python -m pipeline.embed \
    --gemini-dir '${GEMINI_DIR}' \
    --output-dir '${EMBED_DIR}' \
    --backend gemini_embed \
    --output-dim 3072 \
    --batch-size 16

echo
echo '=== Phase 4b: merge ==='
PYTHONPATH=. python -m pipeline.merge \
    --filtered-dir '${FILTERED_DIR}' \
    --gemini-dir '${GEMINI_DIR}' \
    --embed-dir '${EMBED_DIR}' \
    --output-dir '${FINAL_DIR}'

echo
echo '=== Verify ==='
python3 -c \"
import numpy as np
from pathlib import Path
files = sorted(Path('${FINAL_DIR}').glob('trajectories_*.npz'))
print(f'final files: {len(files)}')
d = np.load(files[0], allow_pickle=True)
h = d['hidden_state']
print(f'  hidden shape={h.shape} dtype={h.dtype}')
print(f'  range=[{h.min():.3f}, {h.max():.3f}]  mean={h.mean():.4f}')
nz = (np.abs(h).sum(axis=1) > 0).sum()
print(f'  non-zero hidden rows: {nz}/{len(h)} ({nz/len(h)*100:.1f}%)')
\"
"
