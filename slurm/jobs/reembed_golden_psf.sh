#!/usr/bin/env bash
# Re-embed the existing predict-state-only (PSF) golden Gemini labels with
# qwen3emb and/or gemini_emb backends, so every freeze BC+AWR variant has
# a matching golden dataset.
#
# Inputs (unchanged):
#   oracle_pipeline/filtered_trajectories/trajectories_000000.npz
#   oracle_pipeline/predict_only_gemini_labels/trajectories_000000.jsonl  (PSF texts)
#
# Outputs (new):
#   oracle_pipeline/predict_only_embeddings_<backend>/trajectories_000000_embeddings.npz
#   oracle_pipeline/predict_only_final_<backend>/trajectories_000000.npz
#
# Array index:
#   0 -> qwen3emb  (Qwen3-Embedding-8B, last-token pool, 4096-d)
#   1 -> gemini_emb (gemini-embedding-001, 3072-d, API)
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID:-0}

ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
FILTERED_DIR="${ORACLE_BASE}/filtered_trajectories"
GEMINI_DIR="${ORACLE_BASE}/predict_only_gemini_labels"

case "$ID" in
  0)
    BACKEND="qwen3_embed"
    TAG="qwen3emb"
    DIM=4096
    EXTRA_ARGS=()
    ;;
  1)
    BACKEND="gemini_embed"
    TAG="gemini_emb"
    DIM=3072
    if [[ -z "${GEMINI_API_KEY:-}" ]]; then
        echo "ERROR: GEMINI_API_KEY must be set for gemini_embed backend" >&2
        exit 1
    fi
    EXTRA_ARGS=(--output-dim "${DIM}")
    ;;
  *)
    echo "Unknown array index: $ID" >&2
    exit 1
    ;;
esac

EMBED_DIR="${ORACLE_BASE}/predict_only_embeddings_${TAG}"
FINAL_DIR="${ORACLE_BASE}/predict_only_final_${TAG}"

echo "===================================================================="
echo "Re-embed golden PSF — backend=${BACKEND}  dim=${DIM}  tag=${TAG}"
echo "===================================================================="
echo "  Texts:  ${GEMINI_DIR}"
echo "  Embed:  ${EMBED_DIR}"
echo "  Final:  ${FINAL_DIR}"
echo ""

echo "=== Phase 5: embed ==="
python -m pipeline.embed \
    --gemini-dir "${GEMINI_DIR}" \
    --output-dir "${EMBED_DIR}" \
    --backend "${BACKEND}" \
    --batch-size 16 \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Phase 6: merge ==="
python -m pipeline.merge \
    --filtered-dir "${FILTERED_DIR}" \
    --gemini-dir "${GEMINI_DIR}" \
    --embed-dir "${EMBED_DIR}" \
    --output-dir "${FINAL_DIR}"

echo ""
echo "=== Verify output ==="
python3 -c "
import numpy as np
d = np.load('${FINAL_DIR}/trajectories_000000.npz', allow_pickle=True)
print('Keys:', sorted(d.files))
h = d['hidden_state']
print(f'Hidden: shape={h.shape} dtype={h.dtype}')
print(f'  min={h.min():.3f}  max={h.max():.3f}  mean={h.mean():.4f}')
nz = (np.abs(h).sum(axis=1) > 0).sum()
print(f'  non-zero rows: {nz}/{len(h)} ({nz/len(h)*100:.1f}%)')
assert h.shape[1] == ${DIM}, f'Expected dim ${DIM}, got {h.shape[1]}'
print('OK')
"
echo "Done!"
