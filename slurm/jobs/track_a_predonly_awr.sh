#!/usr/bin/env bash
# Track A: AWR pretrain on predonly-embedded PSF data.
# Uses same hyperparameters as the psf_size_ablation AWR runs for apples-to-apples
# comparison with the baseline v2 AWR (gemini_emb full run).
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
DATA_DIR="${DATA_BASE}/final_trajectories_psf_v2_cadence5_predonly_gemini_emb"
ORACLE_DATA="${ORACLE_BASE}/predict_only_final_v2_cadence5_gemini_emb/trajectories_000000.npz"
SAVE_DIR="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly/awr"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: ${DATA_DIR} not found" >&2
    exit 1
fi

echo "=== Track A: AWR pretrain on predonly data ==="
echo "  Data: ${DATA_DIR}"
echo "  Save: ${SAVE_DIR}"

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${ORACLE_DATA}" \
    --val-freq 5000 \
    --hidden-mode real \
    --layer-width 512 \
    --hidden-dim 3072 \
    --lr 3e-4 \
    --awr-beta 10.0 \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --total-steps 100000 \
    --save-freq 25000 \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.0 \
    --wandb-name "awr_psf_v2_cadence5_predonly" \
    --max-dataset-gb 60

echo "=== DONE Track A AWR pretrain ==="
