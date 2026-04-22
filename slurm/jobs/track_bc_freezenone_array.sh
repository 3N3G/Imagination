#!/usr/bin/env bash
# Track B/C: freezenone BC+AWR fine-tune from the top-2M AWR pretrain.
# Array:
#   0: think
#   1: grounded
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}
case "${ID}" in
    0) TAG="think" ;;
    1) TAG="grounded" ;;
    *) echo "ERROR: unknown array ID ${ID}" >&2; exit 2 ;;
esac

BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
DATA_DIR="${BASE}/final_trajectories_psf_v2_cadence5_${TAG}_predonly_gemini_emb_top2M"
ORACLE_DATA="${ORACLE_BASE}/predict_only_final_v2_cadence5_predonly_gemini_emb/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}_predonly_top2M"
PRETRAINED="${CKPT_BASE}/awr/final.pth"
SAVE_DIR="${CKPT_BASE}/freezenone"

mkdir -p "${SAVE_DIR}"

if [ ! -f "${PRETRAINED}" ]; then
    echo "ERROR: ${PRETRAINED} not found; run track_bc_awr_array.sh first" >&2; exit 1
fi

echo "=== Track ${TAG}: freezenone on top-2M ==="
echo "  Data:       ${DATA_DIR}"
echo "  Pretrained: ${PRETRAINED}"
echo "  Save:       ${SAVE_DIR}"

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${ORACLE_DATA}" \
    --val-freq 2500 \
    --pretrained-checkpoint "${PRETRAINED}" \
    --freeze-mode none \
    --hidden-mode real \
    --layer-width 512 \
    --hidden-dim 3072 \
    --lr 1e-4 \
    --awr-beta 30.0 \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --total-steps 50000 \
    --save-freq 10000 \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.5 \
    --wandb-name "track_${TAG}_predonly_top2M_freezenone" \
    --max-dataset-gb 60

echo "=== DONE freezenone task ${ID} (${TAG}) ==="
