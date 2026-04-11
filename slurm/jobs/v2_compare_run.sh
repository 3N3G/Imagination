#!/usr/bin/env bash
# Architecture comparison: re-run key experiments with ActorCriticAugV2.
# Array job:
#   0: AWR only (no BC) — V2 baseline
#   1: AWR + BC — V2 with oracle BC (matching Apr 8 awr_bc_aug_debug setup)
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

# Data: oracle-mode Gemini labels (matching awr_aug_debug / Apr 8 experiments)
DATA_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/final_trajectories/trajectories_000000.npz"
VAL_DATA="${ORACLE_DATA}"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

TAGS=("v2_awr_aug" "v2_awr_bc_aug")

TAG=${TAGS[$ID]}
SAVE_DIR="${CKPT_BASE}/${TAG}"

echo "=== V2 Architecture Comparison [${ID}]: ${TAG} ==="

CMD=(python -m offline_rl.train_awr_weighted_v2
    --save-dir "${SAVE_DIR}"
    --data-dir "${DATA_DIR}"
    --oracle-data "${ORACLE_DATA}"
    --val-data "${VAL_DATA}"
    --val-freq 5000
    --hidden-mode real
    --layer-width 512
    --lr 3e-4
    --awr-beta 10
    --max-grad-norm 1.0
    --total-steps 100000
    --save-freq 25000
    --arch-v2
    --wandb-name "${TAG}"
)

if [ "$ID" -eq 0 ]; then
    # Pure AWR, no BC
    CMD+=(--oracle-fraction 0.0 --oracle-loss-weight 0.0)
elif [ "$ID" -eq 1 ]; then
    # AWR + BC (matching Apr 8 diagnostic setup)
    CMD+=(--oracle-fraction 0.05 --oracle-loss-weight 0.5
          --dropout 0.2 --weight-decay 1e-3 --entropy-coeff 0.03)
fi

"${CMD[@]}"
