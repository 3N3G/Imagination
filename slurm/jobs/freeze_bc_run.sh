#!/usr/bin/env bash
# Freeze experiment: load AWR-pretrained ActorCriticAugLN, freeze obs/post-merge layers, fine-tune with BC.
# Array job with 4 configs:
#   0: freeze obs_branch, BC only
#   1: freeze obs_branch, BC + AWR
#   2: freeze obs_and_post_merge, BC only
#   3: freeze obs_and_post_merge, BC + AWR
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

# Data: oracle-mode Gemini labels (same as awr_aug_debug pretraining)
DATA_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/final_trajectories/trajectories_000000.npz"
VAL_DATA="${ORACLE_DATA}"
PRETRAINED="/data/group_data/rl/geney/checkpoints/awr_aug_debug/final.pth"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

# Config arrays
FREEZE_MODES=("obs_branch" "obs_branch" "obs_and_post_merge" "obs_and_post_merge")
NO_AWR=(      "yes"        "no"         "yes"                "no")
TAGS=(        "freeze_obs_bc" "freeze_obs_bcawr" "freeze_all_bc" "freeze_all_bcawr")

FREEZE=${FREEZE_MODES[$ID]}
TAG=${TAGS[$ID]}
SAVE_DIR="${CKPT_BASE}/${TAG}"

echo "=== Freeze Experiment [${ID}]: ${TAG} ==="
echo "  Freeze: ${FREEZE}"
echo "  AWR: $([ "${NO_AWR[$ID]}" = "yes" ] && echo "disabled (BC only)" || echo "enabled")"
echo "  Pretrained: ${PRETRAINED}"
echo ""

# Build command
CMD=(python -m offline_rl.train_awr_weighted_v2
    --save-dir "${SAVE_DIR}"
    --data-dir "${DATA_DIR}"
    --oracle-data "${ORACLE_DATA}"
    --val-data "${VAL_DATA}"
    --val-freq 2500
    --pretrained-checkpoint "${PRETRAINED}"
    --freeze-mode "${FREEZE}"
    --hidden-mode real
    --layer-width 512
    --lr 1e-4
    --entropy-coeff 0.01
    --max-grad-norm 1.0
    --total-steps 50000
    --save-freq 10000
    --wandb-name "${TAG}"
    --max-dataset-gb 30
)

if [ "${NO_AWR[$ID]}" = "yes" ]; then
    CMD+=(--no-awr --oracle-loss-weight 1.0)
else
    CMD+=(--oracle-fraction 0.05 --oracle-loss-weight 0.5)
fi

"${CMD[@]}"
