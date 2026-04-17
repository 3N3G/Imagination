#!/usr/bin/env bash
# Rerun of Apr-12 Experiment 1 ("Freeze BC, 4 configs") with predict-state-only
# (PSF) labels on BOTH training and oracle/BC data — matching the eval-time
# embedding distribution.
#
# Array configs:
#   0: freeze obs_branch, BC only
#   1: freeze obs_branch, BC + AWR   (← best config on oracle labels: 16.80)
#   2: freeze obs_and_post_merge, BC only
#   3: freeze obs_and_post_merge, BC + AWR
#
# Data:
#   training:  final_trajectories_psf_qwen3gen  (125 files; files 0-116 train, 117-124 held out)
#   BC/oracle: oracle_pipeline/predict_only_final (golden, PSF labels, 4096-d qwen3gen)
#   pretrained: awr_psf_qwen3gen/final.pth (Apr-12 pure-AWR-on-PSF baseline)
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
DATA_DIR="${DATA_BASE}/final_trajectories_psf_qwen3gen"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="${ORACLE_DATA}"
PRETRAINED="/data/group_data/rl/geney/checkpoints/awr_psf_qwen3gen/final.pth"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

FREEZE_MODES=("obs_branch" "obs_branch" "obs_and_post_merge" "obs_and_post_merge")
NO_AWR=(      "yes"        "no"         "yes"                "no")
TAGS=(
    "psf_freeze_obs_bc"
    "psf_freeze_obs_bcawr"
    "psf_freeze_all_bc"
    "psf_freeze_all_bcawr"
)

FREEZE=${FREEZE_MODES[$ID]}
TAG=${TAGS[$ID]}
SAVE_DIR="${CKPT_BASE}/${TAG}"

echo "===================================================================="
echo "[$ID] ${TAG}"
echo "  Data:       ${DATA_DIR}"
echo "  Oracle:     ${ORACLE_DATA}"
echo "  Pretrained: ${PRETRAINED}"
echo "  Freeze:     ${FREEZE}"
echo "  AWR:        $([ "${NO_AWR[$ID]}" = "yes" ] && echo 'disabled (BC only)' || echo 'enabled')"
echo "===================================================================="

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
