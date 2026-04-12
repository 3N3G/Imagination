#!/usr/bin/env bash
# From-scratch BC+AWR where BC's gradient is blocked from flowing through the
# obs pathway (AWR still updates obs normally). Cleanly separates the
# "BC-toxicity lives in obs-pathway gradient" hypothesis from the
# "freeze_obs_bcawr depends on AWR pretraining" confound.
#
# Compare to:
#   freeze_obs_bcawr (pretrained AWR + frozen obs + BC+AWR fine-tune): 16.80 ± 3.49
#   awr_bc_aug_debug (unrestricted BC+AWR from scratch):                7.46 ± 4.91
#   awr_aug_debug    (pure AWR from scratch):                          16.30 ± 3.61
set -euo pipefail

DATA_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/final_trajectories/trajectories_000000.npz"
VAL_DATA="${ORACLE_DATA}"
CKPT_DIR="/data/group_data/rl/geney/checkpoints/bc_obs_stopgrad"
EVAL_DIR="/data/group_data/rl/geney/eval_results/bc_obs_stopgrad"

echo "======================================================================"
echo "PHASE 1: Train bc_obs_stopgrad (100K from-scratch, BC obs-grad blocked)"
echo "======================================================================"
python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${CKPT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --hidden-mode real \
    --layer-width 512 \
    --lr 3e-4 \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.5 \
    --bc-obs-stopgrad \
    --total-steps 100000 \
    --save-freq 25000 \
    --wandb-name bc_obs_stopgrad \
    --max-dataset-gb 30

echo ""
echo "======================================================================"
echo "PHASE 2: Held-out validation (last 8 files)"
echo "======================================================================"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir "${DATA_DIR}" \
    --file-offset 126 --max-files 8 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --dropout 0.0

echo ""
echo "======================================================================"
echo "PHASE 3: Golden validation"
echo "======================================================================"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir /data/group_data/rl/geney/oracle_pipeline/final_trajectories \
    --file-offset 0 --max-files 1 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --dropout 0.0

echo ""
echo "======================================================================"
echo "PHASE 4: 50-ep online eval"
echo "======================================================================"
python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name eval_bc_obs_stopgrad

echo ""
echo "======================================================================"
echo "ALL DONE — bc_obs_stopgrad"
echo "======================================================================"
