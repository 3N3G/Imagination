#!/usr/bin/env bash
# In-distribution oracle experiments.
# Terminology: "oracle" = Gemini given actual future, "golden" = human-collected trajectories.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

# Data paths
PSF_DATA="/data/group_data/rl/geney/predict_state_full/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
GOLDEN_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"
TOP_BASE="/data/group_data/rl/geney/oracle_pipeline"

case $ID in
  0)
    # Replicate old run: pure AWR on PPO data with ORACLE labels, w512
    echo "=== [0] Pure AWR, oracle labels, w512 (replicate old run) ==="
    python -m offline_rl.train_awr \
        --save-dir "${CKPT_BASE}/pure_awr_oracle_w512" \
        --data-dir "${ORACLE_DATA}" \
        --layer-width 512 \
        --lr 3e-4 \
        --max-dataset-gb 60 \
        --total-steps 100000 \
        --wandb-name "pure_awr_oracle_w512"
    ;;
  1)
    # Same but with PSF labels to see if label type matters
    echo "=== [1] Pure AWR, PSF labels, w512 ==="
    python -m offline_rl.train_awr \
        --save-dir "${CKPT_BASE}/pure_awr_psf_w512" \
        --data-dir "${PSF_DATA}" \
        --layer-width 512 \
        --lr 3e-4 \
        --max-dataset-gb 60 \
        --total-steps 100000 \
        --wandb-name "pure_awr_psf_w512"
    ;;
  2)
    # BC+AWR with top-250 PPO episodes as oracle (in-distribution, PSF labels)
    echo "=== [2] BC+AWR, top-250 PPO oracle (PSF), OW=0.5 ==="
    ORACLE_PATH="${TOP_BASE}/top250_psf/trajectories_000000.npz"
    if [ ! -f "$ORACLE_PATH" ]; then
        echo "Selecting top 250 episodes from PSF data..."
        python tools/select_top_from_final.py \
            --data-dir "${PSF_DATA}" \
            --output "${ORACLE_PATH}" \
            --top-n 250
    fi
    python -m offline_rl.train_awr_weighted_v2 \
        --save-dir "${CKPT_BASE}/bc_top250_psf" \
        --data-dir "${PSF_DATA}" \
        --oracle-data "${ORACLE_PATH}" \
        --layer-width 1024 \
        --lr 1e-4 \
        --dropout 0.2 \
        --weight-decay 1e-3 \
        --entropy-coeff 0.03 \
        --oracle-loss-weight 0.5 \
        --oracle-fraction 0.05 \
        --max-grad-norm 1.0 \
        --val-data "${VAL_DATA}" \
        --val-freq 5000 \
        --total-steps 100000 \
        --arch-v2 \
        --wandb-name "bc_top250_psf"
    ;;
  3)
    # BC+AWR with top-1000 PPO episodes (harder to memorize)
    echo "=== [3] BC+AWR, top-1000 PPO oracle (PSF), OW=0.5 ==="
    ORACLE_PATH="${TOP_BASE}/top1000_psf/trajectories_000000.npz"
    if [ ! -f "$ORACLE_PATH" ]; then
        echo "Selecting top 1000 episodes from PSF data..."
        python tools/select_top_from_final.py \
            --data-dir "${PSF_DATA}" \
            --output "${ORACLE_PATH}" \
            --top-n 1000
    fi
    python -m offline_rl.train_awr_weighted_v2 \
        --save-dir "${CKPT_BASE}/bc_top1000_psf" \
        --data-dir "${PSF_DATA}" \
        --oracle-data "${ORACLE_PATH}" \
        --layer-width 1024 \
        --lr 1e-4 \
        --dropout 0.2 \
        --weight-decay 1e-3 \
        --entropy-coeff 0.03 \
        --oracle-loss-weight 0.5 \
        --oracle-fraction 0.05 \
        --max-grad-norm 1.0 \
        --val-data "${VAL_DATA}" \
        --val-freq 5000 \
        --total-steps 100000 \
        --arch-v2 \
        --wandb-name "bc_top1000_psf"
    ;;
  4)
    # BC+AWR with golden (human) data, OW=0.01 (control — known bad)
    echo "=== [4] BC+AWR, golden (human) oracle, OW=0.01 ==="
    python -m offline_rl.train_awr_weighted_v2 \
        --save-dir "${CKPT_BASE}/bc_golden_ow001" \
        --data-dir "${PSF_DATA}" \
        --oracle-data "${GOLDEN_DATA}" \
        --layer-width 1024 \
        --lr 1e-4 \
        --dropout 0.2 \
        --weight-decay 1e-3 \
        --entropy-coeff 0.03 \
        --oracle-loss-weight 0.01 \
        --oracle-fraction 0.05 \
        --max-grad-norm 1.0 \
        --val-data "${VAL_DATA}" \
        --val-freq 5000 \
        --total-steps 100000 \
        --arch-v2 \
        --wandb-name "bc_golden_ow001"
    ;;
  5)
    # BC+AWR with top-250, high OW=1.0
    echo "=== [5] BC+AWR, top-250 PPO oracle (PSF), OW=1.0 ==="
    ORACLE_PATH="${TOP_BASE}/top250_psf/trajectories_000000.npz"
    if [ ! -f "$ORACLE_PATH" ]; then
        echo "Selecting top 250 episodes from PSF data..."
        python tools/select_top_from_final.py \
            --data-dir "${PSF_DATA}" \
            --output "${ORACLE_PATH}" \
            --top-n 250
    fi
    python -m offline_rl.train_awr_weighted_v2 \
        --save-dir "${CKPT_BASE}/bc_top250_highow" \
        --data-dir "${PSF_DATA}" \
        --oracle-data "${ORACLE_PATH}" \
        --layer-width 1024 \
        --lr 1e-4 \
        --dropout 0.2 \
        --weight-decay 1e-3 \
        --entropy-coeff 0.03 \
        --oracle-loss-weight 1.0 \
        --oracle-fraction 0.05 \
        --max-grad-norm 1.0 \
        --val-data "${VAL_DATA}" \
        --val-freq 5000 \
        --total-steps 100000 \
        --arch-v2 \
        --wandb-name "bc_top250_highow"
    ;;
esac
