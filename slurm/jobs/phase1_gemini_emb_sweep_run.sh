#!/usr/bin/env bash
# Phase 1: grounded gemini_emb sweep building on Exp 18 β=30 peak (15.68).
#
# Three axes:
#   β extension            (0,1): confirm 30 is near the peak or not
#   oracle-grounding sweep (2-5): vary imagination-vs-obs BC mix at β=30
#   freeze-mode ablation   (6,7): test no-freeze + only-hidden-trains
#
# All 50K steps. Data: PSF-labelled gemini_emb trajectories. Pretrained from
# awr_psf_gemini_emb/final.pth. hidden_dim=3072.
#
# Array index -> (variant):
#   0: β=50                                 freeze=obs_branch
#   1: β=100                                freeze=obs_branch
#   2: β=30  oracle_frac=0.02 weight=0.5    freeze=obs_branch
#   3: β=30  oracle_frac=0.10 weight=0.5    freeze=obs_branch
#   4: β=30  oracle_frac=0.05 weight=0.25   freeze=obs_branch
#   5: β=30  oracle_frac=0.05 weight=1.0    freeze=obs_branch
#   6: β=30  oracle_frac=0.05 weight=0.5    freeze=none
#   7: β=30  oracle_frac=0.05 weight=0.5    freeze=obs_and_post_merge
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

BETA=30.0
ORACLE_FRAC=0.05
ORACLE_W=0.5
FREEZE=obs_branch
TAG=""

case "$ID" in
  0) BETA=50.0                                   ; TAG="beta50"        ;;
  1) BETA=100.0                                  ; TAG="beta100"       ;;
  2) ORACLE_FRAC=0.02                            ; TAG="ofrac0.02"     ;;
  3) ORACLE_FRAC=0.10                            ; TAG="ofrac0.10"     ;;
  4) ORACLE_W=0.25                               ; TAG="ow0.25"        ;;
  5) ORACLE_W=1.0                                ; TAG="ow1.0"         ;;
  6) FREEZE=none                                 ; TAG="freezenone"    ;;
  7) FREEZE=obs_and_post_merge                   ; TAG="freezeobspm"   ;;
  *) echo "Unknown array index: $ID" >&2; exit 1 ;;
esac

DATA_DIR="${DATA_BASE}/final_trajectories_psf_gemini_emb"
ORACLE_DATA="${ORACLE_BASE}/predict_only_final_gemini_emb/trajectories_000000.npz"
PRETRAINED="${CKPT_BASE}/awr_psf_gemini_emb/final.pth"
HIDDEN_DIM=3072

FULL_TAG="phase1_gemini_emb_${TAG}"
SAVE_DIR="${CKPT_BASE}/${FULL_TAG}"

echo "===================================================================="
echo "[$ID] ${FULL_TAG}"
echo "  β=${BETA}  oracle_frac=${ORACLE_FRAC}  oracle_weight=${ORACLE_W}  freeze=${FREEZE}"
echo "  data=${DATA_DIR}"
echo "  pretrained=${PRETRAINED}"
echo "===================================================================="

# Note: --pretrained-checkpoint only matters when FREEZE != none. Pass it always
# for consistent init across runs (AWR head stays warm-start even if not frozen).
python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${ORACLE_DATA}" \
    --val-freq 2500 \
    --pretrained-checkpoint "${PRETRAINED}" \
    --freeze-mode "${FREEZE}" \
    --hidden-mode real \
    --layer-width 512 \
    --hidden-dim "${HIDDEN_DIM}" \
    --lr 1e-4 \
    --awr-beta ${BETA} \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --total-steps 50000 \
    --save-freq 10000 \
    --oracle-fraction ${ORACLE_FRAC} \
    --oracle-loss-weight ${ORACLE_W} \
    --wandb-name "${FULL_TAG}" \
    --max-dataset-gb 60

echo ""
echo "=== DONE ${FULL_TAG} ==="
