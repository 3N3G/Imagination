#!/usr/bin/env bash
# BC+AWR v3 ablation — isolate effect of each fix.
# All configs use moderate_combo hparams (drop=0.2, wd=1e-3, ent=0.03, ow=0.5, of=0.05)
# to control for hparam variation. Each config disables exactly one fix.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

# Ablation design (all use same hparams, only fixes differ):
#
#  0: all_fixes        — grad_norm=1.0, prescan=yes, train-only entropy
#                        (shard wraparound fix always on — it's clearly a bug)
#  1: no_clip          — grad_norm=0 (disabled), prescan=yes, train-only entropy
#  2: no_prescan       — grad_norm=1.0, prescan=no (old incremental stats), train-only entropy
#  3: both_ent         — grad_norm=1.0, prescan=yes, both-stream entropy
#  4: clip_5           — grad_norm=5.0, prescan=yes, train-only entropy
#  5: old_behavior     — grad_norm=0, prescan=no, train-only entropy (closest to pre-fix)

TAGS=(       "all_fixes" "no_clip" "no_prescan" "both_ent" "clip_5" "old_behavior")
GNORMS=(     1.0          0         1.0           1.0        5.0      0)
PRESCANS=(   ""           ""        "--no-prescan-stats" ""  ""       "--no-prescan-stats")
BOTH_ENTS=(  ""           ""        ""            "--entropy-both-streams" "" "")

TAG=${TAGS[$ID]}
GN=${GNORMS[$ID]}
PRESCAN=${PRESCANS[$ID]}
BOTH_ENT=${BOTH_ENTS[$ID]}

SAVE_DIR="${CKPT_BASE}/v3_${TAG}"

echo "=== v3 ablation [${ID}]: ${TAG} ==="
echo "  grad_norm=${GN}, prescan=${PRESCAN:-yes}, both_ent=${BOTH_ENT:-no}"
echo ""

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --layer-width 1024 \
    --lr 1e-4 \
    --awr-beta 10 \
    --dropout 0.2 \
    --weight-decay 1e-3 \
    --entropy-coeff 0.03 \
    --oracle-loss-weight 0.5 \
    --oracle-fraction 0.05 \
    --max-grad-norm "${GN}" \
    ${PRESCAN} \
    ${BOTH_ENT} \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --total-steps 100000 \
    --arch-v2 \
    --wandb-name "v3_${TAG}"
