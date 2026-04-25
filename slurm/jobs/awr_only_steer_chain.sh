#!/usr/bin/env bash
# AWR-only steerability chain: does the policy steer when the BC + golden
# oracle finetune is dropped? Same C_grounded_2M data, but using the
# AWR-only intermediate checkpoint (psf_v2_cadence5_grounded_predonly_top2M/awr/
# final.pth, 100k steps, β=10, oracle_loss_weight=0.0).
#
# The freezenone checkpoint (the canonical "C") takes this same AWR checkpoint
# and adds 50k steps of BC+AWR finetune with oracle_loss_weight=0.5,
# oracle_fraction=0.05. This experiment isolates the contribution of that
# finetune phase to steerability.
#
# Cells (one per array index):
#   0: gemini (regular concise, baseline)
#   1: target_descend_v2  (top score-max steerer on freezenone, ret 17.23)
#   2: direction_left_v2  (cleanest direction steer, move-share +6 z on freezenone)
#   3: die_fast_v2        (strongest length steer, z=-3.0 on freezenone)
#   4: avoid_animals_v2   (clearest avoidance, cow_eat -1.5 z on freezenone)
#   5: target_collect_stone_v2 (first net-positive ret on freezenone)
#   6: v2_long_tail       (patch-by-prompt, +2.14 ret on freezenone)

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

VARIANTS=(
    "gemini"                       # 0
    "target_descend_v2"            # 1
    "direction_left_v2"            # 2
    "die_fast_v2"                  # 3
    "avoid_animals_v2"             # 4
    "target_collect_stone_v2"      # 5
    "v2_long_tail"                 # 6
)

MODE="${VARIANTS[$ID]}"

TAG="grounded_predonly_top2M"
TRACK_KEY="grounded_predonly_top2M"

# CRITICAL difference: load /awr/final.pth, not /freezenone/final.pth
CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/awr"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_awr_only"

CKPT="${CKPT_BASE}/final.pth"
# AWR-only checkpoint also has its own hidden_state_stats.npz from the AWR run
STATS="${CKPT_BASE}/hidden_state_stats.npz"
if [ ! -f "${STATS}" ]; then
    # Fall back to freezenone's stats (same data so should be equivalent)
    STATS="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone/hidden_state_stats.npz"
fi

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi
if [ ! -f "${STATS}" ]; then echo "ERROR: ${STATS}" >&2; exit 1; fi

echo "=== awr_only_steer_chain ID=${ID} TRACK=${TRACK_KEY} MODE=${MODE} ==="
echo "  checkpoint: ${CKPT}"
echo "  hidden-stats: ${STATS}"
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode "${MODE}" --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${MODE}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_awr_only_${MODE}_30ep"

echo "=== DONE ID=${ID} MODE=${MODE} ==="
