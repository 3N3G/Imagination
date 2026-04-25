#!/usr/bin/env bash
# Test achievement_max_v3 (floor-1 hunting) on the awr-only checkpoint.
# Critical question: can the floor-1 INTERMEDIATE achievements (find_bow,
# open_chest, drink_potion, defeat_orc_solider, eat_snail, fire_bow) be
# pushed up on awr-only? Both freezenone and awr-only are at ~0% on
# these. If the v3 prompt can elicit them on awr-only, awr-only's
# higher baseline (17.67) + the new achievements would exceed
# freezenone+v2's 18.39.
set -euo pipefail
TAG="grounded_predonly_top2M"; TRACK_KEY="grounded_predonly_top2M"
CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/awr"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_awr_only"
CKPT="${CKPT_BASE}/final.pth"
STATS="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone/hidden_state_stats.npz"
echo "=== awr_only achievement_max_v3 ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode achievement_max_v3 --num-episodes 30 \
    --output-dir "${EVAL_BASE}/achievement_max_v3_30ep" \
    --wandb-name "eval_${TRACK_KEY}_awr_only_achievement_max_v3_30ep"
echo "=== DONE ==="
