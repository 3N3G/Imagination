#!/usr/bin/env bash
# Track A predonly freezenone semantic test suite.
# Single array dispatcher over 6 tasks:
#   0: die content probe            (50-ep live, --embedding-mode die)
#   1: adversarial content probe    (50-ep live, --embedding-mode adversarial)
#   2: aug_zero (zero hidden)       (50-ep live, no embedder)
#   3: direction CF step-0          (50 episodes, 1 Gemini pair each)
#   4: direction CF multistep       (30 episodes x 4 intervention steps)
#   5: HP/Food perturbation         (10 episodes, probe every Gemini call)
#
# All tasks pass --extract-prediction-only to the relevant eval (except aug_zero).
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_semantic"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"
HIDDEN_DIM=3072

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint ${CKPT} not found" >&2
    exit 1
fi

case "${ID}" in
    0)
        echo "=== [${ID}] die content probe (50 ep) ==="
        python -m eval.eval_online \
            --checkpoint "${CKPT}" \
            --hidden-stats "${STATS}" \
            --embed-backend gemini_embed \
            --hidden-dim "${HIDDEN_DIM}" \
            --extract-prediction-only \
            --embedding-mode die \
            --num-episodes 50 \
            --output-dir "${EVAL_BASE}/die_50ep" \
            --wandb-name "eval_predonly_freezenone_die_50ep"
        ;;
    1)
        echo "=== [${ID}] adversarial content probe (50 ep) ==="
        python -m eval.eval_online \
            --checkpoint "${CKPT}" \
            --hidden-stats "${STATS}" \
            --embed-backend gemini_embed \
            --hidden-dim "${HIDDEN_DIM}" \
            --extract-prediction-only \
            --embedding-mode adversarial \
            --num-episodes 50 \
            --output-dir "${EVAL_BASE}/adversarial_50ep" \
            --wandb-name "eval_predonly_freezenone_adversarial_50ep"
        ;;
    2)
        echo "=== [${ID}] constant-embedding (50 ep) ==="
        # Zero-content baseline via eval_online --embedding-mode constant
        # (matches trained LayerNorm arch + 3072-dim hidden).
        python -m eval.eval_online \
            --checkpoint "${CKPT}" \
            --hidden-stats "${STATS}" \
            --embed-backend gemini_embed \
            --hidden-dim "${HIDDEN_DIM}" \
            --embedding-mode constant \
            --num-episodes 50 \
            --output-dir "${EVAL_BASE}/constant_50ep" \
            --wandb-name "eval_predonly_freezenone_constant_50ep"
        ;;
    3)
        echo "=== [${ID}] direction CF step-0 (50 ep) ==="
        python -m eval.eval_direction_counterfactual \
            --checkpoint "${CKPT}" \
            --hidden-stats "${STATS}" \
            --embed-backend gemini_embed \
            --hidden-dim "${HIDDEN_DIM}" \
            --extract-prediction-only \
            --num-episodes 50 \
            --output-dir "${EVAL_BASE}/direction_cf_step0"
        ;;
    4)
        echo "=== [${ID}] direction CF multistep (30 ep, 0/75/150/300) ==="
        python -m eval.eval_direction_counterfactual_multistep \
            --checkpoint "${CKPT}" \
            --hidden-stats "${STATS}" \
            --embed-backend gemini_embed \
            --hidden-dim "${HIDDEN_DIM}" \
            --extract-prediction-only \
            --intervention-steps 0,75,150,300 \
            --num-episodes 30 \
            --output-dir "${EVAL_BASE}/direction_cf_multistep"
        ;;
    5)
        echo "=== [${ID}] HP/Food perturbation (10 ep) ==="
        python -m eval.eval_hp_perturbation \
            --checkpoint "${CKPT}" \
            --hidden-stats "${STATS}" \
            --layer-width 512 \
            --embed-backend gemini_embed \
            --hidden-dim "${HIDDEN_DIM}" \
            --extract-prediction-only \
            --num-episodes 10 \
            --probe-every 1 \
            --output-dir "${EVAL_BASE}/hp_perturb_10ep"
        ;;
    *)
        echo "ERROR: unknown array ID ${ID}" >&2
        exit 2
        ;;
esac

echo "=== DONE task ${ID} ==="
