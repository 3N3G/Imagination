#!/usr/bin/env bash
# SCALING_C Phase 4 — embed (Gemini 3072-d) + merge for the new
# top-4M PPO-RNN-derived data.
#
# Reads:
#   gemini_labels_psf_v3_cadence5_grounded_3flash/  (Phase 3 output)
#   filtered_trajectories_psf_v3_pporn_1e8_top4M/   (Phase 2 output)
# Writes:
#   embeddings_psf_v3_cadence5_grounded_predonly_gemini_emb/
#   final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M/
#
# Embed step needs GPU (Gemini API + a CUDA-friendly batch dim, or
# the embed backend supports CPU). Use small GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Phase 2 wrote to user_data; Phase 3 too; Phase 4 follows suit.
DATA_BASE="/data/user_data/geney/scaling_c_data"
FILTERED_DIR="${DATA_BASE}/filtered_trajectories_psf_v3_pporn_1e8_top4M"
GEMINI_DIR="${DATA_BASE}/gemini_labels_psf_v3_cadence5_grounded_3flash"
EMBED_DIR="${DATA_BASE}/embeddings_psf_v3_cadence5_grounded_predonly_gemini_emb"
FINAL_DIR="${DATA_BASE}/final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M"

# Existence checks happen INSIDE the SLURM job (login node has no /data mount).

# Inner script handles the actual embed+merge pipeline (avoids submit.sh
# CMD_ARGS quoting issue with multi-line bash -c).
"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job "scaling_c_phase4" \
    --gpu A100_80GB \
    --mem 64G \
    --time 6:00:00 \
    "$@" \
    -- bash "${SCRIPT_DIR}/jobs/scaling_c_phase4_inner.sh"
