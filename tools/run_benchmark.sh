#!/usr/bin/env bash
# One-shot benchmark + scorecard for a SCALING_C variant.
#
# Usage:
#   ./tools/run_benchmark.sh <variant_tag> [num_episodes]
#
# Submits 24 eval cells in parallel via SLURM, polls until all complete,
# then runs tools/scorecard.py to print the markdown scorecard.
#
# Estimated wall time: ~50-90 min depending on slot availability.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <variant_tag> [num_episodes]" >&2; exit 2
fi
VARIANT_TAG="$1"
NUM_EPISODES="${2:-30}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Submitting full benchmark for ${VARIANT_TAG} (n=${NUM_EPISODES}) ==="
VARIANT_TAG="${VARIANT_TAG}" NUM_EPISODES="${NUM_EPISODES}" "${REPO}/slurm/jobs/full_benchmark.sh"

echo
echo "=== Polling for completion (every 60s) ==="
EXPECTED_DIR="/data/user_data/geney/eval_results_temp/psf_v3_pporn_1e8_grounded_${VARIANT_TAG}_steer_score"
echo "Watch dir: ${EXPECTED_DIR}"
echo
echo "After all 24 cells finish, run:"
echo "  PYTHONPATH=. python ${REPO}/tools/scorecard.py --variant ${VARIANT_TAG} --num-episodes ${NUM_EPISODES}"
echo
echo "(Polling not implemented in this thin wrapper; check queue with squeue -u geney)"
