#!/usr/bin/env bash
# ==============================================================================
# Universal SLURM job submitter for the Imagination project.
#
# Usage:
#   ./slurm/submit.sh [OPTIONS] -- COMMAND [ARGS...]
#
# Examples:
#   # Basic training job
#   ./slurm/submit.sh --env craftax_fast_llm --job train_awr \
#       -- python -m pipeline.train_awr --total-steps 100000
#
#   # Eval with specific GPU, low memory
#   ./slurm/submit.sh --env craftax --gpu L40S --mem 32G --time 2:00:00 \
#       --job eval_unaug -- python -m pipeline.eval_unaugmented
#
#   # Array job for sweeps
#   ./slurm/submit.sh --env craftax_fast_llm --array 0-3 --job sweep_lr \
#       -- python -m pipeline.train_awr --lr-index \$SLURM_ARRAY_TASK_ID
#
#   # CPU-only pipeline step
#   ./slurm/submit.sh --env craftax_fast_llm --nogpu --partition cpu --mem 128G \
#       --job gemini_label -- python -m pipeline.gemini_label
#
#   # Dry run (print the script without submitting)
#   ./slurm/submit.sh --dry-run --env craftax --job test -- echo hello
#
# All flags have sensible defaults; only --env and the command are required
# for most jobs.
# ==============================================================================
set -euo pipefail

# ---- Defaults ----------------------------------------------------------------
JOB_NAME=""
PARTITION="general"
QOS=""
GPU_TYPE="A100_80GB"
NUM_GPU=1
NO_GPU=false
CPUS=8
MEM="64G"
TIME="12:00:00"
CONDA_ENV=""
ARRAY=""
DRY_RUN=false
EXTRA_SBATCH=""
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"   # Imagination root

# Known conda environments (short name → full path)
declare -A ENV_MAP=(
    [craftax]="/data/user_data/geney/.conda/envs/craftax"
    [craftax_fast_llm]="/data/user_data/geney/.conda/envs/craftax_fast_llm"
    [craftax_vllm_clean]="/data/user_data/geney/.conda/envs/craftax_vllm_clean"
    [imaug]="/data/user_data/geney/.conda/envs/imaug"
    [test]="/data/user_data/geney/.conda/envs/test"
)

# ---- Parse arguments ---------------------------------------------------------
CMD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --job)        JOB_NAME="$2";    shift 2 ;;
        --partition)  PARTITION="$2";   shift 2 ;;
        --qos)        QOS="$2";         shift 2 ;;
        --gpu)        GPU_TYPE="$2";    shift 2 ;;
        --ngpu)       NUM_GPU="$2";     shift 2 ;;
        --nogpu)      NO_GPU=true;      shift ;;
        --cpus)       CPUS="$2";        shift 2 ;;
        --mem)        MEM="$2";         shift 2 ;;
        --time)       TIME="$2";        shift 2 ;;
        --env)        CONDA_ENV="$2";   shift 2 ;;
        --array)      ARRAY="$2";       shift 2 ;;
        --workdir)    WORKDIR="$2";     shift 2 ;;
        --dry-run)    DRY_RUN=true;     shift ;;
        --sbatch)     EXTRA_SBATCH="$2"; shift 2 ;;
        --)           shift; CMD_ARGS=("$@"); break ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            echo "Usage: $0 [OPTIONS] -- COMMAND [ARGS...]" >&2
            exit 1
            ;;
    esac
done

# ---- Validate ----------------------------------------------------------------
if [[ ${#CMD_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: No command specified. Use -- to separate options from command." >&2
    exit 1
fi

if [[ -z "$CONDA_ENV" ]]; then
    echo "ERROR: --env is required. Known envs: ${!ENV_MAP[*]}" >&2
    exit 1
fi

# Resolve short env name to full path
if [[ -n "${ENV_MAP[$CONDA_ENV]+x}" ]]; then
    CONDA_ENV_PATH="${ENV_MAP[$CONDA_ENV]}"
elif [[ "$CONDA_ENV" == /* ]]; then
    CONDA_ENV_PATH="$CONDA_ENV"
else
    CONDA_ENV_PATH="/data/user_data/geney/.conda/envs/$CONDA_ENV"
fi

if [[ -z "$JOB_NAME" ]]; then
    # Auto-generate from the command
    JOB_NAME=$(echo "${CMD_ARGS[*]}" | sed 's/python -m //;s/ .*//' | tr '.' '_')
fi

# ---- Build sbatch script -----------------------------------------------------
LOGDIR="${WORKDIR}/logs"

SBATCH_SCRIPT=$(cat <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
$([ -n "$QOS" ] && echo "#SBATCH --qos=${QOS}")
$([ "$NO_GPU" = false ] && echo "#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPU}")
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/${JOB_NAME}_%j_%a.out
#SBATCH --error=${LOGDIR}/${JOB_NAME}_%j_%a.err
$([ -n "$ARRAY" ] && echo "#SBATCH --array=${ARRAY}")
$([ -n "$EXTRA_SBATCH" ] && echo "#SBATCH ${EXTRA_SBATCH}")

set -e

# ---- Environment setup (standardised) ----
source ~/.bashrc
conda activate ${CONDA_ENV_PATH}

export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

# Weights & Biases temp dirs (avoid NFS contention)
export WANDB_DIR=/tmp/wandb_\${SLURM_JOB_ID}
export WANDB_CACHE_DIR=/tmp/wandb_cache_\${SLURM_JOB_ID}
mkdir -p "\$WANDB_DIR" "\$WANDB_CACHE_DIR"

# HuggingFace / Torch cache on persistent storage
export HF_HOME=/data/user_data/geney/.cache/huggingface
export TORCH_HOME=/data/user_data/geney/.cache/torch

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${WORKDIR}:\${PYTHONPATH:-}"

cd ${WORKDIR}
mkdir -p ${LOGDIR}

# ---- Info header ----
echo "=========================================="
echo "Job:       \${SLURM_JOB_NAME}"
echo "Job ID:    \${SLURM_JOB_ID}"
echo "Task ID:   \${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Host:      \$(hostname)"
echo "GPU:       \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Conda env: ${CONDA_ENV_PATH}"
echo "Workdir:   ${WORKDIR}"
echo "Date:      \$(date)"
echo "=========================================="

# ---- Run command ----
${CMD_ARGS[@]}

# ---- Footer ----
echo "=========================================="
echo "Completed: \$(date)"
echo "=========================================="
SBATCH_EOF
)

# ---- Submit or print ---------------------------------------------------------
if [[ "$DRY_RUN" = true ]]; then
    echo "=== DRY RUN — would submit the following script ==="
    echo "$SBATCH_SCRIPT"
    echo "=== END ==="
else
    mkdir -p "$LOGDIR"
    TMPSCRIPT=$(mktemp /tmp/imagination_sbatch_XXXXXX.sh)
    echo "$SBATCH_SCRIPT" > "$TMPSCRIPT"
    echo "Submitting job '${JOB_NAME}' (env=${CONDA_ENV}, gpu=${GPU_TYPE}x${NUM_GPU}, mem=${MEM}, time=${TIME})"
    sbatch "$TMPSCRIPT"
    rm -f "$TMPSCRIPT"
fi
