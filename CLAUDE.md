# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**Imagination** is a consolidated research repository for training RL agents on the Craftax environment with LLM-augmented "imagination" conditioning. It unifies code from `~/Craftax_Baselines` and the data pipeline at `/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/`.

**Core idea:** Gemini generates future-state narratives, Qwen3-8B embeds them into 4096-dim vectors, and the policy conditions on these embeddings alongside observations.

**Paper**: https://arxiv.org/abs/2402.16801
**Craftax**: https://github.com/MichaelTMatthews/Craftax/

---

## Directory Structure

```
Imagination/
├── models/          # Neural network architectures
│   ├── actor_critic.py       # JAX/Flax (ActorCritic, ActorCriticAug, ActorCriticConv, etc.)
│   ├── actor_critic_aug.py   # PyTorch consolidated (ActorCriticAug, ActorCriticAugLN, ActorCritic)
│   ├── icm.py                # Intrinsic Curiosity Module
│   └── rnd.py                # Random Network Distillation
├── envs/            # Environment utilities
│   ├── wrappers.py           # LogWrapper, AutoReset, OptimisticReset, BatchEnv
│   ├── obs_to_text.py        # Symbolic obs → human-readable text
│   └── image_utils.py        # Image rendering/resizing
├── online_rl/       # JAX PPO variants (run in `craftax` conda env)
│   ├── ppo.py                # Standard PPO (supports --floor_logging flag)
│   ├── ppo_rnn.py, ppo_pixel.py, ppo_rnd.py, ppo_finetune.py
├── offline_rl/      # PyTorch AWR/BC training (run in `craftax_fast_llm` env)
│   ├── train_awr.py          # AWR on imagination-augmented data
│   ├── train_awr_weighted.py # Weighted BC+AWR with oracle upweighting
│   ├── train_awr_weighted_v2.py  # v2: +entropy, +oracle annealing, +weight decay
│   ├── train_ppo_augmented.py    # Online PPO with real-time Gemini+Qwen
│   ├── awr.py, bc.py         # Base algorithms
│   ├── awr_augmented.py      # AWR conditioned on LLM hidden states
│   ├── bc_awr.py             # Combined BC+AWR framework
│   └── awr_vlm_augmented.py  # AWR conditioned on VLM features
├── llm/             # LLM integration
│   ├── llm_play.py           # LLM agent playing Craftax from text obs
│   ├── vllm_policy.py        # vLLM batch inference policy wrapper
│   ├── sglang_policy.py      # SGLang policy wrapper
│   ├── prompts.py            # Prompt templates and text obs filtering
│   ├── extractor.py          # LLM feature extraction interface
│   ├── vllm_hidden_connector.py, vllm_batch_extractor.py
│   ├── vlm_play.py, vlm_server.py
├── pipeline/        # Data processing pipeline
│   ├── config.py             # Shared configuration (paths, models, hyperparams)
│   ├── run.py                # Main orchestrator (phases 4-6)
│   ├── scan_and_filter.py, scan_streaming.py, scan_parallel.py  # Phase 1-2
│   ├── filter_and_repack.py  # Phase 3: filtering + bitpacking
│   ├── gemini_label.py       # Phase 4: Gemini oracle labelling
│   ├── embed.py              # Phase 5: Qwen3-8B embedding extraction
│   ├── merge.py              # Phase 6: combine everything
│   ├── text_utils.py, check_stage.py, repack.py
│   ├── convert_golden_trajs.py, select_top_episodes.py, benchmark_embed.py
├── labelling/       # Redis-based distributed labelling infrastructure
├── eval/            # All evaluation scripts
│   ├── eval_online.py        # Full imagination pipeline eval
│   ├── eval_unaugmented.py   # Obs-only baseline eval
│   ├── eval_aug_zero.py      # Zero-embedding sanity check
│   ├── validate_awr.py       # Policy NLL validation (real/zero/shuffled)
│   ├── eval_awr.py, eval_awr_vlm_augmented.py, eval_policy_wave.py
├── tools/           # Utility scripts
├── analysis/        # Visualization & research scripts
├── scripts/         # Benchmarks, prompt iteration, online_rl_hidden
├── slurm/           # Standardised SLURM job submission
│   ├── submit.sh             # Universal launcher (see below)
│   └── jobs/                 # Pre-configured job wrappers
├── logz/            # Logging utilities (batch_logging)
├── configs/         # YAML/JSON configs (eval, vllm, training)
└── docs/            # Documentation and experiment logs
    └── log/                  # Daily experiment log entries
```

---

## Conda Environments

| Env | Use For | Key Packages |
|-----|---------|-------------|
| `craftax_fast_llm` | **Primary.** All PyTorch training, eval, pipeline, LLM integration | PyTorch, vLLM, transformers |
| `craftax` | JAX online RL only (`online_rl/ppo.py`, `ppo_rnn.py`, `ppo_rnd.py`) | JAX, Flax, distrax, craftax |

Default to `craftax_fast_llm` unless running JAX PPO.

---

## SLURM Job Submission

Use the universal launcher instead of writing ad-hoc sbatch scripts:

```bash
# Basic training
./slurm/submit.sh --env craftax_fast_llm --job train_awr \
    -- python -m offline_rl.train_awr --total-steps 100000

# Specific GPU + resources
./slurm/submit.sh --env craftax_fast_llm --gpu A100_80GB --mem 128G --time 24:00:00 \
    --job embed -- python -m pipeline.embed

# CPU-only (no GPU)
./slurm/submit.sh --env craftax_fast_llm --nogpu --partition cpu \
    --job gemini -- python -m pipeline.gemini_label

# Array job
./slurm/submit.sh --env craftax_fast_llm --array 0-7 \
    --job sweep -- python -m offline_rl.train_awr --seed \$SLURM_ARRAY_TASK_ID

# Dry run (print script without submitting)
./slurm/submit.sh --dry-run --env craftax_fast_llm --job test -- echo hello

# Pre-configured jobs
./slurm/jobs/train_awr.sh
./slurm/jobs/eval_online.sh
./slurm/jobs/ppo_symbolic.sh 2e8
```

The launcher handles: conda activation, TMPDIR=/tmp, PYTHONPATH, WANDB temp dirs, HF_HOME, info headers, error handling.

---

## Model Architectures

**All PyTorch models are in `models/actor_critic_aug.py`** — do NOT define models inline in training scripts.

| Class | Description | Hidden Dim |
|-------|-------------|-----------|
| `ActorCriticAug` | Dual-branch (obs + hidden), tanh | 4096 default |
| `ActorCriticAugLN` | Same + LayerNorm + Dropout | 4096 default |
| `ActorCritic` | Obs-only baseline (no hidden branch) | N/A |

```python
from models.actor_critic_aug import ActorCriticAug, ActorCriticAugLN, ActorCritic
```

JAX/Flax models remain in `models/actor_critic.py`.

---

## Key Data Paths

All data lives on shared storage (NOT in this repo):

| Path | Purpose |
|------|---------|
| `/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories/` | Merged training data (126 train files) |
| `/data/group_data/rl/geney/checkpoints/` | Model checkpoints |
| `/data/group_data/rl/geney/eval_results/` | Evaluation results |
| `/data/group_data/rl/geney/oracle_pipeline/` | Oracle trajectory data |

---

## Import Conventions

All imports use absolute paths from project root (PYTHONPATH is set by `slurm/submit.sh`):

```python
from models.actor_critic_aug import ActorCriticAug   # PyTorch models
from models.actor_critic import ActorCritic           # JAX models
from envs.wrappers import LogWrapper                  # Environment wrappers
from envs.obs_to_text import obs_to_text              # Observation utilities
from llm.prompts import filter_text_obs               # LLM prompt utilities
from llm.extractor import VLLMHiddenStateExtractor    # LLM feature extraction
from logz.batch_logging import batch_log              # Logging
from labelling.obs_to_text import obs_to_text         # Also valid (labelling has own copy)
```

When running locally (not via SLURM), set PYTHONPATH:
```bash
cd ~/Imagination && PYTHONPATH=. python -m offline_rl.train_awr
```

---

## SLURM Script Requirements

When writing new SLURM scripts, ALWAYS use `slurm/submit.sh` or follow these rules:
- `export TMPDIR=/tmp` (required)
- `export PYTHONUNBUFFERED=1`
- Activate conda with full path: `conda activate /data/user_data/geney/.conda/envs/craftax_fast_llm`
- Set `PYTHONPATH` to include project root
- Create WANDB temp dirs under `/tmp/`

---

## Pipeline Overview

```
PPO Training → Raw NPZ → Scan/Filter → Bitpack → Gemini Label → Qwen Embed → Merge → AWR Training → Eval
```

Phases 1-3: Data processing (`pipeline/scan_*.py`, `filter_and_repack.py`)
Phase 4: Gemini labelling (`pipeline/gemini_label.py`)
Phase 5: Qwen embedding (`pipeline/embed.py`)
Phase 6: Merge (`pipeline/merge.py`)

---

## Troubleshooting

**Import errors:** Ensure PYTHONPATH includes project root. Use `slurm/submit.sh` which handles this.

**CUDA/JAX conflicts:** JAX online RL scripts force `JAX_PLATFORMS=cpu` when GPU is needed for PyTorch.

**Wrong conda env:** Use `craftax_fast_llm` for PyTorch work, `craftax` for JAX PPO only.
