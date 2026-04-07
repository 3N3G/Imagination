---
name: project_overview
description: Craftax imagination-augmented RL — architecture, baselines, best results, and current direction
type: project
---

# Imagination-Augmented Offline RL for Craftax

## Goal
Train RL agents that condition on LLM "imagination" of future states. Gemini generates future-state summaries, Qwen3-8B embeds them, policy conditions on embeddings alongside observations.

## Architecture

**Pipeline:** obs → obs_to_text → Gemini prompt → future narrative → Qwen3-8B → 4096-dim embedding → policy conditioning

**Policy variants:**
- `ActorCriticAug` (v1): obs+hidden dual-branch, tanh, orthogonal init (obs gets 1 layer before concat)
- `ActorCriticAugLN`: v1 + LayerNorm + Dropout
- `ActorCriticAugV2`: deep obs branch (3 layers, matching unaugmented) + additive hidden injection + zero-init
- `ActorCriticAugGated`: V2 + learned scalar gate `g ∈ [0,1]` that decides whether to use imagination
- `ActorCritic`: obs-only baseline (3 layers, no hidden branch)

## Data
- 158 Gemini-labelled NPZ files, 126 used for training (12.7M transitions)
- From PPO baseline (200M steps, ~21 return, ~20.9 per-episode mean)
- Embeddings: Qwen3-8B layer-30 mean-pooled, 4096-dim
- Golden/oracle data: 24 human trajectories (38K transitions, ~46.8 return per episode)

## Best Results (live eval, Craftax max=226)

| Model | Return | Training Method |
|-------|--------|-----------------|
| unaug-w1024 | 19.10 ± 1.90 | Pure AWR, train_awr.py |
| aug-w2048 | 18.90 ± 2.14 | Pure AWR + oracle embeddings |
| psf-aug-w2048 | 18.90 ± 1.40 | Pure AWR + predict-only embeddings |
| control_v2_zero | 18.40 ± 2.37 | Pure AWR, V2 arch, hidden=zero |
| BC+AWR best (v8) | 3.60 ± 3.64 | train_awr_weighted_v2.py |
| PPO 200M baseline | ~21.0 | Online RL upper bound |

## Key Findings
1. **Pure AWR works well** — 17-19 return regardless of augmentation
2. **BC+AWR collapses to 2-4 return** — the oracle imitation objective destroys the policy
3. **Embeddings are semantically meaningful** after z-normalization (semantic gap=0.67)
4. **But BC+AWR models don't use them semantically** — they treat hidden as a nonzero cue (KL(real||shuffled) is 100x smaller than KL(real||zero))
5. **Oracle dataset memorized in 20K steps** (loss 1.41→0.03), then BC provides no signal
6. **Architecture doesn't matter** — V1, V2, and unaugmented all get ~18 return when trained with pure AWR
7. Adversarial/die embeddings don't hurt → model not reading semantic content
8. Norm-based source tagging ruled out by direct test

## Current Direction
Testing two hypotheses for making BC+AWR work:
1. **Anti-memorization**: aggressive dropout/wd/entropy to prevent oracle memorization (4-config sweep)
2. **Gated imagination**: learned gate decides when to use hidden state

## Key Paths (~/Imagination layout)
- Models: `models/actor_critic_aug.py` (all PyTorch variants)
- Training: `offline_rl/train_awr_weighted_v2.py` (BC+AWR), `offline_rl/train_awr.py` (pure AWR)
- Eval: `eval/eval_online.py` (augmented), `eval/eval_unaugmented.py` (obs-only)
- Validation: `eval/validate_awr.py`
- Pipeline: `pipeline/` (scan, filter, gemini_label, embed, merge)
- SLURM: `slurm/submit.sh`
- Checkpoints: `/data/group_data/rl/geney/checkpoints/`
- Eval results: `/data/group_data/rl/geney/eval_results/`
