---
name: week_1_mar_22_28
description: "Week 1 (Mar 22-28, 2026): offline data pipeline completion, first AWR imagination training, and a model-size ablation where augmentation only helped at w2048."
type: project
---

### TL;DR
Stood up the full offline data pipeline end-to-end, ran the first AWR imagination-augmented training + online eval, and a w512/w1024/w2048 size ablation in which augmentation only beat the unaugmented baseline at w2048.

### What got built/run
- Phase 2 PPO baseline (job 6721647, 7h52m): 200M steps, 12207 trajectory files.
- Phase 3 filter/repack (job 6723088, 1h55m): 632 files, 2.4GB, 63.7M samples; raw_trajectories deleted after verification.
- Phase 4 Gemini labelling (Mar 23-24): 158 files labelled (quarter of shards).
- Phase 5 Qwen embedding (Mar 24-25): 4096-dim, layer-30, mean-pooled.
- Phase 6 merge (Mar 25): 126 files for training.
- Prompt fix (Mar 23): `oracle_next15_prompt.txt` changed to "exactly 3" summary events; `embed.py` regex updated for `[t+X-t+Y]` range format.
- AWR imagination training (Mar 26): ActorCriticAug, obs(8268)+hidden(4096)->512, 14.3M params, 100K grad steps, batch 256, lr 3e-4, AWR beta 10.0. Validation: real NLL < shuffled < zero.
- Online eval (Mar 26): 10 episodes with live Gemini 2.5 Flash + Qwen3-8B.
- Online augmented PPO (Mar 26-ongoing): initialized from AWR checkpoint, target 100M steps, job 6816399, SPS ~49 (Gemini latency bottleneck), signal handling for graceful SLURM checkpointing.
- Model-size experiment (Mar 28): 6 models, aug/unaug x w512/w1024/w2048, all 100K steps.
- Predict-state-only experiment submitted (Mar 28): top 250 episodes, gemini-3.1-flash-lite, Qwen3-8B embeddings; jobs 6848442-6848448 (select -> label -> embed -> merge -> train w512/w1024/w2048).

### High-level results
- PPO baseline: ~21 return at 200M steps.
- AWR imagination online eval (aug-w512): ~16.20 +/- 5.11 return.
- Model-size ablation (100K steps): aug-w512 = 16.20 (std 5.11), unaug-w512 = 17.70 (2.69), aug-w1024 = 14.50 (4.72), unaug-w1024 = 19.10 (1.90), aug-w2048 = 18.90 (2.14), unaug-w2048 = 17.40 (3.80). Best: unaug-w1024 = 19.10.
- Finding as journaled: augmentation only helps at w2048; below that, unaugmented wins.

### Key decisions
- Adopted "exactly 3" oracle summary format and updated embed-side regex to match.
- Committed to ActorCriticAug dual-branch architecture for AWR training; checkpoint at `checkpoints/awr_imagination/final.pth`.
- Initialized online augmented PPO from the AWR checkpoint rather than from scratch.
- Launched a predict-state-only pipeline (top-250 episodes, gemini-3.1-flash-lite) to eliminate the oracle/predict distribution mismatch.

### Open threads into next week
- Online augmented PPO (job 6816399) still running, bottlenecked by Gemini latency.
- Predict-state-only jobs (6848442-6848448) submitted Mar 28, results pending.
- Augmentation only breaking even/ahead at w2048; unclear if this holds with the predict-state-only data.
