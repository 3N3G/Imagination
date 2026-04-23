---
name: week_3_apr_5_11
description: "Week 3 (Apr 5-11, 2026): v5/v6 BC+AWR sweeps, PSF-consistency fix, BC+AWR failure analysis, embedding-pathway amplification, freeze/V2 experiments launched"
type: project
---

### TL;DR
The week ran a 16-config BC+AWR sweep (v5) to a 6-config PSF-consistent rerun (v6), then diagnosed why BC+AWR collapses to 2-7 return vs 16-19 for pure AWR — tracing the failure to OOD golden BC plus an embedding pathway that amplifies the problem — and queued freeze / V2-architecture interventions.

### What got built/run
- v5 sweep (16 configs) on PSF training data, varying LR/beta/width/dropout/oracle-weight/LN; oracle/val data still oracle-mode Gemini (mismatch discovered mid-week).
- Re-labelled golden trajectories with predict-only Gemini-2.5-flash, built held-out test trajectory (1,175 steps, return=55.35), added advantage/weight histogram logging, launched v6 (6 configs, PSF-consistent).
- Bug fixes in `train_awr_weighted_v2.py`: weighted golden critic loss, removed golden entropy leak, `MAX_DATASET_GB` 30→60, shard wraparound flag, gradient clipping, fresh diag batches, opt-in both-stream entropy, pre-scan hidden stats.
- Diagnostics: hidden-source separability, norm-as-tag test, counterfactual KL (real/zero/shuf/mean), gradient conflict, memorization curves, Δ(real-zero) trajectory.
- New architectures: `ActorCriticAugV2` (deeper obs, additive merge, zero-init hidden) and `ActorCriticAugGated` (learned scalar gate).
- Experiments: anti-memorization sweep (4), gated imagination (1), fix ablation (6), low-oracle-weight sweep (4), partition BC (same-distribution), no-oracle-critic, pure-AWR oracle-vs-PSF label comparison, union-normalization, action-prediction accuracy grid on 807K train + 38K oracle samples, AWR-aug 50-ep online eval.
- Launched: history-conditioned labelling pipeline (history-steps=5, ~1.07M calls, ~$938 est), embedding-comparison pipeline (Qwen3-gen / Qwen3-Embedding / Gemini text-embedding-004), freeze-BC (4 configs on `awr_aug_debug`), V2 architecture comparison (2 configs).

### High-level results
- v5: all oracle configs achieved real>shuf>zero for the first time; best Δ(real-zero)=+0.4734 at lr1e4_ow10; LN critical, LR=1e-4 dominates, oracle-weight and beta insensitive.
- v4 best gameplay: bw01_drop03 = 12.21 return (acc_shuffled >= acc_real in all v4 models — within-data cosine 0.98).
- BC+AWR returns collapse: v6=3.60, v7=2.50, v8=3.60; control_v2_zero=18.40 (V2 arch fine without BC).
- Counterfactual KL at step 7500: KL(real||zero)=2.32 vs KL(real||shuf)=0.046 and KL(real||mean)=0.027; shuffled agree=92.4%, mean-emb agree=94.6%.
- Memorization: BC loss 1.41→0.03 and golden entropy 1.03→0.08 by step 26K; AWR loss 2.85→2.20. Δ(real-zero) peaks +0.48 at 7500, decays to ~0.30 by 40K.
- Low-OW cliff-drop: OW=0→15.40, OW=0.01→5.90, OW=0.05→5.00, OW=0.10→4.70. Partition BC (same-distribution): OW=0.1→18.40 (harmless in-distribution).
- Pure AWR oracle-label vs PSF-label: eval return 13.10 vs 17.10; oracle labels replicate real>>zero>>shuf, PSF shows no discrimination but better gameplay (train/eval label mismatch).
- 50-ep online eval comparison: AWR no-aug 18.38, AWR+BC no-aug 12.28, AWR+BC aug (train-norm) 7.46, naive union 6.68, weighted union 6.46; AWR aug (no BC) 16.30.
- Accuracy grid (oracle 38K): BC models hit ~92% real / ~88-90% shuffled — observation is the primary discriminator, embedding adds ~3pp. Training NLL for AWR+BC noaug = 5.41 vs 2.63 pure AWR noaug.
- Hidden norm stats: raw L2 ~1.2% apart (451.8 vs 446.5), post-norm 62.9 vs 77.7, dim_mean_spread 0.019 vs 0.926, within-oracle cosine 0.58. Weighted union cut norm gap 14.8→11.0 but oracle loss still collapsed (0.04).

### Key decisions and pivots
- Fixed oracle-data/training-data mismatch by re-labelling golden with predict-only and moving to PSF-consistent everywhere (v5 → v6).
- Reframed the bottleneck: BC-on-OOD-golden, not BC-loss formulation (partition BC is harmless); embedding pathway amplifies the collapse but is not sole cause.
- Retreated from "higher accuracy = better gameplay" — unaugmented AWR gets 11.8% golden-action accuracy at 19.1 return; BC models hit ~70%+ at 2-4.
- Shifted intervention design to direct architecture-side controls (freeze obs pathway, V2 merge change) instead of further hparam sweeps.
- OOM/SLURM handling: `--max-dataset-gb 30` to avoid 56GB shards, switch A100→L40S after 6h21m priority wait.

### Open threads into next week
- Freeze BC experiments (4 configs) and V2 architecture comparison (2 configs) pending eval on L40S (jobs 7072895/7072896 → 7072899/7072900).
- History-conditioned labelling (k=5) still running / inadvertently cancelled during embed phase — resumption pending user decision.
- Embedding-comparison pipeline: Qwen3-gen / Qwen3-Embedding running or pending; Gemini text-embedding-004 on CPU after 503 retries.
- Open problem (Apr 7): no setup both teaches semantic embedding use AND yields good gameplay; history-conditioned prediction flagged as a candidate bridge.
- Guess-level interpretations to validate: "hidden branch acts as generic nonzero cue / source tag, not semantic content"; BC+AWR may be a degenerate equilibrium where the hidden branch settles into a weak role.
