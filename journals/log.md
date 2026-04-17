# Experiment Log

## 2026-04-17
- **Content-sensitivity probes on the 3 PSF freeze_obs_bcawr policies** (adversarial prompts + multistep direction counterfactual) separate "ignores embedding" from "reads embedding" regimes.
- **qwen3gen: ignores content.** adv/die prompts flat (17.42, 17.92 vs baseline 17.58); emb-flip action-change 1.7% across 4 intervention steps. Obs-only policy effectively.
- **qwen3emb: reads content but content HURTS.** Adv/die prompts rise to 18.58 / 18.78 (+2pp vs baseline 16.60); emb-flip 11.6%. Kill the narrative → return goes UP.
- **gemini_emb: reads content, mildly helpful.** Adv/die drops to 13.68 / 13.70 (−1.3pp vs baseline 14.96); emb-flip 14.2%. Highest content-sensitivity but lowest absolute return — obs-branch competence is the main gap.
- Best online return (qwen3gen 17.58) comes from the MOST embedding-ignoring policy. "More content-sensitivity" ≠ better policy at these hyperparameters.
- **β sweep on both emb encoders**: qwen3emb peaks at **β=3 = 17.88 ± 5.36** (+1.28 vs β=10 baseline, ~closes gap to qwen3gen 17.58). gemini_emb peaks at **β=30 = 15.68 ± 5.14** (+0.72). Opposite β preferences. Longer training (100K) hurts both (−0.9, −1.4) — more BC memorization, harmful content bites harder on qwen3emb.
- **Grounded gemini_emb sweep (Phase 1, 8 configs)**: β=30 is a genuine local peak; `(ofrac=0.05, ow=0.5)` already near-optimum; **freezenone = 16.20 ± 6.73 (+0.52)** is the only config to beat β=30 baseline — letting the obs-branch keep adapting helps. freezeobspm (only hidden trains) collapses to 11.64. All other Phase 1 configs lose 1.5–3.6pp.
- **HP/Food perturbation probe (per-field V sensitivity)** on 4 policies. qwen3gen: content-blind (mean \|ΔV\|=0.08, arg-change 4%). qwen3emb β=3: content-reactive WRONG-direction (health_low ΔV=+0.03, health_high=−0.06, food_high=−0.13). gemini_emb β=30: content-reactive CORRECT-direction on all 4 probes; food_low ΔV=−0.33 is strongest single signal. freezenone: 3 of 4 probes stronger/correct vs baseline (food_high 10× stronger correct). Correct imagination reading ≠ better policy — bottleneck is obs-branch integration, not signal.
- **3.1-pro text-generator swap at inference (null result)**: 25-ep eval on gemini_emb β=30 (+0.42) and freezenone (−0.46). Both within noise, opposite signs — stronger text generator doesn't lift policies without retraining. Parks the info-bottleneck hypothesis; obs-branch limit is more likely.
- **Val acc is not a memorization signal for gemini_emb**: freezenone val=0.912 AND online=16.20 (both highest). When embedding has correct-direction content, BC memorization grounds in real state content and transfers. Old "high val → online collapse" pattern was specific to content-blind configs.
- **CRITICAL — `obs_to_text` was silently dropping equipment.** Pickaxe/sword/bow/sword-enchantment/bow-enchantment/armour (×4)/armour-enchantments (×4) were decoded but never printed. Every Gemini call in labelling, eval, and gemini_play has been blind to the player's tool/weapon/armour state. Fix applied 2026-04-17 to both `envs/obs_to_text.py` and `labelling/obs_to_text.py`. All pre-fix training shards, embeddings, and eval numbers in this log operated on an equipment-blind obs; relabelling + re-embedding + retraining is required before the next round of tests. Deferred until prompt iteration settles.
- [Detail →](log_2026-04-17.md)

## 2026-04-16
- **PSF-consistent freeze BC+AWR**: rerun Apr-12 Exp 1 (4 configs) + encoder sweep on freeze_obs_bcawr (qwen3gen, qwen3emb, gemini_emb) with PSF labels on training AND golden/BC data.
- **Best PSF freeze result: psf_freeze_obs_bcawr (qwen3gen) = 17.58 ± 3.56**, +0.78 over Apr-12 oracle-labelled version (16.80). Still below unaug 18.38.
- **Pure-BC configs diverge**: obs_bc collapses harder (2.96 vs 5.54) — oracle-embedding shortcut gone; all_bc improves (12.74 vs 8.56) — less room to overfit oracle.
- **Encoder ordering under BC+AWR**: qwen3gen (17.58) > qwen3emb (16.60) > gemini_emb (14.96), ~2.6pp spread. Reverses Apr-12 pure-AWR trend where gemini_emb was marginally best. qwen3emb had the largest real-vs-shuffled validation gap (+15.3pp) but did not translate to higher online return.
- [Detail →](log_2026-04-16.md)

## 2026-04-15
- **freeze_obs_bcawr β sweep (β∈{1,3,10,30})**: all 4 configs within noise (16.1–16.6); held-out real−zero gap ~58pp (content-presence, not content). AWR β is not a useful lever for this config.
- **Gemini-plays iterations**: 3.1-flash-lite-preview 2.60±1.66 (worse than 2.5-flash 4.06); 3.1-pro-preview + ladder-rules addendum ep1=+10.80 (running). Thinking slows calls ~10×.
- **PPO-from-freeze_obs_bcawr** still at ~16.5 (step 5M/100M) — stable, not progressing.
- [Detail →](log_2026-04-15.md)

## 2026-04-14
- Hidden-only baseline (no obs, pure AWR): 1.98 — catastrophic collapse; imagination embedding alone cannot drive the policy.
- hist5 pipeline: embedding ignored (real≈zero≈shuffled), online ~18.7 — matches obs-only level.
- Gemini-plays Craftax pilot: 2.5-flash direct actor = 4.06 ± 1.98 return.
- [Detail →](log_2026-04-14.md)

## 2026-04-12
- **Freeze BC experiment**: freeze_obs_bcawr (BC+AWR with obs_fc1 frozen) **= 16.80 ± 3.49**, matching AWR baseline (16.30). First BC+AWR that does not collapse (previously 7.46 unfrozen). freeze_all_bcawr (14.68) slightly worse; BC-only variants still toxic (5.54, 8.56). freeze_obs_bc shows 97.9% golden acc but 30.7% held-out — pure memorization.
- **V2 arch comparison**: V2 AWR (16.72) matches LN AWR baseline; V2 BC+AWR (4.74) is WORSE than LN (7.46). V2_bc golden acc=91.25% (memorization) while held-out acc=41.18%.
- **Embedding-source comparison (pure AWR, PSF)**: qwen3gen (original layer-30)=18.62±3.20, qwen3emb (dedicated embedding model)=17.86±3.84. All within noise of unaug 18.38±2.69. Real-vs-zero validation <1% acc either encoder → pure-AWR policy essentially ignores the embedding regardless of source. Gemini-embed pending. Dedicated-embedding-is-cleaner hypothesis not supported.
- [Detail →](log_2026-04-12.md)

## 2026-04-11
- **Embedding comparison running**: Gemini embedding-001 (3072-dim) and Qwen3-Embedding-8B (4096-dim) vs Qwen3-8B generative baseline (17.10), same flash-lite PSF labels. Hypothesis: dedicated embedding models may produce cleaner/more semantic representations with less generative noise.
- **hist5 pipeline running**: 4.27M texts labelled with k=5 history context → embedding → train → eval. Hypothesis: snapshot-only PSF predictions may not correspond well to agent actions; history grounds Gemini in the agent's recent behavior.
- [Detail →](log_2026-04-11.md)

## 2026-04-10
- **AWR aug (row 0) 50-ep eval**: **16.30 ± 3.61** — real embeddings alone (no BC) only slightly hurt vs pure AWR (18.38). Embeddings are not harmful without BC; collapse requires both.
- **Action prediction accuracy grid** (all 6 policies, 807K train samples + 38K oracle):
  - AWR no aug: zero (39.9%) > real (34.7%) = shuffled — model ignores embeddings entirely (trained with zero)
  - AWR aug: real (44.4%) > zero (38.5%) > shuffled (33.3%) — uses embeddings semantically on training dist; oracle: zero best (OOD embeddings harmful)
  - BC models (all 4): **92% oracle accuracy with real embeddings** — complete memorization confirmed. Shuffled also gives 88-90% oracle accuracy → hidden state is the primary discriminating signal, not the embedding.
  - BC training NLL: 5.41 (noaug) vs 2.63 (pure AWR) — BC has overwritten PPO policy
- **Mechanistic story complete**: Oracle hidden states are stereotyped → BC teaches model to use them as a mode-switch → at eval, oracle-like hidden states trigger oracle actions in OOD game states
- [Detail →](log_2026-04-10.md)

## 2026-04-08
- **50-ep definitive results** (correct eval mode: online for aug, zero for noaug):
  - AWR no aug (zero emb, no BC): **18.38 ± 2.69** — healthy baseline
  - AWR+BC no aug (zero emb, BC): **12.28 ± 5.55** — OOD golden BC degrades -33%
  - AWR+BC aug (real emb, BC): **7.46 ± 4.91** — embedding worsens collapse further to -59%
  - AWR+BC aug naive union: **6.68 ± 5.46** — normalization change doesn't help
  - AWR+BC aug weighted union: **6.46 ± 4.54** — normalization change doesn't help
- **Key finding:** Embedding pathway amplifies BC toxicity (18→12 without emb, 18→7 with emb). Normalization is not the root cause — separability is inherent to data distributions.
- History-conditioned labelling (job 7010450) running. Hidden norm stats logging added. `eval_online.py` now shows full Gemini text in video.
- Code changes: `pipeline/gemini_label.py`, `pipeline/text_utils.py`, `offline_rl/train_awr_weighted_v2.py` (`--union-norm`), `eval/eval_unaugmented.py` (`--augmented`), `eval/eval_online.py` (full Gemini text in video).
- [Detail →](log_2026-04-08.md)

## 2026-04-07
- **Key finding**: BC+AWR gets 2-4 return vs 17-19 for pure AWR. control_v2_zero (V2 arch, no BC) gets 18.40 — architecture is fine, BC objective is sole cause of collapse.
- Found 2 bugs in train_awr_weighted_v2.py: unweighted oracle critic loss, oracle entropy leak. Fixed.
- Diagnostic run confirms model uses hidden as nonzero cue, not semantic content (KL(real||shuf) 100x smaller than KL(real||zero)).
- Anti-memorization sweep (4 configs): 2.8-4.7 return. Regularization slows memorization but doesn't fix gameplay.
- Gated imagination (ActorCriticAugGated): 2.80 return. Gate architecture doesn't help.
- **6 more bugs fixed**: shard wraparound stats, no grad clipping, stale diagnostic batches, hidden stats bias, Config mutation, entropy scope.
- **Fix ablation** (6 configs, identical hparams): all_fixes=3.70, old_behavior=2.60, no_clip=2.30, both_ent=1.20. Grad clipping at 1.0 is the most helpful fix. Both-stream entropy is harmful.
- **Low OW sweep**: OW=0 → 15.40, OW=0.01 → 5.90, OW=0.05 → 5.00, OW=0.10 → 4.70. Even trace BC (0.01) causes cliff-drop from 15 to 6. Not a dosage issue — BC is fundamentally toxic.
- [Detail →](log_2026-04-07.md)

## 2026-04-06
- v5 all 16 configs complete. v6 sweep with PSF-consistent oracle data. v7 with V2 arch. v8 with bug fixes.
- [Detail →](log_2026-04-06.md)

## 2026-04-05
- v5 sweep 16 configs submitted. PSF training data.
- [Detail →](log_2026-04-05.md)

## 2026-04-04
- bcawr_v4 sweep finished (job 6944688). Best: bw05_drop03 acc_real=0.7004 Δ=+0.24, bw05_drop05 Δ=+0.26. Low oracle weight (0.1) collapses.
- Submitted shuffled val (job 6948121) + live eval (job 6948122) for all 4 v4 models.
- Updated `validate_awr.py` to support v2 arch (LayerNorm + Dropout).
- [Detail →](log_2026-04-03_04.md)

## 2026-04-03
- bcawr_v3 sweep: 4 variants (drop01/drop03 × anneal/no-anneal). Anneal variants collapsed (negative Δ). No-anneal drop01 best (Δ=+0.19).
- v3 live eval: drop01=6.04, drop03=4.44 — very poor, oracle head not translating to gameplay.
- Launched bcawr_v4: halved oracle fraction (0.05), added weight decay (1e-4), ow 0.5 vs 0.1, dropout 0.3 vs 0.5.
- [Detail →](log_2026-04-03_04.md)

## 2026-04-01
- 2x2 factorial (Gemini model × label type): 25-flash predict-only → 13.89 return, 31-lite oracle → 11.85 return.
- Both worse than full-pipeline psf models (16-19) — smaller/cheaper Gemini models degrade quality.
- [Detail →](log_2026-03-29_04-01.md)

## 2026-03-31
- bcawr_v2 sweep: first BC+AWR attempt with oracle head (gentle/entropy/oawr/anneal). Training completed but validation interrupted — architecture mismatch with validate_awr.py (no LayerNorm support at the time).
- Introduced `train_awr_weighted_v2.py` with LayerNorm, Dropout, oracle head.
- [Detail →](log_2026-03-29_04-01.md)

## 2026-03-30
- psf full-pipeline results: w512=16.70, w1024=16.90, w2048=18.90 — matches oracle baselines. Confirms predict-state-only labels work when using the right Gemini model (2.5 Flash).
- Weighted BC experiment: 5.05 return — overweighting oracle data doesn't help.
- Embedding ablation on w2048: adversarial/constant/die/random all ~14-15 vs gemini=14.65 — aug-w2048 not using embeddings semantically.
- psf-w2048 ablation: all modes 16-18 — more robust, less sensitive to embedding content.
- unaug-w2048 100-episode: 17.66 ± 3.89 (confirming 10-ep estimate of 17.40).
- [Detail →](log_2026-03-29_04-01.md)

## 2026-03-29
- Predict-state-only experiment results: ps-w512=7.20, ps-w1024=9.40, ps-w2048=6.60 — terrible. Training on top-250 episodes with 3.1-flash-lite predict labels produces junk policies.
- Hypothesis: either flash-lite too weak, or top-250 filtering too aggressive.
- [Detail →](log_2026-03-29_04-01.md)

## 2026-03-28
- Model size experiment: 6 models (aug/unaug × w512/w1024/w2048). Key finding: augmentation only helps at w2048 (18.90 vs 17.40). unaug-w1024 best overall (19.10).
- Predict-state-only experiment submitted (jobs 6848442-6848448).
- [Detail →](log_2026-03-22_28.md)

## 2026-03-26–27
- AWR imagination training completed (100K steps, 14.3M params).
- Validation: real hidden states beat zero/shuffled — model uses context.
- Online eval: 10 episodes with live Gemini + Qwen pipeline.
- Online augmented PPO started (job 6816399) from AWR checkpoint.
- Craftax_Baselines repo cleanup.
- [Detail →](log_2026-03-22_28.md)

## 2026-03-22–25
- Phase 2 PPO: 200M steps, ~21 return (job 6721647).
- Phase 3 filter: 632 files → 2.4GB, 63.7M samples.
- Prompt fix: "exactly 3" summary events, updated embed.py regex.
- Phases 4-6: Gemini labelling (158 files), Qwen embedding, merge (126 files).
- [Detail →](log_2026-03-22_28.md)
