---
name: week_2_mar_29_apr_4
description: "Week 2 (Mar 29 – Apr 4, 2026): predict-state-only pipeline validation, embedding ablations, and the genesis of BC+AWR (v2/v3/v4) with an oracle discrimination head."
type: project
---

### TL;DR
Week 2 validated the predict-state-only (psf) pipeline with a stronger Gemini, showed embeddings are used weakly at eval, and launched the BC+AWR line (v2 → v3 → v4) where oracle-head discrimination improved but did not yet translate to gameplay.

### What got built / run
- Predict-state-only training (w512/w1024/w2048) with two Gemini tiers: 3.1-flash-lite (ps) and 2.5-flash (psf).
- Embedding-ablation eval harness: gemini / adversarial / constant / die / random modes.
- `train_awr_weighted_v2.py`: LayerNorm + Dropout + oracle discrimination head (real vs zero context).
- BC+AWR sweeps: v2 (Mar 31, 4 variants, w512), v3 (Apr 3, dropout × anneal), v4 (Apr 3–4, oracle-weight × dropout, + weight decay 1e-4).
- Updated `validate_awr.py` with `--dropout` flag for the v2 architecture.
- 2x2 factorial: Gemini model × label type with zero-embed eval.
- Weighted BC run (oracle_frac 0.25, oracle_weight 5.0).

### High-level results (numbers as journaled)
- ps (flash-lite, predict-only): w512 7.20, w1024 9.40, w2048 6.60 — all terrible.
- psf (2.5-flash, predict-only): w512 16.70, w1024 16.90, w2048 18.90 ± 1.40 — matches oracle baselines.
- unaug-w2048 (100 ep): 17.66 ± 3.89.
- Embedding ablations on aug-w2048: gemini 14.65 / adv 14.58 / const 14.19 / die 14.70 / random 13.02 — all ~14-15.
- Embedding ablations on psf-w2048: gemini 17.27 / adv 17.60 / const 16.99 / die 17.18 / random 16.42 — all ~16-18.
- Weighted BC: 5.05 ± 3.52 — overweighting rare oracle data hurts.
- 2x2 factorial (zero-embed eval): 25-flash predict-only 13.89 ± 6.24; 31-lite oracle 11.85 ± 5.97; both worse than full-pipeline 16–19.
- BC+AWR v3 discrimination: drop01 Δ=+0.1949, drop03 Δ=+0.1379; annealed variants negative (Δ≈-0.02 to -0.04). Annealing reduces oracle weight to near zero so the head never learns.
- BC+AWR v3 live eval (100 ep): drop01 6.04 ± 4.74, drop03 4.44 ± 2.85 — discrimination did not translate to gameplay.
- BC+AWR v4 discrimination: bw05_drop03 acc_real 0.7004 Δ +0.2443; bw05_drop05 Δ +0.2596; bw01 variants collapse (Δ +0.14 / +0.05). ow=0.50 clearly better than 0.10.

### Key decisions / pivots
- Confirmed Gemini 2.5-flash is the floor for predict-only labels; abandoned flash-lite for training data.
- Declared weighted-BC (oracle upweight at BC loss) a dead end; pivoted to BC+AWR with an oracle discrimination head.
- v3 → v4: halved oracle_frac (0.10 → 0.05), added weight_decay 1e-4, raised oracle weight (ow_eff 0.10 → 0.50), dropped annealing (confirmed it kills the head).
- Submitted jobs 6948121 (shuffled val, v4 array 0-3) and 6948122 (100-ep live eval, v4 array 0-3).

### Open threads entering next week
- v4 shuffled validation (6948121) and live eval (6948122) outcomes — does ow=0.5 discrimination translate to return?
- If v4 live eval still poor: try even higher oracle weight, different oracle-head architecture, or a curriculum approach.
- If v4 live eval works: scale up (larger model, more training steps).
- Open interpretive thread from reflections: models appear relatively robust to garbage embeddings, which the journal frames as "suggests the hidden branch acts more as a bias term than a semantic signal."
