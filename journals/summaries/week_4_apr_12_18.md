---
name: week 4 apr 12-18
description: "Freeze/stopgrad BC+AWR recipes, PSF-consistent retrain, 3-encoder sweep, β sweep, content-sensitivity probes (adv/die, CF multistep, HP/Food); obs_to_text equipment-gap bug found and fixed."
type: project
---

### TL;DR
Week centered on taming BC+AWR's collapse via pathway isolation (freeze/stopgrad) and on probing whether the imagination embedding is read for content; best PSF-consistent freeze_obs_bcawr reached 17.58 ± 3.56 (qwen3gen) and 16.20 ± 6.73 (gemini_emb freezenone), all still under unaug 18.38, with a critical `obs_to_text` equipment-fields bug discovered that invalidates all prior narrative/embedding content.

### Built / run
- Exp 1 (Apr 12): 4-config freeze BC/BCAWR on pretrained AWR ckpt (oracle-labelled data).
- Exp 2: V2 architecture (3-layer obs + zero-init additive hidden merge) vs LN, AWR and BC+AWR from scratch.
- Exp 3: Encoder comparison under pure AWR + PSF (qwen3gen / qwen3emb / gemini_emb).
- Exp 4/5: Adversarial+die prompts and 200K-step extended training on freeze_obs_bcawr.
- Exp 6/9: bc_obs_stopgrad and dual_stopgrad (both BC and AWR detach hooks) from scratch.
- Exp 7/8: Direction-match analysis and obs/embedding step-0 counterfactual; extended to multistep (steps 0/75/150/300).
- Exp 10: `ActorCriticHiddenOnly` (no obs branch) pure AWR.
- Exp 11: 5-step-history Gemini prompt full pipeline run.
- Exp 12: Gemini-as-actor pilot (`llm/gemini_play.py`) with algorithm + history prompt.
- Exp 15 (Apr 16): PSF-consistent rerun of Exp 1 on qwen3gen + 3-encoder sweep on freeze_obs_bcawr; re-embedded golden PSF for qwen3emb / gemini_emb.
- Exp 16/17: Adversarial/die probe and multistep obs/emb counterfactual on 3 PSF encoders.
- Exp 18: β∈{1,3,30} + long (100K) sweep on qwen3emb and gemini_emb freeze_obs_bcawr.
- Exp 19: 8-config grounded sweep on gemini_emb β=30 (β/ofrac/ow/freeze).
- Exp 20: HP/Food per-field V-sensitivity probe (`eval/eval_hp_perturbation.py`, new).
- Exp 21: Inference-time 3.1-pro text-generator swap on two gemini_emb policies.

### High-level results (numbers as journaled)
- freeze_obs_bcawr (oracle-labelled, pretrained) = 16.80 ± 3.49 online; first BC+AWR not to collapse (vs toxic baseline 7.46 ± 4.91). freeze_all_bcawr = 14.68; BC-only configs still broken (5.54, 8.56).
- Mechanism accounting: BC obs-grad stop = +5.5pt lift (12.98), +AWR hidden stopgrad = +1.4 (14.38), +AWR pretrain warm obs = +2.4 (16.80); ~2.0 to unaug 18.38 unexplained.
- V2 arch neutral for AWR (16.72 vs LN 16.30) and worse for BC+AWR (4.74 vs LN 7.46).
- Pure AWR + PSF across encoders within noise of unaug: qwen3gen 18.62, qwen3emb 17.86, gemini_emb 18.80 (unaug 18.38).
- Hidden-only AWR = 1.98 ± 1.01 — embedding alone cannot drive competent play; real>zero>shuffled on held-out but flat on golden.
- hist5 prompt pipeline: online ~18.69 mid-run, but val real≈zero≈shuffled — embedding ignored.
- Gemini-plays pilot v2 = 4.06 ± 1.98 (4/5 eps at 5.0–5.1, 0 parse failures) with thinking disabled.
- Adversarial/die on oracle freeze_obs_bcawr = 16.00 (both) vs 16.80 baseline — content-invariant.
- Step-0 counterfactual: obs flip changes action 30–84%, emb flip 0–6% across 3 policies. Multistep (0/75/150/300) confirms same pattern; max emb-flip 27.8% (toxic BC+AWR at step 300).
- Direction-match lift vs chance: awr_aug +14.1pp, freeze_obs_bcawr +34.6pp, toxic +26.6pp — reinterpreted as Gemini-as-mirror after CF result.
- PSF-consistent freeze_obs_bcawr qwen3gen = 17.58 ± 3.56 (+0.78 over oracle-label 16.80). PSF psf_freeze_obs_bc collapses harder (2.96 vs 5.54); psf_freeze_all_bc improves (12.74 vs 8.56).
- Encoder ordering under freeze_obs_bcawr PSF: qwen3gen 17.58 > qwen3emb 16.60 > gemini_emb 14.96 (~2.6pp spread).
- Adv/die on PSF encoders: qwen3gen flat; qwen3emb +2pp under harmful prompts (content hurts); gemini_emb −1.3pp (content helps). Mean emb-flip rates 1.7% / 11.6% / 14.2%.
- β sweep: qwen3emb peaks β=3 at 17.88 ± 5.36 (+1.28); gemini_emb peaks β=30 at 15.68 ± 5.14 (+0.72). Long (100K) hurts both (qwen3emb −0.94, gemini_emb −1.44).
- gemini_emb Phase 1 sweep: freezenone = 16.20 ± 6.73 is the only config to exceed β=30 baseline; freezeobspm collapses (11.64).
- HP/Food probe ΔV: qwen3gen content-blind (|ΔV|≈0.08, cos 0.98); qwen3emb β=3 reactive but 3/4 signs wrong; gemini_emb β=30 all 4 signs correct (food_low ΔV=−0.325, strongest single signal); freezenone strengthens 3/4 probes.
- 3.1-pro inference swap: psf_freeze β=30 +0.42, freezenone −0.46 — null, deltas within ≈1.2 combined SE.

### Decisions / pivots
- BC+AWR "collapse" is mechanism-decomposed (obs-grad > hidden-grad > pretrain-warm > unexplained remainder).
- Reinterpreted direction-match lift as Gemini-as-mirror (policy follows obs, Gemini predicts from same state) after obs-flip CF showed embedding pathway near-null.
- Moved training pipeline to PSF-consistent end-to-end; re-embedded golden PSF for qwen3emb / gemini_emb.
- Adopted three-regime framing for encoders: content-blind (qwen3gen) / reads-wrong (qwen3emb) / reads-correctly-but-obs-weaker (gemini_emb).
- freezenone replaces obs-branch-frozen as the gemini_emb default after Phase 1 sweep.
- Inference-time generator swap parked; if stronger text is to help, it must be paid at training time.
- Critical obs_to_text equipment-fields bug fixed Apr 17 in both `envs/obs_to_text.py` and `labelling/obs_to_text.py`; user deferred full relabel/re-embed/retrain until prompt iteration settles.

### Open threads into next week
- Whether qwen3emb content-harmful regime is hyperparameter artifact or inherent (broader sweep of OF, OW, layer width untouched).
- Why gemini_emb's obs-branch converges weaker under the same pretrain budget (3072 vs 4096 dim suspected).
- Re-run content probes (adv/die, CF multistep) on β=3 qwen3emb and β=30 gemini_emb winners to confirm direction of content-use shift.
- Replicate qwen3emb β=3 tight-variance at n=100.
- Full pipeline re-run (relabel PSF + re-embed + retrain) once equipment fix + prompt iteration stabilize; all Exp 12–21 numbers used equipment-blind narratives. Obs-branch may be the real bottleneck (would make stronger text generator pointless regardless of training-dist fix).
