---
name: week 5 apr 19-23
description: "Week 5 (Apr 19-23): v2 pipeline with fixed obs_to_text hits unaug parity at 18.82; cadence=5 alignment is a return null; PSF size ablation shows data is 6x compressible; Track A/B/C fidelity comparison inverts online-return vs held-out content-reading ordering; v2 die/adv prompts decompose format from content and reveal v1 was biased-optimistic on augmented tracks."
type: project
---

### TL;DR

The fixed v2 pipeline reached unaug-parity for the first time (18.82), but three independent axes of follow-on work (cadence alignment, data-size ablation, and a fidelity comparison across Track A/B/C) showed that higher held-out content-reading does not translate into higher online return, and a clean v2 die/adversarial probe revealed that earlier v1 robustness numbers were biased-optimistic via a format escape hatch.

### What got built / run

- Full v2 end-to-end rerun (relabel 158 shards with gemini-3.1-flash-preview, embed with gemini-embedding-001, AWR + freezenone + 50-ep eval). Freezenone checkpoint at `v2_gemini_emb_freezenone`.
- Phase-2 sweep around the v2 winner (6 configs: beta, freeze, ofrac, step-count).
- Cadence-prompt mismatch fix (prompt asks for 5-step forecast; cadence was 15 in config, eval, and train_ppo_augmented; all three moved to 5). Isolated cadence=5 rerun with fresh label directories after discovery that mixing pre/post-prompt-revision labels would silently contaminate the set.
- Thinking-budget bug fix in `eval/eval_online.py` for gemini-3 models.
- PSF size ablation: 5 AWR pretrains on {full 12.7M, 8M, 4M, 2M, 1M} row subsets.
- Track A (predonly extraction): full-data AWR + freezenone + 50-ep eval + semantic suite (direction-CF step-0, multistep, HP/food perturb, constant-embed, v1 die/adv).
- Track B (thinking prompt) and Track C (grounded V6 prompt with future context) on top-2M bitpacked subset: label + embed + merge + AWR + freezenone + 50-ep + semantic suite.
- Track A top-2M fair-size re-training.
- v2 die/adversarial prompt rewrite (algorithm-slot replacement, no override suffix) + 20-sample probe harness + 8-way cross-track eval array.
- PPO and PPO-RNN 1e8 baselines for scoreboard-scale context.
- Infra: DATA_REGISTRY.md added; `pipeline/build_psf_size_subsets.py`, `pipeline/build_bitpacked_top_subset.py`, `tools/in_distribution_semantic_probe.py`, `tools/probe_grounded_prompts.py`, `tools/probe_die_adv_v2.py`, and a stack of new SLURM wrappers.

### High-level results

- v2 pipeline (Exp 22, cadence=15): freezenone = 18.82 +/- 3.89, +2.62 over Apr-17 pre-fix gemini_emb freezenone (16.20) and +0.44 over unaug 18.38. First augmented policy at parity with unaug. Exp 23 sweep of 6 configs around this winner all within ~1 sigma; no config beats default. Exp 24 die/adversarial content probes were both within 1 sigma of the real-gemini baseline: the journal reads the +2.62 as a "better conditioning pipeline" improvement, not content-driven, with the caveat that freeze=obs losing 0.58 is consistent with the obs branch absorbing the gain.
- Cadence=5 rerun (Exp 29): return 18.58 +/- 3.51 vs cadence=15 18.82 +/- 3.89; statistically indistinguishable. KL(real||zero) on val jumps 4.7x (0.035 -> 0.164) but content probes on cadence=5 (Exp 30: die 18.48, adversarial 17.58) remained within noise; journal concludes fresh-horizon alignment does not unlock the imagination pathway.
- PSF size ablation (Exp 31): full 17.48, top8M 17.00, top4M 17.72, top2M 16.86, top1M 15.12. Returns flat from 12.7M down to 2M (~6x reduction at no measurable cost); knee between 2M and 1M. Hidden-branch sensitivity Delta(zero-real) climbs monotonically as data shrinks (0.033 -> 0.145). Caveat: these are AWR-only, not freezenone.
- Track A/B/C fidelity comparison (Apr-22). Online return: A_predonly_full 18.98 +/- 2.53, A_top2M ~18.1, B_thinking_top2M 16.31, C_grounded_top2M 14.66 +/- 5.80. Held-out Delta(shuf-real) NLL: C +0.195 > B +0.085 > A +0.026. Inversion: fidelity raises held-out content-reading but hurts online return. OOD direction-CF argmax-flip: B dominates step-0 (16.7% vs A 4.0%), C dominates multistep (36.8% at step 75, 30.8% at step 300). HP/food food_low Delta-V: A=+0.011 (wrong sign, negligible), B=-0.012 (correct sign, small - the fidelity-axis win), C=+0.098 (wrong sign, 10x A's magnitude). Journal reads C's wrong-sign response as train/eval mismatch (oracle-at-training -> natural-at-inference) materializing as an active behavioral bug, though this is a mechanistic hypothesis rather than a confirmed cause.
- v2 die/adv probes (Apr-23, final n=50 per cell): Delta(v2-regular) die: A_full -0.76, A_top2M -2.28, B_thinking -4.09, C_grounded -4.90; adv: A_full +0.40, A_top2M -2.06, B -2.51, C -2.98. Ordering C > B > A_top2M > A_full matches the held-out Delta_shuf and direction-CF orderings.
- PPO baselines: PPO-RNN 1e8 final ep_ret 27.87 (12.3% of max 226); PPO-symbolic TIMEOUT at 18h / ~33M env steps / ep_ret 17.60. For scoreboard context: published 1B PPO is 11.9%, 1B PPO-GTrXL 18.3%. All imagination-augmented policies remain at or below ~12% raw return in absolute terms.

### Key decisions + pivots

- Labelling-isolation rule adopted after cancelled Apr-19 cadence=5 chain (would have mixed old- and new-prompt labels in one training set). New rule: every labelling-method change (cadence / prompt / model / obs_to_text) writes to its OWN output directory; never rely on resume across method boundaries.
- DATA_REGISTRY.md added as central provenance layer on top of the proliferating v2 variants.
- Validation-file bug caught mid-Track-A: `track_a_predonly_*.sh` pointed `--val-data` at the full-response oracle file even though Track A trained on predonly embeddings; initial val_acc=0.907 / Delta=+0.807 retracted as distribution-mismatch artifacts; correct held-out eval uses each track's own training directory.
- Moved AWR + freezenone training from A100_80GB (saturated / rl partition has only 1 node) to L40S on `general`; AWR pretrain fell from 3h to 15m, freezenone from 2h to ~8m.
- v1 die/adv suffixes diagnosed as format-confounded (21% of adv v1 outputs contained "Instead of"; 40% of die v1 state_understanding contained "override"). Replaced with v2 algorithm-slot replacement; probe harness verified format-clean (zero override, near-zero Instead-of, zero bulleted structures). 0.90 cos-to-concise threshold retired (content shift for die/adv intent is intentionally larger than format-noise floor).
- Pivot in prompt scaffolding: Track C's grounded V6 prompt (concise template + trailing future-context block) was selected out of 6 variants via a cosine-similarity probe against PSF and golden samples (V6 cos 0.931 / 0.894 vs self-noise 0.950 / 0.940).

### Open threads entering next week

- v1-vs-v2 surprise on Track B: v2 drops return further than v1 (die -4.09 vs -2.92; adv -2.51 vs -1.40). Journal hypothesizes v1's "Instead of" / "override" phrasing acted as a format flag the policy could partially discount, and v2's clean-voice bad-play removes that escape hatch; framed as the mask-hypothesis direction confirmed though smaller than the partial-n snapshot suggested. Implication recorded: v1 robustness numbers were biased-optimistic on augmented-trained tracks.
- C_grounded's wrong-sign HP/food response (+0.098 vs A's +0.011 at 10x magnitude) remains the single most interpretable behavioral bug found. v2 die loss (-4.90) strengthens the train/eval-mismatch diagnosis but does not confirm it.
- A_top2M semantic suite incomplete: direction-CF and HP-perturb only ran for A_full; fair-size {predonly, thinking, grounded} at 2M is still missing A_top2M on those axes. Held-out Delta_shuf for A_top2M not yet measured.
- Exploration and verification axes remain untouched by any probe this week.
- Fidelity vs. policy-side bottleneck is not yet resolved: Track B's correct-sign food_low response (-0.012) is the first and only fidelity-axis win on a behavioral sign-test; whether this is scalable or incidental is open.
- Absolute-scale gap: even the best augmented policy (18.98 raw, 8.4% of max) sits below our own 1e8 PPO-RNN baseline (27.87, 12.3%). The imagination-augmentation work has not moved raw performance; its value to date is mechanistic.
