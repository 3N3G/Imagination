# Track A vs B vs C — Per-Track Decomposition (Regular Gemini Eval)

**Date:** 2026-04-24
**Scope:** Decompose where A_full's advantage comes from (vs B_thinking_2M and C_grounded_2M) on the regular freezenone 50-episode evaluation, at the per-achievement, per-step, and action-distribution levels. Used to design a manual video-review pass to discriminate between two failure modes:
- **(i) Gemini quality** — predictions don't match the algorithm-recommended action for the state
- **(ii) Policy fidelity** — the policy's actions don't follow Gemini's prediction

The decomposition is also the seed for two patch-by-prompt experiments (basic-coverage prompt for B, long-tail prompt for C).

**Wandb runs:**
- A_full (predonly, full data): [n7wmnk82](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/n7wmnk82)
- B_thinking_2M: [7itrrqbh](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7itrrqbh)
- C_grounded_2M: [pjb8wf7z](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/pjb8wf7z)

---

## 1. TL;DR — Episode-Level Metrics

| Metric | A_full | B_thinking_2M | C_grounded_2M |
|---|---|---|---|
| n episodes | 50 | **43** | 50 |
| **Mean return** | **18.98 ± 0.36** | 16.31 ± 1.02 | 14.66 ± 0.83 |
| Mean length | 685.7 | **709.9** | 628.8 |
| Median length | 500.5 | **510.0** | **404.5** |
| p10 / p90 length | 281 / 1121 | 252 / 1166 | 242 / 1361 |
| Mean # achievements | **19.52** | 16.60 | 15.08 |
| Achievements / step | **0.02847** | 0.02339 | 0.02398 |
| Return / step | **0.02768** | 0.02297 | 0.02331 |

**Key observations:**
- **A is more efficient per step**, not just per episode. A's ach/step (0.0285) beats B (0.0234) by **22%** and C (0.0240) by **19%**.
- **B is the longest** (709.9 mean steps) but unlocks **fewer** achievements than A — suggesting wasted steps.
- **C dies fastest** (median 404.5 vs A's 500.5) — short-survival regime, but the surviving episodes can run very long (p90=1361 > A's 1121).
- B's SE (1.02) is **3× A's** (0.36) — larger episode-to-episode variance, indicating less consistent behavior.

---

## 2. A → B Drop Analysis (return 18.98 → 16.31, Δ = −2.67)

### B is slower at almost every achievement
B's episodes are slightly longer (+24 steps) but unlocks happen **100–200 steps later**. The 12 achievements where B is ≥100 steps slower than A:

| Achievement | A step (n) | B step (n) | Δ steps | A rate | B rate | rate Δ |
|---|---|---|---|---|---|---|
| **make_torch** | 725.9 (45) | 899.2 (25) | **+173.3** | 0.90 | 0.58 | **−0.32** |
| **place_torch** | 735.8 (44) | 899.2 (25) | **+163.4** | 0.88 | 0.58 | **−0.30** |
| **place_stone** | 684.7 (50) | 848.6 (32) | **+163.9** | 1.00 | 0.74 | **−0.26** |
| **collect_coal** | 725.9 (45) | 915.9 (27) | **+189.9** | 0.90 | 0.63 | **−0.27** |
| **collect_drink** | 701.8 (48) | 890.5 (30) | **+188.7** | 0.96 | 0.70 | **−0.26** |
| **collect_iron** | 733.3 (34) | 888.4 (26) | **+155.1** | 0.68 | 0.60 | −0.08 |
| **make_stone_sword** | 694.1 (49) | 833.7 (32) | **+139.7** | 0.98 | 0.74 | **−0.24** |
| **wake_up** | 715.6 (46) | 851.8 (32) | **+136.2** | 0.92 | 0.74 | **−0.18** |
| **defeat_skeleton** | 727.9 (41) | 860.1 (29) | **+132.2** | 0.82 | 0.67 | −0.15 |
| **defeat_zombie** | 775.4 (32) | 906.0 (28) | **+130.6** | 0.64 | 0.65 | +0.01 |
| **make_stone_pickaxe** | 702.4 (48) | 826.9 (34) | **+124.6** | 0.96 | 0.79 | −0.17 |
| **make_wood_sword** | 694.1 (49) | 824.2 (33) | **+130.1** | 0.98 | 0.77 | −0.21 |
| collect_sapling | 625.1 (46) | 741.5 (40) | +116.4 | 0.92 | 0.93 | +0.01 |
| place_plant | 625.1 (46) | 741.5 (40) | +116.4 | 0.92 | 0.93 | +0.01 |
| place_furnace | 713.3 (46) | 830.4 (32) | +117.1 | 0.92 | 0.74 | −0.18 |
| make_arrow | 733.1 (44) | 825.2 (34) | +92.1 | 0.88 | 0.79 | −0.09 |
| eat_cow | 690.1 (49) | 782.0 (37) | +91.8 | 0.98 | 0.86 | −0.12 |

**The clean exception:** B unlocks `enter_dungeon` 192 steps **earlier** than A and at higher rate (0.30 vs 0.18) — B is the more aggressive descender, which is consistent with the steerability deep-dive (target_descend dominance for thinking variants).

### B's action distribution: too much DO

| Action | A | B | B/A relative |
|---|---|---|---|
| **DO** | 14.16% | **20.73%** | **+46.4%** |
| NOOP | 1.32% | 2.17% | +64.2% |
| REST | 0.10% | 0.34% | +257% |
| SLEEP | 2.14% | 1.28% | **−40.2%** |
| PLACE_STONE | 2.02% | 1.48% | −26.8% |
| UP | 10.81% | 9.11% | −15.7% |
| DOWN | 9.65% | 7.91% | −18.0% |

**Hypothesis (test in videos):** B substitutes movement and craft/place actions with redundant DO actions — DO on tiles that don't yield anything (bumping a stone with a wood pickaxe before standing on the right cell, repeating DO when target is already chopped, etc.). The 46% DO inflation more than accounts for the ~120-step delay across the basic achievement chain. Combined with the **40% SLEEP drop**, B is "doing more, surviving worse, and waking up later."

---

## 3. B → C Drop Analysis (return 16.31 → 14.66, Δ = −1.65)

### C dies sooner — long-tail loop is abandoned
C's median length is **404.5 vs B's 510** (−21%). The drop is not uniform across achievements; instead **specific behavioral loops disappear**:

| Loop | B rate | C rate | Δ | Interpretation |
|---|---|---|---|---|
| **collect_sapling** | 0.93 | **0.38** | **−0.55** | C ignores saplings on the ground |
| **place_plant** | 0.93 | **0.28** | **−0.65** | C plants 3× less often |
| **eat_plant** | 0.14 | **0.00** | −0.14 | Long-cycle loop completely dead |
| **wake_up** | 0.74 | **0.52** | −0.22 | Sleep/wake cycle truncated |
| make_torch | 0.58 | 0.58 | 0.00 | Identical |
| place_torch | 0.58 | 0.52 | −0.06 | |
| place_furnace | 0.74 | 0.62 | −0.12 | |

`eat_cow` is unaffected (rate 0.86=0.86; mean step 782→685). C does the things it does **earlier** (collect_sapling at step 471 vs B's 741), suggesting C reaches mid-game state quickly and then **stops engaging** with the long-tail cycle.

### C's action distribution: more walking, less doing-anything-specific

| Action | B | C | C/B relative |
|---|---|---|---|
| NOOP | 2.17% | **5.32%** | **+146%** |
| LEFT | 8.76% | 11.56% | +32% |
| RIGHT | 8.57% | 12.67% | +48% |
| UP | 9.11% | 13.31% | +46% |
| DOWN | 7.91% | 11.39% | +44% |
| **SLEEP** | 1.28% | **0.65%** | **−49%** |
| **PLACE_PLANT** | 1.05% | 0.52% | **−50%** |
| **PLACE_TORCH** | 1.17% | 0.82% | −30% |
| **MAKE_TORCH** | 1.38% | 0.81% | **−41%** |
| **DESCEND** | 1.49% | 0.73% | **−51%** |
| **ASCEND** | 1.83% | 0.46% | **−75%** |

**Hypothesis (test in videos):** C reaches stone-age, then **just walks around with no plan** — no torches placed, no plants, no sleeping, no descending. The 5.3% NOOP rate (4× A) suggests "stuck pondering" frames where the policy can't pick a meaningful next move. This is the **content-blindness** of grounded labels surfacing as behavioral collapse on the long-tail loop.

---

## 4. Where A's Compounded Advantage Comes From

A doesn't dominate any single achievement by 100×. The pattern is **broad-based 10–30pp gains + 100–200-step head-start**, which compounds:

- A unlocks the basic chain (wood→stone tools, torches, place_stone) by **~step 700**, leaving ~300+ steps for the harder achievements (defeat_skeleton 0.82, enter_dungeon 0.18, make_iron_sword 0.02). B unlocks the same chain by ~step 850, leaving fewer steps for the long tail.
- A's ach/step (0.02847) beats B's 0.02339 by 22% — **even normalizing for survival, A produces achievements faster.**
- The 12 achievements where A beats B by 10pp+ in rate: `collect_coal, collect_drink, collect_stone, defeat_skeleton, eat_cow, make_stone_pickaxe, make_stone_sword, make_torch, make_wood_pickaxe, make_wood_sword, place_furnace, place_stone, place_torch, wake_up`. **None individually decisive; together they sum the +2.67 return gap.**
- The only places A loses to B: `enter_dungeon` (B 0.30 vs A 0.18) and `eat_plant` (B 0.14 vs A 0.08). Both small-n and not enough to offset.

**Compounding mechanism:** earlier basic-tier completion → more residual steps → more attempts at low-rate high-reward achievements → fatter right-tail of the return distribution (A's p90 length 1121 vs B's 1166 is similar, but A converts long episodes into more achievements).

---

## 5. What to Look For in Videos — Failure-Mode Test Plan

For each contrast, a specific visual cue that would discriminate the failure mode. Run with `mpv` on the video paths under `/data/group_data/rl/geney/eval_results/psf_v2_cadence5_<track>/freezenone_50ep/episode_<NN>/gameplay.mp4`.

| Contrast | Hypothesis to test | Specific visual cue |
|---|---|---|
| **A vs B (basic chain)** | B wastes steps on duplicate / mis-targeted DO | Look for: repeated DO on the same tile with no inventory increase; DO when standing next to wrong material; DO with wrong tool equipped; "DO-DO-DO" runs of 3+ steps |
| **A vs B (sleep)** | B fails to sleep when energy is low → slower late-game | Look for: HUD energy bar near zero, no SLEEP triggered; B walking at night without torches |
| **A vs B (place chain)** | B forgets to PLACE_STONE/PLACE_TORCH after collecting | Look for: B has stone/coal in inventory (HUD) but never invokes the place action |
| **B vs C (long-tail loop)** | C reaches mid-game then walks aimlessly | Look for: C with full mid-game inventory wandering across grass/sand without crafting, sleeping, or descending |
| **B vs C (sapling)** | C ignores saplings even when adjacent | Look for: C walks past a sapling tile, no DO, no PLACE_PLANT |
| **B vs C (NOOP)** | C "freezes" at decision points | Look for: 3+ consecutive NOOP frames mid-episode (not at start) |
| **All tracks** | Death cause | Look for: cause of episode termination — drown, zombie/skeleton kill, lava, starvation. C should disproportionately die from neglect (starvation/dehydration), B from combat/dungeon |
| **All tracks** | Gemini-prediction vs action consistency | Pause at each Gemini call boundary (`gemini_log.jsonl` has step indices); compare predicted next-state-text against the next ~5 actions taken |

---

## 6. The Two Failure-Mode Questions

The user will manually score these. The doc cannot automate either yet.

### (i) Gemini quality — does the prediction-text match the algorithm-recommended action?
**Question:** Given the current state (visible in the video frame at step `t`), is Gemini's prediction-text describing the action a Craftax expert would take?

**Data available:**
- `/data/group_data/rl/geney/eval_results/psf_v2_cadence5_<track>/freezenone_50ep/episode_<NN>/gemini_log.jsonl` — one entry per Gemini call (every 5 steps). Each entry includes the prompt context, raw model output, and parsed prediction text.
- Frame-aligned video at the same path, `gameplay.mp4`.
- The Gemini call cadence is **5 steps** (see `pipeline/config.py:GEMINI_STEP_CADENCE`).

**Suggested protocol:** sample 5 Gemini calls per episode × 3 episodes per track (HIGH/MED/LOW return picks) = 45 Gemini calls per track. Score each `{good, ambiguous, bad}` against the algorithm's recommended action for the state.

### (ii) Policy fidelity — do actions follow Gemini's prediction?
**Question:** After Gemini predicts at step `t`, do the policy's actions over `[t, t+5]` move toward fulfilling that prediction?

**Data available:**
- `summary.json` per episode contains the full action sequence.
- `gemini_log.jsonl` contains the prediction text aligned to step `t`.

**Suggested protocol:** for each Gemini call sampled in (i), check the next 5 actions. Score `{follows, partial, ignores}`.

### Picking informative episodes
Use the existing tool:

```bash
PYTHONPATH=. python tools/pick_demo_episodes.py \
    --eval_dir /data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly/freezenone_50ep \
    --eval_dir /data/group_data/rl/geney/eval_results/psf_v2_cadence5_think_predonly_top2M/freezenone_50ep \
    --eval_dir /data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M/freezenone_50ep \
    --picks high,med,low
```

Per-track HIGH/MED/LOW return picks are the minimum useful set. Add a "B-long-but-low-return" pick (B has 7 episodes with length > p90=1166 but return < median) to capture the suspected DO-waste mode.

---

## 7. Patch-by-Prompt Hypothesis (Forward-Looking)

The decomposition above suggests two distinct deficits that may be addressable by **prompt engineering** alone (no retraining of the embedding pipeline needed beyond regenerating labels with the new prompt — full re-pipeline at the chosen variant).

### Patch P-B (basic-coverage prompt, addresses B's gap)
B's failure mode is **uneven basic-chain coverage** — torches at 58%, stone-tier at 74–79%, place_stone at 74%. A new prompt that explicitly emphasizes the basic loop should patch this:

- Always make AND place a torch when collecting coal
- Always craft both wood AND stone sword/pickaxe at each tier
- Always place stone walls when low on HP or before sleeping
- Always sleep when energy is low, before walking further

**Prompt file (will be created separately):**
`configs/training/templates/predict_state_only_prompt_concise_v2_basic_coverage.txt`

### Patch P-C (long-tail prompt, addresses C's gap)
C's failure mode is **abandoned long-tail loop** — saplings/plants halved, sleep halved, descend halved. A new prompt that mentions long-tail behaviors:

- Plant a sapling whenever one is collected
- Sleep safely (with stone walls / inside) when energy is low
- Place torches with collected coal in dark areas
- Descend through the ladder when one is open and HP is full

**Prompt file (will be created separately):**
`configs/training/templates/predict_state_only_prompt_concise_v2_long_tail.txt`

### Eval array job
The two new prompt variants will be queued as an eval array job (one cell per `{B_thinking_2M, C_grounded_2M} × {basic_coverage, long_tail}` = 4 cells) on the existing freezenone checkpoints. **Note:** prompt change requires re-generating Gemini labels at evaluation time; no retraining of the policy is needed for the eval-time prompt swap test (it tests Gemini-quality dimension only). A full validation requires retraining on labels generated with the new prompt — gated on the eval-time probe being non-null.

---

## 8. Caveats & Open Questions

- B has only **n=43** vs A and C's n=50. The 7 missing episodes are likely OOM/timeout failures during eval — **action distribution and ach-rate values are computed only over the 43 completed episodes**. Survivor bias possible.
- B's SE (1.02) is **2.8× A's** (0.36). The 16.31 mean is less stable than A's 18.98 — more episodes might shift it up or down by ~1 return.
- Per-step normalized metrics tell a slightly different story: C's ach/step (0.0240) > B's (0.0234). C's problem is purely **survival**, not within-episode efficiency. B's problem is **per-step efficiency**.
- The `make_iron_sword` step deltas (n=1 for both A and B) are noise.
- The action distribution of the **die_v2 / adversarial** prompt evals (Apr-23) likely contains more useful failure-mode signal than the regular eval — those evals should be cross-checked when looking at policy fidelity.

---

## 9. Files & Paths Referenced

- Per-track eval results: `/data/group_data/rl/geney/eval_results/psf_v2_cadence5_{predonly,think_predonly_top2M,grounded_predonly_top2M}/freezenone_50ep/episode_<NN>/`
  - `gameplay.mp4` — video
  - `summary.json` — per-step metrics + actions + return + length
  - `gemini_log.jsonl` — per Gemini call (cadence 5)
- Pre-computed analysis JSON: `/tmp/track_analysis_data.txt`
- Wandb run map: `/tmp/wandb_url_map.json`
- Demo-episode picker: `/home/geney/Imagination/tools/pick_demo_episodes.py`
- (To be created) Prompt files:
  - `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_v2_basic_coverage.txt`
  - `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_v2_long_tail.txt`
