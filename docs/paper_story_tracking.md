# Paper story tracking

Cross-walks each storyline claim from the working paper outline against
factual empirical evidence in the journals (`journals/log_2026-*.md`) and
canonical docs. Verdict per claim is one of:

- **FOR** — supported with at least one well-controlled probe
- **AGAINST** — contradicted by the data we have
- **MIXED** — partial support with material caveats
- **UNCERTAIN** — limited evidence, sample-size or methodology issue
- **UNTESTED** — claim not probed

Each row links to the journal day where the experiment was run/recorded.
SLURM job IDs and wandb run IDs are included where they exist so that
results can be replicated. All return numbers are raw Craftax episode
return (max=226 by `achievement_mapping`), unless tagged as `% of max`.

---

## 0. Scoreboard / scale anchor

Without this anchor most "X improves over Y" claims are uninterpretable.

| reference | raw return | % of max 226 | source |
|---|---|---|---|
| Craftax max | 226 | 100 | `craftax/craftax/constants.py` `achievement_mapping` |
| 1B-step PPO-GTrXL (published) | ~41.4 | 18.3 | Craftax README scoreboard |
| 1B-step PPO-RNN (published) | ~34.6 | 15.3 | Craftax README scoreboard |
| 1B-step PPO (published) | ~26.9 | 11.9 | Craftax README scoreboard |
| Our PPO-RNN @ 1e8 (10× less compute) | **27.87** | 12.3 | wandb [`fkxga61m`](https://wandb.ai/iris-sobolmark/craftax-baselines-replication/runs/fkxga61m), [log_2026-04-23](../journals/log_2026-04-23.md) |
| Our PPO-symbolic @ 33M (TIMEOUT) | 17.60 | 7.8 | wandb [`tswtiilh`](https://wandb.ai/iris-sobolmark/craftax-baselines-replication/runs/tswtiilh), [log_2026-04-23](../journals/log_2026-04-23.md) |
| Unaug obs-only (offline-RL) | 18.38 ± 3.99 | 8.1 | [log_2026-04-19](../journals/log_2026-04-19.md), [log_2026-04-22](../journals/log_2026-04-22.md) |
| A_full predonly freezenone | 18.98 ± 2.53 | 8.4 | wandb [`n7wmnk82`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/n7wmnk82), [log_2026-04-22](../journals/log_2026-04-22.md) |
| C_grounded_2M baseline | 14.66 ± 0.83 | 6.5 | wandb [`pjb8wf7z`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/pjb8wf7z), [log_2026-04-22](../journals/log_2026-04-22.md) |
| C_grounded_2M + score_max_v2 | 18.33 ± 0.96 | 8.1 | wandb [`0v0j63nw`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0v0j63nw), [log_2026-04-25](../journals/log_2026-04-25.md) |
| **xhighb baseline** (best aug, n=50) | **16.86 ± 1.23** | 7.5 | [log_2026-04-29](../journals/log_2026-04-29.md) |
| **xhighb + score_max_v2 (n=50)** | **18.13 ± 1.32** | 8.0 | [log_2026-04-29](../journals/log_2026-04-29.md) |
| **xxhighb + score_max_v2 (n=30)** | **20.83 ± 1.74** | 9.2 | [log_2026-04-29](../journals/log_2026-04-29.md) |

Punchline: **every aug-policy we have produced is below the 1e8 PPO-RNN
baseline (12.3% of max) and far below the 1B published scoreboard
(18.3%)**. Best aug raw return so far = 20.83 = 9.2% of max. The story
is **not** "imagination policies set new SOTA"; it is at best
"imagination policies match unaug obs-only and add a new control axis".

---

## 1. Storyline claims about LLMs as world models

### 1.1 LLMs are weak at long-horizon prediction in Craftax

**Verdict: UNTESTED directly, FOR indirectly.**

We have not run a controlled long-horizon-only prediction probe. Indirect
evidence from Gemini-as-direct-actor:

- 2.5-flash direct actor return = **4.06 ± 1.98** vs unaug obs-only
  policy 18.38 ([log_2026-04-14](../journals/log_2026-04-14.md))
- 3.1-flash-lite-preview direct actor = 2.60 ± 1.66 ([log_2026-04-15](../journals/log_2026-04-15.md))
- 3.1-pro-preview ep1 = 10.80 (single episode, expensive) ([log_2026-04-15](../journals/log_2026-04-15.md))
- prompt_iter best (10-iter GEPA-style refinement) = **8.10** ([log_2026-04-29](../journals/log_2026-04-29.md))

This says Gemini is a poor *agent*, not specifically that long-horizon
prediction is the failure mode. To anchor the claim properly we would
need a separate direct prediction-accuracy probe (e.g., k-step forward
predictions vs actual env trajectory).

### 1.2 LLMs are weak at low-level / pathfinding prediction

**Verdict: FOR.**

Three independent probes:

1. The HP/Food/Drink perturbation probe ([log_2026-04-22](../journals/log_2026-04-22.md)):
   Track A predonly food_low ΔV = +0.011 (wrong sign). Track C grounded
   food_low ΔV = +0.098 (10× larger but **wrong sign**). Only Track B
   thinking gave correct sign (-0.012). Most pipelines do not encode
   intrinsic-deficit consequences correctly.
2. obs_to_text gap (log_2026-04-17 detail line in [log.md](../journals/log.md)):
   pickaxe/sword/bow/armour/enchantment fields were silently dropped in
   the obs serializer — Gemini was making predictions blind to the
   player's equipment for the entire pre-Apr-17 dataset. Human inspection
   only caught it after the v2 rerun.
3. "Stone-tile navigation" debug noted in [log_2026-04-19](../journals/log_2026-04-19.md):
   user observed Gemini misreads the local stone vs grass layout.

### 1.3 Gemini performs poorly when directly playing or directly predicting in Craftax

**Verdict: FOR.**

Same evidence as 1.1. Best Gemini-direct-actor result on the existing
prompt is 8.10 (prompt_iter v4, [log_2026-04-29](../journals/log_2026-04-29.md))
vs PPO-RNN 1e8 27.87 vs unaug offline-RL 18.38. Gemini is at best ~30%
of obs-only offline-RL.

### 1.4 LLMs encode useful priors at the high-level / planning level

**Verdict: MIXED.**

Supportive evidence:
- Track B (thinking) food_low ΔV = -0.012 (CORRECT sign), the only
  encoder where the policy reads intrinsic deficits in the right
  direction ([log_2026-04-22](../journals/log_2026-04-22.md)).
- Patch-by-prompt: `v2_long_tail` patch on C_grounded_2M improves return
  by **+2.14** (14.66 → 16.80) by clarifying long-tail tasks
  ([log_2026-04-25](../journals/log_2026-04-25.md), wandb [`0uuf13ul`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0uuf13ul)).
- Score-max v2 prompt on C_grounded_2M: +3.67 over baseline (18.33 vs
  14.66) — embedding-conditioned descent + opportunistic milestone
  prompting moves return materially ([log_2026-04-25](../journals/log_2026-04-25.md)).
- Threshold-6 + canonical-ordering prompt on C_grounded_2M: first-ever
  unlocks of `make_iron_armour` and `collect_diamond` (4 distinct eps,
  all length ≥ 1144) on data where these never appear at training
  ([log_2026-04-27](../journals/log_2026-04-27.md), wandb [`7mcbw5p4`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7mcbw5p4)).

Counter-evidence:
- Gemini-as-direct-actor results above show priors alone don't carry the
  game.
- v2 freezenone +2.62 lift was traced to OBS-branch absorbing equipment
  fields, not imagination content (die/adv prompts produced ≈0 Δ on this
  policy — content-invariant, [log_2026-04-19](../journals/log_2026-04-19.md)).
  At least one apparent "imagination win" was actually a tokenizer fix.

### 1.5 LLM priors are insufficient because Craftax requires game-specific procedural knowledge

**Verdict: FOR.**

- Direct-actor results (1.3) are far below offline RL.
- The chain-task prompts on the specificity matrix are zero-rate: 0/30
  iron pickaxes and 0/30 diamonds when *prompted* for them on
  C_grounded_2M ([log_2026-04-25](../journals/log_2026-04-25.md)). Even
  with a Gemini that explicitly says "go craft an iron pickaxe", policy
  cannot execute the chain — the procedural knowledge is missing in the
  policy too.
- Iron-tier unlocks only appear when (a) policy survives ≥1144 steps AND
  (b) prompt provides ordering rules ([log_2026-04-27](../journals/log_2026-04-27.md)).

---

## 2. "Why Craftax is interesting" claims

### 2.1 General-prior tasks helped by LLM

**Verdict: MIXED / partially FOR.**

- `v2_long_tail` patch describes "build torches before exploring,
  cultivate plants" — a domain-priors patch — and lifts C_grounded_2M
  +2.14 ([log_2026-04-25](../journals/log_2026-04-25.md)).
- `target_collect_stone_v2` shifts PLACE_STONE +129% relative on
  C_grounded_2M ([log_2026-04-24](../journals/log_2026-04-24.md), wandb
  [`6phz27zr`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6phz27zr)).

### 2.2 Game-specific knowledge requires learning, not LLM priors

**Verdict: FOR.** See 1.5 — chain-task prompts collapse to 0%, and
Gemini-direct-play tops out at 8.10. We have no probe yet that isolates
"LLM with game-specific RAG / fine-tuning" from "LLM with priors only".

---

## 3. Core hypothesis

> Gemini can produce high-level, state-conditioned semantic information
> that helps a learned policy generalize, steer behavior, or adapt to
> new situations.

### 3.1 Augmented policy beats no-augmentation policy

**Verdict: AGAINST (so far).**

This is the most important null result in the journals.

- Unaug obs-only baseline: **18.38 ± 3.99** ([log_2026-04-22](../journals/log_2026-04-22.md))
- Best aug baseline (xhighb, n=50): **16.86 ± 1.23** ([log_2026-04-29](../journals/log_2026-04-29.md))
- Best aug + best prompt (xxhighb + score_max_v2, n=30): **20.83**
  ([log_2026-04-29](../journals/log_2026-04-29.md))
- Best aug + best prompt at n=50 (xhighb + score_max_v2): **18.13 ± 1.32**
  — within 1σ of unaug ([log_2026-04-29](../journals/log_2026-04-29.md)).

So: aug + good prompt ≈ unaug obs-only. Aug alone < unaug.
**Without prompt steering, augmentation has not been shown to help raw
return.** This is a recurring tension flagged in the journals (e.g.,
[log_2026-04-22](../journals/log_2026-04-22.md) — "fidelity does not lift
return").

### 3.2 Augmented policy generalizes better to OOD

**Verdict: AGAINST on the OOD test we ran.**

OOD steering on never-unlocked achievements (eat_bat, enter_gnomish_mines,
explore_ood_v1) on xxhighb collapsed enter_dungeon 40%→7-20% with NO
new achievements unlocked, even on minimal v2-edits
([log_2026-04-29](../journals/log_2026-04-29.md)). xxhighb has high
content-sensitivity (Δ(real-zero)=+0.7) but is brittle to ANY prompt
divergence from training distribution.

### 3.3 Augmented policy is steerable

**Verdict: FOR — strongest part of the story.**

Multiple independent probes; full canonical writeup in
[`docs/STEERABILITY.md`](STEERABILITY.md). Key results:

- Specificity matrix on C_grounded_2M (n=30 per cell, 21 prompts):
  **12 WIN / 9 NULL / 1 WRONG-WAY** after v3 iteration
  ([log_2026-04-25](../journals/log_2026-04-25.md), [`SPECIFICITY_MATRIX.md`](SPECIFICITY_MATRIX.md)).
- Synthetic embedding arithmetic ("killer probe"): α=+2 along d_die
  produces Δret = **−5.13** matching direct prompt die_v2's Δret = **−4.90**
  on C_grounded_2M (wandb [`4j5pi14i`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/4j5pi14i),
  [`6s40z5tm`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm),
  [log_2026-04-24](../journals/log_2026-04-24.md)). Same intervention on
  A_full = null (-0.74) — fidelity gradient.
- Direction-only steering: `direction_left_v2` shifts LEFT% from 0.24 →
  0.34 (+41% relative) on C_grounded_2M (wandb
  [`w21fwecj`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/w21fwecj),
  [log_2026-04-24](../journals/log_2026-04-24.md)).
- Mid-episode switch (regular → avoid_animals @ step 200): Δret = **+1.81**
  (z=+1.56, n=30) on C_grounded_2M
  ([log_2026-04-24](../journals/log_2026-04-24.md)). Caveat: re-analysis
  in [log_2026-04-25](../journals/log_2026-04-25.md) finds the late
  embedding reads as "go-explore-grass" not literal avoidance — the
  number is real, the *labeled mechanism* is misleading.

### 3.4 Steering improves return (not just changes behavior)

**Verdict: MIXED, mostly UNCERTAIN.**

- `target_collect_stone_v2` Δret = **+1.10** (z=+1.0, NS at n=50) on
  C_grounded_2M — first augmented prompt that *might* be net-positive
  ([log_2026-04-24](../journals/log_2026-04-24.md), wandb
  [`6phz27zr`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6phz27zr)).
  Not statistically significant — needs replication.
- `target_descend_v2` Δret = +1.01 (z=+0.58, n=21) on C_grounded_2M;
  full-n result was within noise.
- `score_max_v2` prompt on C_grounded_2M: Δret = **+3.67** (z=+2.83) —
  this is the strongest score-max result and it is sig
  ([log_2026-04-25](../journals/log_2026-04-25.md), wandb
  [`0v0j63nw`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0v0j63nw)).
- `direction_left_v2` Δret = **−6.72** (z=−5.7) — steering at the cost
  of survival. Most pure-direction steering hurts.
- AWR-only baseline 17.67 vs C_grounded_2M (BC+oracle) 14.66 — BC+oracle
  trades 3 raw points for prompt-headroom; both routes converge at ~18
  with their best prompts ([log_2026-04-25](../journals/log_2026-04-25.md)).

### 3.5 Higher-fidelity LLM training data raises steerability

**Verdict: FOR.**

Track A (predonly) → Track B (thinking) → Track C (grounded oracle) shows
a monotonic content-sensitivity gradient agreed by **three independent
probes** ([log_2026-04-22](../journals/log_2026-04-22.md), [log_2026-04-23](../journals/log_2026-04-23.md)):

| probe | A_full | A_top2M | B_thinking | C_grounded |
|---|---|---|---|---|
| Held-out Δ(shuf−real) NLL | +0.026 | (similar) | +0.085 | **+0.195** |
| Direction-CF hidden-flip multistep step 75 | 7% | — | 5.3% | **36.8%** |
| v2 die return drop | −0.76 | −2.28 | −4.09 | **−4.90** |
| Synthetic α=+2 d_die return drop | −0.74 | — | — | **−5.13** |

Same ordering on 4 different probe types. Tradeoff: **fidelity inverts
on raw return** (A=18.98 > B=16.31 > C=14.66) — high-fidelity grounded
training has a train/eval distribution mismatch (Gemini sees future at
training, not at deploy) that shows up as wrong-sign V at deploy
([log_2026-04-22](../journals/log_2026-04-22.md): C food_low ΔV +0.098,
wrong sign, 10× A_full).

---

## 4. "Why separate LLM labelling from policy" claims

### 4.1 Modularity: LLM and policy can be modified independently

**Verdict: FOR (by construction, but with constraint).**

- Prompt swap at INFERENCE alone, no retraining: yes, the architecture
  supports it. score_max_v2 vs baseline on the same C_grounded_2M
  checkpoint is a clean prompt-only intervention with +3.67 lift
  ([log_2026-04-25](../journals/log_2026-04-25.md)).
- Text-generator swap at inference (3.1-pro replacing 3.1-flash on
  gemini_emb β=30 freezenone): null result, Δret +0.42 / −0.46 within
  noise ([log_2026-04-17 line in log.md](../journals/log.md)). So
  swapping the LLM at inference alone does not help — policy was
  conditioned on flash-style embeddings.

So: **modular at the prompt level, but the LLM/embedder choice is
baked in.** A different LLM at inference requires retraining (or at
least re-embedding). Worth being explicit in the paper.

### 4.2 Prompt changes can alter the policy without retraining

**Verdict: FOR, strongly.** This is the strongest empirical pillar.

Same C_grounded_2M checkpoint, prompt-only interventions:

| prompt | n | return | Δ vs baseline | source |
|---|---|---|---|---|
| baseline (regular) | 50 | 14.66 | — | [log_2026-04-22](../journals/log_2026-04-22.md) |
| score_max_v2 | 30 | 18.33 | +3.67 (z=+2.83) | [log_2026-04-25](../journals/log_2026-04-25.md), wandb [`0v0j63nw`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0v0j63nw) |
| score_max_v2_thresh6 | 30 | 17.63 | +2.97 | [log_2026-04-27](../journals/log_2026-04-27.md), wandb [`7mcbw5p4`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7mcbw5p4) |
| v2_long_tail | 30 | 16.80 | +2.14 | [log_2026-04-25](../journals/log_2026-04-25.md), wandb [`0uuf13ul`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0uuf13ul) |
| die_v2 | 50 | 9.76 | −4.90 | wandb [`6s40z5tm`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm) |

Prompt-only return swing: **−4.90 to +3.67** (8.5 raw points) without
touching policy weights.

### 4.3 Zero-shot LLM labelling generalizes better OOD than fine-tuned model

**Verdict: UNTESTED.** We have no fine-tuned-world-model baseline. This
is a paper claim that needs either (a) a tightly-scoped citation to
fast-WAM-style work or (b) deferring the comparison.

### 4.4 Fine-tuning the world model might improve in-distribution but hurt OOD

**Verdict: UNTESTED in our experiments.** Same caveat as 4.3.

### 4.5 Frozen LLM gives stable priors

**Verdict: FOR (mechanistically).** No experiment, but this is true by
construction since we never train Gemini. Worth noting that the embedder
is also frozen.

---

## 5. OOD / steering claims

### 5.1 Changing the prompt steers the policy without retraining

**Verdict: FOR.** Restated 4.2 / 3.3.

### 5.2 Same policy changes behavior under different Gemini imaginations

**Verdict: FOR.** Specificity matrix 12/22 WINs ([log_2026-04-25](../journals/log_2026-04-25.md)).
v2 die/adv content probe shows track-ordered behavioral response to
content shift ([log_2026-04-23](../journals/log_2026-04-23.md)).

### 5.3 Good prompting beats unconditioned/baseline policies

**Verdict: MIXED.**

- aug + score_max_v2 (18.33 on C_grounded_2M, 20.83 on xxhighb at n=30,
  18.13 on xhighb at n=50) is at parity with unaug obs-only (18.38) at
  n=50 and slightly above at smaller n.
- v2_long_tail patch on C: +2.14 ([log_2026-04-25](../journals/log_2026-04-25.md)).
- However, identical patches on Track A FAIL (`v2_basic_coverage` Δ=−0.4,
  `v2_long_tail` Δ=−0.2 on A_full). The patch-helps story is C-specific —
  A_full doesn't read content enough to be helped by patching
  ([log_2026-04-25](../journals/log_2026-04-25.md), patch-by-prompt
  table).

### 5.4 Policy works in OOD states / objectives / floors

**Verdict: AGAINST.** Three OOD prompts targeting Floor-2/3 achievements
(eat_bat, enter_gnomish_mines, explore_ood_v1) on xxhighb all collapsed
enter_dungeon rate from 40% → 7-20%, no new achievements unlocked
([log_2026-04-29](../journals/log_2026-04-29.md)). Even minimal v2 → v2+1-bullet
edits crash performance.

The closest positive: threshold-6 + canonical-ordering prompt unlocks
**`make_iron_armour` and `collect_diamond` for the first time** on
C_grounded_2M data that has zero such episodes
([log_2026-04-27](../journals/log_2026-04-27.md)). 4 distinct episodes,
all length ≥ 1144. This IS OOD execution, but conditional on
right-tail survival rather than steering breaking new ground.

### 5.5 Steering achievable rare/absent achievements

**Verdict: AGAINST in the strong form, MIXED in the weak form.**

- Strong (zero in training data → policy unlocks via prompt): 0/30 iron
  pickaxes, 0/30 diamonds when prompted directly on existing C
  ([log_2026-04-25](../journals/log_2026-04-25.md)). After SCALING_C
  (PPO-RNN-derived data), make_diamond_pickaxe 1/30 and collect_diamond
  2/30 with thresh6 — first ever ([log_2026-04-27](../journals/log_2026-04-27.md)).
- Weak (rare → uplifted by prompt): score_max_v2 lifts eat_plant 0% → 7%
  on C_grounded_2M, place_torch +15pp, place_plant +15pp
  ([log_2026-04-25](../journals/log_2026-04-25.md)).

---

## 6. Desired triple-result (steerability + performance + OOD generalization)

| Pillar | Status | Note |
|---|---|---|
| Steerability | **FOR** | 12/22 specificity-matrix cells, synthetic-arithmetic killer probe. |
| Performance improvement vs baseline | **MIXED** | aug + best prompt ≈ unaug; aug alone < unaug. Possibly +0.5–2 raw with right prompt; not significant at n=50 against unaug. |
| OOD generalization | **AGAINST** | xxhighb collapses on Floor-2/3 prompts. Threshold-6 unlocks iron-tier on existing C only via right-tail survival, not via OOD execution. |

The paper should NOT make a triple-claim. The defensible claim is:

> A policy trained on grounded LLM future-narrative embeddings is
> measurably steerable by prompt at inference time, with the level of
> steerability set by the fidelity of the training-time LLM
> conditioning. Steering can produce specific behavioral changes
> (achievement-axis shifts of +5–20pp absolute) and modest return
> changes (−7 to +3.7 raw return) without policy retraining.

This excludes "OOD generalization" and excludes "beats unaug baseline"
on raw return.

---

## 7. Baselines we have / don't have

| Baseline | Have? | Source |
|---|---|---|
| PPO @ 1e8 (online) | Partial (TIMEOUT @ 33M, ep_ret 17.60) | wandb [`tswtiilh`](https://wandb.ai/iris-sobolmark/craftax-baselines-replication/runs/tswtiilh), [log_2026-04-23](../journals/log_2026-04-23.md). Resub job 7464286 still pending in log_2026-04-24. |
| PPO-RNN @ 1e8 | YES, ep_ret 27.87 (training) / 26.12 (50-ep eval) | wandb [`fkxga61m`](https://wandb.ai/iris-sobolmark/craftax-baselines-replication/runs/fkxga61m), [log_2026-04-23](../journals/log_2026-04-23.md), [log_2026-04-26](../journals/log_2026-04-26.md), wandb [`xyneriuz`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/xyneriuz) |
| Unconditioned policy | YES, 18.38 ± 3.99 | [log_2026-04-19](../journals/log_2026-04-19.md), [log_2026-04-22](../journals/log_2026-04-22.md) |
| Shuffled / random / zero embeddings | YES (constant-embed: 17.54 on A_full, 12.22 on C_grounded; Δ(shuf−real) NLL on val for all tracks) | [log_2026-04-22](../journals/log_2026-04-22.md). Hidden-only (no obs): 1.98 ([log_2026-04-14](../journals/log_2026-04-14.md)). |
| Gemini directly playing | YES, 4.06 (2.5-flash), 8.10 (prompt_iter v4 best) | [log_2026-04-14](../journals/log_2026-04-14.md), [log_2026-04-15](../journals/log_2026-04-15.md), [log_2026-04-29](../journals/log_2026-04-29.md) |
| DSPy / GEPA prompting | YES (GEPA-style prompt_iter: best 8.10 vs seed 7.70, then collapses iter 3-8 due to seed-from-output bug, fixed) | [log_2026-04-28](../journals/log_2026-04-28.md), [log_2026-04-29](../journals/log_2026-04-29.md) |
| Oracle/future-conditioned labels | YES (Track C grounded uses 5-step future) | [log_2026-04-22](../journals/log_2026-04-22.md) |
| Standard world-model baseline | NO | Untested. |
| 1B-step baseline | NO (10× too expensive) | We're at 1e8, ~10× below scoreboard. |

The unconditioned obs-only baseline at **18.38** is the right comparand
for "is augmentation actually helping". The PPO-RNN @ 1e8 at 26.12 is
the right comparand for "are we anywhere near a strong direct-RL
baseline" (we are NOT — 7+ raw points behind).

---

## 8. Mechanistic / verification claims (the "is the policy reading content" question)

### 8.1 Policy reads embedding-space directions causally

**Verdict: FOR.** Synthetic embedding arithmetic (4.2/3.3 reference)
isolates the embedding from any LLM text change. C_grounded_2M α=+2 on
d_die produces −5.13 matching prompt-based −4.90 ([log_2026-04-24](../journals/log_2026-04-24.md)).

### 8.2 Random-direction control: most C_grounded steerability is direction-magnitude sensitivity, not content-specific

**IMPORTANT CAVEAT (FOR with major qualifier).** Random Gaussian direction
matched to signal norm produces 14.3% argmax flip at α=+2 on C_grounded
vs signal direction's 21.3% — i.e., **~2/3 of the steerability is gain,
~1/3 is content-specific** (signal +2.1σ above random)
([log_2026-04-24 random-control section](../journals/log_2026-04-24.md)).
The value head reads content cleanly (signal ΔV +0.140 vs random ≈−0.009),
the policy head reads it less cleanly. Worth flagging in the paper —
"the policy is high-gain on the embedding axis it reads".

### 8.3 BC+AWR collapses the policy unless obs is frozen

**Verdict: FOR.** Pure AWR no-aug = 18.38; AWR+BC no-aug = 12.28; AWR+BC
with aug embeddings = 7.46 ([log_2026-04-08 in log.md](../journals/log.md)).
**Embedding pathway amplifies BC toxicity** (18 → 12 without emb, 18 → 7
with emb). Fix: freeze obs_fc1 → 16.80 ([log_2026-04-12 in log.md](../journals/log.md)).

This is methodologically important and shapes the C_grounded_2M recipe
(AWR pretrain 100k → BC+AWR finetune 50k freezenone, see [`STEERABILITY.md`](STEERABILITY.md)
"C_grounded recipe" section). The journals contain the full debugging
sequence ([log_2026-04-07, 04-08, 04-10, 04-12](../journals/log.md)).

### 8.4 Cadence-prompt mismatch fix did NOT unlock content reading

**Verdict: AGAINST.** Cadence=15 vs cadence=5 (matching Gemini's "5-step
forecast" prompt with Gemini-call frequency) produced statistically
indistinguishable online return (18.82 vs 18.58) and equally
content-invariant die/adv probes ([log_2026-04-19 Exp 29-30](../journals/log_2026-04-19.md)).
Fresh-horizon alignment was not the bottleneck for this earlier policy
class.

### 8.5 obs_to_text fix vs imagination is responsible for v2 freezenone +2.62 lift

**Verdict: obs-branch carries it.** Die/adv content probes on v2 freezenone
showed Δret ≈ 0 / −0.4 (within noise) — the policy is content-INVARIANT
despite the +2.62 lift over Apr-17 ([log_2026-04-19 Exp 24](../journals/log_2026-04-19.md)).
This is a **caution against over-claiming** any single performance lift.
This finding predates the Track A/B/C work; later C-grounded checkpoints
do read content (see 3.5).

---

## 9. Specific tooling for steerability (paper figure candidates)

### 9.1 Specificity matrix (figure-1 candidate)

12 WIN / 9 NULL / 1 WRONG-WAY on C_grounded_2M. Columns: target axis,
baseline, prompt cell, z, verdict. Source:
[`docs/SPECIFICITY_MATRIX.md`](SPECIFICITY_MATRIX.md), [log_2026-04-25](../journals/log_2026-04-25.md),
data `probe_results/specificity_matrix.json`.

### 9.2 Track A vs B vs C fidelity gradient (figure-2 candidate)

3-probe agreement on C > B > A content-sensitivity ordering. Source:
[log_2026-04-22](../journals/log_2026-04-22.md), [log_2026-04-23](../journals/log_2026-04-23.md).
Held-out Δ(shuf−real), direction-CF hidden-flip, v2 die return drop.

### 9.3 Synthetic embedding arithmetic α-sweep (figure-3 candidate)

C_grounded_2M monotonic argmax flip / KL / ΔV across α ∈ [-2, +2] on
d_die. A_full flat. Source:
[log_2026-04-24 section A/C](../journals/log_2026-04-24.md). Data at
`probe_results/in_dist_embed_arith/` and `probe_results/value_grad_steer/`.

### 9.4 Patch-by-prompt across tracks (figure-4 candidate)

| Track | baseline | v2_basic_coverage | v2_long_tail |
|---|---|---|---|
| A_full | 18.98 | 18.60 (Δ−0.4) | 18.83 (Δ−0.2) |
| B_thinking_2M | 16.31 | 14.93 (Δ−1.4) | 13.67 (Δ−2.6) |
| **C_grounded_2M** | 14.66 | 16.10 (Δ+1.4) | **16.80 (Δ+2.1)** |

Source: [log_2026-04-25 patch section](../journals/log_2026-04-25.md). wandb
[`vrifr64a`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/vrifr64a),
[`uvsixwzo`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/uvsixwzo) (A);
[`xazexbbs`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/xazexbbs),
[`udo35u3d`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/udo35u3d) (B);
[`rsgr6p45`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/rsgr6p45),
[`0uuf13ul`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0uuf13ul) (C).

### 9.5 SCALING_C variant sweep table (figure-5 candidate)

| variant | data | β (s0/s1) | baseline | v2 | thresh6 |
|---|---|---|---|---|---|
| C_grounded_2M (ref) | top-2M PSF v2 | 10/30 | 14.66 | 18.33 | 17.63 |
| top4M (default C_v3) | top-4M PPO-RNN v3 | 10/30 | 11.93 | 10.83 | 12.30 |
| highbeta | top-4M | 30/50 | 13.77 | 17.33 | 15.50 |
| combined | 6M | 10/30 | 12.30 | 14.37 | 15.97 |
| **xhighb** | 6M | 50/100 | 14.73/16.86 | 18.80/18.13 | 18.50/17.50 |
| **xxhighb** | 6M | 70/200 | 14.20 | **20.83** | 17.23 |

Source: [log_2026-04-28](../journals/log_2026-04-28.md), [log_2026-04-29](../journals/log_2026-04-29.md).

---

## 10. Tension points / open issues for the paper

These are issues the journals already flag and the paper should explicitly
address:

1. **No clean "aug > unaug at n=50" win.** Best n=50 result is
   xhighb + score_max_v2 = 18.13 ± 1.32 vs unaug 18.38 ± 3.99. Within
   noise. Either need to push n higher or reframe the claim away from
   raw-return performance. ([log_2026-04-29](../journals/log_2026-04-29.md))
2. **Steerability vs return tradeoff.** Track C is most steerable but
   has lowest raw return. AWR-only checkpoint has highest baseline raw
   return (17.67) but cannot be improved by prompts (saturated at most
   axes). BC+oracle trades 3 raw points for prompt-headroom.
   ([log_2026-04-25 AWR-only ablation](../journals/log_2026-04-25.md))
3. **xxhighb brittleness.** Best-return variant is brittle to ANY prompt
   edit, including minimal v2+1-bullet tweaks
   ([log_2026-04-29](../journals/log_2026-04-29.md)). Suggests an
   inverse correlation between content-sensitivity and OOD robustness
   at high β. The midbeta variant (β=20/40) is in flight to test.
4. **Apparent "imagination win" was actually obs-fix.** v2 freezenone
   +2.62 over Apr-17 was an obs_to_text equipment-field bug, not
   imagination content
   ([log_2026-04-19 Exp 24](../journals/log_2026-04-19.md)). Single-number
   improvements need content-probe sanity checks.
5. **Cadence fix didn't help.** Cadence-aligned policy gave same return
   and same content-invariance ([log_2026-04-19 Exp 29-30](../journals/log_2026-04-19.md)).
   Bottleneck is conditioning architecture or gradient signal, not
   temporal alignment.
6. **Oracle-at-train + natural-at-deploy creates a bug, not a feature.**
   Track C food_low ΔV = +0.098 is wrong-sign and 10× larger than A_full —
   policy learned "food_low + Gemini still confident → safe" because
   Gemini at train time SAW the future. At deploy it's not safe.
   ([log_2026-04-22 HP/Food probe](../journals/log_2026-04-22.md))
7. **Random-direction control halves the steerability headline.** Most
   of C_grounded's argmax-flip rate is from raw direction-magnitude
   sensitivity; only ~1/3 is content-specific
   ([log_2026-04-24 random control](../journals/log_2026-04-24.md)).
   Don't oversell.
8. **Chain-task prompts (iron pickaxe, diamond) at 0% on existing data**
   even when explicitly prompted ([log_2026-04-25 specificity matrix](../journals/log_2026-04-25.md)).
   Steerability does not include "create skills the policy doesn't
   have".
9. **Gemini-as-actor improves fast with prompt iteration** (8.10 vs
   seed 7.70 in 1 GEPA iter) but can collapse without best-prompt
   seeding ([log_2026-04-28 prompt_iter section](../journals/log_2026-04-28.md)).

---

## 11. What's needed for a defensible paper as of 2026-04-29

Adding these would close the most material gaps:

- **Replicate `target_collect_stone_v2` Δret = +1.10 at higher n** (currently
  z=+1.0, NS) — would make "first net-positive augmented prompt" a
  significant headline.
- **Run full 21-cell specificity matrix on xhighb / xxhighb** (in flight
  per [log_2026-04-29](../journals/log_2026-04-29.md), jobs 7596339-55) to
  back the steerability story on the highest-return checkpoint.
- **midbeta variant** (β=20/40 + combined, in flight per
  [log_2026-04-29](../journals/log_2026-04-29.md)) to test the
  steerability ↔ return tradeoff hypothesis.
- **Floor-3 env-state-injection eval** (proposed, not built per
  [log_2026-04-29](../journals/log_2026-04-29.md)): drops the policy onto
  Floor 3 with the necessary prereqs to test pure execution, decoupled
  from path-planning steerability.
- **Larger-n (≥100) replication on aug + best prompt vs unaug** to
  resolve the 3.1 / 5.3 question of whether augmentation strictly
  improves over obs-only at the top of the prompt distribution.
- **Online-RL-with-imagination training** (the user's own claim — that
  steering would close the productivity gap if integrated into RL
  training). Not built. Currently a paper extension.
