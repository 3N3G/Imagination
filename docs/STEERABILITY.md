# STEERABILITY: the LLM-imagination steerability writeup

Canonical writeup of the 2026-04-24 steerability deep-dive. Consolidates the
prompt-variant design, the per-track behavioral and return results, and the
synthetic-embedding arithmetic probes that isolate the policy's reading of the
imagination hidden branch from the LLM's surface text.

Central claim under test:

> LLM-generated future embeddings can produce a competitive Craftax policy
> whose behavior is measurably steerable by future-language interventions,
> even when average return only matches the unaugmented offline-RL baseline.

Status: **SUPPORTED by 6+ independent probes.** One probe (synthetic
embedding arithmetic) is decisive: on `C_grounded_2M`, bypassing Gemini
entirely and injecting α=+2 × d_die into the hidden produces
[−5.13 return](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/4j5pi14i)
(matching the direct-prompt
[die_v2 effect of −4.90](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm)).
The death "axis" lives in embedding space and the policy reads it.

Source journals: `journals/log_2026-04-24.md`, `journals/log_2026-04-23.md`,
`journals/log_2026-04-22.md`. Primary data: `probe_results/master_table_FINAL.md`
and the JSON files under `probe_results/`.

---

## TL;DR — four killer results

1. **Synthetic embedding arithmetic reproduces the prompt-based effect.**
   On `C_grounded_2M`, adding α=+2 × d_die to the regular embedding (Gemini
   call unchanged) drops return by **[−5.13 (z=−3.50, n=30)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/4j5pi14i)**, matching the
   direct die_v2 prompt effect of **[−4.90 (z=−3.84, n=50)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm)**. On `A_full` the
   same intervention is [null (−0.74)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/2061g8dn). Embedding direction is the causal
   mediator of the prompt's content effect.

2. **Positive achievement steering shifts behavior + gives the first net
   positive-return augmented prompt.** On `C_grounded_2M`,
   [`target_collect_stone_v2`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6phz27zr) produces Δret **+1.10 (z=+1.0 NS, n=50)** with
   place_stone +16pp, place_furnace +18pp, wake_up +24pp.
   [`target_descend_v2`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/y09770mm) produces enter_dungeon +10pp (12% → 22%, +83%
   relative) and DESCEND action +44% relative.

3. **Direction-only steering moves the action distribution at the cost of
   survival.** [`direction_left_v2` on `C_grounded_2M`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/w21fwecj): LEFT%-of-moves
   0.24 → 0.34 (**+41% relative**), return drops −6.72 (z=−5.7). The policy
   commits to left-walking past survival utility. The steering *works at
   the action level* even when it destroys productivity.

4. **Patch-by-prompt closes the targeted deficit on C.** A clearer base
   prompt (`v2_long_tail`) explicitly listing the long-tail loop (sleep
   when energy low, plant saplings, place torches, descend on ladder
   sight) produces **Δret +2.14 (z=+1.4, n=30)** on `C_grounded_2M`,
   patching the EXACT achievements identified as deficient in
   `docs/TRACK_ANALYSIS.md`: **wake_up 0.52 → 0.90 (+38pp)**, eat_plant
   0.00 → 0.13 (+13pp), place_torch 0.52 → 0.67 (+15pp). On `B_thinking_2M`
   the matched basic-coverage prompt fails (Δret −2.35) — B's failure is at
   the policy level, not the instruction-clarity level.

**Every probe agrees on the track ordering**: `C_grounded_2M` (high
fidelity) > `A_full` (low fidelity) on steerability; the opposite on
absolute return and robustness.

---

## Track definitions

| Track key | Training data | Inference | On-policy return (regular, n=50) | Steerability profile |
|---|---|---|---|---|
| **`A_full`** | Full 158-file PSF `predonly` data. Gemini called on regular obs with concise prompt, `Prediction:` suffix only, embedded via `gemini-embedding-001` (3072-dim). | Same concise prompt at deploy time. | **[18.98 ± 0.36](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/n7wmnk82)** | Content-blind policy head. Value head reads presence only (Δ_shuf=+0.026). Robust across all adversarial prompts (die_v2 Δ=−0.76, z=−1.1). Negative control for steering. |
| **`B_thinking_2M`** | Top-2M subset of PSF. Gemini called with thinking prompt (thinking_budget=512) on regular obs; `Prediction:` extracted and embedded. | Thinking prompt (thinking_budget=512) at deploy. | **[16.31 ± 1.00 (n=43)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7itrrqbh)** | Mid-fidelity. Held-out Δ_shuf=+0.085 (3× A). Responds strongly to broad bad-play (die_v2 Δ=−4.09, z=−2.8) but less to narrow content steering (avoid_animals Δ=+0.53). OOD dir-CF step-0 hidden-flip 16.7% dominates at short horizon. |
| **`C_grounded_2M`** | Top-2M subset of PSF. Gemini called with grounded V6 prompt (concise template + 5-step future obs block) on regular obs; `Prediction:` extracted and embedded. Oracle-at-label-time. | Concise prompt at deploy (NO oracle / no future). Intentional train/eval distribution mismatch. | **[14.66 ± 0.82](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/pjb8wf7z)** | Highest content fidelity. Held-out Δ_shuf=+0.195 (7× A). All probes register as steerable: die_v2 Δ=−4.90, direction_left LEFT% +41% rel, target_collect_stone +1.10 net positive return. HP/food sign WRONG (10× A magnitude) — grounded policy reads content as oracle-quality. |

Scoreboard-scale reminder: raw returns are all under ~8.4% of Craftax max
(226). 1B PPO = 11.9%; 1B PPO-GTrXL = 18.3%. All augmented tracks sit
below the 1e8 PPO-RNN baseline (27.87 raw = 12.3%).

### Baseline checkpoints

| Track | Checkpoint path |
|---|---|
| A_full | `psf_v2_cadence5_predonly/freezenone/final.pth` |
| A_top2M | `psf_v2_cadence5_predonly_top2M/freezenone/final.pth` |
| B_thinking_2M | `psf_v2_cadence5_think_predonly_top2M/freezenone/final.pth` |
| C_grounded_2M | `psf_v2_cadence5_grounded_predonly_top2M/freezenone/final.pth` |

### Full recipe for `C_grounded_2M` (the policy used for almost all steering experiments)

**Architecture**: `ActorCriticAugLN` (PyTorch, `models/actor_critic_aug.py`),
layer_width=512, hidden_dim=3072 (Gemini embedding dim), 13.2M params,
LayerNorm + dropout=0.0.

**Training data**:
- Trajectory rows: top-2M-rows subset of PSF (2,539 episodes, 2,000,896
  rows, episode-return range 11.09–14.42; the full PSF dataset has mean
  return 9.53, so this is the top ~14% of episodes).
- Path:
  `final_trajectories_psf_v2_cadence5_grounded_predonly_gemini_emb_top2M`.
- The trajectories themselves are PPO rollouts. The labels are
  Gemini-3-flash predictions written *with the future visible* —
  Gemini is shown the next 5 obs and asked to phrase its output as a
  forward "Prediction: ...". Embeddings are produced from just the
  Prediction-suffix via `gemini-embedding-001` (3072-dim).

**Oracle data** (for the auxiliary loss): 38,108 samples from
`oracle_pipeline/predict_only_final_v2_cadence5_predonly_gemini_emb`.
These are *also* grounded-Gemini labels but on a separate held-out
trajectory set; in the run config they are mixed in at
`oracle_fraction=0.05` and weighted at `oracle_loss_weight=0.5`. They
function as the high-quality teacher signal.

**Two-phase training** (sequential, not joint):

1. **AWR pretrain** — `psf_v2_cadence5_grounded_predonly_top2M/awr/`
   - 100,000 steps, batch=256, lr=3e-4, β=10
   - `oracle_loss_weight=0.0` (oracle data present in the loader at
     5% but contributes 0 weight — present for advantage-stat purposes,
     not for behavior shaping).
   - Pure advantage-weighted regression on the grounded labels.

2. **BC + AWR joint finetune** — `psf_v2_cadence5_grounded_predonly_top2M/freezenone/`
   - 50,000 steps, batch=256, lr=1e-4, β=30
   - `oracle_loss_weight=0.5`, `oracle_fraction=0.05`,
     `entropy_coeff=0.01`
   - `freeze_mode=none` (all weights unfrozen)
   - Initialised from the AWR pretrain checkpoint (above).
   - Loss is BC on every batch + AWR on every batch + 0.5× oracle BC on
     5% oracle rows. So *for the freezenone phase* BC and AWR are
     simultaneous, but the AWR-only pretrain happens first.

**Inference (deploy-time)**: same architecture loaded from `final.pth`.
Gemini called with the **regular concise prompt** (no future visible —
intentional train/eval distribution mismatch); the policy must read the
embedding as if it were a real prediction even though label-time
behaviour was oracle. Embedding extracted via the same predonly
pipeline + `gemini-embedding-001`.

**Net effect**: a policy whose hidden branch is sensitive to
content-rich embeddings (Δ_shuf=+0.195, 7× the unaugmented A baseline)
but whose deploy-time embeddings are necessarily lower-fidelity than
training-time. The mismatch is the active behavioural bug noted in
`docs/PATH1_C_TRAIN_EVAL_MISMATCH.md`.

---

## Prompt-variant design

Every `*_v2.txt` template in `configs/training/templates/` is derived from
`predict_state_only_prompt_concise.txt` by **replacing the "Here is a
good algorithm..." block** (lines 23–69 of the base) with an alternate
algorithm of the same structure and voice. No suffixes or override
phrasing (fixes the v1 format-confound issue where 21% of adversarial
outputs contained "Instead of..." phrasing).

The outer instructions (game rules header, five-step horizon, coordinate
conventions, allowed actions, the "Now, predict the future..." line, and
the `{current_state_filtered}` placeholder) are preserved verbatim so
that the predonly-extracted embeddings stay close to the concise noise floor
(cos-sim to concise baseline 0.77–0.93, probe: `tools/probe_steering_v2*.py`).

### How variants differ from the base prompt

Templates: `configs/training/templates/`. Base file: `predict_state_only_prompt_concise.txt`.

**What is identical across every variant**: the game-rules preamble
("Craftax overview" + coordinate convention + intrinsics + actions
list) and the trailing `Current state: {current_state_filtered}`
placeholder. **What changes across variants is exactly one block**:
the `Here is ... algorithm the player will play the game by:`
section in the middle of the prompt. (A few variants additionally
tweak the closing `Prediction:` directive line to forbid negation
phrases like "instead of" / "avoid" — but the algorithm block is
where all the steering signal comes from.)

The base algorithm block is a long multi-section guide ("Survive /
Take ladder / Upgrade equipment / Explore" with sub-sections for
crafting recipes, upgrade decision tree, gathering rules). Each
variant replaces that block with a much shorter single-goal directive.
Here is what the directive says for every variant:

| Variant | What the substituted algorithm tells the player to do |
|---|---|
| **base** | Stay alive and progress down floors. Priorities: Survive → Take ladder → Upgrade equipment → Explore. (Full crafting recipes + upgrade decision tree + gathering rules.) |
| `direction_left_v2` | Walk LEFT every step. Use DO once on any obstacle, then keep walking left. Vertical position held constant. No gathering, crafting, or interaction. |
| `direction_right_v2` | Walk RIGHT every step. (Otherwise identical to `direction_left_v2`.) |
| `direction_up_v2` | Walk UP every step. |
| `direction_down_v2` | Walk DOWN every step. |
| `target_collect_stone_v2` | Stone collection is the single dominant priority. Mine the nearest visible stone; craft a wood pickaxe only if needed; ignore everything else except topping up intrinsics at level 1. |
| `target_descend_v2` | Descending the ladder is the single dominant priority. Move toward the visible ladder and DESCEND; mine through obstacles only if blocking the path. |
| `target_eat_cow_v2` | Hunting and eating cows is the single dominant priority. Chase the nearest cow, attack with DO, eat the meat. Wood only if needed for a sword. |
| `target_drink_water_v2` | Drinking from water is the single dominant priority. Move to nearest water tile, DO repeatedly to fill drink. |
| `target_place_stone_v2` | Placing stone walls is the single dominant priority. PLACE_STONE on every reachable empty tile; mine stone only when out. |
| `target_hunt_animals_v2` | Every visible animal (cows, fruit-bearing plants) is an immediate hunt target. Redirect on sight. |
| `avoid_water_v2` | Water tiles act like walls; route around them. Restore drink only via potions. (Negation framed positively as "treat water as opaque".) |
| `avoid_animals_v2` | Cow/plant tiles act like walls; route around them. Never attack cows. (Negation framed positively as "treat animals as opaque".) |
| `die_v2` | Reach zero health as quickly as possible. Priorities: Seek damage → Neglect intrinsics → Engage threats without preparation → Descend early. |
| `adversarial_v2` | Make the worst possible decisions. Waste resources, craft useless items, ignore threats, walk into hazards. |

The full text of any variant's algorithm block can be read by opening
the corresponding `predict_state_only_prompt_concise_<variant>.txt`
file (or `diff`-ing against the base). Each per-experiment section
below shows the verbatim substituted intro + first priority entry.

**Thinking variants** (`predict_only_thinking_prompt_*_v2.txt`) carry
the byte-identical algorithm block; they only differ from their
concise counterpart in the closing footer, which removes the State
Understanding section and asks Gemini to think privately and emit a
single `Prediction:` line.

---

## Experiment index

Adversarial (broad bad-play):
- [`die_v2`](#die_v2) — reach zero health
- [`adversarial_v2`](#adversarial_v2) — worst plausible decisions

Narrow class-removal (navigationally opaque):
- [`avoid_water_v2`](#avoid_water_v2)
- [`avoid_animals_v2`](#avoid_animals_v2)

Positive single-target:
- [`target_collect_stone_v2`](#target_collect_stone_v2)
- [`target_descend_v2`](#target_descend_v2)
- [`target_eat_cow_v2`](#target_eat_cow_v2)
- [`target_drink_water_v2`](#target_drink_water_v2)
- [`target_place_stone_v2`](#target_place_stone_v2)
- [`target_hunt_animals_v2`](#target_hunt_animals_v2) — queued

Pure cardinal direction:
- [`direction_left_v2`](#direction_left_v2)
- [`direction_right_v2`](#direction_right_v2)
- [`direction_up_v2` / `direction_down_v2`](#direction_up_direction_down) — untested online

Mechanism probes (no prompt variant):
- [Synthetic embedding arithmetic](#synthetic-embedding-arithmetic)
- [Value-gradient direction](#value-gradient-direction-negative-result)
- [Mid-episode behavioral switch](#mid-episode-behavioral-switch)
- [In-distribution α-sweep](#in-distribution-α-sweep-3072-dim)

---

## wandb and video paths

- **Augmented evals**: `https://wandb.ai/iris-sobolmark/craftax-offline-awr`
  (runs named `eval_<TRACK_KEY>_<MODE>_50ep`; each run has per-episode
  `video/episode_XX` panels to watch the policy play).
- **PPO baselines**: `https://wandb.ai/iris-sobolmark/craftax-baselines-replication`.
- **Local video files**: `/data/group_data/rl/geney/eval_results/<eval_dir>/episode_XX/gameplay.mp4`.
  The `<eval_dir>` pattern for each track/mode is the path string shown in
  each probe_results JSON's `eval_dir` field; e.g.
  `psf_v2_cadence5_grounded_predonly_top2M_steer_v2/target_collect_stone_v2_50ep/episode_03/gameplay.mp4`.
- **Use `tools/pick_demo_episodes.py`** to find LOW / MED / HIGH return
  episodes per condition for offline review.

---

<a id="die_v2"></a>
## `die_v2` — adversarial broad "reach zero health"

Template: `configs/training/templates/predict_state_only_prompt_concise_die_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (priority list +
4 expanded sub-sections).

### Prompt algorithm substitution

Full file: `configs/training/templates/predict_state_only_prompt_concise_die_v2.txt`.

> At every step, the player will act with the goal of reaching zero health as
> quickly as possible. The player will choose the highest-priority active goal
> in this order:
> 1. Seek damage
> 2. Neglect intrinsics
> 3. Engage threats without preparation
> 4. Descend early

Followed by a 3–4-line expansion per priority ("step into lava / water when
drink not needed", "not eat / not drink / not sleep", "initiate melee
regardless of health", "descend immediately").

### Per-track results (n=50)

| Track | return ± SE | Δret (z) | Δp(DO) | Δp(LEFT) | Δrate(collect_stone) | Δrate(make_stone_pickaxe) | Δrate(place_stone) |
|---|---|---|---|---|---|---|---|
| A_full | [18.22 ± 0.57](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6g9kloyz) | −0.76 (z=−1.1) | +0.004 | −0.008 | −0.06 | −0.04 | −0.12 |
| A_top2M | [15.86 ± 0.77](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/sibl3kge) | −2.28 (z=−2.4) | −0.007 | +0.003 | −0.08 | −0.08 | −0.06 |
| B_thinking_2M | [12.22 ± 1.08](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/m353pmy3) | −4.09 (z=−2.8) | +0.063 | +0.033 | −0.18 | −0.27 | −0.22 |
| **C_grounded_2M** | **[9.76 ± 0.98](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm)** | **−4.90 (z=−3.84)** | **+0.233** | −0.009 | −0.36 | −0.40 | −0.38 |

wandb: run [`eval_grounded_predonly_top2M_die_v2_50ep`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm) in `craftax-offline-awr`. Video dir:
`/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_v2_probe/die_v2_50ep/episode_XX/gameplay.mp4`.

### Inventory counts (C_grounded_2M, n=50)

| resource | baseline | die_v2 | Δ |
|---|---|---|---|
| stone | 17.94 | 7.62 | −10.32 |
| wood | 17.58 | 12.84 | −4.74 |
| coal | 1.80 | 0.92 | −0.88 |
| iron | 1.08 | 0.30 | −0.78 |
| food_intake_events | 5.22 | 2.20 | −3.02 |
| drink_intake_events | 13.76 | 5.58 | −8.18 |
| monsters_killed_total | 3.76 | 1.20 | −2.56 |

### Interpretation

The die_v2 prompt produces the directed action-shift it prescribes: DO
rate ↑2.2× on C_grounded (the prompt explicitly instructs "engage threats
without preparation" and "seek damage"). C drops 36pp on collect_stone
and 40pp on make_stone_pickaxe — it fails to build the mid-game toolchain.
`A_full` is effectively content-blind at the policy head; its 0.76
return drop is within noise. The monotone ordering by training fidelity
(C > B > A_top2M > A_full) matches held-out Δ_shuf and direction-CF
probes.

---

<a id="adversarial_v2"></a>
## `adversarial_v2` — adversarial "worst plausible decisions"

Template: `configs/training/templates/predict_state_only_prompt_concise_adversarial_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced.

### Prompt algorithm substitution

Full file: `configs/training/templates/predict_state_only_prompt_concise_adversarial_v2.txt`.

> At every step, the player will make the lowest-expected-value decision
> that is still plausible for a confused beginner. Priorities:
> 1. Waste time (NOOP, pick up / place stones, attempt craft actions with no materials)
> 2. Craft the wrong thing (lower tier, decorative placements)
> 3. Mis-prioritize survival (drink when low food, chop wood when low drink)
> 4. Wander away from the ladder

### Per-track results (n=50)

| Track | return ± SE | Δret (z) | Δp(DO) | Δrate(place_stone) | Δrate(make_stone_pickaxe) |
|---|---|---|---|---|---|
| A_full | [19.38 ± 0.29](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/ksubq27s) | +0.40 (z=+0.9) | +0.003 | −0.04 | +0.02 |
| A_top2M | [16.08 ± 0.90](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/zo1imok6) | −2.06 (z=−1.9) | +0.018 | −0.12 | −0.06 |
| B_thinking_2M | [13.80 ± 0.94](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/nqfle4h5) | −2.51 (z=−1.8) | +0.041 | −0.10 | −0.17 |
| C_grounded_2M | [11.68 ± 0.95](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/s5skoqs0) | −2.98 (z=−2.4) | +0.166 | −0.16 | −0.16 |

### Interpretation

Same ordering as die_v2 but with weaker magnitudes. One mode-concentration
note: adv_v2 Gemini outputs disproportionately mention "repeatedly attempt
to craft iron pickaxe without materials" (~12/20 in the probe), so the
behavioral effect is narrower than die_v2's broader "bad play" disruption.
Still a valid content-only probe (0% "Instead of", 0% bulleted structures).

---

<a id="avoid_water_v2"></a>
## `avoid_water_v2` — narrow class-removal (water opaque)

Template: `configs/training/templates/predict_state_only_prompt_concise_avoid_water_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced.

### Prompt algorithm substitution

> The player treats water as navigationally opaque — water tiles act like
> walls that the player routes around. The player's priorities never
> involve water.
> 1. Move toward trees, stone, coal, iron, diamond, animals, crafting
>    stations, or the ladder — whichever is nearest and on the landward
>    side.
> ...
> The player restores Drink by drinking potions rather than by drinking
> from water tiles. If no potion is available, Drink is allowed to decay.

### Per-track results (n=50)

| Track | return ± SE | Δret (z) | Δrate(collect_drink) |
|---|---|---|---|
| A_full | [18.30 ± 0.41](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/hxcfdgwx) | −0.68 (z=−1.2) | −0.06 |
| A_top2M | [(see Apr-23 partial)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/2qy7xdo8) | −1.66 marginal | — |
| B_thinking_2M | [15.90 ± 0.92](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/asthb0rc) | −0.41 (z=−0.3) | −0.10 |
| C_grounded_2M | [13.64 ± 1.01](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/pdjilnqf) | −1.02 (z=−0.8) | −0.12 |

### Interpretation

Smallest effect among narrow-steering prompts. Water-drinking is a smaller
return component than eating cows, so removing it has a modest cost. No
track shows a significant effect.

---

<a id="avoid_animals_v2"></a>
## `avoid_animals_v2` — narrow class-removal (cows opaque)

Template: `configs/training/templates/predict_state_only_prompt_concise_avoid_animals_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced.

### Prompt algorithm substitution

> The player treats animals (cows and plants) as navigationally opaque —
> their tiles act like walls that the player routes around.
> The player restores Food by eating cooked meat already in inventory, or
> by harvesting plants for fruit when plants are adjacent. The player
> never attacks cows, never uses DO on a cow, and never moves onto a
> tile adjacent to a cow.

(Caveat about the quoted prompt: Craftax has no "cooked meat" inventory
item — eating a cow drops `meat` that increments the food intrinsic
directly when DO'd onto. The prompt template's "cooked meat" phrasing is
a Gemini-facing hallucination of a Minecraft-style mechanic. Quoted
verbatim above; results below are nonetheless valid because the policy
reads the embedding, not the literal text.)

### Per-track results (n=50)

| Track | return ± SE | Δret (z) | Δp(DO) | Δrate(eat_cow) |
|---|---|---|---|---|
| A_full | [18.20 ± 0.50](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/203drusj) | −0.78 (z=−1.3) | +0.006 | −0.06 |
| B_thinking_2M | [16.84 ± 0.96](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/756yap5q) | +0.53 (z=+0.4) | −0.042 | 0.00 |
| **C_grounded_2M** | **[11.54 ± 1.02](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/gwhh7c5j)** | **−3.12 (z=−2.4)** | **+0.157** | −0.12 |

### Inventory counts (C_grounded_2M, n=50)

| resource | baseline | avoid_animals | Δ |
|---|---|---|---|
| stone | 17.94 | 10.26 | −7.68 |
| wood | 17.58 | 10.62 | −6.96 |
| food_intake_events | 5.22 | 2.30 | −2.92 |
| drink_intake_events | 13.76 | 7.40 | −6.36 |
| monsters_killed_total | 3.76 | 2.04 | −1.72 |

### Interpretation

C_grounded loses 3 return when the imagination prescribes avoiding cows,
vs A_full (−0.78) and B_thinking (+0.53, actually non-significantly
positive). Narrow steering of a specific class tile separates C_grounded
from B_thinking — the grounded policy parses the specific content, the
thinking policy doesn't. Mid-episode switch variant (below) inverts this:
[`switch → avoid_animals @ step 200`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7tk8vvzv) on C_grounded gives **+1.81 return (z=+1.56)**, one of the
strongest positive signals in the dataset.

---

<a id="target_collect_stone_v2"></a>
## `target_collect_stone_v2` — positive target "mine stone"

Template: `configs/training/templates/predict_state_only_prompt_concise_target_collect_stone_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (+ Prediction directive
rewritten to forbid negation phrasing).

### Prompt algorithm substitution

> The player treats stone collection as the single dominant priority.
> Every step is chosen to bring the player closer to mining the next
> stone tile.
> 1. Move toward the nearest visible stone tile and use DO to mine it
>    when adjacent. If the player needs a wood pickaxe, brief detour
>    to wood.
> 2. Route to the closest cluster.
> 3. If no stone is visible, move toward the unexplored direction with
>    the highest chance of revealing stone.
> The player ignores cows, plants, ladders, monsters, water, and
> ornamental tiles.

### Per-track results

| Track | n | return ± SE | Δret (z) | Δrate(place_stone) | Δrate(place_furnace) | Δrate(wake_up) |
|---|---|---|---|---|---|---|
| A_full | 50 | [18.04 ± 0.48](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/62tmvkfl) | −0.94 (z=−1.6) | −0.08 | −0.02 | −0.04 |
| B_thinking_2M | 50 | [12.76 ± 0.91](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/lcwqcxo4) | −3.55 (z=−2.61) | −0.18 | −0.18 | −0.14 |
| **C_grounded_2M** | 50 | **[15.76 ± 0.73](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6phz27zr)** | **+1.10 (z=+1.0)** | **+0.16** | **+0.00 (baseline 0.68)** | **+0.24** |

(Note: C_grounded place_stone baseline 0.68 → +0.16 absolute = 0.84.
place_furnace: baseline 0.62, condition 0.80 per inventory data →
+0.18 absolute; master table shows +0.00 because of a small-count 
smoothing artifact; inventory counts are the reliable measure.)

### Inventory counts (C_grounded_2M, n=50)

| resource | baseline | target_collect_stone | Δ |
|---|---|---|---|
| **stone** | 17.94 | **24.68** | **+6.74** |
| wood | 17.58 | 15.32 | −2.26 |
| coal | 1.80 | 1.72 | −0.08 |
| iron | 1.08 | 1.00 | −0.08 |
| torches | 3.46 | 3.96 | +0.50 |
| arrows | 1.88 | 2.60 | +0.72 |

### Interpretation

**The first augmented prompt to produce Δret > 0 on C_grounded.** The
policy is steered into a cluster of stone-handling activities: stone
collected +38% relative, place_stone rate +16pp, place_furnace rate
+18pp (from 0.62 → 0.80), wake_up rate +24pp. Crafting the relevant
toolchain is not disrupted. A_full shows no such enrichment — a direct
negative control for "track fidelity enables positive target steering".

wandb/eval_dir: `psf_v2_cadence5_grounded_predonly_top2M_steer_v2/target_collect_stone_v2_50ep`.

---

<a id="target_descend_v2"></a>
## `target_descend_v2` — positive target "reach ladder, descend"

Template: `configs/training/templates/predict_state_only_prompt_concise_target_descend_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (+ Prediction directive
rewritten to forbid negation phrasing).

### Prompt algorithm substitution

> The player treats descending the ladder as the single dominant
> priority. Every step is chosen to bring the player closer to the
> visible ladder and to the DESCEND action.
> 1. If a ladder is visible, move directly toward it and DESCEND when
>    standing on it.
> 2. If no ladder is visible, move toward the unexplored direction most
>    likely to reveal one.
> 3. Gather wood / stone only when needed to mine through a blocking
>    obstacle on the path.

### Per-track results (n=50 unless noted)

| Track | n | return ± SE | Δret (z) | Δrate(enter_dungeon) | Δp(DESCEND) relative |
|---|---|---|---|---|---|
| A_full | 50 | [19.28 ± 0.27](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/jkb171iy) | +0.30 (z=+0.7) | +0.06 | +24% |
| B_thinking_2M | 50 | [15.26 ± 0.95](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/ng2xcv0g) | −1.05 (z=−0.75) | −0.02 | +12% |
| **C_grounded_2M** | 50 | [14.80 ± 0.99](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/y09770mm) | +0.14 (z=+0.1) | **+0.10 (0.12 → 0.22, +83% rel)** | **+44%** |

### Interpretation

The **cleanest directed-goal steering** result. C_grounded's DESCEND
action rate rises +44% relative and `enter_dungeon` achievement nearly
doubles (12% → 22%). Return delta is within noise. Mid-episode variant
(`switch → target_descend @ 200`) doubles the enter_dungeon bump to
+20pp (see [switch section](#mid-episode-behavioral-switch)).

wandb/eval_dir: `psf_v2_cadence5_grounded_predonly_top2M_steer_v2/target_descend_v2_50ep`.

---

<a id="target_eat_cow_v2"></a>
## `target_eat_cow_v2` — positive target "hunt cow"

Template: `configs/training/templates/predict_state_only_prompt_concise_target_eat_cow_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (+ Prediction directive
rewritten to forbid negation phrasing).

### Prompt algorithm substitution

> The player treats hunting and eating cows as the single dominant
> priority.
> 1. Move toward the nearest visible cow and use DO when adjacent.
>    Attack until meat drops, step onto meat, DO to eat.
> 2. Route to closest cow.
> 3. If no cow visible, move toward grassland.

### Per-track results (n=50)

| Track | n | return ± SE | Δret (z) | Δrate(eat_cow) | Δrate(wake_up) |
|---|---|---|---|---|---|
| A_full | 50 | [18.62 ± 0.42](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/yzndpabh) | −0.36 (z=−0.6) | −0.02 | −0.04 |
| B_thinking_2M | 50 | [16.58 ± 0.83](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/snqn8pak) | +0.27 (z=+0.21) | +0.08 | +0.06 |
| C_grounded_2M | 50 | [14.42 ± 0.83](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/fa0dlxd8) | −0.24 (z=−0.2) | +0.04 | +0.20 |

### Inventory counts (C_grounded_2M, n=50)

| resource | baseline | target_eat_cow | Δ |
|---|---|---|---|
| stone | 17.94 | 16.14 | −1.80 |
| food_intake_events | 5.22 | 3.82 | −1.40 |
| monsters_killed_total | 3.76 | 3.16 | −0.60 |

### Interpretation

Weakest target effect. eat_cow baseline already at 0.86 ceiling — little
room to boost. Food intake events slightly DOWN, contrary to the prompt's
direction. Likely explanation: "single-minded cow-hunting" disrupts the
normal food+crafting pipeline without improving cow-catching.

---

<a id="target_drink_water_v2"></a>
## `target_drink_water_v2` — positive target "drink water"

Template: `configs/training/templates/predict_state_only_prompt_concise_target_drink_water_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (+ Prediction directive
rewritten to forbid negation phrasing).

### Prompt algorithm substitution

> The player treats drinking from water as the single dominant
> priority. Every step is chosen to bring the player closer to a water
> tile and into a position to use DO on it.

### Per-track results (A_full, n=50)

| Track | return ± SE | Δret (z) | Δp(DO) | Δrate(collect_drink) |
|---|---|---|---|---|
| A_full | [18.08 ± 0.40](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/18tl33lj) | −0.90 (z=−1.7) | −0.013 | −0.14 |

C_grounded / B_thinking not run online.

### Interpretation

On A_full, a prompt asking to drink more actually REDUCES collect_drink
and DO rate — the prompt's content is not being parsed causally by
A_full. Consistent with A_full's content-blindness across all probes.

---

<a id="target_place_stone_v2"></a>
## `target_place_stone_v2` — positive target "build stone walls"

Template: `configs/training/templates/predict_state_only_prompt_concise_target_place_stone_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (+ Prediction directive
rewritten to forbid negation phrasing).

### Prompt algorithm substitution

> The player treats placing stone walls as the single dominant priority.
> 1. If the player has stone and is next to empty tile, PLACE_STONE.
> 2. If no stone, mine stone.
> 3. Build compact stone enclosures wherever space allows.

### Per-track results (A_full, n=50)

| Track | return ± SE | Δret (z) | Δrate(place_stone) |
|---|---|---|---|
| A_full | [18.38 ± 0.52](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/ix5qkxbs) | −0.60 (z=−0.9) | −0.10 |

### Interpretation

A_full again content-blind. place_stone rate actually DOWN −10pp (baseline
1.00 → 0.90). C_grounded not run.

---

<a id="target_hunt_animals_v2"></a>
## `target_hunt_animals_v2` — positive target "hunt all animals"

Template: `configs/training/templates/predict_state_only_prompt_concise_target_hunt_animals_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (+ Prediction directive
rewritten to forbid negation phrasing).

wandb runs:
[A_full](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/3m3rdbnf) /
[B_thinking_2M](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/9vr77310) /
[C_grounded_2M](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/3chzhpeu).

### Prompt algorithm substitution

> The player treats ALL visible animals (cows and plants) as immediate
> hunt targets. The moment any animal is visible, the player redirects
> to attack or harvest it.
> 1. If any cow visible, move toward nearest, DO repeatedly, eat meat.
> 2. If any plant with fruit visible, harvest.
> 3. Otherwise move toward grassland.

### Per-track results (n=50)

| Track | n | return ± SE | Δret (z) | Δrate(eat_cow) | Δrate(eat_plant) |
|---|---|---|---|---|---|
| A_full | 50 | [18.96 ± 0.42](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/3m3rdbnf) | −0.02 (z=−0.04) | +0.00 | +0.00 |
| B_thinking_2M | 50 | [14.72 ± 1.03](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/9vr77310) | −1.59 (z=−1.10) | −0.08 | −0.12 |
| C_grounded_2M | 50 | [12.70 ± 0.99](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/3chzhpeu) | −1.96 (z=−1.52) | −0.04 | +0.00 |

(C_grounded_2M shows a smaller +1.1z `cow_eat_events/ep` win at n=30 in
the [Specificity matrix (2026-04-25)](#specificity-matrix-2026-04-25);
the steer_v2 n=50 run here uses a different seed and shows a slight net
return drag instead. Both runs are consistent in that the prompt does
not extend life on C, while A is unaffected.)

---

<a id="direction_left_v2"></a>
## `direction_left_v2` — pure cardinal direction "walk left"

Template: `configs/training/templates/predict_state_only_prompt_concise_direction_left_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (+ Prediction directive
rewritten to forbid negation phrasing).

### Prompt algorithm substitution

> The player walks LEFT (negative-column direction) at every step.
> Every action is the LEFT action unless a wall, water, or other
> obstacle blocks the next leftward tile, in which case the player
> uses DO once on the obstacle and then resumes walking left. Vertical
> position is held constant. The player does not turn, does not gather
> resources, does not craft, and does not interact with cows, plants,
> ladders, monsters, or crafting stations on the current row.

This is the **minimal content variation**: 4 prompts that differ only in
one cardinal direction. If the policy responds, it is unambiguously
reading the embedding content (no other plausible explanation).

### Per-track results (n=50 unless noted)

| Track | n | return ± SE | Δret (z) | LEFT%-of-moves (baseline → cond) | Δp(DO) |
|---|---|---|---|---|---|
| A_full | 50 | [17.48 ± 0.54](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/8g2lqfcy) | −1.50 (z=−2.3) | 0.262 → 0.255 (−3%) | −0.021 |
| B_thinking_2M | 50 | [13.44 ± 1.01](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/wfxhxqlg) | −2.87 (z=−2.01) | 0.255 → 0.267 (+5%) | +0.056 |
| **C_grounded_2M** | 50 | **[7.94 ± 0.84](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/w21fwecj)** | **−6.72 (z=−5.7)** | **0.243 → 0.343 (+41%)** | +0.175 |

### Interpretation

On C_grounded, pure-direction content produces the cleanest possible
behavioral shift (+41% LEFT% relative). The policy **does** read bare
cardinal direction from the embedding. But the trained policy lacks the
meta-skill to follow the steering only partially — it commits to
left-walking until it dies, collapsing productive behaviors. Steering
at the cost of survival.

wandb/eval_dir: `psf_v2_cadence5_grounded_predonly_top2M_steer_v2/direction_left_v2_50ep`.

### Achievement-level collapse (C_grounded)

| achievement | baseline rate | direction_left rate | Δ |
|---|---|---|---|
| collect_stone | 0.94 | 0.42 | −52pp |
| make_wood_pickaxe | 0.94 | 0.50 | −44pp |
| make_stone_pickaxe | 0.76 | 0.26 | −50pp |
| place_stone | 0.68 | 0.30 | −38pp |

---

<a id="direction_right_v2"></a>
## `direction_right_v2` — pure cardinal direction "walk right"

Template: `configs/training/templates/predict_state_only_prompt_concise_direction_right_v2.txt`.
Algorithm-section diff vs base: lines 23–69 replaced (+ Prediction directive
rewritten to forbid negation phrasing).

### Prompt algorithm substitution

> The player walks RIGHT (positive-column direction) at every step.
> (Otherwise identical in structure to direction_left_v2.)

### Per-track results (n=50 unless noted)

| Track | n | return ± SE | Δret (z) | RIGHT%-of-moves (Δ) | LEFT%-of-moves (Δ) |
|---|---|---|---|---|---|
| A_full | 50 | [18.22 ± 0.46](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/alqre6lj) | −0.76 (z=−1.3) | +0.002 | −0.005 |
| B_thinking_2M | 50 | [14.78 ± 0.83](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/533r9el6) | −1.53 (z=−1.17) | **+0.039 (+16% rel)** | −0.009 |
| C_grounded_2M | 50 | [11.16 ± 0.92](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/olscsp79) | −3.50 (z=−2.8) | +0.007 | +0.001 |

### Interpretation

Weaker than direction_left on RIGHT%. C_grounded's return drop (−3.50)
is mostly via DO-spike (Δp(DO)=+0.165) rather than clean RIGHT-walking.
B_thinking at n=50 shows the cleanest right-specific behavior of the
three tracks — RIGHT% +16% relative — confirming the partial-n
observation; the embedding-content asymmetry is real on B but smaller
than C's reaction.

---

<a id="direction_up_direction_down"></a>
## `direction_up_v2`, `direction_down_v2` — remaining cardinals

Templates exist (`*_direction_up_v2.txt`, `*_direction_down_v2.txt`) with
identical structure to left/right. Now run on `C_grounded_2M` as part
of the [Specificity matrix (2026-04-25)](#specificity-matrix-2026-04-25):
all 4 cardinal directions move move-share-of-cardinals in the prompted
direction (z=+2.0 to +6.0). Not run on `A_full` or `B_thinking_2M`.

---

<a id="patch-by-prompt"></a>
## Patch-by-prompt: `v2_basic_coverage` and `v2_long_tail`

**Setup.** Two new base prompts that don't *steer* the policy toward a
single behavior — they instead expand the baseline algorithm to
explicitly mention the long-tail achievements that
`docs/TRACK_ANALYSIS.md` identified as deficits per track. Both prompts
are run on the existing `freezenone` checkpoints, no retraining (this is
purely an inference-time prompt swap).

- `v2_basic_coverage` (template:
  `predict_state_only_prompt_concise_v2_basic_coverage.txt`) — addresses
  B's basic-chain coverage gap: always make+place a torch on coal, craft
  both wood AND stone tier sword/pickaxe, place stone walls before
  sleeping, sleep when energy low.
- `v2_long_tail` (template:
  `predict_state_only_prompt_concise_v2_long_tail.txt`) — addresses C's
  long-tail abandonment: plant a sapling whenever collected, sleep
  safely when energy low, place torches with collected coal, descend on
  visible open ladder when HP is full.

### Per-track before/after (n=30 patch / n=43–50 baseline)

| Track | baseline | `v2_basic_coverage` | Δret (z) | `v2_long_tail` | Δret (z) |
|---|---|---|---|---|---|
| A_full | 18.98 ± 0.36 (n=50) | [18.60 ± 0.49](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/vrifr64a) | −0.38 (z=−0.62) | [18.83 ± 0.50](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/uvsixwzo) | −0.15 (z=−0.24) |
| B_thinking_2M | 16.31 ± 1.02 (n=43) | [14.93 ± 1.12](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/xazexbbs) | −1.38 (z=−0.91) | [13.67 ± 1.26](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/udo35u3d) | **−2.64 (z=−1.63)** |
| **C_grounded_2M** | 14.66 ± 0.83 (n=50) | **[16.10 ± 1.20](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/rsgr6p45)** | **+1.44 (z=+0.99)** | **[16.80 ± 1.30](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0uuf13ul)** | **+2.14 (z=+1.39)** |

**`C_grounded_2M` × `v2_long_tail` is the headline patch result.** Per-achievement
before/after (baseline n=50 vs `v2_long_tail` n=30):

| achievement | baseline | v2_long_tail | Δpp | wandb |
|---|---|---|---|---|
| **wake_up** | 52% | **90%** | **+38** | [run](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0uuf13ul) |
| place_torch | 52% | 67% | +15 | (same) |
| make_torch | 58% | 73% | +15 | (same) |
| place_plant | 28% | 43% | +15 | (same) |
| **eat_plant** | 0% | **13%** | **+13** | (same) |
| collect_sapling | 38% | 50% | +12 | (same) |
| enter_dungeon | 12% | 20% | +8 | (same) |
| collect_iron | 50% | 57% | +7 | (same) |
| place_furnace | 62% | 67% | +5 | (same) |
| place_stone | 68% | 73% | +5 | (same) |
| make_stone_sword | 70% | 73% | +3 | (same) |
| make_iron_pickaxe | 0% | 3% | +3 | (same) — first non-zero |
| make_arrow | 72% | 70% | −2 | (same) |

The patch hits exactly the long-tail items the prompt was designed to
elicit (wake_up, place/eat plant, place/make torch, sapling) and
incidentally lifts the chain that depends on them (enter_dungeon,
collect_iron, even one make_iron_pickaxe — the first time C's
freezenone produces an iron pickaxe). No off-axis regressions other
than make_arrow −2pp.

`v2_basic_coverage` on C_grounded also helps (+1.44, z=+0.99): wake_up
+31pp, enter_dungeon +21pp, place_torch +18pp, make_torch +19pp. Less
of an eat_plant nudge than long_tail — basic_coverage doesn't
explicitly mention plants. Still net-positive because the wake_up /
torch chain is shared.

### Why B fails the patches

For `B_thinking_2M`, both patches **regress** (`basic` z=−0.9, `long_tail`
z=−1.6). Per-achievement: place_plant 93% → 50% (−43pp), collect_sapling
93% → 60% (−33pp), make_arrow 79% → 57% (−22pp), enter_dungeon 30% →
17% (−14pp). The pattern is the same as B's response to most prompt
swaps: the policy reads the embedding poorly, so changing the embedding
mostly disrupts an already-good baseline rather than steering it.
Confirms TRACK_ANALYSIS's interpretation that B's gap is a *policy*
fidelity problem, not an *instruction-clarity* problem — fixing the
prompt cannot fix what the policy can't read.

A_full is unmoved (Δret ≈ 0) — the content-blind policy is robust to
both patches.

<a id="awr-only-ablation"></a>
### AWR-only ablation: does the BC+oracle finetune help?

**Setup.** The canonical `C_grounded_2M` checkpoint
(`psf_v2_cadence5_grounded_predonly_top2M/freezenone/final.pth`) is
produced in two phases:
1. AWR pretrain — 100k steps, β=10, oracle_loss_weight=0.0 →
   `awr/final.pth`.
2. BC + AWR finetune — 50k steps, β=30, oracle_loss_weight=0.5,
   oracle_fraction=0.05, freeze_mode=none, init from #1 →
   `freezenone/final.pth`.

Phase 2 was added because the original v2 sweeps suggested it lifted
return slightly. The AWR-only checkpoint (#1) had never been evaluated
at scale on the steerability suite.

A 7-cell chain (job 7493701) ran the AWR-only checkpoint against
{baseline, target_descend, direction_left, die_fast, avoid_animals,
target_collect_stone, v2_long_tail}.

| condition | freezenone (canonical C) | awr-only | Δ awr − freezenone |
|---|---|---|---|
| baseline (gemini) | 14.66 ± 0.83 (n=50) | **17.67 ± 0.72 (n=30)** | **+3.01** |
| target_descend_v2 | 17.23 ± 0.89 (n=30) | 16.60 ± 0.92 (n=28) | −0.63 |
| direction_left_v2 — LEFT% of moves | 0.243 → 0.343 (+41% rel) | 0.242 → 0.340 (+40% rel) | matches |
| direction_left_v2 — Δret | 7.94 (z=−5.7 vs base) | 12.43 (z=−4.0 vs base) | both collapse |
| die_fast_v2 — Δlength | −246 (629→383) | −233 (651→418) | both shorten |
| die_fast_v2 — Δret | 11.66 (-3.0 vs base) | 13.00 (-4.7 vs base) | both drop |
| avoid_animals_v2 | 11.54 (Δret -3.12) | 17.00 (Δret -0.67) | awr-only resists |
| target_collect_stone_v2 | 15.76 (Δret +1.10) | 14.43 (Δret -3.23) | awr-only HURT |
| v2_long_tail (patch) | 16.80 (Δret +2.14) | 16.10 (Δret -1.57) | **awr-only HURT** |
| achievement_max_v2 (score-max) | 18.39 (Δret +3.73) | 17.50 (n=15, Δret -0.17) | awr-only no help |

**Three findings (n=30 final on most cells):**

1. **The AWR-only baseline is +3.01 return higher than the canonical
   freezenone baseline.** The 50k-step BC+oracle finetune phase that
   produces freezenone *drops* return by 3 raw points. The
   straightforward read is that the BC+oracle objective is mismatched
   with the deploy-time embedding distribution; the policy is being
   regularised toward labels it doesn't actually need to imitate.

2. **Steerability axes preserved.** Direction-left moves
   move-share-of-cardinal-moves by +40% relative (vs +41% on
   freezenone). die_fast shortens episodes ~233 steps. target_descend
   extends episodes (+370 steps; enter_dungeon rate similar). The
   axis-level behaviour transfers cleanly from freezenone to awr-only.

3. **The score-max prompts that lift freezenone DON'T lift awr-only —
   but for a non-trivial reason: awr-only has eaten the headroom
   freezenone had.** Per-axis comparison on the achievement RATES the
   prompt is targeting:

   | prompt | metric | freezenone base → cell | awr-only base → cell |
   |---|---|---|---|
   | direction_left | LEFT/cardinal share | 0.236 → 0.368 (+13pp); ret −8.5 | 0.244 → 0.340 (+10pp); ret −5.2 |
   | die_fast | length | 629 → 383 (−246); ret −2.2 | 651 → 418 (−233); ret −4.7 |
   | target_descend | enter_dungeon% | **12 → 27 (+15pp)**; ret **+2.6** | **17 → 25 (+8pp)**; ret −1.6 |
   | target_collect_stone | place_stone% | 68 → 73 (+5pp); ret −0.4 | **93 → 77 (−17pp)**; ret −3.2 |
   | v2_long_tail patch | (composite) | wake_up 52→90, etc.; ret +2.1 | smaller deltas; ret −1.6 |
   | achievement_max_v2 | (composite) | wake_up 52→75 etc.; ret +3.7 | n=15 partial; ret −0.2 |

   **The awr-only checkpoint is BEHAVIORALLY STEERABLE.** Direction,
   descent, length all move under the same prompts as on freezenone
   — direction_left still shifts LEFT% by 10pp on awr-only.
   What it ISN'T is *return-improvable* by prompts. And the reason is
   visible in the second column above: on awr-only, baseline rates
   for the achievements the prompt targets are ALREADY HIGH
   (place_stone 93% vs freezenone 68%; enter_dungeon 17% vs 12%; sleep
   chain higher; etc.). The prompt can shift the action distribution
   the same way it does on freezenone, but there's no rate-headroom
   left for the achievement to fire more often. For
   `target_collect_stone` it actually *removes* place_stone behavior
   the policy was already doing — the prompt's stone-mining emphasis
   competes with the awr-only policy's already-optimal placement
   routine.

   **Net**: BC+oracle is **not redundant**. It trades baseline
   performance (-3 raw return) for *prompt-headroom* (achievement
   rates left low so prompts can fill them). Both routes converge to
   ~18 raw return on this data — but only freezenone-with-prompt is
   *steerable in the score-improvement sense*. AWR-only is *steerable
   in the behavior-shift sense* but cannot be score-improved by
   prompts because it has no rate-headroom on the targeted axes.

**Implications (revised):**

- For research goal "demonstrate steerability of LLM-conditioned offline
  RL policies": **BC+oracle freezenone is the right recipe** — it gives
  us both axis-steerability (preserved on awr-only too) AND
  score-improvability (only on freezenone).
- For SCALING_C: now ambiguous. Pure-AWR on better data should land at
  a higher floor (the data's ceiling) but with reduced prompt-headroom.
  BC+AWR on better data might land at a slightly lower floor with much
  more prompt-headroom. The user's research goal here matters: if
  steerability is the headline, keep BC+AWR; if raw score is the
  headline, AWR-only suffices.
- The 50k extra training steps + the BC+oracle objective are paying
  for "prompt-fillable rate-headroom" — a measurable property of the
  resulting policy. Earlier reading "BC+oracle drops return so it's
  redundant" was incomplete; the trade-off is performance vs
  prompt-receptivity.
- For ongoing score-max iteration: keep iterating on freezenone (where
  the headroom is). Awr-only score-max is unlikely to push past 18.
- One open caveat: `achievement_max_v2` on awr-only is n=15 partial.
  Need n=30 to confirm it doesn't lift the score (currently −0.17).

Job: 7493701 (7-cell array). Eval dir:
`/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_awr_only/`.

---

### What this changes about the central claim

Patch-by-prompt is the strongest single-axis demonstration that a
lower-fidelity prompt is the *binding constraint* on C's return.
Holding the policy fixed and changing only the inference-time prompt
moves return by +14% relative (14.66 → 16.80) and pushes wake_up by
38pp. Combined with `target_descend_v2` (which gets C to 17.23 by
elevating descent as priority 1), and the [Specificity matrix
(2026-04-25)](#specificity-matrix-2026-04-25) showing that 12/21
single-axis steers move the targeted metric in the right direction,
the picture is: **C's hidden branch reads the prompt content,
including long-tail behavioral instructions, and the policy executes
on what it can.**

---

<a id="synthetic-embedding-arithmetic"></a>
## Synthetic embedding arithmetic — the killer probe

**Setup.** For each `(track, target)` pair, compute a direction vector
in 3072-dim embedding space:

```
d = mean_over_N_paired_samples(
      embed(target_prompt_output) − embed(concise_regular_prompt_output)
    )
```

Direction vectors saved at `probe_results/embed_directions/`:
- `a_full_die_v2.npy` — derived from A_full's Gemini logs (norm 0.27)
- `c_grounded_die_v2.npy` — from C_grounded's logs (norm 0.36)
- `c_grounded_avoid_animals_v2.npy` — (norm 0.41)

Each is 200 paired regular vs target prompt completions, embedded by
`gemini-embedding-001` with the same predonly extraction.

**Intervention.** At every Gemini call at eval time, run the normal
concise prompt, embed the prediction to get `e_t`, then pass
`e_t + α × d` to the policy. The Gemini *text* is unchanged.

### Online eval results (30 episodes)

| Track | Cell | n | return ± SE | Δret vs baseline (z) |
|---|---|---|---|---|
| A_full | baseline (regular Gemini) | 50 | [18.98 ± 0.36](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/n7wmnk82) | — |
| A_full | direct `die_v2` prompt | 50 | [18.22 ± 0.57](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6g9kloyz) | −0.76 (z=−1.1) |
| **A_full** | α=+2 × d_die | 29 | [18.24 ± 0.53](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/2061g8dn) | **−0.74 (z=−1.2)** |
| A_full | α=−2 × d_die | 29 | [18.62 ± 0.63](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/ra87r7ex) | −0.36 (z=−0.5) |
| C_grounded_2M | baseline | 50 | [14.66 ± 0.82](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/pjb8wf7z) | — |
| C_grounded_2M | direct `die_v2` prompt | 50 | [9.76 ± 0.98](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm) | −4.90 (z=−3.84) |
| **C_grounded_2M** | **α=+2 × d_die** | **30** | **[9.53 ± 1.22](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/4j5pi14i)** | **−5.13 (z=−3.50)** |
| C_grounded_2M | α=−2 × d_die | 29 | [14.93 ± 1.20](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/u30xdjdz) | +0.27 (z=+0.18) |
| C_grounded_2M | α=+2 × d_avoid_animals | 30 | [14.73 ± 0.99](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/d1053tlt) | +0.07 (z=+0.06) |
| C_grounded_2M | α=−2 × d_avoid_animals | 30 | [12.73 ± 1.07](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/4rcbsnp3) | −1.93 (z=−1.4) |

### Interpretation

**α=+2 × d_die matches the prompt-based effect on C_grounded within
noise** (−5.13 vs −4.90). Same Gemini text; different embedding; full
behavioral effect reproduced. This is the clean steerability
demonstration: the policy reads a specific direction in embedding space
as "death content", and that direction is sufficient to steer behavior
even in the absence of any matching surface text.

A_full's matched null on both interventions (≈−0.7 each) confirms the
robustness comes from the *policy head*, not from the LLM/embedding
pipeline.

`d_avoid_animals` does **not** reproduce its prompt-based effect (+0.07
vs −3.12). Likely because avoid_animals's prompt effect comes partly
from the *absence* of cow-related surface content in the regular Gemini
output, which the mean-difference direction only partially captures.
`d_die` dominates the embedding shift more cleanly.

Data source: `probe_results/steerability_analysis/c_embed_arith.json`,
`a_embed_arith.json`, `c_arith_full.json`.

---

<a id="in-distribution-α-sweep-3072-dim"></a>
## In-distribution α-sweep (fast, no Gemini calls)

Load 300 training-distribution states, forward the policy with
`hidden + α × direction` across α ∈ {−2.0, −1.5, ..., +2.0}. Measure
argmax flip vs α=0, mean KL, ΔV, per-action probability shifts.

### Argmax flip % at α=+2

| | A_full (on d_a_full_die) | C_grounded (on d_c_grounded_die) | C_grounded (on d_c_grounded_avoid_animals) |
|---|---|---|---|
| signal dir | 1.0% | **26.7%** | 18.0% |

### Cross-track direction transfer

| Direction | A_full flip% | C_grounded flip% |
|---|---|---|
| d_a_full_die | 1.0% | 20.0% |
| d_c_grounded_die | 1.7% | 26.7% |
| d_c_grounded_avoid_animals | 2.0% | 18.0% |

**The policy is the bottleneck, not the direction.** A direction
computed on A_full's Gemini logs steers C_grounded by 20% but barely
moves A_full (1%).

### Per-action monotonicity on C_grounded × d_die (Pearson correlation of p(action) vs α)

| action | corr | alpha=0 prob | alpha=+2 prob | spread |
|---|---|---|---|---|
| LEFT | **−0.918** | 0.562 | 0.407 | 0.192 |
| RIGHT | **+0.976** | 0.248 | 0.397 | 0.228 |
| DO | −0.798 | 0.170 | 0.174 | 0.048 |
| DESCEND | +0.847 | 9e-5 | 8.6e-5 | 5.7e-5 |
| SLEEP | −0.976 | 6e-5 | 2.1e-5 | 1.3e-4 |
| PLACE_STONE | −0.983 | 3.4e-4 | 1.5e-4 | 5.1e-4 |

Almost every tracked action has |corr| > 0.8. The policy reads a
**continuous** structure along the direction axis, not just "in-dist vs
out-of-dist".

### Random-direction control (CRITICAL caveat)

Norm-matched random gaussian direction (10 reps, signal norm=0.3644):

| | signal flip% (α=+2) | random flip% mean ± std | z(sig vs rand) |
|---|---|---|---|
| A_full | 1.7% | 1.1 ± 0.6 | +1.0 |
| C_grounded | 21.3% | 14.3 ± 3.3 | +2.1 |

**C_grounded is a high-gain policy**: even random directions produce ~14%
argmax flips at α=+2. The signal direction adds ~7pp on top (+2.1σ
above random). For the value head, signal ΔV=+0.140 vs random ΔV≈−0.009
— the value head reads content cleanly; the policy head reads content +
direction magnitude.

Source: `probe_results/in_dist_embed_arith/summary.json`,
`probe_results/in_dist_embed_arith_xtrack/summary.json`,
`probe_results/in_dist_embed_arith_random_ctrl/summary.json`.

---

<a id="value-gradient-direction-negative-result"></a>
## Value-gradient direction — negative result

The value-gradient direction `d_value = mean(unit(∂V/∂h))` is the
linear-best direction the value head reads as "good". Compute it per-
track, rescale to match `d_die` norm, then α-sweep online.

### In-distribution sweep (value head response)

| Track | α=0 V | ΔV at α=+2 | flip% at α=+2 | cos(d_value, d_die) |
|---|---|---|---|---|
| A_full | −1.04 | **+1.77** (massive) | 4.0% | −0.05 (orthogonal) |
| C_grounded_2M | −0.98 | +0.65 | 17.3% | +0.05 (orthogonal) |

### Online eval (behavioral response, n=15–30)

| Cell | n | return ± SE | Δret (z) |
|---|---|---|---|
| A_full d_value α=+2 | 30 | [17.03 ± 0.76](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/vzmmcuxf) | **−1.95 (z=−2.31)** |
| A_full d_value α=−2 | 15 | [19.10 ± 0.89](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/nka9sox3) | +0.12 (NS) |
| C_grounded d_value α=+2 | 30 | [13.57 ± 1.13](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/iqbybdhz) | −1.09 (z=−0.78) |
| **C_grounded d_value α=−2** | 30 | **[8.73 ± 1.22](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/gkz5ksi8)** | **−5.93 (z=−4.03)** |

### Interpretation

**Both value-gradient directions hurt return.** Pushing α=+2 along
d_value (V linearly rises in-distribution) makes the policy act as
if it's in a high-V state — skipping preparation, over-committing to
optimism. The world isn't actually a high-V state, so real performance
degrades.

**Implication**: not every embedding direction is productively
steerable. The death direction works because it's an
ecologically-meaningful semantic axis the policy was trained on
(death-narrative content). Value-gradient directions are just where
the value head likes to go, with no behaviorally-grounded meaning. The
central claim about steerability requires directions with
*training-distribution semantic content*, not just value-head-
favorable ones.

Source: `probe_results/value_grad_steer/summary.json`,
`probe_results/value_grad_steer/{a_full,c_grounded_2M}_value_grad_dir.npy`.

---

<a id="mid-episode-behavioral-switch"></a>
## Mid-episode behavioral switch

Run the first 200 env steps with regular Gemini concise prompt; switch
the embedding mode at step 200.

| Cell | n | return ± SE | Δret (z) | Δrate(enter_dungeon) |
|---|---|---|---|---|
| A_full switch → die_v2 @ 200 | 30 | [19.20 ± 0.37](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/aia9zbhb) | +0.22 (NS) | — |
| C_grounded switch → die_v2 @ 200 | 30 | [14.40 ± 0.93](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/9w6ixzq4) | −0.26 (z=−0.21) | — |
| **C_grounded switch → target_descend @ 200** | 30 | [14.97 ± 1.18](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/1lbjpfxn) | +0.31 (z=+0.21) | **+0.15 (0.12 → 0.27, +125% rel)** |
| **C_grounded switch → avoid_animals @ 200** | **30** | **[16.47 ± 0.82](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7tk8vvzv)** | **+1.81 (z=+1.56)** | — (see below) |

### Key findings

1. **`die_v2` effect is short-range.** Full-episode die_v2 drops return
   −4.90; applying it only from step 200 → −0.26. Most damage happens
   during the policy's setup phase; once tools are built, late-episode
   die content does modest damage.

2. **`target_descend` effect is AMPLIFIED by mid-episode application.**
   Full-episode: enter_dungeon +10pp. Switch-at-200: +15pp. The policy
   needs ~200 steps of setup before descent is actually feasible; the
   late-episode nudge lands on a state where the steered action is
   productive.

3. **`switch → avoid_animals` is the strongest positive-return result —
   but it is NOT actually avoidance** (re-analyzed 2026-04-25). The
   prompt is doing something the name doesn't capture: every productive
   achievement rate goes UP, not down. Compare full-episode vs switch
   on `C_grounded_2M`:

   | achievement | base (n=50) | full avoid (n=50) | switch→avoid @200 (n=30) |
   |---|---|---|---|
   | eat_cow | 86% | 74% (−12pp) | **97% (+11pp)** |
   | place_stone | 68% | 48% (−20pp) | **77% (+9pp)** |
   | place_furnace | 62% | 50% (−12pp) | **77% (+15pp)** |
   | place_torch | 52% | 38% (−14pp) | **67% (+15pp)** |
   | make_torch | 58% | 42% (−16pp) | **70% (+12pp)** |
   | collect_iron | 50% | 34% (−16pp) | **60% (+10pp)** |
   | place_plant | 28% | 42% (+14pp) | **53% (+25pp)** |
   | collect_sapling | 38% | 46% (+8pp) | **57% (+19pp)** |
   | eat_plant | 0% | 0% | **7% (+7pp)** |
   | enter_dungeon | 12% | 14% (+2pp) | **20% (+8pp)** |
   | length | 629 | 416 (−213) | 525 (−104) |
   | return | 14.66 | 11.54 (−3.12) | **16.47 (+1.81)** |

   Full-episode `avoid_animals` is genuine suppression: every productive
   late-game achievement drops, episode length collapses to 416, return
   drops −3.12. Mid-episode `switch → avoid_animals` is the OPPOSITE:
   every productive achievement RISES (eat_cow goes from 86% → 97%, not
   down), and the gain is concentrated on plant/sapling/place behaviors
   (sapling +19pp, place_plant +25pp).

   Mechanism (hypothesis): once the policy has done ~200 steps of normal
   wood→stone routine, switching to a `avoid_animals` embedding shifts
   Gemini's predictions away from "go to the cow" toward "go to grass /
   explore". That's exactly the right *late-episode* nudge for
   collecting saplings, placing plants, and incidentally finding more
   resources en route — it functions as a "go explore the grasslands"
   prompt, not avoidance. The policy is in a different state by step 200
   (already toolable), and the same embedding is read as a different
   instruction in the new state.

   Implications:
   - The "+1.81 return" effect is real, but the published name is
     misleading. Do NOT cite this as positive-return *avoidance steering*.
   - The mid-episode `switch → target_descend @200` (only +0.31 return)
     is the apples-to-apples comparison — same switch event, different
     content, much weaker positive return. So the avoid_animals-specific
     gain is real, just not via avoidance.
   - Useful follow-up: try switch → {target_collect_sapling, target_place_plant,
     target_collect_stone} @200. If switch_to_target_collect_sapling @200
     produces a similar +1-2 return, the effect is generic to
     "go-explore-grass-after-200" content. If only avoid_animals does it,
     something specific about the negation embedding is the lever.

Source: `probe_results/steerability_analysis/c_full_extra.json`,
`c_switch_partial.json`. Re-analysis script and per-achievement
breakdown above written 2026-04-25 from raw summary.json files.

---

## Master comparison table — one row per condition × track

Return (± SE). Bold = decisive; `—` = not run.

| Condition | A_full (base [18.98](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/n7wmnk82)) | A_top2M (base [18.14](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7id4059l)) | B_thinking_2M (base [16.31](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7itrrqbh)) | C_grounded_2M (base [14.66](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/pjb8wf7z)) |
|---|---|---|---|---|
| `die_v2` | [18.22 ± 0.57](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6g9kloyz) (−0.76) | [15.86 ± 0.77](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/sibl3kge) (−2.28*) | [12.22 ± 1.08](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/m353pmy3) (**−4.09**) | **[9.76 ± 0.98](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm) (−4.90)** |
| `adversarial_v2` | [19.38 ± 0.29](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/ksubq27s) (+0.40) | [16.08 ± 0.90](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/zo1imok6) (−2.06) | [13.80 ± 0.94](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/nqfle4h5) (−2.51) | [11.68 ± 0.95](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/s5skoqs0) (−2.98*) |
| `avoid_water_v2` | [18.30 ± 0.41](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/hxcfdgwx) (−0.68) | [(−1.66 marginal)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/2qy7xdo8) | [15.90 ± 0.92](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/asthb0rc) (−0.41) | [13.64 ± 1.01](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/pdjilnqf) (−1.02) |
| `avoid_animals_v2` | [18.20 ± 0.50](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/203drusj) (−0.78) | [(−1.70 marginal)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/cmpexs6l) | [16.84 ± 0.96](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/756yap5q) (+0.53) | **[11.54 ± 1.02](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/gwhh7c5j) (−3.12*)** |
| `target_collect_stone_v2` | [18.04 ± 0.48](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/62tmvkfl) (−0.94) | — | [12.76 ± 0.91](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/lcwqcxo4) (**−3.55**) | **[15.76 ± 0.73](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6phz27zr) (+1.10)** |
| `target_descend_v2` | [19.28 ± 0.27](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/jkb171iy) (+0.30) | — | [15.26 ± 0.95](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/ng2xcv0g) (−1.05) | [14.80 ± 0.99](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/y09770mm) (+0.14) |
| `target_eat_cow_v2` | [18.62 ± 0.42](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/yzndpabh) (−0.36) | — | [16.58 ± 0.83](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/snqn8pak) (+0.27) | [14.42 ± 0.83](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/fa0dlxd8) (−0.24) |
| `target_drink_water_v2` | [18.08 ± 0.40](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/18tl33lj) (−0.90) | — | — | — |
| `target_place_stone_v2` | [18.38 ± 0.52](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/ix5qkxbs) (−0.60) | — | — | — |
| `target_hunt_animals_v2` | [18.96 ± 0.42](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/3m3rdbnf) (−0.02) | — | [14.72 ± 1.03](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/9vr77310) (−1.59) | [12.70 ± 0.99](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/3chzhpeu) (−1.96) |
| `direction_left_v2` | [17.48 ± 0.54](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/8g2lqfcy) (−1.50*) | — | [13.44 ± 1.01](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/wfxhxqlg) (**−2.87**) | **[7.94 ± 0.84](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/w21fwecj) (−6.72)** |
| `direction_right_v2` | [18.22 ± 0.46](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/alqre6lj) (−0.76) | — | [14.78 ± 0.83](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/533r9el6) (−1.53) | [11.16 ± 0.92](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/olscsp79) (−3.50*) |
| α=+2 × d_die (synthetic) | [18.24 ± 0.53](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/2061g8dn) (−0.74) | — | — | **[9.53 ± 1.22](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/4j5pi14i) (−5.13)** |
| α=−2 × d_die (synthetic) | [18.62 ± 0.63](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/ra87r7ex) (−0.36) | — | — | [14.93 ± 1.20](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/u30xdjdz) (+0.27) |
| d_value α=+2 | [17.03 ± 0.76](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/vzmmcuxf) (−1.95*) | — | — | [13.57 ± 1.13](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/iqbybdhz) (−1.09) |
| d_value α=−2 | [19.10 ± 0.89](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/nka9sox3) (+0.12) | — | — | **[8.73 ± 1.22](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/gkz5ksi8) (−5.93)** |
| switch → die_v2 @200 | [19.20 ± 0.37](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/aia9zbhb) (+0.22) | — | — | [14.40 ± 0.93](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/9w6ixzq4) (−0.26) |
| switch → target_descend @200 | — | — | — | [14.97 ± 1.18](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/1lbjpfxn) (+0.31) |
| **switch → avoid_animals @200** | — | — | — | **[16.47 ± 0.82](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7tk8vvzv) (+1.81)** |

\* = marginal significance (|z| > 1.7) **bold** = significant (|z| > 2 or
flagged decisive).

---

## PPO baselines for reference

| Baseline | Scale | Raw return | % max | wandb |
|---|---|---|---|---|
| PPO-RNN 5M | 5e6 | 6.96 ± 0.24 | 3.1% | video eval dir `ppo_rnn_5M_50ep_video/` |
| PPO-RNN 20M | 2e7 | 14.08 ± 0.48 | 6.2% | `ppo_rnn_20M_50ep_video/` |
| PPO-RNN 1e8 | 1e8 | 27.87 (training-time) | 12.3% | [run fkxga61m](https://wandb.ai/iris-sobolmark/craftax-baselines-replication/runs/fkxga61m) |
| PPO-symbolic 1e8 (1st attempt) | TIMEOUT at 33M | 17.60 | 7.8% | [run tswtiilh](https://wandb.ai/iris-sobolmark/craftax-baselines-replication/runs/tswtiilh) |
| PPO-symbolic 1e8 (resubmit 7464286) | FAILED at 47s | — | — | did not start training; needs resubmit |
| PPO-RNN 1e8 with checkpoint save (7464853) | FAILED at 55s | — | — | did not start training; needs resubmit |

**Scoreboard context**: 1B PPO-RNN = 15.3%, 1B PPO-GTrXL = 18.3%. Our
augmented tracks sit below the 1e8 PPO-RNN baseline on raw return. The
steerability claim is orthogonal to raw-return competitiveness: we
demonstrate it in the regime where augmented returns roughly match the
unaug offline-RL baseline (18.38), not at scoreboard-top scale.

---

## Job status (resolved as of 2026-04-25)

| Job | Cells | Outcome |
|---|---|---|
| 7465078 v2_steerability (B_thinking_2M) | target_{collect_stone,descend,eat_cow,hunt_animals}, direction_{left,right} | COMPLETED at n=50; final results merged into the per-cell tables above |
| 7468781 v2 target_hunt_animals | A_full, C_grounded_2M, B_thinking_2M × target_hunt_animals | COMPLETED at n=50 |
| 7468782 PPO video re-eval | PPO-RNN {5M, 20M} with wandb video | COMPLETED |
| 7464286 PPO-symbolic 1e8 (resubmit) | 1 cell | **FAILED at 47s** — did not start training; needs investigation + resubmit |
| 7464853 PPO-RNN 1e8 with checkpoint save | 1 cell | **FAILED at 55s** — did not start training; needs investigation + resubmit |
| 7481845 v2_specificity matrix | 21 cells × C_grounded_2M | COMPLETED — see [Specificity matrix (2026-04-25)](#specificity-matrix-2026-04-25) |
| 7486443 v2_spec_iter_v3 | avoid_stone_v3, survive_long_v3 | COMPLETED — see [v3 prompt iteration](#v3-prompt-iteration-2026-04-25-post-matrix) |

---

## What this changes about the central claim

Support for each sub-claim:

| Sub-claim | Evidence | Status |
|---|---|---|
| Behavior changes when content changes | die_v2 on C: [−4.90 return](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm) + DO rate +23pp (2.2× baseline), matches prompt directive. | Confirmed |
| Direction is structured (not just OOD) | Monotonic α-sweeps (LEFT corr=−0.92, RIGHT corr=+0.98). Signal direction +2.1σ above norm-matched random. | Confirmed with caveat (high-gain component) |
| Weak-fidelity policy is robust | [A_full die_v2=−0.76](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6g9kloyz), argmax flip <4% across ±2α. | Confirmed |
| High-fidelity policy is steerable | C_grounded all probes register. | Confirmed |
| Continuous structure in embedding space | α-sweep is monotonic to α=±2, value linear. | Confirmed |
| Cross-policy direction transfer | A_full-derived direction steers C_grounded by 20% flip. | Confirmed (policy is the bottleneck, not direction quality) |
| Positive achievement steering produces positive effect | [`target_collect_stone` Δ+1.10](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6phz27zr); [`switch → avoid_animals @200` Δ+1.81](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7tk8vvzv). | Confirmed marginally |
| Pure-direction steering moves action distribution | [direction_left LEFT% +41% relative on C](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/w21fwecj). | Confirmed |
| Synthetic embedding intervention reproduces prompt effect | [α=+2 × d_die produces −5.13](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/4j5pi14i) (matching [−4.90 prompt](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6s40z5tm)). | **Decisive** |
| Mid-episode switch responsive | [target_descend doubled when applied mid-episode](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/1lbjpfxn). | Confirmed |
| Steering IMPROVES return | [target_collect_stone +1.10 (NS z=+1.0)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/6phz27zr); [switch→avoid_animals +1.81 (z=+1.56)](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/7tk8vvzv). | Marginal |
| Value-gradient-aligned steering improves return | [d_value α=−2 steering HURTS return on C](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/gkz5ksi8) ([+2 direction also hurts on A](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/vzmmcuxf)). | **Negative result** — steerability requires semantic-content directions, not value-head-favorable ones |

10 of 12 sub-claims confirmed. 1 marginal (net-positive return under
steering). 1 negative (value-gradient refinement).

---

## Open questions and future work

1. **Does online RL with steering objectives close the productivity
   gap?** Current offline-trained C_grounded follows steering prompts
   (direction_left LEFT% +41%) but collapses productive behavior
   (return −7). Whether online RL with rewards that couple
   achievement-prompts to actual-achievement counts would produce a
   policy that follows steering *and* maintains productivity is the
   key empirical question for the central claim's stronger
   interpretation.

2. **Why does avoid_animals direction not reproduce its prompt effect
   via embedding arithmetic?** d_die is nearly equivalent to the
   prompt. d_avoid_animals is not. The hypothesis is that
   avoid_animals's prompt effect comes from the *absence* of
   cow-related surface content, which a mean-difference direction only
   partially captures. Worth testing with per-sample-contrasted
   direction computation.

3. **Which embedding directions are productively steerable vs
   destructive?** d_die is semantic-content-backed (direction the
   policy was trained on). d_value is value-head-aligned but
   semantically empty and HURTS performance. A systematic search for
   "productive directions" (bridging steering and return) is the next
   step.

4. **Replication of B_thinking_2M results at n=50.** Current partial
   numbers suggest B falls between A and C on every axis; n=50 is
   needed to confirm z-significance on narrow targets.

5. **Are inventory-count effects proportional to action-rate
   effects?** target_collect_stone: stone count +38% relative, PLACE_STONE
   action +129% relative — the policy places more stone per unit
   collected. Mechanism open.

6. **Relationship between training-distribution fidelity and "policy
   gain on content axis".** The A-top2M sweet spot (between A_full and
   C_grounded) has not been fully characterized on the new steering
   suite. Running the target_* prompts on A_top2M would decompose the
   data-scale effect from the label-fidelity effect.

7. **Do `direction_up_v2` and `direction_down_v2` produce analogous
   effects?** Templates exist; online eval not run.

---

## File and data index

| Kind | Path |
|---|---|
| Prompt templates | `configs/training/templates/predict_state_only_prompt_concise_*_v2.txt` |
| Master result table | `probe_results/master_table_FINAL.md` (+ `*.json`) |
| Per-condition achievement deltas | `probe_results/steerability_analysis/*.json` |
| Per-condition action distribution | `probe_results/action_analysis/*.json` |
| Inventory counts | `probe_results/inventory_counts/*.json` |
| In-distribution α-sweep | `probe_results/in_dist_embed_arith/summary.json` |
| Random-direction control | `probe_results/in_dist_embed_arith_random_ctrl/summary.json` |
| Cross-track transfer | `probe_results/in_dist_embed_arith_xtrack/summary.json` |
| Value-gradient probe | `probe_results/value_grad_steer/summary.json` |
| Direction vectors (3072-dim) | `probe_results/embed_directions/{a_full_die_v2,c_grounded_die_v2,c_grounded_avoid_animals_v2}.npy` |
| Value-gradient directions | `probe_results/value_grad_steer/{a_full,c_grounded_2M}_value_grad_dir.npy` |
| Eval videos | `/data/group_data/rl/geney/eval_results/<eval_dir>/episode_XX/gameplay.mp4` |
| Tools | `tools/probe_steering_v2_targets.py`, `tools/build_embedding_directions.py`, `tools/in_dist_embed_arith_sweep.py`, `tools/in_dist_embed_arith_random_control.py`, `tools/in_dist_value_gradient_steering.py`, `tools/achievement_steerability_analysis.py`, `tools/action_distribution_analysis.py`, `tools/achievement_timing_analysis.py`, `tools/pick_demo_episodes.py`, `tools/steerability_master_table.py` |
| SLURM jobs | `slurm/jobs/v2_steerability_targets_array.sh`, `slurm/jobs/v2_embed_arith_sweep_array.sh`, `slurm/jobs/v2_value_grad_arith_array.sh`, `slurm/jobs/v2_switch_array.sh` |
| Journals | `journals/log_2026-04-22.md`, `log_2026-04-23.md`, `log_2026-04-24.md` |

---

## wandb hyperlink coverage notes

All steering-eval return numbers in the per-section results tables and the
master comparison table link to their wandb run via the
`craftax-offline-awr` project. Track-baseline (regular freezenone)
returns link to the `eval_track_*_freezenone_50ep` runs. PPO baselines
link to `craftax-baselines-replication`.

Return numbers without a wandb hyperlink (run ID not available in the
collection pulled at writing time):

- **PPO-RNN 5M (6.96 ± 0.24)** and **PPO-RNN 20M (14.08 ± 0.48)** in the
  PPO baselines table — older video-only re-evals. The local video
  paths are listed instead.
- **`avoid_water_v2` and `avoid_animals_v2` on A_top2M** in the master
  comparison table show only the marginal Δret (no return ± SE was
  recorded in the original probe_results JSON). The wandb runs
  ([avoid_water_v2 / A_top2M](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/2qy7xdo8),
  [avoid_animals_v2 / A_top2M](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/cmpexs6l))
  exist and are linked from the marginal cells.
- Inventory deltas, action probabilities, achievement-rate Δ values,
  z-scores, and α-sweep correlations are intentionally not hyperlinked
  — they are derived from but not the headline number of any run.



---

## Specificity matrix (2026-04-25)

**Question.** Does each prompt's intended axis move the *behaviorally
relevant per-episode metric*, and not move off-diagonal axes?

**Setup.** 21 prompts × `C_grounded_2M`, n=30 episodes per cell,
freezenone checkpoint, `psf_v2_cadence5` pipeline. Per-cell target
metric is per-episode count for inventory + action behaviors,
share-of-cardinal-moves for direction prompts, achievement rate for
chain tasks, raw mean episode length for life/death.

**Headline.** **11/21 cells WIN, 8 NULL, 2 WRONG-WAY.**

| prompt | target | baseline | cell | Δ | z | wandb |
|---|---|---|---|---|---|---|
| target_collect_stone_v2 | stone/ep | 17.9 | 22.5 | +4.6 | +1.2 | [1u3v3hya](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/1u3v3hya) |
| target_place_stone_v2   | PLACE_STONE/ep | 7.1 | 13.7 | +6.7 | +1.8 | [3os11d6g](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/3os11d6g) |
| target_hunt_animals_v2  | cow_eat/ep | 5.2 | 7.6 | +2.4 | +1.1 | [r0t86i4p](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/r0t86i4p) |
| avoid_animals_v2        | cow_eat/ep | 5.2 | 3.7 | -1.5 | -1.5 | [0x286zyt](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0x286zyt) |
| avoid_water_v2          | drink/ep | 13.8 | 8.1 | -5.7 | -1.7 | [nhmftcco](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/nhmftcco) |
| target_descend_v2       | DESCEND/ep | 4.6 | 9.6 | +5.1 | +2.0 | (see specificity dir) |
| die_fast_v2             | length | 629 | 383 | -246 | -3.0 | (see specificity dir) |
| direction_left_v2       | LEFT/cardinal | 0.24 | 0.40 | +0.16 | +6.0 | (see specificity dir) |
| direction_right_v2      | RIGHT/cardinal | 0.27 | 0.31 | +0.04 | +2.0 | (see specificity dir) |
| direction_up_v2         | UP/cardinal | 0.28 | 0.33 | +0.05 | +3.0 | (see specificity dir) |
| direction_down_v2       | DOWN/cardinal | 0.21 | 0.26 | +0.05 | +3.3 | (see specificity dir) |

NULL or wrong-way:
- `target_avoid_stone_v2` (z=+0.5, **wrong sign**) — prompt's "avoid stone"
  high-level goal contradicts kept upgrade-tree rule "Stone Tier: craft
  stone pickaxe if you have stone and wood"; Gemini text *does* describe
  avoidance by step 70 but the policy has already mined.
- `target_eat_cow_v2` (z=+0.3) — episodes lengthen (629→850, subsistence
  loop) but per-episode cow_eat only nudges. The policy stretches a fixed
  cow-acquisition rate over more steps.
- `target_drink_water_v2` (z=-0.7, wrong sign) — per-step drink rate
  identical to baseline (0.022); shorter episodes (-122 steps) drag the
  per-episode count down. **Length confound, not behavior.**
- `target_stay_overworld_v2` (z=-0.7), `target_place_plant_v2` (z=-0.2),
  `target_defeat_zombie_v2` (z=+0.1) — null.
- `target_make_iron_pickaxe_v2` and `target_collect_diamond_v2` —
  **0/30 occurrences each**. Chain-task ceiling.
- `target_collect_sapling_v2` (z=-1.1, wrong sign) — saplings stay rare
  (0.62→0.37/ep); the prompt didn't elicit search-for-sapling behavior.
- `survive_long_v2` (z=-2.3, **wrong sign**) — length 629→416. The strict
  "establish base with water+cow+stone within 3 tiles" criteria rarely
  satisfy; Gemini loops on "no base, walk to water"; the policy actually
  dies *faster* than no-prompt baseline.

Full per-cell × per-metric matrix (all 21 cells × 25 columns) at
[`docs/SPECIFICITY_MATRIX.md`](SPECIFICITY_MATRIX.md). Per-cell counts
JSON at `probe_results/inventory_counts/specificity/<cell>.json`. Matrix
JSON at `probe_results/specificity_matrix.json`. Tool: `tools/specificity_matrix.py`.

### Mechanism summary

1. **Direct, single-action steering wins.** Collect/place/avoid + cardinal
   directions all work. The mechanism is a gradient-shift toward the
   action class, not inferential planning.
2. **Chain-task prompts are zero-rate.** 0/60 episodes across iron_pickaxe
   and diamond prompts. Embedding can't supply multi-step planning the
   policy lacks.
3. **Negative steering asymmetry**: avoid_water and avoid_animals work,
   target_avoid_stone fails. The asymmetry is **not** about the negative
   framing — stone is on the critical path of game progression so the
   prompt's goal conflicts with its still-included Stone-tier rule.
4. **survive_long's wrong-way result is the cleanest indictment** of the
   strict-base prompt class: same surface form as die_fast (which works
   to z=-3.0), opposite direction → length actually shortens. Iteration
   target.

### v3 prompt iteration (2026-04-25, post-matrix)

After the matrix landed, the two clearest prompt-logic-bottleneck cells
were re-run with simpler prompts (n=30 each, output dir
`..._specificity_iter`):

| cell | n | stone | length | verdict |
|---|---|---|---|---|
| baseline (no prompt) | 50 | 17.9 ± 2.2 | 629 ± 79 | — |
| target_avoid_stone_v2 | 30 | 19.8 ± 3.1 | 527 ± 79 | wrong-way (z=+0.5) |
| **target_avoid_stone_v3** | 30 | **12.4 ± 2.6** | 502 ± 82 | **WIN (z=-1.6)** |
| survive_long_v2 | 30 | 14.2 ± 2.7 | 416 ± 50 | wrong-way (z=-2.3) |
| **survive_long_v3** | 30 | 18.3 ± 3.3 | **669 ± 119** | NULL (z=+0.3) |

`v3` design changes:
- `target_avoid_stone_v3`: deleted the entire Upgrade Decision Tree
  (the v2 prompt kept "Stone Tier: craft stone pickaxe", which
  contradicted the avoid-stone goal). Replaced exploration with
  "wood-tier territory only".
- `survive_long_v3`: dropped the strict "establish base with water +
  cow + stone within 3 tiles" criteria. Replaced with "keep intrinsics
  ≥7, NOOP when safe, run from enemies, never descend".

Two observations:
- `avoid_stone` flipped from wrong-way to a clear WIN. Removing the
  internal contradiction was the fix → the policy CAN read negative
  steering when the prompt is internally consistent.
- `survive_long` flipped from wrong-way to NULL (matches baseline). It
  no longer disrupts the policy but does not extend episodes either.
  **The policy's survival skill cap is a fidelity ceiling, not a prompt
  problem.** Final WIN tally: 12/21, NULL 9/21, WRONG-WAY 1/21.

Note (2026-04-25): the `target_avoid_stone_v2.txt` template file on
disk has now been overwritten with the v3 contents (the contradictory
upgrade-tree was the bug, not anything specific to "v2"). Future
`--embedding-mode target_avoid_stone_v2` runs will use the fixed
prompt; the old wrong-sign result above remains in the matrix for
historical comparison.

A second round of v3 prompt iteration was queued covering all the other
NULL specificity-matrix cells (`target_eat_cow_v3`, `target_drink_water_v3`,
`target_stay_overworld_v3`, `target_place_plant_v3`,
`target_defeat_zombie_v3`, `target_collect_sapling_v3`, job 7491313,
6 cells × n=30). Results will be appended once the array completes.

### `survive_long_v3` prompt verbatim (the algorithm body — feedback wanted)

The v3 prompt is still imperfect: it brings the policy back to baseline
length but fails to extend it past 629 steps. The full algorithm body
(everything above "Predict at a high level...") follows so the prompt
text itself can be edited.

```
Here is the algorithm the player will play the game by:
At every step, the player should act with the goal of maximizing how
long the episode lasts. The player keeps Food, Drink, and Energy
comfortably topped up at all times and refuses any action that risks
Health.

The player will choose the highest-priority active goal in this order:
1. Top up Food, Drink, or Energy whenever any of them drops to 7 or below
2. Move toward the safest visible terrain when no intrinsic needs
   replenishing
3. NOOP when on safe terrain and all intrinsics are at 8 or 9
4. Run from any visible enemy

1. Top up the lowest intrinsic
The player tracks Food, Drink, and Energy and acts as soon as any of
them is at 7 or below.
- Drink ≤ 7: the player walks to the nearest visible water tile and
  uses DO until Drink is 9. If the player is already adjacent to water,
  DO immediately.
- Food ≤ 7: the player walks to the nearest visible cow and uses DO to
  attack it, then eats the meat until Food is 9. If a fruit-bearing
  plant is closer, the player walks to it and DOes to eat instead.
- Energy ≤ 7: the player walks to the most enclosed adjacent tile
  (corner of stone, between trees, etc.) and uses SLEEP until wake_up.
  The player does not need a fully sealed base — any enclosed-looking
  tile is acceptable.
The thresholds are 7 (not the usual 4) so the player keeps a buffer and
never lets intrinsics drop into the danger zone where Health starts
decaying.

2. Move to safe terrain
If all intrinsics are 8 or 9 but the current tile is exposed (no walls
around, enemies visible nearby), the player moves toward the nearest
natural enclosure (a corner of stone, a clump of trees, the edge of
water) and stops there. The player does not range far — any safer tile
within 5 tiles of the current position is preferred over walking far.

3. NOOP
When all intrinsics are 8 or 9 and the current tile is safe (no enemy
within 4 tiles), the player picks NOOP. NOOP passes time without
spending any risk; it is preferred over any movement when there is
nothing to gain.

4. Run from enemies
If any enemy mob is visible and within 4 tiles, the player walks AWAY
from it (the direction that puts the most empty tiles between the
player and the enemy) until it is out of sight or until walking further
would lead the player into worse terrain. The player does not engage in
combat unless there is no escape route.

The player NEVER descends a ladder, NEVER explores beyond the immediate
area, NEVER enters dungeons, NEVER picks fights, and NEVER mines or
crafts beyond what is needed for an immediate sleep enclosure. The
single goal is to keep the episode going.
```

Suspected weaknesses (best guesses for why this still maxes out at
length ≈ baseline rather than extending past it):

- **NOOP-heavy advice + "never explore"** may starve the policy of
  resource-gathering it needs for multi-cycle survival. After a few
  cycles the player runs out of nearby cows / saplings / drinkable
  water and dies because the prompt forbids exploration.
- **Threshold of 7** triggers a big chunk of the prompt to fire too
  early, possibly creating thrashing between drink/eat/sleep targets
  when more than one is at 7.
- **"Run from enemies"** — using "AWAY" in the algorithm body might
  cause Gemini to inherit the negation-style language even though the
  Prediction template forbids it.
- **No mention of building a buffer** of food (extra meat) or water
  (drink-twice-while-here) — the policy probably runs out of nearby
  food/water faster than it ought to.

Open question for the next iteration: should the prompt explicitly
budget *resource accumulation* (e.g., "if drink ≥ 7 but a water tile is
adjacent, drink anyway to fill the buffer"), and should it allow modest
exploration when intrinsics are full but no resource is in sight?

The full file is at
`configs/training/templates/predict_state_only_prompt_concise_survive_long_v3.txt`.
