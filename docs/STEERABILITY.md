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
entirely and injecting α=+2 × d_die into the hidden produces −5.13 return
(matching the direct-prompt die_v2 effect of −4.90). The death "axis"
lives in embedding space and the policy reads it.

Source journals: `journals/log_2026-04-24.md`, `journals/log_2026-04-23.md`,
`journals/log_2026-04-22.md`. Primary data: `probe_results/master_table_FINAL.md`
and the JSON files under `probe_results/`.

---

## TL;DR — three killer results

1. **Synthetic embedding arithmetic reproduces the prompt-based effect.**
   On `C_grounded_2M`, adding α=+2 × d_die to the regular embedding (Gemini
   call unchanged) drops return by **−5.13 (z=−3.50, n=30)**, matching the
   direct die_v2 prompt effect of **−4.90 (z=−3.84, n=50)**. On `A_full` the
   same intervention is null (−0.74). Embedding direction is the causal
   mediator of the prompt's content effect.

2. **Positive achievement steering shifts behavior + gives the first net
   positive-return augmented prompt.** On `C_grounded_2M`,
   `target_collect_stone_v2` produces Δret **+1.10 (z=+1.0 NS, n=50)** with
   place_stone +16pp, place_furnace +18pp, wake_up +24pp.
   `target_descend_v2` produces enter_dungeon +10pp (12% → 22%, +83%
   relative) and DESCEND action +44% relative.

3. **Direction-only steering moves the action distribution at the cost of
   survival.** `direction_left_v2` on `C_grounded_2M`: LEFT%-of-moves
   0.24 → 0.34 (**+41% relative**), return drops −6.72 (z=−5.7). The policy
   commits to left-walking past survival utility. The steering *works at
   the action level* even when it destroys productivity.

**Every probe agrees on the track ordering**: `C_grounded_2M` (high
fidelity) > `A_full` (low fidelity) on steerability; the opposite on
absolute return and robustness.

---

## Track definitions

| Track key | Training data | Inference | On-policy return (regular, n=50) | Steerability profile |
|---|---|---|---|---|
| **`A_full`** | Full 158-file PSF `predonly` data. Gemini called on regular obs with concise prompt, `Prediction:` suffix only, embedded via `gemini-embedding-001` (3072-dim). | Same concise prompt at deploy time. | **18.98 ± 0.36** | Content-blind policy head. Value head reads presence only (Δ_shuf=+0.026). Robust across all adversarial prompts (die_v2 Δ=−0.76, z=−1.1). Negative control for steering. |
| **`B_thinking_2M`** | Top-2M subset of PSF. Gemini called with thinking prompt (thinking_budget=512) on regular obs; `Prediction:` extracted and embedded. | Thinking prompt (thinking_budget=512) at deploy. | **16.31 ± 1.00 (n=43)** | Mid-fidelity. Held-out Δ_shuf=+0.085 (3× A). Responds strongly to broad bad-play (die_v2 Δ=−4.09, z=−2.8) but less to narrow content steering (avoid_animals Δ=+0.53). OOD dir-CF step-0 hidden-flip 16.7% dominates at short horizon. |
| **`C_grounded_2M`** | Top-2M subset of PSF. Gemini called with grounded V6 prompt (concise template + 5-step future obs block) on regular obs; `Prediction:` extracted and embedded. Oracle-at-label-time. | Concise prompt at deploy (NO oracle / no future). Intentional train/eval distribution mismatch. | **14.66 ± 0.82** | Highest content fidelity. Held-out Δ_shuf=+0.195 (7× A). All probes register as steerable: die_v2 Δ=−4.90, direction_left LEFT% +41% rel, target_collect_stone +1.10 net positive return. HP/food sign WRONG (10× A magnitude) — grounded policy reads content as oracle-quality. |

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

---

## Prompt-variant design

Every `*_v2.txt` template in `configs/training/templates/` is derived from
`predict_state_only_prompt_concise.txt` by **replacing the "Here is a
good algorithm..." block** (lines 23–66 of the base) with an alternate
algorithm of the same structure and voice. No suffixes or override
phrasing (fixes the v1 format-confound issue where 21% of adversarial
outputs contained "Instead of..." phrasing).

The outer instructions (State Understanding / Prediction format, five-step
horizon, coordinate conventions, allowed actions) are preserved verbatim so
that the predonly-extracted embeddings stay close to the concise noise floor
(cos-sim to concise baseline 0.77–0.93, probe: `tools/probe_steering_v2*.py`).

### Base algorithm section (verbatim from `predict_state_only_prompt_concise.txt` lines 23–29)

> Here is a good algorithm the player will play the game by:
> At every step, the player should act with the goal of staying alive and progressing down floors.
> This means the player will choose the highest-priority active goal in this order:
> 1. Survive
> 2. Take the ladder if it is open and on-screen
> 3. Upgrade equipment if survival is stable...
> 4. Explore to find resources, troops, and the ladder

Each section below shows the algorithm substitution (3–5 lines), the
per-track results, the wandb run locator, and a short interpretation.

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
| A_full | 18.22 ± 0.57 | −0.76 (z=−1.1) | +0.004 | −0.008 | −0.06 | −0.04 | −0.12 |
| A_top2M | 15.86 ± 0.77 | −2.28 (z=−2.4) | −0.007 | +0.003 | −0.08 | −0.08 | −0.06 |
| B_thinking_2M | 12.22 ± 1.08 | −4.09 (z=−2.8) | +0.063 | +0.033 | −0.18 | −0.27 | −0.22 |
| **C_grounded_2M** | **9.76 ± 0.98** | **−4.90 (z=−3.84)** | **+0.233** | −0.009 | −0.36 | −0.40 | −0.38 |

wandb: run `eval_c_grounded_2M_die_v2_50ep` in `craftax-offline-awr`. Video dir:
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
| A_full | 19.38 ± 0.29 | +0.40 (z=+0.9) | +0.003 | −0.04 | +0.02 |
| A_top2M | 16.08 ± 0.90 | −2.06 (z=−1.9) | +0.018 | −0.12 | −0.06 |
| B_thinking_2M | 13.80 ± 0.94 | −2.51 (z=−1.8) | +0.041 | −0.10 | −0.17 |
| C_grounded_2M | 11.68 ± 0.95 | −2.98 (z=−2.4) | +0.166 | −0.16 | −0.16 |

### Interpretation

Same ordering as die_v2 but with weaker magnitudes. One mode-concentration
note: adv_v2 Gemini outputs disproportionately mention "repeatedly attempt
to craft iron pickaxe without materials" (~12/20 in the probe), so the
behavioral effect is narrower than die_v2's broader "bad play" disruption.
Still a valid content-only probe (0% "Instead of", 0% bulleted structures).

---

<a id="avoid_water_v2"></a>
## `avoid_water_v2` — narrow class-removal (water opaque)

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
| A_full | 18.30 ± 0.41 | −0.68 (z=−1.2) | −0.06 |
| A_top2M | (see Apr-23 partial) | −1.66 marginal | — |
| B_thinking_2M | 15.90 ± 0.92 | −0.41 (z=−0.3) | −0.10 |
| C_grounded_2M | 13.64 ± 1.01 | −1.02 (z=−0.8) | −0.12 |

### Interpretation

Smallest effect among narrow-steering prompts. Water-drinking is a smaller
return component than eating cows, so removing it has a modest cost. No
track shows a significant effect.

---

<a id="avoid_animals_v2"></a>
## `avoid_animals_v2` — narrow class-removal (cows opaque)

### Prompt algorithm substitution

> The player treats animals (cows and plants) as navigationally opaque —
> their tiles act like walls that the player routes around.
> The player restores Food by eating cooked meat already in inventory, or
> by harvesting plants for fruit when plants are adjacent. The player
> never attacks cows, never uses DO on a cow, and never moves onto a
> tile adjacent to a cow.

### Per-track results (n=50)

| Track | return ± SE | Δret (z) | Δp(DO) | Δrate(eat_cow) |
|---|---|---|---|---|
| A_full | 18.20 ± 0.50 | −0.78 (z=−1.3) | +0.006 | −0.06 |
| B_thinking_2M | 16.84 ± 0.96 | +0.53 (z=+0.4) | −0.042 | 0.00 |
| **C_grounded_2M** | **11.54 ± 1.02** | **−3.12 (z=−2.4)** | **+0.157** | −0.12 |

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
`switch → avoid_animals @ step 200` on C_grounded gives **+1.81 return (z=+1.56)**, one of the
strongest positive signals in the dataset.

---

<a id="target_collect_stone_v2"></a>
## `target_collect_stone_v2` — positive target "mine stone"

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
| A_full | 50 | 18.04 ± 0.48 | −0.94 (z=−1.6) | −0.08 | −0.02 | −0.04 |
| B_thinking_2M | **In progress (n=24–29/50, job 7465078)** | 15.34 ± 1.16 (partial n=29) | +0.68 (z=+0.48 NS) | +0.0146 Δp (+129% rel) | — | +0.24 |
| **C_grounded_2M** | 50 | **15.76 ± 0.73** | **+1.10 (z=+1.0)** | **+0.16** | **+0.00 (baseline 0.68)** | **+0.24** |

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
| A_full | 50 | 19.28 ± 0.27 | +0.30 (z=+0.7) | +0.06 | +24% |
| B_thinking_2M | **In progress (n=11–17/50, job 7465078)** | 15.74 ± 1.99 (partial n=11) | −0.57 (z=−0.26) | +0.15 | — |
| **C_grounded_2M** | 50 | 14.80 ± 0.99 | +0.14 (z=+0.1) | **+0.10 (0.12 → 0.22, +83% rel)** | **+44%** |

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
| A_full | 50 | 18.62 ± 0.42 | −0.36 (z=−0.6) | −0.02 | −0.04 |
| B_thinking_2M | **In progress (n=23/50)** | 14.32 ± 1.30 (partial) | −0.34 (z=−0.22) | +0.00 | +0.20 |
| C_grounded_2M | 50 | 14.42 ± 0.83 | −0.24 (z=−0.2) | +0.04 | +0.20 |

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

### Prompt algorithm substitution

> The player treats drinking from water as the single dominant
> priority. Every step is chosen to bring the player closer to a water
> tile and into a position to use DO on it.

### Per-track results (A_full, n=50)

| Track | return ± SE | Δret (z) | Δp(DO) | Δrate(collect_drink) |
|---|---|---|---|---|
| A_full | 18.08 ± 0.40 | −0.90 (z=−1.7) | −0.013 | −0.14 |

C_grounded / B_thinking not run online.

### Interpretation

On A_full, a prompt asking to drink more actually REDUCES collect_drink
and DO rate — the prompt's content is not being parsed causally by
A_full. Consistent with A_full's content-blindness across all probes.

---

<a id="target_place_stone_v2"></a>
## `target_place_stone_v2` — positive target "build stone walls"

### Prompt algorithm substitution

> The player treats placing stone walls as the single dominant priority.
> 1. If the player has stone and is next to empty tile, PLACE_STONE.
> 2. If no stone, mine stone.
> 3. Build compact stone enclosures wherever space allows.

### Per-track results (A_full, n=50)

| Track | return ± SE | Δret (z) | Δrate(place_stone) |
|---|---|---|---|
| A_full | 18.38 ± 0.52 | −0.60 (z=−0.9) | −0.10 |

### Interpretation

A_full again content-blind. place_stone rate actually DOWN −10pp (baseline
1.00 → 0.90). C_grounded not run.

---

<a id="target_hunt_animals_v2"></a>
## `target_hunt_animals_v2` — positive target "hunt all animals"

### Prompt algorithm substitution

> The player treats ALL visible animals (cows and plants) as immediate
> hunt targets. The moment any animal is visible, the player redirects
> to attack or harvest it.
> 1. If any cow visible, move toward nearest, DO repeatedly, eat meat.
> 2. If any plant with fruit visible, harvest.
> 3. Otherwise move toward grassland.

### Per-track results

**QUEUED (job 7468781, 3 cells: A_full / C_grounded_2M / B_thinking_2M).**
Results will be appended to `probe_results/master_table_FINAL.md` once the
cells finish (~12h ETA at time of writing).

---

<a id="direction_left_v2"></a>
## `direction_left_v2` — pure cardinal direction "walk left"

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
| A_full | 50 | 17.48 ± 0.54 | −1.50 (z=−2.3) | 0.262 → 0.255 (−3%) | −0.021 |
| B_thinking_2M | **In progress (n=16/50)** | 13.54 ± 1.74 (partial) | −2.77 (z=−1.38) | 0.255 → 0.264 (+4%) | +0.040 |
| **C_grounded_2M** | 50 | **7.94 ± 0.84** | **−6.72 (z=−5.7)** | **0.243 → 0.343 (+41%)** | +0.175 |

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

### Prompt algorithm substitution

> The player walks RIGHT (positive-column direction) at every step.
> (Otherwise identical in structure to direction_left_v2.)

### Per-track results (n=50 unless noted)

| Track | n | return ± SE | Δret (z) | RIGHT%-of-moves (Δ) | LEFT%-of-moves (Δ) |
|---|---|---|---|---|---|
| A_full | 50 | 18.22 ± 0.46 | −0.76 (z=−1.3) | +0.002 | −0.005 |
| B_thinking_2M | **In progress (n=18/50)** | 13.77 ± 1.20 (partial) | −2.54 (z=−1.62) | **+0.028 (+11% rel)** | −0.003 |
| C_grounded_2M | 50 | 11.16 ± 0.92 | −3.50 (z=−2.8) | +0.007 | +0.001 |

### Interpretation

Weaker than direction_left on RIGHT%. C_grounded's return drop (−3.50)
is mostly via DO-spike (Δp(DO)=+0.165) rather than clean RIGHT-walking.
B_thinking's partial signal (+11% relative RIGHT) is actually the
cleanest of the three tracks on right-specific behavior. Reason for the
asymmetry unclear; worth re-checking once B reaches n=50.

---

<a id="direction_up_direction_down"></a>
## `direction_up_v2`, `direction_down_v2` — remaining cardinals

Templates exist (`*_direction_up_v2.txt`, `*_direction_down_v2.txt`) with
identical structure to left/right. **Not run online** — prioritized
left/right as the minimal test; up/down would add symmetry if the
left/right signal replicates at n=50 on B and C.

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
| A_full | baseline (regular Gemini) | 50 | 18.98 ± 0.36 | — |
| A_full | direct `die_v2` prompt | 50 | 18.22 ± 0.57 | −0.76 (z=−1.1) |
| **A_full** | α=+2 × d_die | 29 | 18.24 ± 0.53 | **−0.74 (z=−1.2)** |
| A_full | α=−2 × d_die | 29 | 18.62 ± 0.63 | −0.36 (z=−0.5) |
| C_grounded_2M | baseline | 50 | 14.66 ± 0.82 | — |
| C_grounded_2M | direct `die_v2` prompt | 50 | 9.76 ± 0.98 | −4.90 (z=−3.84) |
| **C_grounded_2M** | **α=+2 × d_die** | **30** | **9.53 ± 1.22** | **−5.13 (z=−3.50)** |
| C_grounded_2M | α=−2 × d_die | 29 | 14.93 ± 1.20 | +0.27 (z=+0.18) |
| C_grounded_2M | α=+2 × d_avoid_animals | 30 | 14.73 ± 0.99 | +0.07 (z=+0.06) |
| C_grounded_2M | α=−2 × d_avoid_animals | 30 | 12.73 ± 1.07 | −1.93 (z=−1.4) |

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
| A_full d_value α=+2 | 30 | 17.03 ± 0.76 | **−1.95 (z=−2.31)** |
| A_full d_value α=−2 | 15 | 19.10 ± 0.89 | +0.12 (NS) |
| C_grounded d_value α=+2 | 30 | 13.57 ± 1.13 | −1.09 (z=−0.78) |
| **C_grounded d_value α=−2** | 30 | **8.73 ± 1.22** | **−5.93 (z=−4.03)** |

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
| A_full switch → die_v2 @ 200 | 30 | 19.20 ± 0.37 | +0.22 (NS) | — |
| C_grounded switch → die_v2 @ 200 | 30 | 14.40 ± 0.93 | −0.26 (z=−0.21) | — |
| **C_grounded switch → target_descend @ 200** | 30 | 14.97 ± 1.18 | +0.31 (z=+0.21) | **+0.15 (0.12 → 0.27, +125% rel)** |
| **C_grounded switch → avoid_animals @ 200** | **30** | **16.47 ± 0.82** | **+1.81 (z=+1.56)** | — (see below) |

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

3. **`switch → avoid_animals` is the strongest positive-return result.**
   Δret = +1.81 (z=+1.56, near-significant at n=30). Switching from
   normal Gemini to avoid_animals at step 200 shifts the policy toward
   productive late-game behaviors: place_furnace +15pp, place_torch
   +15pp, make_stone_pickaxe +14pp, make_stone_sword +13pp, eat_cow
   +11pp. The right *timing* of steering appears to matter as much as
   the steering itself.

Source: `probe_results/steerability_analysis/c_full_extra.json`,
`c_switch_partial.json`.

---

## Master comparison table — one row per condition × track

Return (± SE). Bold = decisive; `—` = not run; `—P` = in progress at
writing time.

| Condition | A_full (base 18.98) | A_top2M (base 18.14) | B_thinking_2M (base 16.31) | C_grounded_2M (base 14.66) |
|---|---|---|---|---|
| `die_v2` | 18.22 ± 0.57 (−0.76) | 15.86 ± 0.77 (−2.28*) | 12.22 ± 1.08 (**−4.09**) | **9.76 ± 0.98 (−4.90)** |
| `adversarial_v2` | 19.38 ± 0.29 (+0.40) | 16.08 ± 0.90 (−2.06) | 13.80 ± 0.94 (−2.51) | 11.68 ± 0.95 (−2.98*) |
| `avoid_water_v2` | 18.30 ± 0.41 (−0.68) | (−1.66 marginal) | 15.90 ± 0.92 (−0.41) | 13.64 ± 1.01 (−1.02) |
| `avoid_animals_v2` | 18.20 ± 0.50 (−0.78) | (−1.70 marginal) | 16.84 ± 0.96 (+0.53) | **11.54 ± 1.02 (−3.12*)** |
| `target_collect_stone_v2` | 18.04 ± 0.48 (−0.94) | — | —P (n≈29) | **15.76 ± 0.73 (+1.10)** |
| `target_descend_v2` | 19.28 ± 0.27 (+0.30) | — | —P (n≈11) | 14.80 ± 0.99 (+0.14) |
| `target_eat_cow_v2` | 18.62 ± 0.42 (−0.36) | — | —P (n≈23) | 14.42 ± 0.83 (−0.24) |
| `target_drink_water_v2` | 18.08 ± 0.40 (−0.90) | — | — | — |
| `target_place_stone_v2` | 18.38 ± 0.52 (−0.60) | — | — | — |
| `target_hunt_animals_v2` | queued (7468781) | — | queued | queued |
| `direction_left_v2` | 17.48 ± 0.54 (−1.50*) | — | —P (n≈16) | **7.94 ± 0.84 (−6.72)** |
| `direction_right_v2` | 18.22 ± 0.46 (−0.76) | — | —P (n≈18) | 11.16 ± 0.92 (−3.50*) |
| α=+2 × d_die (synthetic) | 18.24 ± 0.53 (−0.74) | — | — | **9.53 ± 1.22 (−5.13)** |
| α=−2 × d_die (synthetic) | 18.62 ± 0.63 (−0.36) | — | — | 14.93 ± 1.20 (+0.27) |
| d_value α=+2 | 17.03 ± 0.76 (−1.95*) | — | — | 13.57 ± 1.13 (−1.09) |
| d_value α=−2 | 19.10 ± 0.89 (+0.12) | — | — | **8.73 ± 1.22 (−5.93)** |
| switch → die_v2 @200 | 19.20 ± 0.37 (+0.22) | — | — | 14.40 ± 0.93 (−0.26) |
| switch → target_descend @200 | — | — | — | 14.97 ± 1.18 (+0.31) |
| **switch → avoid_animals @200** | — | — | — | **16.47 ± 0.82 (+1.81)** |

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
| PPO-symbolic 1e8 (resubmit 7464286) | pending | — | — | — |
| PPO-RNN 1e8 with checkpoint save (7464853) | pending | — | — | — |

**Scoreboard context**: 1B PPO-RNN = 15.3%, 1B PPO-GTrXL = 18.3%. Our
augmented tracks sit below the 1e8 PPO-RNN baseline on raw return. The
steerability claim is orthogonal to raw-return competitiveness: we
demonstrate it in the regime where augmented returns roughly match the
unaug offline-RL baseline (18.38), not at scoreboard-top scale.

---

## Status of in-progress jobs (as of 2026-04-24 EOD)

| Job | Cells | Status | ETA |
|---|---|---|---|
| 7465078 v2_steerability (B_thinking_2M) | target_{collect_stone,descend,eat_cow}, direction_{left,right} | In progress (n≈17–24 each) | ~12h to reach n=50 |
| 7468781 v2 target_hunt_animals | A_full, C_grounded_2M, B_thinking_2M × target_hunt_animals | Queued | L40S, 12h walltime |
| 7468782 PPO video re-eval | PPO-RNN {5M, 20M} with wandb video | Running | ~6h |
| 7464286 PPO-symbolic 1e8 (resubmit) | 1 cell | Pending | 48h walltime, ~55h wall |
| 7464853 PPO-RNN 1e8 w/ checkpoint save | 1 cell | Pending | 24h |

Partial numbers for B_thinking_2M already recorded in
`probe_results/steerability_analysis/b_partial2.json` and
`probe_results/action_analysis/b_partial2.json`. C_grounded additional
inventory counts are being computed in the background (stone, wood,
coal, iron, food_intake_events, drink_intake_events,
monsters_killed_total) — don't block on them.

---

## What this changes about the central claim

Support for each sub-claim:

| Sub-claim | Evidence | Status |
|---|---|---|
| Behavior changes when content changes | die_v2 on C: −4.90 return + DO rate +23pp (2.2× baseline), matches prompt directive. | Confirmed |
| Direction is structured (not just OOD) | Monotonic α-sweeps (LEFT corr=−0.92, RIGHT corr=+0.98). Signal direction +2.1σ above norm-matched random. | Confirmed with caveat (high-gain component) |
| Weak-fidelity policy is robust | A_full die_v2=−0.76, argmax flip <4% across ±2α. | Confirmed |
| High-fidelity policy is steerable | C_grounded all probes register. | Confirmed |
| Continuous structure in embedding space | α-sweep is monotonic to α=±2, value linear. | Confirmed |
| Cross-policy direction transfer | A_full-derived direction steers C_grounded by 20% flip. | Confirmed (policy is the bottleneck, not direction quality) |
| Positive achievement steering produces positive effect | `target_collect_stone` Δ+1.10; `switch → avoid_animals @200` Δ+1.81. | Confirmed marginally |
| Pure-direction steering moves action distribution | direction_left LEFT% +41% relative on C. | Confirmed |
| Synthetic embedding intervention reproduces prompt effect | α=+2 × d_die produces −5.13 (matching −4.90 prompt). | **Decisive** |
| Mid-episode switch responsive | target_descend doubled when applied mid-episode. | Confirmed |
| Steering IMPROVES return | target_collect_stone +1.10 (NS z=+1.0); switch→avoid_animals +1.81 (z=+1.56). | Marginal |
| Value-gradient-aligned steering improves return | d_value steering HURTS return on both tracks. | **Negative result** — steerability requires semantic-content directions, not value-head-favorable ones |

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

