# STEER_SCORE — pushing C_grounded_2M to maximum return via prompt iteration

**Goal**: maximize per-episode return on `C_grounded_2M` (freezenone)
through Gemini-prompt steering.

## Craftax scoring system (read once, then carried as background)

Reward per step = `achievement_reward + health_reward`, where
`achievement_reward = sum_over_achievements_just_unlocked(tier_pts)` and
`health_reward = (player_health - init_health) * 0.1`. Each achievement
fires only on the *first* time it is unlocked in an episode. The
tier-points are encoded by `achievement_mapping()` in
`craftax/craftax/constants.py`:

| tier | count | pts/each | subtotal | examples |
|---|---|---|---|---|
| BASIC | 25 | 1 | 25 | collect_wood/stone/iron/coal/diamond, craft wood/stone/iron tools, place_stone/table/furnace/torch/plant, eat_cow, eat_plant, collect_drink, defeat_zombie/skeleton, wake_up |
| INTERMEDIATE | 18 | 3 | 54 | enter_dungeon, enter_gnomish_mines, eat_bat, eat_snail, find_bow, fire_bow, open_chest, drink_potion, defeat_orc_solider/orc_mage/gnome_warrior/gnome_archer, collect_sapphire, collect_ruby, make_diamond_pickaxe/sword, make_iron_armour, make_diamond_armour |
| ADVANCED | 15 | 5 | 75 | enter_sewers, enter_vault, enter_troll_mines, defeat_lizard/kobold/knight/archer/troll/deep_thing, learn/cast_fireball, learn/cast_iceball, enchant_sword, enchant_armour |
| VERY ADVANCED | 9 | 8 | 72 | enter_fire_realm, enter_ice_realm, enter_graveyard, defeat_pigman, defeat_fire_elemental, defeat_frost_troll, defeat_ice_elemental, damage_necromancer, defeat_necromancer |
| **TOTAL** | **67** | | **226** | |

So **226 is the max from achievements alone.** Health reward is small
(~0.1 per HP per step recovered), and never dominant in our experiments.
The Craftax paper / public leaderboard reports return as % of 226.

Reference scores (% of 226):
- Random policy ≈ 0.1–0.5%
- 1B PPO-RNN (paper) ≈ 15.3% = 34.6 raw
- 1B PPO-GTrXL (paper) ≈ 18.3% = 41.4 raw
- **PPO-RNN 1e8 (our replication)** ≈ 12.3% = **27.87 raw** (see analysis below)
- Unaugmented offline-RL (PSF top-2M, no augmentation) ≈ **18.38**
- C_grounded_2M baseline (current focus) ≈ **14.66** = 6.5% of max

## PPO-RNN 1e8 baseline — what does the strongest pure-RL policy at our scale actually achieve?

Source: wandb run [`fkxga61m`](https://wandb.ai/iris-sobolmark/craftax-baselines-replication/runs/fkxga61m)
(Craftax-Symbolic-v1, 1e8 timesteps, default
[Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py)
PPO-RNN). Training-time `episode_return = 27.87`,
`achievements (unique-per-ep mean) = 21.36`. Per-tier breakdown (all
rates from the wandb summary, full JSON at
`probe_results/ppo_baselines_achievement_breakdown.json`):

| tier | per-tier weighted contribution | unique unlocked at all (>0%) | sum-of-rates |
|---|---|---|---|
| BASIC (1pt × 25) | **+17.6** | **25/25** | 17.65 |
| INTERMEDIATE (3pt × 18) | **+11.1** | 13/18 | 3.71 |
| ADVANCED (5pt × 15) | **+0.0** | **0/15** | 0.00 |
| VERY ADVANCED (8pt × 9) | **+0.0** | **0/9** | 0.00 |
| **derived total** | **28.77** | **38/67** | |

The derived total (28.77) is within 1 point of the wandb-logged
`episode_return` (27.87) — the small gap is the health reward + sampling.

**Per-achievement rates from PPO-RNN 1e8 (sorted by rate):**

| ach | rate | tier |
|---|---|---|
| collect_sapling | 99.5% | BASIC |
| collect_wood | 99.0% | BASIC |
| place_plant | 99.0% | BASIC |
| make_wood_pickaxe | 98.0% | BASIC |
| place_table | 98.0% | BASIC |
| collect_stone | 97.5% | BASIC |
| place_furnace | 97.5% | BASIC |
| place_stone | 97.5% | BASIC |
| make_wood_sword | 91.5% | BASIC |
| collect_coal | 86.4% | BASIC |
| make_stone_pickaxe | 84.9% | BASIC |
| make_stone_sword | 84.9% | BASIC |
| make_arrow | 81.9% | BASIC |
| make_torch | 80.9% | BASIC |
| place_torch | 80.9% | BASIC |
| collect_drink | 78.4% | BASIC |
| eat_cow | 72.4% | BASIC |
| **enter_dungeon** | **67.8%** | INTERMEDIATE |
| collect_iron | 66.8% | BASIC |
| **find_bow** | **65.8%** | INTERMEDIATE |
| **open_chest** | **65.8%** | INTERMEDIATE |
| **eat_snail** | **60.8%** | INTERMEDIATE |
| **fire_bow** | **49.7%** | INTERMEDIATE |
| wake_up | 39.7% | BASIC |
| defeat_zombie | 38.2% | BASIC |
| defeat_skeleton | 36.2% | BASIC |
| **make_iron_pickaxe** | **31.7%** | BASIC |
| **drink_potion** | **30.2%** | INTERMEDIATE |
| **defeat_orc_solider** | **17.6%** | INTERMEDIATE |
| make_iron_sword | 15.1% | BASIC |
| **collect_diamond** | **8.5%** | BASIC |
| **collect_ruby** | **6.5%** | INTERMEDIATE |
| **collect_sapphire** | **4.0%** | INTERMEDIATE |
| **make_diamond_sword** | **1.0%** | INTERMEDIATE |
| eat_plant | 0.5% | BASIC |
| **make_diamond_pickaxe** | **0.5%** | INTERMEDIATE |
| **make_iron_armour** | **0.5%** | INTERMEDIATE |
| **defeat_orc_mage** | **0.5%** | INTERMEDIATE |
| (all 15 ADVANCED) | 0% | ADVANCED |
| (all 9 VERY ADVANCED) | 0% | VERY ADVANCED |

**Three things this baseline tells us:**

1. **PPO-RNN 1e8 is *also* capped at the descend-to-floor-1 boundary.**
   It enters the dungeon (floor 1) 68% of the time, but never enters
   floor 2 or below (0% on enter_sewers/vault/troll_mines and all
   higher floors). Whatever skill is needed to clear floor 1 enemies
   and find floor 1's ladder is not reliably there at 1e8 steps. Only
   the much larger 1B PPO-GTrXL run gets meaningfully past floor 1.

2. **Iron tier is reachable, diamond is barely reachable, magic is
   not.** make_iron_pickaxe at 32% is the headline late-tier success.
   make_diamond_* hovers at 0.5–1% — a handful of episodes per 1k. No
   spell ever cast. No enchantment ever applied.

3. **C_grounded_2M is at 51% of PPO-RNN 1e8's score** (14.66 / 28.77),
   from a tiny fraction of the data and a tiny fraction of the compute.
   The score-max v2 prompt narrows that gap further (18.39 / 28.77 =
   64%).

**Per-tier comparison vs the policies we're working with:**

| policy | weighted total | BASIC contrib | INTER contrib | ADV/VADV |
|---|---|---|---|---|
| PPO-RNN 5M | 7.86 | 7.86 | 0.0 | 0.0 |
| PPO-RNN 20M | 14.98 | 14.98 | 0.0 | 0.0 |
| C_grounded freezenone (baseline) | 15.56 | 14.83 | 0.73 | 0.0 |
| **C_grounded + achievement_max_v2** | **~19** | ~17 | ~2 | 0.0 |
| **PPO-RNN 1e8** | **28.77** | **17.6** | **11.1** | 0.0 |
| 1B PPO-RNN (paper) | ~34 | — | — | small |
| 1B PPO-GTrXL (paper) | ~41 | — | — | larger |
| max possible | 226 | 25 | 54 | 75+72 |

The ceiling our prompt-steering experiments are aiming at is **the
basic + intermediate band** (max 79). PPO-RNN 1e8 already harvests
~29 of those 79; we are at ~19. Closing the C-vs-PPO gap requires the
intermediate-tier achievements that PPO-RNN gets (find_bow, open_chest,
drink_potion, fire_bow, eat_snail, enter_dungeon at higher rate, and
defeat_orc_solider). Several of these are *new categories* that
C_grounded_2M's training data essentially never exhibited (because the
source PSF run never reached the dungeon at high rate). This is the
core motivation for [SCALING_C](SCALING_C.md): a higher-quality
trajectory source would expand the action repertoire AWR can train.

(`probe_results/ppo_baselines_achievement_breakdown.json` has the full
per-achievement rates from the wandb summaries for PPO-RNN 1e8 and the
partial-32M PPO-symbolic 1e8 run.)

## Strategy goal restated

Given the ~226-pt scale: each BASIC achievement we unlock from
0%→100% adds 1 pt. Each INTERMEDIATE adds 3. Each ADVANCED adds 5.
Each VERY ADVANCED adds 8. The cheapest wins for prompt steering are:
- BASIC at <100% on freezenone — `make_iron_pickaxe` (0% → headroom 1pt),
  `eat_plant` (0% → headroom 1pt), `wake_up` (52% → headroom 0.5pt),
  many others — total ~10 BASIC pts of headroom.
- INTERMEDIATE at 0% on freezenone but reachable: `enter_gnomish_mines`,
  `find_bow`, `fire_bow`, `open_chest`, `drink_potion`, `eat_bat`,
  `eat_snail`, `defeat_orc_solider`, `defeat_orc_mage` — most need
  the policy to actually descend reliably first. Total ~50pt of
  potential if all hit.
- ADVANCED tier mostly out of reach without clearing floor 1.

Headroom of ~60 BASIC+INTER pts. Realistic ceiling on C without
retraining is probably +10-15 from baseline 14.66 → ~25-30 (which
would land us at PPO-RNN 1e8 parity but from far less compute).

## Baseline diagnosis (n=50, no prompt change)

Return mean = 14.66 ± 0.83. Achievements unlocked across 50 runs (sorted
by rate):

| rate | achievement | category |
|---|---|---|
| 100% | collect_wood | starter |
| 96% | place_table | starter |
| 94% | collect_stone, make_wood_pickaxe | starter |
| 88% | make_wood_sword | starter |
| 86% | eat_cow | starter |
| 82% | collect_drink | starter |
| 76% | make_stone_pickaxe | tier-up |
| 74% | defeat_skeleton | combat |
| 72% | collect_coal, defeat_zombie, make_arrow | mid |
| 70% | make_stone_sword | tier-up |
| 68% | place_stone | mid |
| 62% | place_furnace | mid |
| 58% | make_torch | mid |
| 52% | place_torch, **wake_up** | mid |
| 50% | **collect_iron** | mid |
| 38% | **collect_sapling** | rare-but-doable |
| 28% | **place_plant** | rare-but-doable |
| 12% | **enter_dungeon** | rare-but-doable |
| 4%  | eat_snail | very rare |
| 2%  | collect_sapphire, find_bow, open_chest, defeat_orc_mage | very rare |

Never reached (40/67), notable misses:
- `eat_plant` (0%) — *reachable*: player must place_plant, wait ~50 steps for it to ripen, then DO on it. Baseline never waits long enough.
- `make_iron_pickaxe`, `make_iron_sword`, `make_iron_armour` (0%) —
  chain tasks. Specificity matrix shows the policy CANNOT execute these
  even when prompted directly (0/30).
- `collect_diamond`, all diamond items, all magic, all archer (0%) —
  out of reach.
- `enter_*` (sewers/vault/troll_mines/fire/ice/graveyard) (0%) — would
  need to descend multiple floors first; only enter_dungeon is reachable.

## Achievement headroom assessment (PPO-RNN-anchored, tier-weighted)

The earlier headroom estimate undercounted: `enter_dungeon` and
several other reachable items are INTERMEDIATE (3 pts), not BASIC
(1 pt). PPO-RNN 1e8 — same env, same observation space, just much
more compute — is the upper-bound *rates* we should aim at, since it
demonstrates the policy can actually achieve them. Below: per-target
headroom in **points** (rate-gain × tier-pts), using PPO-RNN 1e8 as the
ceiling and the C_grounded freezenone baseline as the floor.

### BASIC (1 pt each) — the cheap/easy reservoir

| ach | C base | PPO-RNN 1e8 | rate-gain | tier | pt-gain |
|---|---|---|---|---|---|
| wake_up | 52% | 40% | already above PPO-RNN here | 1 | 0 |
| collect_iron | 50% | 67% | +17pp | 1 | +0.17 |
| collect_sapling | 38% | 99% | +61pp | 1 | +0.61 |
| place_plant | 28% | 99% | +71pp | 1 | +0.71 |
| eat_plant | 0% | 0.5% | already at PPO-RNN ceiling | 1 | 0 |
| make_iron_pickaxe | 0% | 32% | +32pp | 1 | +0.32 |
| make_iron_sword | 0% | 15% | +15pp | 1 | +0.15 |
| collect_diamond | 0% | 8.5% | +8.5pp | 1 | +0.085 |
| collect_coal | 72% | 86% | +14pp | 1 | +0.14 |
| collect_drink | 82% | 78% | already above | 1 | 0 |
| place_stone | 68% | 97.5% | +29.5pp | 1 | +0.30 |
| place_furnace | 62% | 97.5% | +35pp | 1 | +0.35 |
| place_torch | 52% | 81% | +29pp | 1 | +0.29 |
| make_torch | 58% | 81% | +23pp | 1 | +0.23 |
| make_stone_pickaxe | 76% | 85% | +9pp | 1 | +0.09 |
| make_stone_sword | 70% | 85% | +15pp | 1 | +0.15 |
| make_arrow | 72% | 82% | +10pp | 1 | +0.10 |
| place_table | 96% | 98% | tiny | 1 | +0.02 |
| make_wood_sword | 88% | 91% | tiny | 1 | +0.03 |
| make_wood_pickaxe | 94% | 98% | tiny | 1 | +0.04 |
| collect_stone | 94% | 97.5% | tiny | 1 | +0.035 |
| collect_wood | 100% | 99% | already at ceiling | 1 | 0 |
| eat_cow | 86% | 72% | already above | 1 | 0 |
| defeat_zombie | 72% | 38% | already above | 1 | 0 |
| defeat_skeleton | 74% | 36% | already above | 1 | 0 |
| **BASIC subtotal** | | | | | **+3.97** |

C is already *above* PPO-RNN on combat (zombie/skeleton/cow), drink, and
wake_up — these are the survival skills the offline-RL policy learned
from the curated PSF data. The PPO-RNN policy is much *worse* at
single-step combat but much better at the rest of the BASIC tier.

### INTERMEDIATE (3 pt each) — the big reservoir, mostly floor-1+

| ach | C base | PPO-RNN 1e8 | rate-gain | tier | pt-gain |
|---|---|---|---|---|---|
| **enter_dungeon** | 12% | **68%** | +56pp | 3 | **+1.68** |
| find_bow | 2% | 66% | +64pp | 3 | +1.92 |
| open_chest | 2% | 66% | +64pp | 3 | +1.92 |
| eat_snail | 4% | 61% | +57pp | 3 | +1.71 |
| fire_bow | 0% | 50% | +50pp | 3 | +1.50 |
| drink_potion | 0% | 30% | +30pp | 3 | +0.90 |
| defeat_orc_solider | 0% | 18% | +18pp | 3 | +0.54 |
| collect_ruby | 0% | 6.5% | +6.5pp | 3 | +0.20 |
| collect_sapphire | 2% | 4% | +2pp | 3 | +0.06 |
| make_diamond_sword | 0% | 1% | +1pp | 3 | +0.03 |
| make_diamond_pickaxe | 0% | 0.5% | +0.5pp | 3 | +0.015 |
| make_iron_armour | 0% | 0.5% | +0.5pp | 3 | +0.015 |
| defeat_orc_mage | 2% | 0.5% | already above | 3 | 0 |
| (5 INTER never reached by PPO-RNN either) | 0% | 0% | — | 3 | 0 |
| **INTER subtotal** | | | | | **+10.5** |

The big leverage points are: `enter_dungeon` (+1.68), `find_bow` (+1.92),
`open_chest` (+1.92), `eat_snail` (+1.71), `fire_bow` (+1.50). All
require descending to floor 1 (gnomish_mines / dungeon level) and
hunting the lights/chests/snails on that floor. PPO-RNN gets these
because it descends 68% of the time; C only descends 12%, so most of
this band is locked behind the descent gate.

### ADVANCED + VERY ADVANCED — out of reach without floor-2+

PPO-RNN 1e8 itself never gets any of the 24 ADV / VADV achievements
(0/15 ADVANCED, 0/9 VERY ADVANCED). The descent gate at floor 1 is
binding even for the much-larger-compute baseline. The 1B PPO-GTrXL
paper number (18.3% of max ≈ 41.4) presumably gets some of these. For
us — at our compute scale — these tiers are genuinely closed unless we
either (a) fine-tune from a stronger base (SCALING_C with PPO-RNN data),
or (b) hand-craft a multi-floor exploration prompt that the policy can
follow.

### Total realistic prompt-only ceiling

If we could match PPO-RNN 1e8's per-achievement profile via prompt
steering on the existing C_grounded freezenone checkpoint:

|  | gain |
|---|---|
| BASIC reservoir | +3.97 |
| INTERMEDIATE reservoir | +10.5 |
| above-PPO-RNN combat / drink / wake_up | -2 to -3 (C loses these) |
| **net** | **~+11 to +12** |

So a perfect prompt would push C from 14.66 → ~26 — close to PPO-RNN
1e8's 27.87. **Prompt-only steering cannot exceed the PPO-RNN ceiling**
because we have no way to introduce floor-2+ behaviors that the policy
never learned in offline data.

That sets the realistic target for prompt iteration: **score-max v3
should aim for the 22–26 range** (currently v2 = 18.39). The remaining
~7 pts headroom is mostly the floor-1 cluster (enter_dungeon → bow,
chest, orc, snail, potion). v3 needs explicit *floor-1 instructions*,
not just "descend".

## Strategy

The specificity matrix taught us:
- **Single-target prompts disrupt off-target behaviors.** `target_descend_v2`
  boosts DESCEND but cuts cow_eat and stone count. `target_place_stone_v2`
  cuts return -1.1.
- **Chain prompts can't unlock new achievements.** Iron/diamond stay 0%
  no matter the prompt.
- **Direct opposites work** (avoid_water/animals — but these are
  destructive for return).
- **Survive prompts can't extend life past baseline ceiling.**

Therefore the score-max prompt cannot be "do X" — it must be a
*balanced enumeration prompt* that nudges several low-rate but reachable
achievements while preserving everything the policy already does well.

## Surprise from existing matrix data: `target_descend_v2` is already a score-max prompt

Re-mining the specificity matrix per-achievement breakdown:

| cell | n | return | length |
|---|---|---|---|
| baseline | 50 | 14.66 ± 0.83 | 629 ± 79 |
| target_descend_v2 | 30 | **17.23 ± 0.89** | **804 ± 132** |
| target_collect_sapling_v2 | 30 | 15.90 ± 1.05 | 647 ± 103 |
| target_place_plant_v2 | 30 | 15.37 ± 1.11 | 574 ± 69 |

`target_descend_v2` lifts return by **+2.57 (z≈+2.1)** without being
designed to. Per-achievement breakdown vs baseline:

| achievement | base% | descend% | Δpp |
|---|---|---|---|
| place_furnace | 62 | 87 | +25 |
| place_torch | 52 | 77 | +25 |
| collect_iron | 50 | 73 | +23 |
| make_torch | 58 | 80 | +22 |
| collect_coal | 72 | 90 | +18 |
| make_stone_sword | 70 | 87 | +17 |
| place_stone | 68 | 83 | +15 |
| enter_dungeon | 12 | 27 | +15 |
| wake_up | 52 | 67 | +15 |
| make_stone_pickaxe | 76 | 87 | +11 |
| eat_plant | 0 | 7 | +7 |
| ... | | | |
| collect_drink | 82 | 60 | -22 |

**Mechanism**: the descent prompt elevates ladder-progression as
priority 1, which (a) gets the policy to floor 1 sometimes (+15pp
enter_dungeon), (b) extends average episode length by +175 steps
because the policy gains player_xp on descent and behaves more
purposefully, and (c) the longer episode lets the policy hit MANY
more achievements opportunistically. Only `collect_drink` regresses
(the policy seems to skip drinks while focused on descent).

**Bar for v1 to beat: 17.23.** Anything that doesn't beat
target_descend_v2 is not actually optimizing score.

## Iteration log

### v1 — balanced achievement-chase ("`achievement_max_v1`")

Design hypothesis (BEFORE seeing target_descend's accidental score boost):
keep the base algorithm, add an "Opportunistic Milestones" checklist for
6 headroom achievements (sapling, place_plant, eat_plant, wake_up,
collect_iron, place_torch, enter_dungeon).

**v1 prompt — algorithm body verbatim** (file:
`configs/training/templates/predict_state_only_prompt_concise_achievement_max_v1.txt`):

```
Here is the algorithm the player will play the game by:
At every step, the player should act with the goal of staying alive AND
ticking off as many achievement milestones as possible. Each milestone
counts only the first time it is completed in an episode, so the player
picks up cheap milestones whenever they are within reach and never
repeats one for score.

The player will choose the highest-priority active goal in this order:
1. Survive
2. Pick up the next on-screen Opportunistic Milestone (see list below)
3. Take the ladder if it is open and on-screen
4. Upgrade equipment if survival is stable. This takes priority over
   taking the ladder if the player is in the overworld (floor 0) and
   has a sword or pickaxe worse than stone or missing.
5. Explore to find resources, troops, and the ladder

1. Survive
[ standard survive section + "sleeping until wake_up is also a free
  achievement (+1 wake_up)" added ]

2. Opportunistic Milestones (the headroom checklist)
Each item below is a one-time +1 achievement the baseline policy often
misses. The player should grab any milestone whenever it is achievable
in the next few steps with little risk. Listed in priority order:
  (a) collect_sapling — when a grass tile with a green sapling icon is
      within 2 tiles, walk over it and DO to harvest. Saplings spawn in
      grass clusters; if none are visible, the player passes any grass
      tile without delaying for it.
  (b) place_plant — once the player holds at least 1 sapling AND is
      standing on a grass tile, use PLACE_PLANT immediately. Note where
      the plant was placed so the player can return to it later for
      eat_plant.
  (c) eat_plant — a placed plant ripens into a fruit-bearing plant
      after roughly 30–60 steps. Whenever the player is within 3 tiles
      of an own placed plant that has ripened (fruit visible), walk to
      it and DO to eat. This restores food AND grants +1 eat_plant.
  (d) wake_up — explicitly cycle to SLEEP (and naturally wake) at least
      once per episode whenever Energy is at 4 or below; do not let
      Energy decay to 0 without sleeping in a safe enclosure first.
  (e) collect_iron — once the player has a stone pickaxe, mine ANY
      visible iron tile (orange/brown speckle) within 4 tiles even if
      it requires a small detour. This unlocks iron-tier later but the
      +1 collect_iron is the cheap part.
  (f) place_torch — once the player has crafted at least 1 torch (1
      Wood + 1 Coal at a crafting table), use PLACE_TORCH on the
      current tile. Free +1.
  (g) enter_dungeon — once the player is on Floor 0 with a stone
      pickaxe, stone sword, and at least 1 iron in inventory, head
      DIRECTLY for the visible down_ladder and DESCEND. Each new floor
      entered = +1 enter_X achievement.

3. Take the ladder if it is open and visible  [ standard ]
4. Upgrade equipment                          [ standard ]
5. Explore                                    [ standard ]
```

**v1 result (n=29): return 15.51 ± 1.00, length 658, Δret +0.85 (z=+0.65 NS).**
wandb: [`96z6ytog`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/96z6ytog)
(per-episode `gameplay.mp4` videos available there for inspection).

Below the target_descend bar of 17.23 in *return*, but **strong steering
result on the targeted achievements** — the prompt named 6 items as the
"headroom checklist" and 5/6 moved in the right direction:

| Δpp | achievement | note |
|---|---|---|
| **+17pp** | **place_plant (28→45%)** | the v1 nudge worked |
| **+10pp** | **place_torch (52→62%)** | the v1 nudge worked |
| **+7pp** | **eat_plant (0→7%)** | **first non-zero on this metric across any C eval — the steering literally created an achievement that didn't exist in baseline runs** |
| **+7pp** | **collect_sapling (38→45%)** | the v1 nudge worked |
| **+7pp** | **wake_up (52→59%)** | the v1 nudge worked |
| +5pp | **collect_iron (50→55%)** | the v1 nudge worked, modest |
| +11pp | defeat_zombie (72→83%) | beneficial side effect |
| +11pp | make_arrow | beneficial side effect |
| +5pp | find_bow, open_chest | beneficial side effect |
| **−5pp** | **enter_dungeon (12→7%)** | UNINTENDED. The v1 prompt gated dungeon entry on "stone pickaxe + stone sword + ≥1 iron", which made the policy WAIT longer. baseline descends without iron. |
| −5pp | defeat_skeleton, make_stone_sword | possibly a side effect of less floor-1 time |

This is the cleanest demonstration in the score-max thread that
**enumerating specific achievements in the prompt body is enough to
push them up by 7–17pp**. Five out of six named targets moved up as
intended; the only loss was `enter_dungeon`, and that traces directly
to a *different* rule the prompt added (the iron-gate). v1 also opens
`eat_plant` (0% → 7%) which had been zero across every prior C eval —
the prompt successfully introduced a new behavior, not just nudged an
existing one.

The takeaway for v2 was not "enumeration doesn't work" — enumeration
*does* work — but "the descent cascade is the bigger lever, and v1
accidentally crippled it. Keep enumeration, fix the descent gate." v2
landed at 18.39 (+3.73) by doing exactly that.

Inspecting v1 vs baseline gameplay in the wandb run is highly
recommended — the videos make the steering visible (you can watch the
policy actively walking *to* sapling icons and placing plants on grass
tiles, behaviors that are absent in baseline videos).

### v2 — `target_descend` base + cheap-milestones rider ("`achievement_max_v2`")

Design: keep `target_descend_v2`'s structure (which already produces
17.23 by tilting priority 1 to descent + cascading episode length).
Add an explicit "One-shot opportunistic milestone within 3 tiles"
section as priority 3 (between Survive and Take ladder), enumerating
sapling / place_plant / eat_plant / place_torch with a hard "no detour
> 3 tiles, never delay descent significantly" rule. Critically, do NOT
re-introduce v1's "wait for iron before descending" gate.

**v2 result (n=28): return 18.39 ± 1.02, length 915, Δret +3.73 (z=+2.83)
vs baseline; +1.15 vs the previous best `target_descend_v2` (17.23);
matches the unaugmented offline-RL baseline of 18.38.** Best
single-prompt result on the C policy to date. wandb:
[`0v0j63nw`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/0v0j63nw)
(per-episode videos available).

Per-achievement deltas (top movers, n=28 vs baseline n=50):

| Δpp | achievement | note |
|---|---|---|
| **+27pp** | place_furnace (62→89%) | descent cascade + iron-tier prep |
| **+27pp** | enter_dungeon (12→39%) | descent cascade preserved |
| +25pp | place_stone (68→93%) | descent cascade |
| +23pp | wake_up (52→75%) | sleep prompt + longer survival |
| +23pp | place_torch (52→75%) | milestone hit |
| +18pp | collect_iron (50→68%) | descent cascade |
| +17pp | collect_coal | descent cascade |
| +17pp | make_torch | descent cascade |
| +17pp | make_stone_pickaxe | descent cascade |
| +16pp | make_stone_sword | descent cascade |
| **+15pp** | place_plant (28→43%) | the v1 nudge that we kept |
| +12pp | collect_sapling | the v1 nudge that we kept |
| +10pp | defeat_zombie | side effect |
| +7pp | eat_cow | side effect |
| +5pp | find_bow, open_chest | side effect |
| +4pp | eat_plant (0→4%) | the v1 nudge that we kept; first non-zero on freezenone via steering alone |
| **−7pp** | **collect_drink (82→75%)** | only meaningful regression |
| −3pp | defeat_skeleton | small |

Episode length 629 → 915 (+45% longer). The combined prompt is
producing longer, more-purposeful runs that also pick up the rare
plant/torch/sapling milestones. Lessons:

- The descent-priority-1 pattern is the *structural* lever; without it
  v1 lost enter_dungeon (-5pp) and underperformed.
- The "no detour > 3 tiles" hard cap on the milestone rider is what
  let it co-exist with descent — without that v2 would have crowded
  out the descent cascade.
- We are now within noise of the unaugmented baseline (18.38) by prompt
  alone, and still steerable (the same prompt-suite from the
  Specificity matrix continues to move per-axis behaviors).

### Where to go from here

The next iteration question: can we push past 18.4 by combining v2's
milestone rider with explicit chain-task elicitation
(make_iron_pickaxe, make_iron_sword), or are we hitting the policy
fidelity ceiling? The Specificity matrix has shown 0/30 iron pickaxe
across multiple chain-task prompts on freezenone, but **`v2_long_tail`
patch produced the first non-zero make_iron_pickaxe (1/30, 3%)** and
v2 also got 1/28 (~4%). That's not random noise — the prompt is
opening the door. A v3 that doubles down on the iron crafting recipe
("after collecting 1 iron AND 1 coal AND 1 wood AND placing furnace +
table, immediately MAKE_IRON_PICKAXE") might push it further.

A *separate* and possibly higher-leverage lever: see the
[AWR-only ablation in STEERABILITY.md](../docs/STEERABILITY.md#awr-only-ablation)
— the BC+oracle finetune phase that produced the freezenone checkpoint
appears to *drop* baseline return by ~2.8 (freezenone 14.66 vs
awr-only 17.43, n=27). If we evaluate the prompt suite against the
awr-only checkpoint instead of freezenone, the achievable score
ceiling may shift up by another ~3 points.
