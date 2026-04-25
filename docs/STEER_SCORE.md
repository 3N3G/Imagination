# STEER_SCORE — pushing C_grounded_2M to maximum return via prompt iteration

**Goal**: maximize per-episode return on `C_grounded_2M` (freezenone)
through Gemini-prompt steering. Each Craftax achievement is worth +1
reward (max possible ≈ 67); baseline score is 14.66 ± 0.83. **Headroom
is large** but the policy has hard fidelity ceilings on chain tasks
(0/30 iron pickaxe, 0/30 diamond per the specificity matrix). This doc
records the prompt-iteration journey.

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

## Achievement headroom assessment

Achievements with **realistic upward headroom** (already achieved
sometimes, just need consistency boost):

| ach | baseline | realistic ceiling | gain |
|---|---|---|---|
| wake_up | 52% | 80% | +0.28 |
| collect_iron | 50% | 80% | +0.30 |
| collect_sapling | 38% | 80% | +0.42 |
| place_plant | 28% | 70% | +0.42 |
| enter_dungeon | 12% | 50% | +0.38 |
| eat_plant | 0% | 30% | +0.30 |
| **Sum** | | | **+2.10** |

If we can lift these 6 achievements without disrupting the existing
high-rate ones, return should rise from 14.66 to ≈ 16.7 (+14% relative).

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

**This may already be the wrong design.** v1 enumerates milestones but
doesn't tilt priorities toward descent — the *structural* lever that
appears most powerful. v1 will tell us whether enumeration alone helps
relative to the baseline (14.66) and to target_descend (17.23).

**v1 result (n=29): return 15.51 ± 1.00, length 658, Δret +0.85 (z=+0.65 NS).**
Below the target_descend bar of 17.23. Per-achievement deltas vs baseline:

| Δpp | achievement | note |
|---|---|---|
| **+17pp** | place_plant (28→45%) | the v1 nudge worked |
| +11pp | defeat_zombie (72→83%) | side effect |
| +11pp | make_arrow | side effect |
| +10pp | place_torch (52→62%) | the v1 nudge worked |
| +7pp | eat_plant (0→7%) | first non-zero on this metric |
| +7pp | collect_sapling (38→45%) | the v1 nudge worked |
| +7pp | wake_up (52→59%) | the v1 nudge worked |
| +5pp | collect_iron, find_bow, open_chest | |
| **−5pp** | **enter_dungeon (12→7%)** | UNINTENDED. The v1 prompt gated dungeon entry on "stone pickaxe + stone sword + ≥1 iron", which made the policy WAIT longer. baseline descends without iron. |
| −5pp | defeat_skeleton, make_stone_sword | possibly a side effect of less floor-1 time |

Reading: the v1 milestone enumeration captured the headroom items it
named (place_plant, eat_plant, sapling, wake_up, place_torch) but the
extra rule "wait for iron before descending" cost the policy its
descent cascade. Net: small win, well below `target_descend_v2`.

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
single-prompt result on the C policy to date.

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
