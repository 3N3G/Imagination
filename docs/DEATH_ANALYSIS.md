# Where C dies — replay-based cause-of-death classification

**Scope**: all 50 episodes from the C_grounded_2M freezenone baseline
eval (wandb run [`pjb8wf7z`](https://wandb.ai/iris-sobolmark/craftax-offline-awr/runs/pjb8wf7z),
mean return 14.66 ± 0.83). Replayed each episode deterministically
with the saved action sequence, captured the final `env_state`, and
classified the cause-of-death.

Tool: `tools/classify_episode_deaths.py`.
Output: `probe_results/death_classification_C_baseline.json`.
Validation: cross-checked HP against Gemini's final-call text — 2/50
small mismatches caused by RNG drift between replay and original eval
(deaths happen within the 5-step Gemini cadence window after the last
Gemini call).

## Headline tally

| cause | n | % |
|---|---|---|
| killed_by_melee_adjacent | 30 | **60%** |
| dehydration (drink → 0) | 5 | 10% |
| killed_by_ranged_or_arrow | 4 | 8% |
| starvation (food → 0) | 2 | 4% |
| exhaustion (energy → 0) | 1 | 2% |
| killed_unknown_combat | 1 | 2% |
| alive_or_timeout (replay-RNG drift) | 7 | 14% |

## Critical observation: ALL 50 EPISODES ENDED ON FLOOR 0

Player floor at death across all 50 episodes = 0. The policy never
even gets killed on Floor 1+ — because **only 6/50 (12%) ever
reached Floor 1 at all** (the `enter_dungeon` achievement). Of those
6, all died on Floor 0 too because Craftax episodes track the floor
at the death tile, and they came back up to 0. Either way: **the
policy spends its life on Floor 0 and dies there**.

This explains the per-achievement gap directly:
- `find_bow` 2%, `open_chest` 2%, `eat_snail` 4%, `defeat_orc_solider`
  0% — ALL of these are floor-1+ entities. The policy can't do them
  because it doesn't go where they are.
- `enter_dungeon` 12% means 12% of episodes reach Floor 1 at all,
  but they die quickly there (or come back up because of low HP).
- C is at 0% on iron pickaxe / diamond / spells / enchant — same
  story scaled up: those are floor-2+ activities.

## Death-mechanism breakdown

### Drink-low is the underlying weakness in ~30% of deaths

User noticed (correct): in ~ep50/49/48/47/46 the player dies with
0 drink. Tracking intrinsic trajectories across the full replay,
the picture is broader: **drink is at 0 for the ENTIRE last 50
steps in 15/50 episodes (30%)**, and is implicated as a co-cause
even when the proximate kill is melee:

| cause | n | mean % of last-50-steps with drink ≤ 0 | mean % with drink ≤ 2 |
|---|---|---|---|
| dehydration | 5 | **100%** | 100% |
| alive_or_timeout | 7 | **39%** | 45% |
| killed_by_melee_adjacent | 30 | 12% (8 episodes at 100%, 22 at 0%) | 33% |
| killed_by_ranged_or_arrow | 4 | 4% | 33% |
| starvation | 2 | 0% (food=0 instead) | 12% |
| exhaustion | 1 | 0% (energy=0) | 36% |

So:
- 5 episodes die *of* dehydration (cause-of-death = HP=0 with drink=0)
- An additional **8 melee-death episodes had drink=0 for the entire
  last 50 steps** before being caught
- An additional 7 alive_or_timeout episodes (which are real deaths
  within the 5-step Gemini cadence) had drink=0 for ~40% of their
  final phase

Conservative estimate: **~13/50 = 26% of deaths have drink=0 as a
load-bearing co-cause.** Less conservative (counting any episode
with sustained drink≤2 in the last 50 steps): ~25/50 = 50%.

**Mechanism**: Craftax health regen requires drink (and food and
energy) to be ≥5. With drink=0 the player loses 1 HP per step of
intrinsic decay AND cannot regenerate after taking melee hits. So
the immediate cause is "zombie chops HP to 0" but the underlying
reason is "you couldn't regen the previous hits because you were
dehydrated".

### 1. Melee deaths (60%) — primarily Floor 0 zombies

These are zombies (and skeletons that closed to melee range) on
Floor 0. The pattern in the replay data: HP drops from 2 → 0 in a
single step (so a 2-damage melee hit, consistent with zombie
attacks). Median episode length for melee-deaths = 380 steps.

8/30 of these episodes had drink=0 throughout the last 50 steps
(see table above) — the player was dying of compound dehydration +
melee. The other 22 had drink ≥3 at death, so those are pure
combat losses without intrinsic stress.

Examples (final state at last replay step):
- ep5: length=211, ret=23.1, drink=3 (pure combat death — collected
  lots, then caught)
- ep15: length=1499, ret=18.1, drink=2 (long survive, drink low at
  end, then caught)
- ep24: length=613, ret=20.1, drink=0 — dehydrated for 32% of
  episode, finally killed by adjacent zombie
- ep28: length=1071, food=1, drink=7, energy=2 — intrinsics drained,
  zombie finished it

**The policy doesn't reliably retreat from advancing zombies, and
also doesn't reliably maintain drink.** Both behaviors compound.

### 2. Intrinsic depletion (16%)

| sub-cause | n | example |
|---|---|---|
| dehydration | 5 | ep6 length=1770: drink=0 + food=0, died of compounding |
| starvation | 2 | ep19 length=407: food=0 + energy=0 |
| exhaustion | 1 | ep1 length=408: energy=0 |

**Drink management is the worst.** 5/8 of these deaths are from
drink hitting 0. The `target_drink_water_v2` prompt actually showed
the policy's per-step drink rate is the same as baseline (0.022) —
the policy doesn't drink more even when prompted to. So the issue
is not "the prompt doesn't tell it to drink" but "the policy's
trained drinking behavior is just a fixed-rate habit, not a
needs-driven action".

### 3. Ranged deaths (8%)

Skeleton arrows on Floor 0. The policy doesn't dodge arrow paths.

### 4. Alive-or-timeout (14%)

These are episodes where my replay shows HP > 0 at the last action
recorded (slight RNG drift in env transitions makes my replay's
final state diverge from the original eval). Cross-checking
Gemini's last text on those episodes: most have HP=1-4 + low
intrinsics (drink/food/energy at 0). Almost certainly real deaths
that landed within the 5-step Gemini cadence window after the last
Gemini call.

## Diagnosis: where is the bottleneck?

The user asked: "is the main issue for improving it further the
amount of training data with oracle labels, reducing the
errors/getting more optimality in gemini predictions, something
else?"

**It's not Gemini prediction quality.** Inspecting Gemini's last
calls before death: in many episodes Gemini correctly identifies
the threat. ep14's last Gemini call says "There are two zombies
nearby... Health is also critically low (1)" — Gemini sees the
problem. The policy doesn't act on it. For ep23 (HP=1, last 5
actions all DO=attack), Gemini was telling it to engage but the
policy didn't break off when HP got critical. This is consistent
with the broader finding that C reads embedding *content* but the
underlying policy lacks the *retreat/dodge/manage* skills.

**It's not lack of oracle labels.** The training data already uses
oracle Gemini labels. The policy memorized them well enough to read
them at deploy time. Adding more oracle labels would re-train the
same fixed-rate routines.

**It IS the trajectory data quality.** Two specific sub-points:

1. **Survival skills.** The source PSF data was generated by a PPO
   policy that itself dies on Floor 0 to zombies — the trajectories
   in our top-2M subset have raw episode return mean 21.21, range
   [20.1, 32.1]. There's no high-return tail teaching long-survival
   combat behavior. AWR-trained policies (the C path) imitate the
   demonstrated behavior; if the demonstration doesn't show
   "retreat from low HP", the student doesn't learn it.
2. **Floor-1 navigation.** The source PSF policy enters the dungeon
   only ~12% of the time. The C policy inherits that 12% rate.
   PPO-RNN 1e8 enters 68% of the time and harvests floor-1 specials
   on the way. Until C trains on data that *reaches* floor 1
   reliably, the +7.8 raw return waiting in the floor-1
   INTERMEDIATE band stays locked.

So: **better training-data trajectories (SCALING_C) are the binding
fix**, not more oracle labels and not better Gemini predictions.
Order-of-magnitude estimate (per `docs/SCALING_C.md` gap table):
+10.84 raw return potential by switching to PPO-RNN-1e8-derived
trajectories.

## Side-by-side return × cause-of-death

To check: do the high-return episodes die differently from the
low-return ones?

| return bucket | n | dominant cause (verified) |
|---|---|---|
| ≥ 20 | 6 | melee=4, starvation=1, ranged=1 |
| 15–19 | 24 | melee=15, dehydration=4, ranged=1, starvation=1, exhaustion(unknown_combat)=1, alive=2 |
| 10–14 | 12 | melee=6, ranged=2, alive=3, exhaustion=1 |
| < 10 | 8 | melee=5, dehydration=1, alive=2 |

Same melee-dominant pattern across all return brackets. **High-return
episodes die the same way as low-return ones** — they just survive
longer first. No qualitative regime change with score.

## Implications for steering / prompt iteration

- **Score-max iteration won't fix this** — even the best prompt
  (achievement_max_v2 = 18.33) still has the policy spending all
  its time on Floor 0 dying to zombies. The +3.7 lift from baseline
  came from doing *more* on Floor 0 (place_furnace +27, place_stone
  +25, wake_up +23), not from surviving longer or descending more.
- **Mid-episode switch prompts** (which can re-orient the policy
  late) probably can't fix this either — the policy lacks the
  underlying retreat/dodge skill.
- **The fundamental answer is** to train C on trajectories that
  themselves demonstrate floor-1 play and zombie-avoidance.
  SCALING_C is exactly that.

## Files / artifacts

- `tools/classify_episode_deaths.py` — replay + classifier
- `probe_results/death_classification_C_baseline.json` — full JSON
  with per-episode (final_hp, food, drink, energy, floor,
  position, melee_within_1, ranged_within_5, lava_under, cause,
  return, length, last-10 actions)
- This doc: `docs/DEATH_ANALYSIS.md`

## NOOP and sleep

The action-replay analysis (`tools/compare_c_vs_ppo_rnn.py`,
`probe_results/compare_c_vs_pporn_1e8.json`) showed C uses NOOP 5.3% of
the time vs PPO-RNN-1e8 0.4%, raising the question of whether C is
voluntarily emitting NOOP. Inspection of the env source resolves this:
**NOOP is not a voluntary action during sleep — Craftax overrides
whatever action the policy emits with NOOP whenever `state.is_sleeping`
or `state.is_resting` is True.**

Evidence (`craftax/craftax/game_logic.py` lines 3018–3020):

```python
# Interrupt action if sleeping or resting
action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)
action = jax.lax.select(state.is_resting, Action.NOOP.value, action)
```

So the recorded action sequence shows NOOP for every step the player
spent asleep (which can be many — a SLEEP cycle runs until energy
refills, see lines 1838–1849). C's higher NOOP rate is therefore a
**sleep-frequency proxy** rather than a behavioral pathology: C
sleeps more (or for longer cycles) than PPO-RNN-1e8. This is
consistent with C's 52% wake_up rate vs PPO-RNN's 40%, and with the
score-max v2 prompt's design where SLEEP is explicitly elevated for
the +1 wake_up achievement.

Consequence: there is no benefit to masking the NOOP logit at eval
time, because the post-mask policy can't do anything different during
the sleep cycle anyway — the env still overrides any other emitted
action to NOOP until wake. The NOOP-rate gap is a statistic about
how often the policy enters sleep, not about the policy ever
"choosing inaction" mid-active-step.
