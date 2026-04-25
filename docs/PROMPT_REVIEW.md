# Steering Prompt Review

This doc lays out every steering-prompt variant alongside the BASE prompt so they can be reviewed quickly. For each variant:

- The intent label, target metric, and steering direction
- The full algorithm-section quote (the section between "Here is ... algorithm ..." and "Predict at a high level ...") verbatim, in a blockquote
- Any change to the closing `Prediction:` directive line
- Sanity-check questions you can answer by eye, to verify the algorithm carries the intent

Read top-to-bottom. For each variant, decide:

- (a) approve as-is
- (b) request edits
- (c) flag as not matching the intended steering direction

The base prompt is at `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt`. Lines 1-22 (preamble + game rules) and lines 77-80 (current-state placeholder) are byte-identical across every variant; only the algorithm section (lines 23-69 of base) and the closing `Prediction:` directive vary.

---

## Base prompt — algorithm section (for comparison)

Path: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt`

> Here is a good algorithm the player will play the game by:
> At every step, the player should act with the goal of staying alive and progressing down floors.
> This means the player will choose the highest-priority active goal in this order:
> 1. Survive
> 2. Take the ladder if it is open and on-screen
> 3. Upgrade equipment if survival is stable. This takes priority over taking the ladder if the player is in the overworld (floor 0) and has a sword or pickaxe worse than stone or missing.
> 4. Explore to find resources, troops, and the ladder
>
> 1. Survive
> The player must track health, food, drink, and energy.  If food is <= 4, get food immediately by killing animals and eating them.  If drink is <= 4, get drink immediately from water tiles.  If energy is <= 4, make a safe enclosure and sleep.  If health is <= 4, restore food, drink, and energy before doing anything risky. To avoid losing health, avoid arrows, and avoid enemies unless you can kill them. The player should never sleep in the open. Before sleeping, block enemies out, for example with stone walls. An easy way for the player to become safe is to mine a tunnel into a cluster of stone and place a stone behind blocking off the tunnel.
>
> 2. Take the ladder if it is open and visible
> If the ladder is open and on screen, the player should prioritize using it unless they are in the overworld. The overworld has the majority of the resources so the player should first acquire at least stone tools before leaving. Note that open and visible are not the same; an open ladder can be used, but the player still needs to find it (down_ladder) to use it. On later floors, the ladder opens only after 8 troops have been killed. Each time and only each time the player descends to a new floor, they will gain one player_xp, which can be used to upgrade one of three attributes:
> 1. Strength: increases max health and physical melee damage
> 2. Dexterity: increases max food, drink, and energy and slows their decrease
> 2. Intelligence: increases max mana, mana regeneration, and spell damage
>
> 3. Upgrade equipment
> The player should upgrade only when survival is stable. The player must always craft the highest-tier item they have the materials for but are missing, skipping lower tiers entirely. Always evaluate the Upgrade Decision Tree below from top to bottom (Highest Tier -> Lowest Tier).
>
> Crafting Costs & Requirements:
> - Pickaxe/Sword: 1 Wood + 1 [Wood/Stone/Iron/Diamond].
> - Armor: 3 [Iron/Diamond].
> - Stations: Crafting Table (3 Wood), Furnace (1 Stone). Crafting anything requires being adjacent to a Crafting Table. Iron and Diamond tier items also require being adjacent to a Furnace.
> - Misc: Arrows (1 Wood + 1 Stone = 2), Torches (1 Wood + 1 Coal = 4).
>
> Upgrade Decision Tree (Evaluate in order 1 -> 4):
> 1. Diamond Tier: If the player has Diamonds, Coal, and Wood: craft Diamond equipment.
> 2. Iron Tier: If the player has at least 1 Iron, 1 Coal, and 1 Wood:
> - If a furnace and a table are adjacent: craft immediately.
> - If no Furnace/Table placed: place a Crafting Table (3 Wood) and Furnace (1 Stone), THEN craft Iron Sword and/or Pickaxe. This takes priority over Stone Tier even if stations aren't yet placed.
> 3. Stone Tier: If the player lacks Iron/Diamond but has Stone and Wood: craft a Stone Pickaxe and Stone Sword.
> 4. Wood/Base Tier: If the player lacks Stone but has Wood: craft Wood tools.
>
> Resource Gathering & Maintenance Rules:
> - Target of Opportunity: Mine Coal whenever seen. Mine Iron whenever seen (requires Stone Pickaxe).
> - Deficit Gathering:
>   - If the player has no useful tools and <10 Wood: gather up to 10 Wood.
>   - If the player has only Wood tools and <10 Stone: mine 10 Stone.
> - Surplus: If the player has extra resources after securing tools/armor, craft arrows or torches.
>
> 4. Explore
> If the player is not in immediate danger, the ladder is not in sight, and no immediate upgrade is available, the player should explore.
> While exploring, the player should:
> - look for the ladder
> - kill troops if the ladder is still closed
> - gather useful nearby resources, especially wood, stone, coal, iron, and diamonds

Base `Prediction:` directive (line 75):

> Prediction: <1 sentence description of the high-level behavior of the player in the next five steps. E.g. "move right to the cluster of trees", or "chase and kill the cow above", or "move down to look for water", or "move up and left to the visible open ladder". Do not reference specific coordinates. >

---

## Summary table

| Variant | Direction | Target metric | New? |
|---|---|---|---|
| die_v2 | down | episode return / survival_steps | existing |
| adversarial_v2 | down | episode return | existing |
| avoid_water_v2 | down | drink_intake events | existing |
| avoid_animals_v2 | down | cow eats + plant eats | existing |
| target_avoid_stone_v2 | down | inventory.stone increments / ep | NEW |
| target_stay_overworld_v2 | down to ~0 | DESCEND action count | NEW |
| target_collect_stone_v2 | up | inventory.stone increments / ep | existing |
| target_descend_v2 | up | DESCEND / enter_dungeon | existing |
| target_eat_cow_v2 | up | cow eats per ep | existing |
| target_drink_water_v2 | up | drink_intake per ep | existing |
| target_place_stone_v2 | up | PLACE_STONE actions | existing |
| target_hunt_animals_v2 | up | cow eats + plant eats | existing |
| target_collect_sapling_v2 | up | saplings collected / ep | NEW |
| target_place_plant_v2 | up | PLACE_PLANT actions | NEW |
| target_defeat_zombie_v2 | up | zombies killed / ep | NEW |
| target_collect_diamond_v2 | up | diamonds mined / ep | NEW |
| target_make_iron_pickaxe_v2 | up | make_iron_pickaxe achievement | NEW |
| direction_left_v2 | up | LEFT action count | existing |
| direction_right_v2 | up | RIGHT action count | existing |
| direction_up_v2 | up | UP action count | existing |
| direction_down_v2 | up | DOWN action count | existing |
| v2_basic_coverage | n/a (better baseline) | basic-tier achievement coverage | existing patch |
| v2_long_tail | n/a (better baseline) | long-tail achievement coverage | existing patch |

---

## Adversarial / "make policy worse"

### die_v2

- **Intent**: make policy seek death (drive return down)
- **Target metric**: episode return / survival steps
- **Direction**: down
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_die_v2.txt`

> Here is the algorithm the player will play the game by:
> At every step, the player will act with the goal of reaching zero health as
> quickly as possible. The player will choose the highest-priority active goal
> in this order:
> 1. Seek damage
> 2. Neglect intrinsics
> 3. Engage threats without preparation
> 4. Descend early
>
> 1. Seek damage
> The player will move toward nearby hazards: lava tiles, zombies, skeletons,
> and arrow traps. If water is adjacent and drink is not needed, the player
> will step into it. The player will stand in enemy line-of-fire when possible.
>
> 2. Neglect intrinsics
> The player will not eat even if food is low, not drink even if drink is low,
> and not sleep even if energy is low. The player will avoid crafting tools or
> armor and will not heal.
>
> 3. Engage threats without preparation
> The player will initiate melee with any visible mob regardless of health or
> equipment. The player will not place stone walls, not flee, and not kite.
>
> 4. Descend early
> If a ladder is visible, the player will descend immediately without gathering
> resources or tools, so that the next floor's threats arrive sooner.

- **Prediction-directive change**: none — base directive unchanged.
- **Sanity-check questions**:
  - Does the algorithm explicitly aim at zero health / death? (intent: yes)
  - Does it avoid healing/eating/drinking/sleeping (the things that prolong life)? (intent: yes)

### adversarial_v2

- **Intent**: worst plausible decisions (drive return down via low-EV play, not direct death-seeking)
- **Target metric**: episode return
- **Direction**: down
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_adversarial_v2.txt`

> Here is the algorithm the player will play the game by:
> At every step, the player will make the lowest-expected-value decision that is
> still plausible for a confused beginner. The player will choose the
> highest-priority active goal in this order:
> 1. Waste time
> 2. Craft the wrong thing
> 3. Mis-prioritize survival
> 4. Wander away from the ladder
>
> 1. Waste time
> The player will NOOP, place and pick up stones, or walk in a tight circle
> around their current tile. The player will repeatedly attempt actions that
> cannot succeed from the current state (e.g. MAKE_IRON_PICKAXE with no iron).
>
> 2. Craft the wrong thing
> When the player does craft, the player will choose lower-tier items over
> higher-tier ones, or craft decorative placements (torches, plants) instead of
> tools when survival needs tools. The player will skip placing a Furnace even
> when iron is available.
>
> 3. Mis-prioritize survival
> If food is low but water is closer, the player will drink; if drink is low
> but a tree is closer, the player will chop wood. The player will sleep in the
> open without walls.
>
> 4. Wander away from the ladder
> If a ladder is visible, the player will turn perpendicular to it and walk
> away, even on non-overworld floors where survival pressure is high.

- **Prediction-directive change**: none — base directive unchanged.
- **Sanity-check questions**:
  - Does the algorithm describe low-EV / wasteful play (NOOP, wrong crafts, mis-priorities)? (intent: yes)
  - Note: this is "play badly" not "die fast" — the death angle belongs to die_v2, this should erode return more gradually.

---

## Avoid (negative-direction steering on a class)

### avoid_water_v2

- **Intent**: drive drink_intake events down (policy stops drinking from water tiles)
- **Target metric**: drink_intake / collect_drink achievement events per episode
- **Direction**: down
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_avoid_water_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats water as navigationally opaque — water tiles act like walls
> that the player routes around. The player's priorities never involve water.
> At every step, the player will choose the highest-priority active goal in
> this order, ignoring any water tile when picking a target:
> 1. Move toward trees, stone, coal, iron, diamond, animals, crafting stations,
>    or the ladder — whichever is nearest and on the landward side.
> 2. Craft, place, or upgrade equipment when standing adjacent to the required
>    stations.
> 3. Explore inland — move toward any dry-tile direction that extends the
>    player's view of unexplored territory.
>
> If the nearest resource is on the far side of a water body, the player will
> route around the water by moving along the shore. If surrounded by water on
> multiple sides, the player will step onto the driest available tile that is
> farthest from water.
>
> The player restores Drink by drinking potions rather than by drinking from
> water tiles. If no potion is available, Drink is allowed to decay.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move left to the cluster of trees.", "Move down-right to the stone cluster.", "Chase the cow directly above.", "Move up and left to the visible open ladder." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation of water — just state the landward direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm say to never drink from water tiles? (intent: yes)
  - Does it route around water rather than through it? (intent: yes)
  - Has the `Prediction:` line been rewritten to ban "instead of / avoid / away from" so the LLM produces positive direction statements? (intent: yes)

### avoid_animals_v2

- **Intent**: drive cow eats AND plant eats both down
- **Target metric**: eat_cow + eat_plant achievement counts per episode
- **Direction**: down
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_avoid_animals_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats animals (cows and plants) as navigationally opaque — their
> tiles act like walls that the player routes around. The player's priorities
> never involve animals. At every step, the player will choose the
> highest-priority active goal in this order, ignoring any cow or plant tile
> when picking a target:
> 1. Move toward trees, stone, coal, iron, diamond, water, crafting stations,
>    or the ladder — whichever is nearest and away from any visible cow.
> 2. Craft, place, or upgrade equipment when standing adjacent to the required
>    stations.
> 3. Explore — move toward any direction that extends the player's view of
>    unexplored territory while keeping distance from visible cows.
>
> If the nearest resource is on the far side of a cow, the player will route
> around the cow by moving along a wider arc. If surrounded by cows on multiple
> sides, the player will step toward the non-cow tile with the largest open
> area.
>
> The player restores Food by eating cooked meat that is already in the
> inventory, or by harvesting plants for fruit when plants are adjacent. The
> player never attacks cows, never uses DO on a cow, and never moves onto a
> tile adjacent to a cow. If no alternative food source is available, Food is
> allowed to decay.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move left to the cluster of trees.", "Move down to the water tiles to drink.", "Move up-right to the stone cluster.", "Move up and left to the visible open ladder." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation of cows/animals — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm route around BOTH cows AND plants, or only cows? **POTENTIAL MISMATCH**: the intro says "treats animals (cows and plants)" but the priority list and detailed rules only call out cows. The food-source rule even says "harvesting plants for fruit when plants are adjacent", which directly enables eat_plant events the intent wants down.
  - Does it tell the player not to attack/eat cows? (intent: yes for cows)
  - Does it forbid plant harvest? (intent: should, but currently allows it)

### target_avoid_stone_v2 (NEW)

- **Intent**: drive stone mined per episode down
- **Target metric**: inventory.stone increments per episode
- **Direction**: down
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_avoid_stone_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats stone tiles as navigationally opaque — stone tiles act like walls that the player routes around. The player's priorities never involve stone. At every step, the player will choose the highest-priority active goal in this order, ignoring any stone tile when picking a target:
> 1. Move toward trees, water, animals, plants, ladders, or crafting stations — whichever is nearest and not behind a stone wall.
> 2. Craft, place, or upgrade equipment using only wood-based materials when standing adjacent to a placed table.
> 3. Explore — move toward any direction that extends the player's view of unexplored grass, water, or forest territory while keeping clear of stone clusters.
>
> If the nearest resource is on the far side of a stone cluster, the player will route around the cluster by moving along its edge. If surrounded by stone on multiple sides, the player will step toward the open tile with the largest non-stone area.
>
> The player keeps wood tools but never upgrades to stone or iron tier. Crafting tables are placed on dirt or grass. Furnaces are NOT placed (they require stone). The player relies on cow meat and plant fruit for food, water for drink, and a sapling-planted area for cover.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move left to the cluster of trees.", "Move down to the water tiles to drink.", "Move up-right to the cow.", "Move up and left to the visible open ladder." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm say to never mine stone? (intent: yes — it bans upgrading past wood tier and bans furnace placement, which together strongly imply no stone mining)
  - Does it route around stone clusters? (intent: yes)

### target_stay_overworld_v2 (NEW)

- **Intent**: drive DESCEND actions to ~0 (player stays on Floor 0)
- **Target metric**: DESCEND action count per episode (also enter_dungeon achievement)
- **Direction**: down (toward 0)
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_stay_overworld_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats DESCENDING THE LADDER as off-limits. The player stays on the overworld (Floor 0) for the entire episode. Ladder tiles are treated as ordinary terrain — the player walks past them without descending. The player's priorities are entirely overworld-bound: gather wood/stone/cows/water and survive on Floor 0.
>
> The player will choose the highest-priority active goal in this order, NEVER selecting DESCEND or ASCEND:
> 1. Survive (food, drink, energy, health).
> 2. Upgrade equipment to stone tier (wood pickaxe, then stone pickaxe + sword).
> 3. Stockpile resources visible on the overworld (extra wood, extra stone, place torches, place plants).
> 4. Explore the overworld in directions away from any visible ladder tile.
>
> If the player encounters a ladder tile, the player walks past it without using DESCEND. If the player is standing on a ladder, the player steps off. The player ignores the ladder entirely — its presence is irrelevant to the algorithm.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move left to the cluster of trees and chop wood.", "Mine the adjacent stone tile to upgrade tools.", "Move up to the cow and attack it for food.", "Place a stone wall to the right to block enemies." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm forbid DESCEND? (intent: yes)
  - Does it tell the player to walk past / step off ladders? (intent: yes)
  - Note: priority 4 says "explore in directions away from any visible ladder tile" — minor issue that this is a soft negation, but the prediction directive scrubs it from the output sentence.

---

## Target (positive-direction on a class)

### target_collect_stone_v2

- **Intent**: drive stone mined per episode up
- **Target metric**: inventory.stone increments per episode (collect_stone achievement)
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_collect_stone_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats stone collection as the single dominant priority. Every
> step is chosen to bring the player closer to mining the next stone tile.
> At every step, the player will choose the highest-priority active goal in
> this order:
> 1. Move toward the nearest visible stone tile and use DO to mine it when
>    adjacent. If the player needs a wood pickaxe to mine stone, the player
>    moves to wood and crafts the pickaxe as a brief detour.
> 2. If multiple stone tiles are visible, route to the closest cluster.
> 3. If no stone is visible, move toward the unexplored direction with the
>    highest chance of revealing stone (typically toward darker tiles or
>    away from grass).
>
> The player will only briefly interrupt stone collection to top up Food,
> Drink, or Energy when they fall to 1 (the moment before health decay).
> Once the immediate intrinsic is restored, the player resumes the stone
> priority. The player ignores cows, plants, ladders, monsters, water (when
> not at 1 drink), and ornamental tiles.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move down-right to the stone cluster.", "Move left to the visible stone tiles and mine them.", "Move up to the wood and then back down to the stone cluster.", "Mine the adjacent stone tile to the left." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm prioritize mining stone above all else? (intent: yes)
  - Does it allow brief detour for wood pickaxe (necessary prereq)? (intent: yes)

### target_descend_v2

- **Intent**: drive DESCEND actions / enter_dungeon up
- **Target metric**: DESCEND action count, enter_dungeon achievement
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_descend_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats descending the ladder as the single dominant priority.
> Every step is chosen to bring the player closer to the visible ladder and
> to the DESCEND action. At every step, the player will choose the
> highest-priority active goal in this order:
> 1. If a ladder is visible, the player moves directly toward it along the
>    shortest available path and uses DESCEND when standing on it.
> 2. If no ladder is visible, the player moves toward the unexplored
>    direction most likely to reveal one (typically deeper into stone or
>    toward the map edge in the largest unexplored quadrant).
> 3. The player gathers wood and stone only when needed to mine through a
>    blocking obstacle on the path to the ladder.
>
> The player will only briefly interrupt the descent priority to top up
> Food, Drink, or Energy when they fall to 1. The player ignores
> cows, plants, monsters that are not blocking the path, optional crafting,
> and ornamental tiles.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move down to the visible open ladder and descend.", "Move right toward the unexplored region to find the ladder.", "Mine the stone wall to the left to clear a path toward the ladder.", "Descend the ladder directly below the player." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm say to take the ladder as soon as one is visible? (intent: yes)
  - Does it avoid distracting goals (crafting, hunting)? (intent: yes)

### target_eat_cow_v2

- **Intent**: drive cow eats per episode up
- **Target metric**: eat_cow achievement count
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_eat_cow_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats hunting and eating cows as the single dominant priority.
> Every step is chosen to bring the player closer to a cow and into a
> position to attack it with DO. At every step, the player will choose the
> highest-priority active goal in this order:
> 1. Move directly toward the nearest visible cow and use DO when adjacent
>    to it. The player attacks repeatedly with DO until the cow drops meat,
>    then steps onto the meat and uses DO again to eat it.
> 2. If multiple cows are visible, route to the closest one. If a cow is
>    directly adjacent, the player attacks it immediately.
> 3. If no cow is visible, move toward the unexplored direction most likely
>    to contain grassland (typically toward open green tiles).
>
> The player will only briefly interrupt cow-hunting to top up Drink or
> Energy when they fall to 1. The player ignores stones, ladders, plants,
> crafting stations, and ornamental tiles. Wood is gathered only if needed
> to make a wood sword for more efficient cow-killing.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Chase the cow directly above and attack it.", "Move right toward the visible cow and attack with DO.", "Move up-right toward the nearest cow.", "Attack the adjacent cow to the left." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm hunt cows aggressively? (intent: yes)
  - Note: a craftax cow drops meat which then must be eaten — the prompt covers both attack and eat-the-meat sub-steps.

### target_drink_water_v2

- **Intent**: drive drink_intake per episode up
- **Target metric**: collect_drink / drink_intake events per episode
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_drink_water_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats drinking from water as the single dominant priority.
> Every step is chosen to bring the player closer to a water tile and into
> a position to use DO on it. At every step, the player will choose the
> highest-priority active goal in this order:
> 1. Move directly toward the nearest visible water tile and use DO when
>    standing adjacent to it.
> 2. If multiple water tiles are visible, route to the closest one. The
>    player will use DO repeatedly while adjacent to water to fill Drink
>    to maximum.
> 3. If no water is visible, move toward the unexplored direction most
>    likely to reveal water (typically toward blue tiles or the largest
>    open quadrant on the map edge).
>
> The player will only briefly interrupt water-drinking to top up Food or
> Energy when they fall to 1. The player ignores cows, plants, monsters,
> ladders, and crafting stations.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move left to the adjacent water tile and drink.", "Move down-right to the visible water and drink.", "Drink from the water tile directly to the left.", "Move up to the river and drink." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm prioritize drinking from water above other goals? (intent: yes)
  - Does it use DO repeatedly while adjacent to water (so drink stays maxed and re-fills as it decays)? (intent: yes)

### target_place_stone_v2

- **Intent**: drive PLACE_STONE actions per episode up
- **Target metric**: PLACE_STONE action count
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_place_stone_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats placing stone walls as the single dominant priority.
> Every step is chosen to bring the player closer to placing a stone tile.
> At every step, the player will choose the highest-priority active goal in
> this order:
> 1. If the player has stone in inventory and is standing next to an empty
>    tile, the player uses PLACE_STONE on that empty tile to build a wall.
> 2. If the player has no stone in inventory, the player moves toward the
>    nearest visible stone tile and mines it with DO when adjacent.
> 3. The player builds compact stone enclosures wherever space allows,
>    placing stone in every direction the player can reach.
>
> The player will only briefly interrupt stone-placing to top up Food,
> Drink, or Energy when they fall to 1. The player ignores cows, plants,
> monsters, ladders, and crafting stations beyond what is needed to
> sustain placing more stone.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Place a stone on the empty tile directly to the right.", "Move left to the stone cluster and mine it for placement material.", "Place a stone above and another one to the left to enclose the player.", "Mine the adjacent stone tile and then place it back." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and action. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm prioritize PLACE_STONE whenever inventory + adjacent empty tile permit? (intent: yes)
  - Does it tell the player to mine more stone as raw material when inventory is empty? (intent: yes)

### target_hunt_animals_v2

- **Intent**: drive cow eats + plant eats both up (combined hunt count)
- **Target metric**: eat_cow + eat_plant achievements per episode
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_hunt_animals_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats ALL visible animals (cows and plants) as immediate hunt
> targets. The moment any animal is visible, the player redirects to attack
> or harvest it. At every step, the player will choose the highest-priority
> active goal in this order:
> 1. If any cow is visible, move directly toward the nearest cow and use DO
>    to attack it repeatedly until it drops meat. Step onto the meat and
>    use DO to eat.
> 2. If any plant with fruit is visible (not a sapling), move directly toward
>    it and use DO to harvest.
> 3. If no animals are visible, move toward the unexplored direction most
>    likely to reveal grassland or plant tiles (typically toward open green
>    tiles).
>
> After each animal is eaten, the player immediately searches for the next
> visible animal and repeats. The player will briefly interrupt hunting
> only to top up Drink or Energy when they fall to 1. The player ignores
> stones, ladders, crafting stations, and ornamental tiles. Wood is gathered
> only if needed to craft a wood sword for more efficient hunting.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Chase the adjacent cow to the left and attack it with DO.", "Move up-right toward the nearest cow a few tiles away.", "Harvest the fruit-bearing plant directly above.", "Attack the cow and then move right to the next visible cow." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm tell the player to hunt BOTH cows AND fruit-bearing plants? (intent: yes — explicitly distinguishes from saplings)
  - Cow priority comes before plant — is that the intended ordering, or should they be tied? (note: ranking cow first means early plants near the player may be skipped if a cow is also visible)

### target_collect_sapling_v2 (NEW)

- **Intent**: drive saplings collected per episode up
- **Target metric**: collect_sapling achievement / sapling inventory increments
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_collect_sapling_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats COLLECTING SAPLINGS as the single dominant priority. Saplings are small green plants that grow on grass tiles. The player uses DO when adjacent to a sapling tile to add it to inventory. The player accumulates as many saplings as possible.
>
> The player will choose the highest-priority active goal in this order:
> 1. Survive (only if intrinsic ≤ 1)
> 2. Move toward the nearest visible sapling tile and use DO when adjacent to collect it.
> 3. If no sapling is visible, move toward the unexplored direction most likely to contain grass / saplings (typically open green areas).
>
> A sapling tile is a small visible plant on a grass tile, distinct from a fruit-bearing plant (a larger plant the player can eat). The player should DO on sapling tiles to add them to inventory. The player ignores stones, ladders, monsters, water (unless drink ≤ 1), and crafting stations.
>
> If the player picks up a sapling, the player continues to look for the next sapling. The goal is the per-episode count of saplings collected — so the player keeps collecting throughout the episode.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move down-right to the visible sapling and collect it with DO.", "Collect the sapling directly to the left and continue right toward the next sapling.", "Move up to the grass area to look for saplings.", "Use DO on the adjacent sapling tile." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm explicitly distinguish saplings from fruit-bearing plants? (intent: yes — saplings are inventory items, not food)
  - Does it tell the player to keep collecting throughout the episode (rather than stop after one)? (intent: yes)

### target_place_plant_v2 (NEW)

- **Intent**: drive PLACE_PLANT actions per episode up
- **Target metric**: PLACE_PLANT action count / place_plant achievement
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_place_plant_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats PLACE_PLANT as the single dominant priority. PLACE_PLANT puts a sapling from the inventory onto a grass tile, where it eventually grows into a fruit-bearing plant. The player accumulates PLACE_PLANT actions throughout the episode.
>
> The player will choose the highest-priority active goal in this order:
> 1. Survive (only if intrinsic ≤ 1)
> 2. If the player has at least one sapling AND is standing on a grass tile, the next action is PLACE_PLANT.
> 3. If the player has a sapling but is not on grass, the next action is to move to the nearest visible grass tile.
> 4. If the player has no saplings, the next action is to walk to the nearest visible sapling and DO to collect it.
> 5. If neither saplings nor grass tiles are visible, explore toward the most-grass-likely direction (typically open green areas).
>
> The player should keep at least one sapling in inventory at all times once any are collected. After PLACE_PLANT-ing, the player immediately looks for another sapling so the cycle continues. The player ignores stones, ladders, monsters that aren't blocking the path, and crafting (other than maybe a wood sword for cow defense).

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Place a plant on the grass tile directly below.", "Move down to the grass area and place a plant.", "Move right to the visible sapling and collect it.", "Place a plant on the adjacent grass tile, then move up to the next sapling." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm chain "collect sapling -> stand on grass -> PLACE_PLANT" so the action can fire? (intent: yes — full pipeline covered)
  - Does it tell the player to repeat the cycle so PLACE_PLANT count keeps growing? (intent: yes)

### target_defeat_zombie_v2 (NEW)

- **Intent**: drive zombies killed per episode up
- **Target metric**: defeat_zombie achievement count per episode
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_defeat_zombie_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats DEFEATING ZOMBIES as the single dominant priority. The player engages every visible zombie and uses DO repeatedly until it is killed. The player accumulates zombie kills throughout the episode.
>
> The player will choose the highest-priority active goal in this order:
> 1. Survive (only if health ≤ 2 — a zombie hits for 1 damage so health ≤ 2 means risk of death from the next hit)
> 2. Move directly toward the nearest visible zombie and use DO repeatedly when adjacent to it.
> 3. If no zombie is visible, move toward the unexplored direction most likely to contain a zombie (typically dark areas, dungeons, night-time spawns).
>
> The player should craft a wood sword (1 Wood, table needed) for ×2 zombie damage, then a stone sword (1 Wood + 1 Stone, table needed) for ×3 damage. After that, every visible zombie is engaged immediately. The player ignores cows (unless food ≤ 2), plants, ladders, water (unless drink ≤ 1), and ornamental tiles.
>
> If multiple zombies are visible, the player engages the closest one first. After killing a zombie, the player immediately searches for the next.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Chase the zombie directly above and attack it with DO.", "Move right toward the visible zombie and attack with DO repeatedly.", "Move up-right toward the zombie a few tiles away.", "Attack the adjacent zombie to the left." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm engage every visible zombie? (intent: yes)
  - Does it tell the player to craft a sword first for higher damage? (intent: yes — a brief detour is allowed because it directly increases the success rate of the target metric)
  - Does it forbid descending? Note: ladders are listed in "ignores", so the player stays where zombies are visible rather than chasing dungeon spawns directly. Reviewer call: do you want zombie-hunting on overworld only (current), or to also descend to find more zombies?

### target_collect_diamond_v2 (NEW)

- **Intent**: drive diamonds mined per episode up
- **Target metric**: collect_diamond achievement count
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_collect_diamond_v2.txt`
- **Note from user**: this prompt explicitly walks through the iron-pickaxe chain (must mine wood -> stone -> coal -> iron -> make_iron_pickaxe -> descend -> mine_diamond) because diamond requires iron pickaxe. The user wanted strong emphasis on the chain without derailing the rest.

> Here is the algorithm the player will play the game by:
> The player treats DIAMOND COLLECTION as the single dominant priority. Diamond requires an Iron Pickaxe to mine, which requires the full crafting chain. The player executes the SHORTEST path to the first diamond and then mines as much diamond as possible.
>
> The player will choose the highest-priority active goal in this order:
> 1. Survive (only if intrinsic ≤ 1, the moment before health decay)
> 2. Advance the diamond-mining chain (the items below, in order)
> 3. Mine diamond when adjacent with iron pickaxe equipped
>
> Diamond-mining chain (do EACH item the moment its prerequisite is met; do NOT skip):
>   i.   Gather 4 Wood (chop trees with DO).
>   ii.  PLACE_TABLE as soon as 3 Wood are on hand.
>   iii. MAKE_WOOD_PICKAXE adjacent to the table.
>   iv.  Mine 5+ Stone with the wood pickaxe.
>   v.   PLACE_FURNACE adjacent to the table (needs 1 Stone).
>   vi.  MAKE_STONE_PICKAXE (needs 1 Wood + 1 Stone next to table).
>   vii. Mine Coal (with stone pickaxe). Coal is required for iron tools and shows as black-flecked stone.
>   viii.Mine Iron (with stone pickaxe). Iron shows as orange-flecked stone.
>   ix.  MAKE_IRON_PICKAXE (needs 1 Wood + 1 Iron + 1 Coal next to BOTH table and furnace). DO NOT SKIP THIS STEP.
>   x.   With iron pickaxe equipped, descend to floors that contain diamond (typically deeper floors). DESCEND open ladder.
>   xi.  Mine every visible diamond tile with DO. Diamond shows as blue-flecked stone.
>
> The player ignores cows, plants, water (unless drink ≤ 1), monsters that are not blocking the path, and ornamental tiles. Combat is avoided unless it is necessary to clear a ladder. Sleep only if energy ≤ 1.
>
> If the player is currently between chain steps, the next action is whatever advances the chain. If the player has all materials for an iron pickaxe but is not at table+furnace, the next action is to walk back to them. If the player has the iron pickaxe but no visible diamond, the next action is to descend or to mine through stone toward unexplored deeper terrain.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move right to mine the iron tile, then return to the furnace to craft the iron pickaxe.", "Descend the visible open ladder to look for diamond on the next floor.", "Move left to chop the tree, then place the table.", "Mine the diamond tile directly above with the iron pickaxe." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm spell out the full wood -> stone -> coal -> iron -> iron-pickaxe -> descend -> diamond chain in order? (intent: yes)
  - Does step ix make iron pickaxe explicit and "do not skip"? (intent: yes — addresses the user's concern about emphasizing the chain)
  - Does it still allow the player to give up combat / cow-hunting to stay on chain? (intent: yes)

### target_make_iron_pickaxe_v2 (NEW)

- **Intent**: drive make_iron_pickaxe achievement up
- **Target metric**: make_iron_pickaxe achievement (binary per ep, so this targets the first-time event)
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_target_make_iron_pickaxe_v2.txt`

> Here is the algorithm the player will play the game by:
> The player treats CRAFTING THE IRON PICKAXE as the single dominant priority. The iron pickaxe requires: 1 Wood + 1 Iron + 1 Coal, and the player must be adjacent to BOTH a placed crafting table AND a placed furnace at the moment of crafting.
>
> The player will choose the highest-priority active goal in this order:
> 1. Survive (only if intrinsic ≤ 1)
> 2. Advance the iron-pickaxe chain (the items below, in order)
> 3. The moment all materials are gathered AND the player is adjacent to table+furnace: MAKE_IRON_PICKAXE
>
> Iron-pickaxe chain (do EACH item at the first feasible opportunity; do NOT skip):
>   a. Gather 4 Wood (chop trees with DO).
>   b. PLACE_TABLE as soon as 3 Wood are on hand.
>   c. MAKE_WOOD_PICKAXE adjacent to the table.
>   d. Mine 3+ Stone with the wood pickaxe.
>   e. PLACE_FURNACE adjacent to the table (needs 1 Stone).
>   f. MAKE_STONE_PICKAXE (needs 1 Wood + 1 Stone next to table).
>   g. Mine Coal (with stone pickaxe). Coal shows as black-flecked stone.
>   h. Mine Iron (with stone pickaxe). Iron shows as orange-flecked stone.
>   i. Walk back to be adjacent to BOTH the table and the furnace.
>   j. MAKE_IRON_PICKAXE.
>
> The player ignores cows (unless food ≤ 1), plants, descending the ladder (unless the current floor has no iron and the next floor likely does), monsters that are not blocking the path, and ornamental tiles. Combat is avoided. Sleep only if energy ≤ 1.
>
> If the player has all required materials but is not at the table+furnace, the next action is to walk back. If the player is at the table+furnace but missing one material, the next action is to gather that material.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move right to mine the visible iron tile, then return to the furnace.", "Walk back left to the table+furnace and craft the iron pickaxe.", "Mine the coal tile directly above, then move down to the iron tile.", "Place the furnace adjacent to the table." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the direction and target. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm cover the full chain (wood -> table -> wood pick -> stone -> furnace -> stone pick -> coal -> iron -> table+furnace -> iron pick)? (intent: yes)
  - Step i ("walk back to BOTH table and furnace") — is this stated explicitly enough that the LLM will catch the dual-adjacency requirement? (intent: yes)
  - Does it allow descending only as a last resort if no iron on current floor? (intent: yes)

---

## Direction (cardinal-direction)

### direction_left_v2

- **Intent**: drive LEFT action count per episode up
- **Target metric**: LEFT action count
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_direction_left_v2.txt`

> Here is the algorithm the player will play the game by:
> The player walks LEFT (negative-column direction) at every step. Every
> action is the LEFT action unless a wall, water, or other obstacle blocks
> the next leftward tile, in which case the player uses DO once on the
> obstacle and then resumes walking left. Vertical position is held
> constant. The player does not turn, does not gather resources, does not
> craft, and does not interact with cows, plants, ladders, monsters, or
> crafting stations on the current row.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move left along the current row.", "Move left, mining the stone tile blocking the path.", "Walk left for five tiles." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the leftward action. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm output strictly LEFT actions? (intent: yes)
  - Does it tell the player NOT to interact with side targets that would consume the LEFT action? (intent: yes)

### direction_right_v2

- **Intent**: drive RIGHT action count per episode up
- **Target metric**: RIGHT action count
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_direction_right_v2.txt`

> Here is the algorithm the player will play the game by:
> The player walks RIGHT (positive-column direction) at every step. Every
> action is the RIGHT action unless a wall, water, or other obstacle blocks
> the next rightward tile, in which case the player uses DO once on the
> obstacle and then resumes walking right. Vertical position is held
> constant. The player does not turn, does not gather resources, does not
> craft, and does not interact with cows, plants, ladders, monsters, or
> crafting stations on the current row.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move right along the current row.", "Move right, mining the stone tile blocking the path.", "Walk right for five tiles." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the rightward action. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Same shape as direction_left_v2 with sign flipped. Approve as a pair if you approve direction_left.

### direction_up_v2

- **Intent**: drive UP action count per episode up
- **Target metric**: UP action count
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_direction_up_v2.txt`

> Here is the algorithm the player will play the game by:
> The player walks UP (negative-row direction) at every step. Every
> action is the UP action unless a wall, water, or other obstacle blocks
> the next upward tile, in which case the player uses DO once on the
> obstacle and then resumes walking up. Horizontal position is held
> constant. The player does not turn, does not gather resources, does not
> craft, and does not interact with cows, plants, ladders, monsters, or
> crafting stations on the current column.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move up along the current column.", "Move up, mining the stone tile blocking the path.", "Walk up for five tiles." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the upward action. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Same shape as direction_left_v2 with axis flipped to vertical. Approve with the family.

### direction_down_v2

- **Intent**: drive DOWN action count per episode up
- **Target metric**: DOWN action count
- **Direction**: up
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_direction_down_v2.txt`

> Here is the algorithm the player will play the game by:
> The player walks DOWN (positive-row direction) at every step. Every
> action is the DOWN action unless a wall, water, or other obstacle blocks
> the next downward tile, in which case the player uses DO once on the
> obstacle and then resumes walking down. Horizontal position is held
> constant. The player does not turn, does not gather resources, does not
> craft, and does not interact with cows, plants, ladders, monsters, or
> crafting stations on the current column.

- **Prediction-directive change**: rewritten to forbid negation phrasing —

> Prediction: <1 positive, direction-stating sentence describing the player's action. Good examples: "Move down along the current column.", "Move down, mining the stone tile blocking the path.", "Walk down for five tiles." Do not use "instead of", "rather than", "avoid", "away from", "refuse", or any other negation — just state the downward action. Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Same shape as direction_up_v2 with sign flipped. Approve with the family.

---

## Patch-by-prompt (better baselines, NOT steering)

### v2_basic_coverage

- **Intent**: patch B's basic-achievement gap (force the LLM to call out missing basic-tier moves like place_table, make_wood_sword, place_furnace)
- **Target metric**: basic-tier achievement coverage (place_table, make_wood_pickaxe, make_wood_sword, place_furnace, make_stone_pickaxe, make_stone_sword, place_torch, place_stone, make_iron_pickaxe, make_iron_sword, sleep, place_plant)
- **Direction**: n/a (this is a better-baseline patch, not a steering direction)
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_v2_basic_coverage.txt`

> Here is a good algorithm the player will play the game by:
> At every step, the player should act with the goal of staying alive and progressing down floors. The player must complete a checklist of basic mid-game actions BEFORE chasing optional goals. Skipping these wastes the resources the player already gathered.
>
> The player will choose the highest-priority active goal in this order:
> 1. Survive (health, food, drink, energy)
> 2. Complete the basic-coverage checklist below — every item is mandatory once the prerequisite is met
> 3. Take the ladder if it is open and visible
> 4. Explore to find resources and the ladder
>
> 1. Survive
> The player must track health, food, drink, and energy. If food <= 4, kill an animal and eat it immediately. If drink <= 4, drink from a water tile immediately. If energy <= 4, build a safe enclosure and SLEEP — never sleep in the open. If health <= 4, restore food/drink/energy first. Use one DO action per intended interaction; do NOT spam DO on a tile that did not respond — instead step away and re-approach if needed.
>
> 2. Basic-coverage checklist (do EACH at the first feasible opportunity)
> Each line below is a mandatory micro-goal. As soon as the prerequisite resources/conditions are met, perform the action; do not skip it to chase a later goal.
>
>   a. PLACE_TABLE as soon as the player has 3 Wood (needed for any crafting).
>   b. MAKE_WOOD_PICKAXE and MAKE_WOOD_SWORD as soon as the player has 1 Wood and is adjacent to a placed table.
>   c. Mine 5+ Stone with the wood pickaxe (provides material for stone tools, walls, and furnace).
>   d. PLACE_FURNACE as soon as the player has 1 Stone and is adjacent to the table (needed for iron tier).
>   e. MAKE_STONE_PICKAXE and MAKE_STONE_SWORD as soon as the player has 1 Wood and 1 Stone next to a table.
>   f. When the player picks up Coal: craft and PLACE_TORCH at least once (1 Wood + 1 Coal -> 4 torches via MAKE_TORCH; place at least one). Torches provide light and prevent monster spawns.
>   g. PLACE_STONE at least once to test wall-building (needed later to make safe enclosures and to block enemies).
>   h. When the player picks up Iron AND has Coal AND Wood AND is next to a table+furnace: MAKE_IRON_PICKAXE and MAKE_IRON_SWORD immediately. Do not skip iron tier.
>   i. When energy <= 5: build a 1-tile stone enclosure and SLEEP. Wake_up restores energy and health.
>   j. When the player collects a sapling: PLACE_PLANT on a grass tile within the next 5 steps. Plants grow into food.
>
> 3. Take the ladder if it is open and visible
> Once the basic-coverage checklist for the current tier is complete, take the visible open ladder. The overworld has the most resources so the player should reach at least stone tools (and ideally place one torch) before descending. Each descend grants 1 player_xp.
>
> 4. Explore
> If the player is not in immediate danger, the ladder is not in sight, and no checklist item applies, explore in the most-unexplored direction. While exploring: look for the ladder, kill troops if the ladder is still closed, gather wood/stone/coal/iron/diamond.

- **Prediction-directive change**: kept base-style examples but with a checklist hint —

> Prediction: <1 sentence description of the high-level behavior of the player in the next five steps. Examples: "Move right to the cluster of stone and mine three tiles.", "Place the table directly above and craft the wood pickaxe.", "Move down-left to the visible coal tile, mine it, then return to the table and make a torch.", "Build a one-tile stone enclosure to the right and SLEEP." Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm enumerate every basic-tier achievement as an explicit checklist item? (intent: yes — covers place_table, wood pick + sword, place_furnace, stone pick + sword, torch, place_stone, iron pick + sword, sleep, place_plant)
  - Is this clearly framed as a stronger-base prompt rather than a steering target (so it gets compared against unaug as a better baseline)? (intent: yes)

### v2_long_tail

- **Intent**: patch C's long-tail gap (force the LLM to surface easy-to-skip differentiator behaviors: plant, sleep, torch, arrow, iron tier, descend, eat plant, collect sapling)
- **Target metric**: long-tail achievement coverage (place_plant, wake_up/sleep, place_torch, make_arrow, make_iron_pickaxe + sword, descend events, eat_plant, collect_sapling)
- **Direction**: n/a (better-baseline patch)
- **Template path**: `/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_v2_long_tail.txt`

> Here is a good algorithm the player will play the game by:
> At every step, the player should act with the goal of staying alive AND completing the long-tail behaviors (sapling planting, sleeping, torch placement, descending) that are easy to skip but together compound into much higher returns. After basic survival and stone-tier crafting, the long-tail loop is the highest-EV way to spend additional steps — do not just keep walking.
>
> The player will choose the highest-priority active goal in this order:
> 1. Survive (health, food, drink, energy)
> 2. Reach stone-tier tools (wood pickaxe -> stone pickaxe + sword)
> 3. Long-tail loop (every item below is mandatory once feasible)
> 4. Take the ladder if it is open and on-screen — DESCEND immediately
> 5. Explore
>
> 1. Survive
> Track health, food, drink, energy. Eat a cow if food <= 4. Drink from water if drink <= 4. SLEEP in a safe enclosure if energy <= 4. The player must NEVER walk past a survival threshold without addressing it. Avoid lava, arrow traps, and unblocked enemies.
>
> 2. Reach stone-tier tools
> Place a crafting table once 3 Wood is on hand. Craft wood pickaxe + wood sword. Mine 5+ stone. Craft stone pickaxe + stone sword. Place a furnace next to the table when 1 stone and table are adjacent.
>
> 3. LONG-TAIL LOOP — these are the differentiators (each item is mandatory the first feasible time):
>
>   a. PLACE_PLANT — whenever the player has a sapling AND stands on a grass tile, PLACE_PLANT immediately. Plants grow into fruit (food source). Without saplings planted, the player has no late-game food backup.
>
>   b. SLEEP — whenever energy <= 5, build a 1-tile stone enclosure (mine a tunnel into a stone cluster, place a stone behind the player) and SLEEP until wake_up. The player CANNOT skip sleep — energy decays unrecoverably otherwise.
>
>   c. MAKE_TORCH + PLACE_TORCH — every time the player picks up Coal AND has Wood, MAKE_TORCH immediately (1 Wood + 1 Coal -> 4 torches) and PLACE_TORCH at least once nearby. Torches reveal map and prevent monster spawns. Without torches the player walks into ambushes.
>
>   d. MAKE_ARROW — whenever the player has 1 Wood + 1 Stone of surplus and a crafting table is reachable, MAKE_ARROW. Arrows are the only ranged option for skeletons and arrow traps.
>
>   e. MAKE_IRON_PICKAXE + MAKE_IRON_SWORD — the moment the player has Iron + Coal + Wood + table+furnace adjacent, craft both. Skipping iron tier is the single largest cause of late-game return loss.
>
>   f. DESCEND — whenever the player stands on an open ladder, DESCEND (do not walk past). Each descend = +1 xp + new resource biome.
>
>   g. EAT_PLANT — whenever the player passes a fruit-bearing plant, harvest it (DO).
>
>   h. COLLECT_SAPLING — whenever the player is adjacent to a sapling tile (small green plant), DO to collect it (saplings let you plant your own food source later).
>
> If the player is in immediate danger or pursuing a survive/stone-tier goal, defer the long-tail item — but RESUME it immediately after the prerequisite is handled. Do not let the policy "drift" into pure walking; every walk step should be heading toward a specific long-tail or ladder goal.
>
> 4. Take the ladder
> If a ladder is open and visible on the current screen, DESCEND immediately (do not skip) — even if the long-tail loop is incomplete on the current floor.
>
> 5. Explore
> Only when (1)-(4) have nothing immediately actionable, move toward the most-unexplored direction. Prefer directions that are likely to expose new resources (coal, iron, ladder) over already-mapped grass.

- **Prediction-directive change**: kept base-style examples but with a long-tail hint —

> Prediction: <1 sentence description of the high-level behavior of the player in the next five steps. Examples: "Plant a sapling on the grass tile directly to the right.", "Mine a stone tunnel two tiles down, place a stone behind, then SLEEP.", "Make a torch and place it against the stone wall to the left.", "Walk down to the open ladder and DESCEND." Do not reference specific coordinates. >

- **Sanity-check questions**:
  - Does the algorithm enumerate every long-tail behavior (place_plant, sleep, torch, arrow, iron tier, descend, eat_plant, collect_sapling)? (intent: yes — all eight covered)
  - Is the framing clear that the long-tail loop is highest-EV after basic survival is handled? (intent: yes)

---

## How to use this doc

For each variant section above, answer the sanity-check questions out loud (or in a follow-up message). If both answers are "yes", approve. If either is "no" or "ambiguous", flag for edit. The two variants most worth a careful look:

1. **avoid_animals_v2** — the body of the algorithm only describes cow-avoidance even though the intro mentions plants. If the intent is to drive eat_cow + eat_plant both down, the algorithm should also forbid harvesting fruit-bearing plants.
2. **target_hunt_animals_v2** — cow priority comes strictly before plant priority. If the metric is the sum of cow eats + plant eats, this ordering may slightly under-weight plants when both are visible.
