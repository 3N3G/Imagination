"""Single source of truth for the Craftax gameplay algorithm and the two
prompts derived from it.

If you want to change how Gemini reasons about Craftax — survival rules,
upgrade order, ladder logic — edit `GAMEPLAY_ALGORITHM` here. Every prompt
(future prediction in labelling / live eval, action selection in
gemini_play, the prompt-iteration webapp) reads from these constants, so
the change propagates automatically.

The static template file
`~/Craftax_Baselines/configs/future_imagination/templates/predict_state_only_prompt_concise.txt`
is a derived artifact for older code paths that read templates from disk
(`pipeline/gemini_label.py`, `eval/eval_online.py`,
`offline_rl/train_ppo_augmented.py`, `eval/eval_direction_counterfactual*.py`,
`eval/eval_hp_perturbation.py`). Regenerate it after editing this module:

    python tools/regenerate_prompt_templates.py
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Algorithm — the single source of truth for "how should the player act?"
# ---------------------------------------------------------------------------
GAMEPLAY_ALGORITHM = """\
At every step, the player should act with the goal of staying alive and progressing down floors.
This means the player will choose the highest-priority active goal in this order:
1. Survive
2. Take the ladder if it is open and on-screen
3. Upgrade equipment if survival is stable. This takes priority over taking the ladder if the player is in the overworld (floor 0) and has a sword or pickaxe worse than stone or missing.
4. Explore to find resources, troops, and the ladder

1. Survive
The player must track health, food, drink, and energy.  If food is <= 4, get food immediately by killing animals and eating them.  If drink is <= 4, get drink immediately from water tiles.  If energy is <= 4, make a safe enclosure and sleep.  If health is <= 4, restore food, drink, and energy before doing anything risky.  The player should never sleep in the open. Before sleeping, block enemies out, for example with stone walls. An easy way for the player to become safe is to mine a tunnel into a cluster of stone and place a stone behind blocking off the tunnel.

2. Take the ladder if it is open and visible
If the ladder is open and on screen, the player should prioritize using it unless they are in the overworld. The overworld has the majority of the resources so the player should first acquire at least stone tools before leaving. Note that open and visible are not the same; an open ladder can be used, but the player still needs to find it (down_ladder) to use it. On later floors, the ladder opens only after 8 troops have been killed. Each time and only each time the player descends to a new floor, they will gain one player_xp, which can be used to upgrade one of three attributes:
1. Strength: increases max health and physical melee damage
2. Dexterity: increases max food, drink, and energy and slows their decrease
2. Intelligence: increases max mana, mana regeneration, and spell damage

3. Upgrade equipment
The player should upgrade only when survival is stable. The player must always craft the highest-tier item they have the materials for but are missing, skipping lower tiers entirely. Always evaluate the Upgrade Decision Tree below from top to bottom (Highest Tier -> Lowest Tier).

Crafting Costs & Requirements:
- Pickaxe/Sword: 1 Wood + 1 [Wood/Stone/Iron/Diamond].
- Armor: 3 [Iron/Diamond].
- Stations: Crafting Table (3 Wood), Furnace (1 Stone). Iron and Diamond tier items require being adjacent to both a Furnace and Crafting Table.
- Misc: Arrows (1 Wood + 1 Stone = 2), Torches (1 Wood + 1 Coal = 4).

Upgrade Decision Tree (Evaluate in order 1 -> 4):
1. Diamond Tier: If the player has Diamonds, Coal, and Wood: craft Diamond equipment.
2. Iron Tier: If the player has at least 1 Iron, 1 Coal, and 1 Wood:
- If a furnace and a table are adjacent: craft immediately.
- If no Furnace/Table placed: place a Crafting Table (3 Wood) and Furnace (1 Stone), THEN craft Iron Sword and/or Pickaxe. This takes priority over Stone Tier even if stations aren't yet placed.
3. Stone Tier: If the player lacks Iron/Diamond but has Stone and Wood: craft a Stone Pickaxe and Stone Sword.
4. Wood/Base Tier: If the player lacks Stone but has Wood: craft Wood tools.

Resource Gathering & Maintenance Rules:
- Target of Opportunity: Mine Coal whenever seen. Mine Iron whenever seen (requires Stone Pickaxe).
- Deficit Gathering:
  - If the player has no useful tools and <10 Wood: gather up to 10 Wood.
  - If the player has only Wood tools and <10 Stone: mine 10 Stone.
- Surplus: If the player has extra resources after securing tools/armor, craft arrows or torches.

4. Explore
If the player is not in immediate danger, the ladder is not in sight, and no immediate upgrade is available, the player should explore.
While exploring, the player should:
- look for the ladder
- kill troops if the ladder is still closed
- gather useful nearby resources, especially wood, stone, coal, iron, and diamonds"""


# ---------------------------------------------------------------------------
# Game-rule preamble (shared overview of mechanics + coordinate convention)
# ---------------------------------------------------------------------------
GAME_RULES_PREAMBLE = """\
Craftax overview:
Craftax is a game about exploring dungeons, mining, crafting and fighting enemies.

Use these game rules:
1) Coordinates are (Row, Column), centered on the player at (0,0).
   - Negative Row = UP, Positive Row = DOWN.
   - Negative Column = LEFT, Positive Column = RIGHT.
2) Intrinsics: Health, Food, Drink, Energy, Mana. All out of 9.
   - Food/Drink/Energy naturally decay.
   - If Food/Drink/Energy reaches 0, Health will decay.
   - If Food/Drink/Energy are maintained, Health can recover.
3) Floor progression uses ladders.
   - Descending requires reaching the ladder on the current floor.
   - On non-overworld floors, the ladder is generally closed until enough mobs are killed.
4) Actions: NOOP, LEFT, RIGHT, UP, DOWN, DO (interact/attack/mine/drink/eat),
   SLEEP, PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT,
   MAKE_{WOOD,STONE,IRON,DIAMOND}_{PICKAXE,SWORD}, REST, DESCEND, ASCEND,
   MAKE_ARROW, SHOOT_ARROW, CAST_FIREBALL, CAST_ICEBALL, PLACE_TORCH,
   DRINK_POTION_*, READ_BOOK, ENCHANT_*, LEVEL_UP_*."""


# ---------------------------------------------------------------------------
# Future prediction prompt (used by labelling, live eval, online RL)
# ---------------------------------------------------------------------------
FUTURE_PREDICT_PROMPT = f"""\
You are forecasting a plausible future for a Craftax state.

{GAME_RULES_PREAMBLE}

Here is a good algorithm the player will play the game by:
{GAMEPLAY_ALGORITHM}

Predict at a high level what the next five steps for the player will look like, given that they are following the algorithm. Do not forecast beyond five time steps! In particular, the player can move at most five tiles during these five steps. Reason during your state understanding about the most immediate next step according to the algorithm and then predict the player's immediate behavior.

State Understanding: <A few sentences analyzing the current scene. Focus on careful spatial reasoning of the relevant tiles or tiles near the player. >

Prediction: <1 sentence description of the high-level behavior of the player in the next five steps. E.g. "move right to the cluster of trees", or "chase and kill the cow above", or "move down to look for water", or "move up and left to the visible open ladder". Do not reference specific coordinates. >

Now, predict the future of the following state.

Current state:
{{current_state_filtered}}
"""


# ---------------------------------------------------------------------------
# Action selection prompt (used by gemini_play and the action-iteration webapp)
# ---------------------------------------------------------------------------
ACTION_SELECT_PROMPT = f"""\
You are playing Craftax. At every step, choose the single action that best follows this algorithm:

{GAMEPLAY_ALGORITHM}

Coordinates: (Row, Column) relative to player at (0,0).
  Negative Row = UP, Positive Row = DOWN.
  Negative Column = LEFT, Positive Column = RIGHT.

Available actions (only use these exact names):
NOOP, LEFT, RIGHT, UP, DOWN, DO, SLEEP, PLACE_STONE, PLACE_TABLE,
PLACE_FURNACE, PLACE_PLANT, MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE,
MAKE_IRON_PICKAXE, MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD,
REST, DESCEND, ASCEND, MAKE_DIAMOND_PICKAXE, MAKE_DIAMOND_SWORD,
MAKE_IRON_ARMOUR, MAKE_DIAMOND_ARMOUR, SHOOT_ARROW, MAKE_ARROW,
CAST_FIREBALL, CAST_ICEBALL, PLACE_TORCH, DRINK_POTION_RED,
DRINK_POTION_GREEN, DRINK_POTION_BLUE, DRINK_POTION_PINK,
DRINK_POTION_CYAN, DRINK_POTION_YELLOW, READ_BOOK, ENCHANT_SWORD,
ENCHANT_ARMOUR, MAKE_TORCH, LEVEL_UP_DEXTERITY, LEVEL_UP_STRENGTH,
LEVEL_UP_INTELLIGENCE, ENCHANT_BOW.

Output format (strict):
REASONING: <couple sentences of rationale>
ACTION: <single action name>

Do not output anything else.

Current state:
{{current_state_filtered}}
"""
