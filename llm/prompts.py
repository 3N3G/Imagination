"""
Shared LLM Prompt Utilities for Craftax

This module provides consistent prompts for all LLM-based components:
- llm_play/llm_play_harnessed.py (LLM gameplay)
- labelling/llm_worker.py (LLM labelling)
- online_rl_llm/online_rl_hidden.py (Online RL with LLM)
- offline_rl/awr_llm_augmented.py (Offline RL with LLM)

All LLM-related code should use these shared prompts for consistency.
"""

import re
import warnings
from typing import Optional


# Background tiles to filter out from observations
BACKGROUND_TILES = {
    "grass", "sand", "gravel", 
    "fire grass", "ice grass", "fire_grass", "ice_grass"
}


SYSTEM_PROMPT = """You are playing Craftax.

Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions and can interact. Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest. This interaction will only happen if the block/creature/chest is directly in front of the player, one step in the direction the player is facing. 
The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. Mana is used for casting spells or enchanting items and will naturally recover. Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. If the player's health falls beneath 0 they will die and the game will restart.

The coordinate system is (Row, Column). Everything is relative to your current position, and the map will show all interesting tiles (so tiles with something besides grass or other background tiles that you will always be able to walk on) within 5 columns and 4 rows of your current position.
- Negative Row is UP. Positive Row is DOWN.
- Negative Column is LEFT. Positive Column is RIGHT.
- (0, 0) is your current position.
- Example: (-1, 0) is one step UP. (0, 1) is one step RIGHT.

To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). There are 9 levels in total.

### GAMEPLAY ALGORITHM
The player should focus on three main aspects: staying alive, collecting resources or tools, and progressing.

**1. Check Intrinsics First**
Look at your health, food, drink, and energy. Without leveling up any stats, the max for each stat is 9.
- If your health is low or medium, make sure to fill up your food, drink, and energy in order to recover health.
- If any of your other intrinsics are medium, either recover it immediately or make sure you will have access to that resource once it becomes low.
- For food and drink: consume animals (cows, snails, bats, etc.) or drink water.
- For energy: you need to sleep. Make sure you are protected (e.g. closed off by stone walls) before going to sleep, otherwise enemies will attack and kill you.

**2. Collect Resources and Tools**
If your intrinsics are fine, collect resources and tools:
- Mine trees with your hand. Note that extra wood is helpful for crafting torches later.
- Once you have at least 2 wood, craft a crafting table. Then if you have wood, craft a wood pickaxe and sword.
- Mine stone and then craft a stone pickaxe and sword. Also mine coal whenever you see it, as it is helpful for crafting iron tools and torches. Note that extra stone is helpful for blocking off enemies to rest and recover.
- If you see iron, mine it, and if you have at least one iron, one coal, and one wood, you can craft an iron sword or pickaxe. This step is not as urgent since sometimes there may not be enough iron for all of these.

**3. Progress**
This means finding the ladder, which means looking for it and (on all floors after the overworld) killing 8 troops. If you see the ladder, and it is open, enter it. Otherwise keep exploring and staying alive.

Actions available: 
0:NOOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:DO (interact/mine/attack), 6:SLEEP, 7:PLACE_STONE,
8:PLACE_TABLE, 9:PLACE_FURNACE, 10:PLACE_PLANT, 11:MAKE_WOOD_PICKAXE, 12:MAKE_STONE_PICKAXE,
13:MAKE_IRON_PICKAXE, 14:MAKE_WOOD_SWORD, 15:MAKE_STONE_SWORD, 16:MAKE_IRON_SWORD, 17:REST,
18:DESCEND, 19:ASCEND, 20:MAKE_DIAMOND_PICKAXE, 21:MAKE_DIAMOND_SWORD, 22:MAKE_IRON_ARMOUR,
23:MAKE_DIAMOND_ARMOUR, 24:SHOOT_ARROW, 25:MAKE_ARROW, 26:CAST_FIREBALL, 27:CAST_ICEBALL,
28:PLACE_TORCH, 29-34:DRINK_POTION_(RED/GREEN/BLUE/PINK/CYAN/YELLOW), 35:READ_BOOK,
36:ENCHANT_SWORD, 37:ENCHANT_ARMOUR, 38:MAKE_TORCH, 39-41:LEVEL_UP_(DEX/STR/INT), 42:ENCHANT_BOW

### COORDINATE & INTERACTION RULES
1. Matrix Coordinates: The map uses (Row, Column) notation. All relevant tiles are shown (for brevity, grass and other background tiles are filtered out). 
   - ROW (1st number): Vertical. Negative is UP, Positive is DOWN.
   - COL (2nd number): Horizontal. Negative is LEFT, Positive is RIGHT.
   - Example: From (0,0), moving UP goes to (-1, 0). Moving DOWN goes to (1, 0).

2. Interaction Targets: The 'DO' action only affects the block you are FACING. You will face the direction you most recently moved in.
   - If facing UP: You act on (-1, 0)
   - If facing DOWN: You act on (1, 0)
   - If facing LEFT: You act on (0, -1)
   - If facing RIGHT: You act on (0, 1)

Note: Do NOT reference coordinates from the examples - those are just to show you how to strategically plan your moves. Only use the CURRENT GAME STATE.
"""

FUTURE_FOCUS_APPENDIX = """
### FUTURE AWARE REASONING
In your <think> reasoning, explicitly model likely near-future outcomes:
- Predict what could happen over the next few steps for multiple plausible action directions.
- Consider likely enemy movement/attacks, intrinsic decay, and immediate resource consequences.
- Highlight robust and risky futures
"""

FUTURE_OPT_SYSTEM_PROMPT = """You are analyzing Craftax future trajectories from one observation.

Predict plausible near-future outcomes with emphasis on:
- Immediate survival risk (damage/death risk, safe vs unsafe lines)
- Intrinsic trends (health/food/drink/energy stability over next few steps)
- Resource/progression implications (tempo gains/losses)

You must not output any action id, action name, or explicit action recommendation.
Use concise forecasting language only.
"""


FEW_SHOT_EXAMPLES_ACTION = """
--- EXAMPLE 1 ---
Game State:
Map (interesting tiles only): 1, 0:crafting_table, 2, -3:tree, 4, 0:stone, 5, 0:stone
Inventory:
Wood: 3
...
Health: 9.0
Food: 9
Drink: 9
Energy: 9
Mana: 9
Direction: right
...
<think> I have full health. I am not hungry. I am not thirsty. I am not tired. I have 3 Wood and I need to craft tools. I see a crafting table already here! I see a 'crafting_table' at (1, 0).
Let me calculate where it is. Crafting table (1, 0) - Me (0,0) = (+1 Row, 0 Col). +1 Row is DOWN. My Direction is 'right'. I need to face 'down' to interact. I must move. Since it is adjacent, I can just interact if I face it. I will move DOWN to face the table.
</think>
**Action:** 4 (DOWN)
   

--- EXAMPLE 2 ---
Game State:
Map (interesting tiles only): 0, -3:tree, 1, 3:stone
Inventory:
Wood: 0
...
Direction: right
...
<think>
I have full health. I am not hungry. I am not thirsty. I have 0 Wood. I need to find a tree to gather wood. Scanning the map...
1. Scan Map: Found 'tree' at (0, -3).
2. Calculate Position:
   - Target: (0, -3)
   - Me: (0, 0)
   - Delta: (0 Row, -3 Col).
   - Analysis: Same row, 3 steps LEFT.
3. Plan: The tree is to my WEST (Left). I am facing EAST (Right). I need to walk over there to chop it.
   - First step: Move LEFT.
</think>
**Action:** 1 (LEFT)

--- EXAMPLE 3 ---
Game State:
Map (interesting tiles only): 0, -2:water, 3, -4:tree, -3, -3:tree, 4, -3:Skeleton on grass
Inventory:
...
Health: 4.0
Food: 5
Drink: 4
...
Direction: down
...
<think>
I am thirsty (Drink: 4). I have low health (4/10). I need to recover health, so I need to eat and drink water. There is water nearby. The skeleton is far away, so I don't need to worry about it.
1. Scan Map: Found 'water' at (0, -2).
2. Calculate Position:
   - Target (0, -2). Me (0, 0).
   - Delta: (0, -2).
   - Analysis: 2 steps LEFT.
3. Plan: This is a simple situation with a clear single objective: reach the water. No threats nearby. I will move LEFT.
</think>
**Action:** 1 (LEFT)

--- EXAMPLE 4 ---
Game State:
Map (interesting tiles only): 0, 2:Arrow , 0, 3:Skeleton, 1, 1:Cow
Inventory:
Wood: 0
...
Health: 3.0
...
Direction: right
...
<think>
I am in critical condition (Health: 3.0). I am hungry. I see a Skeleton to my East shooting at me! There is an Arrow incoming at (0, 2).
1. Analyze Threats:
   - 'Arrow' at (0, 2). It is on my row (Row 0). It will hit me if I stay or move East/West in this row.
   - 'Skeleton' at (0, 3).
2. Analyze Resources:
   - 'Cow' at (1, 1). This is food/healing.
3. Calculate Position:
   - Me: (0, 0).
   - Arrow: (0, 2).
   - Cow: (1, 1).
4. Strategic Decision:
   - I MUST dodge the arrow immediately. I cannot fight (health too low).
   - Moving DOWN (to 1, 0) takes me to Row 1, safely out of the arrow's path (Row 0).
   - Moving DOWN also places me at (1, 0), which is adjacent to the Cow at (1, 1).
   - This achieves two goals: Dodge the threat AND approach healing.
5. Plan: Move DOWN.
</think>
**Action:** 4 (DOWN)

--- EXAMPLE 5 ---
Game State:
Map (interesting tiles only): 1, 0:Orc Soldier on torch on path, 4, 0:Snail on path
Inventory:
Wood: 17
Iron: 4
Stone: 61
Iron Sword with No enchantment
Iron Helmet with No enchantment
...
Health: 6.3
Food: 3
...
Direction: down
...
<think>
I have decent health (6.3). I am hungry. I am not thirsty. I am not tired. There is an Orc Soldier one tile below me. I have Iron gear (Sword, Armor). I am confident I can win.
1. Scan Map: 'Orc Soldier' at (1, 0).
2. Calculate Position:
   - Target (1, 0) - Me (0, 0) = (+1 Row, 0 Col).
   - Analysis: The enemy is 1 step DOWN.
3. Orientation Check: My Direction is 'down'.
   - Result: I am facing the enemy.
4. Strategy: Fight. I have armor and a sword. I need to kill 8 enemies to open the ladder. I probably will not die because I have armor and a sword. I will attack to clear the path and potentially get loot.
Action: DO (Attack).
</think>
**Action:** 5 (DO)

--- END OF EXAMPLES ---
==================================================
>>> LIVE ENVIRONMENT STREAM STARTS HERE <<<
>>> IGNORE ALL COORDINATES FROM EXAMPLES ABOVE <<<
==================================================
"""

FEW_SHOT_EXAMPLES_FUTURE = """
--- EXAMPLE 1 ---
Game State:
Map (interesting tiles only): 1, 0:crafting_table, 2, -3:tree, 4, 0:stone, 5, 0:stone
Inventory:
Wood: 3
...
Health: 9.0
Food: 9
Drink: 9
Energy: 9
Direction: right
...
<think>
Possible near futures:
1. If I path to the crafting table immediately, in a few steps I can convert wood into basic tools and speed up resource gain.
2. If I detour to tree first, I gain extra wood but delay tool crafting and likely slower early progression.
3. No immediate threat pressure (full intrinsics), so both are safe; the tool-first future is higher tempo.
</think>

--- EXAMPLE 2 ---
Game State:
Map (interesting tiles only): 0, -3:tree, 1, 3:stone
Inventory:
Wood: 0
...
Direction: right
...
<think>
Possible near futures:
1. Moving toward the tree likely produces wood in the next few turns, enabling crafting-table and starter tools soon after.
2. Moving away from the tree delays wood economy and can stall early-game progression.
3. There is low immediate danger, so the major future difference is speed of resource unlock.
</think>

--- EXAMPLE 3 ---
Game State:
Map (interesting tiles only): 0, -2:water, 3, -4:tree, -3, -3:tree, 4, -3:Skeleton on grass
Inventory:
...
Health: 4.0
Food: 5
Drink: 4
Direction: down
...
<think>
Possible near futures:
1. If I move toward water now, thirst recovery becomes likely soon, which improves health sustainability.
2. If I spend turns gathering resources first, low health + thirst could compound into a fragile state.
3. Skeleton is currently far enough that short-term hydration-first futures are likely safer.
</think>

--- EXAMPLE 4 ---
Game State:
Map (interesting tiles only): 0, 2:Arrow, 0, 3:Skeleton, 1, 1:Cow
Inventory:
Wood: 0
...
Health: 3.0
Direction: right
...
<think>
Possible near futures:
1. Staying in row 0 risks immediate projectile damage from the incoming arrow.
2. Exiting row 0 quickly likely avoids the arrow path and can transition into a healing path via nearby cow.
3. Trying to force aggression while low health has high downside in the next few steps.
</think>

--- EXAMPLE 5 ---
Game State:
Map (interesting tiles only): 1, 0:Orc Soldier on torch on path, 4, 0:Snail on path
Inventory:
Wood: 17
Iron: 4
Stone: 61
Iron Sword with No enchantment
Iron Helmet with No enchantment
...
Health: 6.3
Food: 3
Direction: down
...
<think>
Possible near futures:
1. Immediate engagement with the nearby Orc could open space and advance ladder-unlock progress if the fight resolves cleanly.
2. If hunger drops further during extended combat, survivability can degrade even with decent gear.
3. A cautious future is to stabilize food before chaining multiple fights; an aggressive future is faster progress with more variance.
</think>

--- END OF EXAMPLES ---
==================================================
>>> LIVE ENVIRONMENT STREAM STARTS HERE <<<
>>> IGNORE ALL COORDINATES FROM EXAMPLES ABOVE <<<
==================================================
"""

FEW_SHOT_EXAMPLES_FUTURE_OPT = """
--- EXAMPLE ---
Game State:
Map (interesting tiles only): 0, 2:Arrow, 0, 3:Skeleton, 1, 1:Cow
Health: 3.0
Food: 5
Drink: 6
Energy: 8
Direction: right
<think>
Key points (<=64 tokens): low health; immediate projectile lane danger on row 0; nearby cow can stabilize.
Future 1: If the trajectory stays in row 0, immediate damage is likely and death risk rises sharply.
Future 2: If the trajectory exits row 0 quickly, near-term danger drops and stabilization becomes more likely.
Future 3: If it forces aggression while fragile, variance increases with a high downside tail.
</think>
--- END EXAMPLE ---
"""


def get_system_prompt(prompt_variant: str = "default") -> str:
    variant = (prompt_variant or "default").strip().lower()
    if variant == "default":
        return SYSTEM_PROMPT
    if variant == "future_based":
        return SYSTEM_PROMPT.rstrip() + "\n\n" + FUTURE_FOCUS_APPENDIX.strip() + "\n"
    if variant == "future_based_opt":
        return FUTURE_OPT_SYSTEM_PROMPT
    raise ValueError(f"Unknown prompt_variant={prompt_variant!r}")


def get_prompt_outline(prompt_variant: str = "default") -> str:
    variant = (prompt_variant or "default").strip().lower()
    if variant == "future_based":
        return (
            "Craftax base system prompt + future-aware appendix: forecast plausible near-future outcomes "
            "(threats, intrinsic decay, resource consequences) without outputting an action."
        )
    if variant == "future_based_opt":
        return (
            "Compact future-forecast prompt: first summarize key state points (~64 tokens), then spend "
            "remaining reasoning budget on three near-future rollout outcomes without action recommendation."
        )
    return (
        "Craftax base system prompt: prioritize survival, resources/tools, and progression with "
        "coordinate-aware tactical reasoning."
    )


def get_generation_prefix(prompt_variant: str = "default") -> str:
    variant = (prompt_variant or "default").strip().lower()
    if variant == "future_based_opt":
        return "<think>\nKey points (<=64 tokens): "
    return ""


def get_generation_stop_sequences(prompt_variant: str = "default") -> list[str]:
    variant = (prompt_variant or "default").strip().lower()
    if variant == "future_based_opt":
        return ["</think>"]
    return []


def get_prompt_sections(
    prompt_variant: str = "default",
    system_prompt: Optional[str] = None,
) -> dict[str, object]:
    """Return canonical prompt sections for a variant."""
    variant = (prompt_variant or "default").strip().lower()
    active_system_prompt = get_system_prompt(variant) if system_prompt is None else system_prompt

    if variant == "future_based":
        few_shot_examples = FEW_SHOT_EXAMPLES_FUTURE
        task_instruction = (
            "You are at (0,0). Output only a <think> block about plausible near-future outcomes. "
            "Do not output an action id, action name, or action recommendation."
        )
    elif variant == "future_based_opt":
        few_shot_examples = FEW_SHOT_EXAMPLES_FUTURE_OPT
        task_instruction = (
            "You are at (0,0). Output exactly:\n"
            "<think>\n"
            "Key points (<=64 tokens): ...\n"
            "Future 1: ...\n"
            "Future 2: ...\n"
            "Future 3: ...\n"
            "</think>\n"
            "Rules: no setup/meta commentary; do not restate the full observation; no action recommendation."
        )
    elif variant == "default":
        few_shot_examples = FEW_SHOT_EXAMPLES_ACTION
        task_instruction = (
            "You are at (0,0). Output your internal reasoning in a <think> block, "
            "then end with: **Action:** <id> (<name>)."
        )
    else:
        raise ValueError(f"Unknown prompt_variant={prompt_variant!r}")

    return {
        "prompt_variant": variant,
        "system_prompt": active_system_prompt,
        "few_shot_examples": few_shot_examples,
        "task_instruction": task_instruction,
        "generation_prefix": get_generation_prefix(variant),
        "stop_sequences": get_generation_stop_sequences(variant),
    }


def build_user_prompt_content(
    text_obs: str,
    few_shot_examples: str,
    task_instruction: str,
) -> str:
    """Build the canonical user message body used by training/eval pipelines."""
    return (
        "Below are examples of good gameplay reasoning. "
        "These are EXAMPLES ONLY, not your actual game history:\n"
        f"{few_shot_examples}\n"
        "YOUR CURRENT GAME STATE (use ONLY this map for coordinates):\n"
        f"{text_obs}\n\n"
        f"{task_instruction}"
    )


MAP_INTERESTING_PREFIX = "Map (interesting tiles only): "
MAP_EMPTY_INTERESTING = "Map: [No interesting tiles in view - all background]"
_MAP_COORD_PREFIX_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:")
_MAP_ENTRY_RE = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*:\s*(.+?)\s*$")


def _split_compact_map_entries(payload: str) -> list[str]:
    """Split compact map payload into per-tile entries without losing row/col commas."""
    content = payload.strip()
    if not content:
        return []
    starts = list(_MAP_COORD_PREFIX_RE.finditer(content))
    if not starts:
        return []
    entries: list[str] = []
    for i, match in enumerate(starts):
        start = match.start()
        end = starts[i + 1].start() if (i + 1) < len(starts) else len(content)
        token = content[start:end].strip().rstrip(",").strip()
        if token:
            entries.append(token)
    return entries


def _parse_map_entry(entry: str) -> Optional[tuple[int, int, str]]:
    """Parse one map token of shape 'row,col:tile'."""
    m = _MAP_ENTRY_RE.fullmatch(entry.strip())
    if m is None:
        return None
    row_s, col_s, tile = m.groups()
    try:
        row = int(row_s)
        col = int(col_s)
    except ValueError:
        return None
    tile_name = tile.strip()
    if not tile_name:
        return None
    return row, col, tile_name


def _collect_map_block(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    """
    Return contiguous map block lines and the next unread index.

    Supports both:
    - compact single-line map: "Map: -4,-5:water, ..."
    - multiline map:
      "Map:"
      "-4, -5: water"
      ...
    """
    first_line = lines[start_idx]
    stripped = first_line.strip()
    block = [first_line]
    next_idx = start_idx + 1

    # Inline compact map payload -> single-line block
    inline_payload = stripped[4:].strip() if stripped.startswith("Map:") else ""
    if inline_payload:
        return block, next_idx

    # Multiline map payload -> consume contiguous map rows
    while next_idx < len(lines):
        candidate = lines[next_idx]
        candidate_stripped = candidate.strip()
        if not candidate_stripped:
            break
        if candidate_stripped.startswith("Inventory:"):
            break
        block.append(candidate)
        next_idx += 1
    return block, next_idx


def _parse_map_block(block_lines: list[str]) -> tuple[bool, list[tuple[int, int, str]]]:
    """Parse map block into (row, col, tile) tuples."""
    if not block_lines:
        return False, []
    first = block_lines[0].strip()
    if not first.startswith("Map:"):
        return False, []

    parsed: list[tuple[int, int, str]] = []

    def _consume_payload(payload: str) -> bool:
        tokens = _split_compact_map_entries(payload)
        if not tokens:
            token = payload.strip().rstrip(",").strip()
            if not token:
                return True
            tokens = [token]
        for token in tokens:
            entry = _parse_map_entry(token)
            if entry is None:
                return False
            parsed.append(entry)
        return True

    inline_payload = first[4:].strip()
    if inline_payload:
        return _consume_payload(inline_payload), parsed

    for line in block_lines[1:]:
        if not _consume_payload(line.strip()):
            return False, []
    return True, parsed


def _format_map_entries(entries: list[tuple[int, int, str]]) -> str:
    return ", ".join(f"{row}, {col}:{tile}" for row, col, tile in entries)


def _validate_interesting_map_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith(MAP_INTERESTING_PREFIX):
        return True
    payload = stripped[len(MAP_INTERESTING_PREFIX):].strip()
    if not payload:
        return False
    tokens = _split_compact_map_entries(payload)
    if not tokens:
        return False
    return all(_parse_map_entry(token) is not None for token in tokens)


def ensure_valid_interesting_map(text_obs: str) -> None:
    """Raise if any emitted 'Map (interesting tiles only)' line has malformed coordinates."""
    for line in text_obs.splitlines():
        if line.strip().startswith(MAP_INTERESTING_PREFIX) and not _validate_interesting_map_line(line):
            raise ValueError(
                "Malformed 'Map (interesting tiles only)' line detected: "
                f"{line!r}. Expected repeated 'row,col:tile' entries."
            )


def filter_text_obs(text_obs: str, strict_map_validation: bool = False) -> str:
    """
    Filter out background tiles from the text observation to reduce token count
    and help the model focus on interesting/interactive tiles.
    
    Handles both map formats:
    - compact: "Map: -5,-4:grass, -4,-4:tree, ..."
    - multiline:
      "Map:"
      "-5, -4: grass"
      ...
    
    Args:
        text_obs: The full text observation from obs_to_text()
        strict_map_validation: if True, raise if malformed interesting-map coordinates
    
    Returns:
        Filtered observation with only interesting tiles shown
    """
    lines = text_obs.split("\n")
    filtered_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("Map:"):
            block_lines, next_idx = _collect_map_block(lines, i)
            parsed_ok, parsed_entries = _parse_map_block(block_lines)
            if parsed_ok:
                interesting_entries: list[tuple[int, int, str]] = []
                for row, col, tile in parsed_entries:
                    tile_type = tile.strip().lower()
                    is_background = tile_type in BACKGROUND_TILES
                    has_entity = " on " in tile_type
                    if (not is_background) or has_entity:
                        interesting_entries.append((row, col, tile.strip()))

                if interesting_entries:
                    formatted = _format_map_entries(interesting_entries)
                    map_line = f"{MAP_INTERESTING_PREFIX}{formatted}"
                    if not _validate_interesting_map_line(map_line):
                        warnings.warn(
                            "Map validation failed after filtering; falling back to original map block.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        for original in block_lines:
                            if original.strip():
                                filtered_lines.append(original)
                    else:
                        filtered_lines.append(map_line)
                else:
                    filtered_lines.append(MAP_EMPTY_INTERESTING)
            else:
                warnings.warn(
                    "Failed to parse map block; keeping original unfiltered map segment.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                for original in block_lines:
                    if original.strip():
                        filtered_lines.append(original)

            i = next_idx
            continue

        if stripped:
            filtered_lines.append(line)
        i += 1

    filtered = "\n".join(filtered_lines)
    if strict_map_validation:
        ensure_valid_interesting_map(filtered)
    return filtered


def create_chat_messages(
    text_obs: str,
    system_prompt: Optional[str] = None,
    prompt_variant: str = "default",
) -> list[dict]:
    """
    Create chat messages for LLM from a text observation.
    
    Args:
        text_obs: The filtered text observation (should already have filter_text_obs applied)
    
    Returns:
        List of chat message dictionaries with 'role' and 'content'
    """
    sections = get_prompt_sections(
        prompt_variant=prompt_variant,
        system_prompt=system_prompt,
    )
    messages = [
        {"role": "system", "content": str(sections["system_prompt"])},
        {
            "role": "user",
            "content": build_user_prompt_content(
                text_obs=text_obs,
                few_shot_examples=str(sections["few_shot_examples"]),
                task_instruction=str(sections["task_instruction"]),
            ),
        },
    ]
    return messages


def create_prompt(
    text_obs: str,
    tokenizer,
    system_prompt: Optional[str] = None,
    prompt_variant: str = "default",
) -> str:
    """
    Create a formatted prompt string for LLM from a text observation.
    
    Args:
        text_obs: The filtered text observation
        tokenizer: HuggingFace tokenizer to apply chat template
    
    Returns:
        Formatted prompt string ready for tokenization
    """
    messages = create_chat_messages(
        text_obs,
        system_prompt=system_prompt,
        prompt_variant=prompt_variant,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
