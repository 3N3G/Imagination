"""Enhanced LLM agent for Craftax with filtered observations and conversation history.

Enhancements over llm_play.py:
1. Filters out background tiles (grass, sand, path, water, darkness) to show only interesting tiles
2. Includes past N observations and model reasonings in context for continuity
3. More compact map representation for better token efficiency
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_text, render_craftax_pixels
from craftax.craftax.constants import Action, Achievement, BLOCK_PIXEL_SIZE_AGENT
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import wandb
import cv2
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Default model - can be overridden with --model flag
# Supported Qwen3 sizes: 0.6B, 1.7B, 4B, 8B, 14B, 32B
# Format: Qwen/Qwen3-{size}B or Qwen/Qwen3-{size}B-Thinking-2507
DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"

# Number of past turns to include in context
HISTORY_LENGTH = 3

# Background tiles to filter out
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


def filter_text_obs(text_obs: str) -> str:
    """
    Filter out background tiles from the text observation to reduce token count
    and help the model focus on interesting/interactive tiles.
    
    Handles the newline-separated format from render_craftax_text:
    Map:
    -5, -4: grass
    -4, -4: tree
    ...
    
    Args:
        text_obs: The full text observation from render_craftax_text()
    
    Returns:
        Filtered observation with only interesting tiles shown
    """
    lines = text_obs.split('\n')
    filtered_lines = []
    in_map_section = False
    interesting_tiles = []
    
    for line in lines:
        stripped = line.strip()
        
        # Detect start of Map section
        if stripped == 'Map:':
            in_map_section = True
            interesting_tiles = []
            continue
        
        # Detect end of Map section (next section header or empty line after tiles)
        if in_map_section:
            # Check if this is a coordinate line (e.g., "-5, -4: grass")
            if ':' in stripped and ',' in stripped.split(':')[0]:
                # Parse coordinate and tile
                parts = stripped.split(':', 1)
                if len(parts) == 2:
                    coord = parts[0].strip()
                    tile = parts[1].strip().lower()
                    
                    # Check if tile is interesting (not pure background)
                    is_background = tile in BACKGROUND_TILES
                    has_entity = ' on ' in tile  # e.g., "Cow on grass"
                    
                    if not is_background or has_entity:
                        interesting_tiles.append(f"{coord}:{parts[1].strip()}")
                continue
            else:
                # End of map section - output filtered tiles
                in_map_section = False
                if interesting_tiles:
                    filtered_lines.append(f"Map (interesting tiles only): {', '.join(interesting_tiles)}")
                else:
                    filtered_lines.append("Map: [No interesting tiles in view - all background]")
                # Now process current line normally
                if stripped:
                    filtered_lines.append(line)
                continue
        
        # Keep all non-map lines
        if stripped:
            filtered_lines.append(line)
    
    # Handle case where map section was the last thing
    if in_map_section:
        if interesting_tiles:
            filtered_lines.append(f"Map (interesting tiles only): {', '.join(interesting_tiles)}")
        else:
            filtered_lines.append("Map: [No interesting tiles in view - all background]")
    
    return '\n'.join(filtered_lines)


def format_history_context(history: list) -> str:
    """
    Format the conversation history for inclusion in the prompt.
    
    Args:
        history: List of dicts with keys: state_summary, reasoning, action, action_name, notable_tiles
    
    Returns:
        Formatted string showing recent game history
    """
    if not history:
        return ""
    
    context = "\n--- RECENT HISTORY (your last moves) ---\n"
    for i, turn in enumerate(history):
        context += f"\nTurn -{len(history) - i}:\n"
        
        # State summary with stats
        state_summary = turn.get('state_summary', '')
        if state_summary:
            context += f"Stats: {state_summary}\n"
        
        # Notable things in view
        notable = turn.get('notable_tiles', '')
        if notable:
            context += f"Nearby: {notable}\n"
        
        # Reasoning - show more of it, truncate only if very long
        reasoning = turn.get('reasoning', '').strip()
        if reasoning:
            # Get the last meaningful part of reasoning if too long
            if len(reasoning) > 600:
                # Try to find a good break point
                context += f"Reasoning (truncated): ...{reasoning[-500:]}\n"
            else:
                context += f"Reasoning: {reasoning}\n"
        
        # Action taken
        context += f"Action: {turn.get('action_name', 'N/A')} ({turn.get('action', '?')})\n"
    
    context += "\n--- END HISTORY ---\n"
    return context


def get_state_summary(text_obs: str) -> str:
    """
    Extract a compact summary of the current state for history.
    
    Args:
        text_obs: Full or filtered text observation
    
    Returns:
        Compact summary string with key stats
    """
    import re
    summary_parts = []
    
    # Extract all numeric stats - be flexible about format
    patterns = [
        (r'Health:?\s*([\d.]+)', 'HP'),
        (r'Food:?\s*(\d+)', 'Food'),
        (r'Drink:?\s*(\d+)', 'Drink'),
        (r'Energy:?\s*(\d+)', 'Energy'),
        (r'Floor:?\s*(\d+)', 'Floor'),
        (r'Wood:?\s*(\d+)', 'Wood'),
        (r'Stone:?\s*(\d+)', 'Stone'),
        (r'Direction:?\s*(\w+)', 'Facing'),
    ]
    
    for pattern, label in patterns:
        match = re.search(pattern, text_obs, re.IGNORECASE)
        if match:
            value = match.group(1)
            # Only include non-zero resources
            if label in ['HP', 'Food', 'Drink', 'Energy', 'Floor', 'Facing']:
                summary_parts.append(f"{label}:{value}")
            elif int(value) > 0:  # Resources - only show if > 0
                summary_parts.append(f"{label}:{value}")
    
    return ", ".join(summary_parts) if summary_parts else "Starting state"


def get_notable_tiles(filtered_text_obs: str) -> str:
    """
    Extract notable tiles from the filtered observation.
    
    Args:
        filtered_text_obs: The filtered text observation (already has background removed)
    
    Returns:
        String listing notable entities and their positions
    """
    # Look for the filtered map line
    if "Map (interesting tiles only):" in filtered_text_obs:
        # Extract just the tiles part
        import re
        match = re.search(r'Map \(interesting tiles only\):\s*(.+?)(?:\n|$)', filtered_text_obs)
        if match:
            tiles_str = match.group(1).strip()
            return tiles_str
    elif "Map:" in filtered_text_obs:
        # Fallback - look for any mobs/special tiles in unfiltered
        import re
        mobs = re.findall(r'(\w+)\s+on\s+\w+', filtered_text_obs, re.IGNORECASE)
        if mobs:
            return "Entities: " + ", ".join(set(mobs[:5]))
    
    return ""


def load_model(model_id: str = None):
    """Load a Qwen model. Supports all Qwen3 sizes (0.6B, 1.7B, 4B, 8B, 14B, 32B)."""
    model_id = model_id or DEFAULT_MODEL
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"Model loaded! Hidden size: {model.config.hidden_size}")
    return model, processor

def get_action(model, processor, text_obs: str, history: list = None) -> tuple[int, dict]:
    """
    Get action from model, handling timeouts and logging full context.
    
    Args:
        model: The loaded model
        processor: The tokenizer/processor
        text_obs: The (optionally filtered) text observation
        history: Optional list of past turn dicts with 'state_summary', 'reasoning', 'action', 'action_name'
    
    Returns:
        action (int): The selected action (0-42).
        log_data (dict): Contains 'full_prompt', 'full_response', 'thinking', 'content', etc.
    """
    history = history or []
    SPLIT_TOKEN_ID = 151668  # </think>
    
    # 1. Prepare Inputs & Full Prompt
    # Few-shot examples showing high-quality gameplay patterns
    # These are formatted WITHOUT chat markers to avoid confusing the model into thinking they're actual history
    few_shot_examples = """
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
    
    # History context removed - each step now only shows current observation
    # history_context = format_history_context(history) if history else ""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Below are examples of good gameplay decisions. These are EXAMPLES ONLY, not your actual game history:\n{few_shot_examples}\nYOUR CURRENT GAME STATE (use ONLY this map for coordinates):\n{text_obs}\n\nYou are at (0,0). Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>)."},
    ]
    
    # Captures the EXACT string passed to the model (System + User)
    full_prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(full_prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # 2. First Generation Pass (Thinking)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.7)
    
    output_ids = outputs[0].tolist()
    patched = False

    # 3. decode full response to check for action
    full_response_text = processor.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
    
    # 4. Try to find action in the initial response
    action = 0
    # Allow optional asterisks to be robust to minor formatting errors
    action_pattern = r'(?:\*\*|)?Action:(?:\*\*|)?\s*(\d+)'
    
    # Find ALL matches and take the LAST one to allow for chain-of-thought self-corrections
    matches = list(re.finditer(action_pattern, full_response_text, re.IGNORECASE))
    
    found_valid_action = False
    if matches:
        last_match = matches[-1]
        try:
            val = int(last_match.group(1))
            if 0 <= val <= 42:
                action = val
                found_valid_action = True
        except ValueError:
            pass

    # 5. Intervention: If no valid action found, FORCE it.
    # This handles both Timeouts (missing split token) and Nonsense (rambling without action) uniformly.
    patched = False
    if not found_valid_action:
        patched = True
        # print("--- Action missing/invalid. Patching... ---")
        
        # We append the trigger string to FORCE the model to output an ID
        # Check if </think> was already generated
        current_context_text = processor.decode(output_ids, skip_special_tokens=True)
        if "</think>" not in current_context_text:
             forced_extension = "\n</think>\nOk I need to output an action now. \n**Action:** "
        else:
             forced_extension = "\nOk I need to output an action now. \n**Action:** "
        
        # Re-tokenize context + extension
        current_context_text = processor.decode(output_ids, skip_special_tokens=True)
        new_full_text = current_context_text + forced_extension
        new_inputs = processor(new_full_text, return_tensors="pt").to(model.device)
        
        # Generate JUST the action ID (short max_new_tokens)
        with torch.no_grad():
            final_outputs = model.generate(
                **new_inputs, 
                max_new_tokens=5, 
                do_sample=True, 
                temperature=0.7
            )
        output_ids = final_outputs[0].tolist()
        
        # Re-decode everything with the extension
        full_response_text = processor.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
        
        # Attempt extraction again on the FORCED text
        matches = list(re.finditer(action_pattern, full_response_text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            try:
                val = int(last_match.group(1))
                if 0 <= val <= 42:
                    action = val
            except ValueError:
                pass

    # 6. Parse Content for Logging
    # Try to clean up "thinking" vs "content" if possible, but for logging simplistic split is fine
    # If explicit split token exists, use it.
    try:
        rev_index = output_ids[::-1].index(SPLIT_TOKEN_ID)
        split_index = len(output_ids) - rev_index
        thinking_content = processor.decode(output_ids[prompt_len:split_index], skip_special_tokens=True).strip()
        content = processor.decode(output_ids[split_index:], skip_special_tokens=True).strip()
    except ValueError:
        # No split token? Then everything is "thinking" or "content" depending on how you view it.
        # Let's just dump it all in full_response and leave thinking empty or full.
        thinking_content = full_response_text
        content = ""

    log_data = {
        "action": action,
        "is_patched": patched,
        "full_prompt": full_prompt_text,
        "full_response": full_response_text,
        "thinking": thinking_content,
        "content": content
    }
    
    return action, log_data

def render_frame_for_video(state, step_num, action_name, reward, total_reward):
    """Render a frame with game pixels and overlay info."""
    pixels = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_AGENT, do_night_noise=False)
    frame = np.array(pixels, dtype=np.uint8)

    # Resize for better visibility
    scale = 4
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Add info overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)

    overlay_height = 60
    frame_with_overlay = np.zeros((frame.shape[0] + overlay_height, frame.shape[1], 3), dtype=np.uint8)
    frame_with_overlay[overlay_height:, :] = frame

    cv2.putText(frame_with_overlay, f"Step: {step_num}", (10, 15), font, font_scale, color, thickness)
    cv2.putText(frame_with_overlay, f"Action: {action_name}", (10, 35), font, font_scale, color, thickness)
    cv2.putText(frame_with_overlay, f"Reward: {reward:.2f} | Total: {total_reward:.2f}", (10, 55), font, font_scale, color, thickness)

    return frame_with_overlay

def get_achieved_achievements(state) -> list[str]:
    achievements_array = np.array(state.achievements)
    achieved = []
    for achievement in Achievement:
        if achievements_array[achievement.value] > 0:
            achieved.append(achievement.name)
    return achieved

def create_action_distribution_chart(action_counts: Counter) -> np.ndarray:
    action_names = [Action(i).name for i in range(43)]
    counts = [action_counts.get(i, 0) for i in range(43)]
    used_actions = [(name, count) for name, count in zip(action_names, counts) if count > 0]
    if not used_actions:
        used_actions = [("NOOP", 0)]
    names, values = zip(*used_actions)

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title('Action Distribution')
    fig.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img

def render_craftax_text_swapped(state):
    # Standard renderer outputs "Col, Row" (x, y)
    # We want "Row, Col" (y, x) to match the prompt and LLM reasoning.
    st = render_craftax_text(state)
    lines = st.split('\n')
    new_lines = []
    
    # Regex to match the coordinate line: "Col, Row: Object"
    # We want "Row, Col: Object"
    # Matches start of line like "-1, 0: " or "0, 0: "
    coord_pattern = re.compile(r"^(-?\d+),\s*(-?\d+):")
    
    for line in lines:
        match = coord_pattern.match(line)
        if match:
            col, row = match.groups()
            # Swap them
            new_line = line.replace(f"{col}, {row}:", f"{row}, {col}:", 1)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
            
    return '\n'.join(new_lines)

# Standard renderer is complex to copy-paste due to dependencies on state structure and exact imports.
# Better strategy: Let render_craftax_text run, then post-process the string to swap "Col, Row" to "Row, Col".
# "y, x: " -> "x, y: "
# Standard output lines look like: "0, -1: tree"
# Regex to match: ^(-?\d+), (-?\d+): 
# Swap groups 1 and 2.
    
def render_craftax_text_swapped(state):
    st = render_craftax_text(state)
    lines = st.split('\n')
    new_lines = []
    
    # Regex to match the coordinate line: "Col, Row: Object"
    # We want "Row, Col: Object"
    coord_pattern = re.compile(r"^(-?\d+),\s*(-?\d+):")
    
    for line in lines:
        match = coord_pattern.match(line)
        if match:
            col, row = match.groups()
            # Swap them
            new_line = line.replace(f"{col}, {row}:", f"{row}, {col}:", 1)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
            
    return '\n'.join(new_lines)

def main():
    parser = argparse.ArgumentParser(description="Enhanced LLM agent for Craftax with filtered observations and history context")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507", help="Model to use")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument("--history_length", type=int, default=HISTORY_LENGTH, help="Number of past turns to include in context")
    parser.add_argument("--wandb_project", type=str, default="craftax-llm-harnessed", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--save_video", action="store_true", help="Save video locally")
    parser.add_argument("--video_dir", type=str, default="./llm_harnessed_videos", help="Directory to save videos")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    args = parser.parse_args()

    # Initialize WandB
    run_name = args.run_name or f"llm-play-harnessed"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_id": args.model,
            "seed": args.seed,
            "max_steps": args.max_steps,
        },
    )
    print(f"[WandB] Initialized run: {run_name}")

    model, processor = load_model(args.model)
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)

    total_reward = 0.0
    step = 0
    action_counts = Counter()
    llm_responses = [] # Store dicts for the table
    frames = []
    
    # History tracking for context
    history = []  # Stores last HISTORY_LENGTH turns

    print("\n" + "="*60)
    print("Starting Craftax with LLM agent (HARNESSED - filtered obs + history)")
    print("="*60)

    # Define columns for WandB table
    columns = [
        "Step", 
        "Full Prompt",
        "Action ID", 
        "Action Name", 
        "Patched", 
        "Thinking", 
        "Final Answer", 
        "Full Response",
        "Filtered State",
        "Unfiltered State"
    ]

    # Initialize Table with log_mode="MUTABLE"
    # columns defined above
    llm_table = wandb.Table(columns=columns, log_mode="MUTABLE")  
    while step < args.max_steps:
        # Get raw text obs (SWAPPED coordinates) and filter it
        raw_text_obs = render_craftax_text_swapped(state)
        filtered_text_obs = filter_text_obs(raw_text_obs)
        
        # Get action with history context
        action, log_data = get_action(model, processor, filtered_text_obs, history)
        action_name = Action(action).name

        action_counts[action] += 1

        llm_table.add_data(
            step, 
            log_data["full_prompt"],
            action, 
            action_name, 
            log_data["is_patched"], 
            log_data["thinking"],
            log_data["content"],
            log_data["full_response"],
            filtered_text_obs,
            raw_text_obs
        )
        
        # Append to llm_responses for final table
        llm_responses.append({
            "step": step,
            "action": action,
            "action_name": action_name,
            "is_patched": log_data["is_patched"],
            "thinking": log_data["thinking"],
            "content": log_data["content"],
            "full_response": log_data["full_response"],
            "full_prompt": log_data["full_prompt"],
            "filtered_state": filtered_text_obs,
            "unfiltered_state": raw_text_obs,
        })

        rng, step_rng = jax.random.split(rng)
        old_achievements = np.array(state.achievements).copy()
        obs, state, reward, done, info = env.step(step_rng, state, action, env_params)
        new_achievements = np.array(state.achievements)

        total_reward += float(reward)
        step += 1
        
        # Update history for next iteration
        history.append({
            "state_summary": get_state_summary(filtered_text_obs),
            "notable_tiles": get_notable_tiles(filtered_text_obs),
            "reasoning": log_data["thinking"],
            "action": action,
            "action_name": action_name,
        })
        # Keep only the last args.history_length turns
        if len(history) > args.history_length:
            history = history[-args.history_length:]

        frame = render_frame_for_video(state, step, action_name, float(reward), total_reward)
        frames.append(frame)

        step_log = {
            "step": step,
            "action": action,
            "reward": float(reward),
            "total_reward": total_reward,
            "player_health": float(state.player_health),
            "player_food": float(state.player_food),
            "player_drink": float(state.player_drink),
            "player_energy": float(state.player_energy),
            "patched_action": int(log_data["is_patched"]), # Scalar metric for quick graphing
        }

        for achievement in Achievement:
            if new_achievements[achievement.value] > old_achievements[achievement.value]:
                step_log[f"achievement/{achievement.name}"] = 1
                print(f"  [Achievement] {achievement.name} earned!")

        if step % 10 == 5:
            # Log intermediate table at step 5, 15, ...
            temp_table = wandb.Table(columns=columns)
            for resp in llm_responses:
                temp_table.add_data(
                    resp["step"], 
                    resp["full_prompt"],
                    resp["action"], 
                    resp["action_name"], 
                    resp["is_patched"], 
                    resp["thinking"],
                    resp["content"],
                    resp["full_response"],
                    resp["filtered_state"],
                    resp["unfiltered_state"]
                )
            wandb.log({f"detailed_logs_step_{step}": temp_table})

        if step % 10 == 0:
            # We don't have the table ready yet for this simple log, so skip or use temp
            pass
            
        wandb.log(step_log)

        if step == 0:
            print(f"Step {step}: {action_name} (action={action}), Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"  Thinking: {log_data['thinking']}...\n\n") 
            print(f"  Full Response: {log_data['full_response']}\n\n")

        if done:
            print(f"\n{'='*60}")
            print(f"Episode ended after {step} steps. Total reward: {total_reward:.2f}")
            print(f"{'='*60}")
            break

    final_achievements = get_achieved_achievements(state)
    print(f"\nAchievements earned ({len(final_achievements)}):")
    for ach in final_achievements:
        print(f"  - {ach}")

    summary_log = {
        "episode/total_reward": total_reward,
        "episode/length": step,
        "episode/num_achievements": len(final_achievements),
        "episode/achievements": final_achievements,
    }

    try:
        action_chart = create_action_distribution_chart(action_counts)
        summary_log["episode/action_distribution_chart"] = wandb.Image(action_chart, caption="Action Distribution")
    except Exception as e:
        print(f"Warning: Failed to create action chart: {e}")

    for action_id, count in action_counts.items():
        summary_log[f"action_counts/{Action(action_id).name}"] = count

    for achievement in Achievement:
        achieved = 1 if achievement.name in final_achievements else 0
        summary_log[f"final_achievements/{achievement.name}"] = achieved

    wandb.log(summary_log)

    # Define columns to include all the new data
    # (columns defined at start of main)
    llm_table = wandb.Table(columns=columns)
    
    for resp in llm_responses:
        llm_table.add_data(
            resp["step"], 
            resp["full_prompt"],
            resp["action"], 
            resp["action_name"], 
            resp["is_patched"], 
            resp["thinking"],
            resp["content"],
            resp["full_response"]
        )
        
    wandb.log({"detailed_logs": llm_table})
    print("Detailed logs table uploaded to WandB.")

    if len(frames) > 0:
        if args.save_video:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"llm_play_harnessed_seed{args.seed}.mp4"

            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"Video saved locally: {video_path}")

        video_array = np.array(frames) 
        video_array = np.transpose(video_array, (0, 3, 1, 2)) 
        wandb.log({
            "episode/video": wandb.Video(video_array, fps=15, format="mp4")
        })
        print("Video logged to WandB")

    print("\nAction Distribution:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
    for action_id, count in sorted_actions[:10]: 
        pct = count / step * 100
        print(f"  {Action(action_id).name}: {count} ({pct:.1f}%)")

    wandb.finish()
    print("\nWandB run finished.")

if __name__ == "__main__":
    main()
