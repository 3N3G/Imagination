"""Simple VLM agent that plays Craftax using text observations."""
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

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

SYSTEM_PROMPT = """You are playing Craftax.

Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions and can interact. Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest. This interaction will only happen if the block/creature/chest is directly in front of the player, one step in the direction the player is facing. 
The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. Mana is used for casting spells or enchanting items and will naturally recover. Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. If the players health falls beneath 0 they will die and the game will restart.

IMPORTANT: To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). There are 9 levels in total.

Here is a rough outline of an example progression of this game:
- Gather wood from nearby trees to build a crafting table and then wood tools
- Find and mine stones to make stone tools or place as a furnace or stone
- Use your tools to mine iron ore and coal to use for building iron tools and armor, and make sure to collect extra coal and wood to use as torches later in the game
- The ladder is always open on the overworld, but once you have iron tools and torches you should be ready to descend. Traverse the overworld looking for the ladder and descend.
- Continue killing creatures, then finding and descending down ladders

Make sure to stay healthy during this process, killing cows (or that level's equivalent peaceful animal), drinking water, and resting enough to keep the intrinsics above 0. 

Actions available: 
0:NOOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:DO (interact/mine/attack), 6:SLEEP, 7:PLACE_STONE,
8:PLACE_TABLE, 9:PLACE_FURNACE, 10:PLACE_PLANT, 11:MAKE_WOOD_PICKAXE, 12:MAKE_STONE_PICKAXE,
13:MAKE_IRON_PICKAXE, 14:MAKE_WOOD_SWORD, 15:MAKE_STONE_SWORD, 16:MAKE_IRON_SWORD, 17:REST,
18:DESCEND, 19:ASCEND, 20:MAKE_DIAMOND_PICKAXE, 21:MAKE_DIAMOND_SWORD, 22:MAKE_IRON_ARMOUR,
23:MAKE_DIAMOND_ARMOUR, 24:SHOOT_ARROW, 25:MAKE_ARROW, 26:CAST_FIREBALL, 27:CAST_ICEBALL,
28:PLACE_TORCH, 29-34:DRINK_POTION_(RED/GREEN/BLUE/PINK/CYAN/YELLOW), 35:READ_BOOK,
36:ENCHANT_SWORD, 37:ENCHANT_ARMOUR, 38:MAKE_TORCH, 39-41:LEVEL_UP_(DEX/STR/INT), 42:ENCHANT_BOW

Important: up is (-1, 0), down is (1, 0), left is (0, -1), and right is (0, 1).
"""
# should we include which achievements exist / which have been gotten in the prompt?


def load_model():
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model loaded!")
    return model, processor

def get_action(model, processor, text_obs: str) -> tuple[int, dict]:
    """
    Get action from model, handling timeouts and logging full context.
    
    Returns:
        action (int): The selected action (0-42).
        log_data (dict): Contains 'full_prompt', 'full_response', 'thinking', 'content', etc.
    """
    SPLIT_TOKEN_ID = 151668  # </think>
    
    # 1. Prepare Inputs & Full Prompt
    # Few-shot examples showing high-quality gameplay patterns
    # These are formatted WITHOUT chat markers to avoid confusing the model into thinking they're actual history
    few_shot_examples = """

--- EXAMPLE 1: Moving to gather wood from a tree ---
Game State:
Map: -5,-4:grass, -4,-4:grass, -3,-4:grass, -2,-4:grass, -1,-4:grass, 0,-4:grass, 1,-4:grass, 2,-4:sand, 3,-4:sand, 4,-4:grass, 5,-4:sand, -5,-3:tree, -4,-3:grass, -3,-3:grass, -2,-3:grass, -1,-3:grass, 0,-3:grass, 1,-3:stone, 2,-3:grass, 3,-3:grass, 4,-3:grass, 5,-3:grass, -5,-2:grass, -4,-2:grass, -3,-2:grass, -2,-2:grass, -1,-2:grass, 0,-2:grass, 1,-2:grass, 2,-2:grass, 3,-2:grass, 4,-2:grass, 5,-2:grass, -5,-1:grass, -4,-1:grass, -3,-1:grass, -2,-1:grass, -1,-1:grass, 0,-1:grass, 1,-1:grass, 2,-1:grass, 3,-1:grass, 4,-1:grass, 5,-1:grass, -5,0:grass, -4,0:grass, -3,0:grass, -2,0:grass, -1,0:grass, 0,0:grass, 1,0:grass, 2,0:grass, 3,0:grass, 4,0:grass, 5,0:grass, -5,1:grass, -4,1:grass, -3,1:grass, -2,1:grass, -1,1:grass, 0,1:Cow on grass, 1,1:grass, 2,1:grass, 3,1:grass, 4,1:grass, 5,1:grass, -5,2:grass, -4,2:grass, -3,2:tree, -2,2:grass, -1,2:Cow on grass, 0,2:sand, 1,2:grass, 2,2:grass, 3,2:grass, 4,2:grass, 5,2:grass, -5,3:tree, -4,3:grass, -3,3:grass, -2,3:tree, -1,3:sand, 0,3:sand, 1,3:sand, 2,3:grass, 3,3:grass, 4,3:grass, 5,3:grass, -5,4:grass, -4,4:grass, -3,4:grass, -2,4:grass, -1,4:sand, 0,4:sand, 1,4:sand, 2,4:sand, 3,4:grass, 4,4:grass, 5,4:grass
Inventory: Wood:0, Stone:0, Coal:0, Iron:0, Diamond:0, Sapphire:0, Ruby:0, Sapling:0, Torch:0, Arrow:0, Book:0, Red potion:0, Green potion:0, Blue potion:0, Pink potion:0, Cyan potion:0, Yellow potion:0, Health:9.0, Food:9, Drink:9, Energy:9, Mana:9, XP:0, Dexterity:1, Strength:1, Intelligence:1, Direction:left, Light:0.906, Is Sleeping:False, Is Resting:False, Learned Fireball:False, Learned Iceball:False, Floor:0, Ladder Open:True, Is Boss Vulnerable:False
Good reasoning: "My inventory has 0 Wood, which is the first resource I need to start crafting. I see a tree located at (-3, 2). I am currently at (0, 0). Thus the tree is 3 steps up and 2 steps right. What is one step up? (-1, 0) is grass. Ok, so I should step up to get closer to the tree."
Good action: 3 (UP)

--- EXAMPLE 2: Escaping from a zombie when health is low ---
Game State:
Map: -5,-4:grass, -4,-4:grass, -3,-4:grass, -2,-4:grass, -1,-4:grass, 0,-4:grass, 1,-4:grass, 2,-4:sand, 3,-4:sand, 4,-4:grass, 5,-4:sand, -5,-3:grass, -4,-3:grass, -3,-3:grass, -2,-3:grass, -1,-3:grass, 0,-3:grass, 1,-3:stone, 2,-3:grass, 3,-3:grass, 4,-3:grass, 5,-3:grass, -5,-2:grass, -4,-2:grass, -3,-2:grass, -2,-2:grass, -1,-2:grass, 0,-2:grass, 1,-2:grass, 2,-2:grass, 3,-2:grass, 4,-2:grass, 5,-2:grass, -5,-1:grass, -4,-1:grass, -3,-1:grass, -2,-1:grass, -1,-1:grass, 0,-1:grass, 1,-1:grass, 2,-1:grass, 3,-1:grass, 4,-1:grass, 5,-1:grass, -5,0:grass, -4,0:grass, -3,0:grass, -2,0:grass, -1,0:grass, 0,0:grass, 1,0:grass, 2,0:grass, 3,0:grass, 4,0:grass, 5,0:grass, -5,1:grass, -4,1:grass, -3,1:grass, -2,1:grass, -1,1:grass, 0,1:Zombie on grass, 1,1:grass, 2,1:grass, 3,1:grass, 4,1:grass, 5,1:grass, -5,2:grass, -4,2:grass, -3,2:grass, -2,2:grass, -1,2:grass, 0,2:grass, 1,2:grass, 2,2:grass, 3,2:grass, 4,2:grass, 5,2:grass, -5,3:grass, -4,3:grass, -3,3:grass, -2,3:grass, -1,3:grass, 0,3:grass, 1,3:grass, 2,3:grass, 3,3:grass, 4,3:grass, 5,3:grass, -5,4:grass, -4,4:grass, -3,4:grass, -2,4:grass, -1,4:grass, 0,4:grass, 1,4:grass, 2,4:grass, 3,4:grass, 4,4:grass, 5,4:grass
Inventory: Wood:1, Stone:0, Coal:0, Iron:0, Diamond:0, Sapphire:0, Ruby:0, Sapling:0, Torch:0, Arrow:0, Book:0, Red potion:0, Green potion:0, Blue potion:0, Pink potion:0, Cyan potion:0, Yellow potion:0, Health:3.0, Food:8, Drink:8, Energy:9, Mana:9, XP:0, Dexterity:1, Strength:1, Intelligence:1, Direction:left, Light:0.922, Is Sleeping:False, Is Resting:False, Learned Fireball:False, Learned Iceball:False, Floor:0, Ladder Open:True, Is Boss Vulnerable:False
Good reasoning: "My health is very low (3.0), so I am in danger of dying. There is a Zombie at (0, 1), which is directly to my right. I am at (0, 0). The zombie is at (0, 1) and therefore adjacent and dangerous. If I fight with only 3 health, I might die. I should escape. Looking around: (0, -1) is grass to my left, (-1, 0) is grass above, (1, 0) is grass below. I should move away from the zombie. Going down (action 4) will take me to (1, 0) which is grass and away from danger."
Good action: 4 (DOWN)

--- EXAMPLE 3: Placing a crafting table to begin tool progression ---
Game State:
Map: -5,-4:grass, -4,-4:grass, -3,-4:grass, -2,-4:grass, -1,-4:grass, 0,-4:grass, 1,-4:grass, 2,-4:sand, 3,-4:sand, 4,-4:grass, 5,-4:sand, -5,-3:tree, -4,-3:grass, -3,-3:grass, -2,-3:grass, -1,-3:grass, 0,-3:grass, 1,-3:stone, 2,-3:grass, 3,-3:grass, 4,-3:grass, 5,-3:grass, -5,-2:grass, -4,-2:grass, -3,-2:grass, -2,-2:grass, -1,-2:grass, 0,-2:grass, 1,-2:grass, 2,-2:grass, 3,-2:grass, 4,-2:grass, 5,-2:grass, -5,-1:grass, -4,-1:grass, -3,-1:grass, -2,-1:grass, -1,-1:grass, 0,-1:grass, 1,-1:grass, 2,-1:grass, 3,-1:grass, 4,-1:grass, 5,-1:grass, -5,0:grass, -4,0:grass, -3,0:grass, -2,0:grass, -1,0:grass, 0,0:grass, 1,0:grass, 2,0:grass, 3,0:grass, 4,0:grass, 5,0:grass, -5,1:grass, -4,1:grass, -3,1:grass, -2,1:grass, -1,1:Cow on grass, 0,1:grass, 1,1:grass, 2,1:grass, 3,1:grass, 4,1:grass, 5,1:grass, -5,2:grass, -4,2:grass, -3,2:tree, -2,2:tree, -1,2:grass, 0,2:Cow on sand, 1,2:grass, 2,2:grass, 3,2:grass, 4,2:grass, 5,2:grass, -5,3:tree, -4,3:grass, -3,3:grass, -2,3:tree, -1,3:sand, 0,3:sand, 1,3:sand, 2,3:grass, 3,3:grass, 4,3:grass, 5,3:grass, -5,4:grass, -4,4:grass, -3,4:grass, -2,4:grass, -1,4:sand, 0,4:sand, 1,4:sand, 2,4:sand, 3,4:grass, 4,4:grass, 5,4:grass
Inventory: Wood:2, Stone:0, Coal:0, Iron:0, Diamond:0, Sapphire:0, Ruby:0, Sapling:0, Torch:0, Arrow:0, Book:0, Red potion:0, Green potion:0, Blue potion:0, Pink potion:0, Cyan potion:0, Yellow potion:0, Health:9.0, Food:9, Drink:9, Energy:9, Mana:9, XP:0, Dexterity:1, Strength:1, Intelligence:1, Direction:up, Light:0.927, Is Sleeping:False, Is Resting:False, Learned Fireball:False, Learned Iceball:False, Floor:0, Ladder Open:True, Is Boss Vulnerable:False
Good reasoning: "I have 2 Wood in my inventory. I need 2 wood to place a crafting table. Looking at my current position (0, 0), I can see there's a stone at (1, -3) and cows nearby at (-1, 1) and (0, 2). I have enough wood now to place a crafting table. This is the first step in the progression: place table → craft wood pickaxe → mine stone → make stone tools. I should place a crafting table here. Action 8 is PLACE_TABLE."
Good action: 8 (PLACE_TABLE)

--- END OF EXAMPLES ---
"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Below are examples of good gameplay decisions. These are EXAMPLES ONLY, not your actual game history:\n{few_shot_examples}\n\nNow here is YOUR CURRENT game state:\n{text_obs}\n\nYou are at (0,0); pick an action. Think about the scene briefly, enough to pick a move and then say the number."},
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

    # 3. Intervention (Patching) Logic
    if SPLIT_TOKEN_ID not in output_ids:
        # print("--- Timeout detected. Patching response... ---") 
        patched = True
        
        # Decode what we have so far to append the string safely
        current_context_text = processor.decode(output_ids, skip_special_tokens=True)
        
        forced_extension = (
            "\nTime is up.\n\n"
            "Given the time I've spent and the approaches I've tried, I should stop thinking "
            "and formulate a final answer based on what I already have.\n"
            "</think>\n\n"
            "Action: "
        )
        
        # Re-tokenize the patched context
        new_full_text = current_context_text + forced_extension
        new_inputs = processor(new_full_text, return_tensors="pt").to(model.device)
        
        # Generate the final number
        with torch.no_grad():
            final_outputs = model.generate(
                **new_inputs, 
                max_new_tokens=128, 
                do_sample=True, 
                temperature=0.7
            )
        output_ids = final_outputs[0].tolist()

    # 4. Decode & Partition
    # We use prompt_len to separate the Prompt from the Response.
    full_response_text = processor.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
    
    # Find split point in the FINAL output_ids
    try:
        rev_index = output_ids[::-1].index(SPLIT_TOKEN_ID)
        split_index = len(output_ids) - rev_index
    except ValueError:
        split_index = prompt_len

    thinking_content = processor.decode(output_ids[prompt_len:split_index], skip_special_tokens=True).strip()
    content = processor.decode(output_ids[split_index:], skip_special_tokens=True).strip()

    # 5. Extract Action
    action = 0 
    numbers = re.findall(r'\b(\d+)\b', content)
    for num_str in numbers:
        try:
            parsed_action = int(num_str)
            if 0 <= parsed_action <= 42:
                action = parsed_action
                break
        except ValueError:
            continue

    # 6. Construct Comprehensive Log
    log_data = {
        "action": action,
        "is_patched": patched,
        "full_prompt": full_prompt_text,     # The System + User prompt
        "full_response": full_response_text, # Thinking + Patch (if any) + Content
        "thinking": thinking_content,        # Just the thinking part
        "content": content                   # Just the final answer part
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

def main():
    parser = argparse.ArgumentParser(description="VLM agent playing Craftax")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument("--wandb_project", type=str, default="craftax-vlm-play", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--save_video", action="store_true", help="Save video locally")
    parser.add_argument("--video_dir", type=str, default="./vlm_play_videos", help="Directory to save videos")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    args = parser.parse_args()

    # Initialize WandB
    run_name = args.run_name or f"vlm-play-seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_id": MODEL_ID,
            "seed": args.seed,
            "max_steps": args.max_steps,
        },
    )
    print(f"[WandB] Initialized run: {run_name}")

    model, processor = load_model()
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

    print("\n" + "="*60)
    print("Starting Craftax with VLM agent")
    print("="*60)

    # Initialize Table with log_mode="MUTABLE"
    columns = ["Step", "Prompt", "Action ID", "Action Name", "Patched", "Thinking", "Final Answer", "Full Response"]
    llm_table = wandb.Table(columns=columns, log_mode="MUTABLE")  
    while step < args.max_steps:
        text_obs = render_craftax_text(state)
        
        action, log_data = get_action(model, processor, text_obs)
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
            log_data["full_response"]
        )

        rng, step_rng = jax.random.split(rng)
        old_achievements = np.array(state.achievements).copy()
        obs, state, reward, done, info = env.step(step_rng, state, action, env_params)
        new_achievements = np.array(state.achievements)

        total_reward += float(reward)
        step += 1

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

        if step % 10 == 0:
            step_log["detailed_logs"] = llm_table
            
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
    columns = [
        "Step", 
        "Action ID", 
        "Action Name", 
        "Patched", 
        "Thinking", 
        "Final Answer", 
        "Full Response",
        "Full Prompt"
    ]
    llm_table = wandb.Table(columns=columns)
    
    for resp in llm_responses:
        llm_table.add_data(
            resp["step"], 
            resp["action"], 
            resp["action_name"], 
            resp["is_patched"], 
            resp["thinking"],
            resp["content"],
            resp["full_response"],
            resp["full_prompt"]
        )
        
    wandb.log({"detailed_logs": llm_table})
    print("Detailed logs table uploaded to WandB.")

    if len(frames) > 0:
        if args.save_video:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"vlm_play_seed{args.seed}.mp4"

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
