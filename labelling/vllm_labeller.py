#!/usr/bin/env python3
"""
vLLM-Based LLM Worker for Craftax Data Labelling

This script is a drop-in replacement for llm_worker.py but uses vLLM
for 10-30x faster inference throughput.

Key differences from llm_worker.py:
- Uses vLLM's LLM class with continuous batching and PagedAttention
- Processes entire NPZ files in large batches (not limited to 16)
- Optionally supports speculative decoding and quantization

Usage:
    python vllm_labeller.py --input /path/to/input.npz --output /path/to/output.npz
    
For Redis queue mode (like original llm_worker.py):
    python vllm_labeller.py --queue-mode
"""

import argparse
import os
import sys
import time
import logging
import socket
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

# Prevent JAX issues
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --- Constants (matching llm_worker.py) ---
QUEUE_NAME = "craftax_llm_job_queue"
RESULTS_DIR = "/data/group_data/rl/geney/craftax_llm_labelled_results/"
LOGS_DIR = "/data/group_data/rl/geney/craftax_llm_job_logs/"
TEMP_NPY_DIR = os.path.join(RESULTS_DIR, "temp_npy")
PROGRESS_DIR = os.path.join(RESULTS_DIR, "progress")

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
TOKENS_GENERATED = 256

# For mmap storage
MAX_TEXT_LEN = 2048
TEXT_DTYPE = f'<U{MAX_TEXT_LEN}'

# Background tiles to filter (matching llm_worker.py)
BACKGROUND_TILES = {
    "grass", "sand", "gravel", 
    "fire grass", "ice grass", "fire_grass", "ice_grass"
}

# --- Logging Setup ---
pid = os.getpid()
hostname = socket.gethostname()


def setup_logging(mode: str = "file"):
    """Setup logging to file and/or stdout."""
    logger = logging.getLogger(f"vllm_worker_{pid}")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Always log to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # Optionally log to file
    if mode == "file":
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_filename = os.path.join(LOGS_DIR, f"vllm_worker_{hostname}_{pid}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# --- System Prompt (matching llm_worker.py) ---
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

FEW_SHOT_EXAMPLES = """
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

--- END OF EXAMPLES ---
==================================================
>>> LIVE ENVIRONMENT STREAM STARTS HERE <<<
>>> IGNORE ALL COORDINATES FROM EXAMPLES ABOVE <<<
==================================================
"""


def filter_text_obs(text_obs: str) -> str:
    """Filter out background tiles from the text observation."""
    lines = text_obs.split('\n')
    filtered_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('Map:'):
            map_content = stripped[4:].strip()
            tiles = [t.strip() for t in map_content.split(',') if ':' in t]
            
            interesting_tiles = []
            for tile in tiles:
                parts = tile.rsplit(':', 1)
                if len(parts) == 2:
                    coord = parts[0].strip()
                    tile_type = parts[1].strip().lower()
                    
                    is_background = tile_type in BACKGROUND_TILES
                    has_entity = ' on ' in tile_type
                    
                    if not is_background or has_entity:
                        interesting_tiles.append(f"{coord}:{parts[1].strip()}")
            
            if interesting_tiles:
                filtered_lines.append(f"Map (interesting tiles only): {', '.join(interesting_tiles)}")
            else:
                filtered_lines.append("Map: [No interesting tiles in view - all background]")
            continue
        
        if stripped:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def create_prompt(text_obs: str) -> str:
    """Create prompt from text observation."""
    return f"Below are examples of good gameplay decisions. These are EXAMPLES ONLY, not your actual game history:\n{FEW_SHOT_EXAMPLES}\nYOUR CURRENT GAME STATE (use ONLY this map for coordinates):\n{text_obs}\n\nYou are at (0,0). Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>)."


def process_npz_file(
    input_path: str,
    output_path: str,
    model_id: str = MODEL_ID,
    max_tokens: int = TOKENS_GENERATED,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Process a single NPZ file using vLLM for fast batch inference.
    
    Args:
        input_path: Path to input NPZ file with 'obs' key
        output_path: Path to save output NPZ file
        model_id: HuggingFace model ID
        max_tokens: Maximum tokens to generate per sample
        tensor_parallel_size: Number of GPUs for tensor parallelism
        quantization: Quantization method ('awq', 'gptq', 'fp8', or None)
        logger: Logger instance
    
    Returns:
        Dict with processing statistics
    """
    if logger is None:
        logger = setup_logging("stdout")
    
    # Import vLLM
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not installed. Run: pip install vllm")
        raise ImportError("vLLM not installed")
    
    # Import obs_to_text for decoding symbolic observations
    from obs_to_text import obs_to_text
    
    logger.info(f"Processing {input_path}")
    
    # Load input data
    data = np.load(input_path, allow_pickle=True)
    num_samples = len(data["obs"])
    logger.info(f"Loaded {num_samples} samples")
    
    # Initialize vLLM
    logger.info(f"Initializing vLLM with model {model_id}")
    
    llm_kwargs = {
        "model": model_id,
        "trust_remote_code": True,
        "dtype": "float16",
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
    }
    
    if quantization:
        llm_kwargs["quantization"] = quantization
        logger.info(f"Using quantization: {quantization}")
    
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    
    # Prepare all prompts
    logger.info("Preparing prompts...")
    prompts = []
    for i in range(num_samples):
        # Decode symbolic observation to text
        if "text_obs" in data and data["text_obs"][i]:
            raw_text_obs = str(data["text_obs"][i])
        else:
            raw_text_obs = obs_to_text(data["obs"][i])
        
        # Filter background tiles
        filtered_text_obs = filter_text_obs(raw_text_obs)
        
        # Create chat-formatted prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": create_prompt(filtered_text_obs)},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
        
        if (i + 1) % 10000 == 0:
            logger.info(f"  Prepared {i + 1}/{num_samples} prompts")
    
    logger.info(f"Prepared {len(prompts)} prompts")
    
    # Configure sampling
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
    )
    
    # Run inference
    logger.info("Starting vLLM batch inference...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    inference_time = end_time - start_time
    logger.info(f"Inference complete in {inference_time:.2f}s ({num_samples / inference_time:.2f} samples/sec)")
    
    # Extract text outputs
    text_outputs = []
    total_tokens = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        text_outputs.append(generated_text)
        total_tokens += len(output.outputs[0].token_ids)
    
    logger.info(f"Total tokens generated: {total_tokens}")
    logger.info(f"Tokens/sec: {total_tokens / inference_time:.2f}")
    
    # Convert to numpy array with fixed string length
    text_outputs_numpy = np.array(text_outputs, dtype=TEXT_DTYPE)
    
    # Save output (matching llm_worker.py format)
    # Note: We don't save hidden states in vLLM mode as it's primarily for throughput
    # Hidden states can be added if needed but require different vLLM configuration
    save_data = {
        "obs": data["obs"],
        "next_obs": data["next_obs"],
        "action": data["action"],
        "reward": data["reward"],
        "done": data["done"],
        "log_prob": data["log_prob"],
        "text_generated": text_outputs_numpy.astype(object),
    }
    
    logger.info(f"Saving to {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    np.savez_compressed(output_path, **save_data)
    
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "num_samples": num_samples,
        "total_tokens": total_tokens,
        "inference_time_s": inference_time,
        "samples_per_sec": num_samples / inference_time,
        "tokens_per_sec": total_tokens / inference_time,
    }
    
    logger.info(f"Complete! Stats: {json.dumps(stats, indent=2)}")
    return stats


def queue_mode(args, logger):
    """Run in queue mode, processing jobs from Redis (like llm_worker.py)."""
    import redis
    
    # Read Redis host
    REDIS_HOST_FILE = "/data/group_data/rl/geney/redis_host.txt"
    try:
        with open(REDIS_HOST_FILE, 'r') as f:
            REDIS_HOST = f.read().strip()
        logger.info(f"Read Redis host from file: {REDIS_HOST}")
    except FileNotFoundError:
        REDIS_HOST = "login1"
        logger.warning(f"Redis host file not found, using fallback: {REDIS_HOST}")
    
    r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    r.ping()
    logger.info(f"Connected to Redis at {REDIS_HOST}:6379")
    
    while True:
        file_path = r.rpop(QUEUE_NAME)
        if file_path is None:
            logger.info("No more jobs! Exiting.")
            break
        
        logger.info(f"Processing job: {file_path}")
        job_basename = os.path.basename(file_path)
        output_path = os.path.join(RESULTS_DIR, job_basename)
        
        try:
            stats = process_npz_file(
                input_path=file_path,
                output_path=output_path,
                model_id=args.model,
                max_tokens=args.max_tokens,
                tensor_parallel_size=args.tensor_parallel,
                quantization=args.quantization,
                logger=logger,
            )
            logger.info(f"Completed job: {file_path}")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
    
    logger.info("Worker finished.")


def main():
    parser = argparse.ArgumentParser(description="vLLM-based LLM labeller for Craftax")
    parser.add_argument("--input", type=str, help="Input NPZ file path")
    parser.add_argument("--output", type=str, help="Output NPZ file path")
    parser.add_argument("--queue-mode", action="store_true", help="Run in Redis queue mode")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID")
    parser.add_argument("--max-tokens", type=int, default=TOKENS_GENERATED, help="Max tokens per sample")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--quantization", type=str, choices=["awq", "gptq", "fp8", None], default=None,
                        help="Quantization method")
    args = parser.parse_args()
    
    logger = setup_logging("file" if args.queue_mode else "stdout")
    
    if args.queue_mode:
        queue_mode(args, logger)
    elif args.input and args.output:
        process_npz_file(
            input_path=args.input,
            output_path=args.output,
            model_id=args.model,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel,
            quantization=args.quantization,
            logger=logger,
        )
    else:
        parser.print_help()
        print("\nError: Must specify --input and --output, or use --queue-mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
