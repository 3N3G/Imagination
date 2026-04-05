#!/usr/bin/env python3
"""Render a text observation (from play_craftax_recorder.py logs) as an image.

Usage:
    python render_text_obs.py examples.jsonl          # Render all entries
    python render_text_obs.py examples.jsonl --step 5 # Render specific step
    python render_text_obs.py examples.jsonl -o output_dir/
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Tile colors (RGB)
TILE_COLORS = {
    # Background
    "grass": (34, 139, 34),
    "sand": (238, 214, 175),
    "gravel": (128, 128, 128),
    "path": (139, 119, 101),
    "water": (65, 105, 225),
    "darkness": (20, 20, 20),
    "wall": (69, 69, 69),
    "fire grass": (178, 34, 34),
    "fire_grass": (178, 34, 34),
    "ice grass": (173, 216, 230),
    "ice_grass": (173, 216, 230),
    
    # Resources
    "tree": (0, 100, 0),
    "stone": (105, 105, 105),
    "coal": (47, 47, 47),
    "iron": (192, 192, 192),
    "diamond": (0, 191, 255),
    "sapphire": (15, 82, 186),
    "ruby": (224, 17, 95),
    
    # Structures
    "crafting_table": (139, 90, 43),
    "furnace": (178, 102, 0),
    "torch": (255, 165, 0),
    "ladder": (184, 134, 11),
    "ladder_open": (50, 205, 50),
    "chest": (218, 165, 32),
    "fountain": (0, 206, 209),
    
    # Mobs (friendly)
    "cow": (139, 69, 19),
    "plant": (0, 128, 0),
    "sapling": (144, 238, 144),
    
    # Mobs (hostile)
    "zombie": (85, 107, 47),
    "skeleton": (245, 245, 220),
    "arrow": (255, 255, 0),
    "gnome": (255, 20, 147),
    "orc": (128, 128, 0),
    "orc soldier": (154, 154, 0),
    "lizard": (0, 128, 128),
    "knight": (192, 192, 192),
    "bat": (75, 0, 130),
    "snail": (255, 182, 193),
    "spider": (139, 0, 0),
    "goblin": (107, 142, 35),
    
    # Player
    "player": (255, 0, 0),
}

# Default color for unknown tiles
DEFAULT_COLOR = (128, 0, 128)  # Purple


def parse_text_obs(text_obs: str) -> dict:
    """Parse a raw text observation into structured data.
    
    Returns:
        dict with 'tiles', 'inventory', 'stats', 'direction'
    """
    result = {
        "tiles": {},  # (row, col) -> tile_name
        "inventory": {},
        "stats": {},
        "direction": "down",
    }
    
    lines = text_obs.strip().split('\n')
    section = None
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Detect sections
        if stripped == "Map:":
            section = "map"
            continue
        elif stripped == "Inventory:":
            section = "inventory"
            continue
        
        # Parse map coordinates
        if section == "map" and ':' in stripped:
            # Format: "row, col: tile_name" or "-1, 2: Cow on grass"
            match = re.match(r'^(-?\d+),\s*(-?\d+):\s*(.+)$', stripped)
            if match:
                row, col, tile = int(match.group(1)), int(match.group(2)), match.group(3).strip()
                result["tiles"][(row, col)] = tile
                continue
        
        # Parse stats and other fields
        if section == "inventory" or section is None:
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == "direction":
                    result["direction"] = value.lower()
                elif key in ["health", "food", "drink", "energy", "mana"]:
                    try:
                        result["stats"][key] = float(value)
                    except ValueError:
                        pass
                elif key in ["wood", "stone", "coal", "iron", "diamond"]:
                    try:
                        result["inventory"][key] = int(value)
                    except ValueError:
                        pass
    
    return result


def get_tile_color(tile_name: str) -> tuple:
    """Get color for a tile, handling 'X on Y' format."""
    tile_lower = tile_name.lower()
    
    # Handle "Entity on Base" format (e.g., "Cow on grass")
    if " on " in tile_lower:
        entity = tile_lower.split(" on ")[0].strip()
        if entity in TILE_COLORS:
            return TILE_COLORS[entity]
    
    # Direct lookup
    if tile_lower in TILE_COLORS:
        return TILE_COLORS[tile_lower]
    
    # Partial match
    for key, color in TILE_COLORS.items():
        if key in tile_lower or tile_lower in key:
            return color
    
    return DEFAULT_COLOR


def render_obs_image(parsed_obs: dict, tile_size: int = 32) -> Image.Image:
    """Render parsed observation as an image.
    
    Args:
        parsed_obs: Output from parse_text_obs()
        tile_size: Size of each tile in pixels
    
    Returns:
        PIL Image
    """
    # Grid is 9 rows x 11 cols, centered at (0,0)
    # Row range: -4 to 4, Col range: -5 to 5
    rows = 9
    cols = 11
    
    width = cols * tile_size
    height = rows * tile_size + 80  # Extra space for stats
    
    img = Image.new('RGB', (width, height), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 8)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Draw grid
    for row in range(-4, 5):
        for col in range(-5, 6):
            # Screen position
            sx = (col + 5) * tile_size
            sy = (row + 4) * tile_size
            
            # Get tile
            tile = parsed_obs["tiles"].get((row, col), "grass")
            color = get_tile_color(tile)
            
            # Draw tile
            draw.rectangle([sx, sy, sx + tile_size - 1, sy + tile_size - 1], fill=color)
            
            # Draw grid lines
            draw.rectangle([sx, sy, sx + tile_size - 1, sy + tile_size - 1], outline=(80, 80, 80))
            
            # Draw coordinate label
            coord_text = f"{row},{col}"
            draw.text((sx + 2, sy + tile_size - 12), coord_text, fill=(200, 200, 200), font=small_font)
    
    # Highlight player position (0, 0)
    px = 5 * tile_size
    py = 4 * tile_size
    draw.rectangle([px + 2, py + 2, px + tile_size - 3, py + tile_size - 3], outline=(255, 0, 0), width=2)
    draw.text((px + tile_size//3, py + 2), "P", fill=(255, 0, 0), font=font)
    
    # Draw direction indicator
    direction = parsed_obs["direction"]
    cx, cy = px + tile_size // 2, py + tile_size // 2
    arrow_len = tile_size // 3
    if direction == "up":
        draw.line([(cx, cy), (cx, cy - arrow_len)], fill=(255, 255, 0), width=2)
    elif direction == "down":
        draw.line([(cx, cy), (cx, cy + arrow_len)], fill=(255, 255, 0), width=2)
    elif direction == "left":
        draw.line([(cx, cy), (cx - arrow_len, cy)], fill=(255, 255, 0), width=2)
    elif direction == "right":
        draw.line([(cx, cy), (cx + arrow_len, cy)], fill=(255, 255, 0), width=2)
    
    # Draw stats at bottom
    stats_y = rows * tile_size + 5
    stats = parsed_obs.get("stats", {})
    stats_text = f"HP: {stats.get('health', '?')}  Food: {stats.get('food', '?')}  Drink: {stats.get('drink', '?')}  Energy: {stats.get('energy', '?')}"
    draw.text((5, stats_y), stats_text, fill=(255, 255, 255), font=font)
    
    inv = parsed_obs.get("inventory", {})
    inv_text = f"Wood: {inv.get('wood', 0)}  Stone: {inv.get('stone', 0)}  Coal: {inv.get('coal', 0)}  Iron: {inv.get('iron', 0)}"
    draw.text((5, stats_y + 15), inv_text, fill=(200, 200, 200), font=font)
    
    dir_text = f"Facing: {direction}"
    draw.text((5, stats_y + 30), dir_text, fill=(255, 255, 0), font=font)
    
    return img


def process_jsonl(jsonl_path: str, output_dir: str = None, step: int = None):
    """Process a JSONL file from play_craftax_recorder.py and render images."""
    jsonl_path = Path(jsonl_path)
    
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = jsonl_path.parent / "rendered"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            entry_step = entry.get("step", 0)
            
            if step is not None and entry_step != step:
                continue
            
            # Render before state
            before_raw = entry.get("before_state_raw", "")
            if before_raw:
                parsed = parse_text_obs(before_raw)
                img = render_obs_image(parsed)
                
                out_path = out_dir / f"step_{entry_step:04d}_before.png"
                img.save(out_path)
                print(f"Saved: {out_path}")
                
                # Also print reasoning
                reasoning = entry.get("reasoning", "")
                action = entry.get("action_name", "")
                print(f"  Action: {action}")
                print(f"  Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"  Reasoning: {reasoning}")


def render_single_text(text: str, output_path: str = None):
    """Render a single text observation."""
    parsed = parse_text_obs(text)
    img = render_obs_image(parsed)
    
    if output_path:
        img.save(output_path)
        print(f"Saved: {output_path}")
    else:
        img.show()
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Render text observations as images")
    parser.add_argument("input", help="JSONL file or text file to render")
    parser.add_argument("-o", "--output", help="Output directory for rendered images")
    parser.add_argument("--step", type=int, help="Only render specific step number")
    parser.add_argument("--text", action="store_true", help="Input is raw text, not JSONL")
    args = parser.parse_args()
    
    if args.text:
        with open(args.input, 'r') as f:
            text = f.read()
        render_single_text(text, args.output)
    else:
        process_jsonl(args.input, args.output, args.step)


if __name__ == "__main__":
    main()
