import argparse
import sys
import time
import json
import os
import datetime
import hashlib
import subprocess
from pathlib import Path
import pygame
import jax
import jax.numpy as jnp
import numpy as np
import re

# Import Craftax components
from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
)
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv as CraftaxEnv
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels, render_craftax_text
from craftax.craftax.play_craftax import CraftaxRenderer, KEY_MAPPING, print_new_achievements, save_compressed_pickle

# --- COPIED FROM llm_play_harnessed.py TO AVOID TORCH DEPENDENCIES ---
BACKGROUND_TILES = {
    "grass", "sand", "darkness", "wall", "gravel", "path", "water",
    "fire grass", "ice grass", "fire_grass", "ice_grass"
}

def filter_text_obs(text_obs: str) -> str:
    """
    Filter out background tiles from the text observation to reduce token count
    and help the model focus on interesting/interactive tiles.
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
# ---------------------------------------------------------------------

    
def pygame_input_box(screen, prompt):
    """
    A simple Pygame loop to capture text input overlay.
    Returns the entered string, or None if cancelled (Escape).
    """
    font = pygame.font.Font(None, 24)
    coord_font = pygame.font.Font(None, 16)
    input_text = ""
    active = True
    
    # Dimensions for the box - move to bottom to see game
    w, h = screen.get_size()
    box_height = 80
    box_width = w
    box_x = 0
    box_y = h - box_height
    
    rect_box = pygame.Rect(box_x, box_y, box_width, box_height)
    
    # CAPTURE CURRENT SCREEN as background so we don't lose the game frame
    background = screen.copy()
    
    # Render translucent coordinates on tiles for orientation
    # OBS_DIM is (9, 11). Tiles are 64x64 on screen.
    # Player is at (0,0) center, coords go -4 to 4 (rows) and -5 to 5 (cols)
    for r in range(-4, 5):
        for c in range(-5, 6):
            coord_str = f"{r},{c}"
            coord_surf = coord_font.render(coord_str, True, (255, 255, 255))
            coord_surf.set_alpha(140)
            
            # Bottom-right of 64x64 tile
            # Tile screen pos: x=(c+5)*64, y=(r+4)*64
            tx = (c + 5) * 64 + 64 - coord_surf.get_width() - 4
            ty = (r + 4) * 64 + 64 - coord_surf.get_height() - 4
            background.blit(coord_surf, (tx, ty))
    
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return input_text
                elif event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode
        
        # 1. Draw the frozen game state background
        screen.blit(background, (0, 0))
        
        # 2. Draw semi-transparent background for the text box ONLY
        s = pygame.Surface((box_width, box_height))
        s.set_alpha(180)            # alpha level (0=transparent, 255=opaque)
        s.fill((0,0,0))             # black background
        screen.blit(s, (box_x, box_y))

        # 3. Draw border
        pygame.draw.rect(screen, (100, 100, 100), rect_box, 1)
        
        # 4. Draw Prompt
        prompt_surf = font.render(prompt, True, (200, 200, 200))
        screen.blit(prompt_surf, (box_x + 10, box_y + 10))
        
        # 5. Draw Input Text
        # Simple scroll: show last N characters if too long
        display_text = input_text
        max_chars = 90
        if len(display_text) > max_chars: 
             display_text = "..." + display_text[-(max_chars-3):]
             
        txt_surf = font.render(display_text, True, (255, 255, 0))
        screen.blit(txt_surf, (box_x + 10, box_y + 40))
        
        pygame.display.flip()
        
    return None


class GameRecorder:
    def __init__(
        self,
        base_dir="golden_examples",
        env_name="Craftax-Symbolic-v1",
        env_params_digest="unknown",
        repo_commit="unknown",
    ):
        self.session_id = datetime.datetime.now().strftime("game_%Y%m%d_%H%M%S")
        self.save_dir = Path(base_dir) / self.session_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir = self.save_dir / "states"
        self.states_dir.mkdir(exist_ok=True)
        self.images_dir = self.save_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.bundles_dir = self.save_dir / "bundles"
        self.bundles_dir.mkdir(exist_ok=True)
        self.log_file = self.save_dir / "examples.jsonl"
        self.schema_version = "2.0.0"
        self.env_name = env_name
        self.env_params_digest = env_params_digest
        self.repo_commit = repo_commit
        print(f"Recording session to: {self.log_file}")
        print(f"States saved to: {self.states_dir}")
        print(f"Images saved to: {self.images_dir}")
        print(f"Bundles saved to: {self.bundles_dir}")
        session_meta = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at": datetime.datetime.now().isoformat(),
            "env_name": self.env_name,
            "env_params_digest": self.env_params_digest,
            "repo_commit": self.repo_commit,
        }
        (self.save_dir / "session_metadata.json").write_text(
            json.dumps(session_meta, indent=2, sort_keys=True)
        )
        
    def save_step(
        self,
        step_num,
        before_state_obj,
        after_state_obj,
        before_state,
        after_state,
        before_obs_vec,
        after_obs_vec,
        reasoning,
        action,
    ):
        """Save step with text, state objects, symbolic observations, and pixel render."""
        timestamp = datetime.datetime.now().isoformat()
        # Save text entry to JSONL
        entry = {
            "timestamp": timestamp,
            "step": step_num,
            "reasoning": reasoning,
            "action_id": action,
            "action_name": Action(action).name,
            "before_state_raw": before_state["raw"],
            "before_state_filtered": before_state["filtered"],
            "after_state_raw": after_state["raw"],
            "after_state_filtered": after_state["filtered"],
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Save state object (pickle)
        state_path = self.states_dir / f"step_{step_num:04d}.pkl"
        save_compressed_pickle(str(state_path), before_state_obj)
        
        # Save pixel render
        pixels = render_craftax_pixels(before_state_obj, 64, do_night_noise=False)
        img_path = self.images_dir / f"step_{step_num:04d}.png"
        import imageio
        imageio.imwrite(str(img_path), np.array(pixels, dtype=np.uint8))

        # Save rich per-step bundle for downstream policy/value/OOD evaluation.
        bundle_dir = self.bundles_dir / f"step_{step_num:04d}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        save_compressed_pickle(str(bundle_dir / "state_before.pbz2"), before_state_obj)
        save_compressed_pickle(str(bundle_dir / "state_after.pbz2"), after_state_obj)
        np.save(bundle_dir / "obs_before.npy", np.asarray(before_obs_vec, dtype=np.float32))
        np.save(bundle_dir / "obs_after.npy", np.asarray(after_obs_vec, dtype=np.float32))
        (bundle_dir / "before_state_raw.txt").write_text(before_state["raw"])
        (bundle_dir / "before_state_filtered.txt").write_text(before_state["filtered"])
        (bundle_dir / "after_state_raw.txt").write_text(after_state["raw"])
        (bundle_dir / "after_state_filtered.txt").write_text(after_state["filtered"])
        bundle_meta = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "step": int(step_num),
            "timestamp": timestamp,
            "action_id": int(action),
            "action_name": Action(action).name,
            "reasoning": reasoning,
            "env_name": self.env_name,
            "env_params_digest": self.env_params_digest,
            "repo_commit": self.repo_commit,
            "paths": {
                "state_before": "state_before.pbz2",
                "state_after": "state_after.pbz2",
                "obs_before": "obs_before.npy",
                "obs_after": "obs_after.npy",
                "before_state_raw": "before_state_raw.txt",
                "before_state_filtered": "before_state_filtered.txt",
                "after_state_raw": "after_state_raw.txt",
                "after_state_filtered": "after_state_filtered.txt",
            },
        }
        (bundle_dir / "metadata.json").write_text(json.dumps(bundle_meta, indent=2, sort_keys=True))
        
        print(f"Saved example at step {step_num} (state + image)")

def get_obs_dict(state):
    raw_text = render_craftax_text(state)
    filtered = filter_text_obs(raw_text)
    return {"raw": raw_text, "filtered": filtered}

def main(args):
    # Initialize Game and Renderer
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    print("\n" + "="*50)
    print("CRAFTAX THOUGHT RECORDER")
    print("="*50)
    print("Controls:")
    for k, v in KEY_MAPPING.items():
        print(f"{pygame.key.name(k)}: {v.name.lower()}")
    print("-" * 30)
    print("SPECIAL RECORDER KEY:")
    print("/ (slash):  RECORD REASONING for next action")
    print("="*50 + "\n")

    if args.god_mode:
        env_params = env_params.replace(god_mode=True)

    env_params_digest = hashlib.sha1(str(env_params).encode("utf-8")).hexdigest()
    try:
        repo_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "unknown"
        )
    except Exception:
        repo_commit = "unknown"

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng, env_params)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN
    renderer = CraftaxRenderer(env, env_params, pixel_render_size=pixel_render_size)
    renderer.render(env_state)

    step_fn = jax.jit(env.step)
    
    # Initialize Recorder components
    game_recorder = GameRecorder(
        env_name="Craftax-Symbolic-v1",
        env_params_digest=env_params_digest,
        repo_commit=repo_commit,
    )
    
    traj_history = {"state": [env_state], "action": [], "reward": [], "done": []}
    
    clock = pygame.time.Clock()
    
    # State flags for recording
    pending_reasoning = None
    step_count = 0

    while not renderer.is_quit_requested():
        # Check for record key (Slash) in the events captured by renderer
        should_record_this_step = False
        for event in renderer.pygame_events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SLASH:
                print("\n[Record Mode Triggered] Pausing for reasoning...")
                should_record_this_step = True
        
        # If record requested, trigger input box
        if should_record_this_step:
            reasoning = pygame_input_box(renderer.screen_surface, "Reasoning for NEXT action:")
            
            # After input box returns, we need to clear the screen surface so it doesn't stick
            # renderer.render(env_state) call below or at start of loop usually handles this,
            # but we just painted over it. Let's force a render to clear the box visual immediately.
            renderer.render(env_state) 
            pygame.display.flip()

            if reasoning:
                print(f"Reasoning captured: {reasoning}")
                print("Now perform the action described...")
                pending_reasoning = reasoning
            else:
                current_time = time.time()
                print("Recording cancelled.")

        # Normal game loop continues
        action = renderer.get_action_from_keypress(env_state)
        
        if action is not None:
            # If we have a pending reasoning, this action completes the Example
            before_obs = None
            before_state_obj = None
            before_obs_vec = None
            if pending_reasoning:
                before_obs = get_obs_dict(env_state)
                before_state_obj = env_state  # Save actual state object
                before_obs_vec = np.asarray(obs, dtype=np.float32).copy()
            
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )
            step_count += 1
            
            # Handle recording if we just finished an action with pending reasoning
            if pending_reasoning:
                after_obs = get_obs_dict(env_state)
                after_obs_vec = np.asarray(obs, dtype=np.float32).copy()
                game_recorder.save_step(
                    step_count, 
                    before_state_obj,
                    env_state,
                    before_obs, 
                    after_obs,
                    before_obs_vec,
                    after_obs_vec,
                    pending_reasoning,
                    action,
                )
                pending_reasoning = None # Reset
            
            new_achievements = env_state.achievements
            print_new_achievements(old_achievements, new_achievements)

            if reward > 0.8:
                print(f"Reward: {reward}\n")

            traj_history["state"].append(env_state)
            traj_history["action"].append(action)
            traj_history["reward"].append(reward)
            traj_history["done"].append(done)

            renderer.render(env_state)

        renderer.update()
        clock.tick(args.fps)

    if args.save_trajectories:
        save_name = f"play_data/trajectories_{int(time.time())}"
        if args.god_mode:
            save_name += "_GM"
        save_name += ".pkl"
        Path("play_data").mkdir(parents=True, exist_ok=True)
        save_compressed_pickle(save_name, traj_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--god_mode", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    
    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
