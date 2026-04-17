#!/usr/bin/env python3
"""Sample 10 length-6 trajectories from PSF shards + golden for the prompt-
iteration webapp. Dumps:
  webapp/data/trajectories.json          (metadata + per-step text)
  webapp/data/images/traj_{i}_step_{j}.png  (local tile view)

Each trajectory is 6 consecutive env steps from a single real episode.
Gemini is intended to be called on the FIRST state only; the other five
are ground-truth future for comparing predictions.
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from labelling.obs_to_text import obs_to_text
from llm.prompts import filter_text_obs
from pipeline.gemini_label import decode_obs_from_bitpacked
from pipeline.config import ACTION_NAMES

SOURCES = [
    (
        "psf_shards",
        "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories_psf_gemini_emb",
    ),
    (
        "psf_golden",
        "/data/group_data/rl/geney/oracle_pipeline/predict_only_final_gemini_emb",
    ),
]
N_TRAJECTORIES = 10
TRAJECTORY_LEN = 6
OUT_DIR = REPO_ROOT / "webapp" / "data"
IMG_DIR = OUT_DIR / "images"

# ---------------------------------------------------------------------------
# Tile renderer
# ---------------------------------------------------------------------------
TILE_COLORS = {
    # Terrain
    "out_of_bounds": (0, 0, 0),
    "grass": (100, 180, 80),
    "fire_grass": (210, 120, 60),
    "ice_grass": (200, 230, 255),
    "sand": (230, 210, 160),
    "gravel": (170, 170, 170),
    "path": (200, 180, 150),
    "water": (80, 140, 220),
    "stone": (130, 130, 140),
    "wall": (120, 100, 80),
    "wall_moss": (100, 130, 90),
    "lava": (220, 80, 40),
    "fire": (255, 100, 50),
    "ice": (200, 240, 255),
    "dark": (30, 30, 40),
    # Resources
    "tree": (40, 110, 60),
    "coal": (50, 50, 50),
    "iron": (180, 160, 120),
    "diamond": (150, 220, 255),
    "sapphire": (60, 90, 200),
    "ruby": (200, 50, 80),
    # Structures
    "chest": (200, 160, 80),
    "fountain": (100, 170, 220),
    "crafting_table": (180, 120, 80),
    "furnace": (80, 80, 100),
    "torch": (255, 200, 80),
    "stalagmite": (140, 130, 120),
    # Plants
    "plant": (80, 200, 100),
    "ripe_plant": (180, 220, 100),
    # Ladders
    "down_ladder": (150, 100, 50),
    "up_ladder": (110, 80, 40),
    "ladder": (150, 100, 50),
    # Necromancer arena
    "necromancer": (140, 40, 160),
    "enchantment_table": (130, 100, 200),
}

ENTITY_LABELS = {
    "zombie": "Z", "skeleton": "S", "orc": "O", "lizard": "L",
    "knight": "K", "troll": "T", "necromancer": "N",
    "cow": "c", "snail": "s", "pigman": "p", "bat": "b",
    "fire_walker": "f", "ice_walker": "i",
    "player_arrow": "→", "arrow": "←",
    "fireball": "F", "iceball": "I",
}


def tile_color(tile: str) -> tuple[int, int, int]:
    tile = tile.strip().lower()
    if " on " in tile:
        tile = tile.split(" on ", 1)[0].strip()
    # exact match first
    if tile in TILE_COLORS:
        return TILE_COLORS[tile]
    # substring match
    for key in sorted(TILE_COLORS, key=lambda k: -len(k)):
        if key in tile:
            return TILE_COLORS[key]
    return (80, 80, 80)


def entity_label(tile: str) -> str:
    """Return a short label if the tile carries a mob/entity (' on X')."""
    if " on " not in tile:
        return ""
    entity = tile.split(" on ", 1)[1].strip().lower().replace(" ", "_")
    for key, label in ENTITY_LABELS.items():
        if key in entity:
            return label
    return entity[:2].upper()


def parse_map_line(text_obs: str) -> dict[tuple[int, int], str]:
    lines = text_obs.split("\n")
    map_line = next((l for l in lines if l.strip().startswith("Map:")), "")
    payload = map_line.replace("Map:", "", 1).strip().rstrip(",")
    tiles: dict[tuple[int, int], str] = {}
    for entry in payload.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        coord_part, tile = entry.split(":", 1)
        try:
            r, c = [int(x) for x in coord_part.split(",")]
        except ValueError:
            continue
        tiles[(r, c)] = tile.strip()
    return tiles


def parse_direction(text_obs: str) -> str:
    m = re.search(r"Direction:(\w+)", text_obs)
    return m.group(1).lower() if m else "right"


def render_local_view(text_obs: str, tile_size: int = 32) -> Image.Image:
    """Render the 11×9 local tile view around the player as a PNG."""
    tiles = parse_map_line(text_obs)
    direction = parse_direction(text_obs)

    rows = range(-4, 5)    # 9 rows
    cols = range(-5, 6)    # 11 cols
    W = len(cols) * tile_size
    H = len(rows) * tile_size
    img = Image.new("RGB", (W, H), (20, 20, 30))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/gnu-free/FreeSansBold.ttf", tile_size // 3
        )
    except Exception:
        font = ImageFont.load_default()

    for r_i, r in enumerate(rows):
        for c_i, c in enumerate(cols):
            tile = tiles.get((r, c), "out_of_bounds")
            x0, y0 = c_i * tile_size, r_i * tile_size
            x1, y1 = x0 + tile_size, y0 + tile_size
            draw.rectangle([x0, y0, x1, y1], fill=tile_color(tile),
                           outline=(0, 0, 0))
            lbl = entity_label(tile)
            if lbl:
                draw.text((x0 + 2, y0 + 2), lbl,
                          fill=(255, 255, 255), font=font)

    # Player marker in center
    center_c = 5 * tile_size + tile_size // 2
    center_r = 4 * tile_size + tile_size // 2
    r = tile_size // 3
    draw.ellipse([center_c - r, center_r - r, center_c + r, center_r + r],
                 fill=(255, 60, 60), outline=(255, 255, 255), width=2)
    # Direction arrow
    dx, dy = {"left": (-1, 0), "right": (1, 0),
              "up": (0, -1), "down": (0, 1)}.get(direction, (1, 0))
    ax = center_c + dx * (tile_size // 2)
    ay = center_r + dy * (tile_size // 2)
    draw.line([center_c, center_r, ax, ay],
              fill=(255, 255, 255), width=3)

    return img


# ---------------------------------------------------------------------------
# Episode discovery + sampling
# ---------------------------------------------------------------------------
def find_episodes(done: np.ndarray):
    """Return list of (start, end_exclusive) indices of complete episodes."""
    done_ix = np.where(done)[0]
    episodes = []
    start = 0
    for di in done_ix:
        episodes.append((start, di + 1))
        start = di + 1
    return episodes


def sample_trajectories(seed: int = 7):
    rng = random.Random(seed)

    # For each source: list of (source_label, file_path, np.load-proxy later)
    source_files = {}
    for label, d in SOURCES:
        fs = sorted([f for f in os.listdir(d) if f.endswith(".npz")])
        source_files[label] = [(label, os.path.join(d, f)) for f in fs]

    # Draw: uniform across trajectories; mix sources. Target 6 from shards, 4 from golden.
    target_counts = {"psf_shards": 6, "psf_golden": 4}
    n_total = sum(target_counts.values())
    assert n_total == N_TRAJECTORIES, f"{n_total} != {N_TRAJECTORIES}"

    picks = []
    used_keys = set()

    def pick_one(source_label):
        for _ in range(200):
            label, path = rng.choice(source_files[source_label])
            d = np.load(path, allow_pickle=True)
            obs_all = decode_obs_from_bitpacked(d)
            episodes = find_episodes(np.asarray(d["done"]))
            long_eps = [(s, e) for s, e in episodes if e - s >= TRAJECTORY_LEN + 2]
            if not long_eps:
                continue
            s, e = rng.choice(long_eps)
            # Skip first/last step of episode to avoid edge effects
            valid_starts = range(s + 1, e - TRAJECTORY_LEN - 1)
            if not valid_starts:
                continue
            start = rng.choice(list(valid_starts))
            key = (label, path, start)
            if key in used_keys:
                continue
            used_keys.add(key)
            # Build the 6-step slice
            traj_slice = {
                "source": label,
                "source_file": os.path.basename(path),
                "start_idx": int(start),
                "obs": obs_all[start:start + TRAJECTORY_LEN],  # (6, 8268)
                "action": np.asarray(d["action"][start:start + TRAJECTORY_LEN]),
                "reward": np.asarray(d["reward"][start:start + TRAJECTORY_LEN]),
                "done": np.asarray(d["done"][start:start + TRAJECTORY_LEN]),
            }
            return traj_slice
        raise RuntimeError(f"Could not sample from source {source_label}")

    for source_label, count in target_counts.items():
        for _ in range(count):
            picks.append(pick_one(source_label))

    rng.shuffle(picks)
    return picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any stale images
    for p in IMG_DIR.glob("*.png"):
        p.unlink()

    trajs_raw = sample_trajectories(seed=7)
    out = []
    for i, t in enumerate(trajs_raw):
        steps = []
        for j in range(TRAJECTORY_LEN):
            obs_vec = t["obs"][j].astype(np.float32)
            raw_text = obs_to_text(obs_vec)
            try:
                filtered = filter_text_obs(raw_text)
            except Exception:
                filtered = raw_text

            img = render_local_view(raw_text)
            img_path = IMG_DIR / f"traj_{i}_step_{j}.png"
            img.save(img_path)

            action_id = int(t["action"][j])
            steps.append({
                "step_in_traj": j,
                "obs_text_full": raw_text,
                "obs_text_filtered": filtered,
                "image": f"images/traj_{i}_step_{j}.png",
                "action": action_id,
                "action_name": ACTION_NAMES[action_id] if 0 <= action_id < len(ACTION_NAMES) else str(action_id),
                "reward": float(t["reward"][j]),
                "done": bool(t["done"][j]),
            })
        out.append({
            "traj_id": i,
            "source": t["source"],
            "source_file": t["source_file"],
            "source_start_idx": t["start_idx"],
            "steps": steps,
        })

    with open(OUT_DIR / "trajectories.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} trajectories → {OUT_DIR / 'trajectories.json'}")
    print(f"Wrote {len(out) * TRAJECTORY_LEN} PNGs → {IMG_DIR}")
    # Quick source breakdown
    from collections import Counter
    counts = Counter(t["source"] for t in out)
    print(f"Source breakdown: {dict(counts)}")


if __name__ == "__main__":
    main()
