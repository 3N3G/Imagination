#!/usr/bin/env python3
"""Sample 10 length-6 trajectories from PSF shards + golden for the prompt-
iteration webapp. Dumps:
  webapp/data/trajectories.json            (metadata + per-step text)
  webapp/data/images/traj_{i}_step_{j}.png  (faithful Craftax-textured frame)

Each trajectory is 6 consecutive env steps from a single real episode.
Gemini is intended to be called on the FIRST state only; the other five
are ground-truth future for comparing predictions.

Frames are rendered directly from the obs vector (no env_state needed) via
`_decode_and_render_frame` from Craftax_Baselines/scripts/render_craftax_obs_frames.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from pathlib import Path

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from labelling.obs_to_text import obs_to_text
from llm.prompts import filter_text_obs
from pipeline.gemini_label import decode_obs_from_bitpacked
from pipeline.config import ACTION_NAMES

# Pull the obs-decoder renderer from Craftax_Baselines (not installed as a module).
RENDER_SCRIPT = Path.home() / "Craftax_Baselines" / "scripts" / "render_craftax_obs_frames.py"
spec = importlib.util.spec_from_file_location("_render_craftax_obs_frames", RENDER_SCRIPT)
render_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_mod)

from craftax.craftax.renderer import TEXTURES

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
BLOCK_SIZE = 16
OUT_DIR = REPO_ROOT / "webapp" / "data"
IMG_DIR = OUT_DIR / "images"


def find_episodes(done: np.ndarray):
    done_ix = np.where(done)[0]
    episodes = []
    start = 0
    for di in done_ix:
        episodes.append((start, di + 1))
        start = di + 1
    return episodes


def sample_windows(seed: int = 7):
    rng = random.Random(seed)

    source_files = {}
    for label, d in SOURCES:
        fs = sorted([f for f in os.listdir(d) if f.endswith(".npz")])
        source_files[label] = [(label, os.path.join(d, f)) for f in fs]

    target = {"psf_shards": 5, "psf_golden": 5}
    assert sum(target.values()) == N_TRAJECTORIES

    # Minimum gap between two picks from the same source_file, to avoid
    # near-duplicate windows when a source has only one file (golden has one).
    MIN_FILE_GAP = 2000

    picks = []
    used = set()
    picked_starts_by_path: dict[str, list[int]] = {}

    def pick_one(source_label):
        for _ in range(800):
            label, path = rng.choice(source_files[source_label])
            d = np.load(path, allow_pickle=True)
            obs_all = decode_obs_from_bitpacked(d)
            episodes = find_episodes(np.asarray(d["done"]))
            long_eps = [(s, e) for s, e in episodes if e - s >= TRAJECTORY_LEN + 3]
            if not long_eps:
                continue
            s, e = rng.choice(long_eps)
            # Skip edges
            valid = list(range(s + 2, e - TRAJECTORY_LEN - 1))
            if not valid:
                continue
            start = rng.choice(valid)
            key = (label, path, start)
            if key in used:
                continue
            # Min-gap within same file
            prior = picked_starts_by_path.get(path, [])
            if any(abs(start - p) < MIN_FILE_GAP for p in prior):
                continue
            used.add(key)
            picked_starts_by_path.setdefault(path, []).append(start)
            return {
                "source": label,
                "source_file": os.path.basename(path),
                "start_idx": int(start),
                "obs": obs_all[start:start + TRAJECTORY_LEN],
                "action": np.asarray(d["action"][start:start + TRAJECTORY_LEN]),
                "reward": np.asarray(d["reward"][start:start + TRAJECTORY_LEN]),
                "done": np.asarray(d["done"][start:start + TRAJECTORY_LEN]),
            }
        raise RuntimeError(f"Could not sample from {source_label}")

    for label, count in target.items():
        for _ in range(count):
            picks.append(pick_one(label))

    rng.shuffle(picks)
    return picks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for p in IMG_DIR.glob("*.png"):
        p.unlink()

    # Load textures once
    textures = {k: np.asarray(v) for k, v in TEXTURES[BLOCK_SIZE].items()}

    picks = sample_windows()
    out = []
    for i, t in enumerate(picks):
        steps_out = []
        for j in range(TRAJECTORY_LEN):
            obs_vec = t["obs"][j].astype(np.float32)
            frame = render_mod._decode_and_render_frame(
                obs_vec, bs=BLOCK_SIZE, textures=textures
            )
            img = Image.fromarray(frame)
            img_path = IMG_DIR / f"traj_{i}_step_{j}.png"
            img.save(img_path)

            raw_text = obs_to_text(obs_vec)
            try:
                filtered = filter_text_obs(raw_text)
            except Exception:
                filtered = raw_text

            a = int(t["action"][j])
            steps_out.append({
                "step_in_traj": j,
                "obs_text_full": raw_text,
                "obs_text_filtered": filtered,
                "image": f"images/traj_{i}_step_{j}.png",
                "action": a,
                "action_name": ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a),
                "reward": float(t["reward"][j]),
                "done": bool(t["done"][j]),
            })
        out.append({
            "traj_id": i,
            "source": t["source"],
            "source_file": t["source_file"],
            "source_start_idx": t["start_idx"],
            "steps": steps_out,
        })
        print(f"  traj {i}: {t['source']}/{t['source_file']} @ {t['start_idx']}")

    with open(OUT_DIR / "trajectories.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} trajectories → {OUT_DIR / 'trajectories.json'}")
    print(f"Wrote {len(out) * TRAJECTORY_LEN} PNGs → {IMG_DIR}")
    from collections import Counter
    print("Source breakdown:", dict(Counter(t["source"] for t in out)))


if __name__ == "__main__":
    main()
