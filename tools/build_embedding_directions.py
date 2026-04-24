"""Compute mean-difference direction vectors in embedding space from saved eval logs.

For each pair (regular_eval, die_v2_eval) of an existing track:
  - Read all gemini_text strings from regular gemini_log.jsonl files
  - Read all gemini_text strings from die_v2 gemini_log.jsonl files
  - Run extract_prediction_suffix to get the same predonly text the eval embedded
  - Embed each text via gemini-embedding-001 (3072-dim)
  - Save mean(die_v2) - mean(regular) as a "death direction" vector

Output: probe_results/embed_directions/<track>_<direction_name>.npy

Usage:
  PYTHONPATH=. GEMINI_API_KEY=... python tools/build_embedding_directions.py \
      --max-pairs 200 --out-dir probe_results/embed_directions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pipeline.embed import _embed_one_gemini, extract_prediction_suffix

EMBED_DIM = 3072
EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-embedding-001:embedContent"
)

# Source eval directories. Direction = mean(target) - mean(reference).
SOURCES = [
    # name → (reference_dir, target_dir)
    ("c_grounded_die_v2", {
        "reference_dir": "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M/freezenone_50ep",
        "target_dir":    "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_v2_probe/die_v2_50ep",
    }),
    ("c_grounded_avoid_animals_v2", {
        "reference_dir": "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M/freezenone_50ep",
        "target_dir":    "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_v2_probe/avoid_animals_v2_50ep",
    }),
    ("a_full_die_v2", {
        "reference_dir": "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly/freezenone_50ep",
        "target_dir":    "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_v2_probe/die_v2_50ep",
    }),
]


def collect_texts(eval_dir: Path, max_n: int) -> List[str]:
    texts: List[str] = []
    ep_dirs = sorted([d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    for ep_dir in ep_dirs:
        log_file = ep_dir / "gemini_log.jsonl"
        if not log_file.exists():
            continue
        with log_file.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    text = rec.get("gemini_text", "")
                    if not text:
                        continue
                    pred, _ = extract_prediction_suffix(text)
                    if pred:
                        texts.append(pred)
                        if len(texts) >= max_n:
                            return texts
                except Exception:
                    continue
    return texts


def embed_batch(texts: List[str], api_key: str) -> np.ndarray:
    url = f"{EMBED_URL}?key={api_key}"
    embeds = []
    for i, t in enumerate(texts):
        try:
            v = _embed_one_gemini(t, url, EMBED_DIM).astype(np.float32)
            embeds.append(v)
        except Exception as e:
            print(f"  embed[{i}] failed: {e}", flush=True)
            continue
        if (i + 1) % 25 == 0:
            print(f"    embedded {i+1}/{len(texts)}", flush=True)
        time.sleep(0.05)
    return np.stack(embeds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=200,
                    help="Max texts per condition (per source pair).")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr); sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, paths in SOURCES:
        ref_dir = Path(paths["reference_dir"])
        tgt_dir = Path(paths["target_dir"])
        if not ref_dir.exists() or not tgt_dir.exists():
            print(f"SKIP {name}: missing {ref_dir if not ref_dir.exists() else tgt_dir}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        ref_texts = collect_texts(ref_dir, args.max_pairs)
        tgt_texts = collect_texts(tgt_dir, args.max_pairs)
        n = min(len(ref_texts), len(tgt_texts))
        print(f"  reference: {len(ref_texts)} texts, target: {len(tgt_texts)} texts -> using {n} each", flush=True)
        ref_texts = ref_texts[:n]; tgt_texts = tgt_texts[:n]

        print(f"  embedding reference ({n})...", flush=True)
        ref_embeds = embed_batch(ref_texts, api_key)
        print(f"  embedding target ({n})...", flush=True)
        tgt_embeds = embed_batch(tgt_texts, api_key)

        ref_mean = ref_embeds.mean(axis=0)
        tgt_mean = tgt_embeds.mean(axis=0)
        direction = tgt_mean - ref_mean
        dir_norm = float(np.linalg.norm(direction))
        ref_mean_norm = float(np.linalg.norm(ref_mean))
        tgt_mean_norm = float(np.linalg.norm(tgt_mean))
        cos_ref_tgt = float(np.dot(ref_mean, tgt_mean) / (ref_mean_norm * tgt_mean_norm))

        out_path = args.out_dir / f"{name}.npy"
        np.save(out_path, direction)
        # save means too
        np.save(args.out_dir / f"{name}_ref_mean.npy", ref_mean)
        np.save(args.out_dir / f"{name}_tgt_mean.npy", tgt_mean)
        # save raw embeddings for reproducibility
        np.savez(args.out_dir / f"{name}_raw.npz",
                 ref_embeds=ref_embeds, tgt_embeds=tgt_embeds)

        print(f"  direction norm: {dir_norm:.4f}", flush=True)
        print(f"  ref_mean norm:  {ref_mean_norm:.4f}", flush=True)
        print(f"  tgt_mean norm:  {tgt_mean_norm:.4f}", flush=True)
        print(f"  cos(ref, tgt):  {cos_ref_tgt:.4f}", flush=True)
        print(f"  saved to {out_path}", flush=True)

        summary[name] = {
            "n_pairs": n,
            "direction_norm": dir_norm,
            "ref_mean_norm": ref_mean_norm,
            "tgt_mean_norm": tgt_mean_norm,
            "cos_ref_tgt": cos_ref_tgt,
            "direction_path": str(out_path),
        }

    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
