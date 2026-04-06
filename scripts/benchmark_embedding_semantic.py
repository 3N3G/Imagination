#!/usr/bin/env python3
"""
Benchmark embedding extraction strategies for SEMANTIC quality.

Tests whether embeddings capture meaningful game-state similarity:
1. Temporal coherence: nearby timesteps should be similar
2. Small perturbations: HP 9→8, drink +1, etc → high similarity
3. Large perturbations: HP 9→1, different floor, different inventory → low similarity
4. Totally different states: early game vs late game → low similarity

For each (layer, pooling) combo, reports a "semantic score" that measures
how well the embedding distances track meaningful state differences.
"""

import argparse
import copy
import json
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/home/geney/Imagination")
from envs.obs_to_text import obs_to_text
from pipeline.config import MAP_OBS_DIM

EMBED_MODEL = "Qwen/Qwen3-8B"
DEVICE = "cuda"
MAX_LENGTH = 2048


def load_obs(npz_path):
    """Load and decode observations from filtered NPZ."""
    d = np.load(npz_path)
    obs_map_bits = d["obs_map_bits"]
    obs_aux = d["obs_aux"].astype(np.float32)
    obs_map = np.unpackbits(obs_map_bits, axis=1, bitorder="little")[:, :MAP_OBS_DIM]
    obs = np.concatenate([obs_map, obs_aux], axis=1).astype(np.float32)
    return obs


def perturb_obs(obs, dim_idx, new_val):
    """Create a copy of obs with one auxiliary dimension changed."""
    o = obs.copy()
    o[MAP_OBS_DIM + dim_idx] = new_val
    return o


# Auxiliary dimension indices (relative to MAP_OBS_DIM=8217)
# From decode_inventory_section in obs_to_text.py, the aux dims are:
# 0-17: inventory (wood, stone, coal, iron, diamond, sapphire, ruby, sapling, torches, arrows, books, potions×6)
# 18-21: sword level (one-hot 0-3)
# 22-26: health, food, drink, energy, mana (floats 0-1)
# 27-30: xp, dex, str, int (floats)
# 31-32: is_sleeping, is_resting
# 33: direction (as 1/num_actions)
# etc.

# Health is at aux index 22, food=23, drink=24, energy=25, mana=26
HP_IDX = 22
FOOD_IDX = 23
DRINK_IDX = 24
ENERGY_IDX = 25
FLOOR_IDX = 43  # approximate


def build_test_pairs(obs_all):
    """
    Build pairs of (text_a, text_b, expected_similarity) where expected_similarity
    is 'high', 'medium', or 'low'.
    """
    n = len(obs_all)
    pairs = []

    # --- 1. Temporal coherence: adjacent steps (expect HIGH similarity) ---
    step_indices = np.linspace(0, n - 2, min(20, n - 1), dtype=int)
    for i in step_indices:
        pairs.append({
            "text_a": obs_to_text(obs_all[i]),
            "text_b": obs_to_text(obs_all[i + 1]),
            "expected": "high",
            "label": f"temporal_adj_{i}",
        })

    # --- 2. Small perturbations (expect HIGH similarity) ---
    anchor_indices = np.linspace(0, n - 1, min(10, n), dtype=int)
    for i in anchor_indices:
        base = obs_all[i]
        base_text = obs_to_text(base)
        hp_val = base[MAP_OBS_DIM + HP_IDX]

        # HP change by ~10% (small)
        if hp_val > 0.1:
            perturbed = perturb_obs(base, HP_IDX, hp_val - 0.1)
            pairs.append({
                "text_a": base_text,
                "text_b": obs_to_text(perturbed),
                "expected": "high",
                "label": f"small_hp_{i}",
            })

        # Drink change by small amount
        drink_val = base[MAP_OBS_DIM + DRINK_IDX]
        if drink_val > 0.1:
            perturbed = perturb_obs(base, DRINK_IDX, drink_val - 0.1)
            pairs.append({
                "text_a": base_text,
                "text_b": obs_to_text(perturbed),
                "expected": "high",
                "label": f"small_drink_{i}",
            })

    # --- 3. Large perturbations (expect LOW similarity) ---
    for i in anchor_indices:
        base = obs_all[i]
        base_text = obs_to_text(base)

        # HP from current to near-death
        perturbed = perturb_obs(base, HP_IDX, 0.05)
        pairs.append({
            "text_a": base_text,
            "text_b": obs_to_text(perturbed),
            "expected": "low",
            "label": f"large_hp_crash_{i}",
        })

        # All resources drained
        perturbed = base.copy()
        for d in [HP_IDX, FOOD_IDX, DRINK_IDX, ENERGY_IDX]:
            perturbed[MAP_OBS_DIM + d] = 0.05
        pairs.append({
            "text_a": base_text,
            "text_b": obs_to_text(perturbed),
            "expected": "low",
            "label": f"large_all_drain_{i}",
        })

    # --- 4. Distant timesteps (expect LOW-MEDIUM similarity) ---
    for _ in range(20):
        i = np.random.randint(0, n // 4)
        j = np.random.randint(3 * n // 4, n)
        pairs.append({
            "text_a": obs_to_text(obs_all[i]),
            "text_b": obs_to_text(obs_all[j]),
            "expected": "low",
            "label": f"distant_{i}_{j}",
        })

    # --- 5. Same state, same text (expect PERFECT similarity, sanity check) ---
    for i in anchor_indices[:5]:
        text = obs_to_text(obs_all[i])
        pairs.append({
            "text_a": text,
            "text_b": text,
            "expected": "perfect",
            "label": f"identical_{i}",
        })

    return pairs


def extract_single(texts, model, tokenizer, layer, pool_method):
    """Extract embeddings for a list of texts."""
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=MAX_LENGTH, padding=True,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[layer]

    mask = attention_mask.unsqueeze(-1).float()

    if pool_method == "mean":
        summed = (hidden.float() * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        emb = summed / lengths
    elif pool_method == "last_token":
        seq_lens = attention_mask.sum(dim=1) - 1
        emb = hidden[torch.arange(len(texts)), seq_lens].float()
    elif pool_method == "max":
        hidden_masked = hidden.float() * mask + (~mask.bool()).float() * (-1e9)
        emb, _ = hidden_masked.max(dim=1)
    else:
        raise ValueError(f"Unknown: {pool_method}")

    return emb.cpu().numpy().astype(np.float32)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_config(pairs, model, tokenizer, layer, pool_method, batch_size=8):
    """Evaluate one (layer, pool) configuration on all pairs."""
    # Collect unique texts
    all_texts = []
    text_to_idx = {}
    for p in pairs:
        for key in ("text_a", "text_b"):
            t = p[key]
            if t not in text_to_idx:
                text_to_idx[t] = len(all_texts)
                all_texts.append(t)

    # Embed all texts
    all_embs = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        emb = extract_single(batch, model, tokenizer, layer, pool_method)
        all_embs.append(emb)
    all_embs = np.concatenate(all_embs, axis=0)

    # Z-normalize (as training would)
    mean = all_embs.mean(axis=0)
    std = all_embs.std(axis=0)
    std[std < 1e-8] = 1.0
    all_embs_z = (all_embs - mean) / std

    # Compute cosine similarities
    results = {"perfect": [], "high": [], "low": []}
    for p in pairs:
        ia = text_to_idx[p["text_a"]]
        ib = text_to_idx[p["text_b"]]
        sim_raw = cosine_sim(all_embs[ia], all_embs[ib])
        sim_z = cosine_sim(all_embs_z[ia], all_embs_z[ib])
        cat = p["expected"]
        results[cat].append({"sim_raw": sim_raw, "sim_z": sim_z, "label": p["label"]})

    return results


def score_results(results):
    """
    Compute a semantic quality score.
    Good embeddings: high sim for 'high' pairs, low sim for 'low' pairs.
    Score = mean(high_sims) - mean(low_sims)  (higher is better, max ~2.0)
    """
    high_z = [r["sim_z"] for r in results.get("high", [])]
    low_z = [r["sim_z"] for r in results.get("low", [])]
    perfect_z = [r["sim_z"] for r in results.get("perfect", [])]

    high_raw = [r["sim_raw"] for r in results.get("high", [])]
    low_raw = [r["sim_raw"] for r in results.get("low", [])]

    return {
        "high_z_mean": np.mean(high_z) if high_z else 0,
        "low_z_mean": np.mean(low_z) if low_z else 0,
        "gap_z": np.mean(high_z) - np.mean(low_z) if high_z and low_z else 0,
        "high_raw_mean": np.mean(high_raw) if high_raw else 0,
        "low_raw_mean": np.mean(low_raw) if low_raw else 0,
        "gap_raw": np.mean(high_raw) - np.mean(low_raw) if high_raw and low_raw else 0,
        "perfect_z_mean": np.mean(perfect_z) if perfect_z else 0,
        "n_high": len(high_z),
        "n_low": len(low_z),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                    default="/data/group_data/rl/geney/oracle_pipeline/test_filtered/trajectories_000000.npz")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    print("Loading observations...")
    obs_all = load_obs(args.data)
    print(f"  {len(obs_all)} observations loaded")

    print("Building test pairs...")
    np.random.seed(42)
    pairs = build_test_pairs(obs_all)
    n_high = sum(1 for p in pairs if p["expected"] == "high")
    n_low = sum(1 for p in pairs if p["expected"] == "low")
    n_perf = sum(1 for p in pairs if p["expected"] == "perfect")
    print(f"  {len(pairs)} pairs: {n_high} high, {n_low} low, {n_perf} perfect")

    # Count unique texts
    unique_texts = set()
    for p in pairs:
        unique_texts.add(p["text_a"])
        unique_texts.add(p["text_b"])
    print(f"  {len(unique_texts)} unique texts to embed")

    print(f"\nLoading model: {EMBED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        EMBED_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"  {num_layers} layers, hidden_size={model.config.hidden_size}")

    # Configs to test
    pool_methods = ["mean", "last_token", "max"]
    layers = [9, 18, 27, 30, 35, 36]

    print(f"\n{'Config':25s} {'high_z':>8s} {'low_z':>8s} {'GAP_z':>8s} {'high_raw':>8s} {'low_raw':>8s} {'GAP_raw':>8s} {'perfect':>8s}")
    print("-" * 100)

    all_scores = []
    for layer in layers:
        for pool in pool_methods:
            label = f"L{layer:02d}_{pool}"
            results = evaluate_config(pairs, model, tokenizer, layer, pool, args.batch_size)
            s = score_results(results)
            all_scores.append((label, s))
            print(
                f"{label:25s} "
                f"{s['high_z_mean']:+8.4f} "
                f"{s['low_z_mean']:+8.4f} "
                f"{s['gap_z']:+8.4f} "
                f"{s['high_raw_mean']:+8.4f} "
                f"{s['low_raw_mean']:+8.4f} "
                f"{s['gap_raw']:+8.4f} "
                f"{s['perfect_z_mean']:+8.4f}"
            )

    # Sort by gap_z
    print("\n=== Ranked by z-normalized gap (high - low) ===")
    all_scores.sort(key=lambda x: -x[1]["gap_z"])
    for i, (label, s) in enumerate(all_scores):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:2d}. {label:25s}  gap_z={s['gap_z']:+.4f}  (high={s['high_z_mean']:+.4f}, low={s['low_z_mean']:+.4f}){marker}")

    # Also print detailed breakdown for top 3
    print("\n=== Detailed pair-level results for top 3 ===")
    for label, s in all_scores[:3]:
        print(f"\n--- {label} ---")
        results = evaluate_config(pairs, model, tokenizer,
                                   int(label.split("_")[0][1:]),
                                   "_".join(label.split("_")[1:]),
                                   args.batch_size)
        for cat in ("high", "low", "perfect"):
            items = results.get(cat, [])
            if items:
                sims = [r["sim_z"] for r in items]
                print(f"  {cat:8s} ({len(items):3d} pairs): z_sim = {np.mean(sims):+.4f} ± {np.std(sims):.4f}  "
                      f"[{np.min(sims):+.4f} .. {np.max(sims):+.4f}]")


if __name__ == "__main__":
    main()
