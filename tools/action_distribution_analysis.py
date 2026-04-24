"""Analyze action-distribution shifts across eval result directories.

For each eval result dir, compute:
  - Per-action frequency (LEFT, RIGHT, UP, DOWN, DO, …)
  - Movement-only fraction (LEFT/RIGHT/UP/DOWN out of all actions)
  - Direction balance: LEFT vs RIGHT, UP vs DOWN
  - DESCEND, SLEEP, PLACE_STONE rates (rare-action rates)

Used as the action-level steerability evidence for direction_X_v2 and
target_X_v2 prompts.

Usage:
  python tools/action_distribution_analysis.py \
      --baseline-dir <regular_eval_dir> \
      --condition-dir <condition_eval_dir> [--condition-dir ...] \
      --condition-label <label> [--condition-label ...] \
      --out probe_results/action_analysis/<run>.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

ACTION_NAMES = [
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
    "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD", "MAKE_IRON_SWORD", "REST", "DESCEND", "ASCEND",
    "MAKE_DIAMOND_PICKAXE", "MAKE_DIAMOND_SWORD", "MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOUR", "SHOOT_ARROW", "MAKE_ARROW", "CAST_FIREBALL",
    "CAST_ICEBALL", "PLACE_TORCH", "DRINK_POTION_RED", "DRINK_POTION_GREEN",
    "DRINK_POTION_BLUE", "DRINK_POTION_PINK", "DRINK_POTION_CYAN",
    "DRINK_POTION_YELLOW", "READ_BOOK", "ENCHANT_SWORD", "ENCHANT_ARMOUR",
    "MAKE_TORCH", "LEVEL_UP_DEX", "LEVEL_UP_STR", "LEVEL_UP_INT",
    "ENCHANT_BOW",
]
NAME_TO_IDX = {n: i for i, n in enumerate(ACTION_NAMES)}
MOVE_NAMES = ["LEFT", "RIGHT", "UP", "DOWN"]


def load_episode_summaries(eval_dir: Path) -> List[dict]:
    eps = []
    if not eval_dir.exists():
        return eps
    for ep_dir in sorted(eval_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        sf = ep_dir / "summary.json"
        if not sf.exists():
            continue
        try:
            with sf.open() as f:
                eps.append(json.load(f))
        except Exception:
            continue
    return eps


def aggregate_actions(eps: List[dict]) -> dict:
    if not eps:
        return {"n": 0, "mean_steps": 0, "action_freq": {}, "move_fraction": 0,
                "left_frac_of_moves": 0, "right_frac_of_moves": 0,
                "up_frac_of_moves": 0, "down_frac_of_moves": 0,
                "do_rate": 0, "sleep_rate": 0, "descend_rate": 0,
                "ascend_rate": 0, "place_stone_rate": 0,
                "rare_action_rates": {}}
    counter = Counter()
    total_steps = 0
    for e in eps:
        for a in e.get("actions", []):
            counter[int(a)] += 1
            total_steps += 1
    if total_steps == 0:
        return {"n": len(eps), "mean_steps": 0, "action_freq": {}}
    freq = {ACTION_NAMES[i]: counter.get(i, 0) / total_steps for i in range(len(ACTION_NAMES))}
    move_total = sum(counter.get(NAME_TO_IDX[m], 0) for m in MOVE_NAMES)
    out = {
        "n": len(eps),
        "total_steps": total_steps,
        "mean_steps": total_steps / len(eps),
        "action_freq": freq,
        "move_fraction": move_total / total_steps,
        "left_frac_of_moves":  counter.get(NAME_TO_IDX["LEFT"],  0) / max(move_total, 1),
        "right_frac_of_moves": counter.get(NAME_TO_IDX["RIGHT"], 0) / max(move_total, 1),
        "up_frac_of_moves":    counter.get(NAME_TO_IDX["UP"],    0) / max(move_total, 1),
        "down_frac_of_moves":  counter.get(NAME_TO_IDX["DOWN"],  0) / max(move_total, 1),
        "do_rate":          freq.get("DO", 0),
        "sleep_rate":       freq.get("SLEEP", 0),
        "descend_rate":     freq.get("DESCEND", 0),
        "ascend_rate":      freq.get("ASCEND", 0),
        "place_stone_rate": freq.get("PLACE_STONE", 0),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--condition-dir", type=Path, action="append", required=True)
    ap.add_argument("--condition-label", type=str, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if len(args.condition_dir) != len(args.condition_label):
        raise SystemExit("Need one --condition-label per --condition-dir")

    base_eps = load_episode_summaries(args.baseline_dir)
    base = aggregate_actions(base_eps)
    print(f"\nbaseline: {args.baseline_dir}")
    print(f"  n={base['n']}  total_steps={base.get('total_steps', 0)}  mean_steps={base['mean_steps']:.1f}")
    print(f"  move_frac={base['move_fraction']:.3f}  L/R/U/D of moves: "
          f"{base['left_frac_of_moves']:.2f}/{base['right_frac_of_moves']:.2f}/"
          f"{base['up_frac_of_moves']:.2f}/{base['down_frac_of_moves']:.2f}")
    print(f"  do_rate={base['do_rate']:.3f}  sleep_rate={base['sleep_rate']:.4f}  "
          f"descend_rate={base['descend_rate']:.4f}  place_stone={base['place_stone_rate']:.4f}")

    rows = []
    for label, cdir in zip(args.condition_label, args.condition_dir):
        eps = load_episode_summaries(cdir)
        cond = aggregate_actions(eps)
        if cond["n"] == 0:
            print(f"\n{label}: NO EPISODES")
            continue
        # deltas relative to baseline
        d = lambda k: cond[k] - base[k]
        print(f"\n{label}:  n={cond['n']}  mean_steps={cond['mean_steps']:.1f}")
        print(f"  move_frac={cond['move_fraction']:.3f} (Δ={d('move_fraction'):+.3f})")
        print(f"  LEFT  of moves: {cond['left_frac_of_moves']:.3f} (Δ={d('left_frac_of_moves'):+.3f})")
        print(f"  RIGHT of moves: {cond['right_frac_of_moves']:.3f} (Δ={d('right_frac_of_moves'):+.3f})")
        print(f"  UP    of moves: {cond['up_frac_of_moves']:.3f} (Δ={d('up_frac_of_moves'):+.3f})")
        print(f"  DOWN  of moves: {cond['down_frac_of_moves']:.3f} (Δ={d('down_frac_of_moves'):+.3f})")
        print(f"  DO rate:        {cond['do_rate']:.3f} (Δ={d('do_rate'):+.3f})")
        print(f"  DESCEND rate:   {cond['descend_rate']:.4f} (Δ={d('descend_rate'):+.4f})")
        print(f"  PLACE_STONE:    {cond['place_stone_rate']:.4f} (Δ={d('place_stone_rate'):+.4f})")
        print(f"  SLEEP rate:     {cond['sleep_rate']:.4f} (Δ={d('sleep_rate'):+.4f})")
        rows.append({"label": label, "dir": str(cdir), "stats": cond,
                     "deltas": {k: float(d(k)) for k in
                                ["move_fraction", "left_frac_of_moves", "right_frac_of_moves",
                                 "up_frac_of_moves", "down_frac_of_moves",
                                 "do_rate", "sleep_rate", "descend_rate",
                                 "ascend_rate", "place_stone_rate"]}})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"baseline_dir": str(args.baseline_dir), "baseline": base, "rows": rows},
                  f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
