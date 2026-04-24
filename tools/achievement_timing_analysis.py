"""Compare WHEN achievements first happen, across conditions.

The achievements field in episode summary.json maps achievement name →
step of first unlock. This script builds a per-achievement step distribution
across episodes for each condition, then compares to baseline.

A meaningful steering signal: target_descend_v2 makes the policy reach
descend_floor / enter_dungeon EARLIER than baseline (smaller mean step),
even if the eventual rate is similar.

Usage:
  PYTHONPATH=. python tools/achievement_timing_analysis.py \
      --baseline-dir <regular_eval> \
      --condition-dir <cond1> --condition-label "label1" \
      --condition-dir <cond2> --condition-label "label2" \
      --achievements-of-interest descend_floor enter_dungeon eat_cow ...
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np


def load_eps(eval_dir: Path) -> List[dict]:
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


def per_ach_timings(eps: List[dict], ach_name: str):
    timings = []
    for e in eps:
        achs = e.get("achievements", {})
        if ach_name in achs:
            timings.append(int(achs[ach_name]))
    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--condition-dir", type=Path, action="append", required=True)
    ap.add_argument("--condition-label", type=str, action="append", required=True)
    ap.add_argument("--achievements-of-interest", nargs="*", default=[
        "collect_wood", "collect_stone", "collect_drink",
        "make_wood_pickaxe", "make_stone_pickaxe",
        "eat_cow", "eat_plant",
        "place_stone", "place_table", "place_furnace",
        "wake_up", "defeat_zombie", "defeat_skeleton",
        "enter_dungeon", "descend_floor",
    ])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if len(args.condition_dir) != len(args.condition_label):
        raise SystemExit("Need one --condition-label per --condition-dir")

    base_eps = load_eps(args.baseline_dir)
    print(f"baseline: {args.baseline_dir} (n={len(base_eps)})")

    rows = []
    print(f"\n{'achievement':<25} {'baseline':<25}  " +
          "  ".join(f"{l[:22]:<25}" for l in args.condition_label))
    for ach in args.achievements_of_interest:
        base_t = per_ach_timings(base_eps, ach)
        base_n = len(base_t)
        if base_n == 0:
            base_str = f"n=0/--          --"
        else:
            base_mean = np.mean(base_t)
            base_str = f"n={base_n:2d}/{len(base_eps):2d} ({base_n/len(base_eps):.0%})  μ={base_mean:5.0f}"
        cond_strs = []
        cond_data = []
        for label, cdir in zip(args.condition_label, args.condition_dir):
            eps = load_eps(cdir)
            cond_t = per_ach_timings(eps, ach)
            cond_n = len(cond_t)
            if cond_n == 0:
                s = f"n=0/{len(eps):2d}          --"
            else:
                cond_mean = np.mean(cond_t)
                if base_n > 0:
                    delta = cond_mean - base_mean
                    s = f"n={cond_n:2d}/{len(eps):2d} ({cond_n/len(eps):.0%})  μ={cond_mean:5.0f} (Δ={delta:+5.0f})"
                else:
                    s = f"n={cond_n:2d}/{len(eps):2d} ({cond_n/len(eps):.0%})  μ={cond_mean:5.0f}"
            cond_strs.append(s)
            cond_data.append({"label": label, "n": cond_n, "n_eps": len(eps),
                              "mean_step": float(np.mean(cond_t)) if cond_n else None,
                              "median_step": float(np.median(cond_t)) if cond_n else None})
        rows.append({"achievement": ach, "baseline": {
                        "n": base_n, "n_eps": len(base_eps),
                        "mean_step": float(np.mean(base_t)) if base_n else None,
                        "median_step": float(np.median(base_t)) if base_n else None,
                    }, "conditions": cond_data})
        print(f"{ach:<25} {base_str:<25}  " + "  ".join(f"{s:<25}" for s in cond_strs))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump({"baseline_dir": str(args.baseline_dir),
                       "rows": rows}, f, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
