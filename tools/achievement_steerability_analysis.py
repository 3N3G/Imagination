"""Analyze achievement counts across a set of eval result directories.

For each eval result dir, compute:
  - per-episode achievement counts (count of unique achievements unlocked)
  - per-achievement frequency across all episodes (e.g. "eat_cow rate")
  - mean return + SE
  - mean episode length
  - mean num_unique_achievements

Then build a comparison table: for each (track, condition) pair, side-by-side
return + per-achievement-of-interest deltas vs the regular baseline.

Output: probe_results/steerability_analysis/<run_label>.json + a summary
markdown table to stdout.

Usage:
  python tools/achievement_steerability_analysis.py \
      --baseline-dir <regular_eval_dir> \
      --condition-dir <condition_eval_dir> [--condition-dir ...] \
      --condition-label <label> [--condition-label ...] \
      --achievements-of-interest collect_stone descend_floor eat_cow \
      --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


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


def aggregate(eps: List[dict]) -> dict:
    if not eps:
        return {"n": 0, "mean_return": 0, "se_return": 0, "mean_length": 0,
                "mean_num_ach": 0, "achievement_counts": {}}
    rets = np.array([e["return"] for e in eps])
    lens = np.array([e["length"] for e in eps])
    n_achs = np.array([e["num_achievements"] for e in eps])
    counter: Counter = Counter()
    for e in eps:
        for ach in e.get("achievements", {}).keys():
            counter[ach] += 1
    return {
        "n": int(len(eps)),
        "mean_return": float(rets.mean()),
        "se_return": float(rets.std(ddof=0) / math.sqrt(len(eps))) if len(eps) > 1 else 0.0,
        "mean_length": float(lens.mean()),
        "mean_num_ach": float(n_achs.mean()),
        "achievement_counts": dict(counter),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--condition-dir", type=Path, action="append", required=True)
    ap.add_argument("--condition-label", type=str, action="append", required=True)
    ap.add_argument("--achievements-of-interest", nargs="*", default=[
        "collect_wood", "collect_stone", "collect_coal", "collect_iron",
        "collect_drink", "collect_diamond",
        "eat_cow", "eat_plant",
        "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
        "make_wood_sword", "make_stone_sword", "make_iron_sword",
        "place_stone", "place_table", "place_furnace", "place_torch",
        "wake_up", "defeat_zombie", "defeat_skeleton",
        "enter_dungeon", "enter_gnomish_mines", "descend_floor",
    ])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if len(args.condition_dir) != len(args.condition_label):
        raise SystemExit("Need one --condition-label per --condition-dir")

    base_eps = load_episode_summaries(args.baseline_dir)
    base_agg = aggregate(base_eps)
    print(f"\nbaseline: {args.baseline_dir}")
    print(f"  n={base_agg['n']}  return={base_agg['mean_return']:.2f}±{base_agg['se_return']:.2f}  "
          f"length={base_agg['mean_length']:.1f}  num_ach={base_agg['mean_num_ach']:.2f}")

    rows = []
    for label, cdir in zip(args.condition_label, args.condition_dir):
        eps = load_episode_summaries(cdir)
        agg = aggregate(eps)
        rets_b = np.array([e["return"] for e in base_eps]) if base_eps else np.array([])
        rets_c = np.array([e["return"] for e in eps]) if eps else np.array([])
        # SE of difference
        se_diff = math.sqrt(
            (rets_b.std(ddof=0)**2 / max(len(rets_b),1)) +
            (rets_c.std(ddof=0)**2 / max(len(rets_c),1))
        ) if len(rets_b) and len(rets_c) else 0.0
        delta_return = agg["mean_return"] - base_agg["mean_return"]
        z_return = delta_return / se_diff if se_diff > 0 else 0.0

        ach_deltas: Dict[str, dict] = {}
        for ach in args.achievements_of_interest:
            base_count = base_agg["achievement_counts"].get(ach, 0)
            cond_count = agg["achievement_counts"].get(ach, 0)
            base_rate = base_count / max(base_agg["n"], 1)
            cond_rate = cond_count / max(agg["n"], 1)
            ach_deltas[ach] = {
                "baseline_count": base_count,
                "cond_count": cond_count,
                "baseline_rate": base_rate,
                "cond_rate": cond_rate,
                "delta_rate": cond_rate - base_rate,
                "ratio": (cond_rate / base_rate) if base_rate > 0 else float("inf") if cond_rate > 0 else 1.0,
            }

        row = {
            "label": label,
            "dir": str(cdir),
            "n": agg["n"],
            "mean_return": agg["mean_return"],
            "se_return": agg["se_return"],
            "delta_return": delta_return,
            "se_diff_return": se_diff,
            "z_return": z_return,
            "mean_length": agg["mean_length"],
            "mean_num_ach": agg["mean_num_ach"],
            "achievement_counts": agg["achievement_counts"],
            "achievement_deltas": ach_deltas,
        }
        rows.append(row)

        print(f"\n{label}:")
        print(f"  n={agg['n']}  return={agg['mean_return']:.2f}±{agg['se_return']:.2f}  "
              f"Δret={delta_return:+.2f} (z={z_return:+.2f})  "
              f"length={agg['mean_length']:.1f}  num_ach={agg['mean_num_ach']:.2f}")
        # show interesting ach deltas (where delta_rate magnitude > 0.05)
        big = [(a, ach_deltas[a]) for a in args.achievements_of_interest
               if abs(ach_deltas[a]["delta_rate"]) > 0.05]
        for a, d in sorted(big, key=lambda x: -abs(x[1]["delta_rate"])):
            arrow = "↑" if d["delta_rate"] > 0 else "↓"
            print(f"    {a:25s} base={d['baseline_rate']:.2f} cond={d['cond_rate']:.2f}  "
                  f"Δrate={d['delta_rate']:+.2f} {arrow}")

    out_data = {
        "baseline_dir": str(args.baseline_dir),
        "baseline_agg": base_agg,
        "rows": rows,
        "achievements_of_interest": args.achievements_of_interest,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
