"""Build the specificity matrix for the 21-cell × C_grounded_2M sweep.

For each cell:
  - Run/read replay_inventory_counts JSON (per-episode counts)
  - Read summary.json files for direction-action counts + achievement rates + length
  - Compute target metric per row (one cell per row, one column per metric)
  - Δ vs baseline + z-score
  - Diagonal cells SHOULD diverge from baseline; off-diagonal SHOULD stay near.

Usage:
  PYTHONPATH=. python tools/specificity_matrix.py \
    --eval-root /data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_specificity \
    --counts-root probe_results/inventory_counts/specificity \
    --baseline probe_results/inventory_counts/c_baseline_full.json \
    --out-md docs/SPECIFICITY_MATRIX.md \
    --out-json probe_results/specificity_matrix.json \
    [--regen-counts]   # re-run replay tool for any cell missing a counts JSON
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


CELLS = [
    ("target_collect_stone_v2",     "stone",                 "UP"),
    ("target_avoid_stone_v2",       "stone",                 "DOWN"),
    ("target_place_stone_v2",       "action_PLACE_STONE",    "UP"),
    ("target_eat_cow_v2",           "cow_eat_events",        "UP"),
    ("target_hunt_animals_v2",      "cow_eat_events",        "UP"),
    ("avoid_animals_v2",            "cow_eat_events",        "DOWN"),
    ("target_drink_water_v2",       "drink_intake_events",   "UP"),
    ("avoid_water_v2",              "drink_intake_events",   "DOWN"),
    ("target_descend_v2",           "action_DESCEND",        "UP"),
    ("target_stay_overworld_v2",    "action_DESCEND",        "DOWN"),
    ("target_collect_sapling_v2",   "sapling",               "UP"),
    ("target_place_plant_v2",       "action_PLACE_PLANT",    "UP"),
    ("target_defeat_zombie_v2",     "monsters_killed_total", "UP"),
    ("target_make_iron_pickaxe_v2", "ach_make_iron_pickaxe", "UP"),
    ("target_collect_diamond_v2",   "diamond",               "UP"),
    ("die_fast_v2",                 "length",                "DOWN"),
    ("survive_long_v2",             "length",                "UP"),
    # Direction cells use share-of-cardinal-moves (length-invariant + handles NOOP shift)
    ("direction_left_v2",           "move_share_LEFT",       "UP"),
    ("direction_right_v2",          "move_share_RIGHT",      "UP"),
    ("direction_up_v2",             "move_share_UP",         "UP"),
    ("direction_down_v2",           "move_share_DOWN",       "UP"),
]

# v3 prompt iterations (overwrites of NULL/wrong-way v2 cells).
# These live in the _specificity_iter eval dir, not _specificity.
CELLS_V3 = [
    ("target_avoid_stone_v3",       "stone",                 "DOWN"),
    ("survive_long_v3",             "length",                "UP"),
    ("target_eat_cow_v3",           "cow_eat_events",        "UP"),
    ("target_drink_water_v3",       "drink_intake_events",   "UP"),
    ("target_stay_overworld_v3",    "action_DESCEND",        "DOWN"),
    ("target_place_plant_v3",       "action_PLACE_PLANT",    "UP"),
    ("target_defeat_zombie_v3",     "monsters_killed_total", "UP"),
    ("target_collect_sapling_v3",   "sapling",               "UP"),
]

# Columns to display in the matrix (the "axes")
COLUMNS = [
    "stone", "action_PLACE_STONE",
    "cow_eat_events", "drink_intake_events",
    "action_DESCEND", "sapling", "action_PLACE_PLANT",
    "monsters_killed_total", "ach_make_iron_pickaxe", "diamond",
    "length",
    "action_LEFT", "action_RIGHT", "action_UP", "action_DOWN",
    "rate_action_LEFT", "rate_action_RIGHT", "rate_action_UP", "rate_action_DOWN",
    "move_share_LEFT", "move_share_RIGHT", "move_share_UP", "move_share_DOWN",
    "return",
]

ACTIONS = ["NOOP","LEFT","RIGHT","UP","DOWN","DO","SLEEP","PLACE_STONE",
           "PLACE_TABLE","PLACE_FURNACE","PLACE_PLANT"]
ACTION_IDX = {n: i for i, n in enumerate(ACTIONS)}


def compute_summary_metrics(eval_dir: Path) -> Dict[str, Dict[str, float]]:
    """Per-episode action counts (+ per-step rates), return, length, and achievement rates."""
    per_ep = {"action_LEFT": [], "action_RIGHT": [], "action_UP": [], "action_DOWN": [],
              "rate_action_LEFT": [], "rate_action_RIGHT": [],
              "rate_action_UP": [], "rate_action_DOWN": [],
              "move_share_LEFT": [], "move_share_RIGHT": [],
              "move_share_UP": [], "move_share_DOWN": [],
              "return": [], "length": []}
    achs = []  # list of sets per ep
    for ep_dir in sorted(eval_dir.iterdir()):
        sf = ep_dir / "summary.json"
        if not sf.exists(): continue
        with sf.open() as f:
            d = json.load(f)
        c = Counter(int(a) for a in d["actions"])
        L = max(d["length"], 1)
        cardinal_total = sum(c.get(ACTION_IDX[n], 0) for n in ["LEFT", "RIGHT", "UP", "DOWN"])
        for n in ["LEFT", "RIGHT", "UP", "DOWN"]:
            cnt = c.get(ACTION_IDX[n], 0)
            per_ep[f"action_{n}"].append(cnt)
            per_ep[f"rate_action_{n}"].append(cnt / L)
            per_ep[f"move_share_{n}"].append(cnt / cardinal_total if cardinal_total > 0 else 0.0)
        per_ep["return"].append(d["return"])
        per_ep["length"].append(d["length"])
        achs.append(set(d.get("achievements", {}).keys()))
    n = len(per_ep["return"])
    out = {}
    for k, v in per_ep.items():
        arr = np.array(v, dtype=float)
        out[k] = {
            "mean": float(arr.mean()) if n > 0 else 0.0,
            "se":   float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "n":    n,
        }
    # achievement rates as ach_<name>
    for ach_name in ["make_iron_pickaxe", "collect_diamond", "defeat_zombie"]:
        rate = sum(1 for s in achs if ach_name in s) / max(n, 1)
        # rate SE = sqrt(p(1-p)/n)
        se = math.sqrt(rate * (1 - rate) / max(n, 1)) if n > 0 else 0.0
        out[f"ach_{ach_name}"] = {"mean": rate, "se": se, "n": n}
    return out


def load_counts_json(jp: Path) -> Dict[str, Dict[str, float]]:
    """Convert replay_inventory_counts JSON output → {key: {mean, se, n}}."""
    with jp.open() as f:
        d = json.load(f)
    s = d["summary"]
    n = s["n"]
    out = {}
    for k, m in s["per_resource_mean_count"].items():
        out[k] = {
            "mean": m,
            "se": s["per_resource_se_count"].get(k, 0.0),
            "n": n,
        }
    return out


def maybe_run_counts(eval_dir: Path, counts_path: Path, max_ep: int = 30):
    if counts_path.exists():
        return
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "tools/replay_inventory_counts.py",
        "--eval-dir", str(eval_dir),
        "--max-episodes", str(max_ep),
        "--out", str(counts_path),
    ]
    print(f"  [running] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_baseline(p: Path) -> Dict[str, Dict[str, float]]:
    with p.open() as f:
        d = json.load(f)
    out = {}
    for k, v in d["per_episode_counts"].items():
        out[k] = {"mean": v["mean"], "se": v["se"], "n": d["n"]}
    out["return"] = {"mean": d["return_mean"], "se": d["return_se"], "n": d["n"]}
    out["length"] = {"mean": d["length_mean"], "se": d["length_se"], "n": d["n"]}
    for ach_name, rate in d["achievement_rates"].items():
        n = d["n"]
        se = math.sqrt(rate * (1 - rate) / n) if n > 0 else 0.0
        out[f"ach_{ach_name}"] = {"mean": rate, "se": se, "n": n}
    # Compute baseline per-step rates for direction actions from absolute count / length
    L = d["length_mean"]
    for n_act in ["LEFT", "RIGHT", "UP", "DOWN"]:
        cnt = out[f"action_{n_act}"]
        rate = cnt["mean"] / L if L > 0 else 0.0
        # Approximate SE via delta method: rate_se ~ count_se/L (length variance ignored)
        rate_se = cnt["se"] / L if L > 0 else 0.0
        out[f"rate_action_{n_act}"] = {"mean": rate, "se": rate_se, "n": cnt["n"]}
    return out


def fmt_cell(metric: str, m: float, se: float, base: float, base_se: float,
             is_target: bool, direction: Optional[str]) -> str:
    """Markdown cell. Bold if target cell. Show z-score in parens."""
    diff = m - base
    pooled = math.sqrt(se ** 2 + base_se ** 2)
    z = diff / pooled if pooled > 0 else 0.0
    sign = "+" if diff >= 0 else ""
    # Decimals: rates 0–1 → 3 decimals, counts 1 decimal
    if metric.startswith("ach_"):
        body = f"{m:.2f} ({sign}{diff:.2f}, z={z:+.1f})"
    elif metric.startswith("rate_"):
        body = f"{m:.3f} ({sign}{diff:.3f}, z={z:+.1f})"
    elif metric == "length":
        body = f"{m:.0f} ({sign}{diff:.0f}, z={z:+.1f})"
    else:
        body = f"{m:.1f} ({sign}{diff:.1f}, z={z:+.1f})"
    if is_target:
        if direction == "UP":
            ok = z >= 1
        else:
            ok = z <= -1
        marker = "**" if ok else "*"
        return f"{marker}{body}{marker}"
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path,
                    default=Path("/data/group_data/rl/geney/eval_results/"
                                 "psf_v2_cadence5_grounded_predonly_top2M_specificity"))
    ap.add_argument("--iter-eval-root", type=Path,
                    default=Path("/data/group_data/rl/geney/eval_results/"
                                 "psf_v2_cadence5_grounded_predonly_top2M_specificity_iter"),
                    help="Root for v3 iteration cells")
    ap.add_argument("--counts-root", type=Path,
                    default=Path("probe_results/inventory_counts/specificity"))
    ap.add_argument("--baseline", type=Path,
                    default=Path("probe_results/inventory_counts/c_baseline_full.json"))
    ap.add_argument("--baseline-eval-dir", type=Path,
                    default=Path("/data/group_data/rl/geney/eval_results/"
                                 "psf_v2_cadence5_grounded_predonly_top2M/freezenone_50ep"))
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--regen-counts", action="store_true",
                    help="Run replay tool for any cell whose counts JSON is missing")
    ap.add_argument("--max-episodes", type=int, default=30)
    args = ap.parse_args()

    base = load_baseline(args.baseline)
    # Augment baseline with summary-derived metrics (move_share_*, rate_action_*) from eval dir
    if args.baseline_eval_dir.exists():
        base_summary = compute_summary_metrics(args.baseline_eval_dir)
        for k, v in base_summary.items():
            if k.startswith("move_share_") or k.startswith("rate_action_"):
                base[k] = v

    def collect(cells, root):
        out = []
        for cell, target, direction in cells:
            eval_dir = root / f"{cell}_30ep"
            counts_path = args.counts_root / f"{cell}.json"
            if not eval_dir.exists():
                print(f"  [skip] {cell}: no eval dir at {eval_dir}")
                out.append((cell, target, direction, 0, None))
                continue
            if args.regen_counts:
                try:
                    maybe_run_counts(eval_dir, counts_path, max_ep=args.max_episodes)
                except subprocess.CalledProcessError as e:
                    print(f"  [error] replay failed for {cell}: {e}")
            cell_metrics = {}
            if counts_path.exists():
                cell_metrics.update(load_counts_json(counts_path))
            cell_metrics.update(compute_summary_metrics(eval_dir))
            n = cell_metrics.get("return", {}).get("n", 0)
            out.append((cell, target, direction, n, cell_metrics))
        return out

    rows = collect(CELLS, args.eval_root)
    iter_rows = collect(CELLS_V3, args.iter_eval_root)

    # Build markdown
    md = []
    md.append("# Specificity matrix — C_grounded_2M (n=30 per cell)")
    md.append("")
    md.append("Each row = one prompt. Each column = one target metric.")
    md.append("Diagonal cells (where row's target = column) marked **bold** if z≥1 in")
    md.append("the wanted direction; `*italic*` if not (failed steering).")
    md.append("")
    md.append("Baseline (no prompt, regular concise) row first.")
    md.append("")
    head = ["Prompt", "n"] + COLUMNS
    md.append("| " + " | ".join(head) + " |")
    md.append("|" + "---|" * len(head))

    # Baseline row
    base_row = ["**baseline**", str(base["return"]["n"])]
    for col in COLUMNS:
        b = base.get(col, {"mean": 0.0, "se": 0.0})
        if col.startswith("ach_"):
            base_row.append(f"{b['mean']:.2f}")
        elif col == "length":
            base_row.append(f"{b['mean']:.0f}")
        else:
            base_row.append(f"{b['mean']:.1f}")
    md.append("| " + " | ".join(base_row) + " |")

    # Per-cell rows
    for cell, target, direction, n, metrics in rows:
        if metrics is None:
            md.append(f"| {cell} | 0 |" + " — |" * len(COLUMNS))
            continue
        row = [cell, str(n)]
        for col in COLUMNS:
            m = metrics.get(col)
            b = base.get(col)
            if m is None or b is None:
                row.append("—")
                continue
            is_target = (col == target)
            row.append(fmt_cell(col, m["mean"], m["se"], b["mean"], b["se"],
                                is_target, direction))
        md.append("| " + " | ".join(row) + " |")

    md.append("")
    md.append("## Diagonal verdict")
    md.append("")
    md.append("| Prompt | target | direction | baseline | cell | Δ | z | verdict |")
    md.append("|---|---|---|---|---|---|---|---|")
    for cell, target, direction, n, metrics in rows:
        if metrics is None:
            md.append(f"| {cell} | {target} | {direction} | — | — | — | — | NO DATA |")
            continue
        m = metrics.get(target)
        b = base.get(target)
        if m is None or b is None:
            md.append(f"| {cell} | {target} | {direction} | — | — | — | — | METRIC MISSING |")
            continue
        diff = m["mean"] - b["mean"]
        pooled = math.sqrt(m["se"] ** 2 + b["se"] ** 2)
        z = diff / pooled if pooled > 0 else 0
        ok = (z >= 1 and direction == "UP") or (z <= -1 and direction == "DOWN")
        wrong = (z <= -1 and direction == "UP") or (z >= 1 and direction == "DOWN")
        verdict = "WIN" if ok else ("WRONG-WAY" if wrong else "NULL")
        if target.startswith("ach_"):
            base_str = f"{b['mean']:.2f}"; cell_str = f"{m['mean']:.2f}"
        elif target.startswith("rate_"):
            base_str = f"{b['mean']:.3f}"; cell_str = f"{m['mean']:.3f}"
        elif target == "length":
            base_str = f"{b['mean']:.0f}"; cell_str = f"{m['mean']:.0f}"
        else:
            base_str = f"{b['mean']:.1f}"; cell_str = f"{m['mean']:.1f}"
        md.append(f"| {cell} | {target} | {direction} | {base_str} | {cell_str} | "
                  f"{diff:+.2f} | {z:+.1f} | {verdict} |")

    # v3 iteration verdict table
    md.append("")
    md.append("## v3 prompt iterations (overwrites of NULL/wrong-way v2 cells)")
    md.append("")
    md.append("Each v3 row shows the same target metric as its v2 progenitor, but with the rewritten prompt run on `psf_v2_cadence5_grounded_predonly_top2M_specificity_iter`.")
    md.append("")
    md.append("| Prompt | target | direction | baseline | cell | Δ | z | verdict |")
    md.append("|---|---|---|---|---|---|---|---|")
    iter_serializable = {}
    for cell, target, direction, n, metrics in iter_rows:
        if metrics is None:
            md.append(f"| {cell} | {target} | {direction} | — | — | — | — | NO DATA |")
            iter_serializable[cell] = {"target": target, "direction": direction, "n": 0, "metrics": {}}
            continue
        m = metrics.get(target)
        b = base.get(target)
        if m is None or b is None:
            md.append(f"| {cell} | {target} | {direction} | — | — | — | — | METRIC MISSING |")
            iter_serializable[cell] = {"target": target, "direction": direction, "n": n, "metrics": metrics}
            continue
        diff = m["mean"] - b["mean"]
        pooled = math.sqrt(m["se"] ** 2 + b["se"] ** 2)
        z = diff / pooled if pooled > 0 else 0
        ok = (z >= 1 and direction == "UP") or (z <= -1 and direction == "DOWN")
        wrong = (z <= -1 and direction == "UP") or (z >= 1 and direction == "DOWN")
        verdict = "WIN" if ok else ("WRONG-WAY" if wrong else "NULL")
        if target.startswith("ach_"):
            base_str = f"{b['mean']:.2f}"; cell_str = f"{m['mean']:.2f}"
        elif target.startswith("rate_"):
            base_str = f"{b['mean']:.3f}"; cell_str = f"{m['mean']:.3f}"
        elif target == "length":
            base_str = f"{b['mean']:.0f}"; cell_str = f"{m['mean']:.0f}"
        else:
            base_str = f"{b['mean']:.1f}"; cell_str = f"{m['mean']:.1f}"
        md.append(f"| {cell} | {target} | {direction} | {base_str} | {cell_str} | "
                  f"{diff:+.2f} | {z:+.1f} | {verdict} |")
        iter_serializable[cell] = {"target": target, "direction": direction, "n": n, "metrics": metrics}

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "baseline": {k: v for k, v in base.items()},
        "cells": {cell: {"target": target, "direction": direction, "n": n,
                          "metrics": metrics or {}}
                  for cell, target, direction, n, metrics in rows},
        "iter_cells": iter_serializable,
    }
    args.out_json.write_text(json.dumps(serializable, indent=2))
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
