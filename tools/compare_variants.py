"""Side-by-side scorecard comparison across multiple SCALING_C variants.

Usage:
    PYTHONPATH=. python tools/compare_variants.py track_c xhighb xxhighb [more...]

Reads the same per-cell results.json + per-episode summary.json files as
tools/scorecard.py, but emits a single table with one row per cell and one
column per variant. Useful for "which variant is best on which axis?".
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scorecard import (
    A, ACTIONS, CELLS, variant_root, find_cell_dir, read_per_ep,
    metric_value, verdict
)


def collect(variant: str, num_episodes: int = 30):
    """Return dict: {cell -> (target_metric_value, return, verdict_vs_baseline)}."""
    root = variant_root(variant)
    bl_dir = None
    for n_try in [num_episodes, 50, 30]:
        bl_dir = find_cell_dir(root, "baseline_concise", n_try)
        if bl_dir: break
    if bl_dir is None and variant == "track_c":
        for sub in root.iterdir() if root.is_dir() else []:
            if "freezenone_50ep" in sub.name:
                bl_dir = sub; break
    bl_eps = read_per_ep(bl_dir) if bl_dir else []
    out = {}
    for tier, cells in CELLS.items():
        for mode, target, direction in cells:
            cd = None
            for n_try in [num_episodes, 50, 30]:
                cd = find_cell_dir(root, mode, n_try)
                if cd: break
            if cd is None: out[mode] = None; continue
            eps = read_per_ep(cd)
            if not eps: out[mode] = None; continue
            n = len(eps)
            ret_mean = sum(e["return"] for e in eps) / n
            tm = metric_value(eps, target)
            if direction == "REF" or not bl_eps:
                v = "REF"; z = 0
            else:
                bm = metric_value(bl_eps, target)
                if bm is None or tm is None: v = "?"; z = 0
                else:
                    delta = tm[0] - bm[0]
                    pse = math.sqrt(bm[2]**2 + tm[2]**2)
                    z = delta/pse if pse > 0 else 0
                    v = verdict(z, direction)
            out[mode] = {"return": ret_mean, "target": tm[0] if tm else None, "z": z, "verdict": v, "n": n,
                         "direction": direction, "target_name": target}
    out["_baseline_n"] = len(bl_eps)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="+")
    p.add_argument("--num-episodes", type=int, default=30)
    args = p.parse_args()

    data = {}
    for v in args.variants:
        data[v] = collect(v, args.num_episodes)

    # Performance table
    print("# Multi-variant comparison\n")
    print(f"_n={args.num_episodes} unless otherwise noted_\n")
    print("## Performance (return)\n")
    print("| cell |", " | ".join(args.variants), "|")
    print("|---|" + "---|" * len(args.variants))
    for mode, _, _ in CELLS["performance"]:
        cells = []
        for v in args.variants:
            d = data[v].get(mode)
            cells.append(f"{d['return']:.2f}" if d else "—")
        print(f"| {mode} | " + " | ".join(cells) + " |")

    # Steerability tables
    for tier, cells in CELLS.items():
        if tier == "performance": continue
        print(f"\n## Steerability — {tier}\n")
        print("| cell | target | dir |", " | ".join(args.variants), "|")
        print("|---|---|---|" + "---|" * len(args.variants))
        for mode, target, direction in cells:
            row = [mode, target, direction]
            for v in args.variants:
                d = data[v].get(mode)
                if d is None: row.append("—")
                else:
                    row.append(f"{d['target']:.2f} (z={d['z']:+.1f}, {d['verdict']})")
            print("| " + " | ".join(row) + " |")

    # WIN totals
    print(f"\n## WIN-rate summary\n")
    for v in args.variants:
        win = nul = wrong = miss = 0
        for tier in ["low_level", "high_level"]:
            for mode, _, dir_ in CELLS[tier]:
                d = data[v].get(mode)
                if d is None: miss += 1; continue
                if d["verdict"] == "WIN": win += 1
                elif d["verdict"] == "NULL": nul += 1
                elif d["verdict"] == "WRONG-WAY": wrong += 1
        total = win + nul + wrong
        print(f"- **{v}**: {win}/{total} WIN ({nul} NULL, {wrong} WRONG-WAY) [missing {miss} cells]")


if __name__ == "__main__":
    main()
