"""Side-by-side comparison of A_full, B_thinking_2M, C_grounded_2M baseline (regular Gemini prompt) eval results.

Goal: surface where the per-track ordering A > B > C in return comes from.
For each metric we compute per-track values + the deltas A−B, B−C, A−C so
it is easy to read off "B drops achievement X relative to A" etc.

Outputs:
  - markdown table to stdout
  - JSON dump at --out

Usage:
  PYTHONPATH=. python tools/baseline_track_comparison.py \
      --out probe_results/baseline_track_comparison.json
"""
from __future__ import annotations

import argparse
import json
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

TRACKS = [
    ("A_full", "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly/freezenone_50ep"),
    ("B_thinking_2M", "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_think_predonly_top2M/freezenone_50ep"),
    ("C_grounded_2M", "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M/freezenone_50ep"),
]
INV_TRACKS = {
    "A_full": "/home/geney/Imagination/probe_results/inventory_counts/a_baseline.json",
    "B_thinking_2M": "/home/geney/Imagination/probe_results/inventory_counts/b_baseline.json",
    "C_grounded_2M": "/home/geney/Imagination/probe_results/inventory_counts/c_baseline.json",
}


def load_eps(eval_dir: Path) -> List[dict]:
    eps = []
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
            pass
    return eps


def aggregate(eps: List[dict]) -> Dict:
    if not eps:
        return {}
    rets = np.array([e["return"] for e in eps])
    lens = np.array([e["length"] for e in eps])
    n_achs = np.array([e["num_achievements"] for e in eps])

    # Achievement unlock rate (fraction of episodes that hit each achievement)
    ach_counter: Counter = Counter()
    # Achievement first-unlock step (mean across episodes that did unlock it)
    ach_step_lists: Dict[str, List[int]] = {}
    for e in eps:
        for ach, step in e.get("achievements", {}).items():
            ach_counter[ach] += 1
            ach_step_lists.setdefault(ach, []).append(int(step))

    # Action distribution
    a_counter = Counter()
    total_steps = 0
    for e in eps:
        for a in e.get("actions", []):
            a_counter[int(a)] += 1
            total_steps += 1

    return {
        "n": len(eps),
        "mean_return": float(rets.mean()),
        "se_return": float(rets.std(ddof=1) / np.sqrt(len(eps))),
        "mean_length": float(lens.mean()),
        "se_length": float(lens.std(ddof=1) / np.sqrt(len(eps))),
        "median_length": float(np.median(lens)),
        "mean_num_ach": float(n_achs.mean()),
        "se_num_ach": float(n_achs.std(ddof=1) / np.sqrt(len(eps))),
        "ach_unlock_rate": {a: ach_counter[a] / len(eps) for a in ach_counter},
        "ach_unlock_step_mean": {a: float(np.mean(ach_step_lists[a])) for a in ach_step_lists},
        "ach_unlock_step_count": {a: len(ach_step_lists[a]) for a in ach_step_lists},
        "action_freq": {ACTION_NAMES[i]: a_counter[i] / max(total_steps, 1) for i in range(len(ACTION_NAMES))},
        "total_steps": total_steps,
        "mean_return_per_step": float(rets.mean() / lens.mean()) if lens.mean() > 0 else 0,
        "mean_ach_per_step": float(n_achs.mean() / lens.mean()) if lens.mean() > 0 else 0,
    }


def fmt_pct(x: float) -> str:
    return f"{100*x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    aggs = {}
    for label, d in TRACKS:
        eps = load_eps(Path(d))
        agg = aggregate(eps)
        # Pull inventory counts if available
        inv_path = INV_TRACKS.get(label)
        if inv_path and Path(inv_path).exists():
            with open(inv_path) as f:
                inv = json.load(f)
            agg["inventory_per_ep"] = inv["summary"]["per_resource_mean_count"]
            agg["inventory_se_per_ep"] = inv["summary"]["per_resource_se_count"]
        else:
            agg["inventory_per_ep"] = {}
            agg["inventory_se_per_ep"] = {}
        aggs[label] = agg

    A, B, C = aggs["A_full"], aggs["B_thinking_2M"], aggs["C_grounded_2M"]

    # ===== Print readable summary =====
    print("# Track baseline comparison (regular Gemini prompt)\n")
    print(f"|                          | A_full       | B_thinking_2M | C_grounded_2M | A−B    | B−C    |")
    print(f"|---|---|---|---|---|---|")
    for label, key, fmt in [
        ("n episodes",          "n",                lambda v: f"{int(v)}"),
        ("mean return",         "mean_return",      lambda v: f"{v:.2f}"),
        ("se return",           "se_return",        lambda v: f"±{v:.2f}"),
        ("mean ep length",      "mean_length",      lambda v: f"{v:.0f}"),
        ("median ep length",    "median_length",    lambda v: f"{v:.0f}"),
        ("mean num achievements","mean_num_ach",    lambda v: f"{v:.1f}"),
        ("return / step",       "mean_return_per_step", lambda v: f"{v:.4f}"),
        ("achievements / step", "mean_ach_per_step", lambda v: f"{v:.4f}"),
    ]:
        a, b, c = A[key], B[key], C[key]
        d_ab = a - b
        d_bc = b - c
        sign_ab = f"{d_ab:+.2f}" if isinstance(a, float) else f"{d_ab:+d}"
        sign_bc = f"{d_bc:+.2f}" if isinstance(b, float) else f"{d_bc:+d}"
        print(f"| {label:<24} | {fmt(a):<12} | {fmt(b):<13} | {fmt(c):<13} | {sign_ab:<7}| {sign_bc:<7}|")

    # ===== Achievement unlock rates =====
    all_achs = sorted(set(A["ach_unlock_rate"]) | set(B["ach_unlock_rate"]) | set(C["ach_unlock_rate"]))
    print(f"\n# Achievement unlock rates (% of episodes that ever achieved it)\n")
    print(f"| achievement | A | B | C | A−B | B−C |")
    print(f"|---|---|---|---|---|---|")
    rows = []
    for ach in all_achs:
        a = A["ach_unlock_rate"].get(ach, 0.0)
        b = B["ach_unlock_rate"].get(ach, 0.0)
        c = C["ach_unlock_rate"].get(ach, 0.0)
        rows.append((ach, a, b, c))
    # Sort by max rate descending for readability
    rows.sort(key=lambda r: -max(r[1], r[2], r[3]))
    for ach, a, b, c in rows:
        d_ab = a - b
        d_bc = b - c
        # Mark notable shifts
        marker_ab = " ←" if abs(d_ab) >= 0.10 else ""
        marker_bc = " ←" if abs(d_bc) >= 0.10 else ""
        print(f"| {ach:<25} | {fmt_pct(a):>4} | {fmt_pct(b):>4} | {fmt_pct(c):>4} | {d_ab:+.2f}{marker_ab} | {d_bc:+.2f}{marker_bc} |")

    # ===== Achievement timing (when does it first happen?) =====
    print(f"\n# Mean first-unlock step (across episodes that achieved it; lower = earlier in episode)\n")
    print(f"| achievement | A μ-step (n) | B μ-step (n) | C μ-step (n) | A→C shift |")
    print(f"|---|---|---|---|---|")
    for ach in all_achs:
        a_step = A["ach_unlock_step_mean"].get(ach)
        b_step = B["ach_unlock_step_mean"].get(ach)
        c_step = C["ach_unlock_step_mean"].get(ach)
        a_n = A["ach_unlock_step_count"].get(ach, 0)
        b_n = B["ach_unlock_step_count"].get(ach, 0)
        c_n = C["ach_unlock_step_count"].get(ach, 0)
        if max(a_n, b_n, c_n) < 5:
            continue  # too few samples
        shift_ac = (c_step - a_step) if (a_step and c_step) else None
        a_str = f"{a_step:.0f} ({a_n})" if a_step else "—"
        b_str = f"{b_step:.0f} ({b_n})" if b_step else "—"
        c_str = f"{c_step:.0f} ({c_n})" if c_step else "—"
        shift_str = f"{shift_ac:+.0f}" if shift_ac is not None else "—"
        print(f"| {ach:<25} | {a_str:<14} | {b_str:<14} | {c_str:<14} | {shift_str:<8} |")

    # ===== Action distribution =====
    print(f"\n# Action frequency (fraction of all steps; LEFT/RIGHT/UP/DOWN are normalized to fraction of MOVE actions in parens)\n")
    interesting = ["LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "DESCEND",
                   "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE", "PLACE_TORCH",
                   "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE",
                   "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD"]
    print(f"| action | A | B | C | A−B | B−C |")
    print(f"|---|---|---|---|---|---|")
    for act in interesting:
        a = A["action_freq"].get(act, 0.0)
        b = B["action_freq"].get(act, 0.0)
        c = C["action_freq"].get(act, 0.0)
        print(f"| {act:<22} | {a:.3f} | {b:.3f} | {c:.3f} | {a-b:+.3f} | {b-c:+.3f} |")

    # ===== Inventory =====
    print(f"\n# Per-episode resource counts (replayed from saved actions; counts ALL increments, not just first)\n")
    inv_keys = sorted(set(list(A["inventory_per_ep"].keys()) + list(B["inventory_per_ep"].keys()) + list(C["inventory_per_ep"].keys())))
    if inv_keys:
        print(f"| resource / event | A | B | C | A−B | B−C |")
        print(f"|---|---|---|---|---|---|")
        for k in inv_keys:
            a = A["inventory_per_ep"].get(k, 0)
            b = B["inventory_per_ep"].get(k, 0)
            c = C["inventory_per_ep"].get(k, 0)
            a_se = A["inventory_se_per_ep"].get(k, 0)
            b_se = B["inventory_se_per_ep"].get(k, 0)
            c_se = C["inventory_se_per_ep"].get(k, 0)
            print(f"| {k:<22} | {a:5.2f}±{a_se:.2f} | {b:5.2f}±{b_se:.2f} | {c:5.2f}±{c_se:.2f} | {a-b:+5.2f} | {b-c:+5.2f} |")
    else:
        print(f"(no inventory data for at least one track)\n")

    # Dump to JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(aggs, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
