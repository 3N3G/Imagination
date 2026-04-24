"""Build the master steerability results table by scanning all eval dirs.

Walks all known psf_v2 cadence5 eval result directories, computes
per-condition (return, length, achievements, action distribution) and
formats as a markdown table for inclusion in the journal.

Per cell columns:
  - return ± SE (n=)
  - Δreturn vs baseline (z)
  - mean episode length
  - num achievements (mean)
  - movement actions (LEFT, RIGHT, UP, DOWN, DO, DESCEND, SLEEP)
  - tracked-achievement rates: collect_stone, eat_cow, descend_floor,
    enter_dungeon, place_stone, sleep_wake_up
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, OrderedDict
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
TRACKED_ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN", "DO", "DESCEND", "SLEEP"]
TRACKED_ACHS = ["collect_stone", "collect_drink", "eat_cow", "eat_plant",
                "descend_floor", "enter_dungeon", "place_stone",
                "wake_up", "make_wood_pickaxe", "make_stone_pickaxe"]

# (track_label, baseline_dir, condition_dirs_pattern_root)
TRACKS = [
    ("A_full",
     "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly/freezenone_50ep",
     {
         "v1_die":              "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly/v1_die_50ep",
         "die_v2":              "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_v2_probe/die_v2_50ep",
         "adversarial_v2":      "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_v2_probe/adversarial_v2_50ep",
         "avoid_water_v2":      "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_v2_probe/avoid_water_v2_50ep",
         "avoid_animals_v2":    "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_v2_probe/avoid_animals_v2_50ep",
         "target_collect_stone_v2": "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_steer_v2/target_collect_stone_v2_50ep",
         "target_descend_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_steer_v2/target_descend_v2_50ep",
         "target_eat_cow_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_steer_v2/target_eat_cow_v2_50ep",
         "target_drink_water_v2":"/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_steer_v2/target_drink_water_v2_50ep",
         "target_place_stone_v2":"/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_steer_v2/target_place_stone_v2_50ep",
         "direction_left_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_steer_v2/direction_left_v2_50ep",
         "direction_right_v2":  "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_full_steer_v2/direction_right_v2_50ep",
     }),
    ("A_top2M",
     "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_top2M/freezenone_50ep",
     {
         "die_v2":              "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_top2M_v2_probe/die_v2_50ep",
         "adversarial_v2":      "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_top2M_v2_probe/adversarial_v2_50ep",
         "target_collect_stone_v2": "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_top2M_steer_v2/target_collect_stone_v2_50ep",
         "target_descend_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_top2M_steer_v2/target_descend_v2_50ep",
         "target_eat_cow_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_top2M_steer_v2/target_eat_cow_v2_50ep",
         "direction_left_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_top2M_steer_v2/direction_left_v2_50ep",
         "direction_right_v2":  "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_top2M_steer_v2/direction_right_v2_50ep",
     }),
    ("B_thinking_2M",
     "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_think_predonly_top2M/freezenone_50ep",
     {
         "die_v2":              "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_think_predonly_top2M_v2_probe/die_v2_50ep",
         "adversarial_v2":      "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_think_predonly_top2M_v2_probe/adversarial_v2_50ep",
         "avoid_water_v2":      "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_think_predonly_top2M_v2_probe/avoid_water_v2_50ep",
         "avoid_animals_v2":    "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_think_predonly_top2M_v2_probe/avoid_animals_v2_50ep",
     }),
    ("C_grounded_2M",
     "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M/freezenone_50ep",
     {
         "die_v2":              "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_v2_probe/die_v2_50ep",
         "adversarial_v2":      "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_v2_probe/adversarial_v2_50ep",
         "avoid_water_v2":      "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_v2_probe/avoid_water_v2_50ep",
         "avoid_animals_v2":    "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_v2_probe/avoid_animals_v2_50ep",
         "target_collect_stone_v2": "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_steer_v2/target_collect_stone_v2_50ep",
         "target_descend_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_steer_v2/target_descend_v2_50ep",
         "target_eat_cow_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_steer_v2/target_eat_cow_v2_50ep",
         "direction_left_v2":   "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_steer_v2/direction_left_v2_50ep",
         "direction_right_v2":  "/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M_steer_v2/direction_right_v2_50ep",
     }),
]


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


def aggregate(eps: List[dict]) -> dict:
    if not eps:
        return None
    rets = np.array([e["return"] for e in eps])
    lens = np.array([e["length"] for e in eps])
    n_achs = np.array([e["num_achievements"] for e in eps])
    counter: Counter = Counter()
    for e in eps:
        for ach in e.get("achievements", {}).keys():
            counter[ach] += 1
    # Action freq aggregated
    a_counter = Counter()
    total_steps = 0
    for e in eps:
        for a in e.get("actions", []):
            a_counter[int(a)] += 1
            total_steps += 1
    action_freq = {n: a_counter.get(NAME_TO_IDX[n], 0) / max(total_steps, 1) for n in TRACKED_ACTIONS}
    return {
        "n": len(eps),
        "mean_return": float(rets.mean()),
        "se_return": float(rets.std(ddof=0) / math.sqrt(len(eps))) if len(eps) > 1 else 0.0,
        "mean_length": float(lens.mean()),
        "mean_num_ach": float(n_achs.mean()),
        "ach_rate": {a: counter[a] / len(eps) for a in TRACKED_ACHS},
        "action_freq": action_freq,
    }


def fmt_pct(x): return f"{100*x:.0f}%" if x is not None else "—"
def fmt_ret(a): return f"{a['mean_return']:.2f}±{a['se_return']:.2f}" if a else "—"
def fmt_dret(a, b):
    if not a or not b: return "—"
    d = a['mean_return'] - b['mean_return']
    se_diff = math.sqrt(a['se_return']**2 + b['se_return']**2)
    z = d / se_diff if se_diff > 0 else 0
    return f"{d:+.2f} (z={z:+.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    full = OrderedDict()

    md_lines = []
    md_lines.append("# Steerability master table\n")
    md_lines.append(f"Auto-generated. Tracked actions: {', '.join(TRACKED_ACTIONS)}.")
    md_lines.append(f"Tracked achievements: {', '.join(TRACKED_ACHS)}.\n")

    for track_label, base_dir, conds in TRACKS:
        base_eps = load_eps(Path(base_dir))
        base = aggregate(base_eps)
        full[track_label] = {"baseline": base, "conditions": {}}
        if not base:
            md_lines.append(f"## {track_label}\n*(no baseline data at {base_dir})*\n")
            continue
        md_lines.append(f"## {track_label}")
        md_lines.append(f"**Baseline (regular)**: n={base['n']} return={fmt_ret(base)} "
                        f"len={base['mean_length']:.0f} num_ach={base['mean_num_ach']:.1f}\n")
        # Action freq baseline
        md_lines.append("Baseline action freq:")
        md_lines.append(" | ".join(f"{a}={base['action_freq'][a]:.3f}" for a in TRACKED_ACTIONS))
        md_lines.append("")
        md_lines.append("Baseline achievement rates:")
        md_lines.append(" | ".join(f"{a}={base['ach_rate'][a]:.2f}" for a in TRACKED_ACHS))
        md_lines.append("")

        # Build table per condition
        head = "| Condition | n | return | Δret (z) | len | n_ach | " + \
               " | ".join(f"Δp({a})" for a in TRACKED_ACTIONS) + " | " + \
               " | ".join(f"Δrate({a})" for a in TRACKED_ACHS) + " |"
        sep = "|" + "---|" * (6 + len(TRACKED_ACTIONS) + len(TRACKED_ACHS))
        md_lines.append(head)
        md_lines.append(sep)

        for cond_label, cond_dir in conds.items():
            cond_eps = load_eps(Path(cond_dir))
            cond = aggregate(cond_eps)
            full[track_label]["conditions"][cond_label] = cond
            if not cond:
                md_lines.append(f"| {cond_label} | 0 | — | — | — | — |"
                                + " — |" * (len(TRACKED_ACTIONS) + len(TRACKED_ACHS)))
                continue
            row = [
                cond_label,
                str(cond['n']),
                fmt_ret(cond),
                fmt_dret(cond, base),
                f"{cond['mean_length']:.0f}",
                f"{cond['mean_num_ach']:.1f}",
            ]
            for a in TRACKED_ACTIONS:
                d = cond['action_freq'][a] - base['action_freq'][a]
                # mark direction-of-interest match in bold (e.g., direction_left should ↑LEFT)
                row.append(f"{d:+.3f}")
            for a in TRACKED_ACHS:
                d = cond['ach_rate'][a] - base['ach_rate'][a]
                row.append(f"{d:+.2f}")
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_md.open("w") as f:
        f.write("\n".join(md_lines))
    with args.out_json.open("w") as f:
        json.dump(full, f, indent=2, default=str)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
