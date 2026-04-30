"""Generate a steerability + performance scorecard for a SCALING_C variant.

Reads per-cell results.json + per-episode summary.json from
/data/user_data/geney/eval_results_temp/psf_v3_pporn_1e8_grounded_<tag>_steer_score/
or for Track C from
/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M/.

Computes:
  - Performance: mean return on baseline / v2 / thresh6
  - Steerability: per-prompt verdict (WIN | NULL | WRONG-WAY) based on
    target metric direction with z-score threshold |z| >= 1.0
  - Total WIN-rate vs. # NULL vs. # WRONG-WAY across the matrix

Usage:
    PYTHONPATH=. python tools/scorecard.py --variant xhighb
    PYTHONPATH=. python tools/scorecard.py --variant track_c
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

ACTIONS = ["NOOP","LEFT","RIGHT","UP","DOWN","DO","SLEEP","PLACE_STONE",
    "PLACE_TABLE","PLACE_FURNACE","PLACE_PLANT","MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE","MAKE_IRON_PICKAXE","MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD","MAKE_IRON_SWORD","REST","DESCEND","ASCEND",
    "MAKE_DIAMOND_PICKAXE","MAKE_DIAMOND_SWORD","MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOUR","SHOOT_ARROW","MAKE_ARROW","CAST_FIREBALL",
    "CAST_ICEBALL","PLACE_TORCH","DRINK_POTION_RED","DRINK_POTION_GREEN",
    "DRINK_POTION_BLUE","DRINK_POTION_PINK","DRINK_POTION_CYAN",
    "DRINK_POTION_YELLOW","READ_BOOK","ENCHANT_SWORD","ENCHANT_ARMOUR",
    "MAKE_TORCH","LEVEL_UP_DEX","LEVEL_UP_STR","LEVEL_UP_INT","ENCHANT_BOW"]
A = {n: i for i, n in enumerate(ACTIONS)}

# Cell definitions: (mode, target_metric_name, direction)
CELLS = {
    "performance": [
        ("baseline_concise", "return", "REF"),
        ("achievement_max_v2", "return", "REF"),
        ("achievement_max_v2_thresh6", "return", "REF"),
    ],
    "low_level": [
        ("direction_left_v2", "move_share_LEFT", "UP"),
        ("direction_right_v2", "move_share_RIGHT", "UP"),
        ("direction_up_v2", "move_share_UP", "UP"),
        ("direction_down_v2", "move_share_DOWN", "UP"),
        ("target_drink_water_v2", "drink_intake_events", "UP"),
        ("avoid_water_v2", "drink_intake_events", "DOWN"),
    ],
    "high_level": [
        ("target_collect_stone_v2", "stone", "UP"),
        ("target_avoid_stone_v2", "stone", "DOWN"),
        ("target_place_stone_v2", "action_PLACE_STONE", "UP"),
        ("target_eat_cow_v2", "cow_eat_events", "UP"),
        ("target_hunt_animals_v2", "cow_eat_events", "UP"),
        ("avoid_animals_v2", "cow_eat_events", "DOWN"),
        ("target_descend_v2", "action_DESCEND", "UP"),
        ("target_stay_overworld_v2", "action_DESCEND", "DOWN"),
        ("target_collect_sapling_v2", "sapling", "UP"),
        ("target_place_plant_v2", "action_PLACE_PLANT", "UP"),
        ("target_defeat_zombie_v2", "monsters_killed", "UP"),
        ("target_make_iron_pickaxe_v2", "ach_make_iron_pickaxe", "UP"),
        ("target_collect_diamond_v2", "diamond", "UP"),
        ("die_fast_v2", "length", "DOWN"),
        ("survive_long_v2", "length", "UP"),
    ],
}


def variant_root(tag: str) -> Path:
    if tag == "track_c":
        return Path("/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M")
    return Path(f"/data/user_data/geney/eval_results_temp/psf_v3_pporn_1e8_grounded_{tag}_steer_score")


def find_cell_dir(root: Path, mode: str, n: int = 30):
    candidates = [root / f"{mode}_{n}ep"]
    if root.is_dir():
        for sub in root.iterdir():
            if sub.is_dir():
                candidates.append(sub / f"{mode}_{n}ep")
                candidates.append(sub / f"{mode}_30ep")
                candidates.append(sub / f"{mode}_50ep")
                candidates.append(sub / f"{mode}")
    for c in candidates:
        if c.is_dir() and (c / "results.json").exists():
            return c
    return None


def read_per_ep(eval_dir: Path):
    eps = []
    if not eval_dir.is_dir(): return eps
    for ep in sorted(eval_dir.iterdir()):
        if not (ep.is_dir() and ep.name.startswith("episode_")): continue
        sf = ep / "summary.json"
        if not sf.exists(): continue
        d = json.load(open(sf))
        actions = d.get("actions", []) or []
        ach = list(d.get("achievements", {}).keys()) if isinstance(d.get("achievements"), dict) else d.get("achievements", [])
        eps.append({"return": d.get("return", 0), "length": d.get("length", 0),
                    "actions": actions, "achievements": ach})
    return eps


def metric_value(eps, metric):
    if not eps: return None
    n = len(eps)
    def per_ep_count(act_id):
        return [sum(1 for a in e["actions"] if a == act_id) for e in eps]
    def per_ep_share(act_id):
        out = []
        for e in eps:
            move = sum(1 for a in e["actions"] if a in (A["LEFT"], A["RIGHT"], A["UP"], A["DOWN"]))
            this = sum(1 for a in e["actions"] if a == act_id)
            out.append(this/move if move > 0 else 0)
        return out
    if metric == "return":               xs = [e["return"] for e in eps]
    elif metric == "length":             xs = [e["length"] for e in eps]
    elif metric == "action_DESCEND":     xs = per_ep_count(A["DESCEND"])
    elif metric == "action_PLACE_STONE": xs = per_ep_count(A["PLACE_STONE"])
    elif metric == "action_PLACE_PLANT": xs = per_ep_count(A["PLACE_PLANT"])
    elif metric == "move_share_LEFT":    xs = per_ep_share(A["LEFT"])
    elif metric == "move_share_RIGHT":   xs = per_ep_share(A["RIGHT"])
    elif metric == "move_share_UP":      xs = per_ep_share(A["UP"])
    elif metric == "move_share_DOWN":    xs = per_ep_share(A["DOWN"])
    elif metric == "stone":              xs = [1 if "collect_stone" in e["achievements"] else 0 for e in eps]
    elif metric == "cow_eat_events":     xs = [1 if "eat_cow" in e["achievements"] else 0 for e in eps]
    elif metric == "drink_intake_events": xs = [1 if "collect_drink" in e["achievements"] else 0 for e in eps]
    elif metric == "sapling":            xs = [1 if "collect_sapling" in e["achievements"] else 0 for e in eps]
    elif metric == "monsters_killed":
        xs = [(1 if "defeat_zombie" in e["achievements"] else 0) +
              (1 if "defeat_skeleton" in e["achievements"] else 0) +
              (1 if "defeat_orc_solider" in e["achievements"] else 0) +
              (1 if "defeat_orc_mage" in e["achievements"] else 0)
              for e in eps]
    elif metric == "ach_make_iron_pickaxe":
        xs = [1 if "make_iron_pickaxe" in e["achievements"] else 0 for e in eps]
    elif metric == "diamond":
        xs = [1 if "collect_diamond" in e["achievements"] else 0 for e in eps]
    else:
        return None
    m = sum(xs) / n
    if n < 2: return (m, 0, 0, n)
    var = sum((x-m)**2 for x in xs) / (n-1)
    sd = var**0.5
    se = sd / math.sqrt(n)
    return (m, sd, se, n)


def verdict(z, direction):
    if direction == "UP":
        if z >= 1.0: return "WIN"
        if z <= -1.0: return "WRONG-WAY"
    elif direction == "DOWN":
        if z <= -1.0: return "WIN"
        if z >= 1.0: return "WRONG-WAY"
    return "NULL"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--baseline-cell", default="baseline_concise")
    args = p.parse_args()

    root = variant_root(args.variant)
    print(f"# Scorecard: {args.variant}")
    print(f"_root: {root}_\n")

    bl_dir = None
    for n_try in [args.num_episodes, 50, 30, 15]:
        bl_dir = find_cell_dir(root, args.baseline_cell, n_try)
        if bl_dir: break
    if bl_dir is None:
        # Track C special case: it has freezenone_50ep as baseline
        if args.variant == "track_c":
            for sub in root.iterdir() if root.is_dir() else []:
                if "freezenone_50ep" in sub.name:
                    bl_dir = sub
                    break
    if bl_dir is None:
        print(f"  ! No baseline ({args.baseline_cell}) for {args.variant}")
        sys.exit(2)

    bl_eps = read_per_ep(bl_dir)
    print(f"## Baseline ({bl_dir.name}, n={len(bl_eps)})\n")
    for k in ["return", "length", "action_DESCEND", "move_share_DOWN", "cow_eat_events", "drink_intake_events"]:
        m = metric_value(bl_eps, k)
        if m: print(f"  {k}: {m[0]:.3f} ± {m[2]:.3f}")
    print()

    print("## Performance\n")
    print("| cell | n | return | length | ach/ep |")
    print("|---|---|---|---|---|")
    perf = {}
    for mode, _, _ in CELLS["performance"]:
        cd = None
        for n_try in [args.num_episodes, 50, 30]:
            cd = find_cell_dir(root, mode, n_try)
            if cd: break
        if cd is None:
            print(f"| {mode} | - | (missing) | - | - |"); continue
        eps = read_per_ep(cd)
        n = len(eps)
        if n == 0: print(f"| {mode} | - | (no eps) | - | - |"); continue
        rets = [e["return"] for e in eps]
        lens = [e["length"] for e in eps]
        achs = [len(e["achievements"]) for e in eps]
        rm = sum(rets)/n
        rsd = (sum((r-rm)**2 for r in rets)/(n-1))**0.5 if n>1 else 0
        rse = rsd/math.sqrt(n)
        perf[mode] = rm
        print(f"| {mode} | {n} | {rm:.2f} ± {rse:.2f} | {sum(lens)/n:.0f} | {sum(achs)/n:.1f} |")

    def steerability_section(label, cells):
        print(f"\n## Steerability — {label}\n")
        print("| cell | target | dir | baseline | cell | Δ | z | verdict |")
        print("|---|---|---|---|---|---|---|---|")
        win = nul = wrong = 0
        for mode, target, direction in cells:
            cd = find_cell_dir(root, mode, args.num_episodes)
            if cd is None:
                print(f"| {mode} | {target} | {direction} | - | (missing) | - | - | - |"); continue
            eps = read_per_ep(cd)
            if not eps:
                print(f"| {mode} | {target} | {direction} | - | (no eps) | - | - | - |"); continue
            bm = metric_value(bl_eps, target)
            cm = metric_value(eps, target)
            if bm is None or cm is None: continue
            delta = cm[0] - bm[0]
            pooled_se = math.sqrt(bm[2]**2 + cm[2]**2)
            z = delta / pooled_se if pooled_se > 0 else 0
            v = verdict(z, direction)
            if v == "WIN": win += 1
            elif v == "NULL": nul += 1
            elif v == "WRONG-WAY": wrong += 1
            print(f"| {mode} | {target} | {direction} | {bm[0]:.3f} | {cm[0]:.3f} | {delta:+.3f} | {z:+.2f} | **{v}** |")
        print(f"\n_{label}: **WIN {win} / NULL {nul} / WRONG-WAY {wrong}**_")
        return win, nul, wrong

    wL, nL, wrL = steerability_section("Low-Level (directional + intrinsic)", CELLS["low_level"])
    wH, nH, wrH = steerability_section("High-Level (target/avoid/atomic)", CELLS["high_level"])

    print("\n## Headline summary\n")
    print(f"- **Performance**: baseline={perf.get('baseline_concise','?')}, "
          f"v2={perf.get('achievement_max_v2','?')}, "
          f"thresh6={perf.get('achievement_max_v2_thresh6','?')}")
    total = wL + nL + wrL + wH + nH + wrH
    print(f"- **Steerability**: {wL+wH}/{total} WIN total ({wL}/{wL+nL+wrL} low-level, {wH}/{wH+nH+wrH} high-level)")


if __name__ == "__main__":
    main()
