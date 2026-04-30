"""Replay episodes from a single eval cell and compute the distribution of
HP / Food / Drink / Energy across all steps.

Output: per-intrinsic histogram (count of steps at each level 0..9),
plus mean/p25/p50/p75 + fraction of steps below thresholds {2,4,6}.

Compares two cells side by side (e.g., baseline_concise vs achievement_max_v2_thresh6).

Usage:
    PYTHONPATH=. python tools/intrinsics_analysis.py \\
        --variant xhighb \\
        --cells baseline_concise_30ep achievement_max_v2_thresh6_30ep \\
        [--max-eps 30]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name


def replay_episode(env, env_params, rng_in, reset_key, actions):
    """Step through a recorded action sequence; return per-step intrinsics."""
    obs, state = env.reset(reset_key, env_params)
    intrinsics = []
    intrinsics.append((float(state.player_health),
                       float(state.player_food),
                       float(state.player_drink),
                       float(state.player_energy)))
    rng = rng_in
    for a in actions:
        rng, sk = jax.random.split(rng)
        obs, state, reward, done, info = env.step(sk, state, int(a), env_params)
        intrinsics.append((float(state.player_health),
                           float(state.player_food),
                           float(state.player_drink),
                           float(state.player_energy)))
        if done: break
    return intrinsics


def analyze_cell(cell_dir: Path, max_eps=30, seed=42):
    """Load episodes' actions, replay, accumulate per-intrinsic counts."""
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(seed)

    eps = []
    for ep in sorted(cell_dir.iterdir()):
        if not (ep.is_dir() and ep.name.startswith("episode_")): continue
        sf = ep / "summary.json"
        if not sf.exists(): continue
        d = json.load(open(sf))
        if not d.get("actions"): continue
        eps.append(d)
    eps = eps[:max_eps]
    print(f"  Replaying {len(eps)} episodes from {cell_dir.name}...", file=sys.stderr)

    hist = {  # intrinsic_name -> {level -> step count}
        "HP": [0]*10, "Food": [0]*10, "Drink": [0]*10, "Energy": [0]*10,
    }
    total_steps = 0
    for ep in eps:
        rng, reset_key = jax.random.split(rng)
        intr = replay_episode(env, env_params, rng, reset_key, ep["actions"])
        for hp, food, drink, energy in intr:
            for name, v in [("HP", hp), ("Food", food), ("Drink", drink), ("Energy", energy)]:
                lvl = max(0, min(9, int(round(v))))
                hist[name][lvl] += 1
            total_steps += 1
    return hist, total_steps


def fmt_distribution(hist, total):
    out = {}
    for name, counts in hist.items():
        pct = [c/total*100 if total else 0 for c in counts]
        # Compute fraction below thresholds
        below = {}
        for thr in [2, 4, 6]:
            below[thr] = sum(counts[:thr+1]) / total * 100 if total else 0
        # Mean
        mean = sum(i*c for i, c in enumerate(counts)) / total if total else 0
        out[name] = {"counts": counts, "pct": pct, "below_pct": below, "mean": mean}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--cells", nargs="+", required=True,
                   help="Cell directory names (e.g., baseline_concise_30ep)")
    p.add_argument("--max-eps", type=int, default=30)
    args = p.parse_args()

    if args.variant == "track_c":
        root = Path("/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M")
    else:
        root = Path(f"/data/user_data/geney/eval_results_temp/psf_v3_pporn_1e8_grounded_{args.variant}_steer_score")

    print(f"# Intrinsics distribution — {args.variant}\n")
    print(f"_root: {root}_\n")

    results = {}
    for cell in args.cells:
        cd = root / cell
        if not cd.is_dir():
            # Try alt for track_c
            if args.variant == "track_c":
                for sub in root.iterdir():
                    if sub.is_dir() and (sub / cell).is_dir():
                        cd = sub / cell; break
        if not cd.is_dir():
            print(f"  ! Missing cell: {cell}"); continue
        hist, total = analyze_cell(cd, max_eps=args.max_eps)
        results[cell] = (fmt_distribution(hist, total), total)

    if not results:
        print("No cells loaded."); return

    cells = list(results.keys())
    print("## Per-intrinsic histogram (% of total steps)\n")
    for intrinsic in ["HP", "Food", "Drink", "Energy"]:
        print(f"### {intrinsic}\n")
        print(f"| level |", " | ".join(cells), "|")
        print("|---|" + "---|" * len(cells))
        for lvl in range(10):
            row = [f"{lvl}"]
            for c in cells:
                pct = results[c][0][intrinsic]["pct"][lvl]
                row.append(f"{pct:>5.1f}%")
            print("| " + " | ".join(row) + " |")
        # Below thresholds
        print(f"\n*Mean / fraction-below-threshold:*")
        for c in cells:
            d = results[c][0][intrinsic]
            print(f"  {c}: mean={d['mean']:.2f} | below≤2: {d['below_pct'][2]:.1f}% | below≤4: {d['below_pct'][4]:.1f}% | below≤6: {d['below_pct'][6]:.1f}%")
        print()

    print("## Total steps per cell\n")
    for c in cells:
        print(f"  {c}: {results[c][1]} steps")


if __name__ == "__main__":
    main()
