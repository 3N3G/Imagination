"""Replay saved episodes deterministically and count inventory increments.

For each episode summary.json with `actions` and `episode` (1-indexed), reset
the env with the matching key (derived from --seed via the same jax key-split
sequence eval_online used), then step through with the saved action sequence
and count: each time inventory.X increases, we increment the count.

Records: per-episode counts of {stone, wood, coal, iron, diamond, drink, food}
where food is approximated by counting eat_cow + eat_plant achievement bumps
(can't disentangle by inventory).

Usage:
  PYTHONPATH=. python tools/replay_inventory_counts.py \
      --eval-dir <eval_results_dir> --seed 42 --max-episodes 50 \
      --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
from craftax.craftax_env import make_craftax_env_from_name


TRACKED = ["stone", "wood", "coal", "iron", "diamond", "ruby", "sapphire",
           "sapling", "torches", "arrows"]

# Events we track beyond inventory:
#   food_intake_events: # steps where player_food went UP (proxy for eat events:
#                       cow meat or plant fruit — both feed the food intrinsic)
#   drink_intake_events: # steps where player_drink went UP
#   monsters_killed_total: sum over monsters_killed[] delta per step


def load_eps(eval_dir: Path) -> List[dict]:
    eps = []
    for ep_dir in sorted(eval_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        sf = ep_dir / "summary.json"
        if not sf.exists():
            continue
        with sf.open() as f:
            d = json.load(f)
        d["_ep_idx"] = int(ep_dir.name.split("_")[-1])
        eps.append(d)
    return eps


def replay_episode(env, env_params, rng_in, reset_key, actions: List[int]):
    """Step env deterministically using the outer rng for step keys (matches eval_online).
    Returns (counts_dict, rng_out). counts_dict has:
      inventory keys (TRACKED), food_intake_events, drink_intake_events,
      monsters_killed_total."""
    obs, state = env.reset(reset_key, env_params)
    counts = {k: 0 for k in TRACKED}
    counts["food_intake_events"] = 0
    counts["drink_intake_events"] = 0
    counts["monsters_killed_total"] = 0
    rng = rng_in
    prev_inv = {k: int(getattr(state.inventory, k)) for k in TRACKED}
    prev_food = int(state.player_food)
    prev_drink = int(state.player_drink)
    prev_mk = int(state.monsters_killed.sum())
    for action in actions:
        rng, sk = jax.random.split(rng)
        obs, state, reward, done, info = env.step(sk, state, int(action), env_params)
        cur_inv = {k: int(getattr(state.inventory, k)) for k in TRACKED}
        for k in TRACKED:
            delta = cur_inv[k] - prev_inv[k]
            if delta > 0:
                counts[k] += int(delta)
        prev_inv = cur_inv
        cur_food = int(state.player_food)
        cur_drink = int(state.player_drink)
        cur_mk = int(state.monsters_killed.sum())
        if cur_food > prev_food:
            counts["food_intake_events"] += 1
        if cur_drink > prev_drink:
            counts["drink_intake_events"] += 1
        if cur_mk > prev_mk:
            counts["monsters_killed_total"] += cur_mk - prev_mk
        prev_food, prev_drink, prev_mk = cur_food, cur_drink, cur_mk
        if done:
            break
    return counts, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    eps = load_eps(args.eval_dir)[: args.max_episodes]
    if not eps:
        raise SystemExit(f"No episodes in {args.eval_dir}")

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params

    # Replicate eval_online's key sequence: rng = PRNGKey(seed); for each ep
    # rng, reset_key = split(rng); ... then within episode rng, step_key = split(rng).
    rng = jax.random.PRNGKey(args.seed)

    rows = []
    for ep_idx, ep in enumerate(eps):
        rng, reset_key = jax.random.split(rng)
        actions = ep.get("actions", [])
        if not actions:
            print(f"  ep {ep_idx+1}: no actions; skipping")
            continue
        counts, rng = replay_episode(env, env_params, rng, reset_key, actions)
        counts["return"] = float(ep["return"])
        counts["length"] = int(ep["length"])
        counts["episode"] = int(ep["_ep_idx"])
        rows.append(counts)
        if (ep_idx + 1) % 5 == 0:
            print(f"  done {ep_idx+1}/{len(eps)}: stone={counts['stone']} wood={counts['wood']}", flush=True)

    ALL_KEYS = TRACKED + ["food_intake_events", "drink_intake_events", "monsters_killed_total"]
    arr = {k: np.array([r[k] for r in rows]) for k in ALL_KEYS}
    summary = {
        "n": len(rows),
        "eval_dir": str(args.eval_dir),
        "per_resource_mean_count": {k: float(arr[k].mean()) for k in ALL_KEYS},
        "per_resource_std_count":  {k: float(arr[k].std(ddof=1)) for k in ALL_KEYS} if len(rows) > 1 else {k: 0.0 for k in ALL_KEYS},
        "per_resource_se_count":   {k: float(arr[k].std(ddof=1) / np.sqrt(len(rows))) for k in ALL_KEYS} if len(rows) > 1 else {k: 0.0 for k in ALL_KEYS},
    }

    print(f"\n=== {args.eval_dir.name} (n={len(rows)}) ===")
    for k in ALL_KEYS:
        m = summary["per_resource_mean_count"][k]
        se = summary["per_resource_se_count"][k]
        print(f"  {k:>22s}: {m:6.2f} ± {se:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
