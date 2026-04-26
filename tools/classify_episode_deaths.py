"""Replay each saved episode deterministically and classify cause of death.

For each episode in the eval dir:
  - Reset env with the same key as eval_online (PRNGKey(seed) split per ep)
  - Step through with the saved actions
  - Capture state at the LAST step before done
  - Classify cause of death:
      * starvation:    food == 0 at death (and probably ate-empty for many steps)
      * dehydration:   drink == 0 at death
      * exhaustion:    energy == 0 at death
      * killed-by-melee: HP went to 0 with a melee mob within 1 tile
      * killed-by-ranged: HP went to 0 with a ranged mob in line-of-sight OR a recent projectile hit
      * lava:          standing on or adjacent to lava at death
      * timeout:       length == max steps (env-specific cap), HP > 0
      * unknown
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from collections import Counter

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import BlockType


def replay_one(env, env_params, rng_in, reset_key, actions):
    """Step env with actions, return state at step N-1 (just before terminal), state at step N,
    and a per-step intrinsics trace [(hp, food, drink, energy), ...]."""
    obs, state = env.reset(reset_key, env_params)
    rng = rng_in
    prev_state = state
    intrinsics_trace = []
    intrinsics_trace.append((float(state.player_health), float(state.player_food), float(state.player_drink), float(state.player_energy)))
    for i, a in enumerate(actions):
        prev_state = state
        rng, sk = jax.random.split(rng)
        obs, state, reward, done, info = env.step(sk, state, int(a), env_params)
        intrinsics_trace.append((float(state.player_health), float(state.player_food), float(state.player_drink), float(state.player_energy)))
        if done:
            break
    return prev_state, state, len(intrinsics_trace) - 1, intrinsics_trace


def classify(prev_state, state, actions, length):
    """Return (cause, details_dict)."""
    pos = np.array(state.player_position)
    hp = float(state.player_health)
    food = float(state.player_food)
    drink = float(state.player_drink)
    energy = float(state.player_energy)
    floor = int(state.player_level)
    p_prev_hp = float(prev_state.player_health)
    p_prev_food = float(prev_state.player_food)

    details = {
        "final_hp": hp, "prev_hp": p_prev_hp,
        "food": food, "drink": drink, "energy": energy,
        "floor": floor, "length": length,
        "pos": pos.tolist(), "hp_drop": p_prev_hp - hp,
    }

    # Detect mobs near player
    floor_idx = floor
    melee_pos = np.asarray(state.melee_mobs.position[floor_idx])
    melee_mask = np.asarray(state.melee_mobs.mask[floor_idx])
    ranged_pos = np.asarray(state.ranged_mobs.position[floor_idx])
    ranged_mask = np.asarray(state.ranged_mobs.mask[floor_idx])

    melee_active = melee_pos[melee_mask.astype(bool)]
    ranged_active = ranged_pos[ranged_mask.astype(bool)]

    melee_dists = np.array([float(np.abs(m - pos).sum()) for m in melee_active]) if len(melee_active) else np.array([])
    ranged_dists = np.array([float(np.abs(m - pos).sum()) for m in ranged_active]) if len(ranged_active) else np.array([])
    melee_within_1 = int((melee_dists <= 1).sum()) if len(melee_dists) else 0
    melee_within_3 = int((melee_dists <= 3).sum()) if len(melee_dists) else 0
    ranged_within_5 = int((ranged_dists <= 5).sum()) if len(ranged_dists) else 0

    # Detect lava under player or adjacent
    map_ = np.asarray(state.map[floor_idx])
    h, w = map_.shape
    py, px = pos
    around = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w:
                around.append(int(map_[ny, nx]))
    lava_block = int(BlockType.LAVA.value)
    lava_under = int(map_[py, px]) == lava_block if 0 <= py < h and 0 <= px < w else False
    lava_near = lava_block in around

    details["melee_within_1"] = melee_within_1
    details["melee_within_3"] = melee_within_3
    details["ranged_within_5"] = ranged_within_5
    details["lava_under"] = bool(lava_under)
    details["lava_near"] = bool(lava_near)

    # Was the player even at "death"?
    if hp <= 0:
        # Identify what killed them
        # 1) intrinsic-zero with intrinsic-decay: food/drink/energy was 0 a few steps before
        # 2) melee-adjacent attack
        # 3) ranged hit
        # 4) lava
        if lava_under:
            return "lava", details
        if melee_within_1 >= 1:
            return "killed_by_melee_adjacent", details
        if ranged_within_5 >= 1 and melee_within_3 == 0:
            return "killed_by_ranged_or_arrow", details
        if food <= 0 or drink <= 0 or energy <= 0:
            # starvation/dehydration/exhaustion (all manifest as HP-decay-from-zero-intrinsic)
            zeros = []
            if food <= 0: zeros.append("food")
            if drink <= 0: zeros.append("drink")
            if energy <= 0: zeros.append("energy")
            details["zero_intrinsics"] = zeros
            if "drink" in zeros:
                return "dehydration", details
            if "food" in zeros:
                return "starvation", details
            return "exhaustion", details
        if melee_within_3 >= 1:
            return "killed_by_melee_nearby", details
        return "killed_unknown_combat", details
    else:
        # length cap or other early termination
        return "alive_or_timeout", details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    eps = []
    for ep_dir in sorted(args.eval_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        sf = ep_dir / "summary.json"
        if not sf.exists(): continue
        with sf.open() as f:
            d = json.load(f)
        d["_ep_idx"] = int(ep_dir.name.split("_")[-1])
        eps.append(d)
    eps = eps[: args.max_episodes]
    print(f"Found {len(eps)} episodes to classify")

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(args.seed)

    rows = []
    for i, ep in enumerate(eps):
        rng, reset_key = jax.random.split(rng)
        actions = ep.get("actions", [])
        if not actions: continue
        prev_state, state, final_step, trace = replay_one(env, env_params, rng, reset_key, actions)
        cause, details = classify(prev_state, state, actions, ep["length"])
        details["episode"] = ep["_ep_idx"]
        details["return"] = ep["return"]
        details["n_achievements"] = len(ep.get("achievements", {}))
        details["actions_last10"] = [int(a) for a in actions[-10:]]
        details["cause"] = cause
        # Intrinsic-trajectory summaries
        import numpy as _np
        hp_arr = _np.array([t[0] for t in trace])
        food_arr = _np.array([t[1] for t in trace])
        drink_arr = _np.array([t[2] for t in trace])
        energy_arr = _np.array([t[3] for t in trace])
        L = len(trace)
        last_n = min(50, L)
        details["frac_last50_drink_le_2"] = float((drink_arr[-last_n:] <= 2).mean())
        details["frac_last50_drink_le_0"] = float((drink_arr[-last_n:] <= 0).mean())
        details["frac_last50_food_le_2"] = float((food_arr[-last_n:] <= 2).mean())
        details["frac_last50_food_le_0"] = float((food_arr[-last_n:] <= 0).mean())
        details["frac_last50_energy_le_2"] = float((energy_arr[-last_n:] <= 2).mean())
        details["frac_last50_hp_le_2"] = float((hp_arr[-last_n:] <= 2).mean())
        details["frac_episode_drink_le_0"] = float((drink_arr <= 0).mean())
        details["frac_episode_food_le_0"] = float((food_arr <= 0).mean())
        details["min_drink_episode"] = float(drink_arr.min())
        details["min_food_episode"] = float(food_arr.min())
        details["min_energy_episode"] = float(energy_arr.min())
        rows.append(details)
        if (i + 1) % 10 == 0:
            print(f"  classified {i+1}/{len(eps)}", flush=True)

    by_cause = Counter(r["cause"] for r in rows)
    print(f"\n=== Cause-of-death tally (n={len(rows)}) ===")
    for cause, n in sorted(by_cause.items(), key=lambda x: -x[1]):
        print(f"  {cause:30s} {n:>4d}  ({n/len(rows)*100:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"summary": dict(by_cause), "rows": rows}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
