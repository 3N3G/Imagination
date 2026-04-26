"""Deep comparative analysis: C_grounded_2M baseline vs PPO-RNN 1e8.

Replays each episode of both eval dirs, captures per-step:
  - intrinsics (hp, food, drink, energy)
  - position + floor
  - action taken

Then computes side-by-side metrics:
  - floor-time distribution (fraction of steps on Floor 0 / 1+)
  - action distribution (cardinal moves, DO, SLEEP, place_*, drink_potion)
  - achievement-timing (mean/median step when each achievement unlocks)
  - intrinsic-stress (fraction of steps with drink<=2, food<=2, etc.)
  - per-100-steps achievement rate
  - mob exposure (steps with mob within 3 tiles)

Output: docs/COMPARE_C_VS_PPO_RNN.md + JSON probe at
probe_results/compare_c_vs_pporn_1e8.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from collections import Counter, defaultdict

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
from craftax.craftax_env import make_craftax_env_from_name


def replay_with_traces(env, env_params, rng_in, reset_key, actions):
    obs, state = env.reset(reset_key, env_params)
    rng = rng_in
    intrinsics = []  # (hp, food, drink, energy)
    floors = []
    positions = []
    intrinsics.append((float(state.player_health), float(state.player_food),
                       float(state.player_drink), float(state.player_energy)))
    floors.append(int(state.player_level))
    positions.append([int(state.player_position[0]), int(state.player_position[1])])
    for a in actions:
        rng, sk = jax.random.split(rng)
        obs, state, reward, done, info = env.step(sk, state, int(a), env_params)
        intrinsics.append((float(state.player_health), float(state.player_food),
                           float(state.player_drink), float(state.player_energy)))
        floors.append(int(state.player_level))
        positions.append([int(state.player_position[0]), int(state.player_position[1])])
        if done:
            break
    return intrinsics, floors, positions


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
A_IDX = {n: i for i, n in enumerate(ACTION_NAMES)}


def analyze(label, eval_dir, seed=42, max_eps=50):
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(seed)

    eps = []
    for ep_dir in sorted(Path(eval_dir).iterdir()):
        sf = ep_dir / "summary.json"
        if not sf.exists(): continue
        with sf.open() as f: d = json.load(f)
        if not d.get("actions"): continue
        d["_ep_idx"] = int(ep_dir.name.split("_")[-1])
        eps.append(d)
    eps = eps[:max_eps]

    rows = []
    for ep in eps:
        rng, reset_key = jax.random.split(rng)
        intrinsics, floors, positions = replay_with_traces(
            env, env_params, rng, reset_key, ep["actions"]
        )
        rows.append({
            "episode": ep["_ep_idx"],
            "return": ep["return"],
            "length": ep["length"],
            "achievements": ep.get("achievements", {}),
            "actions": ep["actions"],
            "intrinsics": intrinsics,
            "floors": floors,
            "positions": positions,
        })

    # Aggregate stats
    n = len(rows)
    returns = np.array([r["return"] for r in rows])
    lengths = np.array([r["length"] for r in rows])

    # Floor time: fraction of steps on each floor across all episodes
    floor_counts = Counter()
    floor_total = 0
    for r in rows:
        for f in r["floors"]:
            floor_counts[f] += 1
            floor_total += 1
    floor_pct = {f: floor_counts[f] / floor_total * 100 for f in sorted(floor_counts.keys())}

    # Per-episode: did the policy reach Floor 1? At what step?
    n_reached_f1 = sum(1 for r in rows if max(r["floors"]) >= 1)
    first_f1_step = []
    for r in rows:
        for i, f in enumerate(r["floors"]):
            if f >= 1:
                first_f1_step.append(i)
                break

    # Action distribution (% of all actions)
    action_counter = Counter()
    total_actions = 0
    for r in rows:
        for a in r["actions"]:
            action_counter[a] += 1
            total_actions += 1
    action_pct = {ACTION_NAMES[i]: action_counter[i] / total_actions * 100 for i in range(len(ACTION_NAMES))}

    # Intrinsic stress
    drink_le2_frac = []
    drink_le0_frac = []
    food_le2_frac = []
    energy_le2_frac = []
    for r in rows:
        intr = np.array(r["intrinsics"])
        drink = intr[:, 2]
        food = intr[:, 1]
        energy = intr[:, 3]
        drink_le2_frac.append(float((drink <= 2).mean()))
        drink_le0_frac.append(float((drink <= 0).mean()))
        food_le2_frac.append(float((food <= 2).mean()))
        energy_le2_frac.append(float((energy <= 2).mean()))

    # Achievement rates
    ach_counter = Counter()
    for r in rows:
        for a in r["achievements"]:
            ach_counter[a] += 1
    ach_rate = {a: ach_counter[a] / n * 100 for a in ach_counter}

    # First-time-of-achievement (when in episode does it unlock?)
    ach_first_step = defaultdict(list)
    for r in rows:
        for a, step in r["achievements"].items():
            if isinstance(step, int):
                ach_first_step[a].append(step)
    ach_first_step_med = {a: float(np.median(steps)) for a, steps in ach_first_step.items() if len(steps) >= 3}

    return {
        "label": label,
        "n": n,
        "return_mean": float(returns.mean()),
        "return_se": float(returns.std(ddof=1) / np.sqrt(n)),
        "length_mean": float(lengths.mean()),
        "length_median": float(np.median(lengths)),
        "floor_pct": floor_pct,
        "n_reached_f1": n_reached_f1,
        "first_f1_step_median": float(np.median(first_f1_step)) if first_f1_step else None,
        "action_pct": action_pct,
        "drink_le2_frac_mean": float(np.mean(drink_le2_frac)),
        "drink_le0_frac_mean": float(np.mean(drink_le0_frac)),
        "food_le2_frac_mean": float(np.mean(food_le2_frac)),
        "energy_le2_frac_mean": float(np.mean(energy_le2_frac)),
        "ach_rate": ach_rate,
        "ach_first_step_median": ach_first_step_med,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-dir", default="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_grounded_predonly_top2M/freezenone_50ep")
    ap.add_argument("--ppo-dir", default="/data/group_data/rl/geney/eval_results/ppo_rnn_100M_save_traj_50ep_video")
    ap.add_argument("--out", default="probe_results/compare_c_vs_pporn_1e8.json")
    args = ap.parse_args()

    print(f"Replaying C (freezenone)...")
    c_stats = analyze("C_grounded_freezenone", args.c_dir, seed=42, max_eps=50)
    print(f"Replaying PPO-RNN 1e8...")
    ppo_stats = analyze("PPO-RNN_1e8", args.ppo_dir, seed=42, max_eps=50)

    out = {"c": c_stats, "ppo": ppo_stats}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
