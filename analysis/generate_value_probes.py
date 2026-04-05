import argparse
import bz2
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax.constants import Action
from craftax.craftax.renderer import render_craftax_text
from craftax.craftax_env import make_craftax_env_from_name

try:
    from labelling.obs_to_text import (
        MAP_OBS_SIZE,
        MAP_CHANNELS,
        NUM_BLOCK_TYPES,
        NUM_ITEM_TYPES,
        NUM_MOB_TYPES,
        OBS_DIM,
        obs_to_text,
    )
except ModuleNotFoundError:
    MAP_OBS_SIZE = 8217
    MAP_CHANNELS = 83
    NUM_BLOCK_TYPES = 37
    NUM_ITEM_TYPES = 5
    NUM_MOB_TYPES = 8
    OBS_DIM = (9, 11)
    obs_to_text = None


ACTION_DIM = len(Action)
DEFAULT_HIGH_STAT = 9.0
DEFAULT_LOW_STAT = 1.0

SPECIAL_START = 43
SPECIAL_FLOOR_INDEX = SPECIAL_START + 5
MOB_CHANNEL_START = NUM_BLOCK_TYPES + NUM_ITEM_TYPES
MOB_CHANNELS = 5 * NUM_MOB_TYPES
LIGHT_CHANNEL = MAP_CHANNELS - 1

OBS_LEVEL_PAIRS = {
    "floor1_to_floor2",
    "floor1_to_floor3",
    "floor1_zombie_adjacent",
    "floor1_skeleton_adjacent",
    "floor2_orc_adjacent",
    "floor2_witch_adjacent",
    "impossible_floor1_witch_adjacent",
}


def _replace_scalar(state, field_name: str, value: float):
    field_value = getattr(state, field_name)
    return state.replace(**{field_name: jnp.asarray(value, dtype=field_value.dtype)})


def _replace_inventory_scalar(state, field_name: str, value: float):
    inv = state.inventory
    inv_field_value = getattr(inv, field_name)
    new_inv = inv.replace(**{field_name: jnp.asarray(value, dtype=inv_field_value.dtype)})
    return state.replace(inventory=new_inv)


def apply_state_pair(state, pair_name: str) -> Tuple[object, object, dict]:
    if pair_name == "health":
        low_value, high_value = DEFAULT_LOW_STAT, DEFAULT_HIGH_STAT
        low = _replace_scalar(state, "player_health", low_value)
        high = _replace_scalar(state, "player_health", high_value)
        meta = {"field": "player_health", "low_value": low_value, "high_value": high_value}
    elif pair_name == "food":
        low_value, high_value = DEFAULT_LOW_STAT, DEFAULT_HIGH_STAT
        low = _replace_scalar(state, "player_food", low_value)
        high = _replace_scalar(state, "player_food", high_value)
        meta = {"field": "player_food", "low_value": low_value, "high_value": high_value}
    elif pair_name == "drink":
        low_value, high_value = DEFAULT_LOW_STAT, DEFAULT_HIGH_STAT
        low = _replace_scalar(state, "player_drink", low_value)
        high = _replace_scalar(state, "player_drink", high_value)
        meta = {"field": "player_drink", "low_value": low_value, "high_value": high_value}
    elif pair_name == "energy":
        low_value, high_value = DEFAULT_LOW_STAT, DEFAULT_HIGH_STAT
        low = _replace_scalar(state, "player_energy", low_value)
        high = _replace_scalar(state, "player_energy", high_value)
        meta = {"field": "player_energy", "low_value": low_value, "high_value": high_value}
    elif pair_name == "wood":
        low_value, high_value = 0, 10
        low = _replace_inventory_scalar(state, "wood", low_value)
        high = _replace_inventory_scalar(state, "wood", high_value)
        meta = {"field": "inventory.wood", "low_value": low_value, "high_value": high_value}
    elif pair_name == "stone":
        low_value, high_value = 0, 10
        low = _replace_inventory_scalar(state, "stone", low_value)
        high = _replace_inventory_scalar(state, "stone", high_value)
        meta = {"field": "inventory.stone", "low_value": low_value, "high_value": high_value}
    else:
        raise ValueError(f"Unsupported state-level pair type: {pair_name}")

    return low, high, meta


def _set_floor(obs_flat: np.ndarray, floor_value: int) -> np.ndarray:
    obs = np.asarray(obs_flat, dtype=np.float32).copy()
    obs[MAP_OBS_SIZE + SPECIAL_FLOOR_INDEX] = float(floor_value) / 10.0
    return obs


def _inject_mob(
    obs_flat: np.ndarray,
    mob_class: int,
    mob_type: int,
    row_offset: int = 1,
    col_offset: int = 0,
    floor_value: Optional[int] = None,
) -> np.ndarray:
    obs = np.asarray(obs_flat, dtype=np.float32).copy()
    if floor_value is not None:
        obs[MAP_OBS_SIZE + SPECIAL_FLOOR_INDEX] = float(floor_value) / 10.0

    map_view = obs[:MAP_OBS_SIZE].reshape(OBS_DIM[0], OBS_DIM[1], MAP_CHANNELS)
    row = int(np.clip(OBS_DIM[0] // 2 + row_offset, 0, OBS_DIM[0] - 1))
    col = int(np.clip(OBS_DIM[1] // 2 + col_offset, 0, OBS_DIM[1] - 1))

    mob_idx = mob_class * NUM_MOB_TYPES + mob_type
    map_view[row, col, MOB_CHANNEL_START : MOB_CHANNEL_START + MOB_CHANNELS] = 0.0
    map_view[row, col, MOB_CHANNEL_START + mob_idx] = 1.0
    map_view[row, col, LIGHT_CHANNEL] = 1.0
    return obs


def apply_obs_pair(base_obs: np.ndarray, pair_name: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    if pair_name == "floor1_to_floor2":
        low = _set_floor(base_obs, 1)
        high = _set_floor(base_obs, 2)
        meta = {"field": "special.floor", "low_value": 1, "high_value": 2}
    elif pair_name == "floor1_to_floor3":
        low = _set_floor(base_obs, 1)
        high = _set_floor(base_obs, 3)
        meta = {"field": "special.floor", "low_value": 1, "high_value": 3}
    elif pair_name == "floor1_zombie_adjacent":
        low = _set_floor(base_obs, 1)
        high = _inject_mob(low, mob_class=0, mob_type=0, row_offset=1, col_offset=0, floor_value=1)
        meta = {"field": "mob.class0.type0", "low_value": "absent", "high_value": "adjacent"}
    elif pair_name == "floor1_skeleton_adjacent":
        low = _set_floor(base_obs, 1)
        high = _inject_mob(low, mob_class=2, mob_type=0, row_offset=1, col_offset=0, floor_value=1)
        meta = {"field": "mob.class2.type0", "low_value": "absent", "high_value": "adjacent"}
    elif pair_name == "floor2_orc_adjacent":
        low = _set_floor(base_obs, 2)
        high = _inject_mob(low, mob_class=0, mob_type=2, row_offset=1, col_offset=0, floor_value=2)
        meta = {"field": "mob.class0.type2", "low_value": "absent", "high_value": "adjacent"}
    elif pair_name == "floor2_witch_adjacent":
        low = _set_floor(base_obs, 2)
        high = _inject_mob(low, mob_class=2, mob_type=6, row_offset=1, col_offset=0, floor_value=2)
        meta = {"field": "mob.class2.type6", "low_value": "absent", "high_value": "adjacent"}
    elif pair_name == "impossible_floor1_witch_adjacent":
        low = _set_floor(base_obs, 1)
        high = _inject_mob(low, mob_class=2, mob_type=6, row_offset=1, col_offset=0, floor_value=1)
        meta = {
            "field": "mob.class2.type6_on_floor1",
            "low_value": "absent",
            "high_value": "adjacent_impossible",
        }
    else:
        raise ValueError(f"Unsupported obs-level pair type: {pair_name}")

    return low.astype(np.float32, copy=False), high.astype(np.float32, copy=False), meta


def collect_states(env, env_params, num_states: int, seed: int, max_steps: int) -> List[object]:
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    _, state = env.reset(reset_rng, env_params)

    states = []
    for _ in range(max_steps):
        if len(states) >= num_states:
            break
        states.append(state)
        rng, step_rng = jax.random.split(rng)
        action = int(np.random.randint(0, ACTION_DIM))
        _, state, _, _, _ = env.step(step_rng, state, action, env_params)
    return states


def save_compressed_pickle(path: Path, obj):
    with bz2.BZ2File(path, "wb") as f:
        pickle.dump(obj, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paired Craftax value probes (state-level and obs-level)."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_states", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[
            "health",
            "food",
            "drink",
            "energy",
            "wood",
            "stone",
            "floor1_zombie_adjacent",
            "floor1_skeleton_adjacent",
            "floor2_witch_adjacent",
        ],
        choices=[
            "health",
            "food",
            "drink",
            "energy",
            "wood",
            "stone",
            "floor1_to_floor2",
            "floor1_to_floor3",
            "floor1_zombie_adjacent",
            "floor1_skeleton_adjacent",
            "floor2_orc_adjacent",
            "floor2_witch_adjacent",
            "impossible_floor1_witch_adjacent",
        ],
    )
    parser.add_argument(
        "--save_text",
        action="store_true",
        help="Also write raw text observations for each low/high probe.",
    )
    return parser.parse_args()


def _obs_to_text_fallback(obs: np.ndarray) -> str:
    if obs_to_text is not None:
        return obs_to_text(obs)
    return "obs_to_text unavailable"


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    vectors_dir = output_dir / "vectors"
    states_dir = output_dir / "states"
    text_dir = output_dir / "text"
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)
    if args.save_text:
        text_dir.mkdir(parents=True, exist_ok=True)

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    base_states = collect_states(
        env=env,
        env_params=env_params,
        num_states=args.num_states,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print(f"Collected {len(base_states)} base states.")

    metadata_path = output_dir / "pairs.jsonl"
    pair_count = 0
    state_pair_count = 0
    obs_pair_count = 0
    with metadata_path.open("w") as f:
        for base_idx, base_state in enumerate(base_states):
            base_obs = np.asarray(env.get_obs(base_state), dtype=np.float32)
            for pair_name in args.pairs:
                pair_id = f"pair_{pair_count:06d}"

                if pair_name in OBS_LEVEL_PAIRS:
                    low_state = None
                    high_state = None
                    low_obs, high_obs, pair_meta = apply_obs_pair(base_obs, pair_name)
                    pair_family = "obs"
                    obs_pair_count += 1
                else:
                    low_state, high_state, pair_meta = apply_state_pair(base_state, pair_name)
                    low_obs = np.asarray(env.get_obs(low_state), dtype=np.float32)
                    high_obs = np.asarray(env.get_obs(high_state), dtype=np.float32)
                    pair_family = "state"
                    state_pair_count += 1

                np.savez_compressed(
                    vectors_dir / f"{pair_id}.npz",
                    obs_low=low_obs,
                    obs_high=high_obs,
                    base_index=np.asarray(base_idx, dtype=np.int32),
                    pair_family=np.asarray(pair_family),
                )

                state_low_path = None
                state_high_path = None
                if low_state is not None and high_state is not None:
                    state_low_path = states_dir / f"{pair_id}_low.pbz2"
                    state_high_path = states_dir / f"{pair_id}_high.pbz2"
                    save_compressed_pickle(state_low_path, low_state)
                    save_compressed_pickle(state_high_path, high_state)

                if args.save_text:
                    if low_state is not None and high_state is not None:
                        low_text = render_craftax_text(low_state)
                        high_text = render_craftax_text(high_state)
                    else:
                        low_text = _obs_to_text_fallback(low_obs)
                        high_text = _obs_to_text_fallback(high_obs)
                    (text_dir / f"{pair_id}_low.txt").write_text(low_text)
                    (text_dir / f"{pair_id}_high.txt").write_text(high_text)

                row = {
                    "pair_id": pair_id,
                    "base_index": base_idx,
                    "pair_name": pair_name,
                    "pair_family": pair_family,
                    "expected_value_relation": "high > low" if pair_name in {"health", "food", "drink", "energy", "wood", "stone"} else "diagnostic_or_high_risk",
                    "low_state_path": str(state_low_path) if state_low_path is not None else None,
                    "high_state_path": str(state_high_path) if state_high_path is not None else None,
                    **pair_meta,
                }
                f.write(json.dumps(row) + "\n")
                pair_count += 1

    summary = {
        "output_dir": str(output_dir),
        "num_base_states": len(base_states),
        "pairs_per_state": len(args.pairs),
        "total_pairs": pair_count,
        "state_pairs": state_pair_count,
        "obs_pairs": obs_pair_count,
        "metadata_file": str(metadata_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
