import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch

from craftax.craftax.constants import Action
from craftax.craftax_env import make_craftax_env_from_name

try:
    from labelling.obs_to_text import (
        MAP_OBS_SIZE,
        MAP_CHANNELS,
        NUM_BLOCK_TYPES,
        NUM_ITEM_TYPES,
        NUM_MOB_TYPES,
        OBS_DIM,
    )
except ModuleNotFoundError:
    # Keep standalone execution resilient if repo root is not on PYTHONPATH.
    MAP_OBS_SIZE = 8217
    MAP_CHANNELS = 83
    NUM_BLOCK_TYPES = 37
    NUM_ITEM_TYPES = 5
    NUM_MOB_TYPES = 8
    OBS_DIM = (9, 11)

from models.actor_critic_aug import ActorCriticAug


ACTION_DIM = len(Action)
LAYER_WIDTH = 512
HIDDEN_STATE_DIM = 2560
DEFAULT_HIGH_STAT = 9.0
DEFAULT_LOW_STAT = 1.0

# Inventory special values start index in inventory block:
# 16 inv + 6 potions + 9 intrinsics + 4 dir + 4 armour + 4 ench = 43
SPECIAL_START = 43
SPECIAL_FLOOR_INDEX = SPECIAL_START + 5

MOB_CHANNEL_START = NUM_BLOCK_TYPES + NUM_ITEM_TYPES
MOB_CHANNELS = 5 * NUM_MOB_TYPES
LIGHT_CHANNEL = MAP_CHANNELS - 1


@dataclass
class PairSpec:
    name: str
    # +1 means high should be higher value, -1 means high should be lower,
    # 0 means diagnostic-only (no directional expectation).
    expected_sign: int


PAIR_EXPECTED_SIGN: Dict[str, int] = {
    "health": 1,
    "food": 1,
    "drink": 1,
    "energy": 1,
    "wood": 1,
    "stone": 1,
    # Floor transitions are harder/deeper, so we expect lower values.
    "floor1_to_floor2": -1,
    "floor1_to_floor3": -1,
    # Mob-threat pairs: high = threat present near player.
    "floor1_zombie_adjacent": -1,
    "floor1_skeleton_adjacent": -1,
    "floor2_orc_adjacent": -1,
    "floor2_witch_adjacent": -1,
    # Keep impossible-state diagnostic available without directional scoring.
    "impossible_floor1_witch_adjacent": 0,
}

OBS_LEVEL_PAIRS = {
    "floor1_to_floor2",
    "floor1_to_floor3",
    "floor1_zombie_adjacent",
    "floor1_skeleton_adjacent",
    "floor2_orc_adjacent",
    "floor2_witch_adjacent",
    "impossible_floor1_witch_adjacent",
}


def load_hidden_stats(checkpoint_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, Path]]:
    stats_path = checkpoint_path.parent / "hidden_state_stats.npz"
    if not stats_path.exists():
        return None
    with np.load(stats_path) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std, stats_path


def normalize_hidden(hidden_vec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if hidden_vec.shape[-1] != mean.shape[-1]:
        raise ValueError(
            f"Hidden dimension mismatch: vec={hidden_vec.shape[-1]} vs stats={mean.shape[-1]}"
        )
    return ((hidden_vec - mean) / std).astype(np.float32, copy=False)


def _replace_scalar(state, field_name: str, value: float):
    field_value = getattr(state, field_name)
    return state.replace(**{field_name: jnp.asarray(value, dtype=field_value.dtype)})


def _replace_inventory_scalar(state, field_name: str, value: float):
    inv = state.inventory
    inv_field_value = getattr(inv, field_name)
    new_inv = inv.replace(**{field_name: jnp.asarray(value, dtype=inv_field_value.dtype)})
    return state.replace(inventory=new_inv)


def apply_state_pair(state, pair_name: str) -> Tuple[object, object]:
    if pair_name == "health":
        low = _replace_scalar(state, "player_health", DEFAULT_LOW_STAT)
        high = _replace_scalar(state, "player_health", DEFAULT_HIGH_STAT)
    elif pair_name == "food":
        low = _replace_scalar(state, "player_food", DEFAULT_LOW_STAT)
        high = _replace_scalar(state, "player_food", DEFAULT_HIGH_STAT)
    elif pair_name == "drink":
        low = _replace_scalar(state, "player_drink", DEFAULT_LOW_STAT)
        high = _replace_scalar(state, "player_drink", DEFAULT_HIGH_STAT)
    elif pair_name == "energy":
        low = _replace_scalar(state, "player_energy", DEFAULT_LOW_STAT)
        high = _replace_scalar(state, "player_energy", DEFAULT_HIGH_STAT)
    elif pair_name == "wood":
        low = _replace_inventory_scalar(state, "wood", 0)
        high = _replace_inventory_scalar(state, "wood", 10)
    elif pair_name == "stone":
        low = _replace_inventory_scalar(state, "stone", 0)
        high = _replace_inventory_scalar(state, "stone", 10)
    else:
        raise ValueError(f"Unsupported state-level pair type: {pair_name}")
    return low, high


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
    if mob_idx < 0 or mob_idx >= MOB_CHANNELS:
        raise ValueError(f"Invalid mob index: class={mob_class} type={mob_type}")

    map_view[row, col, MOB_CHANNEL_START : MOB_CHANNEL_START + MOB_CHANNELS] = 0.0
    map_view[row, col, MOB_CHANNEL_START + mob_idx] = 1.0
    # Ensure the target tile is visible.
    map_view[row, col, LIGHT_CHANNEL] = 1.0
    return obs


def apply_obs_pair(base_obs: np.ndarray, pair_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if pair_name == "floor1_to_floor2":
        low = _set_floor(base_obs, 1)
        high = _set_floor(base_obs, 2)
    elif pair_name == "floor1_to_floor3":
        low = _set_floor(base_obs, 1)
        high = _set_floor(base_obs, 3)
    elif pair_name == "floor1_zombie_adjacent":
        low = _set_floor(base_obs, 1)
        high = _inject_mob(low, mob_class=0, mob_type=0, row_offset=1, col_offset=0, floor_value=1)
    elif pair_name == "floor1_skeleton_adjacent":
        # Craftax class-2/type-0 is floor-1 ranged threat; we treat this as skeleton-like.
        low = _set_floor(base_obs, 1)
        high = _inject_mob(low, mob_class=2, mob_type=0, row_offset=1, col_offset=0, floor_value=1)
    elif pair_name == "floor2_orc_adjacent":
        low = _set_floor(base_obs, 2)
        high = _inject_mob(low, mob_class=0, mob_type=2, row_offset=1, col_offset=0, floor_value=2)
    elif pair_name == "floor2_witch_adjacent":
        # Witch-like proxy: fire mage channel (class-2/type-6) on floor 2.
        low = _set_floor(base_obs, 2)
        high = _inject_mob(low, mob_class=2, mob_type=6, row_offset=1, col_offset=0, floor_value=2)
    elif pair_name == "impossible_floor1_witch_adjacent":
        low = _set_floor(base_obs, 1)
        high = _inject_mob(low, mob_class=2, mob_type=6, row_offset=1, col_offset=0, floor_value=1)
    else:
        raise ValueError(f"Unsupported obs-level pair type: {pair_name}")

    return low.astype(np.float32, copy=False), high.astype(np.float32, copy=False)


def collect_states(env, env_params, num_states: int, seed: int, max_steps: int) -> List[object]:
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    _, state = env.reset(reset_rng, env_params)

    rollout_states: List[object] = []
    for _ in range(max_steps):
        rollout_states.append(state)
        rng, step_rng = jax.random.split(rng)
        action = int(np.random.randint(0, ACTION_DIM))
        _, state, _, _, _ = env.step(step_rng, state, action, env_params)

    if len(rollout_states) <= num_states:
        return rollout_states

    sample_idx = np.linspace(0, len(rollout_states) - 1, num_states, dtype=np.int32)
    return [rollout_states[int(i)] for i in sample_idx]


def build_hidden_input(mode: str, hidden_dim: int, seed: int) -> np.ndarray:
    if mode == "zero":
        return np.zeros((hidden_dim,), dtype=np.float32)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((hidden_dim,), dtype=np.float32)
    raise ValueError(f"Unsupported hidden_input_mode: {mode}")


def build_pair_obs_cache(
    states: List[object],
    pair_specs: List[PairSpec],
    env,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    base_obs = [np.asarray(env.get_obs(s), dtype=np.float32).reshape(-1) for s in states]
    cache: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for spec in pair_specs:
        pair_rows: List[Tuple[np.ndarray, np.ndarray]] = []
        if spec.name in OBS_LEVEL_PAIRS:
            for obs in base_obs:
                pair_rows.append(apply_obs_pair(obs, spec.name))
        else:
            for state in states:
                low_state, high_state = apply_state_pair(state, spec.name)
                low_obs = np.asarray(env.get_obs(low_state), dtype=np.float32).reshape(-1)
                high_obs = np.asarray(env.get_obs(high_state), dtype=np.float32).reshape(-1)
                pair_rows.append((low_obs, high_obs))
        cache[spec.name] = pair_rows
    return cache


def build_llm_hidden_lookup(
    pair_obs_cache: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    llm_model_id: str,
    llm_batch_size: int,
) -> Tuple[Dict[Tuple[str, int, str], np.ndarray], Dict[str, float]]:
    from labelling.obs_to_text import obs_to_text
    from llm.extractor import LLMHiddenStateExtractor
    from llm.prompts import filter_text_obs

    keys: List[Tuple[str, int, str]] = []
    texts: List[str] = []

    for pair_name, obs_pairs in pair_obs_cache.items():
        for state_idx, (low_obs, high_obs) in enumerate(obs_pairs):
            low_text = filter_text_obs(obs_to_text(low_obs))
            high_text = filter_text_obs(obs_to_text(high_obs))

            keys.append((pair_name, state_idx, "low"))
            texts.append(low_text)
            keys.append((pair_name, state_idx, "high"))
            texts.append(high_text)

    print(
        f"Computing real LLM hidden states for {len(texts)} prompts "
        f"(model={llm_model_id}, batch_size={llm_batch_size})"
    )
    extractor = LLMHiddenStateExtractor(model_id=llm_model_id, tokens_generated=1)
    hidden_states, metrics = extractor.extract_hidden_states_no_cot(texts, batch_size=llm_batch_size)

    lookup: Dict[Tuple[str, int, str], np.ndarray] = {}
    for i, key in enumerate(keys):
        lookup[key] = hidden_states[i].astype(np.float32, copy=False)
    return lookup, metrics


def value_for_obs(model, obs: np.ndarray, hidden_vec: np.ndarray, device: torch.device) -> float:
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
    hidden_t = torch.from_numpy(np.asarray(hidden_vec, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        _, value = model(obs_t, hidden_t)
    return float(value.item())


def infer_fusion_mode(state_dict: Dict[str, torch.Tensor], hidden_dim: int) -> str:
    actor_in = int(state_dict["actor_fc1.weight"].shape[1])
    layer_width = int(state_dict["encoder_fc1.weight"].shape[0])
    if actor_in == layer_width + hidden_dim:
        return "concat_raw"
    if actor_in == 2 * layer_width:
        return "gated_proj"
    if actor_in == layer_width:
        return "residual_gated"
    # Backward-compatible fallback if shapes are unexpected.
    if (
        "hidden_gate_logit" in state_dict
        or "hidden_proj.weight" in state_dict
        or "hidden_ln.weight" in state_dict
    ):
        return "gated_proj"
    return "concat_raw"


def load_model(checkpoint_path: Path, obs_dim: int, device: torch.device) -> ActorCriticAug:
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "actor_hidden_fc1.weight" in state_dict:
        hidden_dim = int(state_dict["actor_hidden_fc1.weight"].shape[-1])
    elif "hidden_proj.weight" in state_dict:
        hidden_dim = int(state_dict["hidden_proj.weight"].shape[-1])
    else:
        hidden_dim = HIDDEN_STATE_DIM
    model = ActorCriticAug(
        obs_dim=obs_dim,
        action_dim=ACTION_DIM,
        layer_width=LAYER_WIDTH,
        hidden_state_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def analyze_checkpoint(
    checkpoint_path: Path,
    pair_obs_cache: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    pair_specs: List[PairSpec],
    hidden_input_mode: str,
    device: torch.device,
    shared_hidden_vec: Optional[np.ndarray] = None,
    llm_hidden_lookup: Optional[Dict[Tuple[str, int, str], np.ndarray]] = None,
    normalize_llm_hidden: bool = True,
) -> Dict[str, object]:
    first_pair = pair_obs_cache[pair_specs[0].name][0][0]
    obs_dim = int(np.asarray(first_pair).reshape(-1).shape[0])
    model = load_model(checkpoint_path, obs_dim, device)

    hidden_stats_info = None
    if hidden_input_mode == "llm_no_cot" and normalize_llm_hidden:
        loaded = load_hidden_stats(checkpoint_path)
        if loaded is not None:
            hidden_mean, hidden_std, stats_path = loaded
            hidden_stats_info = {
                "stats_path": str(stats_path),
                "normalized": True,
                "hidden_dim": int(hidden_mean.shape[0]),
            }
        else:
            hidden_mean = None
            hidden_std = None
            hidden_stats_info = {
                "stats_path": None,
                "normalized": False,
                "reason": "hidden_state_stats.npz not found",
            }
    else:
        hidden_mean = None
        hidden_std = None

    results = {}
    for spec in pair_specs:
        deltas = []
        obs_pairs = pair_obs_cache[spec.name]
        for pair_idx, (low_obs, high_obs) in enumerate(obs_pairs):
            if hidden_input_mode == "llm_no_cot":
                if llm_hidden_lookup is None:
                    raise ValueError("llm_hidden_lookup is required for llm_no_cot mode")
                hidden_low = llm_hidden_lookup[(spec.name, pair_idx, "low")]
                hidden_high = llm_hidden_lookup[(spec.name, pair_idx, "high")]
                if hidden_mean is not None and hidden_std is not None:
                    hidden_low = normalize_hidden(hidden_low, hidden_mean, hidden_std)
                    hidden_high = normalize_hidden(hidden_high, hidden_mean, hidden_std)
            else:
                if shared_hidden_vec is None:
                    raise ValueError("shared_hidden_vec is required for non-LLM modes")
                hidden_low = shared_hidden_vec
                hidden_high = shared_hidden_vec

            v_low = value_for_obs(model, low_obs, hidden_low, device)
            v_high = value_for_obs(model, high_obs, hidden_high, device)
            deltas.append(v_high - v_low)

        deltas_np = np.asarray(deltas, dtype=np.float32)
        pair_metrics = {
            "mean_delta_high_minus_low": float(np.mean(deltas_np)),
            "median_delta_high_minus_low": float(np.median(deltas_np)),
            "p10_delta": float(np.percentile(deltas_np, 10)),
            "p90_delta": float(np.percentile(deltas_np, 90)),
            "fraction_positive": float(np.mean(deltas_np > 0.0)),
            "fraction_negative": float(np.mean(deltas_np < 0.0)),
            "num_pairs": int(len(deltas_np)),
            "expected_sign": int(spec.expected_sign),
        }
        if spec.expected_sign != 0:
            pair_metrics["sign_correct_fraction"] = float(
                np.mean((deltas_np * spec.expected_sign) > 0.0)
            )
        else:
            pair_metrics["sign_correct_fraction"] = None
        results[spec.name] = pair_metrics

    return {
        "checkpoint": str(checkpoint_path),
        "hidden_input_mode": hidden_input_mode,
        "hidden_preprocessing": hidden_stats_info,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate value-function monotonicity on controlled Craftax probe pairs."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="One or more .pth checkpoints to evaluate.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
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
        choices=sorted(PAIR_EXPECTED_SIGN.keys()),
    )
    parser.add_argument("--num_states", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hidden_input_mode",
        type=str,
        default="zero",
        choices=["zero", "random", "llm_no_cot"],
        help="Hidden vector used during value evaluation.",
    )
    parser.add_argument(
        "--llm_model_id",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="Model ID used when --hidden_input_mode=llm_no_cot.",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=1,
        help="Batch size for LLM hidden extraction in llm_no_cot mode.",
    )
    parser.add_argument(
        "--normalize_llm_hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When using llm_no_cot, normalize hidden states with checkpoint-local "
            "hidden_state_stats.npz (recommended)."
        ),
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    states = collect_states(
        env=env,
        env_params=env_params,
        num_states=args.num_states,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    if len(states) < args.num_states:
        print(f"Warning: collected {len(states)} states (requested {args.num_states}).")
    else:
        print(f"Collected {len(states)} base states.")

    pair_specs = [PairSpec(name=p, expected_sign=PAIR_EXPECTED_SIGN[p]) for p in args.pairs]
    pair_obs_cache = build_pair_obs_cache(states=states, pair_specs=pair_specs, env=env)

    shared_hidden_vec: Optional[np.ndarray] = None
    llm_hidden_lookup: Optional[Dict[Tuple[str, int, str], np.ndarray]] = None
    llm_metrics: Optional[Dict[str, float]] = None

    if args.hidden_input_mode in {"zero", "random"}:
        shared_hidden_vec = build_hidden_input(args.hidden_input_mode, HIDDEN_STATE_DIM, args.seed)
    else:
        llm_hidden_lookup, llm_metrics = build_llm_hidden_lookup(
            pair_obs_cache=pair_obs_cache,
            llm_model_id=args.llm_model_id,
            llm_batch_size=args.llm_batch_size,
        )

    all_results = []
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        result = analyze_checkpoint(
            checkpoint_path=ckpt_path,
            pair_obs_cache=pair_obs_cache,
            pair_specs=pair_specs,
            hidden_input_mode=args.hidden_input_mode,
            device=device,
            shared_hidden_vec=shared_hidden_vec,
            llm_hidden_lookup=llm_hidden_lookup,
            normalize_llm_hidden=args.normalize_llm_hidden,
        )
        all_results.append(result)

    output = {
        "num_states": len(states),
        "pairs": args.pairs,
        "seed": args.seed,
        "device": str(device),
        "hidden_input_mode": args.hidden_input_mode,
        "checkpoints": all_results,
    }
    if llm_metrics is not None:
        output["llm_metrics"] = llm_metrics

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Wrote analysis to {out_path}")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
