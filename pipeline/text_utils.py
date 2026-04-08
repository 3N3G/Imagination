"""
Observation → text conversion utilities for the pipeline.

Wraps obs_to_text() and filter_text_obs() from Craftax_Baselines, and provides
a lightweight compact-state builder (adapted from future_imagination_eval.py)
that works directly on filtered text without the full TrajectoryStep machinery.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pipeline.config import ACTION_NAMES, ensure_imports

# Lazy imports from Craftax_Baselines (loaded on first call)
_obs_to_text_fn = None
_filter_text_obs_fn = None
_MAP_INTERESTING_PREFIX = None


def _ensure_craftax_imports():
    global _obs_to_text_fn, _filter_text_obs_fn, _MAP_INTERESTING_PREFIX
    if _obs_to_text_fn is not None:
        return
    ensure_imports()
    from labelling.obs_to_text import obs_to_text
    from llm.prompts import MAP_INTERESTING_PREFIX, filter_text_obs

    _obs_to_text_fn = obs_to_text
    _filter_text_obs_fn = filter_text_obs
    _MAP_INTERESTING_PREFIX = MAP_INTERESTING_PREFIX


def obs_to_text(obs):
    """Convert a single (8268,) observation to raw text."""
    _ensure_craftax_imports()
    return _obs_to_text_fn(obs)


def filter_text_obs(text: str) -> str:
    """Filter background tiles from text observation."""
    _ensure_craftax_imports()
    return _filter_text_obs_fn(text)


# ---------------------------------------------------------------------------
# Compact state builder (adapted from future_imagination_eval._build_compact_state)
# ---------------------------------------------------------------------------

_COORD_PREFIX_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:")


def _fmt_num(value) -> str:
    if value is None:
        return "NA"
    v = float(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _parse_features(filtered_text: str) -> Dict:
    """Extract structured features from filtered text obs."""
    features = {
        "health": None, "food": None, "drink": None, "energy": None,
        "mana": None, "xp": None, "floor": None, "direction": None,
        "ladder_open": None, "map_line": "", "inventory": {},
        "equipment_lines": [],
    }

    _ensure_craftax_imports()
    prefix = _MAP_INTERESTING_PREFIX

    stat_float = {"Health", "Food", "Drink", "Energy", "Mana", "XP"}
    inventory_keys = {
        "Wood", "Stone", "Coal", "Iron", "Diamond", "Sapphire", "Ruby",
        "Sapling", "Torch", "Arrow", "Book",
        "Red potion", "Green potion", "Blue potion",
        "Pink potion", "Cyan potion", "Yellow potion",
    }

    # All recognised keys that live on the "Inventory:" line
    all_kv_keys = stat_float | inventory_keys | {
        "Floor", "Direction", "Ladder Open", "Light",
        "Is Sleeping", "Is Resting", "Learned Fireball", "Learned Iceball",
        "Is Boss Vulnerable",
    }

    def _ingest_kv(key: str, val: str):
        """Route a single key-value pair into *features*."""
        key, val = key.strip(), val.strip()
        if key in stat_float:
            try:
                features[key.lower()] = float(val)
            except ValueError:
                pass
        elif key == "Floor":
            try:
                features["floor"] = int(float(val))
            except ValueError:
                pass
        elif key == "Direction":
            features["direction"] = val
        elif key == "Ladder Open":
            features["ladder_open"] = val.strip().lower() in {"true", "1", "yes"}
        elif key in inventory_keys:
            try:
                features["inventory"][key] = float(val)
            except ValueError:
                pass

    for line in filtered_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith(prefix):
            features["map_line"] = stripped
            continue

        # Handle the flat "Inventory: Key:Val, Key:Val, ..." line from
        # filter_text_obs as well as individual "Key: Val" lines.
        if stripped.startswith("Inventory:"):
            # Everything after the leading "Inventory:" is comma-separated
            payload = stripped.split(":", 1)[1]
            for part in payload.split(","):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    _ingest_kv(k, v)
            continue

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            _ingest_kv(key, val)
            if key.strip() not in all_kv_keys:
                if any(kw in stripped for kw in ("with", "Helmet", "Armour", "Sword")):
                    features["equipment_lines"].append(stripped)

    return features


def _compact_inventory(inv: Dict[str, float], max_items: int = 12) -> str:
    non_zero = [(k, v) for k, v in sorted(inv.items()) if abs(v) > 1e-9]
    if not non_zero:
        return "Inventory+: (none)"
    parts = [f"{k}={_fmt_num(v)}" for k, v in non_zero[:max_items]]
    suffix = ", ..." if len(non_zero) > max_items else ""
    return "Inventory+: " + ", ".join(parts) + suffix


def build_compact_state(
    filtered_text: str,
    *,
    action_id: Optional[int] = None,
    reward: Optional[float] = None,
    done: Optional[bool] = None,
) -> str:
    """Build a compact state string from filtered text obs.

    This is the same format as future_imagination_eval._build_compact_state
    but works directly on filtered text instead of requiring TrajectoryStep.
    """
    f = _parse_features(filtered_text)
    lines = []

    lines.append(f["map_line"] or "Map (interesting tiles only): <unavailable>")
    lines.append(
        "Stats: " + ", ".join([
            f"Health={_fmt_num(f['health'])}",
            f"Food={_fmt_num(f['food'])}",
            f"Drink={_fmt_num(f['drink'])}",
            f"Energy={_fmt_num(f['energy'])}",
            f"Mana={_fmt_num(f['mana'])}",
            f"XP={_fmt_num(f['xp'])}",
        ])
    )
    lines.append(
        f"Direction={f['direction'] or 'NA'}, "
        f"Floor={f['floor'] if f['floor'] is not None else 'NA'}, "
        f"LadderOpen={f['ladder_open'] if f['ladder_open'] is not None else 'NA'}"
    )
    lines.append(_compact_inventory(f["inventory"]))
    if f["equipment_lines"]:
        lines.append("Equipment: " + "; ".join(f["equipment_lines"][:4]))

    action_name = ACTION_NAMES[action_id] if action_id is not None and 0 <= action_id < len(ACTION_NAMES) else "NA"
    lines.append(
        f"Action@t={action_name}, "
        f"Reward@t={_fmt_num(reward)}, "
        f"Done@t={done if done is not None else 'NA'}"
    )
    return "\n".join(lines)


def build_future_state_block(
    obs_sequence,
    action_sequence,
    reward_sequence,
    done_sequence,
    base_t_offset: int = 0,
) -> str:
    """Build the future state block for the oracle prompt.

    Args:
        obs_sequence: list/array of (8268,) observations for t+0 through t+N
        action_sequence: corresponding actions
        reward_sequence: corresponding rewards
        done_sequence: corresponding done flags
        base_t_offset: offset for labeling (usually 0)

    Returns:
        Formatted future state block string
    """
    blocks = []
    for i in range(len(obs_sequence)):
        raw_text = obs_to_text(obs_sequence[i])
        filtered = filter_text_obs(raw_text)
        action_id = int(action_sequence[i]) if action_sequence is not None else None
        reward_val = float(reward_sequence[i]) if reward_sequence is not None else None
        done_val = bool(done_sequence[i] > 0.5) if done_sequence is not None else None

        compact = build_compact_state(
            filtered, action_id=action_id, reward=reward_val, done=done_val,
        )
        delta_t = base_t_offset + i
        blocks.append(f"[FUTURE STATE t+{delta_t}]\n{compact}")

    return "\n\n".join(blocks)


def build_history_block(
    obs_sequence,
    action_sequence,
    reward_sequence,
    done_sequence,
    n_history: int = 5,
) -> str:
    """Build a history block from the last N states before the current timestep.

    Args:
        obs_sequence: list/array of (8268,) observations for t-N through t-1
        action_sequence: corresponding actions
        reward_sequence: corresponding rewards
        done_sequence: corresponding done flags
        n_history: number of history states (for labelling offsets)

    Returns:
        Formatted history block string, or "(no history — episode start)" if empty
    """
    if len(obs_sequence) == 0:
        return "(no history — episode start)"

    blocks = []
    n = len(obs_sequence)
    for i in range(n):
        raw_text = obs_to_text(obs_sequence[i])
        filtered = filter_text_obs(raw_text)
        action_id = int(action_sequence[i]) if action_sequence is not None else None
        reward_val = float(reward_sequence[i]) if reward_sequence is not None else None
        done_val = bool(done_sequence[i] > 0.5) if done_sequence is not None else None

        compact = build_compact_state(
            filtered, action_id=action_id, reward=reward_val, done=done_val,
        )
        delta_t = i - n  # negative offset: -N, -N+1, ..., -1
        blocks.append(f"[HISTORY STATE t{delta_t}]\n{compact}")

    return "\n\n".join(blocks)
