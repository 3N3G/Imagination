#!/usr/bin/env python3
"""Future-imagination evaluation over Craftax trajectory text states.

This script is intentionally prompt-template driven:
- you provide prompt text templates in files
- the script injects trajectory/state context variables
- the script queries an LLM provider (Gemini or OpenAI-compatible endpoint)
- the script writes machine-readable logs + markdown/html inspection reports

It is designed so you can iterate prompt content quickly without touching code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]

from llm.prompts import MAP_INTERESTING_PREFIX, filter_text_obs

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis" / "future_imagination"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8000"

PROMPT_PLACEHOLDER_SENTINELS = (
    "[TODO]",
    "<REPLACE",
    "TODO_REPLACE",
)

_MAP_COORD_PREFIX_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:")
_MAP_ENTRY_RE = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*:\s*(.+?)\s*$")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class StepFeatures:
    health: Optional[float] = None
    food: Optional[float] = None
    drink: Optional[float] = None
    energy: Optional[float] = None
    mana: Optional[float] = None
    xp: Optional[float] = None
    floor: Optional[int] = None
    ladder_open: Optional[bool] = None
    direction: Optional[str] = None
    inventory: Dict[str, float] = field(default_factory=dict)
    equipment_lines: List[str] = field(default_factory=list)
    map_entries: List[str] = field(default_factory=list)
    map_line: str = ""


@dataclass
class TrajectoryStep:
    t: int
    episode_id: Optional[int]
    action_id: Optional[int]
    action_name: Optional[str]
    reward: Optional[float]
    done: Optional[bool]
    raw_text_obs: str
    filtered_text_obs: str
    features: StepFeatures
    compact_state: str = ""


@dataclass
class RunSpec:
    run_id: str
    role: str
    description: str
    template_path: Path
    template_text: str
    history_k: int
    history_format: str
    future_stride: int
    future_max_states: int
    future_format: str
    future_event_max: int
    future_include_terminal: bool
    generation: Dict[str, Any]
    stop_sequences: List[str]


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars < 8:
        return text[:max_chars]
    return text[: max_chars - 8] + "\n...<snip>"


def _parse_map_entries_from_filtered(filtered_text_obs: str) -> Tuple[str, List[str]]:
    map_line = ""
    for line in filtered_text_obs.splitlines():
        if line.startswith(MAP_INTERESTING_PREFIX):
            map_line = line.strip()
            break
    if not map_line:
        return "", []
    payload = map_line[len(MAP_INTERESTING_PREFIX):].strip()
    if not payload:
        return map_line, []
    starts = list(_MAP_COORD_PREFIX_RE.finditer(payload))
    if not starts:
        return map_line, []
    entries: List[str] = []
    for idx, match in enumerate(starts):
        start = match.start()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(payload)
        token = payload[start:end].strip().rstrip(",").strip()
        if token:
            entries.append(token)
    return map_line, entries


def _extract_features(filtered_text_obs: str) -> StepFeatures:
    features = StepFeatures()
    map_line, map_entries = _parse_map_entries_from_filtered(filtered_text_obs)
    features.map_line = map_line
    features.map_entries = map_entries

    stat_float_keys = {
        "Health": "health",
        "Food": "food",
        "Drink": "drink",
        "Energy": "energy",
        "Mana": "mana",
        "XP": "xp",
    }
    stat_int_keys = {
        "Floor": "floor",
    }
    inventory_keys = {
        "Wood",
        "Stone",
        "Coal",
        "Iron",
        "Diamond",
        "Sapphire",
        "Ruby",
        "Sapling",
        "Torch",
        "Arrow",
        "Book",
        "Red potion",
        "Green potion",
        "Blue potion",
        "Pink potion",
        "Cyan potion",
        "Yellow potion",
    }

    for raw_line in filtered_text_obs.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key in stat_float_keys:
                setattr(features, stat_float_keys[key], _safe_float(val))
                continue
            if key in stat_int_keys:
                setattr(features, stat_int_keys[key], _safe_int(val))
                continue
            if key == "Direction":
                features.direction = val
                continue
            if key == "Ladder Open":
                features.ladder_open = _safe_bool(val)
                continue
            if key in inventory_keys:
                maybe = _safe_float(val)
                if maybe is not None:
                    features.inventory[key] = maybe
                continue
        else:
            # Track simple gear/status lines for compact displays.
            if "with" in line or "Helmet" in line or "Armour" in line or "Sword" in line:
                features.equipment_lines.append(line)

    return features


def _compact_inventory_line(features: StepFeatures, max_items: int = 12) -> str:
    non_zero: List[Tuple[str, float]] = []
    for key in sorted(features.inventory.keys()):
        value = features.inventory[key]
        if abs(value) > 1e-9:
            non_zero.append((key, value))
    if not non_zero:
        return "Inventory+: (none)"
    shown = non_zero[:max_items]
    parts = [f"{k}={_fmt_num(v)}" for k, v in shown]
    suffix = "" if len(non_zero) <= max_items else ", ..."
    return "Inventory+: " + ", ".join(parts) + suffix


def _build_compact_state(step: TrajectoryStep) -> str:
    f = step.features
    lines: List[str] = []
    if f.map_line:
        lines.append(f.map_line)
    else:
        lines.append("Map (interesting tiles only): <unavailable>")
    lines.append(
        "Stats: "
        + ", ".join(
            [
                f"Health={_fmt_num(f.health)}",
                f"Food={_fmt_num(f.food)}",
                f"Drink={_fmt_num(f.drink)}",
                f"Energy={_fmt_num(f.energy)}",
                f"Mana={_fmt_num(f.mana)}",
                f"XP={_fmt_num(f.xp)}",
            ]
        )
    )
    lines.append(
        f"Direction={f.direction or 'NA'}, Floor={f.floor if f.floor is not None else 'NA'}, "
        f"LadderOpen={f.ladder_open if f.ladder_open is not None else 'NA'}"
    )
    lines.append(_compact_inventory_line(f))
    if f.equipment_lines:
        lines.append("Equipment: " + "; ".join(f.equipment_lines[:4]))
    lines.append(
        f"Action@t={step.action_name or step.action_id or 'NA'}, "
        f"Reward@t={_fmt_num(step.reward)}, Done@t={step.done if step.done is not None else 'NA'}"
    )
    return "\n".join(lines)


def load_trajectory_steps(
    trajectory_dir: Path,
    *,
    strict_map_validation: bool = True,
) -> List[TrajectoryStep]:
    text_obs_path = trajectory_dir / "text_obs.jsonl"
    if not text_obs_path.exists():
        raise FileNotFoundError(f"Missing trajectory text log: {text_obs_path}")

    steps: List[TrajectoryStep] = []
    with text_obs_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = int(rec["t"])
            raw_text = str(rec.get("raw_text_obs", ""))
            if not raw_text:
                raise ValueError(f"Missing raw_text_obs at t={t} in {text_obs_path}")
            filtered = filter_text_obs(raw_text, strict_map_validation=strict_map_validation)
            step = TrajectoryStep(
                t=t,
                episode_id=_safe_int(rec.get("episode_id")),
                action_id=_safe_int(rec.get("action_id")),
                action_name=(str(rec["action_name"]) if rec.get("action_name") is not None else None),
                reward=_safe_float(rec.get("reward")),
                done=_safe_bool(rec.get("done")),
                raw_text_obs=raw_text,
                filtered_text_obs=filtered,
                features=_extract_features(filtered),
            )
            steps.append(step)

    if not steps:
        raise ValueError(f"No trajectory states found in {text_obs_path}")

    steps.sort(key=lambda x: x.t)
    seen = set()
    for s in steps:
        if s.t in seen:
            raise ValueError(f"Duplicate timestep in trajectory: t={s.t}")
        seen.add(s.t)
    for s in steps:
        s.compact_state = _build_compact_state(s)

    return steps


def _resolve_path(path_like: str, *, base_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_specs(
    config: Dict[str, Any],
    *,
    config_path: Path,
    allow_placeholder_templates: bool,
) -> List[RunSpec]:
    raw_specs = config.get("run_specs", [])
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("Config must contain non-empty run_specs list.")

    base_dir = config_path.parent
    out: List[RunSpec] = []
    seen = set()

    for raw in raw_specs:
        run_id = str(raw.get("id", "")).strip()
        if not run_id:
            raise ValueError("Each run_spec must include non-empty 'id'.")
        if run_id in seen:
            raise ValueError(f"Duplicate run_spec id: {run_id}")
        seen.add(run_id)

        role = str(raw.get("role", "predict")).strip().lower()
        if role not in {"oracle", "predict", "other"}:
            raise ValueError(f"run_spec {run_id}: role must be oracle|predict|other, got {role!r}")
        description = str(raw.get("description", "")).strip()

        template_path_raw = raw.get("template_path")
        if not template_path_raw:
            raise ValueError(f"run_spec {run_id}: missing template_path")
        template_path = _resolve_path(str(template_path_raw), base_dir=base_dir)
        if not template_path.exists():
            raise FileNotFoundError(f"run_spec {run_id}: template_path not found: {template_path}")
        template_text = template_path.read_text(encoding="utf-8")
        if not allow_placeholder_templates:
            for sentinel in PROMPT_PLACEHOLDER_SENTINELS:
                if sentinel in template_text:
                    raise ValueError(
                        f"run_spec {run_id}: template {template_path} contains placeholder sentinel "
                        f"{sentinel!r}; update the prompt template or pass --allow-placeholder-templates."
                    )

        history_k = max(1, int(raw.get("history_k", 1)))
        history_format = str(raw.get("history_format", "compact")).strip().lower()
        if history_format not in {"compact", "filtered", "raw"}:
            raise ValueError(
                f"run_spec {run_id}: history_format must be compact|filtered|raw, got {history_format!r}"
            )
        future_stride = max(0, int(raw.get("future_stride", 0)))
        future_max_states = max(0, int(raw.get("future_max_states", 0)))
        future_format = str(raw.get("future_format", "compact")).strip().lower()
        if future_format not in {"compact", "filtered", "raw"}:
            raise ValueError(
                f"run_spec {run_id}: future_format must be compact|filtered|raw, got {future_format!r}"
            )
        future_event_max = max(0, int(raw.get("future_event_max", 0)))
        future_include_terminal = bool(raw.get("future_include_terminal", True))

        generation = dict(raw.get("generation", {}))
        stop_sequences = [str(s) for s in raw.get("stop_sequences", []) if str(s)]

        out.append(
            RunSpec(
                run_id=run_id,
                role=role,
                description=description,
                template_path=template_path,
                template_text=template_text,
                history_k=history_k,
                history_format=history_format,
                future_stride=future_stride,
                future_max_states=future_max_states,
                future_format=future_format,
                future_event_max=future_event_max,
                future_include_terminal=future_include_terminal,
                generation=generation,
                stop_sequences=stop_sequences,
            )
        )

    return out


def _render_state(step: TrajectoryStep, fmt: str) -> str:
    if fmt == "compact":
        return step.compact_state
    if fmt == "filtered":
        return step.filtered_text_obs
    if fmt == "raw":
        return step.raw_text_obs
    raise ValueError(f"Unknown state render format: {fmt!r}")


def _build_history_block(
    steps: List[TrajectoryStep],
    idx: int,
    *,
    history_k: int,
    state_format: str,
) -> Tuple[str, List[int]]:
    start = max(0, idx - history_k + 1)
    indices = list(range(start, idx + 1))
    blocks: List[str] = []
    for j in indices:
        step = steps[j]
        blocks.append(f"[STATE t={step.t}]\n{_render_state(step, state_format)}")
    return "\n\n".join(blocks), [steps[j].t for j in indices]


def _build_future_state_block(
    steps: List[TrajectoryStep],
    idx: int,
    *,
    future_stride: int,
    future_max_states: int,
    include_terminal: bool,
    state_format: str,
) -> Tuple[str, List[int]]:
    if future_stride <= 0 or future_max_states <= 0:
        return "", []

    base_t = steps[idx].t
    selected_indices: List[int] = list(range(idx, len(steps), future_stride))
    if len(selected_indices) > future_max_states:
        selected_indices = selected_indices[:future_max_states]

    terminal_idx = len(steps) - 1
    if include_terminal and terminal_idx not in selected_indices:
        selected_indices.append(terminal_idx)

    selected_indices = sorted(set(selected_indices))

    blocks: List[str] = []
    for j in selected_indices:
        step = steps[j]
        delta_t = step.t - base_t
        blocks.append(
            f"[FUTURE STATE t+{delta_t} | abs_t={step.t}]\n"
            f"{_render_state(step, state_format)}"
        )
    return "\n\n".join(blocks), [steps[j].t for j in selected_indices]


def _build_future_terminal_outcome_note(
    steps: List[TrajectoryStep],
    idx: int,
) -> str:
    base_t = steps[idx].t
    for j in range(idx, len(steps)):
        s = steps[j]
        if s.done is not True:
            continue
        delta_t = s.t - base_t
        action = s.action_name or s.action_id or "NA"
        reward = _fmt_num(s.reward)
        health = s.features.health
        health_txt = f", health={_fmt_num(health)}" if health is not None else ""
        return (
            f"Episode terminates at t+{delta_t} (abs_t={s.t}) with done=True after "
            f"action@t={action}, reward@t={reward}{health_txt}. "
            "Treat this as the player death/game-over endpoint for this rollout."
        )
    return "No terminal event appears in the provided future trajectory segment."


def _inventory_delta_line(prev: StepFeatures, nxt: StepFeatures) -> str:
    keys = sorted(set(prev.inventory.keys()) | set(nxt.inventory.keys()))
    deltas: List[Tuple[str, float, float, float]] = []
    for key in keys:
        a = prev.inventory.get(key, 0.0)
        b = nxt.inventory.get(key, 0.0)
        d = b - a
        if abs(d) > 1e-9:
            deltas.append((key, d, a, b))
    if not deltas:
        return ""
    deltas.sort(key=lambda x: (-abs(x[1]), x[0]))
    parts = []
    for key, delta, before, after in deltas[:6]:
        sign = "+" if delta > 0 else ""
        parts.append(f"{key} {sign}{_fmt_num(delta)} ({_fmt_num(before)}->{_fmt_num(after)})")
    suffix = "" if len(deltas) <= 6 else "; ..."
    return "Inventory delta: " + "; ".join(parts) + suffix


def _map_delta_line(prev: StepFeatures, nxt: StepFeatures) -> str:
    prev_set = set(prev.map_entries)
    next_set = set(nxt.map_entries)
    added_full = sorted(next_set - prev_set)
    removed_full = sorted(prev_set - next_set)
    if not added_full and not removed_full:
        return ""
    add_preview = ", ".join(added_full[:3]) if added_full else ""
    rem_preview = ", ".join(removed_full[:3]) if removed_full else ""
    payload = f"Map delta: +{len(added_full)} / -{len(removed_full)}"
    if add_preview:
        payload += f"; added[{add_preview}]"
    if rem_preview:
        payload += f"; removed[{rem_preview}]"
    return payload


def _scalar_delta_lines(prev: StepFeatures, nxt: StepFeatures) -> List[str]:
    deltas: List[str] = []
    scalar_fields = [
        ("Health", prev.health, nxt.health),
        ("Food", prev.food, nxt.food),
        ("Drink", prev.drink, nxt.drink),
        ("Energy", prev.energy, nxt.energy),
        ("Mana", prev.mana, nxt.mana),
        ("XP", prev.xp, nxt.xp),
        ("Floor", (float(prev.floor) if prev.floor is not None else None), (float(nxt.floor) if nxt.floor is not None else None)),
    ]
    for label, a, b in scalar_fields:
        if a is None or b is None:
            continue
        if abs(float(a) - float(b)) > 1e-9:
            deltas.append(f"{label}: {_fmt_num(float(a))}->{_fmt_num(float(b))}")
    if prev.ladder_open is not None and nxt.ladder_open is not None and prev.ladder_open != nxt.ladder_open:
        deltas.append(f"Ladder Open: {prev.ladder_open}->{nxt.ladder_open}")
    if prev.direction and nxt.direction and prev.direction != nxt.direction:
        deltas.append(f"Direction: {prev.direction}->{nxt.direction}")
    return deltas


def _describe_transition(
    prev_step: TrajectoryStep,
    next_step: TrajectoryStep,
    *,
    base_t: Optional[int] = None,
) -> Optional[str]:
    if base_t is None:
        t_label = f"[t={prev_step.t}->{next_step.t}]"
    else:
        t_label = (
            f"[t+{prev_step.t - base_t}->t+{next_step.t - base_t} | "
            f"abs_t={prev_step.t}->{next_step.t}]"
        )
    parts = [
        t_label,
        f"action={prev_step.action_name or prev_step.action_id or 'NA'}",
    ]
    if prev_step.reward is not None:
        parts.append(f"reward={_fmt_num(prev_step.reward)}")
    if prev_step.done is not None:
        parts.append(f"done={prev_step.done}")

    delta_parts = _scalar_delta_lines(prev_step.features, next_step.features)
    inv_delta = _inventory_delta_line(prev_step.features, next_step.features)
    if inv_delta:
        delta_parts.append(inv_delta)
    map_delta = _map_delta_line(prev_step.features, next_step.features)
    if map_delta:
        delta_parts.append(map_delta)

    reward_nonzero = prev_step.reward is not None and abs(prev_step.reward) > 1e-9
    has_done = bool(prev_step.done)
    if not delta_parts and not reward_nonzero and not has_done:
        return None

    return " | ".join(parts + delta_parts)


def _build_future_event_block(
    steps: List[TrajectoryStep],
    idx: int,
    *,
    future_event_max: int,
    include_terminal: bool,
) -> Tuple[str, int]:
    if future_event_max <= 0:
        return "", 0
    events: List[str] = []
    base_t = steps[idx].t
    for j in range(idx, len(steps) - 1):
        event_line = _describe_transition(steps[j], steps[j + 1], base_t=base_t)
        if event_line is None:
            continue
        events.append(event_line)
        if len(events) >= future_event_max:
            break

    if include_terminal and idx <= len(steps) - 1:
        last = steps[-1]
        if bool(last.done):
            terminal_line = (
                f"[TERMINAL t+{last.t - base_t} | abs_t={last.t}] done=True "
                f"after action={last.action_name or last.action_id or 'NA'} "
                f"reward={_fmt_num(last.reward)}"
            )
            if not events or events[-1] != terminal_line:
                events.append(terminal_line)

    return "\n".join(events), len(events)


class _StrictFormatDict(dict):
    def __missing__(self, key: str) -> Any:
        raise KeyError(key)


def _render_prompt(template_text: str, variables: Dict[str, Any], *, run_id: str) -> str:
    try:
        return template_text.format_map(_StrictFormatDict(variables))
    except KeyError as exc:
        raise KeyError(
            f"run_spec {run_id}: template references missing variable {exc!s}. "
            "Check template placeholders."
        ) from exc


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _word_jaccard(a: str, b: str) -> float:
    sa = set(_words(a))
    sb = set(_words(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


class BaseClient:
    def generate(
        self,
        *,
        prompt: str,
        generation: Dict[str, Any],
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class GeminiClient(BaseClient):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        timeout_s: float = 120.0,
        max_retries: int = 4,
        min_request_interval_s: float = 0.0,
    ):
        if not api_key:
            raise ValueError("Gemini API key is required for provider=gemini.")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.min_request_interval_s = min_request_interval_s
        self._last_request_ts = 0.0

    def _sleep_for_rate_limit(self) -> None:
        if self.min_request_interval_s <= 0:
            return
        now = time.perf_counter()
        elapsed = now - self._last_request_ts
        wait_s = self.min_request_interval_s - elapsed
        if wait_s > 0:
            time.sleep(wait_s)

    def _build_payload(
        self,
        *,
        prompt: str,
        generation: Dict[str, Any],
        stop_sequences: Optional[List[str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ]
        }
        gen_cfg: Dict[str, Any] = {}
        if "temperature" in generation:
            gen_cfg["temperature"] = float(generation["temperature"])
        if "max_output_tokens" in generation:
            gen_cfg["maxOutputTokens"] = int(generation["max_output_tokens"])
        if "top_p" in generation:
            gen_cfg["topP"] = float(generation["top_p"])
        if "thinking_budget" in generation:
            thinking_budget = _safe_int(generation.get("thinking_budget"))
            if thinking_budget is not None:
                gen_cfg["thinkingConfig"] = {
                    "thinkingBudget": max(0, thinking_budget)
                }
        if stop_sequences:
            cleaned = [str(s) for s in stop_sequences if str(s)]
            if cleaned:
                gen_cfg["stopSequences"] = cleaned
        if gen_cfg:
            payload["generationConfig"] = gen_cfg
        return payload

    @staticmethod
    def _extract_text(resp_json: Dict[str, Any]) -> str:
        candidates = resp_json.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        chunks: List[str] = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                chunks.append(str(part["text"]))
        return "".join(chunks)

    @staticmethod
    def _extract_usage(resp_json: Dict[str, Any]) -> Dict[str, Optional[int]]:
        usage = resp_json.get("usageMetadata", {})
        return {
            "prompt_tokens": _safe_int(usage.get("promptTokenCount")),
            "completion_tokens": _safe_int(usage.get("candidatesTokenCount")),
            "total_tokens": _safe_int(usage.get("totalTokenCount")),
            "thoughts_tokens": _safe_int(usage.get("thoughtsTokenCount")),
        }

    def generate(
        self,
        *,
        prompt: str,
        generation: Dict[str, Any],
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        payload = self._build_payload(
            prompt=prompt,
            generation=generation,
            stop_sequences=stop_sequences,
        )
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        backoff_s = 1.0

        for attempt in range(self.max_retries + 1):
            self._sleep_for_rate_limit()
            req = urlrequest.Request(url, data=body, headers=headers, method="POST")
            started = time.perf_counter()
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                    raw_text = resp.read().decode("utf-8")
                self._last_request_ts = time.perf_counter()
                parsed = json.loads(raw_text)
                return {
                    "ok": True,
                    "response_text": self._extract_text(parsed),
                    "usage": self._extract_usage(parsed),
                    "raw_response": parsed,
                    "error": "",
                    "attempt": attempt,
                    "request_s": time.perf_counter() - started,
                }
            except urlerror.HTTPError as exc:
                response_text = ""
                try:
                    response_text = exc.read().decode("utf-8")
                except Exception:
                    response_text = str(exc)
                status = int(exc.code)
                retryable = status in {408, 429, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    time.sleep(backoff_s)
                    backoff_s *= 2.0
                    continue
                return {
                    "ok": False,
                    "response_text": "",
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    "raw_response": {},
                    "error": f"HTTPError {status}: {response_text}",
                    "attempt": attempt,
                    "request_s": time.perf_counter() - started,
                }
            except Exception as exc:  # pragma: no cover - network/runtime issues
                retryable = attempt < self.max_retries
                if retryable:
                    time.sleep(backoff_s)
                    backoff_s *= 2.0
                    continue
                return {
                    "ok": False,
                    "response_text": "",
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    "raw_response": {},
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "attempt": attempt,
                    "request_s": time.perf_counter() - started,
                }


class OpenAICompatibleClient(BaseClient):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str = "",
        timeout_s: float = 120.0,
        max_retries: int = 4,
        min_request_interval_s: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.min_request_interval_s = min_request_interval_s
        self._last_request_ts = 0.0

    def _sleep_for_rate_limit(self) -> None:
        if self.min_request_interval_s <= 0:
            return
        now = time.perf_counter()
        elapsed = now - self._last_request_ts
        wait_s = self.min_request_interval_s - elapsed
        if wait_s > 0:
            time.sleep(wait_s)

    @staticmethod
    def _extract_text(resp_json: Dict[str, Any]) -> str:
        choices = resp_json.get("choices", [])
        if not choices:
            return ""
        first = choices[0]
        if "text" in first:
            return str(first.get("text", ""))
        msg = first.get("message")
        if isinstance(msg, dict):
            return str(msg.get("content", ""))
        return ""

    @staticmethod
    def _extract_usage(resp_json: Dict[str, Any]) -> Dict[str, Optional[int]]:
        usage = resp_json.get("usage", {})
        return {
            "prompt_tokens": _safe_int(usage.get("prompt_tokens")),
            "completion_tokens": _safe_int(usage.get("completion_tokens")),
            "total_tokens": _safe_int(usage.get("total_tokens")),
        }

    def _build_payload(
        self,
        *,
        prompt: str,
        generation: Dict[str, Any],
        stop_sequences: Optional[List[str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if "temperature" in generation:
            payload["temperature"] = float(generation["temperature"])
        if "max_output_tokens" in generation:
            payload["max_tokens"] = int(generation["max_output_tokens"])
        if "top_p" in generation:
            payload["top_p"] = float(generation["top_p"])
        if stop_sequences:
            cleaned = [str(s) for s in stop_sequences if str(s)]
            if cleaned:
                payload["stop"] = cleaned
        return payload

    def generate(
        self,
        *,
        prompt: str,
        generation: Dict[str, Any],
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_payload(
            prompt=prompt,
            generation=generation,
            stop_sequences=stop_sequences,
        )
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        backoff_s = 1.0

        for attempt in range(self.max_retries + 1):
            self._sleep_for_rate_limit()
            req = urlrequest.Request(url, data=body, headers=headers, method="POST")
            started = time.perf_counter()
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_s) as resp:
                    raw_text = resp.read().decode("utf-8")
                self._last_request_ts = time.perf_counter()
                parsed = json.loads(raw_text)
                return {
                    "ok": True,
                    "response_text": self._extract_text(parsed),
                    "usage": self._extract_usage(parsed),
                    "raw_response": parsed,
                    "error": "",
                    "attempt": attempt,
                    "request_s": time.perf_counter() - started,
                }
            except urlerror.HTTPError as exc:
                response_text = ""
                try:
                    response_text = exc.read().decode("utf-8")
                except Exception:
                    response_text = str(exc)
                status = int(exc.code)
                retryable = status in {408, 429, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    time.sleep(backoff_s)
                    backoff_s *= 2.0
                    continue
                return {
                    "ok": False,
                    "response_text": "",
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    "raw_response": {},
                    "error": f"HTTPError {status}: {response_text}",
                    "attempt": attempt,
                    "request_s": time.perf_counter() - started,
                }
            except Exception as exc:  # pragma: no cover
                retryable = attempt < self.max_retries
                if retryable:
                    time.sleep(backoff_s)
                    backoff_s *= 2.0
                    continue
                return {
                    "ok": False,
                    "response_text": "",
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    "raw_response": {},
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "attempt": attempt,
                    "request_s": time.perf_counter() - started,
                }


class HFLocalClient(BaseClient):
    def __init__(
        self,
        *,
        model: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
        trim_to_headline: bool = True,
    ):
        self.model_name = model
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.enable_thinking = bool(enable_thinking)
        self.trim_to_headline = bool(trim_to_headline)
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(self.dtype).lower(), torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch

    def _build_input_text(self, prompt: str) -> str:
        assert self._tokenizer is not None
        try:
            return str(
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            )
        except Exception:
            return prompt

    @staticmethod
    def _apply_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> str:
        if not stop_sequences:
            return text
        stops = [s for s in stop_sequences if isinstance(s, str) and s]
        if not stops:
            return text
        cut = len(text)
        for s in stops:
            pos = text.find(s)
            if pos >= 0 and pos < cut:
                cut = pos
        return text[:cut]

    @staticmethod
    def _trim_reasoning_preamble(text: str) -> str:
        if not text:
            return text
        t = text.strip()

        # Remove explicit <think>...</think> sections if present.
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()

        # Keep only model-formatted answer if Headline is present.
        m = re.search(r"(?im)^headline\s*:", t)
        if m:
            t = t[m.start():].strip()
        return t

    def generate(
        self,
        *,
        prompt: str,
        generation: Dict[str, Any],
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            self._ensure_loaded()
            assert self._tokenizer is not None
            assert self._model is not None
            assert self._torch is not None

            input_text = self._build_input_text(prompt)
            inputs = self._tokenizer(input_text, return_tensors="pt")

            try:
                model_device = self._model.device
            except Exception:
                model_device = next(self._model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            max_new_tokens = int(generation.get("max_output_tokens", 512))
            temperature = float(generation.get("temperature", 0.0))
            top_p = float(generation.get("top_p", 1.0))
            do_sample = temperature > 0.0
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            with self._torch.no_grad():
                out = self._model.generate(**inputs, **gen_kwargs)

            prompt_len = int(inputs["input_ids"].shape[-1])
            out_ids = out[0][prompt_len:]
            text = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            if self.trim_to_headline:
                text = self._trim_reasoning_preamble(text)
            text = self._apply_stop_sequences(text, stop_sequences).strip()
            completion_tokens = int(out_ids.shape[-1])

            return {
                "ok": True,
                "response_text": text,
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_len + completion_tokens,
                },
                "raw_response": {},
                "error": "",
                "attempt": 0,
                "request_s": time.perf_counter() - started,
            }
        except Exception as exc:
            return {
                "ok": False,
                "response_text": "",
                "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "raw_response": {},
                "error": f"{exc.__class__.__name__}: {exc}",
                "attempt": 0,
                "request_s": time.perf_counter() - started,
            }


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _select_timesteps(steps: List[TrajectoryStep], selection_cfg: Dict[str, Any]) -> List[int]:
    t_min = steps[0].t
    t_max = steps[-1].t

    mode = str(selection_cfg.get("mode", "range_plus_explicit")).strip().lower()
    if mode not in {"range_plus_explicit", "explicit_only"}:
        raise ValueError(
            f"Invalid selection.mode={mode!r}; expected 'range_plus_explicit' or 'explicit_only'."
        )

    start_t = _safe_int(selection_cfg.get("start_t"))
    end_t = _safe_int(selection_cfg.get("end_t"))
    stride = _safe_int(selection_cfg.get("stride"))
    max_states = _safe_int(selection_cfg.get("max_states"))
    include_terminal_t = bool(selection_cfg.get("include_terminal_t", True))
    explicit_raw = selection_cfg.get("explicit_timesteps", [])
    explicit_ts = [int(x) for x in explicit_raw] if isinstance(explicit_raw, list) else []

    if start_t is None:
        start_t = t_min
    if end_t is None:
        end_t = t_max
    if stride is None or stride <= 0:
        stride = 1
    if start_t > end_t:
        raise ValueError(f"Invalid selection: start_t ({start_t}) > end_t ({end_t})")

    selected = set()
    if mode == "range_plus_explicit":
        for t in range(start_t, end_t + 1, stride):
            selected.add(t)
    for t in explicit_ts:
        selected.add(t)
    if include_terminal_t:
        selected.add(t_max)

    valid_ts = sorted(t for t in selected if t_min <= t <= t_max)
    if max_states is not None and max_states > 0 and len(valid_ts) > max_states:
        valid_ts = valid_ts[:max_states]
    if not valid_ts:
        raise ValueError("Selection produced zero valid timesteps.")
    return valid_ts


def _sanitize_args_for_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [_json_safe(v) for v in value]
        return value

    d = _json_safe(vars(args).copy())
    if d.get("api_key"):
        d["api_key"] = "<redacted>"
    return d


def _build_client(args: argparse.Namespace) -> BaseClient:
    if args.provider == "gemini":
        api_key = args.api_key or os.getenv(args.api_key_env, "")
        return GeminiClient(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url or DEFAULT_GEMINI_BASE_URL,
            timeout_s=args.request_timeout_s,
            max_retries=args.max_retries,
            min_request_interval_s=args.min_request_interval_s,
        )
    if args.provider == "openai_compatible":
        api_key = args.api_key or os.getenv(args.api_key_env, "")
        return OpenAICompatibleClient(
            model=args.model,
            base_url=args.base_url or DEFAULT_OPENAI_BASE_URL,
            api_key=api_key,
            timeout_s=args.request_timeout_s,
            max_retries=args.max_retries,
            min_request_interval_s=args.min_request_interval_s,
        )
    if args.provider == "hf_local":
        return HFLocalClient(
            model=args.model,
            device_map=args.hf_device_map,
            dtype=args.hf_dtype,
            trust_remote_code=args.hf_trust_remote_code,
            enable_thinking=args.hf_enable_thinking,
            trim_to_headline=args.hf_trim_to_headline,
        )
    raise ValueError(f"Unknown provider: {args.provider!r}")


def _prepare_context(
    *,
    steps: List[TrajectoryStep],
    step_idx: int,
    spec: RunSpec,
    trajectory_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    step = steps[step_idx]

    history_block, history_ts = _build_history_block(
        steps,
        step_idx,
        history_k=spec.history_k,
        state_format=spec.history_format,
    )
    future_state_block, future_state_ts = _build_future_state_block(
        steps,
        step_idx,
        future_stride=spec.future_stride,
        future_max_states=spec.future_max_states,
        include_terminal=spec.future_include_terminal,
        state_format=spec.future_format,
    )
    future_event_block, future_event_count = _build_future_event_block(
        steps,
        step_idx,
        future_event_max=spec.future_event_max,
        include_terminal=spec.future_include_terminal,
    )
    future_terminal_outcome = _build_future_terminal_outcome_note(steps, step_idx)

    vars_for_template: Dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "timestep": step.t,
        "total_timesteps": len(steps),
        "episode_id": step.episode_id if step.episode_id is not None else "NA",
        "future_snapshot_stride": spec.future_stride,
        "action_at_t": step.action_name or step.action_id or "NA",
        "reward_at_t": _fmt_num(step.reward),
        "done_at_t": step.done if step.done is not None else "NA",
        "current_state_raw": step.raw_text_obs,
        "current_state_filtered": step.filtered_text_obs,
        "current_state_compact": step.compact_state,
        "history_block": history_block,
        "future_state_block": future_state_block,
        "future_event_block": future_event_block,
        "future_terminal_outcome": future_terminal_outcome,
    }
    meta = {
        "history_ts": history_ts,
        "future_state_ts": future_state_ts,
        "future_event_count": future_event_count,
    }
    return vars_for_template, meta


def _estimate_workload(
    *,
    steps: List[TrajectoryStep],
    selected_timesteps: List[int],
    run_specs: List[RunSpec],
) -> Dict[str, Any]:
    t_to_idx = {step.t: idx for idx, step in enumerate(steps)}
    per_run: Dict[str, Dict[str, Any]] = {}
    total_prompts = 0
    total_prompt_chars = 0
    for spec in run_specs:
        prompt_chars = 0
        for t in selected_timesteps:
            idx = t_to_idx[t]
            vars_for_template, _ = _prepare_context(
                steps=steps,
                step_idx=idx,
                spec=spec,
                trajectory_id="trajectory",
            )
            prompt = _render_prompt(spec.template_text, vars_for_template, run_id=spec.run_id)
            prompt_chars += len(prompt)
            total_prompts += 1
            total_prompt_chars += len(prompt)
        per_run[spec.run_id] = {
            "num_calls": len(selected_timesteps),
            "prompt_chars_total": prompt_chars,
            "prompt_chars_mean": (prompt_chars / len(selected_timesteps) if selected_timesteps else 0.0),
            "prompt_tokens_rough_total": int(round(prompt_chars / 4.0)),
            "prompt_tokens_rough_mean": (prompt_chars / 4.0 / len(selected_timesteps) if selected_timesteps else 0.0),
        }

    return {
        "selected_timesteps": len(selected_timesteps),
        "run_specs": len(run_specs),
        "total_calls": total_prompts,
        "prompt_chars_total": total_prompt_chars,
        "prompt_tokens_rough_total": int(round(total_prompt_chars / 4.0)),
        "per_run": per_run,
    }


def _write_selected_states(selected_states_path: Path, selected_steps: List[TrajectoryStep]) -> None:
    with selected_states_path.open("w", encoding="utf-8") as f:
        for step in selected_steps:
            payload = {
                "t": step.t,
                "episode_id": step.episode_id,
                "action_id": step.action_id,
                "action_name": step.action_name,
                "reward": step.reward,
                "done": step.done,
                "compact_state": step.compact_state,
                "filtered_text_obs": step.filtered_text_obs,
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _parse_map_entry_token(token: str) -> Optional[Tuple[int, int, str]]:
    match = _MAP_ENTRY_RE.match(token.strip())
    if not match:
        return None
    row = int(match.group(1))
    col = int(match.group(2))
    tile = match.group(3).strip()
    return row, col, tile


def _build_viewer_states(steps: List[TrajectoryStep]) -> Dict[str, Any]:
    states: List[Dict[str, Any]] = []
    min_row: Optional[int] = None
    max_row: Optional[int] = None
    min_col: Optional[int] = None
    max_col: Optional[int] = None

    for step in steps:
        parsed_entries: List[Dict[str, Any]] = []
        for token in step.features.map_entries:
            parsed = _parse_map_entry_token(token)
            if parsed is None:
                continue
            row, col, tile = parsed
            parsed_entries.append({"row": row, "col": col, "tile": tile})
            min_row = row if min_row is None else min(min_row, row)
            max_row = row if max_row is None else max(max_row, row)
            min_col = col if min_col is None else min(min_col, col)
            max_col = col if max_col is None else max(max_col, col)

        states.append(
            {
                "t": step.t,
                "episode_id": step.episode_id,
                "action_name": step.action_name,
                "action_id": step.action_id,
                "reward": step.reward,
                "done": step.done,
                "floor": step.features.floor,
                "direction": step.features.direction,
                "health": step.features.health,
                "food": step.features.food,
                "drink": step.features.drink,
                "energy": step.features.energy,
                "mana": step.features.mana,
                "xp": step.features.xp,
                "ladder_open": step.features.ladder_open,
                "inventory_line": _compact_inventory_line(step.features, max_items=12),
                "state_text": step.compact_state,
                "map_entries": parsed_entries,
            }
        )

    if min_row is None or max_row is None or min_col is None or max_col is None:
        # Default Craftax local view if map parse is unavailable.
        min_row, max_row, min_col, max_col = -5, 5, -4, 4

    return {
        "bounds": {
            "min_row": int(min_row),
            "max_row": int(max_row),
            "min_col": int(min_col),
            "max_col": int(max_col),
        },
        "states": states,
    }


def _attach_viewer_frame_paths(
    viewer_payload: Dict[str, Any],
    *,
    frame_dir: Optional[Path],
    report_dir: Path,
) -> Dict[str, Any]:
    if frame_dir is None:
        return viewer_payload
    if not frame_dir.exists():
        return viewer_payload

    states = viewer_payload.get("states")
    if not isinstance(states, list):
        return viewer_payload

    for state in states:
        t = _safe_int(state.get("t"))
        if t is None:
            state["frame_path"] = ""
            continue
        frame_path = frame_dir / f"t_{t:05d}.png"
        if frame_path.exists():
            rel = os.path.relpath(frame_path, report_dir)
            state["frame_path"] = f"{rel}?v={int(frame_path.stat().st_mtime_ns)}"
        else:
            state["frame_path"] = ""
    return viewer_payload


def _records_index(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    index: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for rec in records:
        t = _safe_int(rec.get("t"))
        run_id = rec.get("run_id")
        if t is None or not run_id:
            continue
        index[(t, str(run_id))] = rec
    return index


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _summarize_by_run(records: List[Dict[str, Any]], run_specs: List[RunSpec]) -> List[Dict[str, Any]]:
    role_by_run = {s.run_id: s.role for s in run_specs}
    desc_by_run = {s.run_id: s.description for s in run_specs}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get("run_id"))].append(rec)

    rows: List[Dict[str, Any]] = []
    for run_id in sorted(grouped.keys()):
        recs = grouped[run_id]
        ok = [r for r in recs if r.get("status") == "ok"]
        errors = [r for r in recs if r.get("status") != "ok"]
        latency_vals = [_safe_float(r.get("latency_s")) for r in ok]
        latency_vals = [x for x in latency_vals if x is not None]
        prompt_tokens = [
            _safe_int((r.get("usage") or {}).get("prompt_tokens"))
            for r in ok
        ]
        prompt_tokens = [x for x in prompt_tokens if x is not None]
        completion_tokens = [
            _safe_int((r.get("usage") or {}).get("completion_tokens"))
            for r in ok
        ]
        completion_tokens = [x for x in completion_tokens if x is not None]

        rows.append(
            {
                "run_id": run_id,
                "role": role_by_run.get(run_id, "unknown"),
                "description": desc_by_run.get(run_id, ""),
                "num_records": len(recs),
                "num_ok": len(ok),
                "num_errors": len(errors),
                "latency_mean_s": (sum(latency_vals) / len(latency_vals) if latency_vals else ""),
                "prompt_tokens_total": (sum(prompt_tokens) if prompt_tokens else ""),
                "completion_tokens_total": (sum(completion_tokens) if completion_tokens else ""),
            }
        )
    return rows


def _build_pairwise_scores(
    records: List[Dict[str, Any]],
    *,
    oracle_run_id: str,
) -> List[Dict[str, Any]]:
    by_t: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        t = _safe_int(rec.get("t"))
        run_id = rec.get("run_id")
        if t is None or not run_id:
            continue
        by_t[t][str(run_id)] = rec

    rows: List[Dict[str, Any]] = []
    for t in sorted(by_t.keys()):
        bucket = by_t[t]
        oracle = bucket.get(oracle_run_id)
        if not oracle or oracle.get("status") != "ok":
            continue
        oracle_text = str(oracle.get("response_text", ""))
        for run_id, rec in bucket.items():
            if run_id == oracle_run_id:
                continue
            if rec.get("status") != "ok":
                continue
            pred_text = str(rec.get("response_text", ""))
            rows.append(
                {
                    "t": t,
                    "oracle_run_id": oracle_run_id,
                    "run_id": run_id,
                    "word_jaccard": _word_jaccard(oracle_text, pred_text),
                    "char_similarity": _char_similarity(oracle_text, pred_text),
                    "oracle_chars": len(oracle_text),
                    "prediction_chars": len(pred_text),
                }
            )
    return rows


def _build_markdown_report(
    *,
    output_path: Path,
    records: List[Dict[str, Any]],
    selected_state_map: Dict[int, Dict[str, Any]],
    run_specs: List[RunSpec],
    summary_rows: List[Dict[str, Any]],
    pairwise_rows: List[Dict[str, Any]],
    oracle_run_id: str,
    max_timesteps: int,
) -> None:
    by_t: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        t = _safe_int(rec.get("t"))
        run_id = str(rec.get("run_id", ""))
        if t is None or not run_id:
            continue
        by_t[t][run_id] = rec

    ordered_run_ids = [spec.run_id for spec in run_specs]
    lines: List[str] = []
    lines.append("# Future Imagination Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat()}`")
    lines.append(f"- Timesteps in run: `{len(by_t)}`")
    lines.append(f"- Oracle run id: `{oracle_run_id}`")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append("| run_id | role | ok | errors | mean_latency_s | prompt_tokens_total | completion_tokens_total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {run_id} | {role} | {num_ok} | {num_errors} | {latency_mean_s} | {prompt_tokens_total} | {completion_tokens_total} |".format(
                **{k: row.get(k, "") for k in row.keys()}
            )
        )
    lines.append("")

    if pairwise_rows:
        lines.append("## Pairwise Scores vs Oracle")
        lines.append("")
        lines.append("| run_id | mean_word_jaccard | mean_char_similarity | n |")
        lines.append("|---|---:|---:|---:|")
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in pairwise_rows:
            grouped[str(row["run_id"])].append(row)
        for run_id in ordered_run_ids:
            rows = grouped.get(run_id, [])
            if not rows:
                continue
            mj = sum(float(x["word_jaccard"]) for x in rows) / len(rows)
            mc = sum(float(x["char_similarity"]) for x in rows) / len(rows)
            lines.append(f"| {run_id} | {mj:.4f} | {mc:.4f} | {len(rows)} |")
        lines.append("")

    shown = 0
    for t in sorted(by_t.keys()):
        shown += 1
        if max_timesteps > 0 and shown > max_timesteps:
            break
        state_payload = selected_state_map.get(t, {})
        lines.append(f"## t={t}")
        lines.append("")
        lines.append("### Current State (Compact)")
        lines.append("```text")
        lines.append(str(state_payload.get("compact_state", "")))
        lines.append("```")
        lines.append("")
        for run_id in ordered_run_ids:
            rec = by_t[t].get(run_id)
            lines.append(f"### {run_id}")
            if rec is None:
                lines.append("_missing_")
                lines.append("")
                continue
            status = rec.get("status", "unknown")
            lines.append(f"- status: `{status}` | latency_s: `{rec.get('latency_s')}`")
            if status != "ok":
                lines.append(f"- error: `{rec.get('error', '')}`")
                lines.append("")
                continue
            lines.append("")
            lines.append("```text")
            lines.append(str(rec.get("response_text", "")))
            lines.append("```")
            prompt_preview = str(rec.get("prompt_preview", ""))
            if prompt_preview:
                lines.append("<details><summary>Prompt Preview</summary>")
                lines.append("")
                lines.append("```text")
                lines.append(prompt_preview)
                lines.append("```")
                lines.append("")
                lines.append("</details>")
            lines.append("")

    omitted = max(0, len(by_t) - shown)
    if omitted > 0:
        lines.append(f"_Omitted {omitted} timesteps from markdown to keep file readable._")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_html_report(
    *,
    output_path: Path,
    records: List[Dict[str, Any]],
    selected_state_map: Dict[int, Dict[str, Any]],
    trajectory_steps: List[TrajectoryStep],
    viewer_frame_dir: Optional[Path],
    run_specs: List[RunSpec],
    summary_rows: List[Dict[str, Any]],
    pairwise_rows: List[Dict[str, Any]],
    oracle_run_id: str,
) -> None:
    by_t: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        t = _safe_int(rec.get("t"))
        run_id = str(rec.get("run_id", ""))
        if t is None or not run_id:
            continue
        by_t[t][run_id] = rec

    ordered_run_ids = [spec.run_id for spec in run_specs]
    run_desc = {spec.run_id: spec.description for spec in run_specs}
    inspection_timesteps = sorted(by_t.keys())
    viewer_payload = _build_viewer_states(trajectory_steps)
    viewer_payload = _attach_viewer_frame_paths(
        viewer_payload,
        frame_dir=viewer_frame_dir,
        report_dir=output_path.parent,
    )
    viewer_payload_json = html.escape(json.dumps(viewer_payload, ensure_ascii=True))
    inspection_timesteps_json = html.escape(json.dumps(inspection_timesteps, ensure_ascii=True))

    h: List[str] = []
    h.append("<!DOCTYPE html>")
    h.append("<html><head><meta charset='utf-8'>")
    h.append("<title>Future Imagination Report</title>")
    h.append(
        "<style>"
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:18px;line-height:1.35;}"
        "table{border-collapse:collapse;width:100%;margin:10px 0 20px 0;}"
        "th,td{border:1px solid #ddd;padding:6px;vertical-align:top;font-size:13px;}"
        "th{background:#f6f6f6;text-align:left;}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#fafafa;border:1px solid #eee;padding:10px;}"
        "details{margin:10px 0;border:1px solid #e6e6e6;padding:8px;border-radius:6px;}"
        ".err{color:#a00;font-weight:600;}"
        ".muted{color:#666;font-size:12px;}"
        ".run-box{border:1px solid #ddd;border-radius:6px;padding:8px;margin:8px 0;}"
        ".viewer-wrap{border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0 18px 0;}"
        ".viewer-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;}"
        ".viewer-controls input[type=range]{min-width:320px;flex:1 1 440px;}"
        ".viewer-frame{display:block;max-width:100%;height:auto;border:1px solid #ddd;background:#111;"
        "image-rendering:pixelated;image-rendering:crisp-edges;margin:8px 0;}"
        ".viewer-grid{display:grid;gap:2px;justify-content:start;margin:8px 0;border:1px solid #ddd;padding:8px;background:#f8f8f8;}"
        ".map-cell{width:28px;height:28px;display:flex;align-items:center;justify-content:center;border:1px solid #e6e6e6;"
        "font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#fff;}"
        ".map-cell.player{background:#d8f5d1;font-weight:700;}"
        ".map-cell.enemy{background:#ffe0e0;}"
        ".map-cell.water{background:#deefff;}"
        ".map-cell.path{background:#f7f0dd;}"
        ".map-cell.resource{background:#e6f7e6;}"
        ".map-cell.wall{background:#efefef;}"
        ".map-cell.darkness{background:#2f3340;color:#e8edf5;}"
        ".viewer-metrics{display:flex;gap:10px;flex-wrap:wrap;font-size:13px;margin-bottom:6px;}"
        "</style>"
    )
    h.append("</head><body>")
    h.append("<h1>Future Imagination Report</h1>")
    h.append(f"<p class='muted'>Generated: {html.escape(datetime.now().isoformat())}</p>")
    h.append(f"<p class='muted'>Oracle run id: <code>{html.escape(oracle_run_id)}</code></p>")

    h.append("<h2>Run Summary</h2>")
    h.append("<table><thead><tr><th>run_id</th><th>role</th><th>ok</th><th>errors</th><th>mean_latency_s</th><th>prompt_tokens_total</th><th>completion_tokens_total</th></tr></thead><tbody>")
    for row in summary_rows:
        h.append(
            "<tr>"
            f"<td><code>{html.escape(str(row.get('run_id', '')))}</code></td>"
            f"<td>{html.escape(str(row.get('role', '')))}</td>"
            f"<td>{html.escape(str(row.get('num_ok', '')))}</td>"
            f"<td>{html.escape(str(row.get('num_errors', '')))}</td>"
            f"<td>{html.escape(str(row.get('latency_mean_s', '')))}</td>"
            f"<td>{html.escape(str(row.get('prompt_tokens_total', '')))}</td>"
            f"<td>{html.escape(str(row.get('completion_tokens_total', '')))}</td>"
            "</tr>"
        )
    h.append("</tbody></table>")

    if pairwise_rows:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in pairwise_rows:
            grouped[str(row["run_id"])].append(row)
        h.append("<h2>Pairwise Scores vs Oracle</h2>")
        h.append("<table><thead><tr><th>run_id</th><th>mean_word_jaccard</th><th>mean_char_similarity</th><th>n</th></tr></thead><tbody>")
        for run_id in ordered_run_ids:
            rows = grouped.get(run_id, [])
            if not rows:
                continue
            mj = sum(float(x["word_jaccard"]) for x in rows) / len(rows)
            mc = sum(float(x["char_similarity"]) for x in rows) / len(rows)
            h.append(
                "<tr>"
                f"<td><code>{html.escape(run_id)}</code></td>"
                f"<td>{mj:.4f}</td>"
                f"<td>{mc:.4f}</td>"
                f"<td>{len(rows)}</td>"
                "</tr>"
            )
        h.append("</tbody></table>")

    h.append("<h2>Trajectory Viewer</h2>")
    h.append(
        "<p class='muted'>Scrub the timeline to inspect observed game states and compare against model summaries.</p>"
    )
    h.append("<div class='viewer-wrap'>")
    h.append("<div class='viewer-controls'>")
    h.append("<button type='button' id='viewer-prev'>&lt;</button>")
    h.append("<input id='viewer-slider' type='range' min='0' max='0' step='1' value='0' />")
    h.append("<button type='button' id='viewer-next'>&gt;</button>")
    h.append("<label>t=<strong id='viewer-t'>0</strong></label>")
    h.append("<label>idx=<span id='viewer-idx'>0</span></label>")
    h.append("</div>")
    h.append("<div class='viewer-controls'>")
    h.append("<label for='viewer-jump'>Jump to inspected t:</label>")
    h.append("<select id='viewer-jump'><option value=''>-- choose --</option></select>")
    h.append("</div>")
    h.append("<div class='viewer-metrics' id='viewer-metrics'></div>")
    h.append("<img id='viewer-frame' class='viewer-frame' alt='Craftax frame' />")
    h.append("<div id='viewer-grid' class='viewer-grid'></div>")
    h.append("<pre id='viewer-state-text'></pre>")
    h.append("</div>")
    h.append(f"<script id='viewer-data' type='application/json'>{viewer_payload_json}</script>")
    h.append(
        f"<script id='viewer-inspection-ts' type='application/json'>{inspection_timesteps_json}</script>"
    )
    h.append(
        "<script>"
        "(function(){"
        "const dataEl=document.getElementById('viewer-data');"
        "const inspEl=document.getElementById('viewer-inspection-ts');"
        "if(!dataEl){return;}"
        "let payload={states:[],bounds:{min_row:-5,max_row:5,min_col:-4,max_col:4}};"
        "let inspectionTs=[];"
        "try{payload=JSON.parse(dataEl.textContent||'{}')||payload;}catch(_){payload=payload;}"
        "try{inspectionTs=JSON.parse(inspEl.textContent||'[]')||[];}catch(_){inspectionTs=[];}"
        "const states=Array.isArray(payload.states)?payload.states:[];"
        "const b=payload.bounds||{};"
        "const bounds={"
        "min_row:Number.isFinite(b.min_row)?b.min_row:-5,"
        "max_row:Number.isFinite(b.max_row)?b.max_row:5,"
        "min_col:Number.isFinite(b.min_col)?b.min_col:-4,"
        "max_col:Number.isFinite(b.max_col)?b.max_col:4"
        "};"
        "const slider=document.getElementById('viewer-slider');"
        "const prevBtn=document.getElementById('viewer-prev');"
        "const nextBtn=document.getElementById('viewer-next');"
        "const tEl=document.getElementById('viewer-t');"
        "const idxEl=document.getElementById('viewer-idx');"
        "const frameEl=document.getElementById('viewer-frame');"
        "const gridEl=document.getElementById('viewer-grid');"
        "const textEl=document.getElementById('viewer-state-text');"
        "const metricsEl=document.getElementById('viewer-metrics');"
        "const jumpEl=document.getElementById('viewer-jump');"
        "if(!slider||!gridEl||!textEl||states.length===0){"
        "if(textEl){textEl.textContent='No trajectory states available for viewer.';}return;}"
        "const tToIdx=new Map();"
        "for(let i=0;i<states.length;i+=1){tToIdx.set(states[i].t,i);}"
        "for(const t of inspectionTs){if(!tToIdx.has(t)){continue;}const opt=document.createElement('option');opt.value=String(t);opt.textContent='t='+String(t);jumpEl.appendChild(opt);}"
        "const tileSymbol=(tile)=>{"
        "const s=(tile||'').toLowerCase();"
        "if(!s){return ' ';}"
        "if(s.includes('player')){return 'P';}"
        "if(s.includes('ladder')){return 'L';}"
        "if(s.includes('path')){return '.';}"
        "if(s.includes('water')){return '~';}"
        "if(s.includes('tree')){return 'T';}"
        "if(s.includes('stone')||s.includes('ore')){return 'O';}"
        "if(s.includes('wall')){return '#';}"
        "if(s.includes('darkness')){return 'D';}"
        "if(s.includes('cow')||s.includes('skeleton')||s.includes('zombie')||s.includes('gnome')||s.includes('orc')||s.includes('troll')){return 'M';}"
        "return (tile||'?').charAt(0).toUpperCase();"
        "};"
        "const tileClass=(tile)=>{"
        "const s=(tile||'').toLowerCase();"
        "if(s.includes('player')){return 'player';}"
        "if(s.includes('cow')||s.includes('skeleton')||s.includes('zombie')||s.includes('gnome')||s.includes('orc')||s.includes('troll')){return 'enemy';}"
        "if(s.includes('water')){return 'water';}"
        "if(s.includes('path')){return 'path';}"
        "if(s.includes('tree')||s.includes('stone')||s.includes('ore')||s.includes('chest')||s.includes('gem')){return 'resource';}"
        "if(s.includes('wall')){return 'wall';}"
        "if(s.includes('darkness')){return 'darkness';}"
        "return '';"
        "};"
        "const esc=(x)=>String(x).replace(/[&<>\"']/g,(m)=>({ '&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;' }[m]));"
        "const rows=bounds.max_row-bounds.min_row+1;"
        "const cols=bounds.max_col-bounds.min_col+1;"
        "gridEl.style.gridTemplateColumns='repeat('+String(cols)+', 28px)';"
        "const render=(idx)=>{"
        "const s=states[idx]||states[0];"
        "slider.value=String(idx);"
        "idxEl.textContent=String(idx);"
        "tEl.textContent=String(s.t);"
        "const fp=String(s.frame_path||'');"
        "if(frameEl&&fp){"
        "frameEl.style.display='block';"
        "frameEl.setAttribute('src', encodeURI(fp));"
        "gridEl.style.display='none';"
        "}else{"
        "if(frameEl){frameEl.style.display='none';frameEl.removeAttribute('src');}"
        "gridEl.style.display='grid';"
        "const byCoord=new Map();"
        "for(const e of (s.map_entries||[])){byCoord.set(String(e.row)+','+String(e.col),e.tile||'');}"
        "let cells='';"
        "for(let r=bounds.min_row;r<=bounds.max_row;r+=1){"
        "for(let c=bounds.min_col;c<=bounds.max_col;c+=1){"
        "const key=String(r)+','+String(c);"
        "const tile=byCoord.get(key)||'';"
        "const sym=tileSymbol(tile);"
        "const cls=tileClass(tile);"
        "const title=tile?key+': '+tile:key+': (empty / not listed)';"
        "cells+='<div class=\"map-cell '+cls+'\" title=\"'+esc(title)+'\">'+esc(sym)+'</div>';"
        "}"
        "}"
        "gridEl.innerHTML=cells;"
        "}"
        "const metrics=["
        "'episode='+(s.episode_id===null||s.episode_id===undefined?'NA':String(s.episode_id)),"
        "'floor='+(s.floor===null||s.floor===undefined?'NA':String(s.floor)),"
        "'dir='+(s.direction||'NA'),"
        "'HP='+(s.health===null||s.health===undefined?'NA':String(s.health)),"
        "'Food='+(s.food===null||s.food===undefined?'NA':String(s.food)),"
        "'Drink='+(s.drink===null||s.drink===undefined?'NA':String(s.drink)),"
        "'Energy='+(s.energy===null||s.energy===undefined?'NA':String(s.energy)),"
        "'XP='+(s.xp===null||s.xp===undefined?'NA':String(s.xp)),"
        "'action='+(s.action_name||String(s.action_id||'NA')),"
        "'reward='+(s.reward===null||s.reward===undefined?'NA':String(s.reward)),"
        "'done='+(s.done===null||s.done===undefined?'NA':String(s.done))"
        "];"
        "metricsEl.textContent='';"
        "for(const m of metrics){const span=document.createElement('span');span.textContent=m;metricsEl.appendChild(span);}"
        "textEl.textContent=String(s.state_text||'');"
        "};"
        "slider.min='0';"
        "slider.max=String(states.length-1);"
        "slider.step='1';"
        "slider.addEventListener('input',()=>render(Number(slider.value||'0')));"
        "if(prevBtn){prevBtn.addEventListener('click',()=>{const n=Math.max(0,Number(slider.value||'0')-1);render(n);});}"
        "if(nextBtn){nextBtn.addEventListener('click',()=>{const n=Math.min(states.length-1,Number(slider.value||'0')+1);render(n);});}"
        "if(jumpEl){jumpEl.addEventListener('change',()=>{const t=Number(jumpEl.value);if(!Number.isFinite(t)||!tToIdx.has(t)){return;}render(tToIdx.get(t));});}"
        "render(0);"
        "})();"
        "</script>"
    )

    h.append("<h2>Per-Timestep Inspection</h2>")
    h.append("<p class='muted'>Use browser search (<code>t=...</code>, <code>run_id</code>, keywords) to jump quickly.</p>")

    for t in sorted(by_t.keys()):
        state_payload = selected_state_map.get(t, {})
        h.append("<details>")
        h.append(f"<summary><strong>t={t}</strong></summary>")
        h.append("<h3>Current State (Compact)</h3>")
        h.append(f"<pre>{html.escape(str(state_payload.get('compact_state', '')))}</pre>")
        for run_id in ordered_run_ids:
            rec = by_t[t].get(run_id)
            h.append("<div class='run-box'>")
            h.append(f"<h4><code>{html.escape(run_id)}</code></h4>")
            if run_desc.get(run_id):
                h.append(f"<div class='muted'>{html.escape(run_desc[run_id])}</div>")
            if rec is None:
                h.append("<div class='err'>missing</div>")
                h.append("</div>")
                continue
            status = str(rec.get("status", "unknown"))
            latency = rec.get("latency_s", "")
            h.append(f"<div class='muted'>status={html.escape(status)} | latency_s={html.escape(str(latency))}</div>")
            if status != "ok":
                h.append(f"<div class='err'>{html.escape(str(rec.get('error', '')))}</div>")
                h.append("</div>")
                continue
            h.append("<div><strong>Response</strong></div>")
            h.append(f"<pre>{html.escape(str(rec.get('response_text', '')))}</pre>")
            prompt_preview = str(rec.get("prompt_preview", ""))
            if prompt_preview:
                h.append("<details><summary>Prompt Preview</summary>")
                h.append(f"<pre>{html.escape(prompt_preview)}</pre>")
                h.append("</details>")
            h.append("</div>")
        h.append("</details>")

    h.append("</body></html>")
    output_path.write_text("\n".join(h), encoding="utf-8")


def _generate_reports(
    *,
    output_dir: Path,
    records_path: Path,
    selected_states_path: Path,
    trajectory_steps: List[TrajectoryStep],
    viewer_frame_dir: Optional[Path],
    run_specs: List[RunSpec],
    oracle_run_id: str,
    max_markdown_timesteps: int,
) -> Dict[str, Path]:
    records = _read_jsonl(records_path)
    selected_states = _read_jsonl(selected_states_path)
    selected_state_map: Dict[int, Dict[str, Any]] = {}
    for rec in selected_states:
        t = _safe_int(rec.get("t"))
        if t is None:
            continue
        selected_state_map[t] = rec

    summary_rows = _summarize_by_run(records, run_specs)
    pairwise_rows = _build_pairwise_scores(records, oracle_run_id=oracle_run_id) if oracle_run_id else []

    records_csv_rows: List[Dict[str, Any]] = []
    for rec in records:
        usage = rec.get("usage", {}) or {}
        records_csv_rows.append(
            {
                "t": rec.get("t"),
                "run_id": rec.get("run_id"),
                "role": rec.get("role"),
                "status": rec.get("status"),
                "latency_s": rec.get("latency_s"),
                "prompt_chars": rec.get("prompt_chars"),
                "response_chars": rec.get("response_chars"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "error": rec.get("error", ""),
                "response_preview": _truncate(str(rec.get("response_text", "")), 220).replace("\n", " "),
            }
        )

    records_csv_path = output_dir / "records.csv"
    summary_csv_path = output_dir / "summary_by_run.csv"
    pairwise_csv_path = output_dir / "pairwise_scores.csv"
    markdown_path = output_dir / "report.md"
    html_path = output_dir / "report.html"

    _write_csv(
        records_csv_path,
        records_csv_rows,
        [
            "t",
            "run_id",
            "role",
            "status",
            "latency_s",
            "prompt_chars",
            "response_chars",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "error",
            "response_preview",
        ],
    )
    _write_csv(
        summary_csv_path,
        summary_rows,
        [
            "run_id",
            "role",
            "description",
            "num_records",
            "num_ok",
            "num_errors",
            "latency_mean_s",
            "prompt_tokens_total",
            "completion_tokens_total",
        ],
    )
    _write_csv(
        pairwise_csv_path,
        pairwise_rows,
        [
            "t",
            "oracle_run_id",
            "run_id",
            "word_jaccard",
            "char_similarity",
            "oracle_chars",
            "prediction_chars",
        ],
    )
    _build_markdown_report(
        output_path=markdown_path,
        records=records,
        selected_state_map=selected_state_map,
        run_specs=run_specs,
        summary_rows=summary_rows,
        pairwise_rows=pairwise_rows,
        oracle_run_id=oracle_run_id,
        max_timesteps=max_markdown_timesteps,
    )
    _build_html_report(
        output_path=html_path,
        records=records,
        selected_state_map=selected_state_map,
        trajectory_steps=trajectory_steps,
        viewer_frame_dir=viewer_frame_dir,
        run_specs=run_specs,
        summary_rows=summary_rows,
        pairwise_rows=pairwise_rows,
        oracle_run_id=oracle_run_id,
    )
    return {
        "records_csv": records_csv_path,
        "summary_csv": summary_csv_path,
        "pairwise_csv": pairwise_csv_path,
        "report_md": markdown_path,
        "report_html": html_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory-dir", type=Path, required=True, help="Path to trajectory folder containing text_obs.jsonl")
    p.add_argument("--config", type=Path, required=True, help="JSON config path defining run_specs + selection")

    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root dir for new run folders")
    p.add_argument("--output-dir", type=Path, default=None, help="Use exact run directory (supports resume)")
    p.add_argument("--run-name", type=str, default="", help="Optional suffix for auto-created run dir name")

    p.add_argument("--provider", choices=["gemini", "openai_compatible", "hf_local"], default="gemini")
    p.add_argument("--model", type=str, required=True, help="Model id/name for the selected provider")
    p.add_argument("--base-url", type=str, default="", help="Provider base URL override")
    p.add_argument("--api-key", type=str, default="", help="API key override (prefer env var)")
    p.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY", help="Env var to read API key from if --api-key is empty")
    p.add_argument("--hf-device-map", type=str, default="auto", help="For provider=hf_local, device_map for transformers load")
    p.add_argument(
        "--hf-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="For provider=hf_local, torch dtype for model load",
    )
    p.add_argument(
        "--hf-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For provider=hf_local, pass trust_remote_code to transformers.",
    )
    p.add_argument(
        "--hf-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For provider=hf_local, request thinking-enabled chat template when supported.",
    )
    p.add_argument(
        "--hf-trim-to-headline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For provider=hf_local, strip preamble and keep response from first 'Headline:' when present.",
    )

    p.add_argument("--dry-run", action="store_true", help="Build prompts + logs but skip provider calls")
    p.add_argument("--estimate-only", action="store_true", help="Only estimate call volume + rough token usage; no calls")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip successful existing records in output_dir")
    p.add_argument("--retry-errors", action="store_true", help="With --resume, rerun records with prior status=error")
    p.add_argument("--max-calls", type=int, default=0, help="If >0, stop after this many new calls")

    p.add_argument("--request-timeout-s", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--min-request-interval-s", type=float, default=0.0, help="Sleep floor between requests")

    p.add_argument("--allow-placeholder-templates", action="store_true", help="Allow templates that still include [TODO]/<REPLACE...> sentinels")
    p.add_argument("--store-full-prompts", action="store_true", help="Write full prompt files to prompts/<run_id>/t_XXXXX.txt")
    p.add_argument("--prompt-preview-chars", type=int, default=1600, help="Prompt preview chars saved per record")

    p.add_argument("--strict-map-validation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--viewer-frame-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing pixel frames named t_00000.png for timeline viewer. "
            "If omitted, defaults to <trajectory-dir>/render_frames_bs16 when present."
        ),
    )

    # Selection overrides (optional)
    p.add_argument("--selection-start", type=int, default=None)
    p.add_argument("--selection-end", type=int, default=None)
    p.add_argument("--selection-stride", type=int, default=None)
    p.add_argument("--selection-max-states", type=int, default=None)

    p.add_argument("--max-markdown-timesteps", type=int, default=120)
    return p


def main() -> int:
    args = _build_parser().parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _load_json(config_path)
    run_specs = load_run_specs(
        config,
        config_path=config_path,
        allow_placeholder_templates=args.allow_placeholder_templates,
    )

    trajectory_dir = args.trajectory_dir.resolve()
    trajectory_id = trajectory_dir.name
    steps = load_trajectory_steps(
        trajectory_dir,
        strict_map_validation=args.strict_map_validation,
    )
    t_to_idx = {s.t: i for i, s in enumerate(steps)}

    selection_cfg = dict(config.get("selection", {}))
    if args.selection_start is not None:
        selection_cfg["start_t"] = args.selection_start
    if args.selection_end is not None:
        selection_cfg["end_t"] = args.selection_end
    if args.selection_stride is not None:
        selection_cfg["stride"] = args.selection_stride
    if args.selection_max_states is not None:
        selection_cfg["max_states"] = args.selection_max_states
    selected_timesteps = _select_timesteps(steps, selection_cfg)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{args.run_name}" if args.run_name else ""
        output_dir = (args.output_root / f"{ts}_{trajectory_id}{suffix}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records_path = output_dir / "records.jsonl"
    selected_states_path = output_dir / "selected_states.jsonl"
    estimate_path = output_dir / "estimate.json"
    resolved_config_path = output_dir / "resolved_run_config.json"

    selected_steps = [steps[t_to_idx[t]] for t in selected_timesteps]
    _write_selected_states(selected_states_path, selected_steps)

    viewer_frame_dir: Optional[Path] = None
    if args.viewer_frame_dir is not None:
        cand = args.viewer_frame_dir.resolve()
        if cand.exists():
            viewer_frame_dir = cand
    else:
        default_frame_dir = (trajectory_dir / "render_frames_bs16").resolve()
        if default_frame_dir.exists():
            viewer_frame_dir = default_frame_dir

    estimate = _estimate_workload(
        steps=steps,
        selected_timesteps=selected_timesteps,
        run_specs=run_specs,
    )
    estimate.update(
        {
            "trajectory_dir": str(trajectory_dir),
            "trajectory_id": trajectory_id,
            "selection": selection_cfg,
            "provider": args.provider,
            "model": args.model,
            "dry_run": bool(args.dry_run),
        }
    )
    estimate_path.write_text(json.dumps(estimate, indent=2, ensure_ascii=True), encoding="utf-8")

    resolved_payload = {
        "generated_at": datetime.now().isoformat(),
        "args": _sanitize_args_for_metadata(args),
        "trajectory_dir": str(trajectory_dir),
        "trajectory_id": trajectory_id,
        "num_total_steps": len(steps),
        "selection": selection_cfg,
        "selected_timesteps": selected_timesteps,
        "run_specs": [
            {
                "id": s.run_id,
                "role": s.role,
                "description": s.description,
                "template_path": str(s.template_path),
                "history_k": s.history_k,
                "history_format": s.history_format,
                "future_stride": s.future_stride,
                "future_max_states": s.future_max_states,
                "future_format": s.future_format,
                "future_event_max": s.future_event_max,
                "future_include_terminal": s.future_include_terminal,
                "generation": s.generation,
                "stop_sequences": s.stop_sequences,
            }
            for s in run_specs
        ],
        "estimate": estimate,
    }
    resolved_config_path.write_text(json.dumps(resolved_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    if args.estimate_only:
        print(f"[estimate-only] wrote {estimate_path}")
        print(json.dumps(estimate, indent=2, ensure_ascii=True))
        return 0

    existing_records = _read_jsonl(records_path) if args.resume else []
    existing_index = _records_index(existing_records)

    client: Optional[BaseClient] = None

    oracle_run_id = str((config.get("comparison", {}) or {}).get("oracle_run_id", ""))
    if not oracle_run_id:
        for spec in run_specs:
            if spec.role == "oracle":
                oracle_run_id = spec.run_id
                break

    new_calls = 0
    for t in selected_timesteps:
        step_idx = t_to_idx[t]
        step = steps[step_idx]
        for spec in run_specs:
            key = (t, spec.run_id)
            previous = existing_index.get(key)
            if previous is not None and args.resume:
                prev_status = str(previous.get("status", ""))
                if prev_status == "ok":
                    continue
                if prev_status != "ok" and not args.retry_errors:
                    continue

            vars_for_template, context_meta = _prepare_context(
                steps=steps,
                step_idx=step_idx,
                spec=spec,
                trajectory_id=trajectory_id,
            )
            prompt = _render_prompt(spec.template_text, vars_for_template, run_id=spec.run_id)
            prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            prompt_preview = _truncate(prompt, args.prompt_preview_chars)

            if args.store_full_prompts:
                prompt_dir = output_dir / "prompts" / spec.run_id
                prompt_dir.mkdir(parents=True, exist_ok=True)
                prompt_path = prompt_dir / f"t_{t:05d}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                prompt_path_value = str(prompt_path)
            else:
                prompt_path_value = ""

            started = time.perf_counter()
            if args.dry_run:
                provider_resp = {
                    "ok": True,
                    "response_text": "[dry-run: no model call]",
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    "raw_response": {},
                    "error": "",
                    "attempt": 0,
                    "request_s": 0.0,
                }
            else:
                if client is None:
                    client = _build_client(args)
                provider_resp = client.generate(
                    prompt=prompt,
                    generation=spec.generation,
                    stop_sequences=spec.stop_sequences,
                )
            latency_s = time.perf_counter() - started

            record = {
                "timestamp": datetime.now().isoformat(),
                "trajectory_id": trajectory_id,
                "t": t,
                "episode_id": step.episode_id,
                "run_id": spec.run_id,
                "role": spec.role,
                "description": spec.description,
                "provider": args.provider,
                "model": args.model,
                "status": ("ok" if provider_resp.get("ok") else "error"),
                "error": provider_resp.get("error", ""),
                "response_text": str(provider_resp.get("response_text", "")),
                "response_chars": len(str(provider_resp.get("response_text", ""))),
                "usage": provider_resp.get("usage", {}),
                "latency_s": round(float(latency_s), 6),
                "request_s": round(float(provider_resp.get("request_s", 0.0)), 6),
                "attempt": provider_resp.get("attempt", 0),
                "prompt_chars": len(prompt),
                "prompt_hash_sha1": prompt_hash,
                "prompt_preview": prompt_preview,
                "prompt_path": prompt_path_value,
                "context_meta": context_meta,
                "raw_response": provider_resp.get("raw_response", {}),
            }
            _append_jsonl(records_path, record)
            new_calls += 1

            if args.max_calls > 0 and new_calls >= args.max_calls:
                break
        if args.max_calls > 0 and new_calls >= args.max_calls:
            break

    artifacts = _generate_reports(
        output_dir=output_dir,
        records_path=records_path,
        selected_states_path=selected_states_path,
        trajectory_steps=steps,
        viewer_frame_dir=viewer_frame_dir,
        run_specs=run_specs,
        oracle_run_id=oracle_run_id,
        max_markdown_timesteps=args.max_markdown_timesteps,
    )

    print(f"output_dir: {output_dir}")
    print(f"records: {records_path}")
    print(f"selected_states: {selected_states_path}")
    print(f"viewer_frame_dir: {viewer_frame_dir if viewer_frame_dir is not None else ''}")
    print(f"estimate: {estimate_path}")
    print(f"resolved_config: {resolved_config_path}")
    for key, value in artifacts.items():
        print(f"{key}: {value}")
    print(f"new_calls: {new_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
