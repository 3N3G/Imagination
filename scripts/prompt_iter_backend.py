#!/usr/bin/env python3
"""Backend utilities for prompt iteration against the online Craftax Qwen/vLLM stack."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from llm.prompts import (
    MAP_INTERESTING_PREFIX,
    build_user_prompt_content,
    ensure_valid_interesting_map,
    filter_text_obs,
    get_prompt_sections,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "prompt_iter" / "fixed_states_v1.json"
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_VLLM_MODEL = "./configs/vllm_hidden_qwen4b"
DEFAULT_VLLM_URL = os.getenv("PROMPT_ITER_VLLM_URL", "http://127.0.0.1:8000")
_COORD_ENTRY_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:[^,]+")
_COORD_PREFIX_RE = re.compile(r"-?\d+\s*,\s*-?\d+\s*:")


@dataclass
class PromptSections:
    system_prompt: str
    few_shot_examples: str
    task_instruction: str
    generation_prefix: str
    stop_sequences: List[str]


@dataclass
class FixedState:
    state_id: str
    label: str
    source_kind: str
    source_path: str
    tags: List[str]
    raw_text_obs: str
    filtered_text_obs: str
    t: Optional[int] = None
    episode_id: Optional[int] = None

    def map_line(self) -> str:
        for line in self.filtered_text_obs.splitlines():
            if line.startswith(MAP_INTERESTING_PREFIX):
                return line
        return ""

    def map_preview(self, max_entries: int = 8) -> str:
        line = self.map_line()
        if not line:
            return ""
        payload = line[len(MAP_INTERESTING_PREFIX):].strip()
        if not payload:
            return ""
        starts = list(_COORD_PREFIX_RE.finditer(payload))
        entries = []
        for i, match in enumerate(starts):
            start = match.start()
            end = starts[i + 1].start() if i + 1 < len(starts) else len(payload)
            token = payload[start:end].strip().rstrip(",").strip()
            if token:
                entries.append(token)
        if not entries:
            entries = [m.group(0).strip() for m in _COORD_ENTRY_RE.finditer(payload)]
        if len(entries) <= max_entries:
            return ", ".join(entries)
        return ", ".join(entries[:max_entries]) + ", ..."


@lru_cache(maxsize=4)
def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def default_prompt_sections(prompt_variant: str = "default") -> PromptSections:
    sections = get_prompt_sections(prompt_variant=prompt_variant)
    return PromptSections(
        system_prompt=str(sections["system_prompt"]),
        few_shot_examples=str(sections["few_shot_examples"]),
        task_instruction=str(sections["task_instruction"]),
        generation_prefix=str(sections["generation_prefix"]),
        stop_sequences=[str(s) for s in sections["stop_sequences"]],
    )


def build_messages(
    filtered_text_obs: str,
    sections: PromptSections,
) -> List[Dict[str, str]]:
    """Build canonical chat messages for the current state."""
    return [
        {"role": "system", "content": sections.system_prompt},
        {
            "role": "user",
            "content": build_user_prompt_content(
                text_obs=filtered_text_obs,
                few_shot_examples=sections.few_shot_examples,
                task_instruction=sections.task_instruction,
            ),
        },
    ]


def build_prompt(
    filtered_text_obs: str,
    sections: PromptSections,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    tokenizer = _load_tokenizer(model_id)
    messages = build_messages(filtered_text_obs, sections)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if sections.generation_prefix:
        prompt += sections.generation_prefix
    return prompt


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_golden_raw_text(entry: Dict[str, Any], root: Path) -> str:
    source = root / str(entry["source_path"])
    obs_variant = str(entry.get("obs_variant", "before")).strip().lower()
    if obs_variant not in {"before", "after"}:
        raise ValueError(f"golden_bundle obs_variant must be before|after, got {obs_variant!r}")

    raw_path = source / f"{obs_variant}_state_raw.txt"
    if raw_path.exists():
        return _read_text(raw_path)

    npy_path = source / f"obs_{obs_variant}.npy"
    if npy_path.exists():
        import numpy as np
        from labelling.obs_to_text import obs_to_text

        obs = np.load(npy_path)
        return obs_to_text(obs)

    raise FileNotFoundError(f"Could not resolve golden raw state text for {source}")


def _resolve_jsonl_state(entry: Dict[str, Any], root: Path) -> tuple[str, Optional[int], Optional[int]]:
    source = root / str(entry["source_path"])
    target_t = int(entry["t"])
    target_episode = entry.get("episode_id")

    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("t", -1)) != target_t:
                continue
            if target_episode is not None and int(record.get("episode_id", -1)) != int(target_episode):
                continue
            raw_text = str(record.get("raw_text_obs", ""))
            if not raw_text:
                raise ValueError(f"Missing raw_text_obs in {source} at t={target_t}")
            return raw_text, record.get("t"), record.get("episode_id")

    raise KeyError(f"No state found in {source} for t={target_t}, episode_id={target_episode}")


def _build_fixed_state(entry: Dict[str, Any], root: Path) -> FixedState:
    source_kind = str(entry["source_kind"]).strip()
    state_id = str(entry["id"])
    label = str(entry.get("label", state_id))
    source_path = str(entry["source_path"])
    tags = [str(tag) for tag in entry.get("tags", [])]

    if source_kind == "golden_bundle":
        raw_text = _resolve_golden_raw_text(entry, root)
        t_val = None
        episode_id_val = None
    elif source_kind == "trajectory_jsonl":
        raw_text, t_val, episode_id_val = _resolve_jsonl_state(entry, root)
    else:
        raise ValueError(f"Unknown source_kind={source_kind!r} in fixed state manifest")

    filtered_text = filter_text_obs(raw_text, strict_map_validation=True)

    ensure_valid_interesting_map(filtered_text)

    return FixedState(
        state_id=state_id,
        label=label,
        source_kind=source_kind,
        source_path=source_path,
        tags=tags,
        raw_text_obs=raw_text,
        filtered_text_obs=filtered_text,
        t=(int(t_val) if t_val is not None else None),
        episode_id=(int(episode_id_val) if episode_id_val is not None else None),
    )


def load_fixed_states(manifest_path: Path | str = DEFAULT_MANIFEST, root: Path = REPO_ROOT) -> List[FixedState]:
    manifest = _read_json(Path(manifest_path))
    states = manifest.get("states", [])
    if not isinstance(states, list) or not states:
        raise ValueError(f"No states listed in {manifest_path}")
    fixed_states = [_build_fixed_state(entry, root) for entry in states]
    if len(fixed_states) != 10:
        raise ValueError(f"Expected exactly 10 fixed states, found {len(fixed_states)}")

    seen = set()
    for state in fixed_states:
        if state.state_id in seen:
            raise ValueError(f"Duplicate state id: {state.state_id}")
        seen.add(state.state_id)
    return fixed_states


def states_by_id(states: Iterable[FixedState]) -> Dict[str, FixedState]:
    return {state.state_id: state for state in states}


def run_completion(
    prompt: str,
    server_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_VLLM_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.7,
    stop_sequences: Optional[List[str]] = None,
    timeout: tuple[int, int] = (10, 300),
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if stop_sequences:
        cleaned = [s for s in stop_sequences if s]
        if cleaned:
            payload["stop"] = cleaned

    resp = requests.post(
        f"{server_url.rstrip('/')}/v1/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def run_chat_completion(
    messages: List[Dict[str, str]],
    server_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_VLLM_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.7,
    stop_sequences: Optional[List[str]] = None,
    timeout: tuple[int, int] = (10, 300),
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if stop_sequences:
        cleaned = [s for s in stop_sequences if s]
        if cleaned:
            payload["stop"] = cleaned

    resp = requests.post(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def check_vllm_health(
    server_url: str = DEFAULT_VLLM_URL,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """Return health metadata for a vLLM endpoint."""
    url = f"{server_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        return {
            "ok": resp.status_code == 200,
            "status_code": int(resp.status_code),
            "url": url,
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "url": url,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def run_state(
    state: FixedState,
    sections: PromptSections,
    *,
    server_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_VLLM_MODEL,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 256,
    temperature: float = 0.7,
    stop_sequences: Optional[List[str]] = None,
    prefer_chat_completions: bool = False,
) -> Dict[str, Any]:
    messages = build_messages(state.filtered_text_obs, sections)
    prompt = ""
    request_mode = "completions"
    response_json: Dict[str, Any]

    t0 = time.perf_counter()
    if prefer_chat_completions:
        request_mode = "chat_completions"
        prompt = json.dumps(messages, indent=2, ensure_ascii=True)
        if sections.generation_prefix:
            prompt += "\n\n# generation_prefix\n" + sections.generation_prefix
        response_json = run_chat_completion(
            messages,
            server_url=server_url,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
    else:
        try:
            prompt = build_prompt(state.filtered_text_obs, sections=sections, model_id=model_id)
            response_json = run_completion(
                prompt,
                server_url=server_url,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )
        except ModuleNotFoundError as exc:
            # Fallback mode: run via chat completions so local transformers is not required.
            request_mode = "chat_completions_fallback"
            prompt = json.dumps(messages, indent=2, ensure_ascii=True)
            if sections.generation_prefix:
                prompt += "\n\n# generation_prefix\n" + sections.generation_prefix
            response_json = run_chat_completion(
                messages,
                server_url=server_url,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )
            response_json["_prompt_iter_note"] = (
                "Fell back to /v1/chat/completions because local transformers is unavailable: "
                f"{exc}"
            )
    elapsed = time.perf_counter() - t0

    response_text = ""
    choices = response_json.get("choices", [])
    if choices:
        first = choices[0]
        if "text" in first:
            response_text = str(first.get("text", ""))
        elif isinstance(first.get("message"), dict):
            response_text = str(first.get("message", {}).get("content", ""))

    return {
        "state_id": state.state_id,
        "label": state.label,
        "source_kind": state.source_kind,
        "source_path": state.source_path,
        "t": state.t,
        "episode_id": state.episode_id,
        "tags": list(state.tags),
        "filtered_text_obs": state.filtered_text_obs,
        "prompt": prompt,
        "request_mode": request_mode,
        "response_text": response_text,
        "response_json": response_json,
        "latency_s": elapsed,
    }


def run_batch(
    states: Iterable[FixedState],
    sections: PromptSections,
    *,
    server_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_VLLM_MODEL,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 256,
    temperature: float = 0.7,
    stop_sequences: Optional[List[str]] = None,
    prefer_chat_completions: bool = False,
) -> List[Dict[str, Any]]:
    results = []
    for state in states:
        results.append(
            run_state(
                state,
                sections,
                server_url=server_url,
                model_name=model_name,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                prefer_chat_completions=prefer_chat_completions,
            )
        )
    return results


def list_state_examples(states: Iterable[FixedState], max_examples: int = 4) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for idx, state in enumerate(states):
        if idx >= max_examples:
            break
        out.append(
            {
                "state_id": state.state_id,
                "label": state.label,
                "source_kind": state.source_kind,
                "map_preview": state.map_preview(10),
            }
        )
    return out


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt iteration backend utilities")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--list-states", action="store_true")
    parser.add_argument("--show-examples", action="store_true")
    parser.add_argument("--state-id", type=str, default=None)
    parser.add_argument("--prompt-variant", type=str, default="default")
    parser.add_argument("--server-url", type=str, default=DEFAULT_VLLM_URL)
    parser.add_argument("--model-name", type=str, default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--prefer-chat-completions", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()

    states = load_fixed_states(args.manifest)
    state_map = states_by_id(states)

    if args.list_states:
        for state in states:
            t_str = "" if state.t is None else f" t={state.t}"
            print(f"{state.state_id}\t{state.label}\t{state.source_kind}{t_str}")

    if args.show_examples:
        examples = list_state_examples(states, max_examples=10)
        print(json.dumps(examples, indent=2, ensure_ascii=True))

    if args.state_id:
        if args.state_id not in state_map:
            raise KeyError(f"Unknown state-id {args.state_id!r}")
        sections = default_prompt_sections(args.prompt_variant)
        if args.stop:
            sections.stop_sequences = [str(s) for s in args.stop]
        result = run_state(
            state_map[args.state_id],
            sections,
            server_url=args.server_url,
            model_name=args.model_name,
            model_id=args.model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop_sequences=sections.stop_sequences,
            prefer_chat_completions=bool(args.prefer_chat_completions),
        )
        output = {
            "state_id": result["state_id"],
            "label": result["label"],
            "latency_s": result["latency_s"],
            "map_preview": state_map[args.state_id].map_preview(10),
            "response_text": result["response_text"],
            "prompt": result["prompt"],
            "response_json": result["response_json"],
        }
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
            print(f"Wrote {args.output_json}")
        else:
            print(json.dumps(output, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
