#!/usr/bin/env python3
"""
Phase 4: Generate Gemini oracle labels for filtered trajectory data.

Reads bitpacked trajectory files from filtered_trajectories/, decodes obs,
finds episode boundaries via done flags, and calls Gemini every 15 steps
within each episode.

Fully resumable: checks which (file, sample_idx) pairs are already done.

Usage:
    python -m pipeline.gemini_label [--api-key KEY] [--max-calls N]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from urllib import error as urlerror
from urllib import request as urlrequest

from pipeline.config import (
    FILTERED_DIR,
    GEMINI_BASE_URL,
    GEMINI_CONCURRENT_REQUESTS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_OUTPUT_DIR,
    GEMINI_REQUESTS_PER_MINUTE,
    GEMINI_STEP_CADENCE,
    GEMINI_TEMPERATURE,
    MAP_OBS_DIM,
    ORACLE_TEMPLATE_PATH,
    PIPELINE_ROOT,
)
from pipeline.text_utils import (
    build_future_state_block,
    build_history_block,
    filter_text_obs,
    obs_to_text,
)


def load_template() -> str:
    if not ORACLE_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Oracle template not found at {ORACLE_TEMPLATE_PATH}. "
            f"Check that Craftax_Baselines is at ~/Craftax_Baselines/"
        )
    return ORACLE_TEMPLATE_PATH.read_text()


def load_completed_indices(output_path: Path) -> set:
    """Load sample_idx values already completed for a given trajectory file."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["sample_idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def decode_obs_from_bitpacked(data) -> np.ndarray:
    """Decode bitpacked NPZ data to float32 obs array (N, 8268).

    Matches the format produced by filter_and_repack.py and consumed by
    decode_obs_array() in awr_llm_augmented.py.
    """
    map_dim = int(data["obs_map_dim"]) if "obs_map_dim" in data.files else MAP_OBS_DIM
    map_bits = np.asarray(data["obs_map_bits"])
    obs_map = np.unpackbits(
        map_bits, axis=1, count=map_dim, bitorder="little"
    ).astype(np.float32, copy=False)
    obs_aux = np.asarray(data["obs_aux"], dtype=np.float32)
    return np.concatenate([obs_map, obs_aux], axis=1)


def find_episodes(done: np.ndarray) -> List[tuple]:
    """Find (start, end_exclusive) indices of complete episodes.

    Episodes are delimited by done=True. Each episode includes the done step.
    """
    done_indices = np.where(done)[0]
    episodes = []
    start = 0
    for di in done_indices:
        episodes.append((start, di + 1))
        start = di + 1
    return episodes


def identify_gemini_calls(
    episodes: List[tuple],
    cadence: int = GEMINI_STEP_CADENCE,
) -> List[Dict]:
    """For each episode, identify Gemini call timesteps (every `cadence` steps).

    Returns list of dicts with: sample_idx, episode_start, within_ep_step,
    future_end_idx, n_future_steps.
    """
    calls = []
    for ep_start, ep_end in episodes:
        ep_len = ep_end - ep_start
        for step_in_ep in range(0, ep_len, cadence):
            sample_idx = ep_start + step_in_ep
            # Future window: up to cadence steps ahead, capped at episode end
            future_end = min(sample_idx + cadence, ep_end - 1)
            n_future = future_end - sample_idx
            calls.append({
                "sample_idx": sample_idx,
                "episode_start": ep_start,
                "within_ep_step": step_in_ep,
                "future_end_idx": future_end,
                "n_future_steps": n_future,
            })
    return calls


# ---------------------------------------------------------------------------
# Gemini API (unchanged from original)
# ---------------------------------------------------------------------------

def call_gemini(
    prompt: str,
    api_key: str,
    model: str = GEMINI_MODEL,
    base_url: str = GEMINI_BASE_URL,
    max_output_tokens: int = GEMINI_MAX_OUTPUT_TOKENS,
    temperature: float = GEMINI_TEMPERATURE,
    max_retries: int = 4,
    timeout_s: float = 120.0,
    use_thinking: bool = True,
) -> Dict[str, Any]:
    """Single synchronous Gemini API call with exponential backoff."""
    url = f"{base_url}/{model}:generateContent?key={api_key}"
    gen_config: Dict[str, Any] = {
        "maxOutputTokens": max_output_tokens,
        "temperature": temperature,
    }
    if use_thinking:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    backoff_s = 1.0

    for attempt in range(max_retries + 1):
        req = urlrequest.Request(url, data=body, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            candidates = parsed.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            usage = parsed.get("usageMetadata", {})
            return {
                "ok": True,
                "text": text,
                "prompt_tokens": usage.get("promptTokenCount"),
                "completion_tokens": usage.get("candidatesTokenCount"),
                "attempt": attempt,
                "latency_s": time.perf_counter() - t0,
            }
        except urlerror.HTTPError as exc:
            status = int(exc.code)
            if status in {408, 429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 60.0)
                continue
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                err_body = str(exc)
            return {"ok": False, "error": f"HTTP {status}: {err_body[:500]}",
                    "attempt": attempt, "latency_s": time.perf_counter() - t0}
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 60.0)
                continue
            return {"ok": False, "error": str(exc)[:500],
                    "attempt": attempt, "latency_s": time.perf_counter() - t0}


def build_prompt(
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    call_info: Dict,
    template: str,
    predict_only: bool = False,
    history_steps: int = 0,
) -> Dict:
    """Build a Gemini prompt for one call.

    Args:
        obs: full float32 obs array (N, 8268) for this file
        action, reward, done: corresponding arrays
        call_info: dict with sample_idx, future_end_idx, n_future_steps, episode_start
        template: oracle prompt template string
        history_steps: number of prior timesteps to include as history (0 = none)
    """
    idx = call_info["sample_idx"]
    end_idx = call_info["future_end_idx"]
    ep_start = call_info.get("episode_start", 0)

    try:
        # Current state (t+0) — use filtered text (not compact) for concise prompt
        raw = obs_to_text(obs[idx])
        filtered = filter_text_obs(raw)

        prompt = template.replace("{current_state_filtered}", filtered)

        # History block (t-N through t-1)
        if history_steps > 0 and "{history_block}" in template:
            hist_start = max(ep_start, idx - history_steps)
            if hist_start < idx:
                hist_obs = [obs[t] for t in range(hist_start, idx)]
                hist_act = action[hist_start:idx]
                hist_rew = reward[hist_start:idx]
                hist_done = done[hist_start:idx]
                history_block = build_history_block(
                    hist_obs, hist_act, hist_rew, hist_done,
                    n_history=history_steps,
                )
            else:
                history_block = "(no history — episode start)"
            prompt = prompt.replace("{history_block}", history_block)

        if not predict_only:
            # Future states (t+1..t+N) — only for oracle mode
            if end_idx > idx:
                future_obs = [obs[t] for t in range(idx + 1, end_idx + 1)]
                future_act = action[idx + 1 : end_idx + 1]
                future_rew = reward[idx + 1 : end_idx + 1]
                future_done = done[idx + 1 : end_idx + 1]
                future_block = build_future_state_block(
                    future_obs, future_act, future_rew, future_done,
                    base_t_offset=1,
                )
            else:
                future_block = "(no future states — episode ended)"
            prompt = prompt.replace("{future_state_block}", future_block)

        return {
            "sample_idx": idx,
            "n_future_steps": call_info["n_future_steps"],
            "prompt": prompt,
        }
    except Exception as e:
        return {"sample_idx": idx, "error": str(e)}


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket rate limiter for concurrent Gemini calls."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.perf_counter()
            wait = self._interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.perf_counter()


def _worker(
    call_info: Dict,
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    done_f32: np.ndarray,
    template: str,
    api_key: str,
    limiter: _RateLimiter,
    gemini_model: str = GEMINI_MODEL,
    predict_only: bool = False,
    use_thinking: bool = True,
    history_steps: int = 0,
) -> Dict:
    """Build prompt and call Gemini for one sample (runs in thread pool)."""
    pr = build_prompt(obs, action, reward, done_f32, call_info, template,
                      predict_only=predict_only, history_steps=history_steps)
    if "error" in pr:
        return {"sample_idx": int(call_info["sample_idx"]),
                "ok": False, "error": f"prompt_build: {pr['error']}"}

    limiter.wait()
    result = call_gemini(pr["prompt"], api_key, model=gemini_model,
                         use_thinking=use_thinking)

    record = {
        "sample_idx": int(call_info["sample_idx"]),
        "n_future_steps": int(pr["n_future_steps"]),
        "ok": result["ok"],
    }
    if result["ok"]:
        record["text"] = result["text"]
        record["prompt_tokens"] = result.get("prompt_tokens")
        record["completion_tokens"] = result.get("completion_tokens")
        record["latency_s"] = result.get("latency_s")
    else:
        record["error"] = result.get("error", "unknown")
    return record


def process_file(
    traj_path: Path,
    template: str,
    api_key: str,
    max_calls: Optional[int] = None,
    output_dir: Optional[Path] = None,
    gemini_model: str = GEMINI_MODEL,
    predict_only: bool = False,
    use_thinking: bool = True,
    history_steps: int = 0,
) -> Dict:
    """Process all Gemini calls for one filtered trajectory file (concurrent)."""
    _output_dir = output_dir or GEMINI_OUTPUT_DIR
    fname = traj_path.stem
    output_path = _output_dir / f"{fname}.jsonl"
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Load file
    try:
        data = np.load(traj_path, allow_pickle=False)
        obs = decode_obs_from_bitpacked(data)
        action = np.asarray(data["action"]).reshape(-1)
        reward = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
        done_arr = np.asarray(data["done"]).reshape(-1).astype(bool)
        data.close()
    except Exception as e:
        print(f"  ERROR loading {fname}: {e}")
        return {"file": fname, "total": 0, "processed": 0, "errors": 1}

    # Find episodes and Gemini call points
    episodes = find_episodes(done_arr)
    gemini_calls = identify_gemini_calls(episodes)

    # Resume
    done_indices = load_completed_indices(output_path)
    pending = [c for c in gemini_calls if c["sample_idx"] not in done_indices]
    if max_calls is not None:
        pending = pending[:max_calls]

    if not pending:
        return {"file": fname, "total": len(gemini_calls),
                "already_done": len(done_indices), "processed": 0, "errors": 0}

    print(f"  {fname}: {len(pending)} pending "
          f"({len(done_indices)} done / {len(gemini_calls)} total)")

    limiter = _RateLimiter(GEMINI_REQUESTS_PER_MINUTE)
    done_f32 = done_arr.astype(np.float32)
    processed = 0
    errors = 0
    write_lock = threading.Lock()

    with open(output_path, "a") as out_f, \
         ThreadPoolExecutor(max_workers=GEMINI_CONCURRENT_REQUESTS) as pool:
        futures = {
            pool.submit(
                _worker, ci, obs, action, reward, done_f32,
                template, api_key, limiter,
                gemini_model=gemini_model,
                predict_only=predict_only,
                use_thinking=use_thinking,
                history_steps=history_steps,
            ): ci
            for ci in pending
        }

        for future in as_completed(futures):
            record = future.result()
            with write_lock:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

            if not record["ok"]:
                errors += 1
            processed += 1

            if processed % 100 == 0 or processed == len(pending):
                print(f"    [{processed}/{len(pending)}] "
                      f"{'OK' if record['ok'] else 'FAIL'} "
                      f"latency={record.get('latency_s', 0):.1f}s")

    del obs, action, reward, done_arr, done_f32

    return {"file": fname, "total": len(gemini_calls),
            "already_done": len(done_indices),
            "processed": processed, "errors": errors}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    api_key: str,
    max_calls_per_file: Optional[int] = None,
    max_files: Optional[int] = None,
    filtered_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    gemini_model: Optional[str] = None,
    template_path: Optional[str] = None,
    predict_only: bool = False,
    history_steps: int = 0,
):
    """Main Gemini labelling entry point."""
    _filtered_dir = Path(filtered_dir) if filtered_dir else FILTERED_DIR
    _output_dir = Path(output_dir) if output_dir else GEMINI_OUTPUT_DIR
    _gemini_model = gemini_model or GEMINI_MODEL
    _use_thinking = _gemini_model.startswith("gemini-2.5")

    if template_path:
        _template_path = Path(template_path)
    elif history_steps > 0:
        _template_path = (
            PIPELINE_ROOT / "configs" / "training" / "templates"
            / "predict_history_k_prompt_concise.txt"
        )
    elif predict_only:
        from pipeline.config import CRAFTAX_BASELINES
        _template_path = (
            CRAFTAX_BASELINES / "configs" / "future_imagination"
            / "templates" / "predict_state_only_prompt_concise.txt"
        )
    else:
        _template_path = ORACLE_TEMPLATE_PATH

    if history_steps > 0:
        mode_str = f"predict+history-{history_steps}"
    elif predict_only:
        mode_str = "predict-state-only"
    else:
        mode_str = "oracle"
    print("=" * 70)
    print(f"PHASE 4: Generate Gemini labels ({mode_str})")
    print("=" * 70)

    if not _template_path.exists():
        raise FileNotFoundError(f"Template not found: {_template_path}")
    template = _template_path.read_text()

    # Discover filtered trajectory files
    files = sorted(_filtered_dir.glob("trajectories_*.npz"))
    if not files:
        print(f"  No trajectory files found in {_filtered_dir}")
        return
    if max_files:
        files = files[:max_files]

    # Quick scan to count total calls
    total_calls = 0
    for fpath in files:
        data = np.load(fpath, allow_pickle=False)
        done_arr = np.asarray(data["done"]).reshape(-1).astype(bool)
        data.close()
        episodes = find_episodes(done_arr)
        total_calls += len(identify_gemini_calls(episodes))

    print(f"  Files to process: {len(files)}")
    print(f"  Total Gemini calls: {total_calls:,}")
    print(f"  Model: {_gemini_model}")
    print(f"  Template: {_template_path.name}")
    print(f"  Mode: {mode_str}")
    print(f"  History steps: {history_steps}")
    print(f"  Use thinking: {_use_thinking}")
    print(f"  Rate limit: {GEMINI_REQUESTS_PER_MINUTE} req/min")
    est_input_tokens = 2300 + history_steps * 540
    est_cost = total_calls * (est_input_tokens * 0.15e-6 + 212 * 0.60e-6)
    print(f"  Estimated cost: ${est_cost:.2f} (at ~{est_input_tokens} input tok/call)")
    print()

    total_processed = 0
    total_errors = 0
    t0 = time.time()

    for i, fpath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {fpath.name}")
        result = process_file(
            fpath, template, api_key,
            max_calls=max_calls_per_file,
            output_dir=_output_dir,
            gemini_model=_gemini_model,
            predict_only=predict_only or history_steps > 0,
            use_thinking=_use_thinking,
            history_steps=history_steps,
        )
        total_processed += result["processed"]
        total_errors += result["errors"]

        elapsed = time.time() - t0
        if total_processed > 0:
            rate = total_processed / elapsed * 60
            remaining = total_calls - total_processed
            eta_min = remaining / rate if rate > 0 else 0
            print(f"  Progress: {total_processed} done, {total_errors} errors, "
                  f"{rate:.0f} calls/min, ETA ~{eta_min:.0f}min")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Gemini labelling complete in {elapsed/60:.1f}min")
    print(f"  Processed: {total_processed}, Errors: {total_errors}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Generate Gemini oracle labels")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--max-calls-per-file", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--filtered-dir", type=str, default=None,
                        help="Override input directory (default: config FILTERED_DIR)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: config GEMINI_OUTPUT_DIR)")
    parser.add_argument("--gemini-model", type=str, default=None,
                        help="Override Gemini model (default: config GEMINI_MODEL)")
    parser.add_argument("--template-path", type=str, default=None,
                        help="Override prompt template path")
    parser.add_argument("--predict-only", action="store_true",
                        help="Predict-state-only mode (no future block)")
    parser.add_argument("--history-steps", type=int, default=0,
                        help="Number of prior timesteps to include as history context (0=none)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: Gemini API key required. Pass --api-key or set GEMINI_API_KEY.")
        return

    run(api_key,
        max_calls_per_file=args.max_calls_per_file,
        max_files=args.max_files,
        filtered_dir=args.filtered_dir,
        output_dir=args.output_dir,
        gemini_model=args.gemini_model,
        template_path=args.template_path,
        predict_only=args.predict_only,
        history_steps=args.history_steps)


if __name__ == "__main__":
    main()
