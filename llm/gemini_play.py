"""Gemini plays Craftax directly.

At each step, sends the current observation + a short action-history to
Gemini, asks for the next action (0-42), parses it, steps the env.

The system prompt is an ALGORITHM-first strategy guide (survive →
ladder → upgrades → explore) derived from the user's rulebook.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Tuple

import jax
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Action
from craftax.craftax.renderer import render_craftax_pixels

from labelling.obs_to_text import obs_to_text
from llm.prompts import filter_text_obs
from pipeline.gemini_label import call_gemini
from pipeline.config import GEMINI_MODEL

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import wandb
except ImportError:
    wandb = None


ACTION_DIM = 43
ACTION_NAMES = [a.name for a in Action]
# Parse "ACTION: <NAME>" — case-insensitive; matches one all-caps-ish token.
ACTION_NAME_RE = re.compile(r'ACTION\s*:\s*\*{0,2}([A-Za-z_]+)\*{0,2}', re.IGNORECASE)

# Map of accepted aliases onto canonical ACTION_NAMES. Enum names are the
# truth; we add a few short aliases the model might emit.
_ALIASES = {
    "LEVEL_UP_STR": "LEVEL_UP_STRENGTH",
    "LEVEL_UP_DEX": "LEVEL_UP_DEXTERITY",
    "LEVEL_UP_INT": "LEVEL_UP_INTELLIGENCE",
}


# Single source of truth for the action-selection prompt — see llm/gameplay.py
# (also used by the prompt-iteration webapp).
from llm.gameplay import ACTION_SELECT_PROMPT as SYSTEM_PROMPT


def summarize_state(filt: str) -> str:
    """Extract one-line compact stats from filtered text obs."""
    parts = []
    for pat, lbl in [
        (r'Health[:\s]*([\d.]+)', 'HP'),
        (r'Food[:\s]*([\d.]+)', 'Food'),
        (r'Drink[:\s]*([\d.]+)', 'Drink'),
        (r'Energy[:\s]*([\d.]+)', 'Energy'),
        (r'Floor[:\s]*(\d+)', 'Floor'),
        (r'Direction[:\s]*(\w+)', 'Dir'),
    ]:
        m = re.search(pat, filt, re.IGNORECASE)
        if m:
            parts.append(f"{lbl}={m.group(1)}")
    return ", ".join(parts) if parts else "?"


def build_prompt(filtered_obs: str, _history_unused=None) -> str:
    """One-shot prompt: fill the single `{current_state_filtered}` slot in
    SYSTEM_PROMPT. history is ignored — kept as a positional for callers."""
    return SYSTEM_PROMPT.replace("{current_state_filtered}", filtered_obs)


def parse_action(text: str) -> Tuple[int, bool, str]:
    """Parse `ACTION: <NAME>` (last occurrence). Returns (id, ok, raw_match)."""
    matches = list(ACTION_NAME_RE.finditer(text))
    if not matches:
        return 0, False, "[no match]"
    raw = matches[-1].group(1).upper()
    name = _ALIASES.get(raw, raw)
    if name in ACTION_NAMES:
        return ACTION_NAMES.index(name), True, matches[-1].group(0)
    return 0, False, matches[-1].group(0)


def make_frame(env_state, step, action_name, response_text, ep_return,
               raw_match: str = ""):
    """Render a frame with the game + a text overlay showing Gemini's reply."""
    pixels = np.array(render_craftax_pixels(env_state, block_pixel_size=16,
                                             do_night_noise=False), dtype=np.uint8)
    # Upscale game
    target_w = 600
    h, w = pixels.shape[:2]
    scale = target_w / w
    new_h = int(h * scale)
    pixels = cv2.resize(pixels, (target_w, new_h), interpolation=cv2.INTER_NEAREST)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 14
    max_chars = 90
    header = f"step={step}  ret={ep_return:+.2f}  extracted=a={action_name}"
    if raw_match:
        header += f"  | raw={raw_match!r}"
    lines = [header]
    if response_text:
        for raw in response_text.splitlines():
            while len(raw) > max_chars:
                lines.append(raw[:max_chars]); raw = raw[max_chars:]
            lines.append(raw)
    lines = lines[:6]
    overlay_h = 10 + len(lines) * line_h + 10
    overlay = np.zeros((overlay_h, target_w, 3), dtype=np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(overlay, ln, (6, 18 + i * line_h), font, 0.38,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([pixels, overlay])


def run_episode(env, env_params, rng, api_key: str, *,
                history_len: int, max_steps: int,
                model: str, verbose: bool,
                record_video: bool = False) -> dict:
    rng, reset_key = jax.random.split(rng)
    obs, env_state = env.reset(reset_key, env_params)

    ep_return = 0.0
    history: List[dict] = []
    traj = []
    prompts_log: List[dict] = []  # full {step, prompt, response} per Gemini call
    frames = []
    parse_fail = 0
    api_fail = 0
    n_calls = 0
    step_latencies = []

    done = False
    step = 0
    while not done and step < max_steps:
        obs_np = np.array(obs, dtype=np.float32)
        text_obs = obs_to_text(obs_np)
        filt = filter_text_obs(text_obs)

        # Skip Gemini entirely while the player is asleep — the env forces a
        # fixed NOOP-like behaviour (sleep auto-continues) and a model call is
        # both wasted cost and misleading for the action log.
        is_sleeping_now = bool(getattr(env_state, "is_sleeping", False))
        if is_sleeping_now:
            action = 0  # NOOP
            parsed_ok = True
            raw_match = "[SLEEPING → NOOP]"
            response_text = "[sleeping — skipping Gemini]"
        else:
            prompt = build_prompt(filt)
            result = call_gemini(prompt, api_key, model=model, use_thinking=False,
                                 thinking_budget=1024,
                                 max_output_tokens=2048, temperature=0.4)
            n_calls += 1
            if not result.get("ok"):
                api_fail += 1
                action = 0
                parsed_ok = False
                raw_match = "[API ERROR]"
                response_text = f"[API ERROR: {result.get('error', 'unknown')}]"
            else:
                response_text = result["text"]
                action, parsed_ok, raw_match = parse_action(response_text)
                if not parsed_ok:
                    parse_fail += 1
                step_latencies.append(result.get("latency_s", 0.0))
            prompts_log.append({
                "step": step,
                "prompt": prompt,
                "response": response_text,
            })

        if record_video and cv2 is not None:
            action_name = ACTION_NAMES[action] if 0 <= action < ACTION_DIM else "?"
            frames.append(make_frame(env_state, step, action_name,
                                      response_text, ep_return,
                                      raw_match=raw_match))

        rng, step_key = jax.random.split(rng)
        next_obs, env_state, reward, done, info = env.step(
            step_key, env_state, action, env_params,
        )
        reward_f = float(reward)
        ep_return += reward_f

        summary = summarize_state(filt)
        history.append({
            "summary": summary,
            "action": action,
            "action_name": ACTION_NAMES[action] if 0 <= action < ACTION_DIM else "?",
            "reward": reward_f,
        })
        history = history[-history_len:]

        traj.append({
            "step": step, "action": action, "reward": reward_f,
            "parsed": parsed_ok,
            "response_head": response_text[:180],
        })

        if verbose and (step < 6 or step % 50 == 0):
            print(f"  step {step:4d} a={action:2d} ({ACTION_NAMES[action]:<16}) "
                  f"r={reward_f:+.2f} ret={ep_return:+.2f} "
                  f"parsed={parsed_ok} | {summary}")

        obs = next_obs
        step += 1

    achievements = info.get("Achievements", {}) if isinstance(info, dict) else {}
    return {
        "return": ep_return,
        "length": step,
        "done": bool(done),
        "n_calls": n_calls,
        "parse_fail": parse_fail,
        "api_fail": api_fail,
        "mean_latency": float(np.mean(step_latencies)) if step_latencies else 0.0,
        "traj": traj,
        "prompts_log": prompts_log,
        "frames": frames,
    }


def save_video(frames: List[np.ndarray], path: Path, fps: int = 15) -> None:
    # Pipe raw RGB straight to ffmpeg → h264 + yuv420p + +faststart.
    # The previous cv2-mp4v writer produced files without a leading moov atom;
    # wandb's HTML5 player couldn't seek past the first buffered chunk, which
    # showed up as videos silently cut off around 26s regardless of episode
    # length. Writing h264 + faststart in one pass avoids that.
    if not frames:
        return
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not on PATH — required to write h264+faststart mp4 for wandb."
        )

    max_h = max(f.shape[0] for f in frames)
    w = frames[0].shape[1]
    out_w = w + (w & 1)
    out_h = max_h + (max_h & 1)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{out_w}x{out_h}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for f in frames:
            if f.shape[0] != out_h or f.shape[1] != out_w:
                padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                padded[:f.shape[0], :f.shape[1]] = f
                f = padded
            proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
    ret = proc.wait(timeout=300)
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with status {ret} writing {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--history-len", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default=GEMINI_MODEL)
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--save-video", action="store_true", default=False)
    p.add_argument("--wandb-name", type=str, default="")
    p.add_argument("--no-wandb", action="store_true", default=False)
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or pass --api-key.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(args.seed)

    use_wandb = (not args.no_wandb) and wandb is not None
    if use_wandb:
        # wandb's service-startup flake on some nodes can hang past the
        # library's own init_timeout (we've seen 10+ min hangs where
        # ServicePollForTokenError loops forever). Enforce a hard SIGALRM
        # ceiling so the run always progresses.
        import signal

        def _wandb_timeout_handler(signum, frame):
            raise TimeoutError("wandb.init exceeded hard timeout (SIGALRM)")

        wandb_ok = False
        for attempt in range(2):
            try:
                signal.signal(signal.SIGALRM, _wandb_timeout_handler)
                signal.alarm(90)
                wandb.init(
                    project="craftax-llm-harnessed",
                    entity="iris-sobolmark",
                    name=args.wandb_name or f"gemini-play-{time.strftime('%Y%m%d-%H%M%S')}",
                    config={
                        "eval_type": "gemini_plays_craftax",
                        "model": args.model,
                        "num_episodes": args.num_episodes,
                        "max_steps": args.max_steps,
                        "history_len": args.history_len,
                        "seed": args.seed,
                    },
                    settings=wandb.Settings(init_timeout=60),
                )
                signal.alarm(0)
                print(f"wandb initialized (attempt {attempt + 1})")
                wandb_ok = True
                break
            except Exception as e:
                signal.alarm(0)
                print(f"wandb init attempt {attempt + 1} failed ({type(e).__name__}: {e})")
                try:
                    wandb.finish(exit_code=1, quiet=True)
                except Exception:
                    pass
        if not wandb_ok:
            print("wandb init failed after retries; continuing without wandb")
            use_wandb = False

    results = []
    t_start = time.time()
    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep+1}/{args.num_episodes} ===")
        rng, ep_rng = jax.random.split(rng)
        t0 = time.time()
        r = run_episode(
            env, env_params, ep_rng, api_key,
            history_len=args.history_len, max_steps=args.max_steps,
            model=args.model, verbose=args.verbose,
            record_video=args.save_video,
        )
        r["wall_s"] = time.time() - t0
        print(f"  Return: {r['return']:+.2f}  Length: {r['length']}  "
              f"Calls: {r['n_calls']}  ParseFail: {r['parse_fail']}  "
              f"APIFail: {r['api_fail']}  MeanLatency: {r['mean_latency']:.2f}s  "
              f"Wall: {r['wall_s']:.1f}s")

        ep_dir = out_dir / f"episode_{ep+1:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        video_path = None
        if args.save_video and r.get("frames"):
            video_path = ep_dir / "gameplay.mp4"
            save_video(r["frames"], video_path)
            print(f"  Video saved: {video_path}")

        prompts_log = r.get("prompts_log") or []
        if prompts_log:
            with open(ep_dir / "prompts.jsonl", "w") as f:
                for entry in prompts_log:
                    f.write(json.dumps(entry) + "\n")

        r_small = {k: v for k, v in r.items() if k not in ("frames", "prompts_log")}
        with open(ep_dir / "summary.json", "w") as f:
            json.dump(r_small, f, indent=2)

        if use_wandb:
            log = {
                "episode/return": r["return"],
                "episode/length": r["length"],
                "episode/parse_fail": r["parse_fail"],
                "episode/api_fail": r["api_fail"],
                "episode/mean_latency_s": r["mean_latency"],
            }
            if video_path and video_path.exists():
                try:
                    log[f"video/episode_{ep+1:02d}"] = wandb.Video(
                        str(video_path), fps=15, format="mp4",
                    )
                except Exception as e:
                    print(f"  wandb video upload failed: {e}")
            wandb.log(log, step=ep + 1)
        r.pop("frames", None)
        r.pop("prompts_log", None)
        results.append(r)

    returns = np.array([r["return"] for r in results], dtype=np.float32)
    lengths = np.array([r["length"] for r in results], dtype=np.float32)
    summary = {
        "model": args.model,
        "n_episodes": len(results),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
        "total_gemini_calls": int(sum(r["n_calls"] for r in results)),
        "total_parse_fails": int(sum(r["parse_fail"] for r in results)),
        "total_api_fails": int(sum(r["api_fail"] for r in results)),
        "wall_s": time.time() - t_start,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if use_wandb:
        wandb.log({
            "summary/return_mean": summary["return_mean"],
            "summary/return_std": summary["return_std"],
            "summary/return_min": summary["return_min"],
            "summary/return_max": summary["return_max"],
            "summary/length_mean": summary["length_mean"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
