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
ACTION_RE = re.compile(r'(?:\*\*|)?\s*Action\s*:?\s*(?:\*\*|)?\s*(\d+)', re.IGNORECASE)
TAIL_NUM_RE = re.compile(r'(\d+)\s*\)?\s*$')


SYSTEM_PROMPT = """You are playing Craftax — a survival-crafting game.

Actions (must pick exactly one, 0-42):
0:NOOP  1:LEFT  2:RIGHT  3:UP  4:DOWN  5:DO (interact/mine/attack/drink/eat)
6:SLEEP  7:PLACE_STONE  8:PLACE_TABLE  9:PLACE_FURNACE  10:PLACE_PLANT
11:MAKE_WOOD_PICKAXE  12:MAKE_STONE_PICKAXE  13:MAKE_IRON_PICKAXE
14:MAKE_WOOD_SWORD  15:MAKE_STONE_SWORD  16:MAKE_IRON_SWORD
17:REST  18:DESCEND  19:ASCEND  20:MAKE_DIAMOND_PICKAXE  21:MAKE_DIAMOND_SWORD
22:MAKE_IRON_ARMOUR  23:MAKE_DIAMOND_ARMOUR  24:SHOOT_ARROW  25:MAKE_ARROW
26:CAST_FIREBALL  27:CAST_ICEBALL  28:PLACE_TORCH
29-34:DRINK_POTION_*  35:READ_BOOK  36:ENCHANT_SWORD  37:ENCHANT_ARMOUR
38:MAKE_TORCH  39:LEVEL_UP_DEX  40:LEVEL_UP_STR  41:LEVEL_UP_INT  42:ENCHANT_BOW

Coordinates are (Row, Col) relative to you at (0,0):
  -Row = UP, +Row = DOWN, -Col = LEFT, +Col = RIGHT.
DO acts on the tile you are facing (= the direction you most recently moved).

=== ALGORITHM ===
At every step, act with the goal of staying alive and progressing down floors.
Pick the highest-priority active goal in this order:
  1. Survive
  2. Take the ladder if it is open
  3. Upgrade equipment if survival is stable
  4. Explore for resources, troops, and the ladder

1. SURVIVE
   Track health, food, drink, energy.
   - Food low → kill an animal (cow, snail, bat, …) and eat it (DO on it).
   - Drink low → drink from a water tile (DO facing water).
   - Energy low → make a safe enclosure (PLACE_STONE around you), then SLEEP.
     Never sleep in the open.
   - Health low → restore food/drink/energy before doing anything risky.

2. TAKE THE LADDER
   If the ladder is OPEN (LadderOpen=True), prioritize finding it and using
   DESCEND. On later floors, ladder opens only after killing 8 troops. On
   Floor 0 you may still gather wood/stone/coal before descending since most
   early resources are there. Every descent grants one XP to spend on
   LEVEL_UP_STR / LEVEL_UP_DEX / LEVEL_UP_INT.

3. UPGRADE EQUIPMENT (only when survival is stable and ladder is not the
   main priority):
   Upgrade order:
     Pickaxe: wood → stone → iron → diamond
     Sword:   wood → stone → iron → diamond
     Armor:            iron → diamond
   Rules:
   - No useful tools → gather wood first (DO on a tree).
   - Crafting needs a crafting table; if none nearby, PLACE_TABLE.
   - After wood tools, mine stone.
   - Mine coal and iron whenever seen.
   - Adjacent to furnace+table with iron+coal+wood → craft iron tools.
   - Extra iron → craft iron armour.
   - Diamonds next to furnace+table: craft diamond equipment.

4. EXPLORE
   Not in danger, ladder not open, no upgrade available → move toward the
   nearest useful unseen direction. Look for the ladder, kill troops if the
   ladder is closed, gather wood/stone/coal/iron/diamonds.

=== OUTPUT FORMAT (STRICT) ===
Respond in AT MOST 2 short lines:
Line 1: one-clause reason (e.g., "Tree at (-1,0), face UP to chop").
Line 2 (REQUIRED, exactly this format): Action: <id>

where <id> is a single integer 0-42. No other text after it.
Example response:
Tree at (-1,0), face UP to chop.
Action: 3
"""


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


def build_user_prompt(filtered_obs: str, history: List[dict]) -> str:
    buf = []
    if history:
        buf.append("--- RECENT HISTORY (last few steps) ---")
        for i, h in enumerate(history):
            buf.append(
                f"t-{len(history)-i}: {h['summary']} | "
                f"took action {h['action']} ({h['action_name']}) reward={h['reward']:+.1f}"
            )
        buf.append("--- END HISTORY ---")
        buf.append("")
    buf.append("CURRENT STATE (use ONLY this for coordinates, you are at (0,0)):")
    buf.append(filtered_obs)
    buf.append("")
    buf.append("Respond in ≤2 lines, last line EXACTLY: Action: <id>")
    return "\n".join(buf)


def parse_action(text: str) -> Tuple[int, bool]:
    matches = list(ACTION_RE.finditer(text))
    if matches:
        try:
            a = int(matches[-1].group(1))
            if 0 <= a < ACTION_DIM:
                return a, True
        except ValueError:
            pass
    stripped = text.strip()
    m = TAIL_NUM_RE.search(stripped)
    if m:
        try:
            a = int(m.group(1))
            if 0 <= a < ACTION_DIM:
                return a, True
        except ValueError:
            pass
    return 0, False


def make_frame(env_state, step, action_name, response_text, ep_return):
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
    lines = [f"step={step}  ret={ep_return:+.2f}  a={action_name}"]
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

        user = build_user_prompt(filt, history)
        prompt = SYSTEM_PROMPT + "\n\n" + user

        # NOTE: call_gemini's use_thinking=True sets thinkingBudget=0 (DISABLES
        # thinking). We disable thinking so the full token budget goes to the
        # answer, not reasoning. The algorithm+history already provide structure.
        result = call_gemini(prompt, api_key, model=model, use_thinking=True,
                             max_output_tokens=256, temperature=0.4)
        n_calls += 1
        if not result.get("ok"):
            api_fail += 1
            action = 0
            parsed_ok = False
            response_text = f"[API ERROR: {result.get('error', 'unknown')}]"
        else:
            response_text = result["text"]
            action, parsed_ok = parse_action(response_text)
            if not parsed_ok:
                parse_fail += 1
            step_latencies.append(result.get("latency_s", 0.0))

        if record_video and cv2 is not None:
            action_name = ACTION_NAMES[action] if 0 <= action < ACTION_DIM else "?"
            frames.append(make_frame(env_state, step, action_name,
                                      response_text, ep_return))

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
        "frames": frames,
    }


def save_video(frames: List[np.ndarray], path: Path) -> None:
    if not frames or cv2 is None:
        return
    max_h = max(f.shape[0] for f in frames)
    w = frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, (w, max_h))
    for f in frames:
        if f.shape[0] < max_h:
            pad = np.zeros((max_h - f.shape[0], w, 3), dtype=np.uint8)
            f = np.vstack([f, pad])
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    import shutil, subprocess
    if shutil.which("ffmpeg"):
        h264 = path.with_suffix(".h264.mp4")
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-pix_fmt", "yuv420p", str(h264)],
            capture_output=True, timeout=180,
        )
        if ret.returncode == 0 and h264.exists():
            h264.rename(path)


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
        wandb.init(
            project="craftax-offline-awr",
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
            settings=wandb.Settings(init_timeout=600, start_method="thread"),
        )
        print("wandb initialized")

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

        r_small = {k: v for k, v in r.items() if k != "frames"}
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
