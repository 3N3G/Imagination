#!/usr/bin/env python3
"""
Online evaluation of the unaugmented (obs-only) AWR policy.

No Gemini or Qwen needed — just obs -> action.
Logs video recordings, per-step rewards, achievements to wandb.

Usage:
    python -m pipeline.eval_unaugmented --num-episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import jax
import torch

from models.actor_critic_aug import ActorCritic
from pipeline.config import ACTION_NAMES

# --- Constants ---
ACTION_DIM = 43
OBS_DIM = 8268
DEFAULT_LAYER_WIDTH = 512
DEFAULT_CKPT_DIR = Path("/data/group_data/rl/geney/checkpoints/awr_unaug_w512")


# ======================================================================
# Video rendering helpers
# ======================================================================
def render_frame(env_state) -> np.ndarray:
    from craftax.craftax.renderer import render_craftax_pixels
    pixels = render_craftax_pixels(env_state, block_pixel_size=16, do_night_noise=False)
    return np.array(pixels, dtype=np.uint8)


def make_video_frame(game_frame, values, rewards, step):
    target_w = 600
    h, w = game_frame.shape[:2]
    scale = target_w / w
    target_h = int(h * scale)
    footer_h = 100

    canvas = np.zeros((target_h + footer_h, target_w, 3), dtype=np.uint8)
    resized = cv2.resize(game_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas[:target_h, :target_w] = resized

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Value graph
    graph_y = target_h + 10
    graph_h = 50
    cv2.rectangle(canvas, (10, graph_y), (target_w - 10, graph_y + graph_h), (30, 30, 30), -1)
    if len(values) > 1:
        v_min, v_max = min(min(values), 0), max(max(values), 1)
        for i in range(len(values) - 1):
            x1 = int(10 + i / len(values) * (target_w - 20))
            x2 = int(10 + (i + 1) / len(values) * (target_w - 20))
            y1 = int(graph_y + graph_h - (values[i] - v_min) / (v_max - v_min + 1e-8) * graph_h)
            y2 = int(graph_y + graph_h - (values[i + 1] - v_min) / (v_max - v_min + 1e-8) * graph_h)
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.putText(canvas, f"Step {step}  V={values[-1]:.2f}  R={sum(rewards):.2f}",
                (10, graph_y - 3), font, 0.4, (200, 200, 200), 1)

    return canvas


# ======================================================================
# Main evaluation
# ======================================================================
def run_eval(args):
    device = args.device
    layer_width = args.layer_width

    # Load policy
    model = ActorCritic(OBS_DIM, ACTION_DIM, layer_width).to(device)
    ckpt_path = args.checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Policy loaded: {ckpt_path} ({total_params:,} params, width={layer_width})")

    # Init Craftax environment
    import craftax.craftax.envs.craftax_pixels_env as pxmod
    from craftax.craftax_env import make_craftax_env_from_name
    Achievement = pxmod.Achievement

    def log_achievements_always(state, done):
        achievements = state.achievements * 100.0
        return {
            f"Achievements/{a.name.lower()}": achievements[a.value]
            for a in Achievement
        }
    pxmod.log_achievements_to_info = log_achievements_always

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Init wandb ---
    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        wandb.init(
            project="craftax-offline-awr",
            entity="iris-sobolmark",
            name=args.wandb_name or f"eval-unaug-w{layer_width}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "eval_type": "online_unaugmented",
                "num_episodes": args.num_episodes,
                "checkpoint": str(ckpt_path),
                "seed": args.seed,
                "layer_width": layer_width,
                "total_params": total_params,
            },
            settings=wandb.Settings(init_timeout=600, start_method="thread"),
        )
        print("wandb initialized")
    elif wandb is None:
        print("WARNING: wandb not installed, logging to files only")

    global_step = 0
    all_results = []

    for ep in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.num_episodes}")
        print(f"{'='*60}")

        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        done = False
        step = 0
        ep_return = 0.0
        ep_rewards = []
        ep_values = []
        ep_frames = []
        ep_achievements = {}
        ep_action_counts = np.zeros(ACTION_DIM, dtype=np.int32)

        while not done and step < 10000:
            obs_np = np.array(obs, dtype=np.float32)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                pi, value = model(obs_t)
                action = pi.sample().item()
                v = value.item()
                entropy = pi.entropy().item()

            ep_values.append(v)
            ep_action_counts[action] += 1

            # Step environment
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                step_key, env_state, action, env_params
            )

            reward_f = float(reward)
            ep_return += reward_f
            ep_rewards.append(reward_f)

            # Track achievements
            for k, val in info.items():
                if k.startswith("Achievements/") and float(val) > 0:
                    name = k.replace("Achievements/", "")
                    if name not in ep_achievements:
                        ep_achievements[name] = step
                        print(f"  [step {step}] Achievement unlocked: {name}")

            # wandb: per-step logging (every 15 steps)
            if use_wandb and step % 15 == 0:
                wandb.log({
                    "step/value": v,
                    "step/entropy": entropy,
                    "step/reward": reward_f,
                    "step/cumulative_return": ep_return,
                    "step/action": action,
                    "step/action_name": ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action),
                    "step/episode": ep + 1,
                    "step/ep_step": step,
                    "step/num_achievements": len(ep_achievements),
                }, step=global_step)

            # Record video frame
            if args.save_video:
                try:
                    game_frame = render_frame(env_state)
                    frame = make_video_frame(game_frame, ep_values, ep_rewards, step)
                    ep_frames.append(frame)
                except Exception:
                    pass

            step += 1
            global_step += 1
            if step % 200 == 0:
                print(f"  Step {step}: return={ep_return:.2f}, value={v:.2f}, "
                      f"action={ACTION_NAMES[action] if action < len(ACTION_NAMES) else action}")

        # --- Episode summary ---
        result = {
            "episode": ep + 1,
            "return": ep_return,
            "length": step,
            "achievements": ep_achievements,
            "num_achievements": len(ep_achievements),
            "mean_value": float(np.mean(ep_values)) if ep_values else 0,
        }
        all_results.append(result)

        print(f"\n  Return: {ep_return:.2f}")
        print(f"  Length: {step}")
        print(f"  Achievements ({len(ep_achievements)}): {list(ep_achievements.keys())}")

        # Save episode log
        ep_dir = out_dir / f"episode_{ep + 1:02d}"
        ep_dir.mkdir(exist_ok=True)
        with open(ep_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)

        # Save video
        video_path = None
        if args.save_video and ep_frames:
            video_path = ep_dir / "gameplay.mp4"
            h, w = ep_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
            for frame in ep_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            import subprocess, shutil
            if shutil.which("ffmpeg"):
                h264_path = ep_dir / "gameplay_h264.mp4"
                ret = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video_path),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-pix_fmt", "yuv420p", str(h264_path)],
                    capture_output=True, timeout=120,
                )
                if ret.returncode == 0 and h264_path.exists():
                    h264_path.rename(video_path)
                    print(f"  Video saved (H.264): {video_path}")
                else:
                    print(f"  Video saved (mp4v fallback): {video_path}")
            else:
                print(f"  Video saved (mp4v): {video_path}")

        # wandb: per-episode logging
        if use_wandb:
            ep_log = {
                "episode/return": ep_return,
                "episode/length": step,
                "episode/num_achievements": len(ep_achievements),
                "episode/mean_value": float(np.mean(ep_values)) if ep_values else 0,
                "episode/std_value": float(np.std(ep_values)) if ep_values else 0,
                "episode/mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0,
            }
            for ach_name, ach_step in ep_achievements.items():
                ep_log[f"achievements/{ach_name}_step"] = ach_step
            top_actions = np.argsort(-ep_action_counts)[:10]
            for rank, ai in enumerate(top_actions):
                if ep_action_counts[ai] > 0:
                    aname = ACTION_NAMES[ai] if ai < len(ACTION_NAMES) else str(ai)
                    ep_log[f"actions/{aname}"] = int(ep_action_counts[ai])
            wandb.log(ep_log, step=global_step)

            if video_path and video_path.exists():
                try:
                    wandb.log({
                        f"video/episode_{ep+1:02d}": wandb.Video(
                            str(video_path), fps=15, format="mp4",
                        ),
                    }, step=global_step)
                except Exception as e:
                    print(f"  wandb video upload failed: {e}")

    # --- Overall summary ---
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    returns = [r["return"] for r in all_results]
    lengths = [r["length"] for r in all_results]
    n_ach = [r["num_achievements"] for r in all_results]
    print(f"Episodes: {args.num_episodes}")
    print(f"Return:       {np.mean(returns):.2f} +/- {np.std(returns):.2f}  "
          f"(min={min(returns):.2f}, max={max(returns):.2f})")
    print(f"Length:       {np.mean(lengths):.0f} +/- {np.std(lengths):.0f}")
    print(f"Achievements: {np.mean(n_ach):.1f} +/- {np.std(n_ach):.1f}")

    all_ach = {}
    for r in all_results:
        for name in r["achievements"]:
            all_ach[name] = all_ach.get(name, 0) + 1
    if all_ach:
        print(f"\nAchievement frequency (out of {args.num_episodes} episodes):")
        for name, count in sorted(all_ach.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}/{args.num_episodes}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "episodes": all_results,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "mean_achievements": float(np.mean(n_ach)),
            "achievement_frequency": all_ach,
        }, f, indent=2)

    if use_wandb:
        wandb.summary["mean_return"] = float(np.mean(returns))
        wandb.summary["std_return"] = float(np.std(returns))
        wandb.summary["mean_length"] = float(np.mean(lengths))
        wandb.summary["mean_achievements"] = float(np.mean(n_ach))
        for name, count in all_ach.items():
            wandb.summary[f"achievement_freq/{name}"] = count / args.num_episodes

        ep_table = wandb.Table(
            columns=["episode", "return", "length", "achievements"],
        )
        for r in all_results:
            ep_table.add_data(r["episode"], r["return"], r["length"], r["num_achievements"])
        wandb.log({"summary/episodes": ep_table})
        wandb.finish()
        print("wandb run finished")

    print(f"\nResults saved to {out_dir}")


# ======================================================================
# CLI
# ======================================================================
def main():
    p = argparse.ArgumentParser(description="Online eval of unaugmented (obs-only) policy")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to policy checkpoint")
    p.add_argument("--output-dir", type=str, required=True,
                    help="Directory for eval outputs")
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-video", dest="save_video", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable wandb logging")
    p.add_argument("--wandb-name", type=str, default=None,
                    help="Custom wandb run name")
    p.add_argument("--layer-width", type=int, default=DEFAULT_LAYER_WIDTH,
                    help="Width of hidden layers (must match training)")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
