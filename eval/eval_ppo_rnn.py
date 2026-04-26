#!/usr/bin/env python3
"""Online 50-ep eval of a PPO-RNN policy saved via online_rl/ppo_rnn.py.

Loads the orbax checkpoint from --checkpoint <dir>, transfers weights into the
PyTorch port (models/ppo_rnn_torch.py), and rolls out on Craftax-Symbolic-v1
with per-episode hidden-state carry and reset-on-done.

Usage:
    python -m eval.eval_ppo_rnn \
        --checkpoint /data/group_data/rl/geney/checkpoints/ppo_rnn_5M_baseline \
        --num-episodes 50 \
        --output-dir /data/group_data/rl/geney/eval_results/ppo_rnn_5M
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

try:
    import cv2
except ImportError:
    cv2 = None

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import torch

from models.ppo_rnn_torch import ActorCriticRNNTorch, load_from_orbax
from pipeline.config import ACTION_NAMES


def _render_frame(env_state) -> np.ndarray:
    from craftax.craftax.renderer import render_craftax_pixels
    pixels = render_craftax_pixels(env_state, block_pixel_size=16, do_night_noise=False)
    return np.array(pixels, dtype=np.uint8)


def _make_video_frame(game_frame, values, rewards, step, action_name):
    """Simpler composition for PPO baselines — game + value graph + action name bar."""
    if cv2 is None:
        return game_frame
    target_w = 600
    h, w = game_frame.shape[:2]
    scale = target_w / w
    target_h = int(h * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    graph_h = 50
    footer_h = graph_h + 60
    canvas = np.zeros((target_h + footer_h, target_w, 3), dtype=np.uint8)
    resized = cv2.resize(game_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas[:target_h, :target_w] = resized
    y0 = target_h + 10
    # value-over-time graph
    if values:
        vs = np.array(values, dtype=np.float32)
        lo, hi = float(vs.min()), float(vs.max())
        if hi - lo < 1e-6: hi = lo + 1.0
        n = min(len(vs), target_w - 20)
        xs = np.linspace(0, len(vs) - 1, n).astype(int)
        ys = y0 + graph_h - ((vs[xs] - lo) / (hi - lo) * graph_h).astype(int)
        for i in range(n - 1):
            cv2.line(canvas, (10 + int(i * (target_w - 20) / max(n - 1, 1)), int(ys[i])),
                     (10 + int((i + 1) * (target_w - 20) / max(n - 1, 1)), int(ys[i + 1])),
                     (120, 200, 120), 1)
    # Text footer: step, return, action
    total_ret = float(np.sum(rewards)) if rewards else 0.0
    cur_v = values[-1] if values else 0.0
    cv2.putText(canvas, f"step={step}  return={total_ret:.1f}  value={cur_v:.2f}  action={action_name}",
                (10, target_h + graph_h + 35), font, 0.5, (230, 230, 230), 1)
    return canvas

ACTION_DIM = 43
OBS_DIM = 8268


def run_eval(args):
    device = torch.device(args.device)

    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_from_orbax(Path(args.checkpoint), action_dim=ACTION_DIM,
                            obs_dim=OBS_DIM, layer_size=args.layer_size)
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Policy loaded: {total_params:,} params, width={args.layer_size}")

    import craftax.craftax.envs.craftax_pixels_env as pxmod
    from craftax.craftax_env import make_craftax_env_from_name
    Achievement = pxmod.Achievement

    def log_achievements_always(state, done):
        achievements = state.achievements * 100.0
        return {f"Achievements/{a.name.lower()}": achievements[a.value] for a in Achievement}
    pxmod.log_achievements_to_info = log_achievements_always

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        wandb.init(
            project="craftax-offline-awr",
            entity="iris-sobolmark",
            name=args.wandb_name or f"eval-ppo-rnn-{time.strftime('%Y%m%d-%H%M%S')}",
            config={"eval_type": "ppo_rnn", "num_episodes": args.num_episodes,
                    "checkpoint": str(args.checkpoint), "seed": args.seed,
                    "layer_size": args.layer_size, "total_params": total_params},
            settings=wandb.Settings(init_timeout=600, start_method="thread"),
        )
        print("wandb initialized")

    global_step = 0
    all_results = []
    for ep in range(args.num_episodes):
        print(f"\n{'='*60}\nEpisode {ep+1}/{args.num_episodes}\n{'='*60}")
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)
        h = model.initial_state(1, device=device)

        done = False; step = 0; ep_return = 0.0
        ep_rewards = []; ep_values = []
        ep_achievements = {}
        ep_action_counts = np.zeros(ACTION_DIM, dtype=np.int32)
        ep_frames = []
        ep_actions = []  # for replay/death-classification downstream

        while not done and step < 10000:
            obs_np = np.array(obs, dtype=np.float32)
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value, h_new = model(obs_t, h)
                pi = torch.distributions.Categorical(logits=logits)
                action = int(pi.sample().item())
                v = float(value.item())
                entropy = float(pi.entropy().item())
            h = h_new

            ep_values.append(v)
            ep_action_counts[action] += 1
            ep_actions.append(int(action))

            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(step_key, env_state, action, env_params)
            reward_f = float(reward)
            ep_return += reward_f
            ep_rewards.append(reward_f)

            for k, val in info.items():
                if k.startswith("Achievements/") and float(val) > 0:
                    name = k.replace("Achievements/", "")
                    if name not in ep_achievements:
                        ep_achievements[name] = step
                        print(f"  [step {step}] Achievement unlocked: {name}")

            if use_wandb and step % 25 == 0:
                wandb.log({"step/value": v, "step/entropy": entropy,
                           "step/reward": reward_f,
                           "step/cumulative_return": ep_return,
                           "step/action": action, "step/episode": ep + 1,
                           "step/ep_step": step}, step=global_step)

            if args.save_video and cv2 is not None:
                try:
                    game_frame = _render_frame(env_state)
                    aname = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
                    frame = _make_video_frame(game_frame, ep_values, ep_rewards, step, aname)
                    ep_frames.append(frame)
                except Exception:
                    pass

            step += 1
            global_step += 1
            if step % 500 == 0:
                print(f"  Step {step}: return={ep_return:.2f}, value={v:.2f}")

        result = {
            "episode": ep + 1,
            "return": ep_return,
            "length": step,
            "achievements": ep_achievements,
            "num_achievements": len(ep_achievements),
            "mean_value": float(np.mean(ep_values)) if ep_values else 0,
            "actions": ep_actions,
        }
        all_results.append(result)
        print(f"\n  Return: {ep_return:.2f}  Length: {step}  Achievements: {len(ep_achievements)}")

        ep_dir = out_dir / f"episode_{ep+1:02d}"
        ep_dir.mkdir(exist_ok=True)
        with open(ep_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)

        if args.save_video and cv2 is not None and ep_frames:
            video_path = ep_dir / "gameplay.mp4"
            max_h = max(f.shape[0] for f in ep_frames)
            w = ep_frames[0].shape[1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, max_h))
            for f_ in ep_frames:
                if f_.shape[0] < max_h:
                    pad = np.zeros((max_h - f_.shape[0], w, 3), dtype=np.uint8)
                    f_ = np.vstack([f_, pad])
                writer.write(cv2.cvtColor(f_, cv2.COLOR_RGB2BGR))
            writer.release()
            # Re-encode to H.264 for browser compatibility
            import subprocess, shutil
            if shutil.which("ffmpeg"):
                h264_path = ep_dir / "gameplay_h264.mp4"
                ret = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video_path),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
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

        if use_wandb:
            ep_log = {"episode/return": ep_return, "episode/length": step,
                      "episode/num_achievements": len(ep_achievements),
                      "episode/mean_value": float(np.mean(ep_values)) if ep_values else 0}
            top = np.argsort(-ep_action_counts)[:10]
            for ai in top:
                if ep_action_counts[ai] > 0:
                    aname = ACTION_NAMES[ai] if ai < len(ACTION_NAMES) else str(ai)
                    ep_log[f"actions/{aname}"] = int(ep_action_counts[ai])
            wandb.log(ep_log, step=global_step)

            video_path = ep_dir / "gameplay.mp4"
            if video_path.exists():
                try:
                    wandb.log({
                        f"video/episode_{ep+1:02d}": wandb.Video(
                            str(video_path), fps=15, format="mp4",
                        ),
                    }, step=global_step)
                except Exception as e:
                    print(f"  wandb video upload failed: {e}")

    # Summary
    print(f"\n{'='*60}\nEVALUATION SUMMARY\n{'='*60}")
    returns = [r["return"] for r in all_results]
    lengths = [r["length"] for r in all_results]
    n_ach = [r["num_achievements"] for r in all_results]
    print(f"Episodes: {args.num_episodes}")
    print(f"Return:       {np.mean(returns):.2f} +/- {np.std(returns, ddof=1):.2f}  "
          f"(min={min(returns):.2f}, max={max(returns):.2f})")
    print(f"Length:       {np.mean(lengths):.0f} +/- {np.std(lengths):.0f}")
    print(f"Achievements: {np.mean(n_ach):.1f} +/- {np.std(n_ach):.1f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "episodes": all_results,
                   "mean_return": float(np.mean(returns)),
                   "std_return": float(np.std(returns, ddof=1)),
                   "mean_length": float(np.mean(lengths)),
                   "mean_achievements": float(np.mean(n_ach))}, f, indent=2)

    if use_wandb:
        wandb.summary["mean_return"] = float(np.mean(returns))
        wandb.summary["std_return"] = float(np.std(returns, ddof=1))
        wandb.summary["mean_length"] = float(np.mean(lengths))
        wandb.summary["mean_achievements"] = float(np.mean(n_ach))
        wandb.finish()

    print(f"\nResults saved to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="dir containing policies/<step>/ (or the policies dir itself)")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer-size", type=int, default=512)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--save-video", action="store_true", default=True,
                   help="Save gameplay.mp4 per episode (default on).")
    p.add_argument("--no-video", dest="save_video", action="store_false")
    run_eval(p.parse_args())


if __name__ == "__main__":
    main()
