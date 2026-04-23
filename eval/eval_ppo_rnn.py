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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import jax
import torch

from models.ppo_rnn_torch import ActorCriticRNNTorch, load_from_orbax
from pipeline.config import ACTION_NAMES

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
        }
        all_results.append(result)
        print(f"\n  Return: {ep_return:.2f}  Length: {step}  Achievements: {len(ep_achievements)}")

        ep_dir = out_dir / f"episode_{ep+1:02d}"
        ep_dir.mkdir(exist_ok=True)
        with open(ep_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)

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
    run_eval(p.parse_args())


if __name__ == "__main__":
    main()
