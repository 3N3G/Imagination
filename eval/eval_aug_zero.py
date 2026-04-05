#!/usr/bin/env python3
"""
Quick online evaluation of augmented models with zero hidden states.

Tests whether an augmented model can still play the game without
imagination embeddings. If a model collapses here, it definitely
won't work with real embeddings either.

Usage:
    python -m pipeline.eval_aug_zero \
        --checkpoint /path/to/final.pth \
        --hidden-stats /path/to/hidden_state_stats.npz \
        --num-episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import jax
import torch

from models.actor_critic_aug import ActorCriticAug

ACTION_DIM = 43
OBS_DIM = 8268
HIDDEN_STATE_DIM = 4096


def make_env():
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    return env


@torch.no_grad()
def run_eval(args):
    device = args.device
    print(f"Loading model from {args.checkpoint}")

    model = ActorCriticAug(OBS_DIM, ACTION_DIM, args.layer_width, HIDDEN_STATE_DIM).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Zero hidden state (normalized: subtract mean, divide by std)
    # This gives -(mean/std) which is the "no information" input
    zero_hidden = torch.zeros(1, HIDDEN_STATE_DIM, dtype=torch.float32, device=device)

    env = make_env()
    rng = jax.random.PRNGKey(args.seed)

    returns = []
    entropies = []

    for ep in range(args.num_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs_jax, env_state = env.reset(reset_rng)
        obs_flat = np.array(obs_jax).reshape(1, -1).astype(np.float32)

        ep_return = 0.0
        ep_entropy = []

        for step in range(1000):  # Craftax has 1000 max steps
            obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device)
            pi, value = model(obs_t, zero_hidden)

            action = pi.sample().item()
            ep_entropy.append(pi.entropy().item())

            rng, step_rng = jax.random.split(rng)
            obs_jax, env_state, reward, done, info = env.step(step_rng, env_state, action)
            obs_flat = np.array(obs_jax).reshape(1, -1).astype(np.float32)

            ep_return += float(reward)

            if bool(done):
                break

        returns.append(ep_return)
        entropies.append(np.mean(ep_entropy))

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{args.num_episodes}: "
                  f"return={ep_return:.1f}, entropy={np.mean(ep_entropy):.3f}")

    returns = np.array(returns)
    mean_entropy = np.mean(entropies)

    print()
    print("=" * 50)
    print(f"Results ({args.num_episodes} episodes, zero embeddings):")
    print(f"  Return: {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"  Min/Max: {returns.min():.1f} / {returns.max():.1f}")
    print(f"  Mean entropy: {mean_entropy:.3f}")
    print("=" * 50)

    # Save results
    out_dir = os.path.dirname(args.checkpoint)
    result = {
        "zero_embed_eval": {
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "mean_entropy": float(mean_entropy),
            "num_episodes": args.num_episodes,
            "returns": returns.tolist(),
        }
    }
    out_path = os.path.join(out_dir, "zero_embed_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--layer-width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
