#!/usr/bin/env python3
"""
Direction-match counterfactual: at step 0 of each episode, compare the policy's
preferred first-movement action under three input variants:

  A. (orig_obs, hidden_orig)             — baseline (same as eval_online step 0)
  B. (flipped_obs, hidden_orig)          — obs 180°-rotated; hidden unchanged
  C. (orig_obs, hidden_flipped)          — hidden from Gemini narrating a 180°-flipped obs

For each we compute argmax over the 4 movement actions {LEFT=1, RIGHT=2, UP=3, DOWN=4}
(non-movement logits masked out).

Gemini is called twice per episode: once on obs_to_text(orig) and once on obs_to_text(flipped).
No environment stepping — single step-0 forward pass per condition.

Writes JSONL of per-episode records at --output-dir/episodes.jsonl.  Run
analysis/direction_match_cf.py afterward to compute match-rate tables.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import jax
import torch

from labelling.obs_to_text import obs_to_text
from llm.prompts import filter_text_obs
from envs.obs_to_text import OBS_DIM, MAP_CHANNELS, MAP_OBS_SIZE, TOTAL_OBS_SIZE
from models.actor_critic_aug import (
    ActorCriticAug as ActorCriticAugBase,
    ActorCriticAugLN,
    ActorCriticAugV2,
)
from pipeline.config import EMBED_HIDDEN_DIM, GEMINI_MODEL

from eval.eval_online import (
    PREDICT_TEMPLATE_PATH,
    ACTION_DIM,
    OBS_DIM as OBS_LEN,
    call_gemini,
    make_embedder,
)

def call_gemini_retry(prompt: str, api_key: str, model: str, use_thinking: bool,
                      max_attempts: int = 4) -> dict:
    """Gemini occasionally returns an empty completion — retry until we get text."""
    import time as _t
    last = None
    for attempt in range(max_attempts):
        last = call_gemini(prompt, api_key, model=model, use_thinking=use_thinking)
        if last.get("text", "").strip():
            return last
        if attempt < max_attempts - 1:
            _t.sleep(1 + attempt)
    raise RuntimeError(f"Gemini returned empty text after {max_attempts} attempts")

MOVE_ACTIONS = [1, 2, 3, 4]  # LEFT, RIGHT, UP, DOWN
MOVE_NAMES = {1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN"}


def flip_obs_180(obs: np.ndarray) -> np.ndarray:
    """Rotate the spatial map 180° around the player (center of 9×11). Inventory unchanged."""
    flat = np.asarray(obs, dtype=np.float32).flatten()
    if flat.size != TOTAL_OBS_SIZE:
        raise ValueError(f"Expected obs of size {TOTAL_OBS_SIZE}, got {flat.size}")
    map_part = flat[:MAP_OBS_SIZE].reshape(OBS_DIM[0], OBS_DIM[1], MAP_CHANNELS)
    flipped_map = map_part[::-1, ::-1, :].copy()
    return np.concatenate([flipped_map.reshape(-1), flat[MAP_OBS_SIZE:]]).astype(np.float32)


def movement_argmax_and_probs(pi) -> tuple[int, list[float]]:
    """Argmax over {LEFT, RIGHT, UP, DOWN} + full softmax over that subset."""
    logits = pi.logits.squeeze(0).detach().cpu().numpy()
    move_logits = np.array([logits[a] for a in MOVE_ACTIONS], dtype=np.float64)
    move_logits = move_logits - move_logits.max()
    probs = np.exp(move_logits)
    probs = probs / probs.sum()
    best = MOVE_ACTIONS[int(np.argmax(move_logits))]
    return best, probs.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hidden-stats", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer-width", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=-1)
    ap.add_argument("--embed-backend", default="qwen3_gen",
                    choices=["qwen3_gen", "qwen3_embed", "gemini_embed"])
    ap.add_argument("--arch-v2", action="store_true")
    ap.add_argument("--no-layernorm", action="store_true")
    ap.add_argument("--gemini-api-key", default=None)
    ap.add_argument("--gemini-model", default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY env var or pass --gemini-api-key")
    gemini_model = args.gemini_model or GEMINI_MODEL
    use_thinking = gemini_model.startswith("gemini-2.5")

    # --- Load policy ---
    if args.arch_v2:
        ModelClass = ActorCriticAugV2
    elif args.no_layernorm:
        ModelClass = ActorCriticAugBase
    else:
        ModelClass = ActorCriticAugLN
    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else EMBED_HIDDEN_DIM
    model_kwargs = dict(obs_dim=OBS_LEN, action_dim=ACTION_DIM,
                        layer_width=args.layer_width, hidden_state_dim=hidden_dim)
    if ModelClass != ActorCriticAugBase:
        model_kwargs["dropout"] = args.dropout
    model = ModelClass(**model_kwargs).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    print(f"Policy: {args.checkpoint}  arch={ModelClass.__name__}  hidden_dim={hidden_dim}")

    stats = np.load(args.hidden_stats)
    hidden_mean = stats["mean"].astype(np.float32)
    hidden_std = stats["std"].astype(np.float32)

    embedder = make_embedder(
        backend=args.embed_backend, device=args.device, api_key=api_key,
        output_dim=hidden_dim,
    )
    print(f"Embedder: {args.embed_backend}")

    template = PREDICT_TEMPLATE_PATH.read_text()

    # --- Env ---
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(args.seed)

    out_path = out_dir / "episodes.jsonl"
    out_f = open(out_path, "w")

    for ep in range(args.num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, _ = env.reset(reset_key, env_params)
        obs_np = np.array(obs, dtype=np.float32)
        flipped_np = flip_obs_180(obs_np)

        text_orig = obs_to_text(obs_np)
        text_flipped = obs_to_text(flipped_np)
        prompt_orig = template.replace("{current_state_filtered}", filter_text_obs(text_orig))
        prompt_flipped = template.replace("{current_state_filtered}", filter_text_obs(text_flipped))

        # Two Gemini calls: on original and on flipped text_obs
        g_orig = call_gemini_retry(prompt_orig, api_key, model=gemini_model, use_thinking=use_thinking)
        g_flipped = call_gemini_retry(prompt_flipped, api_key, model=gemini_model, use_thinking=use_thinking)
        narr_orig = g_orig["text"]
        narr_flipped = g_flipped["text"]

        # Embed both
        hid_orig_raw = embedder.embed(narr_orig)
        hid_flipped_raw = embedder.embed(narr_flipped)
        hid_orig = (hid_orig_raw - hidden_mean) / hidden_std
        hid_flipped = (hid_flipped_raw - hidden_mean) / hidden_std

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=args.device).unsqueeze(0)
        flip_t = torch.tensor(flipped_np, dtype=torch.float32, device=args.device).unsqueeze(0)
        ho_t = torch.tensor(hid_orig, dtype=torch.float32, device=args.device).unsqueeze(0)
        hf_t = torch.tensor(hid_flipped, dtype=torch.float32, device=args.device).unsqueeze(0)

        with torch.no_grad():
            pi_A, _ = model(obs_t, ho_t)    # baseline
            pi_B, _ = model(flip_t, ho_t)   # obs counterfactual
            pi_C, _ = model(obs_t, hf_t)    # emb counterfactual

        act_A, probs_A = movement_argmax_and_probs(pi_A)
        act_B, probs_B = movement_argmax_and_probs(pi_B)
        act_C, probs_C = movement_argmax_and_probs(pi_C)

        record = {
            "episode": ep + 1,
            "narrative_orig": narr_orig,
            "narrative_flipped": narr_flipped,
            "action_A": act_A, "action_B": act_B, "action_C": act_C,
            "action_A_name": MOVE_NAMES[act_A],
            "action_B_name": MOVE_NAMES[act_B],
            "action_C_name": MOVE_NAMES[act_C],
            "probs_A": probs_A, "probs_B": probs_B, "probs_C": probs_C,
            "gemini_orig_tokens": g_orig["prompt_tokens"] + g_orig["completion_tokens"],
            "gemini_flipped_tokens": g_flipped["prompt_tokens"] + g_flipped["completion_tokens"],
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        headline_o = narr_orig.split("\n")[0][:80] if narr_orig else ""
        headline_f = narr_flipped.split("\n")[0][:80] if narr_flipped else ""
        print(f"ep {ep+1:2d}  A={MOVE_NAMES[act_A]:<5s} B={MOVE_NAMES[act_B]:<5s} C={MOVE_NAMES[act_C]:<5s}  "
              f"orig: {headline_o}  flipped: {headline_f}")

    out_f.close()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
