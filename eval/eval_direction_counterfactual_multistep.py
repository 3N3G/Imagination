#!/usr/bin/env python3
"""
Multistep direction-match counterfactual.

Plays each episode forward with the baseline policy (condition A). Gemini is
called every STEP_CADENCE=15 steps to maintain the baseline hidden (mirrors
eval_online). At specified `--intervention-steps`, additionally:
  - runs Gemini on the 180°-flipped obs
  - embeds the flipped narrative
  - computes three movement argmaxes (A/B/C) as in step-0 eval_direction_counterfactual

Writes per-record JSONL of (episode, step, action_A/B/C, narrative_orig,
narrative_flipped, probs, coord parsing done post-hoc).
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
    STEP_CADENCE,
    call_gemini,
    make_embedder,
)
from eval.eval_direction_counterfactual import (
    call_gemini_retry,
    flip_obs_180,
    movement_argmax_and_probs,
    MOVE_ACTIONS,
    MOVE_NAMES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hidden-stats", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-episodes", type=int, default=30)
    ap.add_argument("--intervention-steps", default="0,75,150,300",
                    help="Comma-separated env steps at which to run B/C counterfactuals")
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

    intervention_steps = sorted(int(x) for x in args.intervention_steps.split(","))
    intervention_set = set(intervention_steps)
    max_step = max(intervention_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY env var or pass --gemini-api-key")
    gemini_model = args.gemini_model or GEMINI_MODEL
    use_thinking = gemini_model.startswith("gemini-2.5")

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
    print(f"Intervention steps: {intervention_steps}  (max={max_step})  cadence={STEP_CADENCE}")

    stats = np.load(args.hidden_stats)
    hidden_mean = stats["mean"].astype(np.float32)
    hidden_std = stats["std"].astype(np.float32)

    embedder = make_embedder(
        backend=args.embed_backend, device=args.device, api_key=api_key,
        output_dim=hidden_dim,
    )

    template = PREDICT_TEMPLATE_PATH.read_text()

    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(args.seed)

    out_path = out_dir / "episodes.jsonl"
    out_f = open(out_path, "w")

    for ep in range(args.num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        current_hidden_raw = None
        current_narrative_orig = None
        ep_records = []
        done = False

        for step in range(max_step + 1):
            if done:
                break
            obs_np = np.array(obs, dtype=np.float32)

            # Refresh baseline hidden at STEP_CADENCE (also at step 0) OR when intervention needs fresh coord
            need_orig_call = (step % STEP_CADENCE == 0) or (step in intervention_set and current_hidden_raw is None)
            if need_orig_call:
                text_orig = obs_to_text(obs_np)
                prompt_orig = template.replace("{current_state_filtered}", filter_text_obs(text_orig))
                g_orig = call_gemini_retry(prompt_orig, api_key, model=gemini_model, use_thinking=use_thinking)
                current_narrative_orig = g_orig["text"]
                current_hidden_raw = embedder.embed(current_narrative_orig)

            hidden_orig_normed = (current_hidden_raw - hidden_mean) / hidden_std
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=args.device).unsqueeze(0)
            ho_t = torch.tensor(hidden_orig_normed, dtype=torch.float32, device=args.device).unsqueeze(0)

            if step in intervention_set:
                flipped_np = flip_obs_180(obs_np)
                text_flipped = obs_to_text(flipped_np)
                prompt_flipped = template.replace("{current_state_filtered}", filter_text_obs(text_flipped))
                g_flipped = call_gemini_retry(prompt_flipped, api_key, model=gemini_model, use_thinking=use_thinking)
                narr_flipped = g_flipped["text"]
                hid_flipped_raw = embedder.embed(narr_flipped)
                hid_flipped_normed = (hid_flipped_raw - hidden_mean) / hidden_std

                flip_t = torch.tensor(flipped_np, dtype=torch.float32, device=args.device).unsqueeze(0)
                hf_t = torch.tensor(hid_flipped_normed, dtype=torch.float32, device=args.device).unsqueeze(0)

                with torch.no_grad():
                    pi_A, _ = model(obs_t, ho_t)
                    pi_B, _ = model(flip_t, ho_t)
                    pi_C, _ = model(obs_t, hf_t)

                act_A, probs_A = movement_argmax_and_probs(pi_A)
                act_B, probs_B = movement_argmax_and_probs(pi_B)
                act_C, probs_C = movement_argmax_and_probs(pi_C)

                ep_records.append({
                    "episode": ep + 1,
                    "step": step,
                    "narrative_orig": current_narrative_orig,
                    "narrative_flipped": narr_flipped,
                    "action_A": act_A, "action_B": act_B, "action_C": act_C,
                    "action_A_name": MOVE_NAMES[act_A],
                    "action_B_name": MOVE_NAMES[act_B],
                    "action_C_name": MOVE_NAMES[act_C],
                    "probs_A": probs_A, "probs_B": probs_B, "probs_C": probs_C,
                })

                headline_o = current_narrative_orig.split("\n")[0][:60] if current_narrative_orig else ""
                headline_f = narr_flipped.split("\n")[0][:60] if narr_flipped else ""
                print(f"  ep {ep+1:2d} step {step:3d}  A={MOVE_NAMES[act_A]:<5s} B={MOVE_NAMES[act_B]:<5s} C={MOVE_NAMES[act_C]:<5s}  "
                      f"orig: {headline_o} | flipped: {headline_f}")
            else:
                with torch.no_grad():
                    pi_A, _ = model(obs_t, ho_t)

            action = int(pi_A.sample().item())
            rng, step_key = jax.random.split(rng)
            obs, env_state, _reward, done_jax, _info = env.step(
                step_key, env_state, action, env_params)
            done = bool(done_jax)

        for rec in ep_records:
            out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        print(f"ep {ep+1:2d}: {len(ep_records)} intervention records "
              f"(done_early={done and step < max_step})")

    out_f.close()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
