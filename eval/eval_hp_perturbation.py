#!/usr/bin/env python3
"""
HP/Food/Drink perturbation eval for the imagination-augmented AWR policy.

At each Gemini-call step:
  1. Generate real future text with the true text obs (policy uses this to act).
  2. Also generate future text where a single field in the obs fed to Gemini is
     perturbed to an extreme value: Health∈{1.0, 9.0}, Food∈{1, 9}.
     (Symbolic obs → policy's obs_branch is NOT perturbed. We only change what
     Gemini sees and embed the resulting different future narrative.)
  3. Forward policy on the SAME symbolic obs but with each perturbed hidden.
     Record ΔV, KL(π_real || π_pert), arg-change, and the text of each future.

Output:
  <out>/probes.jsonl            one row per (episode, step, perturbation) probe
  <out>/episodes.jsonl          one row per episode (return, length, etc.)
  <out>/summary.json            aggregate stats across all episodes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
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
from models.actor_critic_aug import (
    ActorCriticAug as ActorCriticAugBase,
    ActorCriticAugLN,
    ActorCriticAugV2,
    ActorCriticAugGated,
    ActorCriticHiddenOnly,
)
from pipeline.config import (
    ACTION_NAMES,
    EMBED_HIDDEN_DIM,
    GEMINI_MODEL,
)

from eval.eval_online import (
    call_gemini,
    make_embedder,
    STEP_CADENCE,
    ACTION_DIM,
    OBS_DIM,
    PREDICT_TEMPLATE_PATH,
)
from pipeline.embed import extract_prediction_suffix


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------
def perturb(text_obs: str, field: str, value) -> str:
    """Overwrite a single intrinsics field in the obs_to_text output."""
    if field == "health":
        return re.sub(r"Health:[\d.]+", f"Health:{float(value):.1f}", text_obs, count=1)
    if field == "food":
        return re.sub(r"Food:\d+", f"Food:{int(value)}", text_obs, count=1)
    if field == "drink":
        return re.sub(r"Drink:\d+", f"Drink:{int(value)}", text_obs, count=1)
    raise ValueError(f"Unknown field: {field}")


# Perturbation set: (tag, field, value).
PROBES = [
    ("health_low",  "health", 1.0),
    ("health_high", "health", 9.0),
    ("food_low",    "food",   1),
    ("food_high",   "food",   9),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hidden-stats", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--embed-backend", type=str, default="qwen3_gen",
                   choices=["qwen3_gen", "qwen3_embed", "gemini_embed"])
    p.add_argument("--hidden-dim", type=int, default=0,
                   help="0 = use EMBED_HIDDEN_DIM (4096) or 3072 for gemini_emb.")
    p.add_argument("--layer-width", type=int, default=512)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--probe-every", type=int, default=1,
                   help="How often (in Gemini-call units) to run perturbation probes. "
                        "1 = at every Gemini call; 2 = every other, etc.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gemini-model", type=str, default=GEMINI_MODEL)
    p.add_argument("--arch-v2", action="store_true")
    p.add_argument("--arch-gated", action="store_true")
    p.add_argument("--arch-hidden-only", action="store_true")
    p.add_argument("--no-layernorm", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--extract-prediction-only", action="store_true",
                   help="Embed only the Prediction: suffix of Gemini output "
                        "(matches predonly-trained policies).")
    args = p.parse_args()

    def _maybe_slice(text: str) -> str:
        if getattr(args, "extract_prediction_only", False):
            suffix, _ = extract_prediction_suffix(text)
            return suffix
        return text

    device = args.device
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set")

    use_thinking = args.gemini_model.startswith("gemini-2.5")

    # Pick model class (mirrors eval_online.py).
    if args.arch_hidden_only:
        ModelClass = ActorCriticHiddenOnly
    elif args.arch_gated:
        ModelClass = ActorCriticAugGated
    elif args.arch_v2:
        ModelClass = ActorCriticAugV2
    elif args.no_layernorm:
        ModelClass = ActorCriticAugBase
    else:
        ModelClass = ActorCriticAugLN
    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else EMBED_HIDDEN_DIM
    kwargs = dict(obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                  layer_width=args.layer_width, hidden_state_dim=hidden_dim)
    if ModelClass != ActorCriticAugBase:
        kwargs["dropout"] = args.dropout
    model = ModelClass(**kwargs).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Policy loaded: {args.checkpoint}  (hidden_dim={hidden_dim})")

    stats = np.load(args.hidden_stats)
    hidden_mean = stats["mean"].astype(np.float32)
    hidden_std = stats["std"].astype(np.float32)

    embedder = make_embedder(
        backend=args.embed_backend, device=device, api_key=api_key,
        output_dim=hidden_dim,
    )

    template = PREDICT_TEMPLATE_PATH.read_text()

    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probes_f = (out_dir / "probes.jsonl").open("w")
    episodes_f = (out_dir / "episodes.jsonl").open("w")

    def forward(obs_np, hidden_vec):
        h = (hidden_vec - hidden_mean) / hidden_std
        o = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        h = torch.tensor(h, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            pi, value = model(o, h)
            probs = pi.probs[0].cpu().numpy().astype(np.float32)
            v = float(value.item())
            argmax = int(np.argmax(probs))
        return v, probs, argmax

    def kl(p, q, eps=1e-8):
        return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

    total_probes = 0
    total_gemini_calls = 0
    probe_counter = 0

    for ep in range(args.num_episodes):
        print(f"\n===== Episode {ep + 1}/{args.num_episodes} =====")
        rng, rk = jax.random.split(rng)
        obs, env_state = env.reset(rk, env_params)

        done = False
        step = 0
        ep_return = 0.0
        current_hidden = np.zeros(hidden_dim, dtype=np.float32)
        current_text = ""
        ep_probe_count = 0

        while not done and step < 10000:
            obs_np = np.array(obs, dtype=np.float32)

            if step % STEP_CADENCE == 0:
                try:
                    text_obs = obs_to_text(obs_np)
                    filtered = filter_text_obs(text_obs)
                    prompt = template.replace("{current_state_filtered}", filtered)

                    t0 = time.perf_counter()
                    gem = call_gemini(prompt, api_key, model=args.gemini_model,
                                      use_thinking=use_thinking)
                    total_gemini_calls += 1
                    current_text = gem["text"]
                    current_hidden = embedder.embed(_maybe_slice(current_text))
                    real_latency = time.perf_counter() - t0

                    # Probe run at every `probe_every` Gemini call.
                    do_probe = (probe_counter % args.probe_every == 0)
                    probe_counter += 1
                    if do_probe:
                        v_real, probs_real, arg_real = forward(obs_np, current_hidden)

                        for tag, field, value in PROBES:
                            try:
                                pert_text_obs = perturb(text_obs, field, value)
                                pert_filtered = filter_text_obs(pert_text_obs)
                                pert_prompt = template.replace(
                                    "{current_state_filtered}", pert_filtered)
                                gem_p = call_gemini(
                                    pert_prompt, api_key, model=args.gemini_model,
                                    use_thinking=use_thinking)
                                total_gemini_calls += 1
                                pert_text = gem_p["text"]
                                pert_hidden = embedder.embed(_maybe_slice(pert_text))

                                v_p, probs_p, arg_p = forward(obs_np, pert_hidden)

                                row = {
                                    "episode": ep + 1,
                                    "step": step,
                                    "probe": tag,
                                    "field": field,
                                    "perturbed_value": value,
                                    "v_real": v_real,
                                    "v_pert": v_p,
                                    "delta_v": v_p - v_real,
                                    "kl_real_to_pert": kl(probs_real, probs_p),
                                    "argmax_real": arg_real,
                                    "argmax_pert": arg_p,
                                    "arg_changed": int(arg_real != arg_p),
                                    "hidden_norm_real": float(np.linalg.norm(current_hidden)),
                                    "hidden_norm_pert": float(np.linalg.norm(pert_hidden)),
                                    "hidden_cosine": float(
                                        np.dot(current_hidden, pert_hidden) /
                                        (np.linalg.norm(current_hidden) *
                                         np.linalg.norm(pert_hidden) + 1e-8)
                                    ),
                                    "real_text": current_text,
                                    "pert_text": pert_text,
                                }
                                probes_f.write(json.dumps(row) + "\n")
                                probes_f.flush()
                                ep_probe_count += 1
                                total_probes += 1
                            except Exception as e:
                                print(f"  [step {step}] probe {tag} failed: {e}")

                        if ep_probe_count <= 3 or ep_probe_count % 10 == 0:
                            print(f"  [step {step}] probed ({ep_probe_count}); "
                                  f"v_real={v_real:.2f}  latency={real_latency:.1f}s")

                except Exception as e:
                    print(f"  [step {step}] gemini/embed error: {e}")

            # Forward with the real hidden to get the action for this step.
            v_act, probs_act, arg_act = forward(obs_np, current_hidden)
            # Sample action from the distribution.
            action = int(np.random.choice(ACTION_DIM, p=probs_act / probs_act.sum()))

            rng, sk = jax.random.split(rng)
            obs, env_state, reward, done_arr, info = env.step(
                sk, env_state, action, env_params)
            ep_return += float(reward)
            done = bool(done_arr)
            step += 1

            if step % 200 == 0:
                print(f"  step={step}  return={ep_return:.2f}  probes={ep_probe_count}")

        ep_row = {
            "episode": ep + 1,
            "return": ep_return,
            "length": step,
            "num_probes": ep_probe_count,
        }
        episodes_f.write(json.dumps(ep_row) + "\n")
        episodes_f.flush()
        print(f"  EP {ep+1}: return={ep_return:.2f}  length={step}  probes={ep_probe_count}")

    probes_f.close()
    episodes_f.close()

    # -----------------------------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------------------------
    rows = [json.loads(l) for l in (out_dir / "probes.jsonl").open()]
    episodes = [json.loads(l) for l in (out_dir / "episodes.jsonl").open()]

    summary = {
        "checkpoint": args.checkpoint,
        "embed_backend": args.embed_backend,
        "hidden_dim": hidden_dim,
        "num_episodes": args.num_episodes,
        "total_probes": total_probes,
        "total_gemini_calls": total_gemini_calls,
        "mean_return": float(np.mean([e["return"] for e in episodes])) if episodes else 0.0,
        "by_probe": {},
    }
    for tag, _, _ in PROBES:
        sub = [r for r in rows if r["probe"] == tag]
        if not sub:
            continue
        dv = np.array([r["delta_v"] for r in sub])
        kls = np.array([r["kl_real_to_pert"] for r in sub])
        arg_changes = np.array([r["arg_changed"] for r in sub])
        cos = np.array([r["hidden_cosine"] for r in sub])
        summary["by_probe"][tag] = {
            "n": int(len(sub)),
            "mean_delta_v": float(dv.mean()),
            "mean_abs_delta_v": float(np.abs(dv).mean()),
            "std_delta_v": float(dv.std()),
            "mean_kl": float(kls.mean()),
            "arg_change_rate": float(arg_changes.mean()),
            "mean_hidden_cosine": float(cos.mean()),
        }
    with (out_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
