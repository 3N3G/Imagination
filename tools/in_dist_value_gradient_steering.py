"""Compute the value-gradient direction in embedding space for each policy.

For a given policy and a sample of training states (obs, hidden):
  - Compute g_i = ∂V/∂hidden | (obs_i, hidden_i)
  - Average across i (after L2-normalizing each gradient) → mean value-gradient
    direction.

Then α-sweep along this direction to test:
  - Does pushing along this direction monotonically increase V?
  - Does the argmax distribution shift?
  - How does this compare to the death direction's value-shifting effect?

This finds the SINGLE BEST direction (in a linearized sense) for steering
the value head. If the policy reads ANY meaningful content axis, this is
the axis that aligns with reading the answer "this is a state worth being
in".

Usage:
  PYTHONPATH=. python tools/in_dist_value_gradient_steering.py \
      --num-samples 300 --out-dir probe_results/value_grad_steer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.eval_online import OBS_DIM as OBS_LEN, ACTION_DIM
from models.actor_critic_aug import ActorCriticAugLN
from tools.in_distribution_semantic_probe import decode_obs_from_shard

CELLS = [
    ("a_full", {
        "ckpt":  "/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly/freezenone/final.pth",
        "stats": "/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly/freezenone/hidden_state_stats.npz",
        "data":  "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories_psf_v2_cadence5_predonly_gemini_emb",
    }),
    ("c_grounded_2M", {
        "ckpt":  "/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_grounded_predonly_top2M/freezenone/final.pth",
        "stats": "/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_grounded_predonly_top2M/freezenone/hidden_state_stats.npz",
        "data":  "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories_psf_v2_cadence5_grounded_predonly_gemini_emb_top2M",
    }),
]
ALPHAS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
DIE_DIR_PATHS = {
    "a_full": "probe_results/embed_directions/a_full_die_v2.npy",
    "c_grounded_2M": "probe_results/embed_directions/c_grounded_die_v2.npy",
}


def load_states(data_dir: Path, num_samples: int, rng: np.random.Generator):
    files = sorted(data_dir.glob("trajectories_*.npz"))[:5]
    obs_list, hid_list = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        obs = decode_obs_from_shard(d)
        hid = np.asarray(d["hidden_state"], dtype=np.float32)
        T = min(len(obs), len(hid))
        n = min(num_samples // len(files) + 5, T)
        idx = rng.choice(T, size=n, replace=False)
        obs_list.append(obs[idx])
        hid_list.append(hid[idx])
        if sum(len(o) for o in obs_list) >= num_samples:
            break
    return (np.concatenate(obs_list, axis=0)[:num_samples],
            np.concatenate(hid_list, axis=0)[:num_samples])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    summary = {}

    for track_name, cfg in CELLS:
        print(f"\n========= track {track_name} =========", flush=True)
        stats = np.load(cfg["stats"])
        h_mean = stats["mean"].astype(np.float32)
        h_std = stats["std"].astype(np.float32)

        ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=True)
        model = ActorCriticAugLN(obs_dim=OBS_LEN, action_dim=ACTION_DIM,
                                 layer_width=512, hidden_state_dim=h_mean.shape[0],
                                 dropout=0.0).to(device)
        model.load_state_dict(ckpt)
        model.eval()

        obs_all, hid_all = load_states(Path(cfg["data"]), args.num_samples, rng)

        # Compute gradients dV/dhidden for each sample.
        # Note: ZG normalizes via (h - h_mean) / h_std before the model.
        # We compute gradient w.r.t. RAW hidden so the direction is in the same
        # space as die_v2 direction.
        h_mean_t = torch.tensor(h_mean, device=device)
        h_std_t = torch.tensor(h_std, device=device)
        grads = []
        chunk = 32
        for i in range(0, len(obs_all), chunk):
            obs_b = torch.tensor(obs_all[i:i+chunk], dtype=torch.float32, device=device)
            hid_b = torch.tensor(hid_all[i:i+chunk], dtype=torch.float32, device=device,
                                 requires_grad=True)
            hid_norm = (hid_b - h_mean_t) / h_std_t
            _, value = model(obs_b, hid_norm)
            value_sum = value.sum()
            grad = torch.autograd.grad(value_sum, hid_b)[0]  # raw-space grad
            grads.append(grad.detach().cpu().numpy())
        grad_arr = np.concatenate(grads, axis=0)  # (N, 3072)
        # L2-normalize each then average
        norms = np.linalg.norm(grad_arr, axis=1, keepdims=True)
        grad_norm = grad_arr / np.clip(norms, 1e-8, None)
        mean_grad_dir = grad_norm.mean(axis=0)
        # Then re-scale to die direction's norm so we can compare flip rates
        die_dir = np.load(DIE_DIR_PATHS[track_name]).astype(np.float32)
        target_norm = float(np.linalg.norm(die_dir))
        cur_norm = float(np.linalg.norm(mean_grad_dir))
        value_grad_dir = mean_grad_dir / cur_norm * target_norm
        cos_with_die = float(np.dot(value_grad_dir, die_dir) /
                             (np.linalg.norm(value_grad_dir) * np.linalg.norm(die_dir)))
        print(f"  mean_grad_dir norm before rescale: {cur_norm:.6f}", flush=True)
        print(f"  rescaled to target_norm = {target_norm:.4f} to match die direction", flush=True)
        print(f"  cos(value_grad_dir, die_dir) = {cos_with_die:+.4f}", flush=True)

        # Now sweep alpha along this direction
        obs_t = torch.tensor(obs_all, dtype=torch.float32, device=device)
        with torch.no_grad():
            base_hid_t = torch.tensor((hid_all - h_mean) / h_std, dtype=torch.float32, device=device)
            base_pi, base_value = model(obs_t, base_hid_t)
            base_probs = F.softmax(base_pi.logits, dim=-1).cpu().numpy()
            base_argmax = base_probs.argmax(axis=-1)
            base_values_np = base_value.squeeze(-1).cpu().numpy()

        sweep = {}
        for alpha in ALPHAS:
            hid_mod = hid_all + alpha * value_grad_dir
            hid_norm = (hid_mod - h_mean) / h_std
            with torch.no_grad():
                hid_t = torch.tensor(hid_norm, dtype=torch.float32, device=device)
                pi, value = model(obs_t, hid_t)
                probs = F.softmax(pi.logits, dim=-1).cpu().numpy()
                values = value.squeeze(-1).cpu().numpy()
            argmax = probs.argmax(axis=-1)
            sweep[alpha] = {
                "flip_pct":         float((argmax != base_argmax).mean() * 100),
                "mean_value":       float(values.mean()),
                "value_delta":      float((values - base_values_np).mean()),
                "mean_left":        float(probs[:, 1].mean()),
                "mean_right":       float(probs[:, 2].mean()),
                "mean_do":          float(probs[:, 5].mean()),
            }

        print(f"  {'alpha':>6} {'flip%':>7} {'mean_V':>10} {'ΔV':>9} {'L':>6} {'R':>6} {'DO':>6}", flush=True)
        for alpha in ALPHAS:
            s = sweep[alpha]
            print(f"  {alpha:>6.1f} {s['flip_pct']:>6.1f}% {s['mean_value']:>10.4f} "
                  f"{s['value_delta']:>+9.4f} {s['mean_left']:>6.3f} {s['mean_right']:>6.3f} "
                  f"{s['mean_do']:>6.3f}", flush=True)

        summary[track_name] = {
            "value_grad_norm_pre_rescale": cur_norm,
            "value_grad_norm_post_rescale": target_norm,
            "cos_value_grad_with_die_dir": cos_with_die,
            "alpha_sweep": {f"{a:+.1f}": sweep[a] for a in ALPHAS},
        }
        np.save(args.out_dir / f"{track_name}_value_grad_dir.npy", value_grad_dir.astype(np.float32))

    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {args.out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
