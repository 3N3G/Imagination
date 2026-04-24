"""Random-direction control for in_dist_embed_arith probe.

Tests whether the policy responds to direction structure (specific
content) or to direction magnitude (anything off-distribution).

For each policy:
  - Run α-sweep with the c_grounded_die_v2 direction (signal)
  - Run α-sweep with N=10 random gaussian directions matched in norm
    to c_grounded_die_v2 direction (controls)
  - Compare argmax flip rate, KL, ΔV at α=+2

If random directions produce smaller effects than the signal direction,
the steerability is content-specific. If similar effect, it's just
noise sensitivity.

Usage:
  PYTHONPATH=. python tools/in_dist_embed_arith_random_control.py \
      --num-samples 300 --num-random 10 \
      --out-dir probe_results/in_dist_embed_arith_random_ctrl
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
SIGNAL_DIRECTION = "probe_results/embed_directions/c_grounded_die_v2.npy"
ALPHAS_TEST = [-2.0, -1.0, +1.0, +2.0]


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


@torch.no_grad()
def forward_batch(model, obs_t, hid_t):
    pi, value = model(obs_t, hid_t)
    logits = pi.logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    values = value.squeeze(-1).cpu().numpy()
    return probs, values


def kl(p, q, eps=1e-8):
    return np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)


def measure_direction(model, obs_t, hid_all, h_mean, h_std, direction, base_probs, base_argmax, base_values, alphas):
    """Return per-alpha (flip%, mean KL, ΔV)."""
    out = {}
    for alpha in alphas:
        hid_mod = hid_all + alpha * direction
        hid_norm = (hid_mod - h_mean) / h_std
        hid_t = torch.tensor(hid_norm, dtype=torch.float32, device=obs_t.device)
        p, v = forward_batch(model, obs_t, hid_t)
        am = p.argmax(axis=-1)
        out[alpha] = {
            "flip_pct": float((am != base_argmax).mean() * 100),
            "mean_kl": float(kl(base_probs, p).mean()),
            "value_delta": float((v - base_values).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--num-random", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    signal_dir = np.load(SIGNAL_DIRECTION).astype(np.float32)
    signal_norm = float(np.linalg.norm(signal_dir))
    print(f"signal direction: {SIGNAL_DIRECTION}  norm={signal_norm:.4f}", flush=True)

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
        obs_t = torch.tensor(obs_all, dtype=torch.float32, device=device)
        hid_t = torch.tensor((hid_all - h_mean) / h_std, dtype=torch.float32, device=device)
        base_probs, base_values = forward_batch(model, obs_t, hid_t)
        base_argmax = base_probs.argmax(axis=-1)

        # signal
        sig = measure_direction(model, obs_t, hid_all, h_mean, h_std,
                                signal_dir, base_probs, base_argmax, base_values, ALPHAS_TEST)
        # randoms (norm-matched)
        rnd_results = []
        for k in range(args.num_random):
            rdir = rng.normal(size=h_mean.shape[0]).astype(np.float32)
            rdir = rdir / np.linalg.norm(rdir) * signal_norm
            r = measure_direction(model, obs_t, hid_all, h_mean, h_std,
                                  rdir, base_probs, base_argmax, base_values, ALPHAS_TEST)
            rnd_results.append(r)

        # Aggregate random
        rnd_summary = {}
        for alpha in ALPHAS_TEST:
            flips = np.array([r[alpha]["flip_pct"] for r in rnd_results])
            kls   = np.array([r[alpha]["mean_kl"]  for r in rnd_results])
            vds   = np.array([r[alpha]["value_delta"] for r in rnd_results])
            rnd_summary[alpha] = {
                "flip_pct_mean": float(flips.mean()),
                "flip_pct_std":  float(flips.std()),
                "flip_pct_p95":  float(np.percentile(flips, 95)),
                "mean_kl_mean":  float(kls.mean()),
                "mean_kl_std":   float(kls.std()),
                "value_delta_mean": float(vds.mean()),
                "value_delta_std":  float(vds.std()),
            }

        print(f"  norm-matched control ({args.num_random} random gaussian dirs):")
        print(f"  {'alpha':>6} {'sig flip%':>10} {'rand flip% (mean ± std)':>26} "
              f"{'sig KL':>8} {'rand KL':>10} {'sig ΔV':>8} {'rand ΔV':>10}")
        for alpha in ALPHAS_TEST:
            s = sig[alpha]; r = rnd_summary[alpha]
            sig_vs_rnd_z = (s['flip_pct'] - r['flip_pct_mean']) / max(r['flip_pct_std'], 0.1)
            print(f"  {alpha:>6.1f} {s['flip_pct']:>9.1f}% "
                  f"  {r['flip_pct_mean']:>6.1f} ± {r['flip_pct_std']:>5.1f}     "
                  f"{s['mean_kl']:>8.3f} {r['mean_kl_mean']:>10.3f} "
                  f"{s['value_delta']:>+8.3f} {r['value_delta_mean']:>+10.3f}   "
                  f"z(sig vs rand)={sig_vs_rnd_z:+.1f}")

        summary[track_name] = {"signal": sig, "random_summary": rnd_summary,
                               "random_per_dir": rnd_results,
                               "signal_norm": signal_norm}

    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {args.out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
