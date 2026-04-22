"""In-distribution semantic probe — complement to the OOD fresh-rollout probes.

For a trained augmented policy, sample N states from the *training distribution*
(merged final_trajectories shards) and run two counterfactuals per state, using
only policy forward passes (no Gemini calls):

  A. baseline      : (obs, hidden)
  B. obs CF        : (obs_from_different_state_j, hidden)   — tests obs reading
  C. hidden CF     : (obs, hidden_from_different_state_j)   — tests hidden reading
  D. hidden perm   : (obs, hidden with dims permuted)       — null baseline

For each, record:
  - argmax action
  - KL(pi_A || pi_X) over full action distribution
  - value delta V_X - V_A
  - whether the movement argmax changed (A ≠ X on {LEFT, RIGHT, UP, DOWN})

The in-distribution hidden-flip rate, compared to the OOD fresh-rollout rate,
tells us whether imagination-reading failure is OOD generalization or
fundamental content-reading failure.

Usage:
  PYTHONPATH=. python -m tools.in_distribution_semantic_probe \\
      --checkpoint /path/to/final.pth \\
      --hidden-stats /path/to/hidden_state_stats.npz \\
      --data-dir /path/to/final_trajectories_psf_v2_cadence5_predonly_gemini_emb \\
      --num-samples 500 \\
      --output-dir /path/to/eval_results/in_distribution_probe/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pipeline.config import EMBED_HIDDEN_DIM
from eval.eval_online import OBS_DIM as OBS_LEN, ACTION_DIM
from models.actor_critic_aug import (
    ActorCriticAug as ActorCriticAugBase,
    ActorCriticAugLN,
    ActorCriticAugV2,
    ActorCriticHiddenOnly,
)
from envs.obs_to_text import OBS_DIM, MAP_CHANNELS, MAP_OBS_SIZE, TOTAL_OBS_SIZE

MOVE_ACTIONS = [1, 2, 3, 4]
MOVE_NAMES = {1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN"}


def flip_obs_180(obs: np.ndarray) -> np.ndarray:
    flat = np.asarray(obs, dtype=np.float32).flatten()
    if flat.size != TOTAL_OBS_SIZE:
        raise ValueError(f"Expected {TOTAL_OBS_SIZE}, got {flat.size}")
    mp = flat[:MAP_OBS_SIZE].reshape(OBS_DIM[0], OBS_DIM[1], MAP_CHANNELS)
    flipped = mp[::-1, ::-1, :].copy()
    return np.concatenate([flipped.reshape(-1), flat[MAP_OBS_SIZE:]]).astype(np.float32)


def decode_obs_from_shard(d) -> np.ndarray:
    """Reconstruct full obs from obs_map_bits + obs_aux. Mirrors label pipeline."""
    bits = np.unpackbits(d["obs_map_bits"], axis=1)[:, : int(d["obs_map_dim"])]
    aux = np.asarray(d["obs_aux"], dtype=np.float32)
    return np.concatenate([bits.astype(np.float32), aux], axis=1)


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def movement_argmax(logits: np.ndarray) -> int:
    m = np.array([logits[a] for a in MOVE_ACTIONS], dtype=np.float64)
    return MOVE_ACTIONS[int(np.argmax(m))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hidden-stats", required=True)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--num-cf-pairs", type=int, default=3,
                    help="How many random counterfactual partners per state (averaged).")
    ap.add_argument("--hidden-dim", type=int, default=3072)
    ap.add_argument("--layer-width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arch-v2", action="store_true")
    ap.add_argument("--no-layernorm", action="store_true")
    ap.add_argument("--arch-hidden-only", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy.
    if args.arch_hidden_only:
        ModelClass = ActorCriticHiddenOnly
    elif args.arch_v2:
        ModelClass = ActorCriticAugV2
    elif args.no_layernorm:
        ModelClass = ActorCriticAugBase
    else:
        ModelClass = ActorCriticAugLN
    kwargs = dict(obs_dim=OBS_LEN, action_dim=ACTION_DIM,
                  layer_width=args.layer_width, hidden_state_dim=args.hidden_dim)
    if ModelClass != ActorCriticAugBase:
        kwargs["dropout"] = args.dropout
    model = ModelClass(**kwargs).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device,
                                     weights_only=True))
    model.eval()
    print(f"Policy: {args.checkpoint}  ({ModelClass.__name__})")

    stats = np.load(args.hidden_stats)
    hidden_mean = stats["mean"].astype(np.float32)
    hidden_std = stats["std"].astype(np.float32)

    # Load enough shards to sample from.
    shards = sorted(args.data_dir.glob("trajectories_*.npz"))
    if not shards:
        raise SystemExit(f"No shards in {args.data_dir}")
    rng = np.random.default_rng(args.seed)

    # Sample files uniformly until we have >= num_samples + cf_pairs*num_samples states.
    print(f"Loading first shard {shards[0].name} for state pool")
    d = np.load(shards[0], allow_pickle=True)
    obs_flat = decode_obs_from_shard(d)
    hidden_raw = np.asarray(d["hidden_state"], dtype=np.float32)
    if obs_flat.shape[0] < args.num_samples + args.num_cf_pairs * args.num_samples:
        print(f"WARN: shard 0 has {obs_flat.shape[0]} rows; may need more shards")
    pool_size = obs_flat.shape[0]
    print(f"  pool size: {pool_size}")

    idx_sample = rng.choice(pool_size, size=args.num_samples, replace=False)

    def normed(h):
        return (h - hidden_mean) / hidden_std

    def forward(obs_np, hidden_raw_np):
        h = normed(hidden_raw_np)
        o_t = torch.tensor(obs_np, dtype=torch.float32, device=args.device).unsqueeze(0)
        h_t = torch.tensor(h, dtype=torch.float32, device=args.device).unsqueeze(0)
        with torch.no_grad():
            pi, value = model(o_t, h_t)
            logits = pi.logits.squeeze(0).cpu().numpy()
            probs = pi.probs.squeeze(0).cpu().numpy().astype(np.float64)
            probs = probs / probs.sum()
            v = float(value.item())
        return logits, probs, v

    # Accumulators.
    results = {
        "obs_flip_rate": [],        # argmax changes when obs is flipped 180°
        "obs_cf_rate": [],          # argmax changes when obs swapped with a random state's obs
        "hidden_cf_rate": [],       # argmax changes when hidden swapped with a random state's hidden
        "hidden_perm_rate": [],     # argmax changes when hidden is permuted
        "kl_obs_flip": [], "kl_obs_cf": [], "kl_hidden_cf": [], "kl_hidden_perm": [],
        "dv_obs_flip": [], "dv_obs_cf": [], "dv_hidden_cf": [], "dv_hidden_perm": [],
    }

    # Permutation for hidden perm CF (fixed across all samples).
    perm = rng.permutation(args.hidden_dim)

    for i, ix in enumerate(idx_sample):
        obs_a = obs_flat[ix]
        h_a = hidden_raw[ix]

        # CF partners (pick k random indices distinct from ix).
        cf_idx = rng.choice(pool_size, size=args.num_cf_pairs, replace=False)
        cf_idx = [int(j) for j in cf_idx if j != ix][: args.num_cf_pairs]
        while len(cf_idx) < args.num_cf_pairs:
            j = int(rng.integers(pool_size))
            if j != ix:
                cf_idx.append(j)

        logits_a, probs_a, v_a = forward(obs_a, h_a)
        move_a = movement_argmax(logits_a)

        # B. obs flipped 180°
        logits_b, probs_b, v_b = forward(flip_obs_180(obs_a), h_a)
        move_b = movement_argmax(logits_b)
        results["obs_flip_rate"].append(int(move_b != move_a))
        results["kl_obs_flip"].append(kl(probs_a, probs_b))
        results["dv_obs_flip"].append(v_b - v_a)

        # B'. obs from another state (in-distribution obs CF)
        obs_rates = []
        for j in cf_idx:
            logits_x, probs_x, v_x = forward(obs_flat[j], h_a)
            obs_rates.append((int(movement_argmax(logits_x) != move_a),
                              kl(probs_a, probs_x), v_x - v_a))
        r_, k_, dv_ = zip(*obs_rates)
        results["obs_cf_rate"].append(float(np.mean(r_)))
        results["kl_obs_cf"].append(float(np.mean(k_)))
        results["dv_obs_cf"].append(float(np.mean(dv_)))

        # C. hidden from another state
        hid_rates = []
        for j in cf_idx:
            logits_x, probs_x, v_x = forward(obs_a, hidden_raw[j])
            hid_rates.append((int(movement_argmax(logits_x) != move_a),
                              kl(probs_a, probs_x), v_x - v_a))
        r_, k_, dv_ = zip(*hid_rates)
        results["hidden_cf_rate"].append(float(np.mean(r_)))
        results["kl_hidden_cf"].append(float(np.mean(k_)))
        results["dv_hidden_cf"].append(float(np.mean(dv_)))

        # D. hidden permuted (null)
        h_perm = h_a[perm]
        logits_d, probs_d, v_d = forward(obs_a, h_perm)
        results["hidden_perm_rate"].append(int(movement_argmax(logits_d) != move_a))
        results["kl_hidden_perm"].append(kl(probs_a, probs_d))
        results["dv_hidden_perm"].append(v_d - v_a)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(idx_sample)}]")

    # Summary.
    def pct(v):
        return f"{100*np.mean(v):.1f}%"

    print(f"\n=== In-distribution semantic probe ({len(idx_sample)} samples) ===")
    print(f"  obs_flip_180    : {pct(results['obs_flip_rate'])}  "
          f"KL={np.mean(results['kl_obs_flip']):.3f}  dV={np.mean(results['dv_obs_flip']):+.3f}")
    print(f"  obs_cf (random) : {pct(results['obs_cf_rate'])}  "
          f"KL={np.mean(results['kl_obs_cf']):.3f}  dV={np.mean(results['dv_obs_cf']):+.3f}")
    print(f"  hidden_cf       : {pct(results['hidden_cf_rate'])}  "
          f"KL={np.mean(results['kl_hidden_cf']):.3f}  dV={np.mean(results['dv_hidden_cf']):+.3f}")
    print(f"  hidden_perm     : {pct(results['hidden_perm_rate'])}  "
          f"KL={np.mean(results['kl_hidden_perm']):.3f}  dV={np.mean(results['dv_hidden_perm']):+.3f}")

    summary = {k: (float(np.mean(v)) if len(v) else None) for k, v in results.items()}
    summary["num_samples"] = len(idx_sample)
    summary["checkpoint"] = str(args.checkpoint)
    summary["data_dir"] = str(args.data_dir)

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(args.output_dir / "raw.npz", **{k: np.array(v) for k, v in results.items()})
    print(f"\nWrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
