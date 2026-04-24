"""In-distribution embedding-arithmetic probe (no Gemini calls).

For each (track, direction) pair:
  - Sample N states from a training shard
  - For each α ∈ ALPHAS:
    - Compute hidden' = real_hidden + α * direction (raw embedding space)
    - Normalize by training hidden_state_stats
    - Forward policy → action distribution, value
    - Record argmax change vs α=0, KL(π_α | π_0), V_α - V_0,
      and per-action (LEFT, RIGHT, UP, DOWN, DESCEND, DO, SLEEP) probability shift.

Output: probe_results/in_dist_embed_arith/<track>_<direction>.json
       per-α aggregate metrics, and summary.json across all cells.

This is FAST — runs entire matrix in minutes on a single GPU.

Usage:
  PYTHONPATH=. python tools/in_dist_embed_arith_sweep.py \
      --num-samples 500 --out-dir probe_results/in_dist_embed_arith
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.config import EMBED_HIDDEN_DIM
from eval.eval_online import OBS_DIM as OBS_LEN, ACTION_DIM
from models.actor_critic_aug import ActorCriticAugLN
from tools.in_distribution_semantic_probe import decode_obs_from_shard

ACTION_NAMES = [
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
    "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD", "MAKE_IRON_SWORD", "REST", "DESCEND", "ASCEND",
    "MAKE_DIAMOND_PICKAXE", "MAKE_DIAMOND_SWORD", "MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOUR", "SHOOT_ARROW", "MAKE_ARROW", "CAST_FIREBALL",
    "CAST_ICEBALL", "PLACE_TORCH", "DRINK_POTION_RED", "DRINK_POTION_GREEN",
    "DRINK_POTION_BLUE", "DRINK_POTION_PINK", "DRINK_POTION_CYAN",
    "DRINK_POTION_YELLOW", "READ_BOOK", "ENCHANT_SWORD", "ENCHANT_ARMOUR",
    "MAKE_TORCH", "LEVEL_UP_DEX", "LEVEL_UP_STR", "LEVEL_UP_INT",
    "ENCHANT_BOW",
]
TRACK_ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN", "DO", "DESCEND", "SLEEP",
                 "PLACE_STONE", "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE"]

# (track_name, ckpt_path, data_dir, dir_name in probe_results/embed_directions)
# Cross-track transfer: each policy is tested with directions from BOTH its own
# data and the other policy's data, to test generality of direction vectors.
CELLS = [
    ("a_full", {
        "ckpt":  "/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly/freezenone/final.pth",
        "stats": "/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly/freezenone/hidden_state_stats.npz",
        "data":  "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories_psf_v2_cadence5_predonly_gemini_emb",
        "directions": [
            "a_full_die_v2",
            "c_grounded_die_v2",            # cross-track transfer
            "c_grounded_avoid_animals_v2",  # cross-track transfer
        ],
    }),
    ("c_grounded_2M", {
        "ckpt":  "/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_grounded_predonly_top2M/freezenone/final.pth",
        "stats": "/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_grounded_predonly_top2M/freezenone/hidden_state_stats.npz",
        "data":  "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories_psf_v2_cadence5_grounded_predonly_gemini_emb_top2M",
        "directions": [
            "c_grounded_die_v2",
            "c_grounded_avoid_animals_v2",
            "a_full_die_v2",                # cross-track transfer
        ],
    }),
]
ALPHAS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
DIR_ROOT = Path("probe_results/embed_directions")


def load_states(data_dir: Path, num_samples: int, rng: np.random.Generator):
    files = sorted(data_dir.glob("trajectories_*.npz"))[:5]
    if not files:
        raise SystemExit(f"No trajectory files in {data_dir}")
    obs_list, hid_list = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        obs = decode_obs_from_shard(d)
        hid = np.asarray(d["hidden_state"], dtype=np.float32)
        # Take a few from each file
        T = min(len(obs), len(hid))
        n = min(num_samples // len(files) + 5, T)
        idx = rng.choice(T, size=n, replace=False)
        obs_list.append(obs[idx])
        hid_list.append(hid[idx])
        if sum(len(o) for o in obs_list) >= num_samples:
            break
    obs_all = np.concatenate(obs_list, axis=0)[:num_samples]
    hid_all = np.concatenate(hid_list, axis=0)[:num_samples]
    return obs_all, hid_all


@torch.no_grad()
def forward_batch(model, obs_batch: torch.Tensor, hid_batch: torch.Tensor):
    pi, value = model(obs_batch, hid_batch)
    logits = pi.logits if hasattr(pi, "logits") else pi.distribution.logits  # torch.Categorical
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    values = value.squeeze(-1).cpu().numpy()
    return probs, values


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    summary = {}

    for track_name, cfg in CELLS:
        print(f"\n========= track {track_name} =========", flush=True)
        # Load stats
        stats = np.load(cfg["stats"])
        h_mean = stats["mean"].astype(np.float32)
        h_std = stats["std"].astype(np.float32)

        # Load model
        ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=True)
        model = ActorCriticAugLN(obs_dim=OBS_LEN, action_dim=ACTION_DIM,
                                 layer_width=512, hidden_state_dim=h_mean.shape[0],
                                 dropout=0.0).to(device)
        model.load_state_dict(ckpt)
        model.eval()
        print(f"  loaded model from {cfg['ckpt']}", flush=True)

        # Load states
        obs_all, hid_all = load_states(Path(cfg["data"]), args.num_samples, rng)
        print(f"  states: obs {obs_all.shape}, hidden {hid_all.shape}", flush=True)

        for dir_name in cfg["directions"]:
            dir_path = DIR_ROOT / f"{dir_name}.npy"
            if not dir_path.exists():
                print(f"  MISSING direction {dir_path}", flush=True); continue
            direction = np.load(dir_path).astype(np.float32)
            print(f"\n  -- direction {dir_name} (norm={np.linalg.norm(direction):.4f}) --", flush=True)

            obs_t = torch.tensor(obs_all, dtype=torch.float32, device=device)

            # Forward at each alpha
            cell_results: Dict[float, Dict] = {}
            argmax_per_alpha: Dict[float, np.ndarray] = {}
            probs_per_alpha: Dict[float, np.ndarray] = {}
            values_per_alpha: Dict[float, np.ndarray] = {}
            for alpha in ALPHAS:
                hid_mod = hid_all + alpha * direction
                hid_norm = (hid_mod - h_mean) / h_std
                hid_t = torch.tensor(hid_norm, dtype=torch.float32, device=device)
                probs, values = forward_batch(model, obs_t, hid_t)
                argmax = probs.argmax(axis=-1)
                argmax_per_alpha[alpha] = argmax
                probs_per_alpha[alpha] = probs
                values_per_alpha[alpha] = values

            # Per-α stats relative to α=0
            base_argmax = argmax_per_alpha[0.0]
            base_probs  = probs_per_alpha[0.0]
            base_values = values_per_alpha[0.0]
            for alpha in ALPHAS:
                p = probs_per_alpha[alpha]
                v = values_per_alpha[alpha]
                am = argmax_per_alpha[alpha]
                kl_vec = kl(base_probs, p)
                argmax_flip = float((am != base_argmax).mean())
                value_delta = float((v - base_values).mean())
                # per-tracked-action probability shift
                action_probs = {}
                for ac in TRACK_ACTIONS:
                    ai = ACTION_NAMES.index(ac)
                    action_probs[ac] = float(p[:, ai].mean())
                cell_results[alpha] = {
                    "alpha": alpha,
                    "argmax_flip_rate_vs_alpha0": argmax_flip,
                    "mean_kl_vs_alpha0": float(kl_vec.mean()),
                    "value_delta_vs_alpha0": value_delta,
                    "tracked_action_probs": action_probs,
                }

            # print summary table
            print(f"    {'alpha':>6} {'flip%':>7} {'KL':>7} {'ΔV':>7}  " +
                  " ".join(f"{a:>5}" for a in TRACK_ACTIONS), flush=True)
            base_actions = cell_results[0.0]["tracked_action_probs"]
            for alpha in ALPHAS:
                cr = cell_results[alpha]
                ap_str = " ".join(
                    f"{cr['tracked_action_probs'][a]:>5.2f}"
                    for a in TRACK_ACTIONS
                )
                print(f"    {alpha:>6.1f} {cr['argmax_flip_rate_vs_alpha0']*100:>6.1f}% "
                      f"{cr['mean_kl_vs_alpha0']:>7.3f} {cr['value_delta_vs_alpha0']:>+7.3f}  "
                      f"{ap_str}", flush=True)

            # monotonicity test: does p(LEFT) increase with α for direction_left direction?
            # Compute for each tracked action, the spearman-ish trend across α.
            alphas_arr = np.array(ALPHAS, dtype=np.float64)
            trend = {}
            for ac in TRACK_ACTIONS:
                vals = np.array([cell_results[a]["tracked_action_probs"][ac] for a in ALPHAS])
                # pearson with α (continuous-monotone signal)
                corr = float(np.corrcoef(alphas_arr, vals)[0, 1]) if vals.std() > 0 else 0.0
                trend[ac] = {"alpha_pearson": corr,
                             "alpha_min2_val": float(vals[0]),
                             "alpha_0_val":    float(vals[len(ALPHAS)//2]),
                             "alpha_p2_val":   float(vals[-1]),
                             "spread":         float(vals.max() - vals.min())}

            cell_id = f"{track_name}_{dir_name}"
            summary[cell_id] = {
                "track": track_name,
                "direction": dir_name,
                "direction_norm": float(np.linalg.norm(direction)),
                "n": int(len(obs_all)),
                "alphas": ALPHAS,
                "per_alpha": {f"{a:+.1f}": cell_results[a] for a in ALPHAS},
                "monotonicity_per_action": trend,
            }
            with (args.out_dir / f"{cell_id}.json").open("w") as f:
                json.dump(summary[cell_id], f, indent=2)

            # print monotonicity verdict
            print(f"\n    monotonicity (Pearson corr of p(action) vs α) — magnitudes >0.5 = strong:")
            for ac in TRACK_ACTIONS:
                t = trend[ac]
                mark = "**" if abs(t["alpha_pearson"]) > 0.5 else "  "
                print(f"      {mark}{ac:<8} corr={t['alpha_pearson']:+.3f}  "
                      f"α=-2:{t['alpha_min2_val']:.3f} α=0:{t['alpha_0_val']:.3f} "
                      f"α=+2:{t['alpha_p2_val']:.3f}  spread={t['spread']:.3f}")

    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {args.out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
