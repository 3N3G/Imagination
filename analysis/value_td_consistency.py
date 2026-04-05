#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from analysis.value_learning import (
    JAXValueModel,
    TorchValueModel,
    checkpoint_kind,
    explained_variance,
    extract_hidden,
    flatten_obs,
    safe_corr,
)


@dataclass
class DatasetMeta:
    files: List[Path]
    counts: List[int]
    total_samples: int
    obs_dim: int
    hidden_dim: int


def index_dataset(dataset_dir: Path, data_glob: str, max_files: Optional[int]) -> DatasetMeta:
    files = sorted(dataset_dir.glob(data_glob))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No dataset files matched {data_glob} under {dataset_dir}")

    valid_files: List[Path] = []
    counts: List[int] = []
    obs_dim = None
    hidden_dim = None
    for path in files:
        try:
            with np.load(path, mmap_mode="r") as data:
                n = int(data["obs"].shape[0])
                valid_files.append(path)
                counts.append(n)
                if obs_dim is None:
                    obs_dim = int(np.prod(data["obs"].shape[1:]))
                if hidden_dim is None and "hidden_state" in data:
                    hs = data["hidden_state"]
                    hidden_dim = int(hs.shape[-1])
        except Exception:
            continue
    if not counts:
        raise RuntimeError("No readable dataset files found.")
    if obs_dim is None:
        raise RuntimeError("Failed to infer observation dimension from dataset.")
    if hidden_dim is None:
        hidden_dim = 2560
    return DatasetMeta(
        files=valid_files,
        counts=counts,
        total_samples=int(sum(counts)),
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
    )


def _shift_interleaved(arr: np.ndarray, done: np.ndarray, num_envs: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    done = np.asarray(done, dtype=np.float32).reshape(-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array to shift, got {arr.shape}")
    n, dim = arr.shape
    out = np.zeros_like(arr, dtype=np.float32)

    if num_envs > 1 and n % num_envs == 0:
        t_len = n // num_envs
        arr_3d = arr.reshape(t_len, num_envs, dim)
        done_2d = done.reshape(t_len, num_envs)
        out_3d = np.zeros_like(arr_3d, dtype=np.float32)
        out_3d[:-1] = arr_3d[1:]
        out_3d = np.where(done_2d[..., None] > 0.5, 0.0, out_3d)
        return out_3d.reshape(n, dim)

    out[:-1] = arr[1:]
    out = np.where(done[:, None] > 0.5, 0.0, out)
    return out


def _derive_next_obs(obs: np.ndarray, done: np.ndarray, num_envs: int) -> np.ndarray:
    return _shift_interleaved(obs, done, num_envs=num_envs)


def _derive_next_hidden(hidden: np.ndarray, done: np.ndarray, num_envs: int) -> np.ndarray:
    return _shift_interleaved(hidden, done, num_envs=num_envs)


def load_td_sample(
    meta: DatasetMeta,
    num_samples: int,
    seed: int,
    num_envs: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(num_samples, meta.total_samples)
    if k <= 0:
        raise ValueError("num_samples must be > 0")

    sample_global = np.sort(rng.choice(meta.total_samples, size=k, replace=False))

    obs_out = np.zeros((k, meta.obs_dim), dtype=np.float32)
    next_obs_out = np.zeros((k, meta.obs_dim), dtype=np.float32)
    hidden_out = np.zeros((k, meta.hidden_dim), dtype=np.float32)
    next_hidden_out = np.zeros((k, meta.hidden_dim), dtype=np.float32)
    reward_out = np.zeros((k,), dtype=np.float32)
    done_out = np.zeros((k,), dtype=np.float32)

    cum = 0
    cursor = 0
    for path, n in zip(meta.files, meta.counts):
        start = cum
        end = cum + n
        l = int(np.searchsorted(sample_global, start, side="left"))
        r = int(np.searchsorted(sample_global, end, side="left"))
        cum = end
        if l >= r:
            continue

        local_idx = (sample_global[l:r] - start).astype(np.int64, copy=False)
        with np.load(path) as data:
            obs = flatten_obs(data["obs"])
            if "next_obs" in data:
                next_obs = flatten_obs(data["next_obs"])
            else:
                next_obs = _derive_next_obs(obs, np.asarray(data["done"], dtype=np.float32), num_envs=num_envs)

            if "hidden_state" in data:
                hidden = extract_hidden(data["hidden_state"])
            else:
                hidden = np.zeros((obs.shape[0], meta.hidden_dim), dtype=np.float32)

            if "next_hidden_state" in data:
                next_hidden = extract_hidden(data["next_hidden_state"])
            else:
                next_hidden = _derive_next_hidden(
                    hidden,
                    np.asarray(data["done"], dtype=np.float32),
                    num_envs=num_envs,
                )

            reward = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
            done = np.asarray(data["done"], dtype=np.float32).reshape(-1)

        m = len(local_idx)
        obs_out[cursor : cursor + m] = obs[local_idx]
        next_obs_out[cursor : cursor + m] = next_obs[local_idx]
        hidden_out[cursor : cursor + m] = hidden[local_idx]
        next_hidden_out[cursor : cursor + m] = next_hidden[local_idx]
        reward_out[cursor : cursor + m] = reward[local_idx]
        done_out[cursor : cursor + m] = done[local_idx]
        cursor += m

    if cursor != k:
        raise RuntimeError(f"Dataset sampling mismatch: expected {k}, got {cursor}")

    return {
        "obs": obs_out,
        "next_obs": next_obs_out,
        "hidden": hidden_out,
        "next_hidden": next_hidden_out,
        "reward": reward_out,
        "done": done_out,
    }


def td_metrics(values: np.ndarray, next_values: np.ndarray, reward: np.ndarray, done: np.ndarray, gamma: float) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    next_values = np.asarray(next_values, dtype=np.float64).reshape(-1)
    reward = np.asarray(reward, dtype=np.float64).reshape(-1)
    done = np.asarray(done, dtype=np.float64).reshape(-1)

    target = reward + gamma * (1.0 - done) * next_values
    td_error = target - values
    abs_td = np.abs(td_error)

    out: Dict[str, object] = {
        "num_samples": int(len(values)),
        "gamma": float(gamma),
        "td_error_mean": float(td_error.mean()),
        "td_error_std": float(td_error.std()),
        "td_abs_mean": float(abs_td.mean()),
        "td_abs_median": float(np.median(abs_td)),
        "td_mse": float(np.mean(td_error ** 2)),
        "td_mae": float(np.mean(abs_td)),
        "corr_value_target": safe_corr(values, target),
        "explained_variance_value_vs_target": explained_variance(target, values),
        "fraction_abs_td_le_0.25": float(np.mean(abs_td <= 0.25)),
        "fraction_abs_td_le_0.50": float(np.mean(abs_td <= 0.50)),
        "fraction_abs_td_le_1.00": float(np.mean(abs_td <= 1.00)),
        "p90_abs_td": float(np.percentile(abs_td, 90)),
        "p99_abs_td": float(np.percentile(abs_td, 99)),
    }

    done_mask = done > 0.5
    not_done = ~done_mask
    out["terminal_fraction"] = float(np.mean(done_mask))
    if np.any(done_mask):
        out["terminal_td_abs_mean"] = float(abs_td[done_mask].mean())
    else:
        out["terminal_td_abs_mean"] = None
    if np.any(not_done):
        out["nonterminal_td_abs_mean"] = float(abs_td[not_done].mean())
    else:
        out["nonterminal_td_abs_mean"] = None

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze one-step Bellman/TD consistency on dataset transitions."
    )
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--data_glob", type=str, default="trajectories_batch_*.npz")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--hidden_mode", type=str, default="real", choices=["real", "zero"])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--action_dim", type=int, default=43)
    parser.add_argument("--layer_width", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=2560)
    parser.add_argument(
        "--normalize_hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For torch checkpoints, use hidden_state_stats.npz if available.",
    )
    parser.add_argument(
        "--skip_missing_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail_on_any_error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If any checkpoint evaluation returns status=missing/error, "
            "exit non-zero after writing output JSON."
        ),
    )
    parser.add_argument("--torch_device", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    meta = index_dataset(dataset_dir, args.data_glob, args.max_files)
    sample = load_td_sample(
        meta=meta,
        num_samples=args.num_samples,
        seed=args.seed,
        num_envs=args.num_envs,
    )

    obs = sample["obs"]
    next_obs = sample["next_obs"]
    hidden = sample["hidden"]
    next_hidden = sample["next_hidden"]
    reward = sample["reward"]
    done = sample["done"]

    results = []
    for ckpt_str in args.checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            msg = f"checkpoint_not_found: {ckpt_path}"
            if args.skip_missing_checkpoints:
                results.append({"checkpoint": ckpt_str, "status": "missing", "error": msg})
                continue
            raise FileNotFoundError(msg)

        kind = checkpoint_kind(ckpt_path)
        try:
            if kind == "torch":
                model = TorchValueModel(
                    checkpoint_path=ckpt_path,
                    obs_dim=obs.shape[1],
                    action_dim=args.action_dim,
                    layer_width=args.layer_width,
                    hidden_dim=hidden.shape[1],
                    device=args.torch_device,
                    normalize_hidden=args.normalize_hidden,
                )
                v = model.predict_values(obs=obs, hidden=hidden, hidden_mode=args.hidden_mode, batch_size=args.batch_size)
                nv = model.predict_values(
                    obs=next_obs,
                    hidden=next_hidden,
                    hidden_mode=args.hidden_mode,
                    batch_size=args.batch_size,
                )
                details = {
                    "kind": "torch",
                    "hidden_mode_eval": args.hidden_mode,
                    "hidden_stats_path": model.hidden_stats_path,
                }
            else:
                model = JAXValueModel(
                    checkpoint_path=ckpt_path,
                    obs_dim=obs.shape[1],
                    action_dim=args.action_dim,
                    layer_width=args.layer_width,
                    hidden_dim=hidden.shape[1],
                )
                v = model.predict_values(obs=obs, hidden=hidden, hidden_mode=args.hidden_mode, batch_size=args.batch_size)
                nv = model.predict_values(
                    obs=next_obs,
                    hidden=next_hidden,
                    hidden_mode=args.hidden_mode,
                    batch_size=args.batch_size,
                )
                details = {
                    "kind": "jax",
                    "load_variant": model.load_variant,
                    "uses_hidden_input": bool(model.use_aug),
                    "hidden_mode_eval": args.hidden_mode if model.use_aug else "ignored",
                    "metadata_available": model.meta is not None,
                }

            metrics = td_metrics(values=v, next_values=nv, reward=reward, done=done, gamma=args.gamma)
            results.append(
                {
                    "checkpoint": str(ckpt_path),
                    "status": "ok",
                    "details": details,
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "checkpoint": str(ckpt_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    output = {
        "dataset": {
            "dataset_dir": str(dataset_dir),
            "data_glob": args.data_glob,
            "indexed_files": len(meta.files),
            "total_samples_indexed": int(meta.total_samples),
            "obs_dim": int(obs.shape[1]),
            "hidden_dim": int(hidden.shape[1]),
            "num_samples_used": int(obs.shape[0]),
            "seed": args.seed,
            "num_envs": args.num_envs,
        },
        "evaluation": {
            "gamma": args.gamma,
            "hidden_mode": args.hidden_mode,
            "normalize_hidden_for_torch": args.normalize_hidden,
            "batch_size": args.batch_size,
        },
        "results": results,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Wrote analysis to {out_path}")
    print(json.dumps(output, indent=2))

    if args.fail_on_any_error:
        bad = [r for r in results if r.get("status") != "ok"]
        if bad:
            labels = ", ".join(f"{r.get('status')}:{r.get('checkpoint')}" for r in bad)
            raise RuntimeError(f"One or more checkpoint evaluations failed: {labels}")


if __name__ == "__main__":
    main()
