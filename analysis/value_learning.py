#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_return_to_go(
    rewards: np.ndarray, dones: np.ndarray, gamma: float, num_envs: int
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    if rewards.shape[0] != dones.shape[0]:
        raise ValueError(
            f"reward/done length mismatch: {rewards.shape[0]} vs {dones.shape[0]}"
        )
    if num_envs > 1 and rewards.shape[0] % num_envs == 0:
        rewards_mat = rewards.reshape(-1, num_envs)
        dones_mat = dones.reshape(-1, num_envs)
        returns_mat = np.zeros_like(rewards_mat, dtype=np.float32)
        next_return = np.zeros(num_envs, dtype=np.float32)
        for t in range(rewards_mat.shape[0] - 1, -1, -1):
            next_return = rewards_mat[t] + gamma * next_return * (1.0 - dones_mat[t])
            returns_mat[t] = next_return
        return returns_mat.reshape(-1)
    returns = np.zeros_like(rewards, dtype=np.float32)
    next_return = 0.0
    for t in range(rewards.shape[0] - 1, -1, -1):
        next_return = rewards[t] + gamma * next_return * (1.0 - dones[t])
        returns[t] = next_return
    return returns


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs)
    if obs.ndim <= 1:
        raise ValueError(f"Expected batched obs, got shape={obs.shape}")
    return obs.reshape(obs.shape[0], -1).astype(np.float32, copy=False)


def extract_hidden(hidden_state: np.ndarray) -> np.ndarray:
    hidden_state = np.asarray(hidden_state)
    if hidden_state.ndim == 3:
        hidden_state = hidden_state.mean(axis=1)
    if hidden_state.ndim != 2:
        raise ValueError(f"Expected hidden_state 2D or 3D, got shape={hidden_state.shape}")
    return hidden_state.astype(np.float32, copy=False)


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    unique_vals, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx in np.where(counts > 1)[0]:
            tie_mask = inverse == idx
            ranks[tie_mask] = ranks[tie_mask].mean()
    return ranks


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    var_y = float(np.var(y_true))
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def calibration_bins(values: np.ndarray, rtg: np.ndarray, num_bins: int = 10) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    rtg = np.asarray(rtg, dtype=np.float64)
    if values.size == 0:
        return {"num_bins": num_bins, "bins": []}
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    bins = []
    for i in range(num_bins):
        lo = float(quantiles[i])
        hi = float(quantiles[i + 1])
        if i < num_bins - 1:
            mask = (values >= lo) & (values < hi)
        else:
            mask = (values >= lo) & (values <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin": i,
                    "count": 0,
                    "value_lo": lo,
                    "value_hi": hi,
                    "pred_mean": None,
                    "rtg_mean": None,
                    "rtg_std": None,
                }
            )
            continue
        bins.append(
            {
                "bin": i,
                "count": count,
                "value_lo": lo,
                "value_hi": hi,
                "pred_mean": float(values[mask].mean()),
                "rtg_mean": float(rtg[mask].mean()),
                "rtg_std": float(rtg[mask].std()),
            }
        )
    valid = [b for b in bins if b["count"] > 0]
    if len(valid) >= 2:
        bin_idx = np.asarray([b["bin"] for b in valid], dtype=np.float64)
        rtg_mean = np.asarray([b["rtg_mean"] for b in valid], dtype=np.float64)
        monotonic_corr = safe_corr(bin_idx, rtg_mean)
    else:
        monotonic_corr = 0.0
    return {"num_bins": num_bins, "rtg_mean_vs_bin_corr": monotonic_corr, "bins": bins}


@dataclass
class DatasetMeta:
    files: List[Path]
    counts: List[int]
    total_samples: int
    obs_dim: int
    hidden_dim: int


def index_dataset(
    dataset_dir: Path, data_glob: str, max_files: Optional[int]
) -> DatasetMeta:
    files = sorted(dataset_dir.glob(data_glob))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"No dataset files matched {data_glob} under {dataset_dir}"
        )
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


def load_dataset_sample(
    meta: DatasetMeta,
    num_samples: int,
    seed: int,
    gamma: float,
    num_envs: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(num_samples, meta.total_samples)
    if k <= 0:
        raise ValueError("num_samples must be > 0")
    sample_global = np.sort(rng.choice(meta.total_samples, size=k, replace=False))

    obs_out = np.zeros((k, meta.obs_dim), dtype=np.float32)
    hidden_out = np.zeros((k, meta.hidden_dim), dtype=np.float32)
    rtg_out = np.zeros((k,), dtype=np.float32)

    cum = 0
    out_cursor = 0
    for path, n in zip(meta.files, meta.counts):
        start = cum
        end = cum + n
        l = int(np.searchsorted(sample_global, start, side="left"))
        r = int(np.searchsorted(sample_global, end, side="left"))
        cum = end
        if l >= r:
            continue
        local_idx = sample_global[l:r] - start
        local_idx = local_idx.astype(np.int64, copy=False)
        with np.load(path) as data:
            obs = flatten_obs(data["obs"])[local_idx]
            if "hidden_state" in data:
                hidden = extract_hidden(data["hidden_state"])[local_idx]
            else:
                hidden = np.zeros((len(local_idx), meta.hidden_dim), dtype=np.float32)
            if "return_to_go" in data:
                rtg = np.asarray(data["return_to_go"], dtype=np.float32)[local_idx]
            elif "reward" in data and "done" in data:
                rtg_full = compute_return_to_go(data["reward"], data["done"], gamma, num_envs)
                rtg = rtg_full[local_idx]
            else:
                raise KeyError(f"File missing return_to_go and reward/done: {path}")
        m = len(local_idx)
        obs_out[out_cursor : out_cursor + m] = obs
        hidden_out[out_cursor : out_cursor + m] = hidden
        rtg_out[out_cursor : out_cursor + m] = rtg
        out_cursor += m

    if out_cursor != k:
        raise RuntimeError(f"Dataset sampling mismatch: expected {k}, got {out_cursor}")
    return {"obs": obs_out, "hidden_state": hidden_out, "return_to_go": rtg_out}


class TorchValueModel:
    def __init__(
        self,
        checkpoint_path: Path,
        obs_dim: int,
        action_dim: int,
        layer_width: int,
        hidden_dim: int,
        device: str,
        normalize_hidden: bool,
    ):
        import torch
        from models.actor_critic_aug import ActorCriticAug

        self.torch = torch
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model = ActorCriticAug(
            obs_dim=obs_dim,
            action_dim=action_dim,
            layer_width=layer_width,
            hidden_state_dim=hidden_dim,
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        stats_path = checkpoint_path.parent / "hidden_state_stats.npz"
        self.hidden_mean = None
        self.hidden_std = None
        self.hidden_stats_path = None
        if normalize_hidden and stats_path.exists():
            with np.load(stats_path) as stats:
                self.hidden_mean = np.asarray(stats["mean"], dtype=np.float32)
                self.hidden_std = np.asarray(stats["std"], dtype=np.float32)
            self.hidden_std = np.where(self.hidden_std < 1e-6, 1.0, self.hidden_std)
            self.hidden_stats_path = str(stats_path)

    def preprocess_hidden(self, hidden: np.ndarray, hidden_mode: str) -> np.ndarray:
        if hidden_mode == "zero":
            proc = np.zeros_like(hidden, dtype=np.float32)
        else:
            proc = hidden.astype(np.float32, copy=False)
        if self.hidden_mean is not None and self.hidden_std is not None:
            if proc.shape[1] != self.hidden_mean.shape[0]:
                raise ValueError(
                    f"Hidden dim mismatch: data={proc.shape[1]} vs stats={self.hidden_mean.shape[0]}"
                )
            proc = (proc - self.hidden_mean) / self.hidden_std
        return proc

    def predict_values(
        self, obs: np.ndarray, hidden: np.ndarray, hidden_mode: str, batch_size: int
    ) -> np.ndarray:
        hidden_proc = self.preprocess_hidden(hidden, hidden_mode)
        values = []
        with self.torch.no_grad():
            for i in range(0, obs.shape[0], batch_size):
                j = min(i + batch_size, obs.shape[0])
                obs_t = self.torch.from_numpy(obs[i:j]).to(self.device)
                hid_t = self.torch.from_numpy(hidden_proc[i:j]).to(self.device)
                _, v = self.model(obs_t, hid_t)
                values.append(v.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(values, axis=0)


class JAXValueModel:
    def __init__(
        self,
        checkpoint_path: Path,
        obs_dim: int,
        action_dim: int,
        layer_width: int,
        hidden_dim: int,
    ):
        import jax
        import jax.numpy as jnp
        from flax import serialization
        import flax.linen as nn
        from flax.linen.initializers import constant, orthogonal

        class ActorCriticValueOnly(nn.Module):
            action_dim: int
            layer_width: int = 512
            activation: str = "tanh"

            @nn.compact
            def __call__(self, x):
                act = nn.relu if self.activation == "relu" else nn.tanh
                actor_mean = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(x)
                actor_mean = act(actor_mean)
                actor_mean = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(actor_mean)
                actor_mean = act(actor_mean)
                actor_mean = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(actor_mean)
                actor_mean = act(actor_mean)
                actor_logits = nn.Dense(
                    self.action_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                )(actor_mean)

                critic = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(x)
                critic = act(critic)
                critic = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(critic)
                critic = act(critic)
                critic = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(critic)
                critic = act(critic)
                critic = nn.Dense(
                    1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
                )(critic)
                return actor_logits, jnp.squeeze(critic, axis=-1)

        class ActorCriticAugValueOnly(nn.Module):
            action_dim: int
            layer_width: int = 512
            hidden_state_dim: int = 2560
            activation: str = "tanh"

            @nn.compact
            def __call__(self, obs, hidden_state):
                act = nn.relu if self.activation == "relu" else nn.tanh
                embedding = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(obs)
                embedding = act(embedding)
                embedding = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(embedding)
                embedding = act(embedding)
                embedding = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(embedding)
                embedding = act(embedding)
                combined = jnp.concatenate([embedding, hidden_state], axis=-1)

                actor_mean = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(combined)
                actor_mean = act(actor_mean)
                actor_logits = nn.Dense(
                    self.action_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                )(actor_mean)

                critic = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(combined)
                critic = act(critic)
                critic = nn.Dense(
                    self.layer_width,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(critic)
                critic = act(critic)
                critic = nn.Dense(
                    1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
                )(critic)
                return actor_logits, jnp.squeeze(critic, axis=-1)

        self.jax = jax
        self.jnp = jnp
        self.hidden_dim = hidden_dim
        meta_path = checkpoint_path.with_suffix(".json")
        self.meta = None
        preferred_mode = None
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text())
                argv = self.meta.get("metadata", {}).get("argv", {})
                if "no_llm" in argv:
                    preferred_mode = "base" if bool(argv["no_llm"]) else "aug"
            except Exception:
                self.meta = None

        ckpt_bytes = checkpoint_path.read_bytes()

        def _build_aug():
            net = ActorCriticAugValueOnly(
                action_dim=action_dim,
                layer_width=layer_width,
                hidden_state_dim=hidden_dim,
            )
            template = net.init(
                jax.random.PRNGKey(0),
                jnp.zeros((1, obs_dim), dtype=jnp.float32),
                jnp.zeros((1, hidden_dim), dtype=jnp.float32),
            )
            params = serialization.from_bytes(template, ckpt_bytes)
            return net, params, True, "aug"

        def _build_base():
            net = ActorCriticValueOnly(action_dim=action_dim, layer_width=layer_width)
            template = net.init(
                jax.random.PRNGKey(0),
                jnp.zeros((1, obs_dim), dtype=jnp.float32),
            )
            params = serialization.from_bytes(template, ckpt_bytes)
            return net, params, False, "base"

        builders = [_build_aug, _build_base]
        if preferred_mode == "base":
            builders = [_build_base, _build_aug]

        load_error = None
        for build in builders:
            try:
                self.network, self.params, self.use_aug, self.load_variant = build()
                break
            except Exception as exc:
                load_error = exc
        else:
            raise RuntimeError(
                f"Failed to deserialize JAX checkpoint with either architecture: {checkpoint_path}"
            ) from load_error

        if self.use_aug:
            @jax.jit
            def _predict(params, obs_batch, hidden_batch):
                _, values = self.network.apply(params, obs_batch, hidden_batch)
                return values
            self._predict = _predict
        else:
            @jax.jit
            def _predict(params, obs_batch):
                _, values = self.network.apply(params, obs_batch)
                return values
            self._predict = _predict

    def predict_values(
        self, obs: np.ndarray, hidden: np.ndarray, hidden_mode: str, batch_size: int
    ) -> np.ndarray:
        if self.use_aug:
            if hidden_mode == "zero":
                hidden_proc = np.zeros_like(hidden, dtype=np.float32)
            else:
                hidden_proc = hidden.astype(np.float32, copy=False)
        else:
            hidden_proc = None
        values = []
        for i in range(0, obs.shape[0], batch_size):
            j = min(i + batch_size, obs.shape[0])
            obs_j = self.jnp.asarray(obs[i:j], dtype=self.jnp.float32)
            if self.use_aug:
                hid_j = self.jnp.asarray(hidden_proc[i:j], dtype=self.jnp.float32)
                v = self._predict(self.params, obs_j, hid_j)
            else:
                v = self._predict(self.params, obs_j)
            values.append(np.asarray(self.jax.device_get(v), dtype=np.float32))
        return np.concatenate(values, axis=0)


def checkpoint_kind(path: Path) -> str:
    if path.suffix == ".pth":
        return "torch"
    if path.suffix == ".msgpack":
        return "jax"
    raise ValueError(f"Unsupported checkpoint format: {path}")


def compute_value_metrics(values: np.ndarray, rtg: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rtg = np.asarray(rtg, dtype=np.float64).reshape(-1)
    if values.shape != rtg.shape:
        raise ValueError(f"Shape mismatch values={values.shape} rtg={rtg.shape}")
    err = values - rtg
    top_k = max(1, int(0.1 * len(values)))
    top_idx = np.argsort(values)[-top_k:]
    bot_idx = np.argsort(values)[:top_k]
    return {
        "num_samples": int(len(values)),
        "rtg_mean": float(rtg.mean()),
        "rtg_std": float(rtg.std()),
        "value_mean": float(values.mean()),
        "value_std": float(values.std()),
        "pearson_r": safe_corr(values, rtg),
        "spearman_r": safe_corr(rankdata_average_ties(values), rankdata_average_ties(rtg)),
        "explained_variance": explained_variance(rtg, values),
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "bias_mean_pred_minus_rtg": float(np.mean(err)),
        "top10pct_rtg_mean": float(rtg[top_idx].mean()),
        "bottom10pct_rtg_mean": float(rtg[bot_idx].mean()),
        "top_minus_bottom_rtg": float(rtg[top_idx].mean() - rtg[bot_idx].mean()),
        "calibration_by_pred_bin": calibration_bins(values, rtg, num_bins=10),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze value-function learning quality on RTG-labelled dataset samples."
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
    sample = load_dataset_sample(
        meta=meta,
        num_samples=args.num_samples,
        seed=args.seed,
        gamma=args.gamma,
        num_envs=args.num_envs,
    )
    obs = sample["obs"]
    hidden = sample["hidden_state"]
    rtg = sample["return_to_go"]

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
                values = model.predict_values(
                    obs=obs,
                    hidden=hidden,
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
                values = model.predict_values(
                    obs=obs,
                    hidden=hidden,
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
            metrics = compute_value_metrics(values=values, rtg=rtg)
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
            "gamma": args.gamma,
            "num_envs": args.num_envs,
        },
        "evaluation": {
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
