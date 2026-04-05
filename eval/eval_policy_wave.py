#!/usr/bin/env python3
"""
Unified policy-wave evaluator for Craftax symbolic policies.

Tracks:
- id: in-distribution gameplay eval
- gameplay_llm: gameplay eval with aligned hidden-refresh generation capture
- ood: gameplay eval under observation-level OOD transforms
- value: value battery orchestration (ranking, counterfactual pairs, TD consistency)
- bundle: qualitative reaction pass over recorder bundle observations
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import requests
import torch
import yaml
from flax import serialization
from flax.training.train_state import TrainState

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from craftax.craftax.constants import Action
from craftax.craftax_env import make_craftax_env_from_name
from labelling.obs_to_text import (
    MAP_OBS_SIZE,
    MAP_CHANNELS,
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_MOB_TYPES,
    OBS_DIM,
    obs_to_text,
)
from models.actor_critic import ActorCritic, ActorCriticAug
from models.actor_critic_aug import ActorCriticAug as TorchActorCriticAug
from llm.extractor import VLLMHiddenStateExtractor
from llm.prompts import filter_text_obs
from envs.wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper


MOB_CHANNEL_START = NUM_BLOCK_TYPES + NUM_ITEM_TYPES
MOB_CHANNELS = 5 * NUM_MOB_TYPES
LIGHT_CHANNEL = MAP_CHANNELS - 1
SPECIAL_START = 43
SPECIAL_FLOOR_INDEX = SPECIAL_START + 5


@dataclass
class ResolvedPolicy:
    policy_id: str
    policy_name: str
    variant_name: str
    policy_type: str
    checkpoint_path: str
    stats_path: Optional[str]
    metadata_path: Optional[str]
    hidden_mode: str
    skip_n: int
    hidden_dim: int
    layer_width: int
    actor_head_layers: int
    critic_head_layers: int
    run_dir: Optional[str] = None
    train_step: Optional[int] = None


class SharedLLMManager:
    def __init__(self, model_id: str, target_layer: int, tokens_to_generate: int):
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
        resp = requests.get(f"{vllm_url}/health", timeout=2)
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM /health returned {resp.status_code}")

        extracted_layers = [8, 16, 24, 35]
        layer_index = -1
        if target_layer != -1 and target_layer in extracted_layers:
            layer_index = extracted_layers.index(target_layer)

        self.tokens_to_generate = int(tokens_to_generate)
        self.extractor = VLLMHiddenStateExtractor(
            server_url=vllm_url,
            model_name="./configs/vllm_hidden_qwen4b",
            model_id=model_id,
            target_layer=layer_index,
        )
        self.hidden_size = int(self.extractor.hidden_size)

    def extract_hidden(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        prompts = [filter_text_obs(obs_to_text(obs)) for obs in obs_batch]
        t0 = time.perf_counter()
        if self.tokens_to_generate == 1:
            hidden, metrics = self.extractor.extract_hidden_states_no_cot(prompts)
            texts = [""] * len(prompts)
        else:
            hidden, texts, metrics = self.extractor.extract_hidden_states(
                prompts,
                batch_size=min(32, len(prompts)),
                max_new_tokens=self.tokens_to_generate,
            )
        dt = time.perf_counter() - t0
        out = dict(metrics)
        out["timing/llm_call_ms"] = dt * 1000.0
        out["generated_text"] = texts
        return hidden.astype(np.float32, copy=False), out

    def generate_for_obs(self, obs_batch: np.ndarray, max_new_tokens: int) -> List[str]:
        prompts = [filter_text_obs(obs_to_text(obs)) for obs in obs_batch]
        _, texts, _ = self.extractor.extract_hidden_states(
            prompts,
            batch_size=min(32, len(prompts)),
            max_new_tokens=max_new_tokens,
        )
        return [" ".join(str(t).replace("\n", " ").split()) for t in texts]


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def flatten(prefix: str, data: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in data.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(flatten(key, v))
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[key] = float(v)
    return out


def parse_seed_list(text: str) -> List[int]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    if not vals:
        return [42]
    return [int(v) for v in vals]


def parse_tracks(text: str) -> List[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return vals or ["id"]


def parse_policy_ids(text: str) -> List[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return vals


def parse_step_from_path(path: str) -> int:
    name = Path(path).name
    for pat in (r"(?:checkpoint_|step_)(\d+)", r"(\d{5,})"):
        m = re.search(pat, name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return -1


def resolve_latest_path(explicit_path: Optional[str], path_glob: Optional[str], required: bool = True) -> Optional[str]:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.exists():
            return str(p.resolve())
        if required:
            raise FileNotFoundError(f"Path not found: {explicit_path}")
        return None

    if path_glob:
        matches = [Path(p).resolve() for p in glob.glob(path_glob)]
        if matches:
            matches.sort(key=lambda p: (parse_step_from_path(str(p)), p.stat().st_mtime, str(p)))
            return str(matches[-1])

    if required:
        raise FileNotFoundError(f"Could not resolve path: explicit={explicit_path} glob={path_glob}")
    return None


def resolve_slice_paths(
    final_path: str,
    slice_glob: Optional[str],
    num_mids: int,
) -> List[str]:
    mids: List[str] = []
    if slice_glob:
        mids = [str(Path(p).resolve()) for p in glob.glob(slice_glob)]
    else:
        parent = Path(final_path).parent
        stem = Path(final_path).suffix
        if stem == ".pth":
            mids = [str(p.resolve()) for p in parent.glob("*_checkpoint_*.pth")]
        elif stem == ".msgpack":
            mids = [str(p.resolve()) for p in parent.glob("*step_*.msgpack")]

    if not mids:
        return [final_path]

    mids = sorted(set(mids), key=lambda p: (parse_step_from_path(p), Path(p).stat().st_mtime, p))
    mids = [p for p in mids if Path(p).resolve() != Path(final_path).resolve()]
    if not mids:
        return [final_path]

    if num_mids > 0 and len(mids) > num_mids:
        idx = np.linspace(0, len(mids) - 1, num_mids, dtype=np.int32)
        mids = [mids[int(i)] for i in sorted(set(idx.tolist()))]

    return mids + [final_path]


def infer_offline_fusion_mode(state_dict: Dict[str, torch.Tensor], hidden_dim: int) -> str:
    if "actor_obs_fc1.weight" in state_dict and "actor_hidden_fc1.weight" in state_dict:
        return "dual_concat"

    actor_in = int(state_dict["actor_fc1.weight"].shape[1])
    layer_width = int(state_dict["encoder_fc1.weight"].shape[0])
    if actor_in == layer_width + hidden_dim:
        return "concat_raw"
    if actor_in == 2 * layer_width:
        return "gated_proj"
    if actor_in == layer_width:
        return "residual_gated"
    if (
        "hidden_gate_logit" in state_dict
        or "hidden_proj.weight" in state_dict
        or "hidden_ln.weight" in state_dict
    ):
        return "gated_proj"
    return "concat_raw"


def load_torch_offline_policy(spec: ResolvedPolicy, device: torch.device):
    state_dict = torch.load(spec.checkpoint_path, map_location=device)

    if "encoder_fc1.weight" in state_dict:
        obs_dim = int(state_dict["encoder_fc1.weight"].shape[1])
        layer_width = int(state_dict["encoder_fc1.weight"].shape[0])
    elif "actor_obs_fc1.weight" in state_dict:
        obs_dim = int(state_dict["actor_obs_fc1.weight"].shape[1])
        layer_width = int(state_dict["actor_obs_fc1.weight"].shape[0])
    else:
        raise KeyError("Unable to infer obs/layer dims from checkpoint state_dict")

    if "actor_out.weight" in state_dict:
        action_dim = int(state_dict["actor_out.weight"].shape[0])
    elif "actor_fc2.weight" in state_dict:
        action_dim = int(state_dict["actor_fc2.weight"].shape[0])
    else:
        raise KeyError("Unable to infer action_dim from checkpoint state_dict")

    if "hidden_proj.weight" in state_dict:
        hidden_dim = int(state_dict["hidden_proj.weight"].shape[1])
    elif "actor_hidden_fc1.weight" in state_dict:
        hidden_dim = int(state_dict["actor_hidden_fc1.weight"].shape[1])
    else:
        hidden_dim = int(spec.hidden_dim)

    stats_path = Path(spec.stats_path).expanduser() if spec.stats_path else None
    if stats_path and stats_path.exists():
        stats = np.load(stats_path)
        hidden_mean = np.asarray(stats["mean"], dtype=np.float32)
        hidden_std = np.asarray(stats["std"], dtype=np.float32)
    else:
        hidden_mean = np.zeros((hidden_dim,), dtype=np.float32)
        hidden_std = np.ones((hidden_dim,), dtype=np.float32)
    hidden_std = np.where(hidden_std < 1e-6, 1.0, hidden_std)

    model = TorchActorCriticAug(
        obs_dim=obs_dim,
        action_dim=action_dim,
        layer_width=layer_width,
        hidden_state_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    def act_fn(obs_np: np.ndarray, hidden_np: np.ndarray, deterministic: bool, rng: jax.Array):
        del rng
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
            hid_t = torch.from_numpy(hidden_np).to(device=device, dtype=torch.float32)
            pi, _ = model(obs_t, hid_t)
            if deterministic:
                actions = torch.argmax(pi.logits, dim=-1)
            else:
                actions = pi.sample()
        return actions.detach().cpu().numpy().astype(np.int32)

    def value_fn(obs_np: np.ndarray, hidden_np: np.ndarray):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
            hid_t = torch.from_numpy(hidden_np).to(device=device, dtype=torch.float32)
            _, value = model(obs_t, hid_t)
        return value.detach().cpu().numpy().astype(np.float32)

    return {
        "framework": "torch",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "hidden_mean": hidden_mean,
        "hidden_std": hidden_std,
        "uses_hidden": True,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "metadata": {
            "fusion_mode": "fixed_dual_branch",
            "checkpoint": spec.checkpoint_path,
            "stats_path": str(stats_path) if stats_path else None,
        },
    }


def load_jax_aug_policy(spec: ResolvedPolicy):
    ckpt = Path(spec.checkpoint_path)
    blob = ckpt.read_bytes()

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)

    hidden_dim = int(spec.hidden_dim)
    layer_width = int(spec.layer_width)

    net = ActorCriticAug(
        action_dim=action_dim,
        layer_width=layer_width,
        hidden_state_dim=hidden_dim,
    )
    template = net.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1, hidden_dim), dtype=jnp.float32),
    )
    params = serialization.from_bytes(template, blob)

    @jax.jit
    def _act(params, obs, hidden, rng, deterministic=False):
        pi, _ = net.apply(params, obs, hidden)
        return jax.lax.cond(
            jnp.asarray(deterministic, dtype=jnp.bool_),
            lambda _: jnp.argmax(pi.logits, axis=-1),
            lambda _: pi.sample(seed=rng),
            operand=None,
        )

    @jax.jit
    def _value(params, obs, hidden):
        _, value = net.apply(params, obs, hidden)
        return value

    def act_fn(obs_np: np.ndarray, hidden_np: np.ndarray, deterministic: bool, rng: jax.Array):
        action = _act(
            params,
            jnp.asarray(obs_np, dtype=jnp.float32),
            jnp.asarray(hidden_np, dtype=jnp.float32),
            rng,
            deterministic,
        )
        return np.asarray(jax.device_get(action), dtype=np.int32)

    def value_fn(obs_np: np.ndarray, hidden_np: np.ndarray):
        value = _value(
            params,
            jnp.asarray(obs_np, dtype=jnp.float32),
            jnp.asarray(hidden_np, dtype=jnp.float32),
        )
        return np.asarray(jax.device_get(value), dtype=np.float32)

    return {
        "framework": "jax",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "hidden_mean": np.zeros((hidden_dim,), dtype=np.float32),
        "hidden_std": np.ones((hidden_dim,), dtype=np.float32),
        "uses_hidden": True,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "metadata": {
            "fusion_mode": "fixed_dual_branch",
            "checkpoint": spec.checkpoint_path,
        },
    }


def load_ppo_msgpack_policy(spec: ResolvedPolicy):
    ckpt = Path(spec.checkpoint_path)
    blob = ckpt.read_bytes()

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)

    layer_width = int(spec.layer_width)
    net = ActorCritic(action_dim=action_dim, layer_width=layer_width)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim), dtype=jnp.float32))
    params = serialization.from_bytes(template, blob)

    @jax.jit
    def _act(params, obs, rng, deterministic=False):
        pi, _ = net.apply(params, obs)
        return jax.lax.cond(
            jnp.asarray(deterministic, dtype=jnp.bool_),
            lambda _: jnp.argmax(pi.logits, axis=-1),
            lambda _: pi.sample(seed=rng),
            operand=None,
        )

    @jax.jit
    def _value(params, obs):
        _, value = net.apply(params, obs)
        return value

    def act_fn(obs_np: np.ndarray, hidden_np: np.ndarray, deterministic: bool, rng: jax.Array):
        del hidden_np
        action = _act(params, jnp.asarray(obs_np, dtype=jnp.float32), rng, deterministic)
        return np.asarray(jax.device_get(action), dtype=np.int32)

    def value_fn(obs_np: np.ndarray, hidden_np: np.ndarray):
        del hidden_np
        value = _value(params, jnp.asarray(obs_np, dtype=jnp.float32))
        return np.asarray(jax.device_get(value), dtype=np.float32)

    return {
        "framework": "jax",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": 0,
        "hidden_mean": None,
        "hidden_std": None,
        "uses_hidden": False,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "metadata": {
            "checkpoint": spec.checkpoint_path,
        },
    }


def load_ppo_orbax_policy(spec: ResolvedPolicy):
    if spec.run_dir is None or spec.train_step is None:
        raise ValueError("ppo_orbax requires run_dir and train_step")

    run_path = Path(spec.run_dir).expanduser().resolve()
    cfg_path = run_path / "config.yaml"
    layer_size = int(spec.layer_width)
    if cfg_path.exists():
        try:
            cfg_raw = yaml.safe_load(cfg_path.read_text())
            cfg = {
                k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
                for k, v in cfg_raw.items()
            }
            layer_size = int(cfg.get("LAYER_SIZE", layer_size))
        except Exception:
            pass

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)

    net = ActorCritic(action_dim=action_dim, layer_width=layer_size)
    params0 = net.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim), dtype=jnp.float32))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(2e-4, eps=1e-5))
    train_state = TrainState.create(apply_fn=net.apply, params=params0, tx=tx)

    ckpt_mgr = ocp.CheckpointManager(
        str(run_path / "policies"),
        ocp.PyTreeCheckpointer(),
        ocp.CheckpointManagerOptions(max_to_keep=1, create=False),
    )
    train_state = ckpt_mgr.restore(int(spec.train_step), items=train_state)

    @jax.jit
    def _act(params, obs, rng, deterministic=False):
        pi, _ = net.apply(params, obs)
        return jax.lax.cond(
            jnp.asarray(deterministic, dtype=jnp.bool_),
            lambda _: jnp.argmax(pi.logits, axis=-1),
            lambda _: pi.sample(seed=rng),
            operand=None,
        )

    @jax.jit
    def _value(params, obs):
        _, value = net.apply(params, obs)
        return value

    def act_fn(obs_np: np.ndarray, hidden_np: np.ndarray, deterministic: bool, rng: jax.Array):
        del hidden_np
        action = _act(train_state.params, jnp.asarray(obs_np, dtype=jnp.float32), rng, deterministic)
        return np.asarray(jax.device_get(action), dtype=np.int32)

    def value_fn(obs_np: np.ndarray, hidden_np: np.ndarray):
        del hidden_np
        value = _value(train_state.params, jnp.asarray(obs_np, dtype=jnp.float32))
        return np.asarray(jax.device_get(value), dtype=np.float32)

    return {
        "framework": "jax",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": 0,
        "hidden_mean": None,
        "hidden_std": None,
        "uses_hidden": False,
        "act_fn": act_fn,
        "value_fn": value_fn,
        "metadata": {
            "run_dir": str(run_path),
            "train_step": int(spec.train_step),
        },
    }


def load_policy(spec: ResolvedPolicy, device: torch.device):
    if spec.policy_type == "torch_offline_aug":
        return load_torch_offline_policy(spec, device)
    if spec.policy_type == "jax_aug_msgpack":
        return load_jax_aug_policy(spec)
    if spec.policy_type == "ppo_msgpack":
        return load_ppo_msgpack_policy(spec)
    if spec.policy_type == "ppo_orbax":
        return load_ppo_orbax_policy(spec)
    raise ValueError(f"Unsupported policy_type: {spec.policy_type}")


def extract_done_episode_achievements(info: Dict, done_mask: np.ndarray) -> Tuple[List[int], Dict[str, float]]:
    ach_keys = sorted(k for k in info.keys() if k.startswith("Achievements/"))
    if not ach_keys:
        return [], {}

    ach_cols = [np.asarray(jax.device_get(info[k]), dtype=np.float32) for k in ach_keys]
    ach_mat = np.stack(ach_cols, axis=-1)

    done_idx = np.nonzero(done_mask)[0]
    counts: List[int] = []
    unlock_sum = np.zeros((ach_mat.shape[1],), dtype=np.float64)
    for i in done_idx:
        unlocked = ach_mat[i] > 0.0
        counts.append(int(unlocked.sum()))
        unlock_sum += unlocked.astype(np.float64)

    rates = {}
    denom = max(1, len(done_idx))
    for j, k in enumerate(ach_keys):
        rates[k] = float(unlock_sum[j] / denom)
    return counts, rates


def _set_floor_obs(obs_batch: np.ndarray, floor_value: int) -> np.ndarray:
    obs = np.asarray(obs_batch, dtype=np.float32).copy()
    obs[:, MAP_OBS_SIZE + SPECIAL_FLOOR_INDEX] = float(floor_value) / 10.0
    return obs


def _inject_mob_obs(
    obs_batch: np.ndarray,
    mob_class: int,
    mob_type: int,
    row_offset: int,
    col_offset: int,
    floor_value: Optional[int],
) -> np.ndarray:
    obs = np.asarray(obs_batch, dtype=np.float32).copy()
    if floor_value is not None:
        obs[:, MAP_OBS_SIZE + SPECIAL_FLOOR_INDEX] = float(floor_value) / 10.0

    mob_idx = mob_class * NUM_MOB_TYPES + mob_type
    row = int(np.clip(OBS_DIM[0] // 2 + row_offset, 0, OBS_DIM[0] - 1))
    col = int(np.clip(OBS_DIM[1] // 2 + col_offset, 0, OBS_DIM[1] - 1))
    for i in range(obs.shape[0]):
        map_view = obs[i, :MAP_OBS_SIZE].reshape(OBS_DIM[0], OBS_DIM[1], MAP_CHANNELS)
        map_view[row, col, MOB_CHANNEL_START : MOB_CHANNEL_START + MOB_CHANNELS] = 0.0
        map_view[row, col, MOB_CHANNEL_START + mob_idx] = 1.0
        map_view[row, col, LIGHT_CHANNEL] = 1.0
    return obs


def _darkness_obs(obs_batch: np.ndarray, visible: float = 0.0) -> np.ndarray:
    obs = np.asarray(obs_batch, dtype=np.float32).copy()
    for i in range(obs.shape[0]):
        map_view = obs[i, :MAP_OBS_SIZE].reshape(OBS_DIM[0], OBS_DIM[1], MAP_CHANNELS)
        map_view[:, :, LIGHT_CHANNEL] = float(visible)
    return obs


def make_ood_transform(scenario: Dict) -> Callable[[np.ndarray], np.ndarray]:
    stype = str(scenario.get("type", "identity"))
    if stype == "identity":
        return lambda obs: np.asarray(obs, dtype=np.float32)
    if stype == "gaussian_noise":
        sigma = float(scenario.get("sigma", 0.05))

        def _fn(obs):
            arr = np.asarray(obs, dtype=np.float32)
            noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
            return arr + noise

        return _fn
    if stype == "set_floor":
        floor_value = int(scenario.get("floor", 2))
        return lambda obs: _set_floor_obs(obs, floor_value=floor_value)
    if stype == "inject_mob":
        mob_class = int(scenario.get("mob_class", 0))
        mob_type = int(scenario.get("mob_type", 0))
        row_offset = int(scenario.get("row_offset", 1))
        col_offset = int(scenario.get("col_offset", 0))
        floor_value = scenario.get("floor", None)
        floor_value = int(floor_value) if floor_value is not None else None
        return lambda obs: _inject_mob_obs(
            obs,
            mob_class=mob_class,
            mob_type=mob_type,
            row_offset=row_offset,
            col_offset=col_offset,
            floor_value=floor_value,
        )
    if stype == "darkness":
        visible = float(scenario.get("visible", 0.0))
        return lambda obs: _darkness_obs(obs, visible=visible)

    raise ValueError(f"Unsupported OOD scenario type: {stype}")


def evaluate_gameplay(
    spec: ResolvedPolicy,
    policy: Dict,
    seed: int,
    num_envs: int,
    target_episodes: int,
    max_env_steps: int,
    deterministic: bool,
    llm_manager: Optional[SharedLLMManager],
    generation_tokens: int,
    ood_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, object]:
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    rng = jax.random.PRNGKey(seed)
    rng, rr = jax.random.split(rng)
    obs, env_state = env.reset(rr, env_params)

    uses_hidden = bool(policy.get("uses_hidden", False))
    hidden_dim = int(policy.get("hidden_dim", 0))
    hidden_mean = policy.get("hidden_mean")
    hidden_std = policy.get("hidden_std")

    if uses_hidden:
        hidden_raw = np.zeros((num_envs, hidden_dim), dtype=np.float32)
        if spec.hidden_mode == "llm" and llm_manager is not None:
            if int(llm_manager.hidden_size) != hidden_dim:
                raise ValueError(
                    f"Hidden dim mismatch for {spec.policy_name}:{spec.variant_name}: "
                    f"policy expects {hidden_dim}, llm extractor gives {llm_manager.hidden_size}"
                )
    else:
        hidden_raw = None

    steps_since_refresh = int(spec.skip_n)
    llm_calls = 0
    llm_call_ms: List[float] = []
    generated_text_samples: List[Dict[str, object]] = []

    returns: List[float] = []
    lengths: List[float] = []
    ach_counts: List[int] = []
    ach_totals: Dict[str, float] = {}

    episode_counter = 0
    steps = 0
    t0 = time.time()

    while episode_counter < target_episodes and steps < max_env_steps:
        obs_np = np.asarray(jax.device_get(obs), dtype=np.float32)
        policy_obs = ood_transform(obs_np) if ood_transform is not None else obs_np

        if uses_hidden:
            if spec.hidden_mode == "llm":
                if llm_manager is None:
                    raise RuntimeError("Policy requires hidden_mode=llm but llm_manager is unavailable")
                if steps_since_refresh >= max(1, int(spec.skip_n)):
                    hidden_raw, metrics = llm_manager.extract_hidden(policy_obs)
                    llm_calls += 1
                    steps_since_refresh = 0
                    llm_call_ms.append(float(metrics.get("timing/llm_call_ms", 0.0)))
                    if generation_tokens > 0:
                        texts = llm_manager.generate_for_obs(policy_obs, max_new_tokens=generation_tokens)
                        if texts:
                            generated_text_samples.append(
                                {
                                    "step": int(steps),
                                    "env0_text": texts[0],
                                    "num_texts": len(texts),
                                }
                            )
            elif spec.hidden_mode == "zero":
                hidden_raw = np.zeros_like(hidden_raw)
            elif spec.hidden_mode == "random":
                hidden_raw = np.random.standard_normal(size=hidden_raw.shape).astype(np.float32)
            else:
                raise ValueError(f"Unsupported hidden_mode for hidden policy: {spec.hidden_mode}")

            if hidden_mean is not None and hidden_std is not None:
                hidden_in = (hidden_raw - hidden_mean[None, :]) / hidden_std[None, :]
            else:
                hidden_in = hidden_raw
        else:
            hidden_in = np.zeros((num_envs, 1), dtype=np.float32)

        rng, ar = jax.random.split(rng)
        actions = policy["act_fn"](policy_obs, hidden_in, deterministic, ar)
        if uses_hidden:
            steps_since_refresh += 1

        rng, sr = jax.random.split(rng)
        obs, env_state, _, _, info = env.step(sr, env_state, jnp.asarray(actions), env_params)
        steps += 1

        done_mask = np.asarray(jax.device_get(info["returned_episode"]), dtype=bool)
        if done_mask.any():
            ret_np = np.asarray(jax.device_get(info["returned_episode_returns"]), dtype=np.float32)
            len_np = np.asarray(jax.device_get(info["returned_episode_lengths"]), dtype=np.float32)
            done_idx = np.nonzero(done_mask)[0]
            ep_ach_counts, ep_ach_rates = extract_done_episode_achievements(info, done_mask)

            for k, v in ep_ach_rates.items():
                ach_totals[k] = ach_totals.get(k, 0.0) + v * len(done_idx)

            for local_pos, idx in enumerate(done_idx):
                returns.append(float(ret_np[idx]))
                lengths.append(float(len_np[idx]))
                ach_counts.append(int(ep_ach_counts[local_pos]) if local_pos < len(ep_ach_counts) else 0)
                episode_counter += 1
                if episode_counter >= target_episodes:
                    break

    runtime_sec = time.time() - t0
    ach_rates = {}
    denom = max(1, episode_counter)
    for k, accum in ach_totals.items():
        ach_rates[k.replace("Achievements/", "").lower()] = float(accum / denom)

    return {
        "policy": spec.policy_name,
        "variant": spec.variant_name,
        "seed": int(seed),
        "episodes": int(episode_counter),
        "env_steps": int(steps),
        "runtime_sec": float(runtime_sec),
        "returns": summarize(returns),
        "lengths": summarize(lengths),
        "achievement_counts": summarize(ach_counts),
        "achievement_unlock_rates": ach_rates,
        "llm_calls": int(llm_calls),
        "llm_call_ms_mean": float(np.mean(llm_call_ms)) if llm_call_ms else 0.0,
        "llm_generation_samples": generated_text_samples[:32],
    }


def aggregate_seed_metrics(seed_runs: List[Dict[str, object]]) -> Dict[str, object]:
    returns = [r.get("returns", {}).get("mean", 0.0) for r in seed_runs]
    lengths = [r.get("lengths", {}).get("mean", 0.0) for r in seed_runs]
    ach = [r.get("achievement_counts", {}).get("mean", 0.0) for r in seed_runs]
    llm_calls = [float(r.get("llm_calls", 0.0)) for r in seed_runs]
    return {
        "num_seeds": len(seed_runs),
        "return_mean_over_seeds": float(np.mean(returns)) if returns else 0.0,
        "return_std_over_seeds": float(np.std(returns)) if returns else 0.0,
        "length_mean_over_seeds": float(np.mean(lengths)) if lengths else 0.0,
        "achievement_mean_over_seeds": float(np.mean(ach)) if ach else 0.0,
        "llm_calls_mean_over_seeds": float(np.mean(llm_calls)) if llm_calls else 0.0,
    }


def evaluate_bundle_reactions(
    spec: ResolvedPolicy,
    policy: Dict,
    bundle_dirs: List[str],
    llm_manager: Optional[SharedLLMManager],
    generation_tokens: int,
) -> Dict[str, object]:
    step_dirs: List[Path] = []
    for b in bundle_dirs:
        step_dirs.extend(sorted(Path(p).resolve() for p in glob.glob(str(Path(b) / "step_*"))))
    step_dirs = sorted(set(step_dirs))

    if not step_dirs:
        return {
            "policy": spec.policy_name,
            "variant": spec.variant_name,
            "num_bundles": 0,
            "records": [],
        }

    records = []
    action_hist: Dict[str, int] = {}

    rng = jax.random.PRNGKey(0)
    for idx, step_dir in enumerate(step_dirs):
        obs_path = step_dir / "obs_before.npy"
        if not obs_path.exists():
            continue
        obs = np.load(obs_path).astype(np.float32).reshape(1, -1)

        if policy.get("uses_hidden", False):
            hidden_dim = int(policy["hidden_dim"])
            if spec.hidden_mode == "llm":
                if llm_manager is None:
                    continue
                if int(llm_manager.hidden_size) != hidden_dim:
                    continue
                hidden_raw, _ = llm_manager.extract_hidden(obs)
                if generation_tokens > 0:
                    llm_text = llm_manager.generate_for_obs(obs, generation_tokens)[0]
                else:
                    llm_text = ""
            elif spec.hidden_mode == "zero":
                hidden_raw = np.zeros((1, hidden_dim), dtype=np.float32)
                llm_text = ""
            else:
                hidden_raw = np.random.standard_normal(size=(1, hidden_dim)).astype(np.float32)
                llm_text = ""

            hm = policy.get("hidden_mean")
            hs = policy.get("hidden_std")
            hidden_in = (hidden_raw - hm[None, :]) / hs[None, :] if hm is not None and hs is not None else hidden_raw
        else:
            hidden_in = np.zeros((1, 1), dtype=np.float32)
            llm_text = ""

        rng, ar = jax.random.split(rng)
        act = int(policy["act_fn"](obs, hidden_in, True, ar)[0])
        value = float(policy["value_fn"](obs, hidden_in)[0])

        name = Action(act).name
        action_hist[name] = action_hist.get(name, 0) + 1

        records.append(
            {
                "bundle": str(step_dir),
                "action_id": act,
                "action_name": name,
                "value": value,
                "llm_text": llm_text,
            }
        )

        if idx >= 255:
            break

    return {
        "policy": spec.policy_name,
        "variant": spec.variant_name,
        "num_bundles": len(records),
        "action_hist": action_hist,
        "records": records[:128],
    }


def run_subprocess(cmd: List[str]):
    print("[subprocess]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_value_battery(
    checkpoints: List[str],
    output_dir: Path,
    dataset_dir: str,
    data_glob: str,
    num_samples: int,
    num_envs: int,
    hidden_mode: str,
    pair_types: List[str],
    torch_device: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only scripts that can consume .pth/.msgpack directly.
    usable = [c for c in checkpoints if Path(c).suffix in {".pth", ".msgpack"}]
    if not usable:
        return {"status": "skipped", "reason": "No usable checkpoints for value battery."}

    value_json = output_dir / "value_learning.json"
    pair_json = output_dir / "value_pairs.json"
    td_json = output_dir / "value_td_consistency.json"

    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "offline_rl" / "analyze_value_learning.py"),
            "--checkpoints",
            *usable,
            "--dataset_dir",
            dataset_dir,
            "--data_glob",
            data_glob,
            "--num_samples",
            str(num_samples),
            "--num_envs",
            str(num_envs),
            "--hidden_mode",
            hidden_mode,
            "--torch_device",
            torch_device,
            "--output_json",
            str(value_json),
        ]
    )

    pth_usable = [c for c in usable if Path(c).suffix == ".pth"]
    if pth_usable:
        run_subprocess(
            [
                sys.executable,
                str(REPO_ROOT / "offline_rl" / "analyze_value_pairs.py"),
                "--checkpoints",
                *pth_usable,
                "--pairs",
                *pair_types,
                "--hidden_input_mode",
                "zero",
                "--device",
                torch_device,
                "--output_json",
                str(pair_json),
            ]
        )
    else:
        pair_json.write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "No .pth checkpoints available for analyze_value_pairs.",
                },
                indent=2,
            )
        )

    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "offline_rl" / "analyze_value_td_consistency.py"),
            "--checkpoints",
            *usable,
            "--dataset_dir",
            dataset_dir,
            "--data_glob",
            data_glob,
            "--num_samples",
            str(num_samples),
            "--num_envs",
            str(num_envs),
            "--hidden_mode",
            hidden_mode,
            "--torch_device",
            torch_device,
            "--output_json",
            str(td_json),
        ]
    )

    return {
        "status": "ok",
        "value_learning_json": str(value_json),
        "value_pairs_json": str(pair_json),
        "value_td_json": str(td_json),
        "checkpoints": usable,
    }


def resolve_manifest_policies(manifest: Dict, include_slices: bool, slice_count: int) -> List[ResolvedPolicy]:
    resolved: List[ResolvedPolicy] = []
    defaults = manifest.get("defaults", {})

    for p in manifest.get("policies", []):
        policy_id = str(p["id"])
        policy_name = str(p.get("name", policy_id))
        policy_type = str(p["policy_type"])

        final_ckpt = resolve_latest_path(
            explicit_path=p.get("checkpoint_path"),
            path_glob=p.get("checkpoint_glob"),
            required=(policy_type != "ppo_orbax"),
        )

        run_dir = None
        train_step = None
        if policy_type == "ppo_orbax":
            run_dir = resolve_latest_path(
                explicit_path=p.get("run_dir"),
                path_glob=p.get("run_dir_glob"),
                required=True,
            )
            train_step = int(p.get("train_step", defaults.get("ppo_train_step", 100000000)))
            final_ckpt = run_dir

        stats_path = resolve_latest_path(
            explicit_path=p.get("stats_path"),
            path_glob=p.get("stats_glob"),
            required=False,
        )
        metadata_path = resolve_latest_path(
            explicit_path=p.get("metadata_path"),
            path_glob=p.get("metadata_glob"),
            required=False,
        )

        slice_paths = [final_ckpt] if final_ckpt else []
        if include_slices and final_ckpt and policy_type != "ppo_orbax":
            slice_paths = resolve_slice_paths(
                final_path=final_ckpt,
                slice_glob=p.get("slice_glob"),
                num_mids=max(0, int(slice_count)),
            )

        for ckpt_path in slice_paths:
            variant_step = parse_step_from_path(ckpt_path)
            variant_name = "final" if variant_step < 0 else f"step_{variant_step}"
            resolved.append(
                ResolvedPolicy(
                    policy_id=policy_id,
                    policy_name=policy_name,
                    variant_name=variant_name,
                    policy_type=policy_type,
                    checkpoint_path=ckpt_path,
                    stats_path=stats_path,
                    metadata_path=metadata_path,
                    hidden_mode=str(p.get("hidden_mode", defaults.get("hidden_mode", "none"))),
                    skip_n=int(p.get("skip_n", defaults.get("skip_n", 1))),
                    hidden_dim=int(p.get("hidden_dim", defaults.get("hidden_dim", 2560))),
                    layer_width=int(p.get("layer_width", defaults.get("layer_width", 512))),
                    actor_head_layers=int(p.get("actor_head_layers", defaults.get("actor_head_layers", 1))),
                    critic_head_layers=int(p.get("critic_head_layers", defaults.get("critic_head_layers", 2))),
                    run_dir=run_dir,
                    train_step=train_step,
                )
            )

    return resolved


def maybe_init_wandb(args, manifest: Dict):
    if args.no_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed; rerun with --no_wandb")

    policy_tag = "all"
    policy_ids = parse_policy_ids(args.policy_ids)
    if policy_ids:
        policy_tag = "-".join(policy_ids[:3])
    policy_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", policy_tag)[:64]

    run_name = (
        f"{manifest.get('wave_name', 'policy_wave')}_{policy_tag}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "manifest": str(args.manifest),
            "tracks": args.tracks,
            "seeds": args.seeds,
            "num_envs": args.num_envs,
            "num_episodes": args.num_episodes,
            "max_env_steps": args.max_env_steps,
            "include_slices": args.include_slices,
            "slice_count": args.slice_count,
            "policy_ids": args.policy_ids,
        },
    )
    return run


def main():
    parser = argparse.ArgumentParser(description="Unified Craftax policy-wave evaluator")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--tracks", type=str, default="id,value,ood,gameplay_llm")
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "analysis" / "policy_wave_v2"))
    parser.add_argument("--summary_path", type=str, default="")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--policy_ids", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_episodes", type=int, default=128)
    parser.add_argument("--max_env_steps", type=int, default=80000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--include_slices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slice_count", type=int, default=3)

    parser.add_argument("--llm_model_id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--llm_layer", type=int, default=-1)
    parser.add_argument("--llm_hidden_tokens", type=int, default=1)
    parser.add_argument("--llm_generation_tokens", type=int, default=0)

    parser.add_argument("--value_dataset_dir", type=str, default="/data/group_data/rl/geney/vllm_craftax_labelled_results")
    parser.add_argument("--value_data_glob", type=str, default="trajectories_batch_*.npz")
    parser.add_argument("--value_num_samples", type=int, default=20000)
    parser.add_argument("--value_hidden_mode", type=str, default="real", choices=["real", "zero"])
    parser.add_argument(
        "--value_pairs",
        type=str,
        default="health,food,drink,energy,wood,stone,floor1_zombie_adjacent,floor1_skeleton_adjacent,floor2_witch_adjacent",
    )

    parser.add_argument("--bundle_dirs", type=str, default="")

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="craftax_symbolic_evals")
    parser.add_argument("--wandb_entity", type=str, default="iris-sobolmark")

    args = parser.parse_args()

    tracks = parse_tracks(args.tracks)
    seeds = parse_seed_list(args.seeds)
    requested_policy_ids = parse_policy_ids(args.policy_ids)
    value_pairs = [x.strip() for x in str(args.value_pairs).split(",") if x.strip()]
    bundle_dirs = [x.strip() for x in str(args.bundle_dirs).split(",") if x.strip()]

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text())
    if requested_policy_ids:
        requested_set = set(requested_policy_ids)
        source_policies = manifest.get("policies", [])
        selected_policies = [p for p in source_policies if str(p.get("id", "")) in requested_set]
        resolved_ids = {str(p.get("id", "")) for p in selected_policies}
        missing = sorted(requested_set - resolved_ids)
        if missing:
            raise RuntimeError(
                f"Requested policy_ids missing from manifest: {missing}. "
                f"Manifest={manifest_path}"
            )
        manifest = dict(manifest)
        manifest["policies"] = selected_policies

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    specs = resolve_manifest_policies(manifest, include_slices=args.include_slices, slice_count=args.slice_count)
    if not specs:
        raise RuntimeError("No policy specs resolved from manifest")

    needs_llm = any(s.hidden_mode == "llm" for s in specs) and any(
        t in tracks for t in ["id", "ood", "gameplay_llm", "bundle"]
    )
    llm_manager = None
    if needs_llm:
        llm_manager = SharedLLMManager(
            model_id=args.llm_model_id,
            target_layer=args.llm_layer,
            tokens_to_generate=args.llm_hidden_tokens,
        )

    # Resolve OOD scenario library from manifest (with defaults).
    ood_scenarios = manifest.get(
        "ood_scenarios",
        [
            {"id": "ood_floor2", "type": "set_floor", "floor": 2},
            {"id": "ood_floor3", "type": "set_floor", "floor": 3},
            {
                "id": "ood_floor1_zombie",
                "type": "inject_mob",
                "mob_class": 0,
                "mob_type": 0,
                "row_offset": 1,
                "col_offset": 0,
                "floor": 1,
            },
            {
                "id": "ood_floor1_skeleton",
                "type": "inject_mob",
                "mob_class": 2,
                "mob_type": 0,
                "row_offset": 1,
                "col_offset": 0,
                "floor": 1,
            },
            {
                "id": "ood_floor2_witch",
                "type": "inject_mob",
                "mob_class": 2,
                "mob_type": 6,
                "row_offset": 1,
                "col_offset": 0,
                "floor": 2,
            },
            {
                "id": "diag_impossible_floor1_witch",
                "type": "inject_mob",
                "mob_class": 2,
                "mob_type": 6,
                "row_offset": 1,
                "col_offset": 0,
                "floor": 1,
                "diagnostic_only": True,
            },
        ],
    )

    run = maybe_init_wandb(args, manifest)

    output = {
        "timestamp": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "tracks": tracks,
        "num_policies": len(specs),
        "policy_ids": [spec.policy_id for spec in specs],
        "seeds": seeds,
        "results": {
            "id": {},
            "gameplay_llm": {},
            "ood": {},
            "bundle": {},
            "value": {},
        },
    }

    loaded_policies: Dict[Tuple[str, str], Dict] = {}
    for spec in specs:
        key = (spec.policy_name, spec.variant_name)
        loaded_policies[key] = load_policy(spec, device=device)

    if "id" in tracks:
        for spec in specs:
            key = (spec.policy_name, spec.variant_name)
            policy = loaded_policies[key]
            seed_runs = []
            for seed in seeds:
                metrics = evaluate_gameplay(
                    spec=spec,
                    policy=policy,
                    seed=seed,
                    num_envs=args.num_envs,
                    target_episodes=args.num_episodes,
                    max_env_steps=args.max_env_steps,
                    deterministic=args.deterministic,
                    llm_manager=llm_manager,
                    generation_tokens=0,
                    ood_transform=None,
                )
                seed_runs.append(metrics)
            entry = {
                "seed_runs": seed_runs,
                "aggregate": aggregate_seed_metrics(seed_runs),
            }
            output["results"]["id"][f"{spec.policy_name}:{spec.variant_name}"] = entry
            if run is not None:
                wandb.log({
                    f"id/{spec.policy_name}/{spec.variant_name}/return_mean": entry["aggregate"]["return_mean_over_seeds"],
                    f"id/{spec.policy_name}/{spec.variant_name}/achievement_mean": entry["aggregate"]["achievement_mean_over_seeds"],
                })

    if "gameplay_llm" in tracks:
        for spec in specs:
            if spec.hidden_mode != "llm":
                continue
            key = (spec.policy_name, spec.variant_name)
            policy = loaded_policies[key]
            seed_runs = []
            for seed in seeds:
                metrics = evaluate_gameplay(
                    spec=spec,
                    policy=policy,
                    seed=seed,
                    num_envs=args.num_envs,
                    target_episodes=args.num_episodes,
                    max_env_steps=args.max_env_steps,
                    deterministic=args.deterministic,
                    llm_manager=llm_manager,
                    generation_tokens=max(0, int(args.llm_generation_tokens)),
                    ood_transform=None,
                )
                seed_runs.append(metrics)
            output["results"]["gameplay_llm"][f"{spec.policy_name}:{spec.variant_name}"] = {
                "seed_runs": seed_runs,
                "aggregate": aggregate_seed_metrics(seed_runs),
            }

    if "ood" in tracks:
        for scenario in ood_scenarios:
            sid = str(scenario.get("id", scenario.get("type", "ood")))
            transform = make_ood_transform(scenario)
            output["results"]["ood"][sid] = {
                "diagnostic_only": bool(scenario.get("diagnostic_only", False)),
                "policies": {},
            }

            for spec in specs:
                key = (spec.policy_name, spec.variant_name)
                policy = loaded_policies[key]
                seed_runs = []
                for seed in seeds:
                    metrics = evaluate_gameplay(
                        spec=spec,
                        policy=policy,
                        seed=seed,
                        num_envs=args.num_envs,
                        target_episodes=args.num_episodes,
                        max_env_steps=args.max_env_steps,
                        deterministic=args.deterministic,
                        llm_manager=llm_manager,
                        generation_tokens=0,
                        ood_transform=transform,
                    )
                    seed_runs.append(metrics)
                output["results"]["ood"][sid]["policies"][f"{spec.policy_name}:{spec.variant_name}"] = {
                    "seed_runs": seed_runs,
                    "aggregate": aggregate_seed_metrics(seed_runs),
                }

    if "bundle" in tracks:
        all_bundle_dirs = bundle_dirs or manifest.get("bundle_dirs", [])
        for spec in specs:
            key = (spec.policy_name, spec.variant_name)
            policy = loaded_policies[key]
            output["results"]["bundle"][f"{spec.policy_name}:{spec.variant_name}"] = evaluate_bundle_reactions(
                spec=spec,
                policy=policy,
                bundle_dirs=all_bundle_dirs,
                llm_manager=llm_manager,
                generation_tokens=max(0, int(args.llm_generation_tokens)),
            )

    if "value" in tracks:
        checkpoint_paths = sorted(
            {
                spec.checkpoint_path
                for spec in specs
                if spec.policy_type != "ppo_orbax"
                and Path(spec.checkpoint_path).suffix in {".pth", ".msgpack"}
            }
        )
        output["results"]["value"] = run_value_battery(
            checkpoints=checkpoint_paths,
            output_dir=output_dir / "value_battery",
            dataset_dir=args.value_dataset_dir,
            data_glob=args.value_data_glob,
            num_samples=args.value_num_samples,
            num_envs=args.num_envs,
            hidden_mode=args.value_hidden_mode,
            pair_types=value_pairs,
            torch_device=str(device),
        )

    if args.summary_path:
        summary_path = Path(args.summary_path).expanduser()
        if not summary_path.is_absolute():
            summary_path = output_dir / summary_path
        summary_path = summary_path.resolve()
    else:
        summary_path = output_dir / "policy_wave_v2_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote summary: {summary_path}")

    if run is not None:
        run.summary["summary_json"] = str(summary_path)
        run.finish()


if __name__ == "__main__":
    main()
