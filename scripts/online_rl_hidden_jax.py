#!/usr/bin/env python3
"""
Online RL with LLM Hidden States - Optimized JAX Implementation

Two-phase architecture for maximum performance:
- Phase A (Python): Text rendering + vLLM hidden state extraction every skip_n steps
- Phase B (JIT-compiled): Run env steps + PPO update via jax.lax.scan

Verification modes:
- --no-llm: Uses ActorCritic (no hidden state), should match ppo.py exactly
- --skip-n 1: LLM every step, matches online_rl_hidden.py behavior
- --skip-n N: LLM every N steps for speed/quality tradeoff
"""

import argparse
import errno
import glob
import json
import os
import pickle
import shutil
import signal
import sys
from collections import deque
from datetime import datetime

# GPU memory sharing: JAX (XLA) and vLLM (PyTorch/CUDA) can coexist on the
# same GPU. Key settings to avoid conflicts:
#   - Disable JAX CUDA command buffers (CUDA graphs): they share a limited pool
#     with vLLM's graphs and cause "command buffer OOM" errors at instantiation.
#   - Don't preallocate: let JAX allocate on demand, vLLM already holds 60%.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # JAX gets 30%, vLLM gets 60%
# Disable CUDA command buffers so JAX doesn't compete with vLLM's CUDA graphs.
# Without this, JAX tries to instantiate CUDA graphs that OOM against vLLM's pool.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import time
from typing import Dict, List, NamedTuple, Optional, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
import wandb
from craftax.craftax.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name

# Import wrappers (same as ppo.py)
from envs.wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper

# Import models from shared module (same as ppo.py uses)
from models.actor_critic import ActorCritic, ActorCriticAug

# Import text processing and vLLM interface
from llm.prompts import (
    create_prompt,
    filter_text_obs,
    get_generation_prefix,
    get_prompt_outline,
    get_system_prompt,
)
from llm.extractor import VLLMHiddenStateExtractor
from labelling.obs_to_text import obs_to_text
import requests


TERMINATION_REQUESTED = False
TERMINATION_SIGNAL = None


def _signal_handler(signum, _frame):
    global TERMINATION_REQUESTED, TERMINATION_SIGNAL
    TERMINATION_REQUESTED = True
    TERMINATION_SIGNAL = signum
    print(
        f"\nReceived signal {signum}; will checkpoint and exit at the next safe boundary.",
        flush=True,
    )


# =============================================================================
# Configuration (matches ppo.py defaults)
# =============================================================================

@dataclass
class Config:
    # Environment
    ENV_NAME: str = "Craftax-Symbolic-v1"

    # LLM
    MODEL_ID: str = "Qwen/Qwen3-4B"
    HIDDEN_SIZE: int = 2560

    # Policy network
    LAYER_SIZE: int = 512

    # PPO hyperparameters (SAME as ppo.py defaults)
    LR: float = 2e-4
    ANNEAL_LR: bool = True
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.8
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 1.0
    NUM_STEPS: int = 64
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 8

    # WandB
    WANDB_PROJECT: str = "craftax-online-rl-llm"
    WANDB_ENTITY: str = "iris-sobolmark"


SYMBOLIC_MAP_DIM = 8217
SYMBOLIC_AUX_DIM = 51
SYMBOLIC_OBS_DIM = SYMBOLIC_MAP_DIM + SYMBOLIC_AUX_DIM


def _extract_step_from_text(text: str) -> int:
    m = re.search(r"_step(\d+)", text)
    if m is None:
        return -1
    return int(m.group(1))


class TrajectoryWriter:
    def __init__(
        self,
        enabled: bool,
        save_dir: Optional[str],
        save_every_updates: int,
        min_free_gb: float,
        run_name: str,
        schema: str = "minimal_core",
    ):
        self.enabled = bool(enabled)
        self.save_dir = os.path.expanduser(save_dir) if save_dir else None
        self.save_every_updates = max(1, int(save_every_updates))
        self.min_free_bytes = int(float(min_free_gb) * (1024 ** 3))
        self.run_name = run_name
        self.schema = schema
        self._disabled_reason = None
        self._disable_logged_at_update = -1
        self._next_batch_idx = 0
        if self.enabled and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def _free_bytes(self) -> int:
        if not self.save_dir:
            return 0
        stat = shutil.disk_usage(self.save_dir)
        return int(stat.free)

    def _disable(self, reason: str, update_idx: int):
        if self._disabled_reason is None:
            self._disabled_reason = reason
            marker = {
                "disabled": True,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "run_name": self.run_name,
            }
            if self.save_dir:
                marker_path = os.path.join(
                    self.save_dir, f"{self.run_name}_trajectory_disabled.json"
                )
                try:
                    with open(marker_path, "w", encoding="utf-8") as f:
                        json.dump(marker, f, indent=2, sort_keys=True)
                except Exception:
                    pass
        # Keep reminders sparse but visible. Always log the first disable event.
        if self._disable_logged_at_update < 0 or update_idx - self._disable_logged_at_update >= 100:
            print(
                "[trajectory] DISABLED permanently for this run segment: "
                f"{self._disabled_reason} (update={update_idx}). "
                "Future trajectory writes will be skipped.",
                flush=True,
            )
            self._disable_logged_at_update = update_idx

    def export_state(self) -> Dict:
        return {
            "enabled": bool(self.enabled),
            "save_dir": self.save_dir,
            "save_every_updates": int(self.save_every_updates),
            "min_free_bytes": int(self.min_free_bytes),
            "run_name": self.run_name,
            "schema": self.schema,
            "disabled_reason": self._disabled_reason,
            "next_batch_idx": int(self._next_batch_idx),
        }

    @classmethod
    def from_state(cls, state: Dict):
        writer = cls(
            enabled=bool(state.get("enabled", False)),
            save_dir=state.get("save_dir"),
            save_every_updates=int(state.get("save_every_updates", 1)),
            min_free_gb=float(state.get("min_free_bytes", 0)) / (1024 ** 3),
            run_name=state.get("run_name", "run"),
            schema=state.get("schema", "minimal_core"),
        )
        writer._disabled_reason = state.get("disabled_reason")
        writer._next_batch_idx = int(state.get("next_batch_idx", 0))
        return writer

    def _pack_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        obs_flat = np.asarray(obs)
        if obs_flat.ndim != 2:
            obs_flat = obs_flat.reshape(obs_flat.shape[0], -1)
        if obs_flat.shape[1] != SYMBOLIC_OBS_DIM:
            return {"obs": obs_flat.astype(np.float16, copy=False), "obs_packed": np.array(0, dtype=np.uint8)}

        map_part = obs_flat[:, :SYMBOLIC_MAP_DIM]
        aux_part = obs_flat[:, SYMBOLIC_MAP_DIM:]
        rounded = np.rint(map_part)
        # Pack only if the map slice is binary to avoid lossy conversion.
        is_binary = np.max(np.abs(map_part - rounded)) < 1e-5
        payload = {"obs_aux": aux_part.astype(np.float16, copy=False)}
        if is_binary:
            bits = rounded.astype(np.uint8, copy=False)
            payload["obs_map_bits"] = np.packbits(bits, axis=1, bitorder="little")
            payload["obs_packed"] = np.array(1, dtype=np.uint8)
        else:
            payload["obs_map"] = map_part.astype(np.float16, copy=False)
            payload["obs_packed"] = np.array(0, dtype=np.uint8)
        return payload

    def maybe_save(self, traj_batch, update_idx: int, total_steps: int):
        if not self.enabled:
            return
        if self._disabled_reason is not None:
            self._disable(self._disabled_reason, update_idx)
            return
        if (update_idx + 1) % self.save_every_updates != 0:
            return
        if not self.save_dir:
            self._disable("missing save_dir", update_idx)
            return

        free_bytes = self._free_bytes()
        if free_bytes < self.min_free_bytes:
            self._disable(
                f"free space {free_bytes / (1024 ** 3):.1f}GB below floor "
                f"{self.min_free_bytes / (1024 ** 3):.1f}GB",
                update_idx,
            )
            return

        batch_data = {
            "action": np.asarray(jax.device_get(traj_batch.action), dtype=np.int16).reshape(-1),
            "reward": np.asarray(jax.device_get(traj_batch.reward), dtype=np.float16).reshape(-1),
            "done": np.asarray(jax.device_get(traj_batch.done), dtype=np.uint8).reshape(-1),
        }

        obs_flat = np.asarray(jax.device_get(traj_batch.obs))
        obs_flat = obs_flat.reshape(-1, obs_flat.shape[-1])
        batch_data.update(self._pack_obs(obs_flat))

        if hasattr(traj_batch, "hidden_state"):
            hidden = np.asarray(jax.device_get(traj_batch.hidden_state))
            hidden = hidden.reshape(-1, hidden.shape[-1]).astype(np.float16, copy=False)
            batch_data["hidden_state"] = hidden

        batch_data["obs_dim"] = np.array(SYMBOLIC_OBS_DIM, dtype=np.int32)
        batch_data["obs_map_dim"] = np.array(SYMBOLIC_MAP_DIM, dtype=np.int32)
        batch_data["obs_aux_dim"] = np.array(SYMBOLIC_AUX_DIM, dtype=np.int32)

        out_path = os.path.join(
            self.save_dir,
            f"{self.run_name}_traj_step{int(total_steps):012d}_batch{self._next_batch_idx:06d}.npz",
        )
        try:
            np.savez_compressed(out_path, **batch_data)
            self._next_batch_idx += 1
            print(
                f"[trajectory] saved batch={self._next_batch_idx} step={total_steps} "
                f"path={out_path}",
                flush=True,
            )
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                self._disable("no space left on device; disabling future writes", update_idx)
            else:
                self._disable(f"OSError while saving trajectories: {exc}", update_idx)
        except Exception as exc:
            self._disable(f"unexpected error while saving trajectories: {exc}", update_idx)

# =============================================================================
# Transition storage for PPO
# =============================================================================

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


class TransitionAug(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    hidden_state: jnp.ndarray
    info: dict


def _extract_episode_metrics(traj_info: dict) -> Dict[str, float]:
    """Compute completed-episode aggregate metrics from traj info."""
    metrics: Dict[str, float] = {}
    done = traj_info["returned_episode"]
    completed = float(jax.device_get(jnp.sum(done)))
    metrics["train/completed_episodes"] = completed
    if completed <= 0:
        return metrics

    metrics["train/episode_return"] = float(
        jax.device_get(jnp.sum(traj_info["returned_episode_returns"] * done) / completed)
    )
    metrics["train/episode_length"] = float(
        jax.device_get(jnp.sum(traj_info["returned_episode_lengths"] * done) / completed)
    )
    return metrics


def _extract_achievement_metrics(
    log_env_state,
    lifetime_any_unlocked: Optional[np.ndarray],
    lifetime_slot_unlocked: Optional[np.ndarray],
) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute snapshot + lifetime achievement metrics.

    Snapshot metrics are non-monotonic and reflect current active env states.
    Lifetime metrics are monotonic over the run.
    """
    metrics: Dict[str, float] = {}
    try:
        achievements_np = np.asarray(jax.device_get(log_env_state.env_state.achievements)).astype(bool)
        if achievements_np.ndim != 2:
            return metrics, lifetime_any_unlocked, lifetime_slot_unlocked

        # Snapshot (current state across env workers)
        snapshot_any = achievements_np.any(axis=0)
        snapshot_total_unique = int(snapshot_any.sum())
        snapshot_total_unlocks = int(achievements_np.sum())
        metrics["achievements/total_unique"] = float(snapshot_total_unique)
        metrics["achievements/total_unlocks"] = float(snapshot_total_unlocks)
        metrics["achievements/snapshot_total_unique"] = float(snapshot_total_unique)
        metrics["achievements/snapshot_total_unlocks"] = float(snapshot_total_unlocks)

        # Lifetime (monotonic for this run)
        if lifetime_any_unlocked is None:
            lifetime_any_unlocked = np.zeros_like(snapshot_any, dtype=bool)
        if lifetime_slot_unlocked is None:
            lifetime_slot_unlocked = np.zeros_like(achievements_np, dtype=bool)
        lifetime_any_unlocked |= snapshot_any
        lifetime_slot_unlocked |= achievements_np
        metrics["achievements/lifetime_total_unique"] = float(lifetime_any_unlocked.sum())
        metrics["achievements/lifetime_total_unlocks"] = float(lifetime_slot_unlocked.sum())

        snapshot_rate = achievements_np.mean(axis=0)
        lifetime_rate = lifetime_slot_unlocked.mean(axis=0)
        for idx, ach in enumerate(Achievement):
            if idx >= snapshot_rate.shape[0]:
                break
            ach_name = ach.name.lower().replace(" ", "_")
            metrics[f"achievements/{ach_name}_unlock_rate"] = float(snapshot_rate[idx])
            metrics[f"achievements/{ach_name}_lifetime_unlock_rate"] = float(lifetime_rate[idx])
    except Exception:
        pass
    return metrics, lifetime_any_unlocked, lifetime_slot_unlocked


def _extract_loss_metrics(loss_info: jnp.ndarray) -> Dict[str, float]:
    """Summarize PPO losses from [epochs, minibatches, 6] tensor."""
    metrics: Dict[str, float] = {}
    try:
        loss_np = np.asarray(jax.device_get(loss_info))
        if loss_np.ndim != 3 or loss_np.shape[-1] < 6:
            return metrics
        means = loss_np.mean(axis=(0, 1))
        metrics["train/total_loss"] = float(means[0])
        metrics["train/value_loss"] = float(means[1])
        metrics["train/policy_loss"] = float(means[2])
        metrics["train/entropy"] = float(means[3])
        metrics["train/approx_kl"] = float(means[4])
        metrics["train/clipfrac"] = float(means[5])
    except Exception:
        pass
    return metrics


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = float(np.var(y_true))
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def _truncate_log_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r", "")
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3] + "..."


def _build_cot_samples(
    text_observations: List[str],
    generated_texts: List[str],
    tokenizer,
    system_prompt: str,
    prompt_variant: str,
    generation_prefix: str,
    max_samples: int,
    max_chars: int,
) -> List[Dict[str, str]]:
    if not generated_texts or not text_observations or max_samples <= 0:
        return []

    n = min(len(generated_texts), len(text_observations))
    sample_count = min(max_samples, n)
    if sample_count <= 0:
        return []

    if sample_count == 1:
        sample_indices = [0]
    else:
        sample_indices = np.linspace(0, n - 1, num=sample_count, dtype=np.int32).tolist()

    deduped_indices: List[int] = []
    seen = set()
    for idx in sample_indices:
        idx_int = int(idx)
        if idx_int in seen:
            continue
        seen.add(idx_int)
        deduped_indices.append(idx_int)

    samples: List[Dict[str, str]] = []
    for env_idx in deduped_indices:
        response_text = generated_texts[env_idx] or ""
        full_prompt = create_prompt(
            text_observations[env_idx],
            tokenizer,
            system_prompt=system_prompt,
            prompt_variant=prompt_variant,
        )
        if generation_prefix:
            full_prompt += generation_prefix
        samples.append(
            {
                "prompt": _truncate_log_text(full_prompt, max_chars),
                "response": _truncate_log_text(response_text, max_chars),
            }
        )
    return samples


def _cot_samples_to_wandb_payload(samples: List[Dict[str, str]]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for idx, sample in enumerate(samples):
        payload[f"cot/sample_{idx}_prompt"] = str(sample.get("prompt", ""))
        payload[f"cot/sample_{idx}_response"] = str(sample.get("response", ""))
    return payload


def _append_jsonl_record(path: str, payload: Dict[str, object]) -> bool:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        return True
    except Exception as exc:
        print(f"[cot-log] WARNING: failed to append to {path}: {exc}", flush=True)
        return False


def _cot_log_due_for_update(
    update_idx_1based: int,
    first_updates: int,
    interval_updates: int,
) -> bool:
    if update_idx_1based <= 0:
        return False
    if first_updates > 0 and update_idx_1based <= first_updates:
        return True
    if interval_updates <= 0:
        return False
    return (update_idx_1based - max(0, first_updates)) % interval_updates == 0


def _maybe_save_policy(
    policy_save_dir: str,
    run_name: str,
    params,
    summary: Dict[str, float],
    metadata: Optional[Dict] = None,
) -> str:
    os.makedirs(policy_save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{run_name}_{ts}"
    params_path = os.path.join(policy_save_dir, f"{base}.msgpack")
    meta_path = os.path.join(policy_save_dir, f"{base}.json")
    params_cpu = jax.device_get(params)
    with open(params_path, "wb") as f:
        f.write(serialization.to_bytes(params_cpu))
    payload = dict(summary)
    if metadata is not None:
        payload["metadata"] = metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return params_path


def _save_policy_snapshot(
    policy_save_dir: str,
    run_name: str,
    tag: str,
    params,
    summary: Dict[str, float],
    metadata: Optional[Dict] = None,
) -> str:
    os.makedirs(policy_save_dir, exist_ok=True)
    base = f"{run_name}_{tag}"
    params_path = os.path.join(policy_save_dir, f"{base}.msgpack")
    meta_path = os.path.join(policy_save_dir, f"{base}.json")
    params_cpu = jax.device_get(params)
    with open(params_path, "wb") as f:
        f.write(serialization.to_bytes(params_cpu))
    payload = dict(summary)
    if metadata is not None:
        payload["metadata"] = metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return params_path


def _to_host_tree(tree):
    def _to_host_leaf(x):
        try:
            x_host = jax.device_get(x)
        except Exception:
            return x
        if isinstance(x_host, np.ndarray):
            return x_host
        if np.isscalar(x_host):
            return np.asarray(x_host)
        return x_host

    return jax.tree_util.tree_map(_to_host_leaf, tree)


def _save_resumable_checkpoint(
    checkpoint_dir: str,
    run_name: str,
    payload: Dict,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    total_steps = int(payload["total_steps"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{run_name}_resume_step{total_steps:012d}_{ts}"
    ckpt_path = os.path.join(checkpoint_dir, f"{base}.pkl")
    meta_path = os.path.join(checkpoint_dir, f"{base}.json")

    with open(ckpt_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_payload = {
        "run_name": run_name,
        "mode": payload.get("mode"),
        "total_steps": total_steps,
        "update_idx": int(payload.get("update_idx", 0)),
        "llm_calls": int(payload.get("llm_calls", 0)),
        "steps_since_llm": int(payload.get("steps_since_llm", 0)),
        "checkpoint_path": ckpt_path,
        "timestamp": ts,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, sort_keys=True)

    latest_meta_path = os.path.join(checkpoint_dir, "latest_resume.json")
    with open(latest_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, sort_keys=True)
    return ckpt_path


def _extract_step_from_resume_path(path: str) -> int:
    m = re.search(r"_resume_step(\d+)", os.path.basename(path))
    if m is None:
        return -1
    return int(m.group(1))


def _resolve_resume_checkpoint(resume_from: str) -> str:
    if os.path.isfile(resume_from):
        return resume_from
    if not os.path.isdir(resume_from):
        raise FileNotFoundError(f"Resume path is neither file nor directory: {resume_from}")

    latest_meta_path = os.path.join(resume_from, "latest_resume.json")
    if os.path.exists(latest_meta_path):
        with open(latest_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ckpt_path = meta.get("checkpoint_path")
        if ckpt_path and os.path.exists(ckpt_path):
            return ckpt_path

    candidates = glob.glob(os.path.join(resume_from, "*_resume_step*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No resumable checkpoints found under: {resume_from}")
    candidates.sort(key=lambda p: (_extract_step_from_resume_path(p), p))
    return candidates[-1]


def _load_resumable_checkpoint(resume_from: str) -> Tuple[Dict, str]:
    ckpt_path = _resolve_resume_checkpoint(resume_from)
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    return payload, ckpt_path


# =============================================================================
# Text Observation Processing
# =============================================================================

def render_craftax_text_swapped(state):
    """Render text with Row,Col coordinates."""
    from craftax.craftax.renderer import render_craftax_text
    st = render_craftax_text(state)
    lines = st.split('\n')
    new_lines = []
    coord_pattern = re.compile(r"^(-?\d+),\s*(-?\d+):")
    for line in lines:
        match = coord_pattern.match(line)
        if match:
            col, row = match.groups()
            new_line = line.replace(f"{col}, {row}:", f"{row}, {col}:", 1)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


# =============================================================================
# LLM Hidden State Manager
# =============================================================================

class LLMHiddenStateManager:
    def __init__(
        self,
        model_id: str = Config.MODEL_ID,
        target_layer: int = -1,
        tokens_to_generate: int = 1,
        prompt_variant: str = "default",
        hidden_pooling: str = "last_token",
        hidden_pooling_k: int = 8,
        temperature: float = 0.7,
        log_cot_text: bool = False,
        cot_log_samples: int = 2,
        cot_log_max_chars: int = 512,
    ):
        self.tokens_to_generate = tokens_to_generate
        self.temperature = temperature
        self.prompt_variant = (prompt_variant or "default").strip().lower()
        self.system_prompt = get_system_prompt(self.prompt_variant)
        self.prompt_outline = get_prompt_outline(self.prompt_variant)
        self.generation_prefix = get_generation_prefix(self.prompt_variant)
        self.log_cot_text = bool(log_cot_text and tokens_to_generate > 1)
        self.cot_log_samples = max(1, int(cot_log_samples))
        self.cot_log_max_chars = max(0, int(cot_log_max_chars))
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
        try:
            resp = requests.get(f"{vllm_url}/health", timeout=2)
            if resp.status_code != 200:
                raise Exception(f"Server returned status {resp.status_code}")
        except Exception as e:
            print(f"\n❌ ERROR: vLLM server not available at {vllm_url}")
            print(f"   Error: {e}")
            print(f"\n📝 To start: bash scripts/start_vllm_hidden.sh --mode last_token")
            sys.exit(1)

        print(f"✅ vLLM server connected at {vllm_url}")
        model_name = "./configs/vllm_hidden_qwen4b"
        extracted_layers = [8, 16, 24, 35]
        layer_index = -1 if target_layer == -1 else (extracted_layers.index(target_layer) if target_layer in extracted_layers else -1)

        self.llm = VLLMHiddenStateExtractor(
            server_url=vllm_url,
            model_name=model_name,
            model_id=model_id,
            target_layer=layer_index,
            hidden_pooling=hidden_pooling,
            hidden_pooling_k=hidden_pooling_k,
            prompt_variant=self.prompt_variant,
            system_prompt=self.system_prompt,
        )
        self.hidden_size = self.llm.hidden_size
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   Prompt variant: {self.prompt_variant}")

    def extract(
        self,
        obs_batch: jnp.ndarray,
        num_envs: int,
    ) -> Tuple[jnp.ndarray, Dict[str, float], Optional[List[Dict[str, str]]]]:
        t_start = time.perf_counter()
        # Convert once to host and decode text from symbolic observations.
        # This is dramatically faster than render_craftax_text(state) on per-env state objects.
        obs_host = np.asarray(jax.device_get(obs_batch))
        text_observations = []
        for i in range(num_envs):
            raw_text = obs_to_text(obs_host[i])
            # Hard fail on malformed interesting-map coordinates so training does
            # not silently continue with corrupted prompt geometry.
            filtered_text = filter_text_obs(raw_text, strict_map_validation=True)
            text_observations.append(filtered_text)
        t_text = time.perf_counter() - t_start

        t_llm_start = time.perf_counter()
        cot_samples = None
        if self.tokens_to_generate == 1:
            hidden_np, llm_metrics = self.llm.extract_hidden_states_no_cot(text_observations)
        else:
            hidden_np, generated_texts, llm_metrics = self.llm.extract_hidden_states(
                text_observations,
                batch_size=min(32, len(text_observations)),
                max_new_tokens=self.tokens_to_generate,
                temperature=self.temperature,
            )
            if generated_texts:
                lengths = np.asarray([len(t or "") for t in generated_texts], dtype=np.float32)
                if lengths.size > 0:
                    llm_metrics["llm/generated_chars_mean"] = float(lengths.mean())
                    llm_metrics["llm/generated_chars_max"] = float(lengths.max())
                    llm_metrics["llm/generated_chars_min"] = float(lengths.min())
            if self.log_cot_text:
                cot_samples = _build_cot_samples(
                    text_observations,
                    generated_texts,
                    tokenizer=self.llm.tokenizer,
                    system_prompt=self.system_prompt,
                    prompt_variant=self.prompt_variant,
                    generation_prefix=self.generation_prefix,
                    max_samples=self.cot_log_samples,
                    max_chars=self.cot_log_max_chars,
                )
        t_llm = time.perf_counter() - t_llm_start

        return jnp.asarray(hidden_np), {
            "timing/text_render_ms": t_text * 1000,
            "timing/llm_inference_ms": t_llm * 1000,
            **llm_metrics,
        }, cot_samples


# =============================================================================
# PPO Training - No LLM Mode (matches ppo.py exactly)
# =============================================================================

def make_train_no_llm(config, network, env, env_params):
    """Create JIT-compiled training functions for no-LLM mode."""

    @jax.jit
    def _env_step(carry, unused):
        train_state, env_state, last_obs, rng = carry
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(train_state.params, last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        transition = Transition(done=done, action=action, value=value, reward=reward, log_prob=log_prob, obs=last_obs, info=info)
        return (train_state, env_state, obsv, rng), transition

    @jax.jit
    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae
        _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
        return advantages, advantages + traj_batch.value

    @jax.jit
    def _update_epoch(update_state, unused):
        def _update_minibatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            def _loss_fn(params, traj_batch, gae, targets):
                pi, value = network.apply(params, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_loss = 0.5 * jnp.maximum(jnp.square(value - targets), jnp.square(value_pred_clipped - targets)).mean()
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                log_ratio = log_prob - traj_batch.log_prob
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor = -jnp.minimum(ratio * gae, jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae).mean()
                entropy = pi.entropy().mean()
                total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])
                return total_loss, (value_loss, loss_actor, entropy, approx_kl, clipfrac)
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, aux), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
            value_loss, loss_actor, entropy, approx_kl, clipfrac = aux
            loss_vec = jnp.asarray([total_loss, value_loss, loss_actor, entropy, approx_kl, clipfrac], dtype=jnp.float32)
            return train_state.apply_gradients(grads=grads), loss_vec

        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        permutation = jax.random.permutation(_rng, batch_size)
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), (traj_batch, advantages, targets))
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch)
        train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
        return (train_state, traj_batch, advantages, targets, rng), losses

    @jax.jit
    def _ppo_update(train_state, traj_batch, last_obs, rng):
        _, last_val = network.apply(train_state.params, last_obs)
        advantages, targets = _calculate_gae(traj_batch, last_val)
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        return update_state[0], update_state[-1], loss_info

    return _env_step, _ppo_update


# =============================================================================
# PPO Training - With LLM Hidden States
# =============================================================================

def make_train_with_llm(config, network, env, env_params):
    """Create JIT-compiled training functions for LLM-augmented mode."""

    @jax.jit
    def _env_step(carry, unused):
        train_state, env_state, last_obs, hidden_states, rng = carry
        rng, _rng = jax.random.split(rng)
        pi, value = network.apply(train_state.params, last_obs, hidden_states)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        transition = TransitionAug(done=done, action=action, value=value, reward=reward, log_prob=log_prob, obs=last_obs, hidden_state=hidden_states, info=info)
        return (train_state, env_state, obsv, hidden_states, rng), transition

    @jax.jit
    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae
        _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
        return advantages, advantages + traj_batch.value

    @jax.jit
    def _update_epoch(update_state, unused):
        def _update_minibatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            def _loss_fn(params, traj_batch, gae, targets):
                pi, value = network.apply(params, traj_batch.obs, traj_batch.hidden_state)
                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_loss = 0.5 * jnp.maximum(jnp.square(value - targets), jnp.square(value_pred_clipped - targets)).mean()
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                log_ratio = log_prob - traj_batch.log_prob
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor = -jnp.minimum(ratio * gae, jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae).mean()
                entropy = pi.entropy().mean()
                total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])
                return total_loss, (value_loss, loss_actor, entropy, approx_kl, clipfrac)
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, aux), grads = grad_fn(train_state.params, traj_batch, advantages, targets)
            value_loss, loss_actor, entropy, approx_kl, clipfrac = aux
            loss_vec = jnp.asarray([total_loss, value_loss, loss_actor, entropy, approx_kl, clipfrac], dtype=jnp.float32)
            return train_state.apply_gradients(grads=grads), loss_vec

        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        permutation = jax.random.permutation(_rng, batch_size)
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), (traj_batch, advantages, targets))
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch)
        train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
        return (train_state, traj_batch, advantages, targets, rng), losses

    @jax.jit
    def _ppo_update(train_state, traj_batch, last_obs, hidden_states, rng):
        _, last_val = network.apply(train_state.params, last_obs, hidden_states)
        advantages, targets = _calculate_gae(traj_batch, last_val)
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        return update_state[0], update_state[-1], loss_info

    return _env_step, _ppo_update


# =============================================================================
# Training Loop - No LLM (matches ppo.py)
# =============================================================================

def run_training_no_llm(
    num_envs: int,
    total_timesteps: int,
    num_steps: int,
    use_wandb: bool,
    seed: int,
    verbose: bool,
    save_policy: bool,
    policy_save_dir: str,
    run_name: str,
    checkpoint_every_steps: int,
    checkpoint_dir: Optional[str],
    resume_from: Optional[str],
    run_metadata: Optional[Dict] = None,
) -> Dict:
    print("=" * 70)
    print("Online RL - NO LLM MODE (matches ppo.py)")
    print("=" * 70)

    config = {
        "NUM_ENVS": num_envs, "NUM_STEPS": num_steps, "NUM_MINIBATCHES": Config.NUM_MINIBATCHES,
        "UPDATE_EPOCHS": Config.UPDATE_EPOCHS, "MINIBATCH_SIZE": num_envs * num_steps // Config.NUM_MINIBATCHES,
        "NUM_UPDATES": total_timesteps // num_steps // num_envs, "LR": Config.LR, "GAMMA": Config.GAMMA,
        "GAE_LAMBDA": Config.GAE_LAMBDA, "CLIP_EPS": Config.CLIP_EPS, "ENT_COEF": Config.ENT_COEF,
        "VF_COEF": Config.VF_COEF, "MAX_GRAD_NORM": Config.MAX_GRAD_NORM,
    }

    env = make_craftax_env_from_name(Config.ENV_NAME, auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    network = ActorCritic(env.action_space(env_params).n, Config.LAYER_SIZE)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
    network_params = network.init(init_rng, init_x)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    _env_step, _ppo_update = make_train_no_llm(config, network, env, env_params)

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)

    effective_checkpoint_dir = checkpoint_dir or policy_save_dir

    total_steps = 0
    episode_returns = deque(maxlen=100)
    start_update_idx = 0
    start_time = time.perf_counter()
    last_log_time, last_log_steps = start_time, 0
    last_log_update = 0
    lifetime_any_unlocked = None
    lifetime_slot_unlocked = None

    if resume_from:
        payload, resume_path = _load_resumable_checkpoint(resume_from)
        if payload.get("mode") != "no_llm":
            raise ValueError(
                f"Resume checkpoint mode mismatch: expected no_llm, got {payload.get('mode')}"
            )
        train_state = serialization.from_state_dict(train_state, payload["train_state_state"])
        env_state = serialization.from_state_dict(env_state, payload["env_state_state"])
        obs = jnp.asarray(payload["obs"])
        rng = jnp.asarray(payload["rng"])
        total_steps = int(payload.get("total_steps", 0))
        start_update_idx = int(
            payload.get("update_idx", total_steps // max(1, num_steps * num_envs))
        )
        for ret in payload.get("episode_returns_tail", []):
            episode_returns.append(float(ret))
        if payload.get("lifetime_any_unlocked") is not None:
            lifetime_any_unlocked = np.asarray(payload["lifetime_any_unlocked"]).astype(bool)
        if payload.get("lifetime_slot_unlocked") is not None:
            lifetime_slot_unlocked = np.asarray(payload["lifetime_slot_unlocked"]).astype(bool)
        print(
            f"Resumed from {resume_path}: total_steps={total_steps}, "
            f"start_update={start_update_idx}/{config['NUM_UPDATES']}"
        )

        # Reset timing anchors so SPS after resume reflects post-resume runtime.
        start_time = time.perf_counter()
        last_log_time, last_log_steps = start_time, total_steps
        last_log_update = start_update_idx

    if checkpoint_every_steps > 0:
        next_checkpoint_step = ((total_steps // checkpoint_every_steps) + 1) * checkpoint_every_steps
    else:
        next_checkpoint_step = None

    final_update_idx = start_update_idx
    for update_idx in range(start_update_idx, config["NUM_UPDATES"]):
        if TERMINATION_REQUESTED:
            print(
                f"Termination requested before update {update_idx}; exiting loop.",
                flush=True,
            )
            break
        carry = (train_state, env_state, obs, rng)
        carry, traj_batch = jax.lax.scan(_env_step, carry, None, num_steps)
        train_state, env_state, obs, rng = carry
        total_steps += num_steps * num_envs
        final_update_idx = update_idx + 1

        rng, update_rng = jax.random.split(rng)
        train_state, rng, loss_info = _ppo_update(train_state, traj_batch, obs, update_rng)

        completed_mask = traj_batch.info["returned_episode"].flatten()
        completed_returns = traj_batch.info["returned_episode_returns"].flatten()[completed_mask]
        if len(completed_returns) > 0:
            for ret in completed_returns.tolist():
                episode_returns.append(float(ret))

        current_time = time.perf_counter()
        if (update_idx + 1) % 10 == 0:
            elapsed = current_time - last_log_time
            update_delta = (update_idx + 1) - last_log_update
            sps = (total_steps - last_log_steps) / elapsed
            updates_per_sec = update_delta / elapsed
            episode_metrics = _extract_episode_metrics(traj_batch.info)
            achievement_metrics, lifetime_any_unlocked, lifetime_slot_unlocked = _extract_achievement_metrics(
                env_state, lifetime_any_unlocked, lifetime_slot_unlocked
            )
            loss_metrics = _extract_loss_metrics(loss_info)
            targets_np = np.asarray(jax.device_get(traj_batch.reward + (config["GAMMA"] * traj_batch.value * (1 - traj_batch.done))))
            values_np = np.asarray(jax.device_get(traj_batch.value))
            perf_metrics = {
                "perf/updates_per_sec": updates_per_sec,
                "train/explained_variance": _explained_variance(values_np.reshape(-1), targets_np.reshape(-1)),
            }
            mean_return = episode_metrics.get("train/episode_return", 0.0)
            if verbose:
                print(
                    f"Update {update_idx+1:4d}/{config['NUM_UPDATES']} | Steps: {total_steps:,} "
                    f"| SPS: {sps:,.0f} | Return: {mean_return:.1f}"
                )
            if use_wandb:
                wandb.log(
                    {
                        "timestep": total_steps,
                        "perf/sps": sps,
                        **episode_metrics,
                        **loss_metrics,
                        **perf_metrics,
                        **achievement_metrics,
                    },
                    step=total_steps,
                )
            last_log_time, last_log_steps = current_time, total_steps
            last_log_update = update_idx + 1

        if save_policy and next_checkpoint_step is not None and total_steps >= next_checkpoint_step:
            mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
            checkpoint_summary = {
                "timestep": int(total_steps),
                "sps": float(total_steps / max(1e-6, (time.perf_counter() - start_time))),
                "final_return": mean_return,
                "intermediate": True,
            }
            tag = f"step{total_steps:012d}"
            policy_path = _save_policy_snapshot(
                policy_save_dir=policy_save_dir,
                run_name=run_name,
                tag=tag,
                params=train_state.params,
                summary=checkpoint_summary,
                metadata=run_metadata,
            )
            resume_payload = {
                "mode": "no_llm",
                "run_name": run_name,
                "total_steps": int(total_steps),
                "update_idx": int(update_idx + 1),
                "train_state_state": _to_host_tree(serialization.to_state_dict(train_state)),
                "env_state_state": _to_host_tree(serialization.to_state_dict(env_state)),
                "obs": np.asarray(jax.device_get(obs)),
                "rng": np.asarray(jax.device_get(rng)),
                "episode_returns_tail": list(episode_returns),
                "lifetime_any_unlocked": None
                if lifetime_any_unlocked is None
                else np.asarray(lifetime_any_unlocked, dtype=bool),
                "lifetime_slot_unlocked": None
                if lifetime_slot_unlocked is None
                else np.asarray(lifetime_slot_unlocked, dtype=bool),
                "saved_policy_path": policy_path,
            }
            resume_path = _save_resumable_checkpoint(
                checkpoint_dir=effective_checkpoint_dir,
                run_name=run_name,
                payload=resume_payload,
            )
            print(
                f"Saved intermediate checkpoint at step {total_steps}: "
                f"policy={policy_path} resume={resume_path}"
            )
            next_checkpoint_step = ((total_steps // checkpoint_every_steps) + 1) * checkpoint_every_steps

    total_time = time.perf_counter() - start_time
    final_return = float(np.mean(episode_returns)) if episode_returns else 0
    final_metrics = {
        "sps": total_steps / max(total_time, 1e-6),
        "final_return": final_return,
        "terminated_by_signal": bool(TERMINATION_REQUESTED),
        "termination_signal": TERMINATION_SIGNAL,
    }
    if save_policy:
        save_path = _maybe_save_policy(policy_save_dir, run_name, train_state.params, final_metrics, run_metadata)
        print(f"Saved policy checkpoint: {save_path}")
        final_metrics["policy_path"] = save_path
        final_resume_payload = {
            "mode": "no_llm",
            "run_name": run_name,
            "total_steps": int(total_steps),
            "update_idx": int(final_update_idx),
            "train_state_state": _to_host_tree(serialization.to_state_dict(train_state)),
            "env_state_state": _to_host_tree(serialization.to_state_dict(env_state)),
            "obs": np.asarray(jax.device_get(obs)),
            "rng": np.asarray(jax.device_get(rng)),
            "episode_returns_tail": list(episode_returns),
            "lifetime_any_unlocked": None
            if lifetime_any_unlocked is None
            else np.asarray(lifetime_any_unlocked, dtype=bool),
            "lifetime_slot_unlocked": None
            if lifetime_slot_unlocked is None
            else np.asarray(lifetime_slot_unlocked, dtype=bool),
            "saved_policy_path": save_path,
            "terminated_by_signal": bool(TERMINATION_REQUESTED),
            "termination_signal": TERMINATION_SIGNAL,
        }
        final_resume_path = _save_resumable_checkpoint(
            checkpoint_dir=effective_checkpoint_dir,
            run_name=run_name,
            payload=final_resume_payload,
        )
        final_metrics["resume_path"] = final_resume_path
        print(f"Saved resumable checkpoint: {final_resume_path}")
    print(f"\nDone. SPS: {total_steps/total_time:,.0f}, Return: {final_return:.1f}")
    return final_metrics


# =============================================================================
# Training Loop - With LLM
# =============================================================================

def run_training_with_llm(
    num_envs: int,
    total_timesteps: int,
    skip_n: int,
    num_steps: int,
    model_id: str,
    target_layer: int,
    tokens_to_generate: int,
    prompt_variant: str,
    use_wandb: bool,
    seed: int,
    verbose: bool,
    save_policy: bool,
    policy_save_dir: str,
    run_name: str,
    checkpoint_every_steps: int,
    checkpoint_dir: Optional[str],
    resume_from: Optional[str],
    run_metadata: Optional[Dict] = None,
    hidden_pooling: str = "last_token",
    hidden_pooling_k: int = 8,
    temperature: float = 0.7,
    save_traj_online: bool = False,
    traj_save_dir: Optional[str] = None,
    traj_save_every_updates: int = 50,
    traj_free_space_min_gb: float = 150.0,
    traj_schema: str = "minimal_core",
    log_cot_text: bool = False,
    cot_log_first_updates: int = 10,
    cot_log_every_updates: int = 100,
    cot_log_samples: int = 2,
    cot_log_max_chars: int = 0,
    cot_log_file: Optional[str] = None,
) -> Dict:
    print("=" * 70)
    print(f"Online RL with LLM Hidden States (skip_n={skip_n})")
    print(f"Prompt variant: {prompt_variant}")
    print("=" * 70)

    config = {
        "NUM_ENVS": num_envs, "NUM_STEPS": num_steps, "NUM_MINIBATCHES": Config.NUM_MINIBATCHES,
        "UPDATE_EPOCHS": Config.UPDATE_EPOCHS, "MINIBATCH_SIZE": num_envs * num_steps // Config.NUM_MINIBATCHES,
        "NUM_UPDATES": total_timesteps // num_steps // num_envs, "LR": Config.LR, "GAMMA": Config.GAMMA,
        "GAE_LAMBDA": Config.GAE_LAMBDA, "CLIP_EPS": Config.CLIP_EPS, "ENT_COEF": Config.ENT_COEF,
        "VF_COEF": Config.VF_COEF, "MAX_GRAD_NORM": Config.MAX_GRAD_NORM,
    }

    env = make_craftax_env_from_name(Config.ENV_NAME, auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    cot_text_logging_enabled = bool(log_cot_text and tokens_to_generate > 1)
    if log_cot_text and tokens_to_generate <= 1:
        print("CoT text logging requested but disabled because --tokens <= 1.")
    if cot_text_logging_enabled:
        print(
            f"CoT text logging: enabled (updates 1..{cot_log_first_updates}, "
            f"then every {cot_log_every_updates}, "
            f"samples={cot_log_samples}, max_chars={cot_log_max_chars})"
        )
        if cot_log_file:
            print(f"CoT text local mirror: {cot_log_file}")

    llm_manager = LLMHiddenStateManager(
        model_id=model_id,
        target_layer=target_layer,
        tokens_to_generate=tokens_to_generate,
        prompt_variant=prompt_variant,
        hidden_pooling=hidden_pooling,
        hidden_pooling_k=hidden_pooling_k,
        temperature=temperature,
        log_cot_text=cot_text_logging_enabled,
        cot_log_samples=cot_log_samples,
        cot_log_max_chars=cot_log_max_chars,
    )

    network = ActorCriticAug(
        action_dim=env.action_space(env_params).n,
        layer_width=Config.LAYER_SIZE,
        hidden_state_dim=llm_manager.hidden_size,
    )
    print("Using fixed ActorCriticAug architecture (dual-branch concat).")
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    obs_dim = env.observation_space(env_params).shape[0]
    network_params = network.init(init_rng, jnp.zeros((1, obs_dim)), jnp.zeros((1, llm_manager.hidden_size)))

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    print("Creating JIT-compiled training functions...", flush=True)
    _env_step, _ppo_update = make_train_with_llm(config, network, env, env_params)
    print("  Done.", flush=True)

    print("Resetting environment...", flush=True)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    hidden_states = jnp.zeros((num_envs, llm_manager.hidden_size))
    print("  Done.", flush=True)

    effective_checkpoint_dir = checkpoint_dir or policy_save_dir

    total_steps = 0
    llm_calls = 0
    episode_returns = deque(maxlen=100)
    steps_since_llm = skip_n  # Force LLM on first iter when not resuming
    start_update_idx = 0
    start_time = time.perf_counter()
    print("Starting training loop...", flush=True)
    last_log_time, last_log_steps = start_time, 0
    last_log_update, last_log_llm_calls = 0, 0
    lifetime_any_unlocked = None
    lifetime_slot_unlocked = None
    trajectory_writer = TrajectoryWriter(
        enabled=save_traj_online,
        save_dir=traj_save_dir,
        save_every_updates=traj_save_every_updates,
        min_free_gb=traj_free_space_min_gb,
        run_name=run_name,
        schema=traj_schema,
    )
    latest_cot_samples: Optional[List[Dict[str, str]]] = None
    cot_log_file_disabled = False

    if resume_from:
        payload, resume_path = _load_resumable_checkpoint(resume_from)
        if payload.get("mode") != "with_llm":
            raise ValueError(
                f"Resume checkpoint mode mismatch: expected with_llm, got {payload.get('mode')}"
            )
        train_state = serialization.from_state_dict(train_state, payload["train_state_state"])
        env_state = serialization.from_state_dict(env_state, payload["env_state_state"])
        obs = jnp.asarray(payload["obs"])
        rng = jnp.asarray(payload["rng"])
        hidden_states = jnp.asarray(payload["hidden_states"])
        total_steps = int(payload.get("total_steps", 0))
        llm_calls = int(payload.get("llm_calls", 0))
        steps_since_llm = int(payload.get("steps_since_llm", skip_n))
        start_update_idx = int(
            payload.get("update_idx", total_steps // max(1, num_steps * num_envs))
        )
        for ret in payload.get("episode_returns_tail", []):
            episode_returns.append(float(ret))
        if payload.get("lifetime_any_unlocked") is not None:
            lifetime_any_unlocked = np.asarray(payload["lifetime_any_unlocked"]).astype(bool)
        if payload.get("lifetime_slot_unlocked") is not None:
            lifetime_slot_unlocked = np.asarray(payload["lifetime_slot_unlocked"]).astype(bool)
        if payload.get("trajectory_writer_state") is not None:
            trajectory_writer = TrajectoryWriter.from_state(payload["trajectory_writer_state"])
            if traj_save_dir:
                trajectory_writer.save_dir = os.path.expanduser(traj_save_dir)
                os.makedirs(trajectory_writer.save_dir, exist_ok=True)
        print(
            f"Resumed from {resume_path}: total_steps={total_steps}, llm_calls={llm_calls}, "
            f"start_update={start_update_idx}/{config['NUM_UPDATES']}"
        )

        # Reset timing anchors so performance metrics after resume are sane.
        start_time = time.perf_counter()
        last_log_time, last_log_steps = start_time, total_steps
        last_log_update, last_log_llm_calls = start_update_idx, llm_calls

    if checkpoint_every_steps > 0:
        next_checkpoint_step = ((total_steps // checkpoint_every_steps) + 1) * checkpoint_every_steps
    else:
        next_checkpoint_step = None

    def _save_resume(update_idx: int, saved_policy_path: Optional[str], intermediate: bool):
        resume_payload = {
            "mode": "with_llm",
            "run_name": run_name,
            "total_steps": int(total_steps),
            "update_idx": int(update_idx),
            "llm_calls": int(llm_calls),
            "steps_since_llm": int(steps_since_llm),
            "train_state_state": _to_host_tree(serialization.to_state_dict(train_state)),
            "env_state_state": _to_host_tree(serialization.to_state_dict(env_state)),
            "obs": np.asarray(jax.device_get(obs)),
            "rng": np.asarray(jax.device_get(rng)),
            "hidden_states": np.asarray(jax.device_get(hidden_states)),
            "episode_returns_tail": list(episode_returns),
            "lifetime_any_unlocked": None
            if lifetime_any_unlocked is None
            else np.asarray(lifetime_any_unlocked, dtype=bool),
            "lifetime_slot_unlocked": None
            if lifetime_slot_unlocked is None
            else np.asarray(lifetime_slot_unlocked, dtype=bool),
            "saved_policy_path": saved_policy_path,
            "trajectory_writer_state": trajectory_writer.export_state(),
            "terminated_by_signal": bool(TERMINATION_REQUESTED),
            "termination_signal": TERMINATION_SIGNAL,
            "intermediate": bool(intermediate),
        }
        return _save_resumable_checkpoint(
            checkpoint_dir=effective_checkpoint_dir,
            run_name=run_name,
            payload=resume_payload,
        )

    final_update_idx = start_update_idx
    for update_idx in range(start_update_idx, config["NUM_UPDATES"]):
        if TERMINATION_REQUESTED:
            print(
                f"Termination requested before update {update_idx}; exiting loop.",
                flush=True,
            )
            break

        llm_metrics = {}
        steps_collected = 0
        all_transitions = []

        while steps_collected < num_steps:
            if steps_since_llm >= skip_n:
                hidden_states, llm_metrics, cot_samples = llm_manager.extract(obs, num_envs)
                steps_since_llm = 0
                llm_calls += 1
                if cot_samples:
                    latest_cot_samples = cot_samples

            steps_this_chunk = min(skip_n - steps_since_llm, num_steps - steps_collected)
            steps_this_chunk = max(1, steps_this_chunk)

            carry = (train_state, env_state, obs, hidden_states, rng)
            if update_idx == 0 and steps_collected == 0:
                print(f"  First scan ({steps_this_chunk} steps)...", flush=True)
            carry, traj_chunk = jax.lax.scan(_env_step, carry, None, steps_this_chunk)
            if update_idx == 0 and steps_collected == 0:
                print(f"  First scan complete.", flush=True)
            train_state, env_state, obs, hidden_states, rng = carry

            all_transitions.append(traj_chunk)
            steps_collected += steps_this_chunk
            steps_since_llm += steps_this_chunk

        traj_batch = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_transitions)
        total_steps += num_steps * num_envs
        final_update_idx = update_idx + 1

        rng, update_rng = jax.random.split(rng)
        train_state, rng, loss_info = _ppo_update(train_state, traj_batch, obs, hidden_states, update_rng)
        trajectory_writer.maybe_save(traj_batch, update_idx, total_steps)

        completed_mask = traj_batch.info["returned_episode"].flatten()
        completed_returns = traj_batch.info["returned_episode_returns"].flatten()[completed_mask]
        if len(completed_returns) > 0:
            for ret in completed_returns.tolist():
                episode_returns.append(float(ret))

        current_update_idx = update_idx + 1
        perf_log_due = (current_update_idx % 10) == 0
        cot_log_due = (
            cot_text_logging_enabled
            and latest_cot_samples is not None
            and _cot_log_due_for_update(
                current_update_idx,
                first_updates=max(0, int(cot_log_first_updates)),
                interval_updates=max(1, int(cot_log_every_updates)),
            )
        )
        current_time = time.perf_counter()
        if perf_log_due or cot_log_due:
            elapsed = current_time - last_log_time
            step_delta = total_steps - last_log_steps
            update_delta = current_update_idx - last_log_update
            llm_delta = llm_calls - last_log_llm_calls
            log_payload = {"timestep": total_steps}
            if perf_log_due:
                sps = step_delta / elapsed
                updates_per_sec = update_delta / elapsed
                llm_calls_per_sec = llm_delta / elapsed if elapsed > 0 else 0.0
                steps_per_llm_call = step_delta / max(llm_delta, 1)
                episode_metrics = _extract_episode_metrics(traj_batch.info)
                achievement_metrics, lifetime_any_unlocked, lifetime_slot_unlocked = _extract_achievement_metrics(
                    env_state, lifetime_any_unlocked, lifetime_slot_unlocked
                )
                loss_metrics = _extract_loss_metrics(loss_info)
                targets_np = np.asarray(
                    jax.device_get(
                        traj_batch.reward + (config["GAMMA"] * traj_batch.value * (1 - traj_batch.done))
                    )
                )
                values_np = np.asarray(jax.device_get(traj_batch.value))
                perf_metrics = {
                    "perf/updates_per_sec": updates_per_sec,
                    "perf/llm_calls_per_sec": llm_calls_per_sec,
                    "perf/steps_per_llm_call": steps_per_llm_call,
                    "train/explained_variance": _explained_variance(values_np.reshape(-1), targets_np.reshape(-1)),
                }
                mean_return = episode_metrics.get("train/episode_return", 0.0)
                text_ms = llm_metrics.get("timing/text_render_ms")
                llm_ms = llm_metrics.get("timing/llm_inference_ms")
                timing_suffix = ""
                if text_ms is not None and llm_ms is not None:
                    timing_suffix = f" | TextMS: {text_ms:.1f} | LLMMS: {llm_ms:.1f}"
                if verbose:
                    print(
                        f"Update {current_update_idx:4d}/{config['NUM_UPDATES']} | Steps: {total_steps:,} "
                        f"| SPS: {sps:,.0f} | Return: {mean_return:.1f} | LLM: {llm_calls}{timing_suffix}"
                    )
                log_payload.update(
                    {
                        "perf/sps": sps,
                        "perf/llm_calls": llm_calls,
                        **episode_metrics,
                        **loss_metrics,
                        **perf_metrics,
                        **achievement_metrics,
                        **llm_metrics,
                    }
                )
            else:
                log_payload["perf/llm_calls"] = llm_calls
            if cot_log_due:
                log_payload.update(_cot_samples_to_wandb_payload(latest_cot_samples))
                log_payload["cot/prompt_variant"] = llm_manager.prompt_variant
                log_payload["cot/prompt_outline"] = llm_manager.prompt_outline
            if use_wandb:
                wandb.log(
                    log_payload,
                    step=total_steps,
                )
            if cot_log_due and cot_log_file and not cot_log_file_disabled:
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "run_name": run_name,
                    "update_idx": int(current_update_idx),
                    "timestep": int(total_steps),
                    "llm_calls": int(llm_calls),
                    "prompt_variant": llm_manager.prompt_variant,
                    "prompt_outline": llm_manager.prompt_outline,
                    "samples": latest_cot_samples,
                }
                if not _append_jsonl_record(cot_log_file, record):
                    cot_log_file_disabled = True
                    print(
                        "[cot-log] disabling local CoT mirror after write failure.",
                        flush=True,
                    )
                elif verbose:
                    print(
                        f"[cot-log] wrote {len(latest_cot_samples)} samples "
                        f"to {cot_log_file}",
                        flush=True,
                    )
            if perf_log_due:
                last_log_time, last_log_steps = current_time, total_steps
                last_log_update, last_log_llm_calls = current_update_idx, llm_calls

        if save_policy and next_checkpoint_step is not None and total_steps >= next_checkpoint_step:
            mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
            checkpoint_summary = {
                "timestep": int(total_steps),
                "sps": float(total_steps / max(1e-6, (time.perf_counter() - start_time))),
                "llm_calls": int(llm_calls),
                "final_return": mean_return,
                "intermediate": True,
            }
            tag = f"step{total_steps:012d}"
            policy_path = _save_policy_snapshot(
                policy_save_dir=policy_save_dir,
                run_name=run_name,
                tag=tag,
                params=train_state.params,
                summary=checkpoint_summary,
                metadata=run_metadata,
            )
            resume_path = _save_resume(update_idx + 1, policy_path, intermediate=True)
            print(
                f"Saved intermediate checkpoint at step {total_steps}: "
                f"policy={policy_path} resume={resume_path}"
            )
            next_checkpoint_step = ((total_steps // checkpoint_every_steps) + 1) * checkpoint_every_steps

        if TERMINATION_REQUESTED:
            print(
                f"Termination requested after update {update_idx + 1}; saving resumable checkpoint.",
                flush=True,
            )
            resume_path = _save_resume(update_idx + 1, None, intermediate=True)
            print(f"Saved termination checkpoint: {resume_path}", flush=True)
            break

    total_time = time.perf_counter() - start_time
    final_return = float(np.mean(episode_returns)) if episode_returns else 0
    final_metrics = {
        "sps": total_steps / max(total_time, 1e-6),
        "llm_calls": llm_calls,
        "final_return": final_return,
        "terminated_by_signal": bool(TERMINATION_REQUESTED),
        "termination_signal": TERMINATION_SIGNAL,
    }
    if save_policy:
        save_path = _maybe_save_policy(policy_save_dir, run_name, train_state.params, final_metrics, run_metadata)
        print(f"Saved policy checkpoint: {save_path}")
        final_metrics["policy_path"] = save_path
        final_resume_path = _save_resume(final_update_idx, save_path, intermediate=False)
        final_metrics["resume_path"] = final_resume_path
        print(f"Saved resumable checkpoint: {final_resume_path}")
    print(
        f"\nDone. SPS: {total_steps / max(total_time, 1e-6):,.0f}, "
        f"LLM calls: {llm_calls}, Return: {final_return:.1f}"
    )
    return final_metrics


# =============================================================================
# Entry Point
# =============================================================================

def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(description="Online RL with LLM hidden states (Optimized JAX)")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--timesteps", type=lambda x: int(float(x)), default=1e6)
    parser.add_argument("--skip-n", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (matches ppo.py)")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="default",
        choices=["default", "future_based", "future_based_opt"],
        help="Prompt construction variant for the LLM policy context.",
    )
    parser.add_argument(
        "--hidden-pooling",
        type=str,
        default="last_token",
        choices=["last_token", "mean_last_k"],
        help="How to pool token-wise hidden states from the selected layer.",
    )
    parser.add_argument(
        "--hidden-pooling-k",
        type=int,
        default=8,
        help="Token window size for --hidden-pooling mean_last_k.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM generation temperature when --tokens > 1.",
    )
    parser.add_argument("--model", type=str, default=Config.MODEL_ID)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=Config.WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=Config.WANDB_ENTITY)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-save-dir", type=str, default="/data/group_data/rl/geney/online_rl_hidden_models")
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=10_000_000,
        help="Save intermediate checkpoints every N env steps (0 disables periodic checkpointing).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional directory for resumable checkpoints. Defaults to --policy-save-dir.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a checkpoint file or a checkpoint directory containing latest_resume.json.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run name. Defaults to online-jax-{envs}env-{mode}.",
    )
    parser.add_argument(
        "--save-traj-online",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save compact trajectory shards during online RL.",
    )
    parser.add_argument(
        "--traj-save-dir",
        type=str,
        default=None,
        help="Output directory for online trajectory shards.",
    )
    parser.add_argument(
        "--traj-save-every-updates",
        type=int,
        default=50,
        help="Save one trajectory shard every N PPO updates.",
    )
    parser.add_argument(
        "--traj-free-space-min-gb",
        type=float,
        default=150.0,
        help="Disable trajectory saving when free space drops below this threshold.",
    )
    parser.add_argument(
        "--traj-schema",
        type=str,
        default="minimal_core",
        choices=["minimal_core"],
        help="Trajectory schema version.",
    )
    parser.add_argument(
        "--log-cot-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log sampled CoT text traces (only applies when --tokens > 1).",
    )
    parser.add_argument(
        "--cot-log-every-updates",
        type=int,
        default=100,
        help="After warmup logging, log CoT text every N PPO updates.",
    )
    parser.add_argument(
        "--cot-log-first-updates",
        type=int,
        default=10,
        help="Log CoT text on updates 1..N before switching to periodic logging.",
    )
    parser.add_argument(
        "--cot-log-samples",
        type=int,
        default=2,
        help="Number of env samples to log per CoT logging event.",
    )
    parser.add_argument(
        "--cot-log-max-chars",
        type=int,
        default=0,
        help="Max characters per CoT prompt/response field (0 disables truncation).",
    )
    parser.add_argument(
        "--cot-log-file",
        type=str,
        default=None,
        help="Optional JSONL mirror path for sampled CoT logs.",
    )
    args = parser.parse_args()

    if args.checkpoint_every_steps < 0:
        parser.error("--checkpoint-every-steps must be >= 0")
    if args.hidden_pooling_k <= 0:
        parser.error("--hidden-pooling-k must be > 0")
    if args.traj_save_every_updates <= 0:
        parser.error("--traj-save-every-updates must be > 0")
    if args.traj_free_space_min_gb < 0:
        parser.error("--traj-free-space-min-gb must be >= 0")
    if args.cot_log_every_updates <= 0:
        parser.error("--cot-log-every-updates must be > 0")
    if args.cot_log_first_updates < 0:
        parser.error("--cot-log-first-updates must be >= 0")
    if args.cot_log_samples <= 0:
        parser.error("--cot-log-samples must be > 0")
    if args.cot_log_max_chars < 0:
        parser.error("--cot-log-max-chars must be >= 0")

    use_wandb = args.use_wandb and not args.no_wandb
    mode_str = "no-llm" if args.no_llm else f"skip{args.skip_n}"
    run_name = args.run_name or f"online-jax-{args.envs}env-{mode_str}"
    policy_save_dir = os.path.expanduser(args.policy_save_dir)
    checkpoint_dir = os.path.expanduser(args.checkpoint_dir) if args.checkpoint_dir else None
    resume_from = os.path.expanduser(args.resume_from) if args.resume_from else None
    traj_save_dir = os.path.expanduser(args.traj_save_dir) if args.traj_save_dir else None
    cot_log_file = os.path.expanduser(args.cot_log_file) if args.cot_log_file else None
    if args.save_traj_online and traj_save_dir is None:
        traj_save_dir = os.path.join(policy_save_dir, "online_traj", run_name)
    if args.log_cot_text and cot_log_file is None:
        cot_log_file = os.path.join(policy_save_dir, "cot_logs", f"{run_name}.jsonl")
    run_metadata = {
        "argv": vars(args),
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "git_commit": os.environ.get("GIT_COMMIT"),
        "prompt_pipeline": "filter_text_obs(obs_to_text(symbolic_obs))",
        "prompt_variant": args.prompt_variant,
        "prompt_outline": get_prompt_outline(args.prompt_variant),
        "hidden_pooling": args.hidden_pooling,
        "hidden_pooling_k": args.hidden_pooling_k,
        "temperature": args.temperature,
        "save_traj_online": args.save_traj_online,
        "traj_save_dir": traj_save_dir,
        "traj_schema": args.traj_schema,
        "log_cot_text": args.log_cot_text,
        "cot_log_first_updates": args.cot_log_first_updates,
        "cot_log_every_updates": args.cot_log_every_updates,
        "cot_log_samples": args.cot_log_samples,
        "cot_log_max_chars": args.cot_log_max_chars,
        "cot_log_file": cot_log_file,
    }
    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args))
        run_metadata["wandb_run_id"] = wandb.run.id if wandb.run is not None else None
        run_metadata["wandb_run_name"] = wandb.run.name if wandb.run is not None else None

    if args.no_llm:
        results = run_training_no_llm(
            args.envs, args.timesteps, args.num_steps, use_wandb, args.seed, not args.quiet,
            args.save_policy, policy_save_dir, run_name, args.checkpoint_every_steps, checkpoint_dir, resume_from, run_metadata
        )
    else:
        results = run_training_with_llm(
            args.envs, args.timesteps, args.skip_n, args.num_steps, args.model, args.layer, args.tokens,
            args.prompt_variant,
            use_wandb, args.seed, not args.quiet,
            args.save_policy, policy_save_dir, run_name, args.checkpoint_every_steps, checkpoint_dir, resume_from, run_metadata,
            hidden_pooling=args.hidden_pooling,
            hidden_pooling_k=args.hidden_pooling_k,
            temperature=args.temperature,
            save_traj_online=args.save_traj_online,
            traj_save_dir=traj_save_dir,
            traj_save_every_updates=args.traj_save_every_updates,
            traj_free_space_min_gb=args.traj_free_space_min_gb,
            traj_schema=args.traj_schema,
            log_cot_text=args.log_cot_text,
            cot_log_first_updates=args.cot_log_first_updates,
            cot_log_every_updates=args.cot_log_every_updates,
            cot_log_samples=args.cot_log_samples,
            cot_log_max_chars=args.cot_log_max_chars,
            cot_log_file=cot_log_file,
        )

    if use_wandb:
        wandb.finish()
    print("\n✅ Done!")
    return results


if __name__ == "__main__":
    main()
