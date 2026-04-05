#!/usr/bin/env python3
"""
Online PPO training with imagination augmentation for Craftax.

Each env-step cycle:
  1. Every 15 env-steps per env: obs_to_text → Gemini prediction → Qwen embed
  2. Policy(obs, hidden_state) → action
  3. Standard PPO update (GAE, clipped surrogate, value clipping)

Warm-starts from an AWR checkpoint (same ActorCriticAug architecture).

Usage:
    python -m pipeline.train_ppo_augmented --total-timesteps 100000000
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

# JAX memory — set before importing jax
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.optim as optim

from models.actor_critic_aug import ActorCriticAug

# -- Craftax / pipeline imports --
from craftax.craftax_env import make_craftax_env_from_name
from labelling.obs_to_text import obs_to_text
from llm.prompts import filter_text_obs
from envs.wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper

from pipeline.config import (
    ACTION_NAMES,
    EMBED_HIDDEN_DIM,
    EMBED_LAYER,
    EMBED_MODEL,
    GEMINI_BASE_URL,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
)

# ======================================================================
# Constants
# ======================================================================
STEP_CADENCE = 15
ACTION_DIM = 43
OBS_DIM = 8268
LAYER_WIDTH = 512

AWR_CKPT_DIR = Path("/data/group_data/rl/geney/checkpoints/awr_imagination")
PREDICT_TEMPLATE_PATH = (
    Path.home()
    / "Craftax_Baselines/configs/future_imagination/templates"
    / "predict_state_only_prompt_concise.txt"
)
SAVE_DIR_DEFAULT = "/data/group_data/rl/geney/checkpoints/ppo_augmented"
TRAJ_DIR_DEFAULT = "/data/group_data/rl/geney/online_trajectories"

MAP_DIM = 8217
AUX_DIM = OBS_DIM - MAP_DIM  # 51


# ======================================================================
# Online trajectory writer — saves rollout data in offline-compatible format
# ======================================================================
class OnlineTrajectoryWriter:
    """Accumulates per-env episodes and writes .npz shards matching the
    offline pipeline format (obs_map_bits, obs_aux, hidden_state, etc.)."""

    def __init__(self, num_envs: int, output_dir: str,
                 samples_per_file: int = 100_000, gamma: float = 0.99):
        self.num_envs = num_envs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_file = samples_per_file
        self.gamma = gamma

        # Per-env episode buffers (lists of per-step arrays)
        self._obs = [[] for _ in range(num_envs)]       # float32 (8268,)
        self._hidden = [[] for _ in range(num_envs)]     # float32 (4096,)
        self._action = [[] for _ in range(num_envs)]     # int32 scalar
        self._reward = [[] for _ in range(num_envs)]     # float32 scalar
        self._log_prob = [[] for _ in range(num_envs)]   # float32 scalar
        self._gemini_text = [[] for _ in range(num_envs)]  # str
        self._gemini_step_idx = [[] for _ in range(num_envs)]  # int32

        # Current Gemini text per env (carried forward between calls)
        self._current_text = [""] * num_envs

        # Shard accumulator
        self._acc_obs_bits = []
        self._acc_obs_aux = []
        self._acc_hidden = []
        self._acc_action = []
        self._acc_reward = []
        self._acc_done = []
        self._acc_log_prob = []
        self._acc_rtg = []
        self._acc_text = []
        self._acc_gemini_idx = []
        self._acc_samples = 0

        # Count existing shards so we don't overwrite
        existing = list(self.output_dir.glob("trajectories_*.npz"))
        self._shard_idx = len(existing)
        self._total_episodes = 0
        self._total_samples = 0

    def set_gemini_text(self, env_idx: int, text: str):
        """Called when a new Gemini response arrives for an env."""
        self._current_text[env_idx] = text

    def add_step(self, env_idx: int, obs: np.ndarray, hidden: np.ndarray,
                 action: int, log_prob: float, reward: float,
                 step_counter: int):
        """Record one transition for one environment."""
        self._obs[env_idx].append(obs)
        self._hidden[env_idx].append(hidden)
        self._action[env_idx].append(action)
        self._log_prob[env_idx].append(log_prob)
        self._reward[env_idx].append(reward)
        self._gemini_text[env_idx].append(self._current_text[env_idx])
        self._gemini_step_idx[env_idx].append(step_counter % STEP_CADENCE)

    def on_episode_done(self, env_idx: int):
        """Finalize and flush the completed episode for env_idx."""
        n = len(self._obs[env_idx])
        if n == 0:
            return

        # Stack episode arrays
        obs_f32 = np.array(self._obs[env_idx], dtype=np.float32)      # (n, 8268)
        hidden = np.array(self._hidden[env_idx], dtype=np.float32)     # (n, 4096)
        action = np.array(self._action[env_idx], dtype=np.int32)       # (n,)
        reward = np.array(self._reward[env_idx], dtype=np.float32)     # (n,)
        log_prob = np.array(self._log_prob[env_idx], dtype=np.float32) # (n,)
        texts = self._gemini_text[env_idx]                             # list[str]
        gidx = np.array(self._gemini_step_idx[env_idx], dtype=np.int32)

        # Bitpack obs
        obs_f16 = obs_f32.astype(np.float16)
        map_section = np.round(obs_f16[:, :MAP_DIM]).astype(np.uint8)
        obs_map_bits = np.packbits(map_section, axis=1, bitorder="little")
        obs_aux = obs_f16[:, MAP_DIM:]

        # Return-to-go
        rtg = np.zeros(n, dtype=np.float32)
        running = 0.0
        for t in reversed(range(n)):
            running = reward[t] + self.gamma * running
            rtg[t] = running

        # Done array: all False except last step
        done = np.zeros(n, dtype=np.uint8)
        done[-1] = 1

        # Text array (object dtype)
        text_arr = np.empty(n, dtype=object)
        for i in range(n):
            text_arr[i] = texts[i]

        # Add to shard accumulator
        self._acc_obs_bits.append(obs_map_bits)
        self._acc_obs_aux.append(obs_aux)
        self._acc_hidden.append(hidden.astype(np.float16))
        self._acc_action.append(action)
        self._acc_reward.append(reward)
        self._acc_done.append(done)
        self._acc_log_prob.append(log_prob)
        self._acc_rtg.append(rtg)
        self._acc_text.append(text_arr)
        self._acc_gemini_idx.append(gidx)
        self._acc_samples += n
        self._total_episodes += 1

        # Clear per-env buffers
        self._obs[env_idx].clear()
        self._hidden[env_idx].clear()
        self._action[env_idx].clear()
        self._reward[env_idx].clear()
        self._log_prob[env_idx].clear()
        self._gemini_text[env_idx].clear()
        self._gemini_step_idx[env_idx].clear()
        self._current_text[env_idx] = ""

        # Flush shard if large enough
        if self._acc_samples >= self.samples_per_file:
            self.flush_shard()

    def flush_shard(self):
        """Write accumulated episodes to a compressed .npz shard."""
        if self._acc_samples == 0:
            return

        path = self.output_dir / f"trajectories_{self._shard_idx:06d}.npz"
        np.savez_compressed(
            path,
            obs_map_bits=np.concatenate(self._acc_obs_bits),
            obs_map_dim=np.int32(MAP_DIM),
            obs_aux=np.concatenate(self._acc_obs_aux),
            action=np.concatenate(self._acc_action),
            reward=np.concatenate(self._acc_reward),
            done=np.concatenate(self._acc_done),
            log_prob=np.concatenate(self._acc_log_prob),
            return_to_go=np.concatenate(self._acc_rtg),
            hidden_state=np.concatenate(self._acc_hidden),
            text_generated=np.concatenate(self._acc_text),
            gemini_step_idx=np.concatenate(self._acc_gemini_idx),
        )

        self._total_samples += self._acc_samples
        print(f"  Trajectory shard saved: {path.name} "
              f"({self._acc_samples:,} samples, "
              f"{self._total_episodes} episodes total, "
              f"{self._total_samples:,} samples total)")

        # Reset accumulator
        self._shard_idx += 1
        self._acc_obs_bits.clear()
        self._acc_obs_aux.clear()
        self._acc_hidden.clear()
        self._acc_action.clear()
        self._acc_reward.clear()
        self._acc_done.clear()
        self._acc_log_prob.clear()
        self._acc_rtg.clear()
        self._acc_text.clear()
        self._acc_gemini_idx.clear()
        self._acc_samples = 0

    @property
    def stats(self) -> dict:
        pending = self._acc_samples
        return {
            "total_episodes": self._total_episodes,
            "total_samples_saved": self._total_samples,
            "pending_samples": pending,
            "shards_written": self._shard_idx,
        }


# ======================================================================
# Qwen3-8B embedder (layer-30 mean-pool) with batch support
# ======================================================================
class QwenEmbedder:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        print("Loading Qwen3-8B (31 layers, SDPA)...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBED_MODEL, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        config.num_hidden_layers = EMBED_LAYER + 1  # layers 0..30
        self.model = AutoModelForCausalLM.from_pretrained(
            EMBED_MODEL, config=config, dtype=torch.float16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.device = device
        print(f"  Loaded in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed N texts → (N, 4096) float32."""
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", truncation=True,
                max_length=2048, padding=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            out = self.model.model(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            last_hs = out.last_hidden_state  # (B, seq, 4096)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_embeds.append(pooled.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeds, axis=0)


# ======================================================================
# Gemini API
# ======================================================================
def call_gemini(prompt: str, api_key: str, max_retries: int = 3) -> dict:
    from urllib import error as urlerror, request as urlrequest

    url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            "temperature": GEMINI_TEMPERATURE,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            req = urlrequest.Request(url, data=body, headers=headers, method="POST")
            with urlrequest.urlopen(req, timeout=30.0) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)

            candidates = parsed.get("candidates", [])
            text = ""
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(
                    p.get("text", "") for p in parts if isinstance(p, dict)
                )

            usage = parsed.get("usageMetadata", {})
            return {
                "text": text,
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "latency_s": time.perf_counter() - t0,
            }
        except urlerror.HTTPError as e:
            if e.code == 429 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return {
                "text": "", "error": str(e),
                "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0,
            }
        except Exception as e:
            return {
                "text": "", "error": str(e),
                "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0,
            }


def call_gemini_batch(
    prompts: list[str], api_key: str, max_workers: int = 50,
) -> list[dict]:
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(call_gemini, p, api_key): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


# ======================================================================
# Rollout buffer
# ======================================================================
class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, hidden_dim, device):
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.hidden = torch.zeros(num_steps, num_envs, hidden_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

    def compute_gae(self, last_value, gamma, gae_lambda):
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            next_val = last_value if t == self.num_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns


# ======================================================================
# PPO update
# ======================================================================
def ppo_update(
    model, optimizer, buf, advantages, returns,
    clip_eps, ent_coef, vf_coef, max_grad_norm,
    update_epochs, num_minibatches,
):
    batch_size = buf.num_steps * buf.num_envs
    mb_size = batch_size // num_minibatches

    b_obs = buf.obs.reshape(batch_size, -1)
    b_hid = buf.hidden.reshape(batch_size, -1)
    b_act = buf.actions.reshape(batch_size)
    b_logp = buf.log_probs.reshape(batch_size)
    b_val = buf.values.reshape(batch_size)
    b_adv = advantages.reshape(batch_size)
    b_ret = returns.reshape(batch_size)

    metrics = dict(policy_loss=0, value_loss=0, entropy=0, approx_kl=0, clip_frac=0)
    n = 0

    for _ in range(update_epochs):
        idx = torch.randperm(batch_size, device=buf.device)
        for start in range(0, batch_size, mb_size):
            mb = idx[start : start + mb_size]

            pi, val = model(b_obs[mb], b_hid[mb])
            logp = pi.log_prob(b_act[mb])
            ent = pi.entropy().mean()

            ratio = torch.exp(logp - b_logp[mb])
            adv = b_adv[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            pg1 = -adv * ratio
            pg2 = -adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            pi_loss = torch.max(pg1, pg2).mean()

            v_unclip = (val - b_ret[mb]) ** 2
            v_clip = b_val[mb] + torch.clamp(val - b_val[mb], -clip_eps, clip_eps)
            v_clip_loss = (v_clip - b_ret[mb]) ** 2
            vf_loss = 0.5 * torch.max(v_unclip, v_clip_loss).mean()

            loss = pi_loss + vf_coef * vf_loss - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["policy_loss"] += pi_loss.item()
                metrics["value_loss"] += vf_loss.item()
                metrics["entropy"] += ent.item()
                metrics["approx_kl"] += ((ratio - 1) - ratio.log()).mean().item()
                metrics["clip_frac"] += ((ratio - 1).abs() > clip_eps).float().mean().item()
            n += 1

    return {k: v / max(n, 1) for k, v in metrics.items()}


# ======================================================================
# Checkpoint helpers
# ======================================================================
def save_checkpoint(model, optimizer, global_step, save_dir, extra=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"ppo_step_{global_step}.pth")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  Checkpoint saved: {path}")
    return path


# ======================================================================
# Main training loop
# ======================================================================
def train(args):
    device = args.device
    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY env var or pass --gemini-api-key")

    num_envs = args.num_envs
    num_steps = args.num_steps
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // (num_steps * num_envs)
    batch_size = num_steps * num_envs
    est_gemini_calls = total_timesteps // STEP_CADENCE
    est_gemini_cost = est_gemini_calls * 0.0005

    print("=" * 70)
    print("Imagination-Augmented PPO — Online Training")
    print("=" * 70)
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"  Num envs:         {num_envs}")
    print(f"  Num steps:        {num_steps}")
    print(f"  Num updates:      {num_updates:,}")
    print(f"  Batch size:       {batch_size:,}")
    print(f"  LR:               {args.lr} (anneal={args.anneal_lr})")
    print(f"  Device:           {device}")
    print(f"  Est. Gemini calls: {est_gemini_calls:,}")
    print(f"  Est. Gemini cost:  ~${est_gemini_cost:,.0f}")
    print()

    # -- Load prompt template --
    template = PREDICT_TEMPLATE_PATH.read_text()
    print(f"Prompt template: {PREDICT_TEMPLATE_PATH.name}")

    # -- Model --
    model = ActorCriticAug(OBS_DIM, ACTION_DIM, LAYER_WIDTH, EMBED_HIDDEN_DIM).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Loaded AWR checkpoint: {args.init_checkpoint}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # -- Hidden state normalization (from offline training) --
    stats_path = args.hidden_stats or str(AWR_CKPT_DIR / "hidden_state_stats.npz")
    stats = np.load(stats_path)
    hidden_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    hidden_std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    print(f"Hidden stats: {stats_path}")

    # -- Qwen embedder --
    embedder = QwenEmbedder(device=device)

    # -- Craftax environment --
    print(f"Setting up Craftax-Symbolic-v1 with {num_envs} envs...")
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=num_envs)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    obs_jax, env_state = env.reset(reset_key, env_params)
    print(f"  Obs shape: {obs_jax.shape}")

    # -- Rollout buffer --
    buf = RolloutBuffer(num_steps, num_envs, OBS_DIM, EMBED_HIDDEN_DIM, device)

    # -- Per-env imagination state --
    current_hidden = np.zeros((num_envs, EMBED_HIDDEN_DIM), dtype=np.float32)
    step_counters = np.zeros(num_envs, dtype=np.int32)  # steps since last Gemini

    # -- Trajectory writer (saves online rollouts in offline-compatible format) --
    traj_dir = args.trajectory_dir
    traj_writer = OnlineTrajectoryWriter(
        num_envs=num_envs, output_dir=traj_dir, gamma=args.gamma,
    )
    print(f"Trajectory writer: {traj_dir}")

    # -- wandb --
    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        wandb_name = args.wandb_name or f"ppo-augmented-{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_name,
            config=vars(args),
            settings=wandb.Settings(init_timeout=600, start_method="thread"),
        )
        print("wandb initialized")

    # -- Save dir --
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "total_params": total_params,
        "num_updates": num_updates,
        "est_gemini_calls": est_gemini_calls,
    }
    with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ----------------------------------------------------------------
    # Graceful shutdown on SIGTERM / SIGUSR1 (SLURM pre-emption)
    # ----------------------------------------------------------------
    _shutdown_requested = False

    def _request_shutdown(signum, frame):
        nonlocal _shutdown_requested
        _shutdown_requested = True
        sig_name = signal.Signals(signum).name
        print(f"\n*** Received {sig_name} — will save checkpoint after current update ***")

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGUSR1, _request_shutdown)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    global_step = 0
    total_gemini_calls = 0
    total_gemini_cost = 0.0
    t_train_start = time.time()
    last_save_time = t_train_start

    for update in range(1, num_updates + 1):
        t0 = time.time()

        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            for pg in optimizer.param_groups:
                pg["lr"] = frac * args.lr

        # ---- Collect rollout ----
        model.eval()
        gemini_calls_upd = 0
        gemini_cost_upd = 0.0
        gemini_errors_upd = 0
        ep_returns = []
        ep_lengths = []

        for step in range(num_steps):
            # -- Imagination: Gemini + Qwen for envs at cadence --
            needs_gemini = np.where(step_counters % STEP_CADENCE == 0)[0]

            if len(needs_gemini) > 0:
                obs_np_all = np.array(obs_jax, dtype=np.float32)
                prompts = []
                for i in needs_gemini:
                    text_obs = obs_to_text(obs_np_all[i])
                    filtered = filter_text_obs(text_obs)
                    prompts.append(
                        template.replace("{current_state_filtered}", filtered)
                    )

                results = call_gemini_batch(
                    prompts, api_key, max_workers=args.gemini_workers,
                )

                valid_texts, valid_idx = [], []
                for j, (env_idx, res) in enumerate(zip(needs_gemini, results)):
                    if res.get("text"):
                        valid_texts.append(res["text"])
                        valid_idx.append(env_idx)
                        gemini_cost_upd += (
                            res.get("prompt_tokens", 0) * 0.15e-6
                            + res.get("completion_tokens", 0) * 0.60e-6
                        )
                    else:
                        gemini_errors_upd += 1

                gemini_calls_upd += len(needs_gemini)

                if valid_texts:
                    embeddings = embedder.embed_batch(
                        valid_texts, batch_size=args.embed_batch_size,
                    )
                    for j, env_idx in enumerate(valid_idx):
                        current_hidden[env_idx] = embeddings[j]
                        traj_writer.set_gemini_text(env_idx, valid_texts[j])

            # -- Normalize hidden states --
            hidden_t = torch.tensor(
                current_hidden, dtype=torch.float32, device=device,
            )
            hidden_normed = (hidden_t - hidden_mean) / hidden_std

            # -- Policy forward --
            obs_t = torch.tensor(
                np.array(obs_jax, dtype=np.float32), device=device,
            )
            with torch.no_grad():
                pi, value = model(obs_t, hidden_normed)
                action = pi.sample()
                log_prob = pi.log_prob(action)

            # -- Store transition --
            buf.obs[step] = obs_t
            buf.hidden[step] = hidden_normed
            buf.actions[step] = action
            buf.log_probs[step] = log_prob
            buf.values[step] = value
            # (reward and done stored after env step)

            # -- Step environment (save pre-step obs for trajectory writer) --
            obs_jax_pre = obs_jax
            action_jax = jnp.array(action.cpu().numpy(), dtype=jnp.int32)
            rng, step_key = jax.random.split(rng)
            obs_jax, env_state, reward_jax, done_jax, info = env.step(
                step_key, env_state, action_jax, env_params,
            )

            reward_np = np.array(reward_jax, dtype=np.float32)
            done_np = np.array(done_jax, dtype=np.float32)

            buf.rewards[step] = torch.tensor(reward_np, device=device)
            buf.dones[step] = torch.tensor(done_np, device=device)

            # -- Record transitions for trajectory saving --
            obs_np_save = np.array(obs_jax_pre, dtype=np.float32)
            action_np = action.cpu().numpy()
            logp_np = log_prob.cpu().numpy()
            for ei in range(num_envs):
                traj_writer.add_step(
                    env_idx=ei,
                    obs=obs_np_save[ei],
                    hidden=current_hidden[ei],
                    action=int(action_np[ei]),
                    log_prob=float(logp_np[ei]),
                    reward=float(reward_np[ei]),
                    step_counter=int(step_counters[ei]),
                )

            # -- Track completed episodes --
            done_mask = done_np > 0.5
            if done_mask.any():
                ret = np.array(info["returned_episode_returns"])
                length = np.array(info["returned_episode_lengths"])
                ep_returns.extend(ret[done_mask].tolist())
                ep_lengths.extend(length[done_mask].tolist())
                for ei in np.where(done_mask)[0]:
                    traj_writer.on_episode_done(ei)

            # -- Update per-env counters --
            step_counters += 1
            step_counters[done_mask] = 0
            current_hidden[done_mask] = 0.0

            global_step += num_envs

        # ---- Bootstrap value ----
        obs_t = torch.tensor(
            np.array(obs_jax, dtype=np.float32), device=device,
        )
        hidden_t = torch.tensor(current_hidden, dtype=torch.float32, device=device)
        hidden_normed = (hidden_t - hidden_mean) / hidden_std
        with torch.no_grad():
            _, last_value = model(obs_t, hidden_normed)

        # ---- GAE ----
        advantages, returns = buf.compute_gae(
            last_value, args.gamma, args.gae_lambda,
        )

        # ---- PPO update ----
        model.train()
        ppo_metrics = ppo_update(
            model, optimizer, buf, advantages, returns,
            args.clip_eps, args.ent_coef, args.vf_coef,
            args.max_grad_norm, args.update_epochs, args.num_minibatches,
        )

        # ---- Bookkeeping ----
        total_gemini_calls += gemini_calls_upd
        total_gemini_cost += gemini_cost_upd
        update_time = time.time() - t0
        sps = num_steps * num_envs / update_time

        # ---- Logging ----
        if update % args.log_freq == 0 or update == 1:
            elapsed = time.time() - t_train_start
            eta_h = (num_updates - update) * (elapsed / update) / 3600
            ret_str = (
                f"mean={np.mean(ep_returns):.2f} ({len(ep_returns)} eps)"
                if ep_returns else "no completed episodes"
            )
            print(
                f"[{update}/{num_updates}] step={global_step:,}  "
                f"return: {ret_str}  "
                f"pi_loss={ppo_metrics['policy_loss']:.4f}  "
                f"vf_loss={ppo_metrics['value_loss']:.4f}  "
                f"ent={ppo_metrics['entropy']:.3f}  "
                f"gemini={gemini_calls_upd}(err={gemini_errors_upd})  "
                f"SPS={sps:.0f}  ETA={eta_h:.1f}h"
            )

        if use_wandb and (update % args.log_freq == 0 or update == 1):
            log_dict = {
                "train/policy_loss": ppo_metrics["policy_loss"],
                "train/value_loss": ppo_metrics["value_loss"],
                "train/entropy": ppo_metrics["entropy"],
                "train/approx_kl": ppo_metrics["approx_kl"],
                "train/clip_frac": ppo_metrics["clip_frac"],
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/sps": sps,
                "gemini/calls_this_update": gemini_calls_upd,
                "gemini/errors_this_update": gemini_errors_upd,
                "gemini/cost_this_update": gemini_cost_upd,
                "gemini/total_calls": total_gemini_calls,
                "gemini/total_cost_usd": total_gemini_cost,
            }
            if ep_returns:
                log_dict["episode/mean_return"] = np.mean(ep_returns)
                log_dict["episode/median_return"] = np.median(ep_returns)
                log_dict["episode/min_return"] = np.min(ep_returns)
                log_dict["episode/max_return"] = np.max(ep_returns)
                log_dict["episode/num_episodes"] = len(ep_returns)
            if ep_lengths:
                log_dict["episode/mean_length"] = np.mean(ep_lengths)
            ts = traj_writer.stats
            log_dict["trajectories/total_samples"] = ts["total_samples_saved"]
            log_dict["trajectories/total_episodes"] = ts["total_episodes"]
            log_dict["trajectories/shards_written"] = ts["shards_written"]
            wandb.log(log_dict, step=global_step)

        # ---- Checkpoint (update-based or time-based every 2h) ----
        now = time.time()
        time_since_save = now - last_save_time
        if update % args.save_freq == 0 or time_since_save >= 7200:
            save_checkpoint(model, optimizer, global_step, save_dir)
            last_save_time = now

        # ---- Graceful shutdown ----
        if _shutdown_requested:
            print(f"\nShutdown requested at update {update}, saving final checkpoint...")
            save_checkpoint(
                model, optimizer, global_step, save_dir,
                extra={"interrupted": True, "update": update},
            )
            traj_writer.flush_shard()
            traj_stats = traj_writer.stats
            total_time = now - t_train_start
            print(f"Ran {update}/{num_updates} updates in {total_time / 3600:.1f}h")
            print(f"Gemini calls: {total_gemini_calls:,} (${total_gemini_cost:.2f})")
            print(f"Trajectories saved: {traj_stats['total_samples_saved']:,} samples, "
                  f"{traj_stats['total_episodes']} episodes, "
                  f"{traj_stats['shards_written']} shards")
            if use_wandb:
                wandb.summary["total_gemini_calls"] = total_gemini_calls
                wandb.summary["total_gemini_cost_usd"] = total_gemini_cost
                wandb.summary["total_time_h"] = total_time / 3600
                wandb.summary["interrupted"] = True
                wandb.finish()
            print("Exiting cleanly.")
            return

    # ---- Final ----
    save_checkpoint(model, optimizer, global_step, save_dir, extra={"final": True})
    traj_writer.flush_shard()
    traj_stats = traj_writer.stats
    total_time = time.time() - t_train_start
    print(f"\nTraining complete: {total_time / 3600:.1f}h")
    print(f"Total Gemini calls: {total_gemini_calls:,} (${total_gemini_cost:.2f})")
    print(f"Trajectories saved: {traj_stats['total_samples_saved']:,} samples, "
          f"{traj_stats['total_episodes']} episodes, "
          f"{traj_stats['shards_written']} shards")

    if use_wandb:
        wandb.summary["total_gemini_calls"] = total_gemini_calls
        wandb.summary["total_gemini_cost_usd"] = total_gemini_cost
        wandb.summary["total_time_h"] = total_time / 3600
        wandb.finish()


# ======================================================================
# CLI
# ======================================================================
def main():
    p = argparse.ArgumentParser(
        description="Online PPO with imagination augmentation",
    )
    # Environment
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=64)
    p.add_argument("--total-timesteps", type=lambda x: int(float(x)), default=int(1e8))
    p.add_argument("--seed", type=int, default=42)

    # PPO
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.8)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=8)

    # Model / init
    p.add_argument(
        "--init-checkpoint", type=str,
        default=str(AWR_CKPT_DIR / "final.pth"),
        help="AWR checkpoint to warm-start from (empty string to train from scratch)",
    )
    p.add_argument("--hidden-stats", type=str, default=None)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")

    # Imagination
    p.add_argument("--gemini-api-key", type=str, default=None)
    p.add_argument("--gemini-workers", type=int, default=50)
    p.add_argument("--embed-batch-size", type=int, default=32)

    # Logging / saving
    p.add_argument("--save-dir", type=str, default=SAVE_DIR_DEFAULT)
    p.add_argument("--trajectory-dir", type=str, default=TRAJ_DIR_DEFAULT,
                    help="Directory to save online trajectory shards for offline reuse")
    p.add_argument("--save-freq", type=int, default=100,
                    help="Save checkpoint every N updates (~4.5h at SPS=49)")
    p.add_argument("--log-freq", type=int, default=10,
                    help="Log to stdout/wandb every N updates")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="craftax-ppo-augmented")
    p.add_argument("--wandb-entity", type=str, default="iris-sobolmark")

    args = p.parse_args()
    if args.init_checkpoint == "":
        args.init_checkpoint = None
    train(args)


if __name__ == "__main__":
    main()
