#!/usr/bin/env python3
"""
Online RL with LLM Hidden States - PyTorch Implementation

PPO training with LLM hidden state augmentation via vLLM server.
Uses the same algorithm as online_rl_hidden_jax.py but in PyTorch.

Key features:
- PPO training (not just inference)
- Modular hidden state extraction with skip_n parameter
- WandB logging with training metrics
- Compatible with pre-trained AWR checkpoints

Usage:
    python online_rl_hidden.py --envs 8 --timesteps 1e6 --skip-n 1
    python online_rl_hidden.py --envs 8 --timesteps 1e6 --skip-n 4  # Reuse hidden states
"""

import argparse
import os
import sys

import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb
from craftax.craftax.constants import Achievement


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # LLM
    MODEL_ID: str = "Qwen/Qwen3-4B"
    HIDDEN_SIZE: int = 2560  # Qwen3-4B hidden size

    # Policy
    ACTION_DIM: int = 43
    LAYER_WIDTH: int = 512
    OBS_DIM: int = 8268  # Will be detected from environment

    # PPO Hyperparameters (match ppo.py / online_rl_hidden_jax.py)
    LR: float = 2e-4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.8
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 1.0
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 8
    NUM_STEPS: int = 64  # Steps per PPO update

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # WandB
    WANDB_PROJECT: str = "craftax-online-rl-llm"
    WANDB_ENTITY: str = "iris-sobolmark"


# =============================================================================
# Observation Processing
# =============================================================================

from llm.prompts import filter_text_obs, create_prompt


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


from llm.extractor import VLLMHiddenStateExtractor
import requests


# =============================================================================
# Policy Network with Hidden State Augmentation
# =============================================================================

from models.actor_critic_aug import ActorCriticAug


# =============================================================================
# Rollout Storage for PPO
# =============================================================================

class RolloutBuffer:
    """Storage for PPO rollout data."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, hidden_dim: int, device: str):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        # Pre-allocate tensors
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.hidden_states = torch.zeros((num_steps, num_envs, hidden_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)

        self.step = 0

    def add(self, obs, hidden_state, action, log_prob, value, reward, done):
        self.obs[self.step] = obs
        self.hidden_states[self.step] = hidden_state
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.step += 1

    def reset(self):
        self.step = 0

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + self.values
        return returns, advantages

    def get_minibatches(self, returns, advantages, num_minibatches):
        """Generate shuffled minibatches for PPO update."""
        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // num_minibatches

        # Flatten everything
        obs_flat = self.obs.reshape(batch_size, -1)
        hidden_flat = self.hidden_states.reshape(batch_size, -1)
        actions_flat = self.actions.reshape(batch_size)
        log_probs_flat = self.log_probs.reshape(batch_size)
        values_flat = self.values.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)

        # Shuffle
        indices = torch.randperm(batch_size, device=self.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield (
                obs_flat[mb_indices],
                hidden_flat[mb_indices],
                actions_flat[mb_indices],
                log_probs_flat[mb_indices],
                values_flat[mb_indices],
                returns_flat[mb_indices],
                advantages_flat[mb_indices],
            )


# =============================================================================
# LLM Hidden State Manager
# =============================================================================

class LLMHiddenStateManager:
    """Manages LLM hidden state extraction via vLLM server."""

    def __init__(
        self,
        model_id: str = Config.MODEL_ID,
        target_layer: int = -1,
        tokens_to_generate: int = 1,
    ):
        self.tokens_to_generate = tokens_to_generate

        # Check vLLM server
        vllm_url = "http://localhost:8000"
        try:
            resp = requests.get(f"{vllm_url}/health", timeout=2)
            if resp.status_code != 200:
                raise Exception(f"Server returned status {resp.status_code}")
        except Exception as e:
            print(f"\n❌ ERROR: vLLM server not available at {vllm_url}")
            print(f"   Error: {e}")
            print(f"\n📝 To start the server:")
            print(f"   bash scripts/start_vllm_hidden.sh --mode last_token")
            sys.exit(1)

        print(f"✅ vLLM server connected at {vllm_url}")

        # Initialize vLLM extractor
        model_name = "./configs/vllm_hidden_qwen4b"
        extracted_layers = [8, 16, 24, 35]

        if target_layer == -1:
            layer_index = -1
        elif target_layer in extracted_layers:
            layer_index = extracted_layers.index(target_layer)
        else:
            print(f"⚠️ Warning: Layer {target_layer} not in {extracted_layers}, using last")
            layer_index = -1

        self.llm = VLLMHiddenStateExtractor(
            server_url=vllm_url,
            model_name=model_name,
            model_id=model_id,
            target_layer=layer_index,
        )
        self.hidden_size = self.llm.hidden_size
        print(f"   Hidden size: {self.hidden_size}")

    def extract(self, states, num_envs: int, device: str) -> Tuple[torch.Tensor, Dict]:
        """Extract hidden states for all environments."""
        t_start = time.perf_counter()

        # Render text for all envs
        text_observations = []
        for i in range(num_envs):
            raw_text = render_craftax_text_swapped(states[i])
            # Hard fail on malformed interesting-map coordinates so training does
            # not silently continue with corrupted prompt geometry.
            filtered_text = filter_text_obs(raw_text, strict_map_validation=True)
            text_observations.append(filtered_text)

        t_text = time.perf_counter() - t_start

        # Get hidden states from vLLM
        t_llm_start = time.perf_counter()

        if self.tokens_to_generate == 1:
            hidden_np, llm_metrics = self.llm.extract_hidden_states_no_cot(text_observations)
        else:
            hidden_np, _, llm_metrics = self.llm.extract_hidden_states(
                text_observations,
                batch_size=min(32, len(text_observations)),
                max_new_tokens=self.tokens_to_generate
            )

        t_llm = time.perf_counter() - t_llm_start

        metrics = {
            "timing/text_render_ms": t_text * 1000,
            "timing/llm_inference_ms": t_llm * 1000,
            **llm_metrics
        }

        hidden_tensor = torch.tensor(hidden_np, dtype=torch.float32, device=device)
        return hidden_tensor, metrics


# =============================================================================
# PPO Update
# =============================================================================

def ppo_update(
    policy: ActorCriticAug,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: torch.Tensor,
    config: dict,
) -> Dict[str, float]:
    """Perform PPO update on collected rollout."""

    # Compute returns and advantages
    returns, advantages = buffer.compute_returns_and_advantages(
        last_value, config["gamma"], config["gae_lambda"]
    )

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO update epochs
    total_loss_sum = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    entropy_sum = 0
    num_updates = 0

    for epoch in range(config["update_epochs"]):
        for mb_data in buffer.get_minibatches(returns, advantages, config["num_minibatches"]):
            obs, hidden, actions, old_log_probs, old_values, mb_returns, mb_advantages = mb_data

            # Forward pass
            pi, values = policy(obs, hidden)
            log_probs = pi.log_prob(actions)
            entropy = pi.entropy().mean()

            # Policy loss (clipped)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"]) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            values_clipped = old_values + torch.clamp(
                values - old_values, -config["clip_eps"], config["clip_eps"]
            )
            value_loss1 = (values - mb_returns) ** 2
            value_loss2 = (values_clipped - mb_returns) ** 2
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

            # Total loss
            loss = policy_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy

            # Gradient update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
            optimizer.step()

            total_loss_sum += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy.item()
            num_updates += 1

    return {
        "loss/total": total_loss_sum / num_updates,
        "loss/policy": policy_loss_sum / num_updates,
        "loss/value": value_loss_sum / num_updates,
        "loss/entropy": entropy_sum / num_updates,
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def run_training(
    num_envs: int = 8,
    total_timesteps: int = 1_000_000,
    skip_n: int = 1,
    num_steps: int = 64,
    model_id: str = Config.MODEL_ID,
    checkpoint_path: Optional[str] = None,
    target_layer: int = -1,
    tokens_to_generate: int = 1,
    use_wandb: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    PPO training with LLM hidden states.

    Args:
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        skip_n: Extract LLM hidden states every N env steps
        num_steps: Steps per PPO update
        model_id: LLM model ID
        checkpoint_path: Optional path to load policy checkpoint
        target_layer: Which LLM layer to extract (-1 for last)
        tokens_to_generate: Tokens to generate (1 for prompt-only)
        use_wandb: Enable WandB logging
        seed: Random seed
        verbose: Print progress
    """

    print("=" * 70)
    print("Online RL with LLM Hidden States - PyTorch PPO")
    print("=" * 70)
    print(f"  Environments:     {num_envs}")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"  Skip-N:           {skip_n} (LLM every {skip_n} env steps)")
    print(f"  Steps per update: {num_steps}")
    print(f"  Device:           {Config.DEVICE}")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Import JAX for environment
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    # Initialize environments
    print("\n[1/4] Initializing Craftax environments...")
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params

    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, num_envs)

    states = []
    obs_list = []
    for i, r in enumerate(rngs):
        obs, state = env.reset(r, env_params)
        states.append(state)
        obs_list.append(np.array(obs))

    obs_dim = obs_list[0].shape[0]
    Config.OBS_DIM = obs_dim
    print(f"  Obs dim: {obs_dim}")

    # Initialize LLM manager
    print("\n[2/4] Connecting to vLLM server...")
    llm_manager = LLMHiddenStateManager(
        model_id=model_id,
        target_layer=target_layer,
        tokens_to_generate=tokens_to_generate,
    )

    # Initialize policy
    print("\n[3/4] Initializing policy network...")
    policy = ActorCriticAug(
        obs_dim=obs_dim,
        action_dim=Config.ACTION_DIM,
        layer_width=Config.LAYER_WIDTH,
        hidden_state_dim=llm_manager.hidden_size,
    ).to(Config.DEVICE)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint: {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=Config.LR, eps=1e-5)

    # Rollout buffer
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_dim=obs_dim,
        hidden_dim=llm_manager.hidden_size,
        device=Config.DEVICE,
    )

    # PPO config
    ppo_config = {
        "gamma": Config.GAMMA,
        "gae_lambda": Config.GAE_LAMBDA,
        "clip_eps": Config.CLIP_EPS,
        "ent_coef": Config.ENT_COEF,
        "vf_coef": Config.VF_COEF,
        "max_grad_norm": Config.MAX_GRAD_NORM,
        "update_epochs": Config.UPDATE_EPOCHS,
        "num_minibatches": Config.NUM_MINIBATCHES,
    }

    # Training state
    print("\n[4/4] Starting training...")
    print("=" * 70)

    total_steps = 0
    update_count = 0
    llm_calls = 0
    episode_returns = []
    current_episode_rewards = np.zeros(num_envs)

    # Hidden state tracking
    cached_hidden = torch.zeros((num_envs, llm_manager.hidden_size), device=Config.DEVICE)
    steps_since_llm = skip_n  # Force LLM call on first step

    # Achievement tracking
    achievement_names = [a.name for a in Achievement]
    total_achievements = set()
    prev_achievements = [None] * num_envs

    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_steps = 0

    num_updates_total = total_timesteps // (num_steps * num_envs)

    for update_idx in range(num_updates_total):
        buffer.reset()
        llm_metrics = {}

        # Collect rollout
        for step in range(num_steps):
            # LLM extraction if needed
            if steps_since_llm >= skip_n:
                cached_hidden, llm_metrics = llm_manager.extract(states, num_envs, Config.DEVICE)
                steps_since_llm = 0
                llm_calls += 1

            # Get observations
            obs_batch = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=Config.DEVICE)

            # Get actions
            with torch.no_grad():
                pi, values = policy(obs_batch, cached_hidden)
                actions = pi.sample()
                log_probs = pi.log_prob(actions)

            actions_np = actions.cpu().numpy()

            # Step environments
            new_states = []
            rewards_np = np.zeros(num_envs)
            dones_np = np.zeros(num_envs)

            for i in range(num_envs):
                rng, step_rng = jax.random.split(rngs[i])
                rngs = rngs.at[i].set(rng)

                obs, new_state, reward, done, _ = env.step(
                    step_rng, states[i], int(actions_np[i]), env_params
                )

                obs_list[i] = np.array(obs)
                new_states.append(new_state)
                rewards_np[i] = float(reward)
                dones_np[i] = float(done)

                current_episode_rewards[i] += float(reward)

                # Track achievements
                curr_ach = np.array(new_state.achievements)
                if prev_achievements[i] is not None:
                    new_ach = curr_ach & ~prev_achievements[i]
                    for ach_idx in np.where(new_ach)[0]:
                        total_achievements.add(achievement_names[ach_idx])
                prev_achievements[i] = curr_ach

                if done:
                    episode_returns.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0
                    prev_achievements[i] = None

            states = new_states
            steps_since_llm += 1

            # Store transition
            rewards_tensor = torch.tensor(rewards_np, dtype=torch.float32, device=Config.DEVICE)
            dones_tensor = torch.tensor(dones_np, dtype=torch.float32, device=Config.DEVICE)

            buffer.add(obs_batch, cached_hidden, actions, log_probs, values, rewards_tensor, dones_tensor)

        total_steps += num_steps * num_envs

        # Get last value for GAE
        obs_batch = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=Config.DEVICE)
        with torch.no_grad():
            _, last_value = policy(obs_batch, cached_hidden)

        # PPO update
        loss_metrics = ppo_update(policy, optimizer, buffer, last_value, ppo_config)
        update_count += 1

        # Logging
        current_time = time.perf_counter()
        if (update_idx + 1) % 10 == 0 or update_idx == 0:
            elapsed = current_time - last_log_time
            steps_done = total_steps - last_log_steps
            sps = steps_done / elapsed if elapsed > 0 else 0
            mean_return = np.mean(episode_returns[-100:]) if episode_returns else 0

            if verbose:
                print(f"Update {update_idx+1:4d}/{num_updates_total} | "
                      f"Steps: {total_steps:,} | "
                      f"SPS: {sps:,.0f} | "
                      f"Return: {mean_return:.1f} | "
                      f"LLM: {llm_calls}")

            if use_wandb:
                log_dict = {
                    "timestep": total_steps,
                    "perf/sps": sps,
                    "perf/llm_calls": llm_calls,
                    "train/episode_return": mean_return,
                    "train/episodes": len(episode_returns),
                    "train/updates": update_count,
                    "achievements/unique": len(total_achievements),
                }
                log_dict.update(loss_metrics)
                log_dict.update(llm_metrics)
                wandb.log(log_dict, step=total_steps)

            last_log_time = current_time
            last_log_steps = total_steps

    # Final results
    total_time = time.perf_counter() - start_time
    final_sps = total_steps / total_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total timesteps:  {total_steps:,}")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Average SPS:      {final_sps:,.0f}")
    print(f"  LLM calls:        {llm_calls}")
    print(f"  PPO updates:      {update_count}")
    if episode_returns:
        print(f"  Final return:     {np.mean(episode_returns[-100:]):.1f}")
    print(f"  Achievements:     {len(total_achievements)}")

    return {
        "sps": final_sps,
        "total_time": total_time,
        "total_steps": total_steps,
        "llm_calls": llm_calls,
        "final_return": np.mean(episode_returns[-100:]) if episode_returns else 0,
    }


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Online RL with LLM hidden states (PyTorch PPO)")

    # Core settings
    parser.add_argument("--envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--timesteps", type=lambda x: int(float(x)), default=1e6,
                        help="Total training timesteps")
    parser.add_argument("--skip-n", type=int, default=1,
                        help="LLM hidden state extraction every N env steps")
    parser.add_argument("--num-steps", type=int, default=64,
                        help="Steps per rollout before PPO update")

    # LLM settings
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which layer to extract (-1 for last)")
    parser.add_argument("--tokens", type=int, default=1,
                        help="Tokens to generate (1 for prompt-only)")
    parser.add_argument("--model", type=str, default=Config.MODEL_ID)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to policy checkpoint")

    # Logging
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=Config.WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=Config.WANDB_ENTITY)
    parser.add_argument("--quiet", action="store_true")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    use_wandb = args.use_wandb and not args.no_wandb

    # Initialize WandB
    if use_wandb:
        run_name = f"online-pytorch-{args.envs}env-skip{args.skip_n}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
        print(f"WandB: {args.wandb_project}/{run_name}")

    results = run_training(
        num_envs=args.envs,
        total_timesteps=args.timesteps,
        skip_n=args.skip_n,
        num_steps=args.num_steps,
        model_id=args.model,
        checkpoint_path=args.checkpoint,
        target_layer=args.layer,
        tokens_to_generate=args.tokens,
        use_wandb=use_wandb,
        seed=args.seed,
        verbose=not args.quiet,
    )

    if use_wandb:
        wandb.finish()

    print("\n✅ Done!")
    return results


if __name__ == "__main__":
    main()
