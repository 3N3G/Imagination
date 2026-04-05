import argparse
import os
import sys
import time
from tqdm import tqdm
from typing import Tuple


import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from craftax.craftax_env import make_craftax_env_from_name
from envs.wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper

from models.actor_critic import ActorCriticConvImAug

from PIL import Image
from gemma import gm


class PPOConfig:
    def __init__(self, **kwargs):
        self.env_name = kwargs.get("env_name", "Craftax-Pixels-v1")
        self.num_envs = kwargs.get("num_envs", 32)
        self.total_timesteps = int(kwargs.get("total_timesteps", 1e6))
        self.num_steps = kwargs.get("num_steps", 64)
        self.update_epochs = kwargs.get("update_epochs", 4)
        self.num_minibatches = kwargs.get("num_minibatches", 8)
        self.lr = kwargs.get("lr", 2e-4)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_eps = kwargs.get("clip_eps", 0.2)
        self.ent_coef = kwargs.get("ent_coef", 0.01)
        self.vf_coef = kwargs.get("vf_coef", 0.5)
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.layer_size = kwargs.get("layer_size", 512)
        self.seed = kwargs.get("seed", 0)
        self.prompt = kwargs.get(
            "prompt",
            "Analyze this scene in a 2D Minecraft game called Craftax. Consider the actions the player can take, and things the player should consider.",
        )
        self.use_wandb = kwargs.get("use_wandb", False)
        self.wandb_project = kwargs.get("wandb_project", None)
        self.wandb_entity = kwargs.get("wandb_entity", None)
        self.log_interval = kwargs.get("log_interval", 10)


def setup_gemma(prompt: str):
    print("Loading model")
    model = gm.nn.Gemma3_4B()
    checkpoint_dir = gm.ckpts.CheckpointPath.GEMMA3_4B_IT
    params = gm.ckpts.load_params(checkpoint_dir)

    tokenizer = gm.text.Gemma3Tokenizer()

    def embed_images(pil_images: list) -> np.ndarray:
        """Return embeddings for a list of images. Couldn't figure out actual embeddings so just using logits.

        Output shape: (batch, vocab_size), dtype: np.float32
        """
        assert len(pil_images) > 0
        print(f"Embedding {len(pil_images)} images")
        text = (
            f"<start_of_turn>user\n{prompt}\n\n<start_of_image>\n<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        # Convert PIL images to numpy arrays and stack them
        image_arrays = [np.asarray(img, dtype=np.uint8) for img in pil_images]
        images_batch = np.stack(image_arrays, axis=0)  # (batch, h, w, c)
        batch_size = images_batch.shape[0]

        # Model takes in (batch, num_images, h, w, c)
        images = jnp.asarray(images_batch, dtype=jnp.uint8)
        images = jnp.expand_dims(images, axis=1)  # (batch, 1, h, w, c)

        # Encode prompt and broadcast to batch size
        prompt_tokens = jnp.asarray(tokenizer.encode(text, add_bos=True))  # (seq_len,)
        prompt_tokens = jnp.expand_dims(prompt_tokens, axis=0)  # (1, seq_len)
        prompt_tokens = jnp.broadcast_to(
            prompt_tokens, (batch_size, prompt_tokens.shape[1])
        )  # (batch, seq_len)

        # Apply model to entire batch at once
        out = model.apply(
            {"params": params},
            prompt_tokens,
            images=images,
        )

        # Reduce logits (batch, seq_len, vocab_size) -> (batch, vocab_size)
        z = out.logits[:, -1, :]
        return np.asarray(z, dtype=np.float32)

    return embed_images


def obs_to_pil_batch(obs_uint8_bhwc: np.ndarray) -> list:
    """Convert a batch of uint8 images (N,H,W,C) to a list of PIL.Image in RGB format."""
    # Ensure RGB format (H,W,3) for each image
    imgs = []
    for i in range(obs_uint8_bhwc.shape[0]):
        img = obs_uint8_bhwc[i]
        # If grayscale or single channel, convert to RGB
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        # If RGBA, convert to RGB
        elif img.shape[-1] == 4:
            img = img[..., :3]
        # Ensure it's RGB (3 channels)
        assert img.shape[-1] == 3, f"Expected 3 channels (RGB), got {img.shape[-1]}"
        imgs.append(Image.fromarray(img))  # PIL auto-detects RGB from (H,W,3) shape
    return imgs


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Compute Generalized Advantage Estimation.

    rewards, values, dones: shape (T, N)
    last_value: shape (N,)
    Returns advantages, returns with shape (T, N)
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros((N,), dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_loss_fn(
    params, apply_fn, obs, z, actions, old_log_probs, advantages, returns, config
):
    pi, value = apply_fn(params, obs, z)
    log_probs = pi.log_prob(actions)

    # Policy loss
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = jnp.exp(log_probs - old_log_probs)
    loss_actor1 = ratio * advantages
    loss_actor2 = (
        jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages
    )
    policy_loss = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))

    # Value loss (clip around old value estimate)
    value_pred = value
    value_pred_clipped = returns + (value_pred - returns).clip(
        -config.clip_eps, config.clip_eps
    )
    value_losses = jnp.square(value_pred - returns)
    value_losses_clipped = jnp.square(value_pred_clipped - returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

    entropy = jnp.mean(pi.entropy())
    total_loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
    return total_loss, (policy_loss, value_loss, entropy)


def train(config: PPOConfig):
    assert (
        "Symbolic" not in config.env_name
    ), "This script only supports pixel environments."

    print("Loading environment")
    # Environment
    env = make_craftax_env_from_name(config.env_name, False)
    env_params = env.default_params
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env = BatchEnvWrapper(env, num_envs=config.num_envs)

    # Seed
    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)

    print("Loading Gemma 3 4B")
    # Gemma embedder (Python/NumPy/PIL path)
    embed_images = setup_gemma(config.prompt)

    # Derive Gemma embedding dim by running once on a dummy frame
    obs_np_uint8 = np.asarray(jnp.clip(obs * 255.0, 0, 255), dtype=np.uint8)
    z_dummy = embed_images(obs_to_pil_batch(obs_np_uint8))  # shape (N, D)
    z_dim = z_dummy.shape[-1]

    print("Loading network")
    # Policy network and optimizer
    network = ActorCriticConvImAug(env.action_space(env_params).n, config.layer_size)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros_like(obs, dtype=jnp.float32)
    dummy_z = jnp.zeros((config.num_envs, z_dim), dtype=jnp.float32)
    params = network.init(init_rng, dummy_obs, dummy_z)

    print("Loading optimizer")
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr, eps=1e-5),
    )
    state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    num_updates = config.total_timesteps // (config.num_steps * config.num_envs)
    batch_size = config.num_steps * config.num_envs
    minibatch_size = batch_size // config.num_minibatches

    # Optional Weights & Biases
    if config.use_wandb:
        try:
            import wandb

            wandb.init(
                project=config.wandb_project or "craftax",
                entity=config.wandb_entity,
                config={k: v for k, v in vars(config).items() if not callable(v)},
                name=f"{config.env_name}-pixel",
            )
        except Exception as e:
            print(f"wandb init failed: {e}")
            config.use_wandb = False

    print("Training!")
    for update in range(int(num_updates)):
        t_update_start = time.time()
        # Storage
        obs_buf = []
        z_buf = []
        actions_buf = []
        logprobs_buf = []
        values_buf = []
        rewards_buf = []
        dones_buf = []

        for t in tqdm(range(config.num_steps), desc="steps", dynamic_ncols=True):
            # Compute z for current observations (expects uint8 images)
            _t0 = time.time()
            obs_uint8 = np.asarray(jnp.clip(obs * 255.0, 0, 255), dtype=np.uint8)
            pil_batch = obs_to_pil_batch(obs_uint8)
            _t1 = time.time()
            z_np = embed_images(pil_batch)  # (N, D)
            _t2 = time.time()
            z_t = jnp.asarray(z_np, dtype=jnp.float32)

            # Policy forward
            pi, value = state.apply_fn(state.params, obs, z_t)
            rng, act_rng = jax.random.split(rng)
            action = pi.sample(seed=act_rng)
            log_prob = pi.log_prob(action)
            _t3 = time.time()

            # Step env
            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, reward, done, info = env.step(
                step_rng, env_state, action, env_params
            )
            _t4 = time.time()

            # Minimal profiling output for the first few steps of the first update
            if update == 0 and t < 10:
                print(
                    f"profiling step {t}: to_pil={_t1 - _t0:.2f}s, gemma={_t2 - _t1:.2f}s, policy+sample={_t3 - _t2:.2f}s, env.step={_t4 - _t3:.2f}s, total={_t4 - _t0:.2f}s"
                )

            # Store
            obs_buf.append(obs)
            z_buf.append(z_t)
            actions_buf.append(action)
            logprobs_buf.append(log_prob)
            values_buf.append(value)
            rewards_buf.append(reward)
            dones_buf.append(done)

            obs = next_obs

        # Stack buffers to (T, N, ...)
        obs_traj = jnp.stack(obs_buf)  # (T, N, H, W, C)
        z_traj = jnp.stack(z_buf)  # (T, N, D)
        actions_traj = jnp.stack(actions_buf)  # (T, N)
        logprobs_traj = jnp.stack(logprobs_buf)  # (T, N)
        values_traj = jnp.stack(values_buf)  # (T, N)
        rewards_traj = jnp.stack(rewards_buf)  # (T, N)
        dones_traj = jnp.stack(dones_buf)  # (T, N)

        # Bootstrap value
        _tb0 = time.time()
        _obs_uint8_boot = np.asarray(jnp.clip(obs * 255.0, 0, 255), dtype=np.uint8)
        _pil_boot = obs_to_pil_batch(_obs_uint8_boot)
        _tb1 = time.time()
        # Embed in micro-batches to reduce peak memory (no caching)
        _tb2 = time.time()
        _chunks = [embed_images([im]) for im in _pil_boot]
        _z_boot_np = np.concatenate(_chunks, axis=0)
        _tb3 = time.time()
        _z_boot = jnp.asarray(_z_boot_np, dtype=jnp.float32)
        _, last_value = state.apply_fn(state.params, obs, _z_boot)
        _tb4 = time.time()
        if update == 0:
            print(
                f"profiling bootstrap: to_pil={_tb1 - _tb0:.2f}s, gemma={_tb3 - _tb2:.2f}s, value_apply={_tb4 - _tb3:.2f}s, total={_tb4 - _tb0:.2f}s"
            )

        # Move to NumPy for simple GAE implementation
        advantages_np, returns_np = compute_gae(
            np.asarray(rewards_traj),
            np.asarray(values_traj),
            np.asarray(dones_traj),
            np.asarray(last_value),
            config.gamma,
            config.gae_lambda,
        )

        # Flatten to (batch, ...)
        def flatten_time_env(x):
            x = np.asarray(x)
            T, N = x.shape[0], x.shape[1]
            return x.reshape(T * N, *x.shape[2:])

        obs_flat = flatten_time_env(obs_traj)
        z_flat = flatten_time_env(z_traj)
        actions_flat = flatten_time_env(actions_traj)
        logprobs_flat = flatten_time_env(logprobs_traj)
        adv_flat = flatten_time_env(advantages_np)
        ret_flat = flatten_time_env(returns_np)

        # Training epochs
        idxs = np.arange(batch_size)
        policy_losses = []
        value_losses = []
        entropies = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = idxs[start:end]

                mb_obs = jnp.asarray(obs_flat[mb_idx], dtype=jnp.float32)
                mb_z = jnp.asarray(z_flat[mb_idx], dtype=jnp.float32)
                mb_actions = jnp.asarray(actions_flat[mb_idx])
                mb_old_logp = jnp.asarray(logprobs_flat[mb_idx], dtype=jnp.float32)
                mb_adv = jnp.asarray(adv_flat[mb_idx], dtype=jnp.float32)
                mb_ret = jnp.asarray(ret_flat[mb_idx], dtype=jnp.float32)

                (loss, aux), grads = jax.value_and_grad(ppo_loss_fn, has_aux=True)(
                    state.params,
                    state.apply_fn,
                    mb_obs,
                    mb_z,
                    mb_actions,
                    mb_old_logp,
                    mb_adv,
                    mb_ret,
                    config,
                )
                state = state.apply_gradients(grads=grads)
                # Accumulate metrics
                policy_losses.append(float(aux[0]))
                value_losses.append(float(aux[1]))
                entropies.append(float(aux[2]))

        # Per-update metrics
        ep_return = (
            jnp.where(dones_traj, rewards_traj, jnp.zeros_like(rewards_traj))
            .sum(axis=0)
            .mean()
        )
        t_update_end = time.time()
        sps = batch_size / max(1e-6, (t_update_end - t_update_start))

        if ((update + 1) % config.log_interval) == 0:
            mean_pol = float(np.mean(policy_losses)) if policy_losses else 0.0
            mean_val = float(np.mean(value_losses)) if value_losses else 0.0
            mean_ent = float(np.mean(entropies)) if entropies else 0.0
            total_seen = int((update + 1) * batch_size)
            print(
                f"Update {update+1}/{int(num_updates)} | steps {total_seen} | SPS {sps:.1f} | "
                f"ret {float(ep_return):.2f} | pi {mean_pol:.3f} | vf {mean_val:.3f} | ent {mean_ent:.3f}"
            )
            if config.use_wandb:
                try:
                    import wandb

                    wandb.log(
                        {
                            "update": update + 1,
                            "timesteps": total_seen,
                            "sps": sps,
                            "return/avg": float(ep_return),
                            "loss/policy": mean_pol,
                            "loss/value": mean_val,
                            "entropy": mean_ent,
                        }
                    )
                except Exception as e:
                    print(f"wandb log failed: {e}")

        # Explicitly drop large per-update device arrays before next update
        try:
            del (
                obs_traj,
                z_traj,
                actions_traj,
                logprobs_traj,
                values_traj,
                rewards_traj,
                dones_traj,
            )
        except NameError:
            pass
        try:
            del obs_flat, z_flat, actions_flat, logprobs_flat, adv_flat, ret_flat
        except NameError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Analyze this scene in a 2d minecraft game called craftax. Consider the actions the player can take, and things the player should consider.",
    )
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=1)

    args, rest_args = parser.parse_known_args()
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    config = PPOConfig(**vars(args))
    t0 = time.time()
    train(config)
    t1 = time.time()
    print("Time to run:", t1 - t0)


if __name__ == "__main__":
    main()
