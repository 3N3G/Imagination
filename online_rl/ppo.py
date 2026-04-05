import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_AGENT, Achievement

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax import serialization
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logz.batch_logging import batch_log, create_log_dict
from models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
)
from models.icm import ICMEncoder, ICMForward, ICMInverse
from envs.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)


from pathlib import Path
import pickle
from craftax.craftax.renderer import render_craftax_text, render_craftax_pixels
import imageio

# ---------------------------------------------------------------------------
# Floor-change logging (enabled via --floor_logging)
# ---------------------------------------------------------------------------
_max_floor_reached = [0]
_floors_logged = set()


def _floor_change_callback(player_levels, achievements, obs_flat, update_step, num_envs):
    """Print text observation when a new floor is first reached.

    Called via jax.experimental.io_callback — runs outside JAX tracing.
    """
    from labelling.obs_to_text import obs_to_text as _obs_to_text

    player_levels = np.array(player_levels)
    achievements = np.array(achievements)
    obs_flat = np.array(obs_flat)

    floor_names = {
        0: "Overworld", 1: "Gnomish Mines", 2: "Dungeon",
        3: "Sewers", 4: "Vault", 5: "Troll Mines",
        6: "Fire Realm", 7: "Ice Realm", 8: "Graveyard (Boss)",
    }

    for env_idx in range(num_envs):
        current_floor = int(player_levels[env_idx])
        if current_floor > _max_floor_reached[0]:
            _max_floor_reached[0] = current_floor
            print("\n" + "=" * 80)
            print(f"NEW FLOOR REACHED: {current_floor} ({floor_names.get(current_floor, 'Unknown')})")
            print(f"Update step: {update_step}, Env: {env_idx}")
            print("=" * 80)
            try:
                print(_obs_to_text(obs_flat[env_idx]))
            except Exception as e:
                print(f"Error converting observation to text: {e}")
            env_achievements = achievements[env_idx]
            achieved = [a.name for a in Achievement if env_achievements[a.value] > 0]
            print(f"\nAchievements unlocked ({len(achieved)}): {achieved}")
            print("=" * 80 + "\n")
            sys.stdout.flush()


def save_trajectory_batch(batch_data, batch_idx, save_path, env_states=None, env=None):
    """Save a batch of trajectories to disk using numpy compressed format
    
    Args:
        batch_data: Dict with obs, next_obs, action, reward, done, log_prob, and optionally text_obs
        batch_idx: Batch number for filename
        save_path: Directory to save to
        env_states: Optional list of EnvState objects for text rendering (legacy, not used)
        env: Optional environment for reconstruction (legacy, not used)
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    filepath = save_path / f"trajectories_batch_{batch_idx:06d}.npz"
    
    # Save all keys from batch_data (including text_obs if present)
    np.savez_compressed(filepath, **batch_data)

    # Report what was saved
    keys_saved = list(batch_data.keys())
    print(
        f"Saved batch {batch_idx} ({len(batch_data['obs'])} transitions) to {filepath}"
    )
    print(f"  Keys: {keys_saved}")


def save_policy_checkpoint_msgpack(
    params,
    timestep: int,
    save_dir: str,
    metadata: dict | None = None,
) -> tuple[str, str]:
    save_path = Path(save_dir).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    params_host = jax.device_get(params)
    ckpt_path = save_path / f"ppo_symbolic_step_{int(timestep):012d}.msgpack"
    with ckpt_path.open("wb") as f:
        f.write(serialization.to_bytes(params_host))

    meta_payload = {
        "timestep": int(timestep),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_path),
    }
    if metadata:
        meta_payload.update(metadata)

    meta_path = ckpt_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, sort_keys=True)

    latest_path = save_path / "latest_policy.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, sort_keys=True)

    print(f"Saved policy checkpoint: {ckpt_path}")
    return str(ckpt_path), str(meta_path)


# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
        else:
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
        init_policy_path = str(config.get("INIT_POLICY_PATH", "")).strip()
        if init_policy_path:
            init_path = Path(init_policy_path).expanduser()
            if not init_path.exists():
                raise FileNotFoundError(f"init policy checkpoint not found: {init_path}")
            with init_path.open("rb") as f:
                init_blob = f.read()
            network_params = serialization.from_bytes(network_params, init_blob)
            print(f"Loaded initial policy params from {init_path}")
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Exploration state
        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
        }

        if config["TRAIN_ICM"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Encoder
            icm_encoder_network = ICMEncoder(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_encoder_network_params = icm_encoder_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_encoder"] = TrainState.create(
                apply_fn=icm_encoder_network.apply,
                params=icm_encoder_network_params,
                tx=tx,
            )

            # Forward
            icm_forward_network = ICMForward(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
                num_actions=env.num_actions,
            )
            rng, _rng = jax.random.split(rng)
            icm_forward_network_params = icm_forward_network.init(
                _rng, jnp.zeros((1, config["ICM_LATENT_SIZE"])), jnp.zeros((1,))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_forward"] = TrainState.create(
                apply_fn=icm_forward_network.apply,
                params=icm_forward_network_params,
                tx=tx,
            )

            # Inverse
            icm_inverse_network = ICMInverse(
                num_layers=3,
                output_dim=env.num_actions,
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_inverse_network_params = icm_inverse_network.init(
                _rng,
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_inverse"] = TrainState.create(
                apply_fn=icm_inverse_network.apply,
                params=icm_inverse_network_params,
                tx=tx,
            )

            if config["USE_E3B"]:
                ex_state["e3b_matrix"] = (
                    jnp.repeat(
                        jnp.expand_dims(
                            jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                        ),
                        config["NUM_ENVS"],
                        axis=0,
                    )
                    / config["E3B_LAMBDA"]
                )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                reward_i = jnp.zeros(config["NUM_ENVS"])

                if config["TRAIN_ICM"]:
                    latent_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, last_obs
                    )
                    latent_next_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, obsv
                    )

                    latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                        ex_state["icm_forward"].params, latent_obs, action
                    )
                    error = (latent_next_obs - latent_next_obs_pred) * (
                        1 - done[:, None]
                    )
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["ICM_REWARD_COEFF"]

                    if config["USE_E3B"]:
                        # Embedding is (NUM_ENVS, 128)
                        # e3b_matrix is (NUM_ENVS, 128, 128)
                        us = jax.vmap(jnp.matmul)(ex_state["e3b_matrix"], latent_obs)
                        bs = jax.vmap(jnp.dot)(latent_obs, us)

                        def update_c(c, b, u):
                            return c - (1.0 / (1 + b)) * jnp.outer(u, u)

                        updated_cs = jax.vmap(update_c)(ex_state["e3b_matrix"], bs, us)
                        new_cs = (
                            jnp.repeat(
                                jnp.expand_dims(
                                    jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                                ),
                                config["NUM_ENVS"],
                                axis=0,
                            )
                            / config["E3B_LAMBDA"]
                        )
                        ex_state["e3b_matrix"] = jnp.where(
                            done[:, None, None], new_cs, updated_cs
                        )

                        e3b_bonus = jnp.where(
                            done, jnp.zeros((config["NUM_ENVS"],)), bs
                        )

                        reward_i = e3b_bonus * config["E3B_REWARD_COEFF"]

                reward = reward_e + reward_i

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
            ) = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            # Add achievement tracking (same as online_rl_hidden.py)
            # Get achievements from all environments over all steps
            # env_state shape: LogEnvState with env_state field
            # We need to track achievements across the trajectory
            # Since we're in JAX, we track this numerically rather than with sets

            # Get final env_state achievements
            final_achievements = env_state.env_state.achievements  # Shape: (NUM_ENVS, num_achievements)

            # Count total unique achievements (any env that has unlocked each achievement)
            any_unlocked = jnp.any(final_achievements, axis=0)  # Shape: (num_achievements,)
            total_unique = jnp.sum(any_unlocked)

            # Count total unlocks (sum of all achievements across all envs)
            total_unlocks = jnp.sum(final_achievements)

            metric["achievements/total_unique"] = total_unique
            metric["achievements/total_unlocks"] = total_unlocks

            # Floor-change logging (optional)
            if config.get("FLOOR_LOGGING", False):
                jax.experimental.io_callback(
                    _floor_change_callback,
                    None,
                    env_state.env_state.player_level,
                    env_state.env_state.achievements,
                    last_obs,
                    update_step,
                    config["NUM_ENVS"],
                )

            # Add timestep tracking to match online_rl_hidden.py
            # Each update processes NUM_STEPS * NUM_ENVS environment steps
            metric["timestep"] = (
                int(config.get("TIMESTEP_OFFSET", 0))
                + (update_step + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    def _inverse_loss_fn(
                        icm_encoder_params, icm_inverse_params, traj_batch
                    ):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.next_obs
                        )

                        action_pred_logits = ex_state["icm_inverse"].apply_fn(
                            icm_inverse_params, latent_obs, latent_next_obs
                        )
                        true_action = jax.nn.one_hot(
                            traj_batch.action, num_classes=action_pred_logits.shape[-1]
                        )

                        bce = -jnp.mean(
                            jnp.sum(
                                action_pred_logits
                                * true_action
                                * (1 - traj_batch.done[:, None]),
                                axis=1,
                            )
                        )

                        return bce * config["ICM_INVERSE_LOSS_COEF"]

                    inverse_grad_fn = jax.value_and_grad(
                        _inverse_loss_fn,
                        has_aux=False,
                        argnums=(
                            0,
                            1,
                        ),
                    )
                    inverse_loss, grads = inverse_grad_fn(
                        ex_state["icm_encoder"].params,
                        ex_state["icm_inverse"].params,
                        traj_batch,
                    )
                    icm_encoder_grad, icm_inverse_grad = grads
                    ex_state["icm_encoder"] = ex_state["icm_encoder"].apply_gradients(
                        grads=icm_encoder_grad
                    )
                    ex_state["icm_inverse"] = ex_state["icm_inverse"].apply_gradients(
                        grads=icm_inverse_grad
                    )

                    def _forward_loss_fn(icm_forward_params, traj_batch):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.next_obs
                        )

                        latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                            icm_forward_params, latent_obs, traj_batch.action
                        )

                        error = (latent_next_obs - latent_next_obs_pred) * (
                            1 - traj_batch.done[:, None]
                        )
                        return (
                            jnp.square(error).mean() * config["ICM_FORWARD_LOSS_COEF"]
                        )

                    forward_grad_fn = jax.value_and_grad(
                        _forward_loss_fn, has_aux=False
                    )
                    forward_loss, icm_forward_grad = forward_grad_fn(
                        ex_state["icm_forward"].params, traj_batch
                    )
                    ex_state["icm_forward"] = ex_state["icm_forward"].apply_gradients(
                        grads=icm_forward_grad
                    )

                    losses = (inverse_loss, forward_loss)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["TRAIN_ICM"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["icm_inverse_loss"] = ex_loss[0].mean()
                metric["icm_forward_loss"] = ex_loss[1].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            if config["SAVE_TRAJ"]:

                def save_callback(traj_batch, update_step):
                    if update_step % config["SAVE_TRAJ_EVERY"] == 0:
                        # Convert to numpy and reshape
                        # traj_batch shape is (NUM_STEPS, NUM_ENVS, ...)
                        # Reshape to (NUM_STEPS * NUM_ENVS, ...)
                        batch_data = {
                            "obs": np.array(traj_batch.obs).reshape(
                                -1, *traj_batch.obs.shape[2:]
                            ),
                            "next_obs": np.array(traj_batch.next_obs).reshape(
                                -1, *traj_batch.next_obs.shape[2:]
                            ),
                            "action": np.array(traj_batch.action).reshape(-1),
                            "reward": np.array(traj_batch.reward).reshape(
                                -1
                            ),  # I'm assuming no intrisic reward, so reward_e = reward
                            "done": np.array(traj_batch.done).reshape(-1),
                            "log_prob": np.array(traj_batch.log_prob).reshape(-1),
                        }

                        save_trajectory_batch(
                            batch_data, update_step, config["TRAJ_SAVE_PATH"]
                        )

                jax.experimental.io_callback(save_callback, None, traj_batch, update_step)

            if config["SAVE_VIDEO"]:

                def video_callback(params, update_step, rng_key):
                    if update_step % config["SAVE_VIDEO_EVERY"] == 0:
                        save_path = Path(config["VIDEO_SAVE_PATH"])
                        save_path.mkdir(parents=True, exist_ok=True)
                        
                        video_filename = save_path / f"video_update_{update_step:06d}.mp4"
                        
                        # Initialize a rollout env
                        test_env = make_craftax_env_from_name(config["ENV_NAME"], True)
                        test_env_params = test_env.default_params
                        
                        # Re-initialize network for rollout
                        if "Symbolic" in config["ENV_NAME"]:
                            test_network = ActorCritic(test_env.action_space(test_env_params).n, config["LAYER_SIZE"])
                        else:
                            test_network = ActorCriticConv(test_env.action_space(test_env_params).n, config["LAYER_SIZE"])
                        
                        # Define rollout
                        def run_rollout(rng):
                            obs, state = test_env.reset(rng, test_env_params)
                            frames = []
                            done = False
                            step_count = 0
                            
                            while not done and step_count < config["MAX_VIDEO_STEPS"]:
                                # Render frame (already returns 0-255 values)
                                frame = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_AGENT, do_night_noise=False)
                                frame_uint8 = np.array(frame, dtype=np.uint8)
                                frames.append(frame_uint8)
                                
                                # Act
                                rng, _rng = jax.random.split(rng)
                                pi, _ = test_network.apply(params, obs[None]) # add batch dim
                                action = pi.sample(seed=_rng)[0] # remove batch dim
                                
                                obs, state, reward, done, info = test_env.step(_rng, state, action, test_env_params)
                                step_count += 1
                                
                            return frames

                        frames = run_rollout(rng_key)
                        
                        if frames:
                            # Resize frames for better visibility using PIL
                            from PIL import Image
                            scale = 4
                            resized_frames = []
                            for frame in frames:
                                img = Image.fromarray(frame)
                                new_size = (img.width * scale, img.height * scale)
                                resized = img.resize(new_size, Image.NEAREST)
                                resized_frames.append(np.array(resized))
                            
                            # Save as GIF locally (imageio supports this natively)
                            gif_filename = save_path / f"video_update_{update_step:06d}.gif"
                            imageio.mimsave(gif_filename, resized_frames, duration=1000/15)  # 15fps
                            print(f"Saved video to {gif_filename}")
                            
                            if config["USE_WANDB"]:
                                # For wandb, pass the numpy array directly (T, C, H, W)
                                video_array = np.array(resized_frames)  # (T, H, W, C)
                                video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
                                wandb.log({
                                    "video": wandb.Video(video_array, fps=15, format="mp4"),
                                    "video_update_step": int(update_step)
                                })



                jax.experimental.io_callback(video_callback, None, train_state.params, update_step, rng)

            if config.get("SAVE_POLICY_EVERY_STEPS", 0) > 0 and config.get("POLICY_SAVE_DIR"):
                save_every = int(config["SAVE_POLICY_EVERY_STEPS"])
                step_stride = int(config["NUM_STEPS"] * config["NUM_ENVS"])
                timestep_offset = int(config.get("TIMESTEP_OFFSET", 0))

                prev_timestep = timestep_offset + update_step * step_stride
                curr_timestep = prev_timestep + step_stride
                crossed_save_boundary = (curr_timestep // save_every) > (prev_timestep // save_every)

                def policy_callback(params, timestep):
                    save_policy_checkpoint_msgpack(
                        params=params,
                        timestep=int(timestep),
                        save_dir=config["POLICY_SAVE_DIR"],
                        metadata=config.get("POLICY_METADATA", {}),
                    )

                def _save_policy(payload):
                    params, timestep = payload
                    jax.experimental.io_callback(policy_callback, None, params, timestep)
                    return None

                jax.lax.cond(
                    crossed_save_boundary,
                    _save_policy,
                    lambda _: None,
                    (train_state.params, curr_timestep),
                )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}  # , "info": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}
    config["RUN_START_UTC"] = datetime.now(timezone.utc).isoformat()
    config["POLICY_METADATA"] = {
        "env_name": config["ENV_NAME"],
        "num_envs": int(config["NUM_ENVS"]),
        "num_steps": int(config["NUM_STEPS"]),
        "total_timesteps": int(config["TOTAL_TIMESTEPS"]),
        "seed": int(config["SEED"]),
    }
    if config.get("INIT_POLICY_PATH"):
        config["POLICY_METADATA"]["init_policy_path"] = str(config["INIT_POLICY_PATH"])
    config["POLICY_METADATA"]["timestep_offset"] = int(config.get("TIMESTEP_OFFSET", 0))

    final_timestep = int(config.get("TIMESTEP_OFFSET", 0)) + int(config["TOTAL_TIMESTEPS"])

    if config.get("SAVE_POLICY_EVERY_STEPS", 0) < 0:
        raise ValueError("--save_policy_every_steps must be >= 0")

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M"
            + ("-floor-logging" if config.get("FLOOR_LOGGING", False) else ""),
        )

    rng = jax.random.PRNGKey(config["SEED"])

    if config["USE_WANDB"]:
        # Create env to get first frame
        env = make_craftax_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
        )
        env_params = env.default_params
        test_rng = jax.random.PRNGKey(0)
        obsv, _ = env.reset(test_rng, env_params)
        first_frame = np.array(obsv)  # Now it's concrete, not a tracer
        # Only log as image if it's actually an image (not symbolic 1D vector)
        if len(first_frame.shape) >= 2:
            wandb.log(
                {
                    "first_frame": wandb.Image(first_frame),
                    "frame_shape": str(first_frame.shape),
                }
            )
        else:
            wandb.log({"frame_shape": str(first_frame.shape)})

        if config.get("FLOOR_LOGGING", False):
            env_fl = make_craftax_env_from_name(
                config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
            )
            _, fl_state = env_fl.reset(jax.random.PRNGKey(0), env_fl.default_params)
            print("\n" + "=" * 80)
            print("FLOOR 0 (Overworld) - Initial State")
            print("=" * 80)
            print(render_craftax_text(fl_state))
            print("=" * 80 + "\n")
            sys.stdout.flush()

    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config.get("FLOOR_LOGGING", False):
        print(f"\n{'=' * 80}")
        print(f"Training Complete! Maximum floor reached: {_max_floor_reached[0]}")
        print(f"{'=' * 80}\n")

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                final_timestep,
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")

    if config.get("SAVE_POLICY_FINAL", True) and config.get("POLICY_SAVE_DIR"):
        train_states = out["runner_state"][0]
        train_state = jax.tree.map(lambda x: x[0], train_states)
        save_policy_checkpoint_msgpack(
            params=train_state.params,
            timestep=final_timestep,
            save_dir=config["POLICY_SAVE_DIR"],
            metadata={
                **config.get("POLICY_METADATA", {}),
                "final_checkpoint": True,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e9
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument(
        "--policy_save_dir",
        type=str,
        default="",
        help="Directory for msgpack policy checkpoints (periodic + final).",
    )
    parser.add_argument(
        "--save_policy_every_steps",
        type=int,
        default=0,
        help="Save msgpack policy checkpoint every N env steps (0 disables periodic saves).",
    )
    parser.add_argument(
        "--save_policy_final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save final msgpack policy checkpoint at total_timesteps.",
    )
    parser.add_argument(
        "--init_policy_path",
        type=str,
        default="",
        help="Optional msgpack policy checkpoint path to initialize training from.",
    )
    parser.add_argument(
        "--timestep_offset",
        type=int,
        default=0,
        help="Absolute timestep offset used for logging/checkpoint naming when continuing runs.",
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument(
        "--floor_logging",
        action="store_true",
        help="Print text observations when new floors are reached (Craftax-Symbolic).",
    )
    parser.add_argument("--save_traj", action="store_true")
    parser.add_argument(
        "--traj_save_path",
        type=str,
        default="/data/group_data/rl/geney/craftax_unlabelled_symbolic_with_text/",
    )
    parser.add_argument("--save_traj_every", type=int, default=1)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_video_every", type=int, default=100)
    parser.add_argument("--max_video_steps", type=int, default=1000)
    parser.add_argument("--video_save_path", type=str, default="videos")
    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ICM
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
