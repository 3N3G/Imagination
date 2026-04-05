"""
Fine-tune an offline-trained PyTorch model using online PPO (JAX).

This script:
1. Loads a PyTorch AWR model checkpoint
2. Converts the weights to JAX/Flax format  
3. Runs online PPO training with the converted weights as initialization

Usage:
    python online_rl/ppo_finetune_from_offline.py \
        --checkpoint /path/to/awr_checkpoint_100000.pth \
        --env_name Craftax-Pixels-v1 \
        --total_timesteps 1e8
"""
import argparse
import os
import sys
import time


import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logz.batch_logging import batch_log, create_log_dict
from models.actor_critic import ActorCriticConv
from envs.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from models.actor_critic import ActorCriticConv, ActorCriticAug


# ==============================================================================
# Weight Conversion: PyTorch -> JAX
# ==============================================================================

def load_pytorch_checkpoint(checkpoint_path: str) -> dict:
    """Load PyTorch checkpoint and return state dict."""
    print(f"Loading PyTorch checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    return state_dict


def convert_pytorch_to_jax_weights(pytorch_state_dict: dict, jax_params: dict) -> dict:
    """
    Convert PyTorch ActorCriticConv weights to JAX ActorCriticConv format.
    
    PyTorch architecture (from offline_rl/awr.py):
        conv1: Conv2d(3, 32, kernel_size=5, padding=2)
        pool: MaxPool2d(kernel_size=3, stride=3)
        conv2: Conv2d(32, 32, kernel_size=5, padding=2)
        conv3: Conv2d(32, 32, kernel_size=5, padding=2)
        actor_fc1: Linear(512, layer_width)
        actor_fc2: Linear(layer_width, action_dim)
        actor_fc3: Linear(action_dim, action_dim)
        critic_fc1: Linear(512, layer_width)
        critic_fc2: Linear(layer_width, 1)
    
    JAX architecture (from models/actor_critic.py):
        Conv_0: Conv(features=32, kernel_size=(5, 5))
        Conv_1: Conv(features=32, kernel_size=(5, 5))
        Conv_2: Conv(features=32, kernel_size=(5, 5))
        Dense_0: Dense(layer_width)  # actor_fc1
        Dense_1: Dense(action_dim)   # actor_fc2
        Dense_2: Dense(action_dim)   # actor_fc3
        Dense_3: Dense(layer_width)  # critic_fc1
        Dense_4: Dense(1)            # critic_fc2
    
    If ActorCriticAug keys are present (encoder_fc1), maps to ActorCriticAug structure:
        Dense_0..Dense_8 matches JAX ActorCriticAug declaration order.
    """
    import copy
    jax_params = copy.deepcopy(jax_params)
    params = jax_params['params']
    
    def pt_to_jax_conv(pt_weight, pt_bias):
        """Convert PyTorch conv2d weights to JAX format.
        
        PyTorch conv: (out_channels, in_channels, H, W)
        JAX conv: (H, W, in_channels, out_channels)
        """
        weight = pt_weight.numpy()
        weight = np.transpose(weight, (2, 3, 1, 0))  # OIHW -> HWIO
        bias = pt_bias.numpy()
        return {'kernel': weight, 'bias': bias}
    
    def pt_to_jax_linear(pt_weight, pt_bias):
        """Convert PyTorch linear weights to JAX format.
        
        PyTorch linear: (out_features, in_features)
        JAX dense: (in_features, out_features)
        """
        weight = pt_weight.numpy().T  # Transpose
        bias = pt_bias.numpy()
        return {'kernel': weight, 'bias': bias}
    
    # ActorCriticAug Support
    if 'encoder_fc1.weight' in pytorch_state_dict:
        print("Detected ActorCriticAug (LLM-augmented) checkpoint")
        
        # Encoder
        params['Dense_0'] = pt_to_jax_linear(pytorch_state_dict['encoder_fc1.weight'], pytorch_state_dict['encoder_fc1.bias'])
        params['Dense_1'] = pt_to_jax_linear(pytorch_state_dict['encoder_fc2.weight'], pytorch_state_dict['encoder_fc2.bias'])
        params['Dense_2'] = pt_to_jax_linear(pytorch_state_dict['encoder_fc3.weight'], pytorch_state_dict['encoder_fc3.bias'])
        
        # Actor
        params['Dense_3'] = pt_to_jax_linear(pytorch_state_dict['actor_fc1.weight'], pytorch_state_dict['actor_fc1.bias'])
        params['Dense_4'] = pt_to_jax_linear(pytorch_state_dict['actor_fc2.weight'], pytorch_state_dict['actor_fc2.bias'])
        
        # Critic
        params['Dense_5'] = pt_to_jax_linear(pytorch_state_dict['critic_fc1.weight'], pytorch_state_dict['critic_fc1.bias'])
        params['Dense_6'] = pt_to_jax_linear(pytorch_state_dict['critic_fc2.weight'], pytorch_state_dict['critic_fc2.bias'])
        params['Dense_7'] = pt_to_jax_linear(pytorch_state_dict['critic_fc3.weight'], pytorch_state_dict['critic_fc3.bias'])
        params['Dense_8'] = pt_to_jax_linear(pytorch_state_dict['critic_out.weight'], pytorch_state_dict['critic_out.bias'])

        return jax_params

    # ActorCriticConv Support
    if 'conv1.weight' in pytorch_state_dict:
        print("Detected ActorCriticConv (Pixel-based) checkpoint")
        # Convert convolutions
        params['Conv_0'] = pt_to_jax_conv(pytorch_state_dict['conv1.weight'], pytorch_state_dict['conv1.bias'])
        params['Conv_1'] = pt_to_jax_conv(pytorch_state_dict['conv2.weight'], pytorch_state_dict['conv2.bias'])
        params['Conv_2'] = pt_to_jax_conv(pytorch_state_dict['conv3.weight'], pytorch_state_dict['conv3.bias'])
        
        # Convert actor layers
        params['Dense_0'] = pt_to_jax_linear(pytorch_state_dict['actor_fc1.weight'], pytorch_state_dict['actor_fc1.bias'])
        params['Dense_1'] = pt_to_jax_linear(pytorch_state_dict['actor_fc2.weight'], pytorch_state_dict['actor_fc2.bias'])
        params['Dense_2'] = pt_to_jax_linear(pytorch_state_dict['actor_fc3.weight'], pytorch_state_dict['actor_fc3.bias'])
        
        # Convert critic layers
        params['Dense_3'] = pt_to_jax_linear(pytorch_state_dict['critic_fc1.weight'], pytorch_state_dict['critic_fc1.bias'])
        params['Dense_4'] = pt_to_jax_linear(pytorch_state_dict['critic_fc2.weight'], pytorch_state_dict['critic_fc2.bias'])
        
        return jax_params
    
    print("Successfully converted PyTorch weights to JAX format")
    return jax_params


# ==============================================================================
# PPO Training (adapted from online_rl/ppo.py)
# ==============================================================================

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config, pretrained_params=None):
    """Create training function, optionally initializing with pretrained params."""
    
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
        network = ActorCriticConv(
            action_dim=env.action_space(env_params).n,
            layer_width=config["LAYER_SIZE"],
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))
        init_params = network.init(_rng, init_x)
        
        # If pretrained params provided, use them instead of random init
        if pretrained_params is not None:
            print("Using pretrained parameters for initialization")
            network_params = pretrained_params
        else:
            network_params = init_params

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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, update_i, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state = (train_state, env_state, last_obs, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state

            # CALCULATE ADVANTAGE
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
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"]
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"] * config["NUM_STEPS"])
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((config["NUM_STEPS"] * config["NUM_ENVS"],) + x.shape[2:]),
                    batch,
                )
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # LOGGING
            metric = traj_batch.info
            runner_state = (train_state, env_state, last_obs, update_i + 1, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def main():
    parser = argparse.ArgumentParser(description="Fine-tune offline model with online PPO")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch checkpoint (.pth file)")
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e8)
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
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--anneal_lr", type=bool, default=True)
    parser.add_argument("--use_optimistic_resets", type=bool, default=True)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="craftax-finetune")
    parser.add_argument("--save_policy", action="store_true")
    args = parser.parse_args()

    config = {
        "ENV_NAME": args.env_name,
        "NUM_ENVS": args.num_envs,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "LR": args.lr,
        "NUM_STEPS": args.num_steps,
        "UPDATE_EPOCHS": args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": args.vf_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "LAYER_SIZE": args.layer_size,
        "ANNEAL_LR": args.anneal_lr,
        "USE_OPTIMISTIC_RESETS": args.use_optimistic_resets,
        "OPTIMISTIC_RESET_RATIO": args.optimistic_reset_ratio,
        "SEED": args.seed,
        "USE_WANDB": args.use_wandb,
        "SAVE_POLICY": args.save_policy,
        "CHECKPOINT_PATH": args.checkpoint,
    }

    # Initialize WandB
    if config["USE_WANDB"]:
        wandb.init(
            project=args.wandb_project,
            config=config,
            name=f"finetune_{os.path.basename(args.checkpoint)}",
        )

    print("=" * 60)
    print("FINE-TUNING OFFLINE MODEL WITH ONLINE PPO")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Environment: {args.env_name}")
    print(f"Total timesteps: {args.total_timesteps:.0e}")
    print("=" * 60)

    # Load PyTorch checkpoint
    pytorch_state_dict = load_pytorch_checkpoint(args.checkpoint)

    # Create JAX model to get parameter structure
    rng = jax.random.PRNGKey(args.seed)
    env = make_craftax_env_from_name(args.env_name, auto_reset=True)
    env_params = env.default_params
    
    # Determine Architecture based on checkpoint content
    # For initialization purposes
    if 'encoder_fc1.weight' in pytorch_state_dict:
        print("Using ActorCriticAug architecture")
        network = ActorCriticAug(
            action_dim=env.action_space(env_params).n,
            layer_width=args.layer_size,
            hidden_state_dim=2560 # Default Qwen embedding size, ideally passed in args
        )
        dummy_hidden = jnp.zeros((1, 2560))
        # ActorCriticAug requires (obs, hidden)
        # We need to reshape dummy_obs to simple 1D if symbolic? 
        # Env returns (1345,) symbolic or (64,64,3) pixel?
        # Craftax-Symbolic-v1 returns 1345 symbolic.
        # But make_craftax_env returns vectorized env?
        # dummy_obs is (1, *shape).
    else:
        print("Using ActorCriticConv architecture")
        network = ActorCriticConv(
            action_dim=env.action_space(env_params).n,
            layer_width=args.layer_size,
        )
    
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, *env.observation_space(env_params).shape))
    
    if isinstance(network, ActorCriticAug):
        dummy_hidden = jnp.zeros((1, 2560))
        init_params = network.init(init_rng, dummy_obs, dummy_hidden)
    else:
        init_params = network.init(init_rng, dummy_obs)

    # Convert weights
    pretrained_params = convert_pytorch_to_jax_weights(pytorch_state_dict, init_params)

    # Run training with pretrained params
    rng, train_rng = jax.random.split(rng)
    
    train_fn = make_train(config, pretrained_params)
    train_jit = jax.jit(train_fn)
    
    t0 = time.time()
    out = train_jit(train_rng)
    t1 = time.time()
    
    print(f"\nTraining complete!")
    print(f"Time: {t1 - t0:.1f}s")
    print(f"SPS: {config['TOTAL_TIMESTEPS'] / (t1 - t0):.0f}")

    # Save final policy
    if config["SAVE_POLICY"] and config["USE_WANDB"]:
        train_state = out["runner_state"][0]
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        path = os.path.join(wandb.run.dir, "policies")
        checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
        save_args = orbax_utils.save_args_from_target(train_state)
        checkpoint_manager.save(
            config["TOTAL_TIMESTEPS"],
            train_state,
            save_kwargs={"save_args": save_args},
        )
        print(f"Saved policy to {path}")

    if config["USE_WANDB"]:
        wandb.finish()


if __name__ == "__main__":
    main()
