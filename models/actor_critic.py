import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax


class ActorCriticConvSymbolicCraftax(nn.Module):
    action_dim: Sequence[int]
    map_obs_shape: Sequence[int]
    layer_width: int

    @nn.compact
    def __call__(self, obs):
        # Split into map and flat obs
        flat_map_obs_shape = (
            self.map_obs_shape[0] * self.map_obs_shape[1] * self.map_obs_shape[2]
        )
        image_obs = obs[:, :flat_map_obs_shape]
        image_dim = self.map_obs_shape
        image_obs = image_obs.reshape((image_obs.shape[0], *image_dim))

        flat_obs = obs[:, flat_map_obs_shape:]

        # Convolutions on map
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_obs)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_embedding)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)
        # image_embedding = jnp.concatenate([image_embedding, obs[:, : CraftaxEnv.get_flat_map_obs_shape()]], axis=-1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)
        embedding = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        embedding = x.reshape(x.shape[0], -1)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticWithEmbedding(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_emb)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_emb


class ActorCriticConvImAug(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, z):
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        embedding = x.reshape(x.shape[0], -1)

        combined_embedding = jnp.concatenate([embedding, z], axis=-1)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(combined_embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticAug(nn.Module):
    """Single augmented architecture used for both online and offline runs.

    Actor and critic do not share a backbone:
    - actor: obs->512, hidden->512, concat, 512, 512, action logits
    - critic: obs->512, hidden->512, concat, 512, 512, value
    """
    action_dim: Sequence[int]
    layer_width: int = 512
    hidden_state_dim: int = 2560
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, hidden_state):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_obs = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_obs_fc1",
        )(obs)
        actor_obs = activation(actor_obs)

        actor_hidden = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_hidden_fc1",
        )(hidden_state)
        actor_hidden = activation(actor_hidden)

        actor_combined = jnp.concatenate([actor_obs, actor_hidden], axis=-1)
        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_fc1",
        )(actor_combined)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_fc2",
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic_obs = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_obs_fc1",
        )(obs)
        critic_obs = activation(critic_obs)

        critic_hidden = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_hidden_fc1",
        )(hidden_state)
        critic_hidden = activation(critic_hidden)

        critic_combined = jnp.concatenate([critic_obs, critic_hidden], axis=-1)
        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_fc1",
        )(critic_combined)
        critic = activation(critic)
        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_fc2",
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)
