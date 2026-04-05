"""
Consolidated augmented actor-critic architectures (PyTorch).

All offline RL training and eval scripts should import from here instead of
defining models inline.  The JAX/Flax equivalents live in actor_critic.py.

Variants
--------
ActorCriticAug       – dual-branch (obs + hidden), tanh, no normalization
ActorCriticAugLN     – dual-branch with LayerNorm + optional Dropout
ActorCritic          – obs-only baseline (no hidden branch)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Weight initialisation (matches JAX orthogonal init)
# ---------------------------------------------------------------------------
def orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


# ---------------------------------------------------------------------------
# Augmented actor-critic (basic – no normalisation)
# ---------------------------------------------------------------------------
class ActorCriticAug(nn.Module):
    """Dual-branch actor-critic conditioned on obs + LLM hidden states.

    Architecture (each of actor/critic):
        obs  → Linear(obs_dim, W) → tanh
        hidden → Linear(hidden_dim, W) → tanh
        concat(2W) → Linear(2W, W) → tanh → Linear(W, W) → tanh → output
    """

    def __init__(
        self,
        obs_dim: int = 8268,
        action_dim: int = 43,
        layer_width: int = 512,
        hidden_state_dim: int = 4096,
    ):
        super().__init__()
        # Actor
        self.actor_obs_fc1 = nn.Linear(obs_dim, layer_width)
        self.actor_hidden_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.actor_fc1 = nn.Linear(2 * layer_width, layer_width)
        self.actor_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_out = nn.Linear(layer_width, action_dim)
        # Critic
        self.critic_obs_fc1 = nn.Linear(obs_dim, layer_width)
        self.critic_hidden_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.critic_fc1 = nn.Linear(2 * layer_width, layer_width)
        self.critic_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_out = nn.Linear(layer_width, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in [
            self.actor_obs_fc1, self.actor_hidden_fc1,
            self.actor_fc1, self.actor_fc2,
            self.critic_obs_fc1, self.critic_hidden_fc1,
            self.critic_fc1, self.critic_fc2,
        ]:
            orthogonal_init(layer, gain=np.sqrt(2))
        orthogonal_init(self.actor_out, gain=0.01)
        orthogonal_init(self.critic_out, gain=1.0)

    def forward(self, obs, hidden_state):
        ao = torch.tanh(self.actor_obs_fc1(obs))
        ah = torch.tanh(self.actor_hidden_fc1(hidden_state))
        ax = torch.tanh(self.actor_fc1(torch.cat([ao, ah], dim=1)))
        ax = torch.tanh(self.actor_fc2(ax))
        pi = Categorical(logits=self.actor_out(ax))

        co = torch.tanh(self.critic_obs_fc1(obs))
        ch = torch.tanh(self.critic_hidden_fc1(hidden_state))
        cx = torch.tanh(self.critic_fc1(torch.cat([co, ch], dim=1)))
        cx = torch.tanh(self.critic_fc2(cx))
        value = self.critic_out(cx).squeeze(-1)

        return pi, value


# ---------------------------------------------------------------------------
# Augmented actor-critic with LayerNorm + Dropout
# ---------------------------------------------------------------------------
class ActorCriticAugLN(nn.Module):
    """Same architecture as ActorCriticAug but with LayerNorm after each
    linear layer and optional dropout.  Used by train_awr_weighted_v2 and
    later experiments.
    """

    def __init__(
        self,
        obs_dim: int = 8268,
        action_dim: int = 43,
        layer_width: int = 512,
        hidden_state_dim: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Actor
        self.actor_obs_fc1 = nn.Linear(obs_dim, layer_width)
        self.actor_hidden_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.actor_fc1 = nn.Linear(2 * layer_width, layer_width)
        self.actor_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_out = nn.Linear(layer_width, action_dim)
        self.actor_ln1 = nn.LayerNorm(layer_width)
        self.actor_ln2 = nn.LayerNorm(layer_width)
        self.actor_ln3 = nn.LayerNorm(layer_width)
        self.actor_ln4 = nn.LayerNorm(layer_width)
        # Critic
        self.critic_obs_fc1 = nn.Linear(obs_dim, layer_width)
        self.critic_hidden_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.critic_fc1 = nn.Linear(2 * layer_width, layer_width)
        self.critic_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_out = nn.Linear(layer_width, 1)
        self.critic_ln1 = nn.LayerNorm(layer_width)
        self.critic_ln2 = nn.LayerNorm(layer_width)
        self.critic_ln3 = nn.LayerNorm(layer_width)
        self.critic_ln4 = nn.LayerNorm(layer_width)

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for layer in [
            self.actor_obs_fc1, self.actor_hidden_fc1,
            self.actor_fc1, self.actor_fc2,
            self.critic_obs_fc1, self.critic_hidden_fc1,
            self.critic_fc1, self.critic_fc2,
        ]:
            orthogonal_init(layer, gain=np.sqrt(2))
        orthogonal_init(self.actor_out, gain=0.01)
        orthogonal_init(self.critic_out, gain=1.0)

    def forward(self, obs, hidden_state):
        ao = self.drop(torch.tanh(self.actor_ln1(self.actor_obs_fc1(obs))))
        ah = self.drop(torch.tanh(self.actor_ln2(self.actor_hidden_fc1(hidden_state))))
        ax = self.drop(torch.tanh(self.actor_ln3(self.actor_fc1(torch.cat([ao, ah], dim=1)))))
        ax = self.drop(torch.tanh(self.actor_ln4(self.actor_fc2(ax))))
        pi = Categorical(logits=self.actor_out(ax))

        co = self.drop(torch.tanh(self.critic_ln1(self.critic_obs_fc1(obs))))
        ch = self.drop(torch.tanh(self.critic_ln2(self.critic_hidden_fc1(hidden_state))))
        cx = self.drop(torch.tanh(self.critic_ln3(self.critic_fc1(torch.cat([co, ch], dim=1)))))
        cx = self.drop(torch.tanh(self.critic_ln4(self.critic_fc2(cx))))
        value = self.critic_out(cx).squeeze(-1)

        return pi, value


# ---------------------------------------------------------------------------
# Unaugmented baseline (obs-only, no hidden branch)
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Observation-only actor-critic (no hidden state / augmentation).

    Architecture (each of actor/critic):
        obs → Linear(obs_dim, W) → tanh → Linear(W, W) → tanh
            → Linear(W, W) → tanh → output
    """

    def __init__(
        self,
        obs_dim: int = 8268,
        action_dim: int = 43,
        layer_width: int = 512,
    ):
        super().__init__()
        self.actor_fc1 = nn.Linear(obs_dim, layer_width)
        self.actor_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_fc3 = nn.Linear(layer_width, layer_width)
        self.actor_out = nn.Linear(layer_width, action_dim)

        self.critic_fc1 = nn.Linear(obs_dim, layer_width)
        self.critic_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_fc3 = nn.Linear(layer_width, layer_width)
        self.critic_out = nn.Linear(layer_width, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in [
            self.actor_fc1, self.actor_fc2, self.actor_fc3,
            self.critic_fc1, self.critic_fc2, self.critic_fc3,
        ]:
            orthogonal_init(layer, gain=np.sqrt(2))
        orthogonal_init(self.actor_out, gain=0.01)
        orthogonal_init(self.critic_out, gain=1.0)

    def forward(self, obs, hidden_state=None):
        ax = torch.tanh(self.actor_fc1(obs))
        ax = torch.tanh(self.actor_fc2(ax))
        ax = torch.tanh(self.actor_fc3(ax))
        pi = Categorical(logits=self.actor_out(ax))

        cx = torch.tanh(self.critic_fc1(obs))
        cx = torch.tanh(self.critic_fc2(cx))
        cx = torch.tanh(self.critic_fc3(cx))
        value = self.critic_out(cx).squeeze(-1)

        return pi, value
