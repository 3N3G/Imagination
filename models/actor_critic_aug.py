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

    def forward(self, obs, hidden_state, obs_detach: bool = False):
        ao = self.drop(torch.tanh(self.actor_ln1(self.actor_obs_fc1(obs))))
        ah = self.drop(torch.tanh(self.actor_ln2(self.actor_hidden_fc1(hidden_state))))
        if obs_detach:
            ao = ao.detach()
        ax = self.drop(torch.tanh(self.actor_ln3(self.actor_fc1(torch.cat([ao, ah], dim=1)))))
        ax = self.drop(torch.tanh(self.actor_ln4(self.actor_fc2(ax))))
        pi = Categorical(logits=self.actor_out(ax))

        co = self.drop(torch.tanh(self.critic_ln1(self.critic_obs_fc1(obs))))
        ch = self.drop(torch.tanh(self.critic_ln2(self.critic_hidden_fc1(hidden_state))))
        if obs_detach:
            co = co.detach()
        cx = self.drop(torch.tanh(self.critic_ln3(self.critic_fc1(torch.cat([co, ch], dim=1)))))
        cx = self.drop(torch.tanh(self.critic_ln4(self.critic_fc2(cx))))
        value = self.critic_out(cx).squeeze(-1)

        return pi, value


# ---------------------------------------------------------------------------
# Augmented actor-critic v2: deep obs branch + late hidden injection + residual
# ---------------------------------------------------------------------------
class ActorCriticAugV2(nn.Module):
    """Augmented actor-critic that matches the unaugmented obs pathway depth.

    Architecture (each of actor/critic):
        obs  → fc1(W) → LN → tanh → fc2(W) → LN → tanh → fc3(W) → LN → tanh
        hidden → h_fc1(W) → LN → tanh → h_fc2(W) → LN → tanh
        combined = obs_out + hidden_out   (additive injection)
        combined → merge(W) → LN → tanh → output

    Key differences from ActorCriticAugLN:
        - Obs branch has 3 layers (matching unaugmented ActorCritic)
        - Hidden injected late via addition (not concatenation)
        - Residual: obs signal passes through even if hidden is garbage
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
        # Actor — obs branch (3 layers, same as ActorCritic)
        self.actor_obs_fc1 = nn.Linear(obs_dim, layer_width)
        self.actor_obs_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_obs_fc3 = nn.Linear(layer_width, layer_width)
        self.actor_obs_ln1 = nn.LayerNorm(layer_width)
        self.actor_obs_ln2 = nn.LayerNorm(layer_width)
        self.actor_obs_ln3 = nn.LayerNorm(layer_width)
        # Actor — hidden branch (2 layers → project to same width)
        self.actor_h_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.actor_h_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_h_ln1 = nn.LayerNorm(layer_width)
        self.actor_h_ln2 = nn.LayerNorm(layer_width)
        # Actor — merge + output
        self.actor_merge = nn.Linear(layer_width, layer_width)
        self.actor_merge_ln = nn.LayerNorm(layer_width)
        self.actor_out = nn.Linear(layer_width, action_dim)

        # Critic — obs branch (3 layers)
        self.critic_obs_fc1 = nn.Linear(obs_dim, layer_width)
        self.critic_obs_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_obs_fc3 = nn.Linear(layer_width, layer_width)
        self.critic_obs_ln1 = nn.LayerNorm(layer_width)
        self.critic_obs_ln2 = nn.LayerNorm(layer_width)
        self.critic_obs_ln3 = nn.LayerNorm(layer_width)
        # Critic — hidden branch (2 layers)
        self.critic_h_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.critic_h_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_h_ln1 = nn.LayerNorm(layer_width)
        self.critic_h_ln2 = nn.LayerNorm(layer_width)
        # Critic — merge + output
        self.critic_merge = nn.Linear(layer_width, layer_width)
        self.critic_merge_ln = nn.LayerNorm(layer_width)
        self.critic_out = nn.Linear(layer_width, 1)

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for layer in [
            self.actor_obs_fc1, self.actor_obs_fc2, self.actor_obs_fc3,
            self.actor_h_fc1, self.actor_h_fc2, self.actor_merge,
            self.critic_obs_fc1, self.critic_obs_fc2, self.critic_obs_fc3,
            self.critic_h_fc1, self.critic_h_fc2, self.critic_merge,
        ]:
            orthogonal_init(layer, gain=np.sqrt(2))
        orthogonal_init(self.actor_out, gain=0.01)
        orthogonal_init(self.critic_out, gain=1.0)
        # Init hidden branch output near zero so model starts from obs-only behavior
        nn.init.zeros_(self.actor_h_fc2.weight)
        nn.init.zeros_(self.actor_h_fc2.bias)
        nn.init.zeros_(self.critic_h_fc2.weight)
        nn.init.zeros_(self.critic_h_fc2.bias)

    def forward(self, obs, hidden_state):
        # Actor — obs branch
        ao = self.drop(torch.tanh(self.actor_obs_ln1(self.actor_obs_fc1(obs))))
        ao = self.drop(torch.tanh(self.actor_obs_ln2(self.actor_obs_fc2(ao))))
        ao = self.drop(torch.tanh(self.actor_obs_ln3(self.actor_obs_fc3(ao))))
        # Actor — hidden branch
        ah = self.drop(torch.tanh(self.actor_h_ln1(self.actor_h_fc1(hidden_state))))
        ah = self.drop(torch.tanh(self.actor_h_ln2(self.actor_h_fc2(ah))))
        # Actor — additive merge + output
        ax = ao + ah  # residual: if hidden is zero/noise, obs signal passes through
        ax = self.drop(torch.tanh(self.actor_merge_ln(self.actor_merge(ax))))
        pi = Categorical(logits=self.actor_out(ax))

        # Critic — obs branch
        co = self.drop(torch.tanh(self.critic_obs_ln1(self.critic_obs_fc1(obs))))
        co = self.drop(torch.tanh(self.critic_obs_ln2(self.critic_obs_fc2(co))))
        co = self.drop(torch.tanh(self.critic_obs_ln3(self.critic_obs_fc3(co))))
        # Critic — hidden branch
        ch = self.drop(torch.tanh(self.critic_h_ln1(self.critic_h_fc1(hidden_state))))
        ch = self.drop(torch.tanh(self.critic_h_ln2(self.critic_h_fc2(ch))))
        # Critic — additive merge + output
        cx = co + ch
        cx = self.drop(torch.tanh(self.critic_merge_ln(self.critic_merge(cx))))
        value = self.critic_out(cx).squeeze(-1)

        return pi, value


# ---------------------------------------------------------------------------
# Augmented actor-critic v3: gated imagination (learned 0/1 gate)
# ---------------------------------------------------------------------------
class ActorCriticAugGated(nn.Module):
    """V2 architecture + a learned scalar gate that decides whether to use imagination.

    Gate: obs → fc(W) → fc(1) → sigmoid → g ∈ [0,1]
    combined = obs_out + g * hidden_out

    The gate is conditioned on obs only (not hidden), so the model learns
    WHEN imagination is useful from the observation alone. If g→0 the model
    falls back to obs-only behavior.
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
        # Actor — obs branch (3 layers)
        self.actor_obs_fc1 = nn.Linear(obs_dim, layer_width)
        self.actor_obs_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_obs_fc3 = nn.Linear(layer_width, layer_width)
        self.actor_obs_ln1 = nn.LayerNorm(layer_width)
        self.actor_obs_ln2 = nn.LayerNorm(layer_width)
        self.actor_obs_ln3 = nn.LayerNorm(layer_width)
        # Actor — hidden branch (2 layers)
        self.actor_h_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.actor_h_fc2 = nn.Linear(layer_width, layer_width)
        self.actor_h_ln1 = nn.LayerNorm(layer_width)
        self.actor_h_ln2 = nn.LayerNorm(layer_width)
        # Actor — gate (obs-conditioned, shares first obs layer representation)
        self.actor_gate_fc = nn.Linear(layer_width, layer_width)
        self.actor_gate_out = nn.Linear(layer_width, 1)
        # Actor — merge + output
        self.actor_merge = nn.Linear(layer_width, layer_width)
        self.actor_merge_ln = nn.LayerNorm(layer_width)
        self.actor_out = nn.Linear(layer_width, action_dim)

        # Critic — obs branch (3 layers)
        self.critic_obs_fc1 = nn.Linear(obs_dim, layer_width)
        self.critic_obs_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_obs_fc3 = nn.Linear(layer_width, layer_width)
        self.critic_obs_ln1 = nn.LayerNorm(layer_width)
        self.critic_obs_ln2 = nn.LayerNorm(layer_width)
        self.critic_obs_ln3 = nn.LayerNorm(layer_width)
        # Critic — hidden branch (2 layers)
        self.critic_h_fc1 = nn.Linear(hidden_state_dim, layer_width)
        self.critic_h_fc2 = nn.Linear(layer_width, layer_width)
        self.critic_h_ln1 = nn.LayerNorm(layer_width)
        self.critic_h_ln2 = nn.LayerNorm(layer_width)
        # Critic — gate
        self.critic_gate_fc = nn.Linear(layer_width, layer_width)
        self.critic_gate_out = nn.Linear(layer_width, 1)
        # Critic — merge + output
        self.critic_merge = nn.Linear(layer_width, layer_width)
        self.critic_merge_ln = nn.LayerNorm(layer_width)
        self.critic_out = nn.Linear(layer_width, 1)

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for layer in [
            self.actor_obs_fc1, self.actor_obs_fc2, self.actor_obs_fc3,
            self.actor_h_fc1, self.actor_h_fc2, self.actor_merge,
            self.critic_obs_fc1, self.critic_obs_fc2, self.critic_obs_fc3,
            self.critic_h_fc1, self.critic_h_fc2, self.critic_merge,
            self.actor_gate_fc, self.critic_gate_fc,
        ]:
            orthogonal_init(layer, gain=np.sqrt(2))
        orthogonal_init(self.actor_out, gain=0.01)
        orthogonal_init(self.critic_out, gain=1.0)
        # Zero-init hidden output so model starts as obs-only
        nn.init.zeros_(self.actor_h_fc2.weight)
        nn.init.zeros_(self.actor_h_fc2.bias)
        nn.init.zeros_(self.critic_h_fc2.weight)
        nn.init.zeros_(self.critic_h_fc2.bias)
        # Init gate bias negative so gate starts near 0 (off)
        nn.init.zeros_(self.actor_gate_out.weight)
        nn.init.constant_(self.actor_gate_out.bias, -2.0)  # sigmoid(-2) ≈ 0.12
        nn.init.zeros_(self.critic_gate_out.weight)
        nn.init.constant_(self.critic_gate_out.bias, -2.0)

    def forward(self, obs, hidden_state):
        # Actor — obs branch
        ao1 = self.drop(torch.tanh(self.actor_obs_ln1(self.actor_obs_fc1(obs))))
        ao = self.drop(torch.tanh(self.actor_obs_ln2(self.actor_obs_fc2(ao1))))
        ao = self.drop(torch.tanh(self.actor_obs_ln3(self.actor_obs_fc3(ao))))
        # Actor — hidden branch
        ah = self.drop(torch.tanh(self.actor_h_ln1(self.actor_h_fc1(hidden_state))))
        ah = self.drop(torch.tanh(self.actor_h_ln2(self.actor_h_fc2(ah))))
        # Actor — gate (conditioned on obs representation after first layer)
        ag = torch.tanh(self.actor_gate_fc(ao1.detach()))  # detach so gate doesn't affect obs branch
        ag = torch.sigmoid(self.actor_gate_out(ag))  # (B, 1)
        # Actor — gated merge + output
        ax = ao + ag * ah
        ax = self.drop(torch.tanh(self.actor_merge_ln(self.actor_merge(ax))))
        pi = Categorical(logits=self.actor_out(ax))

        # Critic — obs branch
        co1 = self.drop(torch.tanh(self.critic_obs_ln1(self.critic_obs_fc1(obs))))
        co = self.drop(torch.tanh(self.critic_obs_ln2(self.critic_obs_fc2(co1))))
        co = self.drop(torch.tanh(self.critic_obs_ln3(self.critic_obs_fc3(co))))
        # Critic — hidden branch
        ch = self.drop(torch.tanh(self.critic_h_ln1(self.critic_h_fc1(hidden_state))))
        ch = self.drop(torch.tanh(self.critic_h_ln2(self.critic_h_fc2(ch))))
        # Critic — gate
        cg = torch.tanh(self.critic_gate_fc(co1.detach()))
        cg = torch.sigmoid(self.critic_gate_out(cg))
        # Critic — gated merge + output
        cx = co + cg * ch
        cx = self.drop(torch.tanh(self.critic_merge_ln(self.critic_merge(cx))))
        value = self.critic_out(cx).squeeze(-1)

        # Store gate values for logging
        self._last_actor_gate = ag.detach()
        self._last_critic_gate = cg.detach()

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
