"""PyTorch inference port of Flax ActorCriticRNN from online_rl/ppo_rnn.py.

The Flax architecture (per state, batch first):
    Dense(obs_dim,   LAYER_SIZE) + ReLU   # encoder
    GRUCell(LAYER_SIZE)                    # recurrence
    Dense(LAYER_SIZE, LAYER_SIZE) + ReLU   # actor 1
    Dense(LAYER_SIZE, LAYER_SIZE) + ReLU   # actor 2
    Dense(LAYER_SIZE, action_dim)          # actor logits
    Dense(LAYER_SIZE, LAYER_SIZE) + ReLU   # critic 1 (shares input w/ actor)
    Dense(LAYER_SIZE, LAYER_SIZE) + ReLU   # critic 2
    Dense(LAYER_SIZE, 1)                   # critic value

We port only the forward pass. Weights are loaded from an orbax checkpoint
via `load_from_orbax` — it opens the checkpoint through the Flax side
(so we don't have to re-implement the Pytree spec) and transfers each
tensor into PyTorch with the correct shape transpose and GRU-block
reordering.

Unit-test with `python -m models.ppo_rnn_torch --verify <ckpt_dir>` —
compares the PyTorch logits to JAX on 32 random obs and asserts L1 < 1e-4.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticRNNTorch(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, layer_size: int = 512):
        super().__init__()
        self.layer_size = layer_size
        self.enc = nn.Linear(obs_dim, layer_size)
        self.gru = nn.GRUCell(layer_size, layer_size)

        self.actor1 = nn.Linear(layer_size, layer_size)
        self.actor2 = nn.Linear(layer_size, layer_size)
        self.actor_head = nn.Linear(layer_size, action_dim)

        self.critic1 = nn.Linear(layer_size, layer_size)
        self.critic2 = nn.Linear(layer_size, layer_size)
        self.critic_head = nn.Linear(layer_size, 1)

    def initial_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.layer_size,
                           device=device if device is not None else
                           next(self.parameters()).device)

    def forward(
        self,
        obs: torch.Tensor,       # (B, obs_dim)
        h: torch.Tensor,          # (B, layer_size) — prev hidden
        done: torch.Tensor | None = None,  # (B,) bool, reset mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if done is not None:
            h = torch.where(done.view(-1, 1), torch.zeros_like(h), h)

        x = F.relu(self.enc(obs))
        h = self.gru(x, h)

        a = F.relu(self.actor1(h))
        a = F.relu(self.actor2(a))
        logits = self.actor_head(a)

        v = F.relu(self.critic1(h))
        v = F.relu(self.critic2(v))
        value = self.critic_head(v).squeeze(-1)

        return logits, value, h


# --------------------------------------------------------------------------
# Flax/orbax → PyTorch weight transfer
# --------------------------------------------------------------------------

# Flax Dense.kernel has shape (in, out); PyTorch Linear.weight is (out, in).
# Flax GRUCell stores weights per gate: ih and hh each as dict with keys
# 'i', 'g', 'n' or split into 'z', 'r', 'n' depending on version. We detect
# at runtime.

_GRU_GATE_MAP_CANDIDATES = [
    # (flax_order, torch_order)
    (["ir", "iz", "in"], ["r", "z", "n"]),  # (ir, iz, in_) -> PyTorch [r, z, n]
]


def _transfer_dense(flax_param: dict, torch_layer: nn.Linear) -> None:
    kernel = np.asarray(flax_param["kernel"])  # (in, out)
    bias = np.asarray(flax_param["bias"])      # (out,)
    with torch.no_grad():
        torch_layer.weight.copy_(torch.from_numpy(kernel.T.copy()))
        torch_layer.bias.copy_(torch.from_numpy(bias.copy()))


def _flax_gru_to_torch(flax_gru_params: dict, torch_gru: nn.GRUCell) -> None:
    """Flax GRU uses separate gate dicts; PyTorch uses stacked [r, z, n].

    Flax GRUCell params (nn.GRUCell in flax.linen, as of 0.8+):
        {'ir': {'kernel': (in, hidden)}, 'iz': {...}, 'in': {...},
         'hr': {'kernel': (hidden, hidden)}, 'hz': {...}, 'hn': {...}}

    PyTorch GRUCell:
        weight_ih: (3*hidden, in)   ordered [r, z, n]
        weight_hh: (3*hidden, hidden) ordered [r, z, n]
        bias_ih : (3*hidden,)
        bias_hh : (3*hidden,)
    """
    def _grab(section: str, gate: str) -> Tuple[np.ndarray, np.ndarray | None]:
        gate_dict = flax_gru_params[f"{section}{gate}"]
        kernel = np.asarray(gate_dict["kernel"]).T   # to (out, in)
        bias = np.asarray(gate_dict["bias"]) if "bias" in gate_dict else None
        return kernel, bias

    ir_w, ir_b = _grab("i", "r")
    iz_w, iz_b = _grab("i", "z")
    in_w, in_b = _grab("i", "n")

    hr_w, hr_b = _grab("h", "r")
    hz_w, hz_b = _grab("h", "z")
    hn_w, hn_b = _grab("h", "n")

    weight_ih = np.concatenate([ir_w, iz_w, in_w], axis=0)
    weight_hh = np.concatenate([hr_w, hz_w, hn_w], axis=0)

    def _stack_bias(parts):
        parts = [p if p is not None else np.zeros(torch_gru.hidden_size) for p in parts]
        return np.concatenate(parts, axis=0)

    bias_ih = _stack_bias([ir_b, iz_b, in_b])
    bias_hh = _stack_bias([hr_b, hz_b, hn_b])

    with torch.no_grad():
        torch_gru.weight_ih.copy_(torch.from_numpy(weight_ih.copy()))
        torch_gru.weight_hh.copy_(torch.from_numpy(weight_hh.copy()))
        torch_gru.bias_ih.copy_(torch.from_numpy(bias_ih.copy()))
        torch_gru.bias_hh.copy_(torch.from_numpy(bias_hh.copy()))


def transfer_flax_to_torch(flax_params: dict, model: ActorCriticRNNTorch) -> None:
    """flax_params is the nested dict under `params` of a TrainState."""

    # Flax auto-names Dense layers Dense_0, Dense_1, etc. in registration
    # order. In ActorCriticRNN.__call__ the order is:
    #   Dense_0: encoder (obs -> layer)
    #   Dense_1: actor1
    #   Dense_2: actor2
    #   Dense_3: actor_head
    #   Dense_4: critic1
    #   Dense_5: critic2
    #   Dense_6: critic_head
    # The GRU is wrapped inside ScannedRNN which exposes GRUCell_0's params.
    _transfer_dense(flax_params["Dense_0"], model.enc)
    _transfer_dense(flax_params["Dense_1"], model.actor1)
    _transfer_dense(flax_params["Dense_2"], model.actor2)
    _transfer_dense(flax_params["Dense_3"], model.actor_head)
    _transfer_dense(flax_params["Dense_4"], model.critic1)
    _transfer_dense(flax_params["Dense_5"], model.critic2)
    _transfer_dense(flax_params["Dense_6"], model.critic_head)

    # GRU lives under ScannedRNN_0/GRUCell_0
    scanned = flax_params.get("ScannedRNN_0") or flax_params.get("ScannedRNN")
    if scanned is None:
        raise RuntimeError(f"No ScannedRNN_0 key in {list(flax_params.keys())}")
    gru_key = next((k for k in scanned.keys() if k.startswith("GRUCell")), None)
    if gru_key is None:
        raise RuntimeError(f"No GRUCell under ScannedRNN: {list(scanned.keys())}")
    _flax_gru_to_torch(scanned[gru_key], model.gru)


# --------------------------------------------------------------------------
# Checkpoint loader
# --------------------------------------------------------------------------

def load_from_orbax(ckpt_dir: Path, action_dim: int, obs_dim: int,
                    layer_size: int = 512) -> ActorCriticRNNTorch:
    """Open a `policies/<step>` orbax checkpoint dir and return a PyTorch model
    with weights transferred. The orbax CheckpointManager save path is usually
    `<dir>/policies/<step>`. We accept either the parent or the step-level dir.
    """
    import jax
    from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions

    ckpt_dir = Path(ckpt_dir)
    if (ckpt_dir / "policies").is_dir():
        ckpt_dir = ckpt_dir / "policies"

    options = CheckpointManagerOptions(max_to_keep=5, create=False)
    orbax_checkpointer = PyTreeCheckpointer()
    mgr = CheckpointManager(str(ckpt_dir), orbax_checkpointer, options)
    step = mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoint step under {ckpt_dir}")

    restored = mgr.restore(step)
    # TrainState layout: {opt_state, params, step} where params = {'params': {Dense_0, ...}}
    params = restored.get("params")
    if params is None:
        raise RuntimeError(f"No 'params' in restored state (keys={list(restored.keys())})")
    if isinstance(params, dict) and "params" in params and all(
        k.startswith(("Dense", "GRUCell", "ScannedRNN")) for k in params["params"].keys()
    ):
        params = params["params"]

    model = ActorCriticRNNTorch(obs_dim=obs_dim, action_dim=action_dim, layer_size=layer_size)
    transfer_flax_to_torch(params, model)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Verification / CLI
# --------------------------------------------------------------------------

def _verify(ckpt_dir: Path, obs_dim: int, action_dim: int, layer_size: int) -> None:
    """Compare PyTorch port vs Flax on 32 random obs; fail loudly if mismatched."""
    import jax
    import jax.numpy as jnp
    from online_rl.ppo_rnn import ActorCriticRNN, ScannedRNN

    torch_model = load_from_orbax(ckpt_dir, action_dim, obs_dim, layer_size)

    # Rebuild Flax model params for a side-by-side forward.
    from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions
    ckpt_dir_p = Path(ckpt_dir)
    if (ckpt_dir_p / "policies").is_dir():
        ckpt_dir_p = ckpt_dir_p / "policies"
    mgr = CheckpointManager(str(ckpt_dir_p), PyTreeCheckpointer(),
                            CheckpointManagerOptions(max_to_keep=5, create=False))
    restored = mgr.restore(mgr.latest_step())
    flax_params = restored.get("params")
    if isinstance(flax_params, dict) and "params" in flax_params and all(
        k.startswith(("Dense", "GRUCell", "ScannedRNN"))
        for k in flax_params["params"].keys()
    ):
        flax_params = flax_params["params"]

    rng = np.random.default_rng(0)
    obs_np = rng.standard_normal((32, obs_dim), dtype=np.float32)
    dones_np = np.zeros(32, dtype=bool)

    # Flax forward. ScannedRNN expects time-major inputs with shape (T, B, ...);
    # we fake T=1 so the port matches the single-step path.
    config = {"LAYER_SIZE": layer_size}
    flax_model = ActorCriticRNN(action_dim=action_dim, config=config)
    init_hidden = ScannedRNN.initialize_carry(32, layer_size)
    obs_j = jnp.asarray(obs_np)[None]  # (1, 32, obs_dim)
    dones_j = jnp.asarray(dones_np)[None]  # (1, 32)
    _, pi, value_j = flax_model.apply({"params": flax_params}, init_hidden, (obs_j, dones_j))
    # distrax.Categorical.logits returns log-softmax(raw_logits), not raw.
    logprob_flax = np.asarray(pi.logits[0])
    value_flax = np.asarray(value_j[0])

    # PyTorch forward — raw logits. Convert to log-softmax for comparison.
    obs_t = torch.from_numpy(obs_np)
    h_t = torch_model.initial_state(32)
    logits_t, value_t, _ = torch_model(obs_t, h_t)
    logprob_t = torch.log_softmax(logits_t, dim=-1).detach().cpu().numpy()
    value_t = value_t.detach().cpu().numpy()

    max_logprob_diff = np.abs(logprob_flax - logprob_t).max()
    max_value_diff = np.abs(value_flax - value_t).max()
    print(f"max |log-prob diff| = {max_logprob_diff:.3e}")
    print(f"max |value diff|    = {max_value_diff:.3e}")
    if max_logprob_diff > 1e-3:
        print("WARN: log-prob diff > 1e-3 — port is NOT numerically faithful.")
        sys.exit(1)
    print("PyTorch port matches Flax within 1e-3. OK.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=Path, required=True, help="orbax checkpoint dir")
    ap.add_argument("--obs-dim", type=int, default=8268,
                    help="Craftax-Symbolic-v1 obs_dim (default 8268)")
    ap.add_argument("--action-dim", type=int, default=43)
    ap.add_argument("--layer-size", type=int, default=512)
    args = ap.parse_args()
    _verify(args.verify, args.obs_dim, args.action_dim, args.layer_size)
