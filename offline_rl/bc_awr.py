"""Minimal joint BC+AWR offline training scaffold for imagination-augmented policies.

Each optimizer step samples from:
- a small golden dataset for strong behavioral cloning supervision
- a larger dataset for AWR-style policy/value learning

Both losses are applied to the same policy at the same time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.actor_critic_aug import ActorCriticAug

Batch = Dict[str, torch.Tensor]
LogFn = Callable[[Mapping[str, float]], None]


class BatchSource(Protocol):
    """Minimal adapter for offline batch providers."""

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        """Return a batch dictionary for one training update."""


class InMemoryBatchSource:
    """Simple in-memory source for bring-up and smoke tests."""

    def __init__(self, data: Mapping[str, torch.Tensor]):
        if "obs" not in data or "action" not in data:
            raise ValueError("InMemoryBatchSource requires `obs` and `action`.")

        self.data = {key: torch.as_tensor(value) for key, value in data.items()}
        first_key = next(iter(self.data))
        self.size = int(self.data[first_key].shape[0])
        if self.size <= 0:
            raise ValueError("InMemoryBatchSource data is empty.")

        for key, value in self.data.items():
            if value.shape[0] != self.size:
                raise ValueError(
                    f"All fields must share batch dimension 0. `{key}` has "
                    f"{value.shape[0]} vs expected {self.size}."
                )

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        idx = torch.randint(0, self.size, (batch_size,))
        return {
            key: value.index_select(0, idx).to(device=device)
            for key, value in self.data.items()
        }


SimpleImaginationAugPolicy = ActorCriticAug


@dataclass
class GoldenBCConfig:
    batch_size: int = 64
    loss_weight: float = 1.0
    value_loss_weight: float = 0.0


@dataclass
class AWRConfig:
    batch_size: int = 256
    loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    beta: float = 1.0
    max_weight: float = 20.0
    normalize_advantage: bool = True


@dataclass
class OfflineTrainConfig:
    steps: int = 5000
    lr: float = 3e-4
    log_every: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip_norm: Optional[float] = None


class JointBcAwrTrainer:
    """Joint trainer that sums golden BC and dataset-wide AWR losses each step."""

    def __init__(
        self,
        policy: nn.Module,
        golden_source: Optional[BatchSource],
        awr_source: Optional[BatchSource],
        bc_config: Optional[GoldenBCConfig] = None,
        awr_config: Optional[AWRConfig] = None,
        train_config: Optional[OfflineTrainConfig] = None,
        log_fn: Optional[LogFn] = None,
    ):
        self.policy = policy
        self.golden_source = golden_source
        self.awr_source = awr_source
        self.bc_config = bc_config or GoldenBCConfig()
        self.awr_config = awr_config or AWRConfig()
        self.train_config = train_config or OfflineTrainConfig()
        self.log_fn = log_fn
        self.device = torch.device(self.train_config.device)
        self.policy.to(self.device)

        self.use_bc = self.bc_config.loss_weight > 0.0
        self.use_awr = self.awr_config.loss_weight > 0.0

        if not self.use_bc and not self.use_awr:
            raise ValueError("At least one of BC or AWR must have a positive loss weight.")
        if self.use_bc and self.golden_source is None:
            raise ValueError("golden_source is required when BC loss is enabled.")
        if self.use_awr and self.awr_source is None:
            raise ValueError("awr_source is required when AWR loss is enabled.")

    def train(self) -> None:
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.train_config.lr)
        self.policy.train()

        for step in range(1, self.train_config.steps + 1):
            total_loss = None
            metrics: Dict[str, float] = {"step": float(step)}

            if self.use_bc:
                bc_batch = self._prepare_batch(
                    self.golden_source.sample(self.bc_config.batch_size, self.device),  # type: ignore[union-attr]
                    required_keys=("obs", "action"),
                )
                bc_loss, bc_value_loss = self._compute_bc_loss(bc_batch)
                weighted_bc_loss = (
                    self.bc_config.loss_weight * bc_loss
                    + self.bc_config.value_loss_weight * bc_value_loss
                )
                total_loss = weighted_bc_loss if total_loss is None else total_loss + weighted_bc_loss
                metrics["bc_loss"] = float(bc_loss.detach().cpu())
                metrics["bc_value_loss"] = float(bc_value_loss.detach().cpu())

            if self.use_awr:
                awr_batch = self._prepare_batch(
                    self.awr_source.sample(self.awr_config.batch_size, self.device),  # type: ignore[union-attr]
                    required_keys=("obs", "action"),
                )
                awr_actor_loss, awr_value_loss, avg_weight = self._compute_awr_loss(
                    awr_batch
                )
                weighted_awr_loss = (
                    self.awr_config.loss_weight * awr_actor_loss
                    + self.awr_config.value_loss_weight * awr_value_loss
                )
                total_loss = weighted_awr_loss if total_loss is None else total_loss + weighted_awr_loss
                metrics["awr_actor_loss"] = float(awr_actor_loss.detach().cpu())
                metrics["awr_value_loss"] = float(awr_value_loss.detach().cpu())
                metrics["awr_avg_weight"] = float(avg_weight.detach().cpu())

            if total_loss is None:
                raise RuntimeError("No active loss was computed.")

            self._optimize(optimizer, total_loss)
            metrics["total_loss"] = float(total_loss.detach().cpu())

            if step % self.train_config.log_every == 0 or step == 1:
                self._log(metrics)

    def _compute_bc_loss(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        dist, values = self._forward(batch)
        action = batch["action"].long()
        bc_loss = -dist.log_prob(action).mean()

        value_target = self._value_target(batch)
        if value_target is None:
            bc_value_loss = values.new_zeros(())
        else:
            bc_value_loss = F.mse_loss(values, value_target)

        return bc_loss, bc_value_loss

    def _compute_awr_loss(
        self,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self._forward(batch)
        action = batch["action"].long()

        advantage = self._advantage(batch, values)
        if self.awr_config.normalize_advantage:
            advantage = (advantage - advantage.mean()) / advantage.std(
                unbiased=False
            ).clamp_min(1e-6)

        weights = torch.exp(advantage / self.awr_config.beta).clamp(
            max=self.awr_config.max_weight
        )
        actor_loss = -(weights.detach() * dist.log_prob(action)).mean()

        value_target = self._value_target(batch)
        if value_target is None:
            value_loss = values.new_zeros(())
        else:
            value_loss = F.mse_loss(values, value_target)

        return actor_loss, value_loss, weights.mean()

    def _forward(self, batch: Batch) -> Tuple[Categorical, torch.Tensor]:
        obs = batch["obs"]
        hidden_state = batch.get("imagination")
        if hidden_state is None:
            hidden_state = batch.get("hidden_state")

        policy_out, value = self.policy(obs, hidden_state)
        if isinstance(policy_out, Categorical):
            return policy_out, value
        return Categorical(logits=policy_out), value

    def _value_target(self, batch: Batch) -> Optional[torch.Tensor]:
        for key in ("returns", "return_to_go", "value_target"):
            if key in batch:
                return batch[key].float()
        return None

    def _advantage(self, batch: Batch, values: torch.Tensor) -> torch.Tensor:
        for key in ("advantage", "advantages"):
            if key in batch:
                return batch[key].float()

        value_target = self._value_target(batch)
        if value_target is None:
            raise ValueError(
                "AWR batch must provide `advantage`/`advantages` or one of "
                "`returns`, `return_to_go`, `value_target`."
            )
        return value_target - values.detach()

    def _prepare_batch(self, batch: Batch, required_keys: Tuple[str, ...]) -> Batch:
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required batch key: `{key}`")

        normalized: Batch = {}
        for key, value in batch.items():
            normalized[key] = torch.as_tensor(value, device=self.device)
        return normalized

    def _optimize(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.train_config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.train_config.grad_clip_norm
            )
        optimizer.step()

    def _log(self, metrics: Mapping[str, float]) -> None:
        if self.log_fn is not None:
            self.log_fn(metrics)
        else:
            print(dict(metrics))
