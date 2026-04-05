"""
Modified vLLM Hidden States Connector

Drop-in replacement for ExampleHiddenStatesConnector that supports:
  - mode="all"        : save all token hidden states (default, original behavior)
  - mode="last_token" : save only the last token's hidden state per layer

Configure via kv_connector_extra_config:
  {
    "shared_storage_path": "/tmp/hidden_states",
    "mode": "last_token"   # or "all"
  }

Installation:
  Copy this file over the installed connector.py:
    cp utils/vllm_hidden_connector.py \
       $(python -c "import vllm_hidden_states_extractor; import os; print(os.path.dirname(vllm_hidden_states_extractor.__file__))")/connector.py
"""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import safetensors
import torch

from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_hidden_states_extractor.model import CacheOnlyAttentionLayer
from vllm_hidden_states_extractor.utils import reshape_hidden_states_from_kv_cache

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    num_tokens: int  # actual tokens (excluding padding from block alignment)

    @staticmethod
    def make_meta(
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> "ReqMeta":
        token_ids_tensor = torch.tensor(token_ids)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()
        return ReqMeta(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            num_tokens=len(token_ids),
        )


@dataclass
class ExampleHiddenStatesConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(req_id, filename, token_ids, block_ids, block_size)
        )


class ExampleHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """
    KV Connector that extracts hidden states from CacheOnlyAttentionLayers.

    Supports two modes via kv_connector_extra_config:
      - "all" (default): Save hidden states for all tokens
      - "last_token": Save only the last token's hidden state

    Output safetensors format:
      - "hidden_states": [num_layers, seq_len, hidden_size]  (mode="all")
                      or [num_layers, 1, hidden_size]         (mode="last_token")
      - "token_ids":     [seq_len]
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp/hidden_states"
        )
        # Mode: "all" or "last_token"
        self._mode = self._kv_transfer_config.get_from_extra_config(
            "mode", "all"
        )
        assert self._mode in ("all", "last_token"), (
            f"Invalid mode: {self._mode}. Must be 'all' or 'last_token'"
        )

        self.cache_layers = []
        logger.info("Hidden states connector config:")
        logger.info("  storage_path: %s", self._storage_path)
        logger.info("  mode: %s", self._mode)

        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )
        logger.info("  num_hidden_states: %d", self.num_hidden_states)

        self._request_filenames: dict[str, str] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, kv_caches.keys()
        )
        self.cache_layers = list(layers.keys())
        logger.info(f"Found {len(self.cache_layers)} CacheOnlyAttentionLayers")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
            num_tokens: int,
        ) -> torch.Tensor:
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            padded_kv = layer.reshape(2, num_pages * page_size, -1)[
                :, slot_mapping, ...
            ]
            return padded_kv[:, :num_tokens, ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata)
        os.makedirs(self._storage_path, exist_ok=True)
        attn_slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        cursor = 0
        if isinstance(attn_metadata, MLACommonMetadata):
            kv_capacity = int(kv_layer.shape[0] * kv_layer.shape[1])
        else:
            kv_capacity = int(kv_layer.shape[1] * kv_layer.shape[2])

        for request in connector_metadata.requests:
            req_num_tokens = int(request.num_tokens)
            req_slot_mapping = request.slot_mapping[:req_num_tokens]
            req_token_ids = request.token_ids[: req_slot_mapping.shape[0]]
            if isinstance(attn_slot_mapping, torch.Tensor):
                slot_count = int(req_slot_mapping.shape[0])
                candidate = attn_slot_mapping[cursor : cursor + slot_count]
                cursor += slot_count
                if candidate.numel() > 0:
                    req_slot_mapping = candidate
                    req_token_ids = req_token_ids[: candidate.shape[0]]
            valid_mask = (req_slot_mapping >= 0) & (req_slot_mapping < kv_capacity)
            if valid_mask.numel() == 0 or not bool(valid_mask.any().item()):
                logger.warning(
                    "No valid slot mappings for req %s (kv_capacity=%d, num_slots=%d)",
                    getattr(request, "req_id", "<unknown>"),
                    kv_capacity,
                    int(req_slot_mapping.numel()),
                )
                continue
            if not bool(valid_mask.all().item()):
                num_dropped = int((~valid_mask).sum().item())
                slot_min = int(req_slot_mapping.min().item())
                slot_max = int(req_slot_mapping.max().item())
                logger.warning(
                    "Dropping %d invalid slot mappings for req %s (kv_capacity=%d, slot_min=%d, slot_max=%d)",
                    num_dropped,
                    getattr(request, "req_id", "<unknown>"),
                    kv_capacity,
                    slot_min,
                    slot_max,
                )
                req_slot_mapping = req_slot_mapping[valid_mask]
                req_token_ids = req_token_ids[valid_mask.detach().cpu()]
            if self._mode == "last_token" and req_slot_mapping.numel() > 1:
                req_slot_mapping = req_slot_mapping[-1:]
                req_token_ids = req_token_ids[-1:]
            req_num_tokens = int(req_slot_mapping.shape[0])
            kv_cache = extract_kv_from_layer(
                kv_layer, req_slot_mapping, req_num_tokens
            )
            hidden_states = reshape_hidden_states_from_kv_cache(
                kv_cache, self.num_hidden_states
            )
            # hidden_states shape: [num_layers, seq_len, hidden_size]

            if self._mode == "last_token":
                # Only keep the last token's hidden state
                hidden_states = hidden_states[:, -1:, :]

            tensors = {
                "hidden_states": hidden_states.detach().cpu(),
                "token_ids": req_token_ids[:req_num_tokens].detach().cpu(),
            }
            safetensors.torch.save_file(tensors, request.filename)

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = ExampleHiddenStatesConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            filename = os.path.join(self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
                block_ids=new_req.block_ids[0],
                block_size=self._block_size,
            )
            self._request_filenames[new_req.req_id] = filename

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)
        return False, {"hidden_states_path": req_filename}

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # HMA path: hidden-state artifact is keyed by request id and does not
        # depend on per-group block ids, so we can reuse the single-group logic.
        primary_group = block_ids[0] if block_ids else []
        return self.request_finished(request, primary_group)

    def clear_connector_metadata(self):
        # Request metadata must be cleared between engine steps. Leaving stale
        # connector metadata around can corrupt the next request's slot mapping.
        self._connector_metadata = None

    def real_clear_connector_metadata(self):
        self._connector_metadata = None
