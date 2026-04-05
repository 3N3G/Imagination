#!/usr/bin/env python3
"""
Create a vLLM hidden-state extractor config from a Hugging Face model config.

This is a model-agnostic variant of `scripts/create_vllm_config.py` intended for
Qwen family models beyond Qwen3-4B.

Examples:
  python scripts/create_vllm_config_auto.py \
    --model Qwen/Qwen3.5-9B \
    --layers auto4 \
    --output configs/vllm_hidden_qwen35_9b

  python scripts/create_vllm_config_auto.py \
    --model Qwen/Qwen3.5-27B \
    --layers last \
    --output configs/vllm_hidden_qwen35_27b_last
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
from urllib import parse as urlparse
from urllib import request as urlrequest


def _resolve_layers(spec: str, num_layers: int) -> List[int]:
    spec = spec.strip().lower()
    if spec == "last":
        return [num_layers - 1]
    if spec == "auto4":
        # Evenly spread taps with an explicit final-layer tap.
        step = max(1, num_layers // 4)
        raw = [step - 1, 2 * step - 1, 3 * step - 1, num_layers - 1]
        return sorted({max(0, min(num_layers - 1, x)) for x in raw})
    raise ValueError(f"Unsupported layer spec '{spec}'. Use one of: auto4, last.")


def _normalize_explicit_layers(raw_layers: List[int], num_layers: int) -> List[int]:
    norm: List[int] = []
    for v in raw_layers:
        idx = int(v)
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {v} out of range for num_layers={num_layers}")
        norm.append(idx)
    return sorted(set(norm))


def _load_hf_config_dict(model_id: str) -> dict:
    """
    Load raw config.json for a model.

    Prefer huggingface_hub direct download so we are not blocked by the local
    transformers version understanding newer model_type values.
    """
    local_path = Path(model_id).expanduser()
    candidate = local_path / "config.json" if local_path.is_dir() else local_path
    if candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1) Raw HTTP fetch works even when local transformers/hf_hub are old/missing.
    try:
        quoted = urlparse.quote(model_id, safe="/")
        url = f"https://huggingface.co/{quoted}/raw/main/config.json"
        with urlrequest.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass

    # 2) huggingface_hub fallback
    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass

    # 3) transformers fallback
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return cfg.to_dict()


def _cfg_lookup(cfg: dict, key: str, default=None):
    if key in cfg and cfg.get(key) is not None:
        return cfg.get(key)
    for nested_key in ("text_config", "llm_config"):
        nested = cfg.get(nested_key)
        if isinstance(nested, dict) and nested.get(key) is not None:
            return nested.get(key)
    return default


def _require(v, name: str):
    if v is None:
        raise ValueError(f"Missing required config key: {name}")
    return v


def _build_config(
    model_id: str,
    layer_ids: List[int],
    verifier_archs_override: Optional[List[str]] = None,
    verifier_name_or_path: Optional[str] = None,
) -> dict:
    cfg = _load_hf_config_dict(model_id)

    hidden_size = int(_require(_cfg_lookup(cfg, "hidden_size"), "hidden_size"))
    num_layers = int(
        _require(_cfg_lookup(cfg, "num_hidden_layers"), "num_hidden_layers")
    )
    num_attention_heads = int(
        _require(_cfg_lookup(cfg, "num_attention_heads"), "num_attention_heads")
    )
    head_dim = int(
        _cfg_lookup(cfg, "head_dim", hidden_size // max(1, num_attention_heads))
    )
    intermediate_size = int(_cfg_lookup(cfg, "intermediate_size", hidden_size * 4))
    max_position_embeddings = int(_cfg_lookup(cfg, "max_position_embeddings", 32768))
    hidden_act = str(_cfg_lookup(cfg, "hidden_act", "silu"))
    vocab_size = int(_cfg_lookup(cfg, "vocab_size", 151936))
    rms_norm_eps = float(_cfg_lookup(cfg, "rms_norm_eps", 1e-6))
    rope_theta = float(_cfg_lookup(cfg, "rope_theta", 5000000.0))
    if verifier_archs_override:
        verifier_archs = list(verifier_archs_override)
    else:
        verifier_archs = list(cfg.get("architectures") or ["Qwen3ForCausalLM"])

    for layer in layer_ids:
        if not (0 <= layer < num_layers):
            raise ValueError(
                f"Invalid layer {layer}; must satisfy 0 <= layer < {num_layers}"
            )

    num_extracted = len(layer_ids)
    speculator_hidden = hidden_size * num_extracted
    # Heuristic carried from the existing config builder.
    spec_num_heads = max(1, min(64, speculator_hidden // max(1, head_dim) // 4))

    return {
        "architectures": ["Eagle3Speculator"],
        "auto_map": {"": "eagle3.Eagle3SpeculatorConfig"},
        "draft_vocab_size": 32000,
        "eagle_aux_hidden_state_layer_ids": layer_ids,
        "has_no_defaults_at_init": False,
        "norm_before_residual": True,
        "speculators_config": {
            "algorithm": "eagle3",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "accept_tolerance": 0.0,
                    "proposal_type": "greedy",
                    "speculative_tokens": 1,
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "architectures": verifier_archs,
                "name_or_path": verifier_name_or_path or model_id,
            },
        },
        "speculators_model_type": "extract_hidden_states",
        "speculators_version": "0.2.0.dev11",
        "torch_dtype": "bfloat16",
        "transformer_layer_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "head_dim": head_dim,
            "hidden_act": hidden_act,
            "hidden_size": speculator_hidden,
            "initializer_range": 0.02,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": spec_num_heads,
            "num_hidden_layers": 1,
            "num_key_value_heads": spec_num_heads,
            "pretraining_tp": 1,
            "rms_norm_eps": rms_norm_eps,
            "rope_scaling": None,
            "rope_theta": rope_theta,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": vocab_size,
        },
        "transformers_version": "4.57.1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create model-aware vLLM hidden extractor config."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id (e.g. Qwen/Qwen3.5-9B).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for generated config.json.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["auto4"],
        help=(
            "Layer selection. Either one of: auto4, last, or explicit indices "
            "(supports negative indices, e.g. -1)."
        ),
    )
    parser.add_argument(
        "--verifier-architectures",
        nargs="+",
        default=None,
        help=(
            "Optional override for verifier architectures in generated config "
            "(e.g. Qwen3ForCausalLM)."
        ),
    )
    parser.add_argument(
        "--verifier-name-or-path",
        default=None,
        help=(
            "Optional override for verifier.name_or_path in generated config "
            "(e.g. local patched model directory)."
        ),
    )
    args = parser.parse_args()

    hf_cfg = _load_hf_config_dict(args.model)
    num_layers = int(
        _require(_cfg_lookup(hf_cfg, "num_hidden_layers"), "num_hidden_layers")
    )

    explicit_ok = all(
        tok.lstrip("-").isdigit() for tok in args.layers
    )
    if len(args.layers) == 1 and not explicit_ok:
        layer_ids = _resolve_layers(args.layers[0], num_layers)
    else:
        layer_ids = _normalize_explicit_layers([int(x) for x in args.layers], num_layers)

    config = _build_config(
        args.model,
        layer_ids,
        args.verifier_architectures,
        args.verifier_name_or_path,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    meta_path = out_dir / "metadata.json"

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metadata = {
        "model_id": args.model,
        "num_hidden_layers": num_layers,
        "selected_layers": layer_ids,
        "hidden_size_per_layer": int(
            _require(_cfg_lookup(hf_cfg, "hidden_size"), "hidden_size")
        ),
        "verifier_architectures": list(hf_cfg.get("architectures") or []),
        "verifier_architectures_used": list(
            args.verifier_architectures or (hf_cfg.get("architectures") or [])
        ),
        "verifier_name_or_path_used": args.verifier_name_or_path or args.model,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"config_path={cfg_path}")
    print(f"metadata_path={meta_path}")
    print(f"model={args.model}")
    print(f"num_hidden_layers={num_layers}")
    print(f"selected_layers={layer_ids}")


if __name__ == "__main__":
    main()
