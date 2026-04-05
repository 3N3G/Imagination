#!/usr/bin/env python3
"""
Create vLLM hidden states config for specific layers.

This script generates a config that extracts only the specified layers,
allowing flexible hidden state extraction without hardcoding.

Usage:
    # Extract only last layer (35)
    python scripts/create_vllm_config.py --layers 35 --output configs/vllm_hidden_last/

    # Extract multiple layers
    python scripts/create_vllm_config.py --layers 8 16 24 35 --output configs/vllm_hidden_multi/

    # Extract all layers (expensive!)
    python scripts/create_vllm_config.py --layers $(seq 0 35) --output configs/vllm_hidden_all/
"""

import argparse
import json
import os
from pathlib import Path


def create_config(layers, output_dir, model_id="Qwen/Qwen3-4B-Thinking-2507"):
    """
    Create vLLM config for extracting specific hidden state layers.

    Args:
        layers: List of layer indices to extract (0-35 for Qwen3-4B)
        output_dir: Directory to save config
        model_id: HuggingFace model ID
    """

    # Qwen3-4B model parameters
    HIDDEN_SIZE = 2560
    NUM_LAYERS = 36
    HEAD_DIM = 128

    # Validate layers
    for layer in layers:
        if not (0 <= layer < NUM_LAYERS):
            raise ValueError(f"Invalid layer {layer}. Must be 0-{NUM_LAYERS-1}")

    # Calculate speculator hidden size (concatenated layers)
    num_extracted = len(layers)
    speculator_hidden = HIDDEN_SIZE * num_extracted

    # Calculate attention heads for speculator
    # This is a bit of voodoo from the EAGLE architecture
    # Formula: speculator_hidden / head_dim / 4 * 2 (for k,v split and internal multiplier)
    # Simplified: just set to a reasonable value
    num_heads = max(1, min(20, speculator_hidden // HEAD_DIM // 4))

    config = {
        "architectures": ["Eagle3Speculator"],
        "auto_map": {
            "": "eagle3.Eagle3SpeculatorConfig"
        },
        "draft_vocab_size": 32000,
        "eagle_aux_hidden_state_layer_ids": layers,
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
                    "verifier_accept_k": 1
                }
            ],
            "verifier": {
                "architectures": ["Qwen3ForCausalLM"],
                "name_or_path": model_id
            }
        },
        "speculators_model_type": "extract_hidden_states",
        "speculators_version": "0.2.0.dev11",
        "torch_dtype": "bfloat16",
        "transformer_layer_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "head_dim": HEAD_DIM,
            "hidden_act": "silu",
            "hidden_size": speculator_hidden,
            "initializer_range": 0.02,
            "intermediate_size": 9728,  # From original Qwen3-4B config
            "max_position_embeddings": 262144,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": num_heads,
            "num_hidden_layers": 1,  # Speculator has 1 layer
            "num_key_value_heads": num_heads,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 5000000,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 151936  # Qwen3 vocab size
        },
        "transformers_version": "4.57.1"
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved to {config_path}")

    # Create README
    readme_content = f"""# vLLM Hidden States Config

Auto-generated config for extracting hidden states from specific layers.

## Configuration
- **Model**: {model_id}
- **Extracted Layers**: {layers}
- **Number of Layers**: {num_extracted}
- **Hidden Size per Layer**: {HIDDEN_SIZE}
- **Total Hidden Size**: {speculator_hidden}

## Usage

```bash
# Start server
vllm serve "{output_dir}" \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.95 \\
    --kv-transfer-config '{{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{{"shared_storage_path":"/tmp/hidden_states","mode":"last_token"}}}}'
```

## Notes
- Mode "last_token" extracts only the last token (for prompt-only inference)
- Mode "all" extracts all tokens (for generation)
- Output shape: [{num_extracted}, seq_len, {HIDDEN_SIZE}]
"""

    readme_path = output_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"✅ README saved to {readme_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create vLLM config for specific layers")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[35],
        help="Layer indices to extract (0-35 for Qwen3-4B). Default: 35 (last layer)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/vllm_hidden_custom/",
        help="Output directory for config"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="HuggingFace model ID"
    )

    args = parser.parse_args()

    print(f"Creating config for layers: {args.layers}")
    output_path = create_config(args.layers, args.output, args.model)

    print(f"\n✨ Config created successfully!")
    print(f"📁 Location: {output_path}")
    print(f"\nTo use:")
    print(f"  vllm serve {output_path} --max-model-len 8192 ...")


if __name__ == "__main__":
    main()