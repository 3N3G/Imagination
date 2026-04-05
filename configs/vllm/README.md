# vLLM Hidden States Config

Auto-generated config for extracting hidden states from specific layers.

## Configuration
- **Model**: Qwen/Qwen3-4B-Thinking-2507
- **Extracted Layers**: [35]
- **Number of Layers**: 1
- **Hidden Size per Layer**: 2560
- **Total Hidden Size**: 2560

## Usage

```bash
# Start server
vllm serve "configs/vllm_hidden_last/" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --kv-transfer-config '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/hidden_states","mode":"last_token"}}'
```

## Notes
- Mode "last_token" extracts only the last token (for prompt-only inference)
- Mode "all" extracts all tokens (for generation)
- Output shape: [1, seq_len, 2560]
