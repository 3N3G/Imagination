"""
Optimized vLLM batch extraction for 128+ environments.

Sends all prompts in a single batched request to vLLM for maximum efficiency.
"""

import time
import numpy as np
import requests
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer

from llm.prompts import create_prompt


class VLLMBatchExtractor:
    """
    Optimized vLLM hidden state extractor using true batched inference.

    Sends all prompts in a single request instead of concurrent individual requests.
    This should provide ~10-20x speedup for large batches.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model_name: str = "./configs/vllm_hidden_qwen4b",
        model_id: str = "Qwen/Qwen3-4B-Thinking-2507",
        hidden_states_path: str = "/tmp/hidden_states",
        hidden_size: int = 2560,
        target_layer: int = -1,
    ):
        self.server_url = server_url
        self.model_name = model_name
        self.hidden_states_path = hidden_states_path
        self.hidden_size = hidden_size
        self.target_layer = target_layer

        # Load tokenizer for prompt formatting
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Verify server
        try:
            resp = requests.get(f"{server_url}/health", timeout=5)
            print(f"vLLM server health: {resp.status_code}")
        except Exception as e:
            print(f"WARNING: vLLM server not reachable at {server_url}: {e}")

    def extract_batch(
        self,
        observations: List[str],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extract hidden states for all observations in a single batched request.

        Args:
            observations: List of filtered text observations

        Returns:
            hidden_states: (N, hidden_size) hidden states
            metrics: Timing metrics
        """
        if not observations:
            return np.array([]), {}

        start_time = time.perf_counter()

        # Format all prompts
        prompts = [create_prompt(obs, self.tokenizer) for obs in observations]

        # Send single batched request
        response = requests.post(
            f"{self.server_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": self.model_name,
                "prompt": prompts,  # List of prompts for batched inference
                "max_tokens": 1,
                "temperature": 0.0,
                "n": 1,
            },
            timeout=30,  # Longer timeout for large batches
        )
        response.raise_for_status()
        result = response.json()

        # Extract hidden states from batched response
        all_hidden = []
        for choice in result.get("choices", []):
            kv_params = choice.get("kv_transfer_params", {})
            hs_path = kv_params.get("hidden_states_path")

            if hs_path:
                # Load from safetensors file
                import safetensors.torch
                hs_dict = safetensors.torch.load_file(hs_path)

                # Get target layer hidden state
                target_key = f"hidden_states.{self.target_layer}"
                if target_key in hs_dict:
                    hs_tensor = hs_dict[target_key]
                    # For last_token mode: already (1, hidden_size)
                    # For all mode: take last token from (1, seq_len, hidden_size)
                    if hs_tensor.dim() == 3:
                        hs_tensor = hs_tensor[0, -1, :]
                    elif hs_tensor.dim() == 2:
                        hs_tensor = hs_tensor[0]

                    all_hidden.append(hs_tensor.numpy())
                else:
                    # Fallback to zero
                    all_hidden.append(np.zeros(self.hidden_size, dtype=np.float32))
            else:
                all_hidden.append(np.zeros(self.hidden_size, dtype=np.float32))

        elapsed = time.perf_counter() - start_time

        hidden_states = np.stack(all_hidden, axis=0)

        metrics = {
            "llm/batch_size": len(observations),
            "llm/inference_time": elapsed,
            "llm/throughput": len(observations) / elapsed,
        }

        return hidden_states, metrics