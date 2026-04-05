
import collections
import concurrent.futures
import glob
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# Torch is optional - only needed for LLMHiddenStateExtractor (HuggingFace)
# VLLMHiddenStateExtractor works without torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from llm.prompts import create_prompt, get_generation_prefix, get_system_prompt

# Default Configuration
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

TOKENS_GENERATED = 256


def compute_hidden_state_diagnostics(hidden_states: np.ndarray) -> Dict[str, float]:
    """
    Compute diagnostics for hidden state quality and diversity.
    
    Args:
        hidden_states: (batch, hidden_dim) pooled hidden states
        
    Returns:
        Dictionary of diagnostic metrics
    """
    if len(hidden_states) == 0:
        return {}
    
    # Basic statistics
    metrics = {
        "hidden/mean": float(np.mean(hidden_states)),
        "hidden/std": float(np.std(hidden_states)),
        "hidden/min": float(np.min(hidden_states)),
        "hidden/max": float(np.max(hidden_states)),
        "hidden/norm_mean": float(np.mean(np.linalg.norm(hidden_states, axis=1))),
    }
    
    # Per-dimension statistics (variance across batch for each dim)
    per_dim_var = np.var(hidden_states, axis=0)
    metrics["hidden/dim_var_mean"] = float(np.mean(per_dim_var))
    metrics["hidden/dim_var_std"] = float(np.std(per_dim_var))
    metrics["hidden/active_dims"] = int(np.sum(per_dim_var > 0.01))  # Dims with variance
    
    # Diversity: pairwise cosine similarity
    if len(hidden_states) >= 2:
        # Normalize
        norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normalized = hidden_states / norms
        
        # Compute pairwise cosine similarities (sample if too many)
        n = len(normalized)
        if n > 32:
            # Sample subset for efficiency
            idx = np.random.choice(n, 32, replace=False)
            normalized = normalized[idx]
        
        cos_sim = normalized @ normalized.T
        # Get upper triangle (exclude diagonal)
        upper_idx = np.triu_indices(len(cos_sim), k=1)
        pairwise_cos = cos_sim[upper_idx]
        
        metrics["hidden/diversity_mean_cos_sim"] = float(np.mean(pairwise_cos))
        metrics["hidden/diversity_std_cos_sim"] = float(np.std(pairwise_cos))
        metrics["hidden/diversity_min_cos_sim"] = float(np.min(pairwise_cos))
        metrics["hidden/diversity_max_cos_sim"] = float(np.max(pairwise_cos))
    
    return metrics


class LLMHiddenStateExtractor:
    """
    Extracts hidden states from LLM reasoning using HuggingFace + Flash Attention.
    Generates text and returns pooled hidden states.

    Requires torch and transformers to be installed.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        tokens_generated: int = TOKENS_GENERATED,
        device: str = None,
        prompt_variant: str = "default",
        system_prompt: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "LLMHiddenStateExtractor requires torch. "
                "Use VLLMHiddenStateExtractor instead (works without torch)."
            )
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.tokens_generated = tokens_generated
        self.device = device
        self.prompt_variant = (prompt_variant or "default").strip().lower()
        self.system_prompt = (
            get_system_prompt(self.prompt_variant) if system_prompt is None else system_prompt
        )
        self.generation_prefix = get_generation_prefix(self.prompt_variant)
        
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"  # Required for batched generation with decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model with Flash Attention 2...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            self.flash_enabled = True
        except Exception as e:
            print(f"Flash Attention/BFloat16 not available: {e}, falling back to float16")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.flash_enabled = False
        
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded (flash_attention={self.flash_enabled}, hidden_size={self.hidden_size})")
        
        # Metrics
        self.total_samples = 0
        self.total_time = 0.0
        self.batch_times = []
        self._missing_path_count = 0
        self._zero_vector_fallback_count = 0
    
    def extract_hidden_states(
        self, 
        observations: List[str],
        batch_size: int = 8,
    ) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """
        Generate text and extract hidden states.
        
        Args:
            observations: List of filtered text observations
            batch_size: Batch size for processing
            
        Returns:
            hidden_states: (N, hidden_size) pooled hidden states
            generated_texts: List of generated text strings
            metrics: Timing and diagnostic metrics
        """
        if not observations:
            return np.array([]), [], {}
        
        start_time = time.perf_counter()
        
        all_hidden = []
        all_texts = []
        batch_latencies = []
        
        for batch_idx in range(0, len(observations), batch_size):
            batch_start = time.perf_counter()
            batch_obs = observations[batch_idx:batch_idx + batch_size]
            prompts = [
                create_prompt(
                    obs,
                    self.tokenizer,
                    system_prompt=self.system_prompt,
                    prompt_variant=self.prompt_variant,
                )
                for obs in batch_obs
            ]
            if self.generation_prefix:
                prompts = [p + self.generation_prefix for p in prompts]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)
            
            prompt_len = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.tokens_generated,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=0.7,
                )
            
            # Extract hidden states from last layer
            last_layer_states = [s[-1] for s in outputs.hidden_states]
            # (batch, seq, hidden)
            generated_hidden = torch.cat(last_layer_states, dim=1)
            
            # Truncate/pad to tokens_generated
            # We want to capture the reasoning process. 
            # Note: outputs.hidden_states usually contains (prompt + gen).
            # But here we are concatenating across generation steps?
            # Actually HuggingFace `generate` with output_hidden_states returns a tuple of tuples.
            # outputs.hidden_states is a tuple (one per step) of tuples (one per layer).
            # We just want the last layer for each step.
            
            # Correct logic for extracting generated hidden states:
            # outputs.hidden_states is tuple of length `generated_tokens`.
            # Each element is tuple of length `num_layers`.
            # We want stacked (batch, gen_len, hidden).
            
            # If we used `inputs` directly, we'd get prompt hidden states too. 
            # But `generate` returns hidden states for *generated* tokens only usually?
            # Let's verify standard HF behavior:
            # "The hidden states of the model at the output of each layer plus the initial embedding outputs."
            # It usually returns hidden states for each generated token.
            
            # Extract hidden states from the last layer. 
            # In generate(), each step returns hidden states for the new tokens.
            # We take the last token's hidden state to ensure consistent shape (batch, hidden_size).
            hs_per_step = [step_hs[-1][:, -1, :] for step_hs in outputs.hidden_states]
            generated_hidden = torch.stack(hs_per_step, dim=1)  # (batch, seq_len, hidden_size)
            
            # Truncate/pad to tokens_generated
            seq_len = generated_hidden.shape[1]
            if seq_len > self.tokens_generated:
                generated_hidden = generated_hidden[:, :self.tokens_generated, :]
            elif seq_len < self.tokens_generated:
                padding = torch.zeros(
                    (len(batch_obs), self.tokens_generated - seq_len, self.hidden_size),
                    device=generated_hidden.device,
                    dtype=generated_hidden.dtype,
                )
                generated_hidden = torch.cat([generated_hidden, padding], dim=1)
            
            # Mean pool to (batch, hidden_size)
            pooled = generated_hidden.mean(dim=1).to(torch.float32).cpu().numpy()
            all_hidden.append(pooled)
            
            # Decode text
            generated_ids = outputs.sequences[:, prompt_len:]
            texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_texts.extend(texts)
            
            batch_latencies.append(time.perf_counter() - batch_start)
        
        elapsed = time.perf_counter() - start_time
        self.total_samples += len(observations)
        self.total_time += elapsed
        self.batch_times.extend(batch_latencies)
        
        all_hidden = np.concatenate(all_hidden, axis=0)
        
        # Compute metrics
        metrics = {
            "llm/inference_time_s": elapsed,
            "llm/samples_per_sec": len(observations) / elapsed,
            "llm/batch_latency_mean_s": np.mean(batch_latencies),
            "llm/total_samples": self.total_samples,
            "llm/total_time_s": self.total_time,
        }
        
        # Add hidden state diagnostics
        hidden_diag = compute_hidden_state_diagnostics(all_hidden)
        metrics.update(hidden_diag)
        
        return all_hidden, all_texts, metrics
    
    def get_metrics(self) -> Dict:
        return {
            "total_samples": self.total_samples,
            "total_time_s": self.total_time,
            "samples_per_sec": self.total_samples / max(0.001, self.total_time),
        }

    def extract_hidden_states_no_cot(
        self,
        observations: List[str],
        batch_size: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extract hidden states via prompt-only forward pass (no generation).
        
        Takes the last token's hidden state from the last layer after processing
        the full prompt. This is deterministic, ~34x faster than generation, and
        produces better observation-discriminating representations (see analysis/
        hidden_state_noise_v2.py results).
        
        Args:
            observations: List of filtered text observations
            batch_size: Batch size for processing
            
        Returns:
            hidden_states: (N, hidden_size) last-token hidden states
            metrics: Timing and diagnostic metrics
        """
        if not observations:
            return np.array([]), {}
        
        start_time = time.perf_counter()
        
        all_hidden = []
        batch_latencies = []
        
        for batch_idx in range(0, len(observations), batch_size):
            batch_start = time.perf_counter()
            batch_obs = observations[batch_idx:batch_idx + batch_size]
            prompts = [
                create_prompt(
                    obs,
                    self.tokenizer,
                    system_prompt=self.system_prompt,
                    prompt_variant=self.prompt_variant,
                )
                for obs in batch_obs
            ]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)
            
            with torch.no_grad():
                # Avoid CausalLM logits allocation for no-CoT extraction.
                # Calling the backbone model directly significantly lowers memory.
                backbone = self.model.model if hasattr(self.model, "model") else self.model
                outputs = backbone(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            
            # Pre-norm: use [-2] to get the last transformer layer output BEFORE
            # the final RMSNorm. This matches vLLM plugin behavior and preserves
            # more information. ([-1] would be post-norm, with smaller norms.)
            last_layer = outputs.hidden_states[-2]  # (batch, seq, hidden)
            pooled = last_layer[:, -1, :].to(torch.float32).cpu().numpy()
            all_hidden.append(pooled)
            
            batch_latencies.append(time.perf_counter() - batch_start)
        
        elapsed = time.perf_counter() - start_time
        self.total_samples += len(observations)
        self.total_time += elapsed
        self.batch_times.extend(batch_latencies)
        
        all_hidden = np.concatenate(all_hidden, axis=0)
        
        metrics = {
            "llm/inference_time_s": elapsed,
            "llm/samples_per_sec": len(observations) / elapsed,
            "llm/batch_latency_mean_s": np.mean(batch_latencies),
            "llm/total_samples": self.total_samples,
            "llm/total_time_s": self.total_time,
        }
        
        hidden_diag = compute_hidden_state_diagnostics(all_hidden)
        metrics.update(hidden_diag)
        
        return all_hidden, metrics


class VLLMHiddenStateExtractor:
    """
    Extracts hidden states from a running vLLM server with the
    vllm-hidden-states-extractor plugin.
    
    Drop-in replacement for LLMHiddenStateExtractor.
    Requires:
      1. vLLM server running with the hidden states plugin config
      2. ExampleHiddenStatesConnector configured (mode="last_token" or "all")
    
    Usage:
        extractor = VLLMHiddenStateExtractor(
            server_url="http://localhost:8000",
            model_name="./configs/vllm_hidden_qwen4b",
        )
        hidden_states, metrics = extractor.extract_hidden_states_no_cot(observations)
        # hidden_states: (N, hidden_size) numpy array
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model_name: str = "./configs/vllm_hidden_qwen4b",
        model_id: str = DEFAULT_MODEL_ID,
        hidden_states_path: str = "/tmp/hidden_states",
        hidden_size: int = 2560,
        num_extracted_layers: int = 4,
        target_layer: int = -1,
        max_workers: int = 16,
        hidden_pooling: str = "last_token",
        hidden_pooling_k: int = 8,
        prompt_variant: str = "default",
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            server_url: URL of the running vLLM server
            model_name: Model name for the vLLM API (matches the config dir)
            model_id: HuggingFace model ID (for tokenizer, used in prompt formatting)
            hidden_states_path: Directory where hidden states safetensors are saved
            hidden_size: Base model hidden size (2560 for Qwen3-4B)
            num_extracted_layers: Number of layers extracted by the plugin (default 4)
            target_layer: Which layer index to use from extracted layers (-1 = last)
            max_workers: Max concurrent HTTP requests
        """
        import requests as req_lib
        self._requests = req_lib
        
        self.server_url = server_url
        self.model_name = model_name
        self.hidden_states_path = hidden_states_path
        self.hidden_size = hidden_size
        self.num_extracted_layers = num_extracted_layers
        self.target_layer = target_layer
        self.max_workers = max_workers
        self.hidden_pooling = hidden_pooling
        self.hidden_pooling_k = max(1, int(hidden_pooling_k))
        self.prompt_variant = (prompt_variant or "default").strip().lower()
        self.system_prompt = (
            get_system_prompt(self.prompt_variant) if system_prompt is None else system_prompt
        )
        self.generation_prefix = get_generation_prefix(self.prompt_variant)
        if self.hidden_pooling not in ("last_token", "mean_last_k"):
            raise ValueError(
                f"Invalid hidden_pooling={self.hidden_pooling}; expected one of: "
                "'last_token', 'mean_last_k'"
            )

        # Load tokenizer for prompt formatting (transformers works without torch for tokenizers)
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Verify server is reachable
        try:
            resp = self._requests.get(f"{server_url}/health", timeout=5)
            print(f"vLLM server health: {resp.status_code}")
        except Exception as e:
            print(f"WARNING: vLLM server not reachable at {server_url}: {e}")
            print("  Make sure to start the server before calling extract methods.")

        # Metrics
        self.total_samples = 0
        self.total_time = 0.0
        self.batch_times = []
        
        print(f"VLLMHiddenStateExtractor initialized:")
        print(f"  server_url: {server_url}")
        print(f"  model_name: {model_name}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  target_layer: {target_layer} (of {num_extracted_layers})")
        print(f"  hidden_pooling: {self.hidden_pooling}")
        if self.hidden_pooling == "mean_last_k":
            print(f"  hidden_pooling_k: {self.hidden_pooling_k}")

    def _send_request(self, prompt: str, max_tokens: int = 1, temperature: float = 0.0) -> dict:
        """Send a single completion request to the vLLM server with retries."""
        last_error = None
        for attempt in range(3):
            try:
                resp = self._requests.post(
                    f"{self.server_url}/v1/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=(10, 180),
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"vLLM request failed after retries: {last_error}") from last_error

    def _get_hidden_state_path(self, result: dict) -> Optional[str]:
        """Extract hidden-states file path from response with completion-id fallback."""
        # Check in choices
        for choice in result.get("choices", []):
            kv_params = choice.get("kv_transfer_params", {})
            if kv_params and kv_params.get("hidden_states_path"):
                return kv_params["hidden_states_path"]
        # Check top-level
        kv_params = result.get("kv_transfer_params", {})
        if kv_params and kv_params.get("hidden_states_path"):
            return kv_params["hidden_states_path"]

        # Some vLLM responses may omit kv_transfer_params while still writing the
        # hidden-state file. Fall back to completion-id pattern lookup.
        completion_id = result.get("id")
        if completion_id:
            pattern = os.path.join(self.hidden_states_path, f"{completion_id}-*.safetensors")
            for attempt in range(5):
                matches = glob.glob(pattern)
                if matches:
                    return max(matches, key=os.path.getmtime)
                time.sleep(0.05 * (attempt + 1))
        return None

    def _load_hidden_state(self, path: str) -> Optional[np.ndarray]:
        """Load hidden state from safetensors file and return target layer."""
        from safetensors import safe_open

        if not path:
            return None
        if not os.path.exists(path):
            for attempt in range(8):
                time.sleep(0.05 * (attempt + 1))
                if os.path.exists(path):
                    break
            else:
                return None

        hs = None
        last_error = None
        for attempt in range(10):
            try:
                if TORCH_AVAILABLE:
                    # Prefer pt loader: robust for bfloat16 tensors.
                    with safe_open(path, framework="pt") as f:
                        hs = f.get_tensor("hidden_states").to(torch.float32).cpu().numpy()
                else:
                    with safe_open(path, framework="numpy") as f:
                        hs = f.get_tensor("hidden_states")  # [num_layers, seq_len, hidden_size]
                break
            except (OSError, EOFError, ValueError) as exc:
                # Handles stale file handle/truncated write races from shared FS.
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
            except Exception as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))

        if hs is None:
            print(f"WARNING: Failed to load hidden state from {path}: {last_error}")
            try:
                os.remove(path)
            except OSError:
                pass
            return None

        # Select target layer with configurable pooling.
        layer_hs = hs[self.target_layer]
        if layer_hs.ndim != 2:
            print(f"WARNING: unexpected hidden state shape at {path}: {layer_hs.shape}")
            result = np.zeros(self.hidden_size, dtype=np.float32)
        elif self.hidden_pooling == "last_token":
            result = layer_hs[-1, :].astype(np.float32)
        else:
            k = min(self.hidden_pooling_k, layer_hs.shape[0])
            result = layer_hs[-k:, :].mean(axis=0).astype(np.float32)

        # Clean up the file
        try:
            os.remove(path)
        except OSError:
            pass

        return result

    def extract_hidden_states_no_cot(
        self,
        observations: List[str],
        batch_size: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extract hidden states via the vLLM server plugin.
        
        Sends concurrent HTTP requests to the vLLM server, which extracts
        hidden states from specified layers and saves them to disk.
        
        API-compatible with LLMHiddenStateExtractor.extract_hidden_states_no_cot().
        
        Args:
            observations: List of filtered text observations
            batch_size: Number of concurrent requests per batch
            
        Returns:
            hidden_states: (N, hidden_size) last-token hidden states from target layer
            metrics: Timing and diagnostic metrics
        """
        if not observations:
            return np.array([]), {}

        start_time = time.perf_counter()
        
        # Format prompts using the tokenizer chat template
        prompts = [
            create_prompt(
                obs,
                self.tokenizer,
                system_prompt=self.system_prompt,
                prompt_variant=self.prompt_variant,
            )
            for obs in observations
        ]

        all_hidden = []
        batch_latencies = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_t0 = time.perf_counter()
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            
            # Send concurrent requests
            workers = min(self.max_workers, len(batch_prompts))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._send_request, prompt) for prompt in batch_prompts]
                results = []
                for fut in futures:
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        print(f"WARNING: request failed: {exc}")
                        results.append(None)
            
            # Extract hidden states from responses
            for result in results:
                if result is None:
                    all_hidden.append(np.zeros(self.hidden_size, dtype=np.float32))
                    continue
                hs_path = self._get_hidden_state_path(result)
                if hs_path is None:
                    self._missing_path_count += 1
                    if self._missing_path_count <= 5 or self._missing_path_count % 100 == 0:
                        completion_id = result.get("id", "<unknown>")
                        finish_reason = None
                        if result.get("choices"):
                            finish_reason = result["choices"][0].get("finish_reason")
                        print(
                            "WARNING: hidden_states_path missing "
                            f"(id={completion_id}, finish_reason={finish_reason})"
                        )
                hs = self._load_hidden_state(hs_path)
                if hs is not None:
                    all_hidden.append(hs)
                else:
                    # Fallback: zero vector (shouldn't happen in normal operation)
                    self._zero_vector_fallback_count += 1
                    if (
                        self._zero_vector_fallback_count <= 10
                        or self._zero_vector_fallback_count % 100 == 0
                    ):
                        print(
                            "WARNING: Failed to load hidden state, using zero vector "
                            f"(count={self._zero_vector_fallback_count})"
                        )
                    all_hidden.append(np.zeros(self.hidden_size, dtype=np.float32))

            batch_latencies.append(time.perf_counter() - batch_t0)

        elapsed = time.perf_counter() - start_time
        self.total_samples += len(observations)
        self.total_time += elapsed
        self.batch_times.extend(batch_latencies)

        all_hidden = np.stack(all_hidden, axis=0)  # (N, hidden_size)

        metrics = {
            "llm/inference_time_s": elapsed,
            "llm/samples_per_sec": len(observations) / elapsed,
            "llm/batch_latency_mean_s": float(np.mean(batch_latencies)),
            "llm/total_samples": self.total_samples,
            "llm/total_time_s": self.total_time,
        }

        hidden_diag = compute_hidden_state_diagnostics(all_hidden)
        metrics.update(hidden_diag)

        return all_hidden, metrics

    def extract_hidden_states(
        self,
        observations: List[str],
        batch_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """
        Generate text and extract hidden states via the vLLM server.

        Note: vLLM's hidden state plugin extracts states from the LAST token
        of the generation. So for generation mode, we generate N tokens and
        take the hidden state from the final token position.

        Args:
            observations: List of filtered text observations
            batch_size: Number of concurrent requests per batch
            max_new_tokens: Number of tokens to generate

        Returns:
            hidden_states: (N, hidden_size) last-token hidden states from target layer
            generated_texts: List of generated text strings
            metrics: Timing and diagnostic metrics
        """
        if not observations:
            return np.array([]), [], {}

        start_time = time.perf_counter()

        # Format prompts using the tokenizer chat template
        prompts = [
            create_prompt(
                obs,
                self.tokenizer,
                system_prompt=self.system_prompt,
                prompt_variant=self.prompt_variant,
            )
            for obs in observations
        ]
        if self.generation_prefix:
            prompts = [p + self.generation_prefix for p in prompts]

        all_hidden = []
        all_texts = []
        batch_latencies = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_t0 = time.perf_counter()
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            # Send concurrent requests with generation
            workers = min(self.max_workers, len(batch_prompts))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(self._send_request, prompt, max_new_tokens, temperature)
                    for prompt in batch_prompts
                ]
                results = []
                for fut in futures:
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        print(f"WARNING: generation request failed: {exc}")
                        results.append(None)

            # Extract hidden states and text from responses
            for result in results:
                if result is None:
                    all_hidden.append(np.zeros(self.hidden_size, dtype=np.float32))
                    all_texts.append("")
                    continue
                # Get hidden state
                hs_path = self._get_hidden_state_path(result)
                if hs_path is None:
                    self._missing_path_count += 1
                    if self._missing_path_count <= 5 or self._missing_path_count % 100 == 0:
                        completion_id = result.get("id", "<unknown>")
                        finish_reason = None
                        if result.get("choices"):
                            finish_reason = result["choices"][0].get("finish_reason")
                        print(
                            "WARNING: hidden_states_path missing "
                            f"(id={completion_id}, finish_reason={finish_reason})"
                        )
                hs = self._load_hidden_state(hs_path)
                if hs is not None:
                    all_hidden.append(hs)
                else:
                    # Fallback: zero vector (shouldn't happen in normal operation)
                    self._zero_vector_fallback_count += 1
                    if (
                        self._zero_vector_fallback_count <= 10
                        or self._zero_vector_fallback_count % 100 == 0
                    ):
                        print(
                            "WARNING: Failed to load hidden state, using zero vector "
                            f"(count={self._zero_vector_fallback_count})"
                        )
                    all_hidden.append(np.zeros(self.hidden_size, dtype=np.float32))

                # Get generated text
                if result.get("choices") and len(result["choices"]) > 0:
                    text = result["choices"][0].get("text", "")
                    all_texts.append(text)
                else:
                    all_texts.append("")

            batch_latencies.append(time.perf_counter() - batch_t0)

        elapsed = time.perf_counter() - start_time
        self.total_samples += len(observations)
        self.total_time += elapsed
        self.batch_times.extend(batch_latencies)

        all_hidden = np.stack(all_hidden, axis=0)  # (N, hidden_size)

        metrics = {
            "llm/inference_time_s": elapsed,
            "llm/samples_per_sec": len(observations) / elapsed,
            "llm/batch_latency_mean_s": float(np.mean(batch_latencies)),
            "llm/total_samples": self.total_samples,
            "llm/total_time_s": self.total_time,
            "llm/tokens_generated": max_new_tokens,
            "llm/temperature": float(temperature),
            "llm/hidden_pooling_k": float(self.hidden_pooling_k),
        }

        hidden_diag = compute_hidden_state_diagnostics(all_hidden)
        metrics.update(hidden_diag)

        return all_hidden, all_texts, metrics

    def get_metrics(self) -> Dict:
        return {
            "total_samples": self.total_samples,
            "total_time_s": self.total_time,
            "samples_per_sec": self.total_samples / max(0.001, self.total_time),
        }
