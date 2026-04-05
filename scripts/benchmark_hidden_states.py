#!/usr/bin/env python3
"""
Benchmark Hidden State Extraction Methods

Compares different approaches for getting hidden states during LLM inference:
1. HuggingFace with output_hidden_states=True (baseline, like llm_worker.py)
2. SGLang (if it supports hidden state extraction)
3. vLLM generate + HuggingFace forward pass (hybrid)
4. vLLM generate + HuggingFace encode-only (just prefill hidden states)

All methods must return hidden states for 256 generated tokens.

Usage:
    python benchmark_hidden_states.py --backend huggingface --samples 10
    python benchmark_hidden_states.py --backend all --samples 10
"""

import argparse
import time
import gc
import os
from typing import Dict, Any, List, Optional
import numpy as np

# Prevent conflicts
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
TOKENS_GENERATED = 256  # Must match llm_worker.py
MAX_PROMPT_LEN = 2048

# Sample prompts (simplified Craftax observations)
SAMPLE_PROMPTS = [
    "Map (interesting tiles only): 0,1:tree, -1,0:stone\nInventory: Wood:0\nHealth:9\nDirection:right\n\nWhat action should you take?",
    "Map (interesting tiles only): 1,0:water, 0,-1:Cow on grass\nInventory: Wood:5\nHealth:7\nDirection:down\n\nWhat action should you take?",
    "Map (interesting tiles only): -1,-1:Skeleton on grass\nInventory: Wood:3, Stone:2\nHealth:5\nDirection:up\n\nWhat action should you take?",
    "Map: [No interesting tiles in view]\nInventory: Wood:10, Stone:5\nHealth:9\nDirection:left\n\nWhat action should you take?",
]


def get_test_prompts(n: int) -> List[str]:
    """Generate n test prompts by cycling through samples."""
    prompts = []
    for i in range(n):
        prompts.append(SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)])
    return prompts


# =============================================================================
# Method 1: HuggingFace (Baseline)
# =============================================================================

def benchmark_huggingface(
    model_id: str,
    prompts: List[str],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Benchmark HuggingFace with output_hidden_states=True.
    This is the baseline method used in llm_worker.py.
    """
    print(f"\n{'='*60}")
    print("Method 1: HuggingFace (baseline)")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model: {model_id}")
    load_start = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")
    
    hidden_size = model.config.hidden_size
    print(f"Hidden size: {hidden_size}")
    
    # Process in batches
    all_hidden_states = []
    all_outputs = []
    latencies = []
    
    total_start = time.perf_counter()
    
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        current_batch_size = len(batch_prompts)
        
        batch_start = time.perf_counter()
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN,
        ).to(model.device)
        
        prompt_len = inputs['input_ids'].shape[1]
        
        # Generate with hidden states
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=TOKENS_GENERATED,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=0.7,
            )
        
        # Extract hidden states from last layer
        # outputs.hidden_states is tuple of (step, layer, batch, seq, hidden)
        last_layer_states = [s[-1] for s in outputs.hidden_states]
        generated_hidden = torch.cat(last_layer_states, dim=1)
        
        # Truncate/pad to TOKENS_GENERATED
        seq_len = generated_hidden.shape[1]
        if seq_len > TOKENS_GENERATED:
            generated_hidden = generated_hidden[:, :TOKENS_GENERATED, :]
        elif seq_len < TOKENS_GENERATED:
            padding = torch.zeros(
                (current_batch_size, TOKENS_GENERATED - seq_len, hidden_size),
                device=generated_hidden.device,
                dtype=generated_hidden.dtype,
            )
            generated_hidden = torch.cat([generated_hidden, padding], dim=1)
        
        # Mean pool to get (batch, hidden_size)
        pooled = generated_hidden.mean(dim=1).cpu().numpy()
        all_hidden_states.append(pooled)
        
        # Decode text
        generated_ids = outputs.sequences[:, prompt_len:]
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_outputs.extend(texts)
        
        batch_time = time.perf_counter() - batch_start
        latencies.append(batch_time)
        
        print(f"  Batch {batch_idx//batch_size + 1}: {batch_time:.2f}s "
              f"({current_batch_size/batch_time:.2f} samples/sec)")
    
    total_time = time.perf_counter() - total_start
    
    # Combine results
    all_hidden_states = np.concatenate(all_hidden_states, axis=0)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "backend": "huggingface",
        "total_time_s": total_time,
        "samples_per_sec": len(prompts) / total_time,
        "mean_latency_s": np.mean(latencies),
        "hidden_shape": all_hidden_states.shape,
        "hidden_size": hidden_size,
        "tokens_generated": TOKENS_GENERATED,
        "num_samples": len(prompts),
        "sample_output": all_outputs[0][:200] if all_outputs else "",
    }


# =============================================================================
# Method 2: SGLang (check if hidden states supported)
# =============================================================================

def benchmark_sglang(
    model_id: str,
    prompts: List[str],
    server_url: str = "http://localhost:30000",
) -> Dict[str, Any]:
    """
    Test if SGLang supports hidden state extraction.
    """
    print(f"\n{'='*60}")
    print("Method 2: SGLang")
    print(f"{'='*60}")
    
    try:
        import requests
    except ImportError:
        return {"backend": "sglang", "error": "requests not installed"}
    
    # Check server health
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.status_code != 200:
            return {"backend": "sglang", "error": f"Server not healthy: {health.status_code}"}
    except Exception as e:
        return {"backend": "sglang", "error": f"Cannot connect to server: {e}"}
    
    print(f"Server is healthy at {server_url}")
    
    # Check if hidden states endpoint exists
    # SGLang may have /v1/embeddings for pooling models
    try:
        # Try embeddings endpoint
        embed_payload = {
            "input": prompts[0],
            "model": model_id,
        }
        embed_resp = requests.post(f"{server_url}/v1/embeddings", json=embed_payload, timeout=30)
        
        if embed_resp.status_code == 200:
            result = embed_resp.json()
            print(f"SGLang embeddings endpoint works!")
            print(f"Response keys: {result.keys() if isinstance(result, dict) else type(result)}")
            
            # Check if this gives us what we need
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0].get("embedding", [])
                return {
                    "backend": "sglang",
                    "supports_embeddings": True,
                    "embedding_dim": len(embedding) if embedding else 0,
                    "note": "SGLang supports embeddings but NOT per-token hidden states during generation",
                }
        else:
            print(f"Embeddings endpoint returned: {embed_resp.status_code}")
    except Exception as e:
        print(f"Embeddings endpoint error: {e}")
    
    # Try generate with hidden states request (unlikely to work)
    try:
        gen_payload = {
            "text": prompts[0],
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": TOKENS_GENERATED,
            },
            "return_hidden_states": True,  # This likely won't exist
        }
        gen_resp = requests.post(f"{server_url}/generate", json=gen_payload, timeout=60)
        result = gen_resp.json()
        
        if "hidden_states" in result:
            return {
                "backend": "sglang",
                "supports_hidden_states": True,
                "hidden_shape": np.array(result["hidden_states"]).shape,
            }
        else:
            return {
                "backend": "sglang",
                "supports_hidden_states": False,
                "note": "SGLang does not support hidden states during generation. Use HuggingFace.",
                "available_keys": list(result.keys()) if isinstance(result, dict) else str(type(result)),
            }
    except Exception as e:
        return {
            "backend": "sglang",
            "supports_hidden_states": False,
            "error": str(e),
            "note": "SGLang does not support hidden states during generation",
        }


# =============================================================================
# Method 3: vLLM Generate + HuggingFace Forward (Hybrid)
# =============================================================================

def benchmark_vllm_hf_hybrid(
    model_id: str,
    prompts: List[str],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Hybrid approach:
    1. Use vLLM to generate text quickly (256 tokens)
    2. Use HuggingFace to do a forward pass on prompt+generation to get hidden states
    
    This gets the speed of vLLM generation + hidden states from HF.
    """
    print(f"\n{'='*60}")
    print("Method 3: vLLM Generate + HuggingFace Forward (Hybrid)")
    print(f"{'='*60}")
    
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return {"backend": "vllm_hf_hybrid", "error": "vLLM not installed"}
    
    # Phase 1: vLLM generation
    print("Phase 1: vLLM text generation...")
    vllm_start = time.perf_counter()
    
    llm = LLM(
        model=model_id,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.5,  # Leave room for HF model
        max_model_len=4096,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=TOKENS_GENERATED,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [o.outputs[0].text for o in outputs]
    
    vllm_time = time.perf_counter() - vllm_start
    print(f"  vLLM generation: {vllm_time:.2f}s ({len(prompts)/vllm_time:.2f} samples/sec)")
    
    # Cleanup vLLM
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # Phase 2: HuggingFace forward pass for hidden states
    print("Phase 2: HuggingFace forward pass for hidden states...")
    hf_start = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    hidden_size = model.config.hidden_size
    all_hidden_states = []
    
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        batch_texts = generated_texts[batch_idx:batch_idx + batch_size]
        
        # Combine prompt + generated text
        full_texts = [p + t for p, t in zip(batch_prompts, batch_texts)]
        
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN + TOKENS_GENERATED,
        ).to(model.device)
        
        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Get last layer hidden states
        # outputs.hidden_states[-1] shape: (batch, seq_len, hidden_size)
        last_hidden = outputs.hidden_states[-1]
        
        # Take the last TOKENS_GENERATED tokens (the generated part)
        # and mean pool
        gen_hidden = last_hidden[:, -TOKENS_GENERATED:, :]
        pooled = gen_hidden.mean(dim=1).cpu().numpy()
        all_hidden_states.append(pooled)
    
    hf_time = time.perf_counter() - hf_start
    print(f"  HuggingFace forward: {hf_time:.2f}s")
    
    total_time = vllm_time + hf_time
    
    all_hidden_states = np.concatenate(all_hidden_states, axis=0)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "backend": "vllm_hf_hybrid",
        "total_time_s": total_time,
        "vllm_gen_time_s": vllm_time,
        "hf_forward_time_s": hf_time,
        "samples_per_sec": len(prompts) / total_time,
        "hidden_shape": all_hidden_states.shape,
        "hidden_size": hidden_size,
        "tokens_generated": TOKENS_GENERATED,
        "num_samples": len(prompts),
        "note": "vLLM generates text, HF does forward pass to get hidden states",
    }


# =============================================================================
# Method 4: HuggingFace with Flash Attention (optimized baseline)
# =============================================================================

def benchmark_huggingface_flash(
    model_id: str,
    prompts: List[str],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    HuggingFace with Flash Attention 2 enabled for faster inference.
    """
    print(f"\n{'='*60}")
    print("Method 4: HuggingFace + Flash Attention 2")
    print(f"{'='*60}")
    
    # Load model with flash attention
    print(f"Loading model with flash_attention_2: {model_id}")
    load_start = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        flash_enabled = True
    except Exception as e:
        print(f"Flash Attention 2 not available: {e}")
        print("Falling back to default attention")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        flash_enabled = False
    
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s (flash_attention: {flash_enabled})")
    
    hidden_size = model.config.hidden_size
    
    # Process in batches
    all_hidden_states = []
    latencies = []
    
    total_start = time.perf_counter()
    
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        current_batch_size = len(batch_prompts)
        
        batch_start = time.perf_counter()
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_LEN,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=TOKENS_GENERATED,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=0.7,
            )
        
        last_layer_states = [s[-1] for s in outputs.hidden_states]
        generated_hidden = torch.cat(last_layer_states, dim=1)
        
        seq_len = generated_hidden.shape[1]
        if seq_len > TOKENS_GENERATED:
            generated_hidden = generated_hidden[:, :TOKENS_GENERATED, :]
        elif seq_len < TOKENS_GENERATED:
            padding = torch.zeros(
                (current_batch_size, TOKENS_GENERATED - seq_len, hidden_size),
                device=generated_hidden.device,
                dtype=generated_hidden.dtype,
            )
            generated_hidden = torch.cat([generated_hidden, padding], dim=1)
        
        pooled = generated_hidden.mean(dim=1).cpu().numpy()
        all_hidden_states.append(pooled)
        
        batch_time = time.perf_counter() - batch_start
        latencies.append(batch_time)
        
        print(f"  Batch {batch_idx//batch_size + 1}: {batch_time:.2f}s "
              f"({current_batch_size/batch_time:.2f} samples/sec)")
    
    total_time = time.perf_counter() - total_start
    all_hidden_states = np.concatenate(all_hidden_states, axis=0)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "backend": "huggingface_flash",
        "flash_attention_enabled": flash_enabled,
        "total_time_s": total_time,
        "samples_per_sec": len(prompts) / total_time,
        "mean_latency_s": np.mean(latencies),
        "hidden_shape": all_hidden_states.shape,
        "hidden_size": hidden_size,
        "tokens_generated": TOKENS_GENERATED,
        "num_samples": len(prompts),
    }


# =============================================================================
# Main
# =============================================================================

def print_results(results: List[Dict[str, Any]]):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Backend':<25} {'Time (s)':<12} {'Samples/sec':<12} {'Hidden Shape':<20}")
    print("-" * 70)
    
    for r in results:
        if "error" in r:
            print(f"{r['backend']:<25} ERROR: {r['error']}")
        elif "supports_hidden_states" in r and not r.get("supports_hidden_states", True):
            print(f"{r['backend']:<25} NOT SUPPORTED: {r.get('note', 'N/A')}")
        else:
            time_s = r.get("total_time_s", 0)
            sps = r.get("samples_per_sec", 0)
            shape = str(r.get("hidden_shape", "N/A"))
            print(f"{r['backend']:<25} {time_s:<12.2f} {sps:<12.2f} {shape:<20}")
    
    print("\n" + "=" * 70)
    
    # Find best
    valid_results = [r for r in results if "samples_per_sec" in r and r["samples_per_sec"] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x["samples_per_sec"])
        print(f"\n✅ BEST: {best['backend']} with {best['samples_per_sec']:.2f} samples/sec")
        
        # Training time estimates
        sps = best["samples_per_sec"]
        print(f"\n📊 Training Time Estimates (at {sps:.1f} samples/sec):")
        for steps, label in [(100_000, "100K"), (500_000, "500K"), (1_000_000, "1M")]:
            hours = steps / sps / 3600
            days = hours / 24
            print(f"   {label}: {hours:.1f} hours ({days:.2f} days)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark hidden state extraction methods")
    parser.add_argument("--backend", type=str, default="huggingface",
                        choices=["huggingface", "sglang", "vllm_hf_hybrid", "huggingface_flash", "all"],
                        help="Backend to benchmark")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID")
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000",
                        help="SGLang server URL")
    args = parser.parse_args()
    
    prompts = get_test_prompts(args.samples)
    print(f"Testing with {len(prompts)} samples, batch_size={args.batch_size}")
    print(f"Model: {args.model}")
    print(f"Tokens to generate: {TOKENS_GENERATED}")
    
    results = []
    
    if args.backend in ["huggingface", "all"]:
        result = benchmark_huggingface(args.model, prompts, args.batch_size)
        results.append(result)
    
    if args.backend in ["huggingface_flash", "all"]:
        result = benchmark_huggingface_flash(args.model, prompts, args.batch_size)
        results.append(result)
    
    if args.backend in ["sglang", "all"]:
        result = benchmark_sglang(args.model, prompts, args.sglang_url)
        results.append(result)
    
    if args.backend in ["vllm_hf_hybrid", "all"]:
        result = benchmark_vllm_hf_hybrid(args.model, prompts, args.batch_size)
        results.append(result)
    
    print_results(results)
    
    # Save results
    import json
    output_file = f"hidden_state_benchmark_{args.backend}_{args.samples}samples.json"
    with open(output_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in results:
            r_copy = r.copy()
            if "hidden_shape" in r_copy:
                r_copy["hidden_shape"] = list(r_copy["hidden_shape"])
            json_results.append(r_copy)
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
