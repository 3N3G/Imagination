#!/usr/bin/env python3
"""
LLM Inference Benchmarking for Craftax

Compares throughput and latency across different inference backends:
1. HuggingFace Transformers (baseline - current llm_worker.py approach)
2. vLLM (offline batch mode)
3. SGLang (server mode with RadixAttention)

Usage:
    python benchmark_inference.py --backend huggingface --samples 100
    python benchmark_inference.py --backend vllm --samples 100
    python benchmark_inference.py --backend sglang --samples 100 --server-url http://localhost:30000

Output: Tokens/sec, samples/sec, latency distribution
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import os

# Prevent JAX from hogging GPU memory
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# Sample prompts for benchmarking (matching llm_worker.py format)
SYSTEM_PROMPT = """You are playing Craftax.

Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions and can interact.

Actions available: 
0:NOOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:DO (interact/mine/attack), 6:SLEEP, 7:PLACE_STONE,
8:PLACE_TABLE, 9:PLACE_FURNACE, 10:PLACE_PLANT, 11:MAKE_WOOD_PICKAXE, 12:MAKE_STONE_PICKAXE,
13:MAKE_IRON_PICKAXE, 14:MAKE_WOOD_SWORD, 15:MAKE_STONE_SWORD, 16:MAKE_IRON_SWORD, 17:REST,
18:DESCEND, 19:ASCEND, 20:MAKE_DIAMOND_PICKAXE, 21:MAKE_DIAMOND_SWORD, 22:MAKE_IRON_PICKAXE

Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>).
"""

# Realistic sample observations for benchmarking
SAMPLE_OBS = [
    "Map (interesting tiles only): -1,0:tree, 1,0:stone, 2,3:water\nInventory: Wood:0, Stone:0\nHealth: 9.0\nFood: 9\nDrink: 9\nEnergy: 9\nDirection: right",
    "Map (interesting tiles only): 0,-1:crafting_table, 1,2:tree, -2,0:Cow on grass\nInventory: Wood:5, Stone:3\nHealth: 7.0\nFood: 6\nDrink: 8\nEnergy: 9\nDirection: down",
    "Map (interesting tiles only): 0,1:Skeleton on grass, 2,0:stone, -1,-2:tree\nInventory: Wood:2, Stone:8, Iron Sword\nHealth: 5.0\nFood: 4\nDrink: 3\nEnergy: 7\nDirection: right",
    "Map (interesting tiles only): -1,0:ladder_down, 0,2:Orc Soldier on path\nInventory: Wood:10, Stone:20, Coal:5\nHealth: 8.0\nFood: 9\nDrink: 9\nEnergy: 9\nDirection: up",
    "Map: [No interesting tiles in view - all background]\nInventory: Wood:0\nHealth: 9.0\nFood: 9\nDrink: 9\nEnergy: 9\nDirection: left",
]


def create_prompts(num_samples: int) -> List[str]:
    """Create benchmark prompts cycling through sample observations."""
    prompts = []
    for i in range(num_samples):
        obs = SAMPLE_OBS[i % len(SAMPLE_OBS)]
        prompt = f"YOUR CURRENT GAME STATE:\n{obs}\n\nYou are at (0,0). Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>)."
        prompts.append(prompt)
    return prompts


# ============================================================================
# Backend: HuggingFace Transformers
# ============================================================================
def benchmark_huggingface(model_id: str, prompts: List[str], max_new_tokens: int, batch_size: int) -> Dict[str, Any]:
    """Benchmark HuggingFace Transformers (matching llm_worker.py approach)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"Loading model {model_id} with HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Apply chat template
    formatted_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)
    
    # Benchmark in batches (matching llm_worker.py)
    total_tokens = 0
    latencies = []
    
    print(f"Running {len(prompts)} samples with batch_size={batch_size}...")
    start_total = time.perf_counter()
    
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(model.device)
        
        prompt_len = inputs['input_ids'].shape[1]
        
        start_batch = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        end_batch = time.perf_counter()
        
        batch_tokens = (outputs.shape[1] - prompt_len) * outputs.shape[0]
        total_tokens += batch_tokens
        latencies.append((end_batch - start_batch) / len(batch))
        
        print(f"  Batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}: {batch_tokens} tokens, {end_batch - start_batch:.2f}s")
    
    end_total = time.perf_counter()
    
    return {
        "backend": "huggingface",
        "total_time_s": end_total - start_total,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / (end_total - start_total),
        "samples_per_sec": len(prompts) / (end_total - start_total),
        "mean_latency_s": np.mean(latencies),
        "p50_latency_s": np.percentile(latencies, 50),
        "p95_latency_s": np.percentile(latencies, 95),
        "p99_latency_s": np.percentile(latencies, 99),
    }


# ============================================================================
# Backend: vLLM
# ============================================================================
def benchmark_vllm(model_id: str, prompts: List[str], max_new_tokens: int, batch_size: int) -> Dict[str, Any]:
    """Benchmark vLLM offline batch inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return {"error": "vLLM not installed. Run: pip install vllm"}
    
    print(f"Loading model {model_id} with vLLM...")
    llm = LLM(model=model_id, trust_remote_code=True, dtype="float16", max_model_len=4096, gpu_memory_utilization=0.9)
    
    # vLLM handles chat formatting differently
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    formatted_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_new_tokens,
    )
    
    print(f"Running {len(prompts)} samples with vLLM (batched)...")
    start_total = time.perf_counter()
    
    # vLLM processes all prompts in an optimized batch
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    end_total = time.perf_counter()
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    # Calculate per-sample latency (approximate - vLLM batches internally)
    latency_per_sample = (end_total - start_total) / len(prompts)
    
    return {
        "backend": "vllm",
        "total_time_s": end_total - start_total,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / (end_total - start_total),
        "samples_per_sec": len(prompts) / (end_total - start_total),
        "mean_latency_s": latency_per_sample,
        "p50_latency_s": latency_per_sample,  # Approximate
        "p95_latency_s": latency_per_sample,  # Approximate
        "p99_latency_s": latency_per_sample,  # Approximate
    }


# ============================================================================
# Backend: SGLang
# ============================================================================
def benchmark_sglang(model_id: str, prompts: List[str], max_new_tokens: int, server_url: str) -> Dict[str, Any]:
    """Benchmark SGLang server mode (requires running server)."""
    try:
        import requests
    except ImportError:
        return {"error": "requests not installed. Run: pip install requests"}
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    formatted_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)
    
    # Test server connectivity
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.status_code != 200:
            return {"error": f"SGLang server not healthy at {server_url}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Cannot connect to SGLang server at {server_url}: {e}"}
    
    print(f"Running {len(prompts)} samples with SGLang server at {server_url}...")
    
    latencies = []
    total_tokens = 0
    start_total = time.perf_counter()
    
    for i, prompt in enumerate(formatted_prompts):
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": max_new_tokens,
            }
        }
        
        start_req = time.perf_counter()
        response = requests.post(f"{server_url}/generate", json=payload)
        end_req = time.perf_counter()
        
        if response.status_code == 200:
            result = response.json()
            tokens = len(tokenizer.encode(result.get("text", "")))
            total_tokens += tokens
            latencies.append(end_req - start_req)
        else:
            print(f"  Request {i} failed: {response.status_code}")
            latencies.append(float('inf'))
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(prompts)}")
    
    end_total = time.perf_counter()
    
    valid_latencies = [l for l in latencies if l != float('inf')]
    
    return {
        "backend": "sglang",
        "total_time_s": end_total - start_total,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / (end_total - start_total) if total_tokens > 0 else 0,
        "samples_per_sec": len(prompts) / (end_total - start_total),
        "mean_latency_s": np.mean(valid_latencies) if valid_latencies else float('inf'),
        "p50_latency_s": np.percentile(valid_latencies, 50) if valid_latencies else float('inf'),
        "p95_latency_s": np.percentile(valid_latencies, 95) if valid_latencies else float('inf'),
        "p99_latency_s": np.percentile(valid_latencies, 99) if valid_latencies else float('inf'),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference backends")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model ID to benchmark")
    parser.add_argument("--backend", type=str, choices=["huggingface", "vllm", "sglang", "all"],
                        default="huggingface", help="Backend to benchmark")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to benchmark")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per sample")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (for HF/vLLM)")
    parser.add_argument("--server-url", type=str, default="http://localhost:30000",
                        help="SGLang server URL")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    args = parser.parse_args()
    
    prompts = create_prompts(args.samples)
    results = {}
    
    print("=" * 60)
    print("LLM Inference Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples: {args.samples}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    if args.backend in ["huggingface", "all"]:
        print("\n[1/3] Benchmarking HuggingFace Transformers...")
        results["huggingface"] = benchmark_huggingface(
            args.model, prompts, args.max_tokens, args.batch_size
        )
        print(f"\nHuggingFace Results:")
        for k, v in results["huggingface"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    if args.backend in ["vllm", "all"]:
        print("\n[2/3] Benchmarking vLLM...")
        results["vllm"] = benchmark_vllm(
            args.model, prompts, args.max_tokens, args.batch_size
        )
        print(f"\nvLLM Results:")
        for k, v in results["vllm"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    if args.backend in ["sglang", "all"]:
        print("\n[3/3] Benchmarking SGLang...")
        results["sglang"] = benchmark_sglang(
            args.model, prompts, args.max_tokens, args.server_url
        )
        print(f"\nSGLang Results:")
        for k, v in results["sglang"].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        baseline = results.get("huggingface", {}).get("samples_per_sec", 1)
        for backend, r in results.items():
            if "error" not in r:
                speedup = r["samples_per_sec"] / baseline if baseline > 0 else float('inf')
                print(f"{backend:15} | {r['samples_per_sec']:.2f} samples/s | {r['tokens_per_sec']:.0f} tok/s | {speedup:.1f}x speedup")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
