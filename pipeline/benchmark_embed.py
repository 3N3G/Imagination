#!/usr/bin/env python3
"""
Benchmark: HF+SDPA vs vLLM for extracting Qwen3-8B layer-30 hidden states.

Three approaches:
  1. HF full model (36 layers) + output_hidden_states → extract layer 30
  2. HF truncated model (31 layers) → last hidden state IS layer 30
  3. vLLM encode() with truncated model (31 layers) → mean-pooled last layer

Approach 2 skips 5 unnecessary layers and avoids storing all hidden states.
Approach 3 adds vLLM's PagedAttention + continuous batching on top.

Usage:
    python -m pipeline.benchmark_embed [--n-texts 200] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

EMBED_MODEL = "Qwen/Qwen3-8B"
EMBED_LAYER = 30  # 0-indexed, of 36 total
HIDDEN_DIM = 4096
GEMINI_DIR = Path("/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/gemini_labels")


def load_sample_texts(n: int) -> list[str]:
    """Load real Gemini label texts for benchmarking."""
    texts = []
    for jsonl in sorted(GEMINI_DIR.glob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("ok") and rec.get("text"):
                    texts.append(rec["text"])
                    if len(texts) >= n:
                        return texts
    if not texts:
        raise RuntimeError(f"No Gemini texts found in {GEMINI_DIR}")
    return texts


# ==========================================================================
# Approach 1: HF full model, extract layer 30 from output_hidden_states
# ==========================================================================
def bench_hf_full(texts: list[str], batch_size: int, device: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n--- [1] HF full model (36 layers) + output_hidden_states ---")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        EMBED_MODEL, dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    load_time = time.time() - t_load
    print(f"  Model loaded in {load_time:.1f}s ({model.config.num_hidden_layers} layers)")

    embeddings = np.empty((len(texts), HIDDEN_DIM), dtype=np.float16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=2048, padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model.model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
            layer_hs = out.hidden_states[EMBED_LAYER]

        mask = attention_mask.unsqueeze(-1).float()
        mean_pooled = ((layer_hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1)).half()
        embeddings[i:i + len(batch)] = mean_pooled.cpu().numpy()

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "method": "hf_full_36L",
        "load_time": load_time,
        "inference_time": elapsed,
        "texts_per_sec": len(texts) / elapsed,
        "peak_gpu_mb": peak_mb,
        "output_shape": embeddings.shape,
        "sample_norm": float(np.linalg.norm(embeddings[0].astype(np.float32))),
        "embeddings": embeddings,
    }


# ==========================================================================
# Approach 2: HF truncated model (31 layers), last hidden = layer 30
# ==========================================================================
def bench_hf_truncated(texts: list[str], batch_size: int, device: str) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("\n--- [2] HF truncated model (31 layers) → last hidden = layer 30 ---")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    config.num_hidden_layers = EMBED_LAYER + 1  # 31 layers: 0..30
    model = AutoModelForCausalLM.from_pretrained(
        EMBED_MODEL, config=config, dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    load_time = time.time() - t_load
    print(f"  Model loaded in {load_time:.1f}s ({model.config.num_hidden_layers} layers)")

    embeddings = np.empty((len(texts), HIDDEN_DIM), dtype=np.float16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=2048, padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            # No output_hidden_states needed — last layer IS layer 30
            out = model.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hs = out.last_hidden_state  # (B, seq, 4096)

        mask = attention_mask.unsqueeze(-1).float()
        mean_pooled = ((last_hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1)).half()
        embeddings[i:i + len(batch)] = mean_pooled.cpu().numpy()

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "method": "hf_truncated_31L",
        "load_time": load_time,
        "inference_time": elapsed,
        "texts_per_sec": len(texts) / elapsed,
        "peak_gpu_mb": peak_mb,
        "output_shape": embeddings.shape,
        "sample_norm": float(np.linalg.norm(embeddings[0].astype(np.float32))),
        "embeddings": embeddings,
    }


# ==========================================================================
# Approach 3: vLLM encode() with truncated model
# ==========================================================================
def bench_vllm(texts: list[str], batch_size: int) -> dict:
    print("\n--- [3] vLLM encode() with truncated model (31 layers) ---")

    try:
        from vllm import LLM, PoolingParams
        from vllm.config.pooler import PoolerConfig
    except ImportError:
        print("  SKIP: vllm not installed in this environment")
        return {"method": "vllm_truncated_31L", "skipped": True}

    import torch

    t_load = time.time()
    # Truncate to 31 layers so last hidden = layer 30.
    # Use pooler_config for mean pooling (matching HF approach).
    llm = LLM(
        model=EMBED_MODEL,
        dtype="half",
        max_model_len=2048,
        enforce_eager=True,  # avoid cuda graph overhead for benchmark
        hf_overrides={"num_hidden_layers": EMBED_LAYER + 1},
        pooler_config=PoolerConfig(pooling_type="MEAN"),
        trust_remote_code=True,
    )
    load_time = time.time() - t_load
    print(f"  Engine loaded in {load_time:.1f}s")

    pooling_params = PoolingParams()

    torch.cuda.synchronize()
    t_start = time.time()

    outputs = llm.encode(texts, pooling_params)

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    # Extract embeddings from outputs
    embeddings = np.empty((len(texts), HIDDEN_DIM), dtype=np.float16)
    for i, out in enumerate(outputs):
        vec = out.outputs.data
        if hasattr(vec, 'cpu'):
            vec = vec.cpu().float().numpy()
        else:
            vec = np.array(vec, dtype=np.float32)
        # vLLM encode may return different pooling — check shape
        if vec.ndim == 1 and vec.shape[0] == HIDDEN_DIM:
            embeddings[i] = vec.astype(np.float16)
        else:
            print(f"  WARNING: unexpected output shape {vec.shape} for sample {i}")
            embeddings[i] = 0

    del llm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "method": "vllm_truncated_31L",
        "load_time": load_time,
        "inference_time": elapsed,
        "texts_per_sec": len(texts) / elapsed,
        "peak_gpu_mb": peak_mb,
        "output_shape": embeddings.shape,
        "sample_norm": float(np.linalg.norm(embeddings[0].astype(np.float32))),
        "embeddings": embeddings,
    }


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-texts", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-vllm", action="store_true",
                        help="Skip vLLM benchmark (e.g. if not installed)")
    args = parser.parse_args()

    print("=" * 70)
    print("Embedding Benchmark: HF+SDPA vs vLLM")
    print("=" * 70)
    print(f"  Model: {EMBED_MODEL}")
    print(f"  Target layer: {EMBED_LAYER} (of 36)")
    print(f"  Texts: {args.n_texts}")
    print(f"  Batch size: {args.batch_size}")

    texts = load_sample_texts(args.n_texts)
    print(f"  Loaded {len(texts)} texts")
    avg_len = np.mean([len(t.split()) for t in texts])
    print(f"  Avg text length: {avg_len:.0f} words")

    results = []

    # Run benchmarks
    r1 = bench_hf_full(texts, args.batch_size, args.device)
    results.append(r1)

    r2 = bench_hf_truncated(texts, args.batch_size, args.device)
    results.append(r2)

    if not args.skip_vllm:
        r3 = bench_vllm(texts, args.batch_size)
        results.append(r3)

    # === Performance summary ===
    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)
    print(f"{'Method':<25} {'Load(s)':>8} {'Infer(s)':>9} {'Texts/s':>9} {'GPU MB':>9}")
    print("-" * 70)
    for r in results:
        if r.get("skipped"):
            print(f"{r['method']:<25} {'SKIPPED':>8}")
            continue
        print(f"{r['method']:<25} {r['load_time']:>8.1f} {r['inference_time']:>9.1f} "
              f"{r['texts_per_sec']:>9.1f} {r['peak_gpu_mb']:>9.0f}")

    # === Correctness verification ===
    print("\n" + "=" * 70)
    print("CORRECTNESS")
    print("=" * 70)

    all_passed = True

    def check_basic(r: dict) -> bool:
        """Per-method sanity checks."""
        name = r["method"]
        ok = True
        emb = r["embeddings"].astype(np.float32)

        # Shape
        expected = (len(texts), HIDDEN_DIM)
        if emb.shape != expected:
            print(f"  FAIL [{name}] shape {emb.shape} != expected {expected}")
            ok = False
        else:
            print(f"  OK   [{name}] shape = {emb.shape}")

        # NaN / Inf
        n_nan = int(np.isnan(emb).sum())
        n_inf = int(np.isinf(emb).sum())
        if n_nan > 0 or n_inf > 0:
            print(f"  FAIL [{name}] NaN={n_nan}, Inf={n_inf}")
            ok = False
        else:
            print(f"  OK   [{name}] no NaN/Inf")

        # Norms (should be nonzero and finite)
        norms = np.linalg.norm(emb, axis=1)
        if np.any(norms < 1e-6):
            n_zero = int(np.sum(norms < 1e-6))
            print(f"  FAIL [{name}] {n_zero}/{len(norms)} near-zero norm vectors")
            ok = False
        else:
            print(f"  OK   [{name}] norms: min={norms.min():.2f} mean={norms.mean():.2f} "
                  f"max={norms.max():.2f}")

        # Check not all identical (degenerate model)
        if np.allclose(emb[0], emb[1]):
            print(f"  FAIL [{name}] first two embeddings are identical (degenerate)")
            ok = False
        else:
            self_cos = np.dot(emb[0], emb[1]) / (norms[0] * norms[1] + 1e-8)
            print(f"  OK   [{name}] emb[0] vs emb[1] cosine = {self_cos:.4f} (not degenerate)")

        return ok

    def compare_pair(emb_a: np.ndarray, emb_b: np.ndarray,
                     name_a: str, name_b: str,
                     expect_exact: bool) -> bool:
        """Compare two embedding matrices element-wise and per-sample."""
        a = emb_a.astype(np.float32)
        b = emb_b.astype(np.float32)
        ok = True
        label = f"{name_a} vs {name_b}"

        # Per-sample cosine similarity
        norms_a = np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)
        norms_b = np.linalg.norm(b, axis=1, keepdims=True).clip(1e-8)
        cos_sims = np.sum((a / norms_a) * (b / norms_b), axis=1)

        print(f"\n  --- {label} ---")
        print(f"  Cosine sim: min={cos_sims.min():.6f} mean={cos_sims.mean():.6f} "
              f"max={cos_sims.max():.6f} std={cos_sims.std():.6f}")

        # Element-wise absolute difference
        abs_diff = np.abs(a - b)
        print(f"  Abs diff:   mean={abs_diff.mean():.6e} max={abs_diff.max():.6e} "
              f"median={np.median(abs_diff):.6e}")

        # Relative difference (where values are non-negligible)
        mask = np.abs(a) > 1e-4
        if mask.any():
            rel_diff = abs_diff[mask] / np.abs(a[mask])
            print(f"  Rel diff:   mean={rel_diff.mean():.6e} max={rel_diff.max():.6e} "
                  f"median={np.median(rel_diff):.6e}")

        # Find worst sample
        worst_idx = int(cos_sims.argmin())
        print(f"  Worst sample: idx={worst_idx}, cosine={cos_sims[worst_idx]:.6f}, "
              f"max_abs_diff={abs_diff[worst_idx].max():.6e}")

        if expect_exact:
            # HF full vs HF truncated: same layers, should be identical
            # Allow tiny fp16 rounding (values are stored as fp16)
            threshold = 0.999999
            if cos_sims.min() < threshold:
                print(f"  FAIL: expected near-identical (min cosine >= {threshold}), "
                      f"got {cos_sims.min():.6f}")
                ok = False
            else:
                print(f"  OK:   near-identical as expected (min cosine = {cos_sims.min():.6f})")

            # Also check all-close in absolute terms
            if not np.allclose(a, b, atol=1e-2, rtol=1e-3):
                n_mismatch = int(np.sum(~np.isclose(a, b, atol=1e-2, rtol=1e-3)))
                total = a.size
                print(f"  WARN: {n_mismatch}/{total} elements not allclose "
                      f"(atol=1e-2, rtol=1e-3)")
            else:
                print(f"  OK:   np.allclose passed (atol=1e-2, rtol=1e-3)")
        else:
            # HF vs vLLM: allow numerical differences from different kernels
            # but embeddings should be highly correlated
            if cos_sims.mean() < 0.95:
                print(f"  FAIL: mean cosine {cos_sims.mean():.4f} < 0.95 — "
                      f"likely different computation (wrong layer? different pooling?)")
                ok = False
            elif cos_sims.mean() < 0.99:
                print(f"  WARN: mean cosine {cos_sims.mean():.4f} — moderate difference, "
                      f"check pooling strategy")
            else:
                print(f"  OK:   high agreement (mean cosine = {cos_sims.mean():.6f})")

            # If cosine is low, check if it's a pooling mismatch
            if cos_sims.mean() < 0.95:
                print(f"  INFO: vLLM likely uses last-token pooling vs our mean pooling.")
                print(f"        This is expected — outputs represent the same layer")
                print(f"        but pool differently across the sequence.")

        return ok

    # Basic checks for each method
    for r in results:
        if r.get("skipped"):
            continue
        if not check_basic(r):
            all_passed = False

    # Pairwise comparisons
    if "embeddings" in r1 and "embeddings" in r2:
        if not compare_pair(r1["embeddings"], r2["embeddings"],
                            "hf_full_36L", "hf_truncated_31L",
                            expect_exact=True):
            all_passed = False

    for r in results:
        if r.get("skipped") or "embeddings" not in r:
            continue
        if r["method"].startswith("vllm") and "embeddings" in r2:
            if not compare_pair(r2["embeddings"], r["embeddings"],
                                "hf_truncated_31L", "vllm_truncated_31L",
                                expect_exact=False):
                all_passed = False

    # Final verdict
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CORRECTNESS CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see details above")
    print("=" * 70)


if __name__ == "__main__":
    main()
