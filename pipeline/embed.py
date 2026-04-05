#!/usr/bin/env python3
"""
Phase 5: Extract mean-pooled Qwen3-8B embeddings from Gemini oracle texts.

Extracts layer-30 hidden states and mean-pools across all tokens to produce
a single (4096,) float16 vector per sample.

Supports two backends:
  --backend hf      : HuggingFace transformers (default, reliable)
  --backend vllm    : vLLM extract_hidden_states (faster, needs vllm >=0.15)

Usage:
    python -m pipeline.embed [--batch-size 16] [--backend hf] [--max-files N] [--file-offset K]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline.config import (
    EMBED_HIDDEN_DIM,
    EMBED_LAYER,
    EMBED_MODEL,
    EMBED_OUTPUT_DIR,
    GEMINI_OUTPUT_DIR,
)


def load_gemini_texts(jsonl_path: Path) -> Dict[int, str]:
    """Load completed Gemini texts from a JSONL file."""
    if not jsonl_path.exists():
        return {}
    texts = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ok") and rec.get("text"):
                    texts[rec["sample_idx"]] = rec["text"]
            except (json.JSONDecodeError, KeyError):
                continue
    return texts


# ---------------------------------------------------------------------------
# Backend: HuggingFace transformers
# ---------------------------------------------------------------------------

def _extract_hf(
    texts: List[str],
    model,
    tokenizer,
    extract_layer: int,
    device: str,
    max_length: int = 2048,
) -> np.ndarray:
    """Mean-pooled hidden states via HF transformers. Returns (N, hidden_dim) float16."""
    import torch

    batch_size = len(texts)
    hidden_dim = model.config.hidden_size
    result = np.empty((batch_size, hidden_dim), dtype=np.float16)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        layer_hidden = outputs.hidden_states[extract_layer]  # (B, seq_len, H)

    # Mean pool over non-padding tokens
    mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
    summed = (layer_hidden.float() * mask).sum(dim=1)  # (B, H)
    lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
    mean_pooled = (summed / lengths).half()  # (B, H)

    result = mean_pooled.cpu().numpy()
    return result


def _run_hf(
    all_texts: List[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Load HF model and extract all embeddings."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading HF model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        EMBED_MODEL,
        dtype=torch.float16,
        device_map=device if device != "cpu" else "cpu",
        attn_implementation="sdpa" if device != "cpu" else "eager",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"    Hidden size: {model.config.hidden_size}")
    print(f"    Num layers: {model.config.num_hidden_layers}")

    if model.config.hidden_size != EMBED_HIDDEN_DIM:
        raise RuntimeError(
            f"Model hidden_size={model.config.hidden_size} != "
            f"EMBED_HIDDEN_DIM={EMBED_HIDDEN_DIM}"
        )

    all_embeddings = np.empty((len(all_texts), EMBED_HIDDEN_DIM), dtype=np.float16)
    t0 = time.time()

    for batch_start in range(0, len(all_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(all_texts))
        batch_texts = all_texts[batch_start:batch_end]

        batch_emb = _extract_hf(
            batch_texts, model, tokenizer,
            extract_layer=EMBED_LAYER, device=device,
        )
        all_embeddings[batch_start:batch_end] = batch_emb

        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 0
        remaining = len(all_texts) - batch_end
        eta = remaining / rate if rate > 0 else 0
        print(f"    {batch_end}/{len(all_texts)} done, "
              f"{rate:.1f} texts/s, ETA {eta/60:.1f}min")

    del model
    if device != "cpu":
        import torch
        torch.cuda.empty_cache()

    return all_embeddings


# ---------------------------------------------------------------------------
# Backend: vLLM (for future use when env is updated)
# ---------------------------------------------------------------------------

def _run_vllm(
    all_texts: List[str],
    batch_size: int,
) -> np.ndarray:
    """Extract embeddings via vLLM hidden state extraction."""
    import tempfile
    from safetensors import safe_open
    from vllm import LLM, SamplingParams
    import vllm_hidden_states_extractor
    vllm_hidden_states_extractor.register()

    all_embeddings = np.empty((len(all_texts), EMBED_HIDDEN_DIM), dtype=np.float16)
    t0 = time.time()

    for batch_start in range(0, len(all_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(all_texts))
        batch_texts = all_texts[batch_start:batch_end]

        with tempfile.TemporaryDirectory() as tmpdir:
            llm = LLM(
                model=EMBED_MODEL,
                speculative_config={
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": {
                        "hf_config": {
                            "eagle_aux_hidden_state_layer_ids": [EMBED_LAYER],
                        }
                    },
                },
                kv_transfer_config={
                    "kv_connector": "ExampleHiddenStatesConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                        "shared_storage_path": tmpdir,
                    },
                },
                max_model_len=2048,
                enforce_eager=True,
            )

            sampling_params = SamplingParams(max_tokens=1)
            outputs = llm.generate(batch_texts, sampling_params)

            for i, output in enumerate(outputs):
                hs_path = output.kv_transfer_params.get("hidden_states_path")
                if hs_path is None:
                    raise RuntimeError(
                        f"vLLM did not produce hidden states for batch item {i} "
                        f"(request_id={output.request_id})"
                    )
                with safe_open(hs_path, framework="pt") as f:
                    hidden_states = f.get_tensor("hidden_states")
                layer_hs = hidden_states[0].float()
                mean_pooled = layer_hs.mean(dim=0).half().numpy()
                if mean_pooled.shape != (EMBED_HIDDEN_DIM,):
                    raise RuntimeError(
                        f"Expected ({EMBED_HIDDEN_DIM},), got {mean_pooled.shape}"
                    )
                all_embeddings[batch_start + i] = mean_pooled

            del llm

        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 0
        remaining = len(all_texts) - batch_end
        eta = remaining / rate if rate > 0 else 0
        print(f"    {batch_end}/{len(all_texts)} done, "
              f"{rate:.1f} texts/s, ETA {eta/60:.1f}min")

    return all_embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    max_files: Optional[int] = None,
    file_offset: int = 0,
    batch_size: int = 16,
    backend: str = "hf",
    device: str = "cuda",
    gemini_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    _gemini_dir = Path(gemini_dir) if gemini_dir else GEMINI_OUTPUT_DIR
    _output_dir = Path(output_dir) if output_dir else EMBED_OUTPUT_DIR

    print("=" * 70)
    print(f"PHASE 5: Extract mean-pooled Qwen3-8B embeddings ({backend})")
    print("=" * 70)
    print(f"  Model: {EMBED_MODEL}")
    print(f"  Extract layer: {EMBED_LAYER}")
    print(f"  Hidden dim: {EMBED_HIDDEN_DIM}")
    print(f"  Batch size: {batch_size}")
    print(f"  Backend: {backend}")
    print(f"  Gemini dir: {_gemini_dir}")
    print(f"  Output dir: {_output_dir}")
    print()

    jsonl_files = sorted(_gemini_dir.glob("trajectories_*.jsonl"))
    if not jsonl_files:
        print(f"  No Gemini JSONL files found in {_gemini_dir}")
        return
    if file_offset:
        jsonl_files = jsonl_files[file_offset:]
    if max_files:
        jsonl_files = jsonl_files[:max_files]

    _output_dir.mkdir(parents=True, exist_ok=True)

    # Collect texts from files that haven't been embedded yet
    all_sample_indices: List[int] = []
    all_texts: List[str] = []
    file_ranges: Dict[str, tuple] = {}

    for jsonl_path in jsonl_files:
        stem = jsonl_path.stem
        output_path = _output_dir / f"{stem}_embeddings.npz"
        if output_path.exists():
            print(f"  Skipping {stem} (already done)")
            continue

        gemini_texts = load_gemini_texts(jsonl_path)
        if not gemini_texts:
            print(f"  WARNING: no Gemini texts in {stem}, skipping")
            continue

        sorted_items = sorted(gemini_texts.items())
        start = len(all_texts)
        for idx, text in sorted_items:
            all_sample_indices.append(idx)
            all_texts.append(text)
        file_ranges[stem] = (start, len(all_texts))

    if not all_texts:
        print("  Nothing to embed.")
        return

    print(f"  Total texts to embed: {len(all_texts)} across {len(file_ranges)} files")
    print()

    # Run embedding extraction
    if backend == "hf":
        all_embeddings = _run_hf(all_texts, batch_size, device)
    elif backend == "vllm":
        all_embeddings = _run_vllm(all_texts, batch_size)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'hf' or 'vllm'.")

    # Save per-file
    for stem, (start, end) in file_ranges.items():
        output_path = _output_dir / f"{stem}_embeddings.npz"
        file_indices = all_sample_indices[start:end]
        file_embeddings = all_embeddings[start:end]

        np.savez_compressed(
            output_path,
            sample_indices=np.array(file_indices, dtype=np.int32),
            embeddings=file_embeddings,
        )
        print(f"  Saved {output_path.name}: {file_embeddings.shape}")

    print(f"\n{'=' * 70}")
    print(f"Embedding extraction complete")
    print(f"  Embedded: {len(all_texts)} texts")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Extract Qwen3-8B embeddings")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--file-offset", type=int, default=0,
                        help="Skip first K files (for parallel sharding)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", type=str, default="hf",
                        choices=["hf", "vllm"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gemini-dir", type=str, default=None,
                        help="Override Gemini labels directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override embeddings output directory")
    args = parser.parse_args()
    run(max_files=args.max_files, file_offset=args.file_offset,
        batch_size=args.batch_size, backend=args.backend, device=args.device,
        gemini_dir=args.gemini_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
