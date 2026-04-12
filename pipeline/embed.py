#!/usr/bin/env python3
"""
Phase 5: Extract embeddings from Gemini oracle texts.

Saves each file's embeddings immediately after processing (incremental),
so progress is never lost on job timeout.

Supports backends:
  --backend hf           : HuggingFace transformers, Qwen3-8B layer-30 mean-pool (default)
  --backend vllm         : vLLM extract_hidden_states (faster, needs vllm >=0.15)
  --backend qwen3_embed  : Qwen/Qwen3-Embedding, last-token pooling
  --backend gemini_embed : Gemini embedding API (gemini-embedding-001)

Usage:
    python -m pipeline.embed [--batch-size 16] [--backend hf] [--max-files N] [--file-offset K]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Backend: HuggingFace transformers (Qwen3-8B, layer-30 mean-pool)
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

    return mean_pooled.cpu().numpy()


def _load_model_hf(device: str):
    """Load Qwen3-8B model and tokenizer. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading HF model {EMBED_MODEL}...")
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
    print(f"  Model loaded in {time.time() - t0:.1f}s, "
          f"hidden_size={model.config.hidden_size}, "
          f"layers={model.config.num_hidden_layers}")

    if model.config.hidden_size != EMBED_HIDDEN_DIM:
        raise RuntimeError(
            f"Model hidden_size={model.config.hidden_size} != "
            f"EMBED_HIDDEN_DIM={EMBED_HIDDEN_DIM}"
        )
    return model, tokenizer


def _embed_hf(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Embed texts using already-loaded HF model. Returns (N, hidden_dim) float16."""
    all_embeddings = np.empty((len(texts), EMBED_HIDDEN_DIM), dtype=np.float16)
    t0 = time.time()

    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_emb = _extract_hf(
            texts[batch_start:batch_end], model, tokenizer, EMBED_LAYER, device
        )
        all_embeddings[batch_start:batch_end] = batch_emb

        elapsed = time.time() - t0
        rate = batch_end / elapsed if elapsed > 0 else 0
        remaining = len(texts) - batch_end
        eta = remaining / rate if rate > 0 else 0
        print(f"    {batch_end}/{len(texts)} done, "
              f"{rate:.1f} texts/s, ETA {eta/60:.1f}min")

    return all_embeddings


# ---------------------------------------------------------------------------
# Backend: Qwen3-Embedding (dedicated embedding model, last-token pooling)
# ---------------------------------------------------------------------------

def _load_model_qwen3embed(device: str):
    """Load Qwen/Qwen3-Embedding model. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    MODEL_ID = "Qwen/Qwen3-Embedding-8B"
    print(f"  Loading {MODEL_ID}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Qwen3-Embedding requires left-padding

    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=device if device != "cpu" else "cpu",
        attn_implementation="sdpa" if device != "cpu" else "eager",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s, hidden_dim={model.config.hidden_size}")
    return model, tokenizer


def _embed_qwen3embed(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Embed texts using already-loaded Qwen3-Embedding model (last-token pool)."""
    import torch

    hidden_dim = model.config.hidden_size
    all_embeddings = np.empty((len(texts), hidden_dim), dtype=np.float16)
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", truncation=True,
            max_length=2048, padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(len(chunk), device=device)
            last_hs = out.last_hidden_state[batch_idx, seq_lens]  # (B, H)

        all_embeddings[i : i + len(chunk)] = last_hs.half().cpu().numpy()

        elapsed = time.time() - t0
        done = i + len(chunk)
        rate = done / elapsed if elapsed > 0 else 0
        remaining = len(texts) - done
        eta = remaining / rate if rate > 0 else 0
        print(f"    {done}/{len(texts)} done, "
              f"{rate:.1f} texts/s, ETA {eta/60:.1f}min")

    return all_embeddings


# ---------------------------------------------------------------------------
# Backend: Gemini embedding API (parallel HTTP calls)
# ---------------------------------------------------------------------------

def _embed_one_gemini(text: str, url: str, output_dim: int) -> np.ndarray:
    """Call Gemini Embedding API for a single text. Retries on 429."""
    import json
    from urllib import request as urlrequest, error as urlerror

    payload = json.dumps({
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": output_dim,
    }).encode()
    headers = {"Content-Type": "application/json"}

    for attempt in range(6):
        try:
            req = urlrequest.Request(url, data=payload, headers=headers, method="POST")
            with urlrequest.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read())
            return np.array(d["embedding"]["values"], dtype=np.float32)
        except urlerror.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini embed HTTP {e.code}: {e.read()[:200]}")
        except (urlerror.URLError, TimeoutError) as e:
            if attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini embed network error: {e}")
    raise RuntimeError("Gemini embed: max retries exceeded")


def _run_gemini_embed(
    all_texts: List[str],
    output_dim: int,
    api_key: str,
    max_workers: int = 100,
) -> np.ndarray:
    """Embed texts via Gemini embedding API (parallel requests)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    url = f"{BASE_URL}/gemini-embedding-001:embedContent?key={api_key}"

    all_embeddings = np.empty((len(all_texts), output_dim), dtype=np.float32)
    t0 = time.time()
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_embed_one_gemini, text, url, output_dim): i
            for i, text in enumerate(all_texts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            all_embeddings[i] = fut.result()
            done_count += 1
            if done_count % 1000 == 0:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = len(all_texts) - done_count
                eta = remaining / rate if rate > 0 else 0
                print(f"    {done_count}/{len(all_texts)} done, "
                      f"{rate:.1f} texts/s, ETA {eta/60:.1f}min")

    return all_embeddings.astype(np.float16)


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
    output_dim: int = 3072,
    gemini_api_key: Optional[str] = None,
):
    _gemini_dir = Path(gemini_dir) if gemini_dir else GEMINI_OUTPUT_DIR
    _output_dir = Path(output_dir) if output_dir else EMBED_OUTPUT_DIR

    print("=" * 70)
    print(f"PHASE 5: Extract embeddings — backend={backend}")
    print("=" * 70)
    print(f"  Backend: {backend}")
    if backend in ("hf", "vllm"):
        print(f"  Model:   {EMBED_MODEL}  (layer {EMBED_LAYER}, mean-pool)")
        print(f"  Dim:     {EMBED_HIDDEN_DIM}")
    elif backend == "qwen3_embed":
        print(f"  Model:   Qwen/Qwen3-Embedding  (last-token pool)")
    elif backend == "gemini_embed":
        print(f"  Model:   gemini-embedding-001  (API)")
        print(f"  Dim:     {output_dim}")
    print(f"  Batch:   {batch_size}")
    print(f"  Input:   {_gemini_dir}")
    print(f"  Output:  {_output_dir}")
    print()

    jsonl_files = sorted(_gemini_dir.glob("trajectories_*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No Gemini JSONL files found in {_gemini_dir}")
    if file_offset:
        jsonl_files = jsonl_files[file_offset:]
    if max_files:
        jsonl_files = jsonl_files[:max_files]

    _output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once for model-based backends
    _model, _tokenizer = None, None
    if backend == "hf":
        _model, _tokenizer = _load_model_hf(device)
    elif backend == "qwen3_embed":
        _model, _tokenizer = _load_model_qwen3embed(device)
    elif backend == "gemini_embed":
        import os
        _api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not _api_key:
            raise ValueError(
                "--gemini-api-key or GEMINI_API_KEY env var required for gemini_embed"
            )

    files_done = 0
    texts_total = 0
    t_global = time.time()

    for file_idx, jsonl_path in enumerate(jsonl_files):
        stem = jsonl_path.stem
        output_path = _output_dir / f"{stem}_embeddings.npz"

        if output_path.exists():
            print(f"  [{file_idx + 1}/{len(jsonl_files)}] Skipping {stem} (already done)")
            continue

        gemini_texts = load_gemini_texts(jsonl_path)
        if not gemini_texts:
            print(f"  WARNING: no Gemini texts in {stem}, skipping")
            continue

        sorted_items = sorted(gemini_texts.items())
        indices = [idx for idx, _ in sorted_items]
        texts = [text for _, text in sorted_items]

        print(f"\n  [{file_idx + 1}/{len(jsonl_files)}] {stem}  ({len(texts)} texts)")

        if backend == "hf":
            embeddings = _embed_hf(texts, _model, _tokenizer, batch_size, device)
        elif backend == "qwen3_embed":
            embeddings = _embed_qwen3embed(texts, _model, _tokenizer, batch_size, device)
        elif backend == "gemini_embed":
            embeddings = _run_gemini_embed(texts, output_dim, _api_key)
        elif backend == "vllm":
            embeddings = _run_vllm(texts, batch_size)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        np.savez_compressed(
            output_path,
            sample_indices=np.array(indices, dtype=np.int32),
            embeddings=embeddings,
        )
        print(f"  Saved {output_path.name}: {embeddings.shape}")

        files_done += 1
        texts_total += len(texts)

    # Cleanup
    if _model is not None:
        del _model
        if device != "cpu":
            import torch
            torch.cuda.empty_cache()

    elapsed = time.time() - t_global
    print(f"\n{'=' * 70}")
    print(f"Embedding extraction complete")
    print(f"  Files processed: {files_done}")
    print(f"  Texts embedded:  {texts_total}")
    print(f"  Total time:      {elapsed/60:.1f}min")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Extract text embeddings for offline RL")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--file-offset", type=int, default=0,
                        help="Skip first K files (for parallel sharding)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", type=str, default="hf",
                        choices=["hf", "vllm", "qwen3_embed", "gemini_embed"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gemini-dir", type=str, default=None,
                        help="Override Gemini labels directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override embeddings output directory")
    parser.add_argument("--output-dim", type=int, default=3072,
                        help="Output dimensionality for gemini_embed backend (default: 3072)")
    parser.add_argument("--gemini-api-key", type=str, default=None,
                        help="Gemini API key (or GEMINI_API_KEY env var)")
    args = parser.parse_args()
    run(
        max_files=args.max_files,
        file_offset=args.file_offset,
        batch_size=args.batch_size,
        backend=args.backend,
        device=args.device,
        gemini_dir=args.gemini_dir,
        output_dir=args.output_dir,
        output_dim=args.output_dim,
        gemini_api_key=args.gemini_api_key,
    )


if __name__ == "__main__":
    main()
