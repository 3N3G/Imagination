#!/usr/bin/env python3
"""
Benchmark different embedding extraction strategies for diversity.
Tests: pooling method × layer depth on a small set of Gemini texts.
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

EMBED_MODEL = "Qwen/Qwen3-8B"
DEVICE = "cuda"
MAX_LENGTH = 2048


def load_texts(jsonl_path, max_texts=200):
    texts = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ok") and d.get("text"):
                texts.append(d["text"])
            if len(texts) >= max_texts:
                break
    return texts


def extract_embeddings(texts, model, tokenizer, layer, pool_method, device=DEVICE):
    """Extract embeddings with different pooling strategies."""
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=MAX_LENGTH, padding=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[layer]  # (B, seq_len, H)

    mask = attention_mask.unsqueeze(-1).float()

    if pool_method == "mean":
        summed = (hidden.float() * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        emb = summed / lengths
    elif pool_method == "last_token":
        # Last non-padding token for each sequence
        seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
        emb = hidden[torch.arange(len(texts)), seq_lens].float()
    elif pool_method == "max":
        # Max pool over non-padding tokens
        hidden_masked = hidden.float() * mask + (~mask.bool()).float() * (-1e9)
        emb, _ = hidden_masked.max(dim=1)
    elif pool_method == "mean_centered":
        # Mean pool then subtract corpus mean (two-pass, but we approximate inline)
        summed = (hidden.float() * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        emb = summed / lengths
        emb = emb - emb.mean(dim=0, keepdim=True)
    elif pool_method == "first_token":
        emb = hidden[:, 0].float()
    else:
        raise ValueError(f"Unknown pool method: {pool_method}")

    return emb.cpu().numpy().astype(np.float32)


def analyze_diversity(embs, label):
    """Compute diversity metrics for a set of embeddings."""
    n = len(embs)
    norms = np.linalg.norm(embs, axis=1)

    # Remove any zero-norm rows
    valid = norms > 1e-6
    embs = embs[valid]
    norms = norms[valid]
    n = len(embs)

    if n < 10:
        print(f"  {label}: too few valid embeddings ({n})")
        return

    # Random pair cosine similarity (raw)
    rng = np.random.RandomState(42)
    idx1 = rng.randint(0, n, 500)
    idx2 = rng.randint(0, n, 500)
    raw_sims = []
    for i, j in zip(idx1, idx2):
        if i == j:
            continue
        a, b = embs[i], embs[j]
        sim = np.dot(a, b) / (norms[i] * norms[j])
        raw_sims.append(sim)
    raw_sims = np.array(raw_sims)

    # After centering
    centered = embs - embs.mean(axis=0)
    c_norms = np.linalg.norm(centered, axis=1)
    c_valid = c_norms > 1e-6
    centered_sims = []
    for i, j in zip(idx1, idx2):
        if i == j or not c_valid[i] or not c_valid[j]:
            continue
        a, b = centered[i], centered[j]
        sim = np.dot(a, b) / (c_norms[i] * c_norms[j])
        centered_sims.append(sim)
    centered_sims = np.array(centered_sims)

    # After z-score normalization (per-dimension, like training does)
    std = embs.std(axis=0)
    std[std < 1e-8] = 1.0
    znormed = (embs - embs.mean(axis=0)) / std
    z_norms = np.linalg.norm(znormed, axis=1)
    z_sims = []
    for i, j in zip(idx1, idx2):
        if i == j:
            continue
        a, b = znormed[i], znormed[j]
        sim = np.dot(a, b) / (z_norms[i] * z_norms[j])
        z_sims.append(sim)
    z_sims = np.array(z_sims)

    # PCA: effective dimensionality
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    var_explained = np.cumsum(S ** 2) / np.sum(S ** 2)
    eff_dim_90 = np.searchsorted(var_explained, 0.90) + 1
    eff_dim_95 = np.searchsorted(var_explained, 0.95) + 1

    print(
        f"  {label:40s}  "
        f"raw_cos={raw_sims.mean():+.4f}±{raw_sims.std():.3f}  "
        f"centered_cos={centered_sims.mean():+.4f}±{centered_sims.std():.3f}  "
        f"znorm_cos={z_sims.mean():+.4f}±{z_sims.std():.3f}  "
        f"eff_dim_90={eff_dim_90:4d}  "
        f"eff_dim_95={eff_dim_95:4d}  "
        f"norm={norms.mean():.1f}±{norms.std():.1f}"
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=str,
                    default="/data/group_data/rl/geney/oracle_pipeline/predict_only_gemini_labels/trajectories_000000.jsonl")
    p.add_argument("--max-texts", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    print(f"Loading texts from {args.jsonl} (max={args.max_texts})")
    texts = load_texts(args.jsonl, args.max_texts)
    print(f"  Loaded {len(texts)} texts")

    print(f"\nLoading model: {EMBED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        EMBED_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE)
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"  Loaded. {num_layers} layers, hidden_size={model.config.hidden_size}")

    pool_methods = ["mean", "last_token", "max", "mean_centered", "first_token"]
    layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1, num_layers]
    # layer indices: layer 0 = embedding, layer N = final layer output

    print(f"\nExtracting embeddings for {len(pool_methods)} methods × {len(layers)} layers...")
    print(f"  Layers: {layers}")
    print(f"  Methods: {pool_methods}")
    print()

    header = f"  {'Config':40s}  {'raw_cos':>18s}  {'centered_cos':>18s}  {'znorm_cos':>18s}  {'eff90':>8s}  {'eff95':>8s}  {'norm':>12s}"
    print(header)
    print("  " + "-" * len(header))

    for layer in layers:
        for pool in pool_methods:
            label = f"L{layer:02d}_{pool}"
            all_embs = []
            for i in range(0, len(texts), args.batch_size):
                batch = texts[i:i + args.batch_size]
                emb = extract_embeddings(batch, model, tokenizer, layer, pool)
                all_embs.append(emb)
            embs = np.concatenate(all_embs, axis=0)
            analyze_diversity(embs, label)
        print()


if __name__ == "__main__":
    main()
