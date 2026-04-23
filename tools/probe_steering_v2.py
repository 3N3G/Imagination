"""Probe v2 steering prompts (avoid_water, avoid_animals) against concise regular.

User's explicit format criterion (journals/log_2026-04-23, morning note):
  The prediction must read as a POSITIVE direction-stating sentence.
  E.g. "walking left towards the trees" — NOT "instead of looking for water"
  or "walking away from the water".

Criteria tracked:
  - cos sim vs concise regular (content shift is expected to be meaningful
    but not extreme since we only disable one tile class).
  - "instead of", "rather than", "refuse", "avoid", "away from", "ignor"
    phrase frequencies — all target 0%.
  - bulleted structures, "override", "Worst Possible Future" — target 0%.
  - "water"/"cow"/"animal" mention frequency (informational — non-zero is
    fine, negation phrasing is what we care about).
  - Direction words present (up/down/left/right/toward) — target 100%.

Usage:
  PYTHONPATH=. GEMINI_API_KEY=... python tools/probe_steering_v2.py \
      --num-samples 20 --out-dir probe_results/steering_v2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from labelling.obs_to_text import obs_to_text
from pipeline.text_utils import filter_text_obs
from pipeline.gemini_label import call_gemini, decode_obs_from_bitpacked
from pipeline.embed import _embed_one_gemini, extract_prediction_suffix

MODEL = "gemini-3-flash-preview"
EMBED_DIM = 3072
EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-embedding-001:embedContent"
)

PSF_FILE = Path(
    "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
    "/final_trajectories_psf_v2_cadence5_gemini_emb/trajectories_000000.npz"
)

TEMPLATE_DIR = Path("configs/training/templates")
CONCISE = TEMPLATE_DIR / "predict_state_only_prompt_concise.txt"
AVOID_WATER = TEMPLATE_DIR / "predict_state_only_prompt_concise_avoid_water_v2.txt"
AVOID_ANIMALS = TEMPLATE_DIR / "predict_state_only_prompt_concise_avoid_animals_v2.txt"

NORMAL_MAX_OUT = 512


def sample_current_obs(npz_path, num_samples, rng_seed=0):
    print(f"Loading {npz_path}", flush=True)
    d = np.load(npz_path, allow_pickle=True)
    obs = decode_obs_from_bitpacked(d)
    T = len(obs)
    rng = np.random.default_rng(rng_seed)
    picks = sorted(rng.choice(T, size=min(num_samples, T), replace=False).tolist())
    return [{"t": int(t), "obs_t_text": filter_text_obs(obs_to_text(obs[t]))} for t in picks]


def embed_text(text, api_key):
    url = f"{EMBED_URL}?key={api_key}"
    return _embed_one_gemini(text, url, EMBED_DIM).astype(np.float32)


def cos_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na and nb) else 0.0


# negation / "avoiding" phrases that would indicate prompt leakage
NEG_PHRASES = [
    r"\binstead of\b",
    r"\brather than\b",
    r"\brefuse\b",
    r"\bavoid\b",
    r"\baway from\b",
    r"\bignor",  # ignore, ignoring, ignores
    r"\bopaque\b",
    r"\broute around\b",
]
NEG_RE = re.compile("|".join(NEG_PHRASES), re.I)
DIR_RE = re.compile(r"\b(up|down|left|right|toward|towards|north|south|east|west)\b", re.I)
BULLET_RE = re.compile(r"(?m)^\s*(?:[1-9]\.|[-*•])\s")
OVERRIDE_RE = re.compile(r"\boverride\b", re.I)
WATER_RE = re.compile(r"\bwater\b", re.I)
COW_RE = re.compile(r"\b(cow|cows|animal|animals)\b", re.I)


def format_metrics(text: str, pred: str) -> Dict:
    return {
        "chars_total": len(text),
        "chars_pred": len(pred),
        "bulleted_lines": len(BULLET_RE.findall(text)),
        "neg_hits_pred": len(NEG_RE.findall(pred)),
        "has_override_pred": bool(OVERRIDE_RE.search(pred)),
        "dir_hits_pred": len(DIR_RE.findall(pred)),
        "water_mention_pred": bool(WATER_RE.search(pred)),
        "cow_mention_pred": bool(COW_RE.search(pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr); sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "concise":        CONCISE.read_text(),
        "concise_rerun":  CONCISE.read_text(),
        "avoid_water_v2": AVOID_WATER.read_text(),
        "avoid_animals_v2": AVOID_ANIMALS.read_text(),
    }

    samples = sample_current_obs(PSF_FILE, args.num_samples, rng_seed=args.seed)

    per_variant: Dict[str, Dict] = {v: {"texts": [], "preds": [], "embeds": [], "metrics": []}
                                    for v in variants}

    for i, sample in enumerate(samples):
        print(f"\n-- sample {i+1}/{len(samples)} (t={sample['t']}) --", flush=True)
        for vname, tmpl in variants.items():
            prompt = tmpl.replace("{current_state_filtered}", sample["obs_t_text"])
            r = call_gemini(prompt, api_key, model=MODEL, thinking_budget=0, max_output_tokens=NORMAL_MAX_OUT)
            text = r.get("text", "") if r.get("ok", True) else ""
            if not text:
                print(f"    {vname}: empty, retrying", flush=True); time.sleep(1)
                r = call_gemini(prompt, api_key, model=MODEL, thinking_budget=0, max_output_tokens=NORMAL_MAX_OUT)
                text = r.get("text", "") if r.get("ok", True) else ""
            pred, status = extract_prediction_suffix(text)
            emb = embed_text(pred, api_key)
            per_variant[vname]["texts"].append(text)
            per_variant[vname]["preds"].append(pred)
            per_variant[vname]["embeds"].append(emb)
            per_variant[vname]["metrics"].append(format_metrics(text, pred))
            print(f"    {vname:18s} status={status} pred: {pred.split(chr(10))[0][:90]}", flush=True)

    # cos sim vs concise
    print("\n=== cos sim of predonly embedding vs concise ===", flush=True)
    print(f"{'variant':<18} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}", flush=True)
    cos_vs: Dict[str, List[float]] = {}
    for v in variants:
        sims = [cos_sim(per_variant["concise"]["embeds"][k], per_variant[v]["embeds"][k])
                for k in range(len(samples))]
        cos_vs[v] = sims
        a = np.array(sims)
        print(f"{v:<18} {a.mean():>7.4f} {a.std():>7.4f} {a.min():>7.4f} {a.max():>7.4f}", flush=True)

    # format metrics
    print("\n=== format metrics (per-sample means or % with flag) ===", flush=True)
    hdr = f"{'variant':<18} {'chars_pred':>11} {'neg_hits':>9} {'override_%':>11} {'dir_hits':>9} {'water_%':>8} {'cow_%':>6} {'bullet':>7}"
    print(hdr, flush=True)
    format_summary = {}
    for v in variants:
        ms = per_variant[v]["metrics"]
        s = {
            "chars_pred_mean":     float(np.mean([m["chars_pred"] for m in ms])),
            "neg_hits_mean":       float(np.mean([m["neg_hits_pred"] for m in ms])),
            "override_pct":        100.0 * float(np.mean([m["has_override_pred"] for m in ms])),
            "dir_hits_mean":       float(np.mean([m["dir_hits_pred"] for m in ms])),
            "water_pct":           100.0 * float(np.mean([m["water_mention_pred"] for m in ms])),
            "cow_pct":             100.0 * float(np.mean([m["cow_mention_pred"] for m in ms])),
            "bulleted_mean":       float(np.mean([m["bulleted_lines"] for m in ms])),
        }
        format_summary[v] = s
        print(f"{v:<18} {s['chars_pred_mean']:>11.1f} {s['neg_hits_mean']:>9.2f} "
              f"{s['override_pct']:>11.1f} {s['dir_hits_mean']:>9.2f} "
              f"{s['water_pct']:>8.1f} {s['cow_pct']:>6.1f} {s['bulleted_mean']:>7.2f}", flush=True)

    # verdict
    print("\n=== verdict ===", flush=True)
    for v in ("avoid_water_v2", "avoid_animals_v2"):
        s = format_summary[v]
        checks = [
            ("neg_hits_mean == 0 in Prediction line",  s["neg_hits_mean"] == 0),
            ("override_% == 0 in Prediction line",     s["override_pct"] == 0),
            ("dir_hits_mean >= 1",                      s["dir_hits_mean"] >= 1.0),
            ("bulleted_mean <= 0.5",                    s["bulleted_mean"] <= 0.5),
        ]
        print(f"\n-- {v} --", flush=True)
        for msg, ok in checks:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {msg}", flush=True)

    # dump
    dumpable = {
        "variant_names": list(variants.keys()),
        "samples_t": [s["t"] for s in samples],
        "cos_vs_concise": cos_vs,
        "format_summary": format_summary,
        "per_variant_preds": {v: per_variant[v]["preds"] for v in variants},
        "per_variant_texts": {v: per_variant[v]["texts"] for v in variants},
        "per_variant_metrics": {v: per_variant[v]["metrics"] for v in variants},
    }
    out = args.out_dir / "probe_results.json"
    with out.open("w") as f:
        json.dump(dumpable, f, indent=2, default=str)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
