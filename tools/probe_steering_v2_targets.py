"""Probe v2 positive-target and pure-direction steering prompts.

Variants probed:
  - target_collect_stone_v2
  - target_descend_v2
  - target_eat_cow_v2
  - target_drink_water_v2
  - target_place_stone_v2
  - direction_left_v2 / right_v2 / up_v2 / down_v2

Same checks as probe_steering_v2.py: cos sim of predonly embedding vs
concise regular, neg-phrase hit count, override flag, bulleted lines,
direction-word presence.

Usage:
  PYTHONPATH=. GEMINI_API_KEY=... python tools/probe_steering_v2_targets.py \
      --num-samples 15 --out-dir probe_results/steering_v2_targets
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

# (label, file)
NEW_VARIANTS = [
    ("target_collect_stone_v2",  "predict_state_only_prompt_concise_target_collect_stone_v2.txt"),
    ("target_descend_v2",        "predict_state_only_prompt_concise_target_descend_v2.txt"),
    ("target_eat_cow_v2",        "predict_state_only_prompt_concise_target_eat_cow_v2.txt"),
    ("target_drink_water_v2",    "predict_state_only_prompt_concise_target_drink_water_v2.txt"),
    ("target_place_stone_v2",    "predict_state_only_prompt_concise_target_place_stone_v2.txt"),
    ("direction_left_v2",        "predict_state_only_prompt_concise_direction_left_v2.txt"),
    ("direction_right_v2",       "predict_state_only_prompt_concise_direction_right_v2.txt"),
    ("direction_up_v2",          "predict_state_only_prompt_concise_direction_up_v2.txt"),
    ("direction_down_v2",        "predict_state_only_prompt_concise_direction_down_v2.txt"),
]

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


NEG_PHRASES = [
    r"\binstead of\b", r"\brather than\b", r"\brefuse\b",
    r"\bavoid\b", r"\baway from\b", r"\bignor",
    r"\bopaque\b", r"\broute around\b",
]
NEG_RE = re.compile("|".join(NEG_PHRASES), re.I)
DIR_RE = re.compile(r"\b(up|down|left|right|toward|towards|north|south|east|west|above|below)\b", re.I)
BULLET_RE = re.compile(r"(?m)^\s*(?:[1-9]\.|[-*•])\s")
OVERRIDE_RE = re.compile(r"\boverride\b", re.I)


def format_metrics(text: str, pred: str) -> Dict:
    return {
        "chars_total": len(text),
        "chars_pred": len(pred),
        "bulleted_lines": len(BULLET_RE.findall(text)),
        "neg_hits_pred": len(NEG_RE.findall(pred)),
        "has_override_pred": bool(OVERRIDE_RE.search(pred)),
        "dir_hits_pred": len(DIR_RE.findall(pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr); sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants: Dict[str, str] = {"concise": CONCISE.read_text()}
    for label, fname in NEW_VARIANTS:
        path = TEMPLATE_DIR / fname
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr); sys.exit(2)
        variants[label] = path.read_text()

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
            print(f"    {vname:28s} status={status} pred: {pred.split(chr(10))[0][:90]}", flush=True)

    # cos sim vs concise
    print("\n=== cos sim of predonly embedding vs concise ===", flush=True)
    print(f"{'variant':<28} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}", flush=True)
    cos_vs: Dict[str, List[float]] = {}
    for v in variants:
        sims = [cos_sim(per_variant["concise"]["embeds"][k], per_variant[v]["embeds"][k])
                for k in range(len(samples))]
        cos_vs[v] = sims
        a = np.array(sims)
        print(f"{v:<28} {a.mean():>7.4f} {a.std():>7.4f} {a.min():>7.4f} {a.max():>7.4f}", flush=True)

    # format metrics
    print("\n=== format metrics ===", flush=True)
    hdr = f"{'variant':<28} {'chars_pred':>11} {'neg_hits':>9} {'override_%':>11} {'dir_hits':>9} {'bullet':>7}"
    print(hdr, flush=True)
    format_summary = {}
    for v in variants:
        ms = per_variant[v]["metrics"]
        s = {
            "chars_pred_mean": float(np.mean([m["chars_pred"] for m in ms])),
            "neg_hits_mean":   float(np.mean([m["neg_hits_pred"] for m in ms])),
            "override_pct":    100.0 * float(np.mean([m["has_override_pred"] for m in ms])),
            "dir_hits_mean":   float(np.mean([m["dir_hits_pred"] for m in ms])),
            "bulleted_mean":   float(np.mean([m["bulleted_lines"] for m in ms])),
        }
        format_summary[v] = s
        print(f"{v:<28} {s['chars_pred_mean']:>11.1f} {s['neg_hits_mean']:>9.2f} "
              f"{s['override_pct']:>11.1f} {s['dir_hits_mean']:>9.2f} "
              f"{s['bulleted_mean']:>7.2f}", flush=True)

    # verdict — same criteria as steering v2
    print("\n=== verdict ===", flush=True)
    for v_label, _ in NEW_VARIANTS:
        s = format_summary[v_label]
        checks = [
            ("neg_hits_mean == 0 in Prediction line", s["neg_hits_mean"] == 0),
            ("override_% == 0 in Prediction line",    s["override_pct"] == 0),
            ("dir_hits_mean >= 1",                    s["dir_hits_mean"] >= 1.0),
            ("bulleted_mean <= 0.5",                  s["bulleted_mean"] <= 0.5),
        ]
        print(f"\n-- {v_label} --", flush=True)
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
