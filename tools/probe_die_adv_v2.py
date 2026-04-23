"""Probe die/adversarial v2 prompts against concise regular.

Criteria (from handoff_2026-04-22_overnight.md):
 - cos sim of predonly-extracted embeddings vs concise regular > 0.90
   (ideally near self-noise floor ~0.94)
 - "Instead of" phrase frequency ~ 0%
 - bulleted / multi-sentence "Worst Possible Future" structures rare
 - avg Prediction length within concise range

Outputs stdout report + JSON to --out-dir.

Usage:
  PYTHONPATH=. GEMINI_API_KEY=... python tools/probe_die_adv_v2.py \
      --num-samples 20 --out-dir probe_results/die_adv_v2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
CONCISE_TMPL = TEMPLATE_DIR / "predict_state_only_prompt_concise.txt"
DIE_V2_TMPL = TEMPLATE_DIR / "predict_state_only_prompt_concise_die_v2.txt"
ADV_V2_TMPL = TEMPLATE_DIR / "predict_state_only_prompt_concise_adversarial_v2.txt"

# v1 suffixes (verbatim from eval/eval_online.py)
ADVERSARIAL_V1_SUFFIX = """

IMPORTANT OVERRIDE: Instead of predicting what the player WILL do, predict a future
where the player makes the WORST possible decisions. The player should waste resources,
craft useless items, ignore threats, walk into lava, let hunger/thirst deplete,
and generally play as poorly as possible. Make the future plausible but terrible."""

DIE_V1_SUFFIX = """

IMPORTANT OVERRIDE: Instead of predicting what the player WILL do, predict a future
where the player actively tries to DIE as quickly as possible. The player should
seek out enemies, walk into lava or water when thirsty/hungry, refuse to eat or drink,
engage monsters without weapons, descend floors without preparation, and take every
possible action that leads to rapid death."""

NORMAL_MAX_OUT = 512


def sample_current_obs(npz_path: Path, num_samples: int, rng_seed: int = 0) -> List[Dict]:
    print(f"Loading {npz_path}", flush=True)
    d = np.load(npz_path, allow_pickle=True)
    obs = decode_obs_from_bitpacked(d)
    T = len(obs)
    rng = np.random.default_rng(rng_seed)
    picks = sorted(rng.choice(T, size=min(num_samples, T), replace=False).tolist())
    return [
        {"t": int(t), "obs_t_text": filter_text_obs(obs_to_text(obs[t]))}
        for t in picks
    ]


def render_prompt(template: str, sample: Dict) -> str:
    return template.replace("{current_state_filtered}", sample["obs_t_text"])


def embed_text(text: str, api_key: str) -> np.ndarray:
    url = f"{EMBED_URL}?key={api_key}"
    return _embed_one_gemini(text, url, EMBED_DIM).astype(np.float32)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na and nb) else 0.0


# ---------------------------------------------------------------------------
# Format checks
# ---------------------------------------------------------------------------

BULLETED_LINE_RE = re.compile(r"(?m)^\s*(?:[1-9]\.|[-*•])\s")
WORST_FUTURE_RE = re.compile(r"worst\s+possible\s+future", re.I)
INSTEAD_OF_RE = re.compile(r"\binstead of\b", re.I)
OVERRIDE_RE = re.compile(r"\boverride\b", re.I)


def format_metrics(text: str, pred: str) -> Dict:
    return {
        "chars_total": len(text),
        "chars_pred": len(pred),
        "bulleted_lines": len(BULLETED_LINE_RE.findall(text)),
        "has_worst_future": bool(WORST_FUTURE_RE.search(text)),
        "has_instead_of": bool(INSTEAD_OF_RE.search(text)),
        "has_override": bool(OVERRIDE_RE.search(text)),
        "pred_num_sentences": len(re.findall(r"[.!?](?:\s|$)", pred)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    concise_src = CONCISE_TMPL.read_text()
    die_v2_src = DIE_V2_TMPL.read_text()
    adv_v2_src = ADV_V2_TMPL.read_text()

    variants: Dict[str, str] = {
        "concise":       concise_src,
        "concise_rerun": concise_src,
        "die_v1":        concise_src + DIE_V1_SUFFIX,
        "adv_v1":        concise_src + ADVERSARIAL_V1_SUFFIX,
        "die_v2":        die_v2_src,
        "adv_v2":        adv_v2_src,
    }

    samples = sample_current_obs(PSF_FILE, args.num_samples, rng_seed=args.seed)

    per_variant: Dict[str, Dict] = {v: {"texts": [], "preds": [], "embeds": [],
                                        "metrics": []} for v in variants}

    for i, sample in enumerate(samples):
        print(f"\n-- sample {i+1}/{len(samples)} (t={sample['t']}) --", flush=True)
        for vname, tmpl in variants.items():
            prompt = render_prompt(tmpl, sample)
            r = call_gemini(prompt, api_key, model=MODEL,
                            thinking_budget=0, max_output_tokens=NORMAL_MAX_OUT)
            text = r.get("text", "") if r.get("ok", True) else ""
            if not text:
                print(f"    {vname}: empty, retrying", flush=True); time.sleep(1)
                r = call_gemini(prompt, api_key, model=MODEL,
                                thinking_budget=0, max_output_tokens=NORMAL_MAX_OUT)
                text = r.get("text", "") if r.get("ok", True) else ""
            pred, status = extract_prediction_suffix(text)
            emb = embed_text(pred, api_key)
            per_variant[vname]["texts"].append(text)
            per_variant[vname]["preds"].append(pred)
            per_variant[vname]["embeds"].append(emb)
            per_variant[vname]["metrics"].append(format_metrics(text, pred))
            head = pred.split("\n")[0][:80]
            print(f"    {vname:14s} status={status} pred: {head}", flush=True)

    # ------- cosine sim analysis vs concise
    print("\n=== cosine sim of predonly embedding vs concise ===", flush=True)
    print(f"{'variant':<15} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'pass(>.90)':>10}", flush=True)
    cos_vs_concise: Dict[str, List[float]] = {}
    for v in variants:
        sims = [cos_sim(per_variant["concise"]["embeds"][k],
                        per_variant[v]["embeds"][k])
                for k in range(len(samples))]
        cos_vs_concise[v] = sims
        a = np.array(sims)
        flag = "Y" if a.mean() > 0.90 else ("~" if a.mean() > 0.85 else "N")
        print(f"{v:<15} {a.mean():>7.4f} {a.std():>7.4f} {a.min():>7.4f} {a.max():>7.4f} {flag:>10}", flush=True)

    # ------- format metrics aggregated
    print("\n=== format metrics (per-sample means unless noted) ===", flush=True)
    hdr = f"{'variant':<15} {'chars_tot':>10} {'chars_pred':>11} {'bulleted':>9} {'worst_fut_%':>12} {'instead_of_%':>13} {'override_%':>11} {'pred_sents':>11}"
    print(hdr, flush=True)
    format_summary: Dict[str, Dict] = {}
    for v in variants:
        ms = per_variant[v]["metrics"]
        n = len(ms)
        s = {
            "chars_total_mean": np.mean([m["chars_total"] for m in ms]),
            "chars_pred_mean":  np.mean([m["chars_pred"] for m in ms]),
            "bulleted_mean":    np.mean([m["bulleted_lines"] for m in ms]),
            "worst_future_pct": 100.0 * np.mean([m["has_worst_future"] for m in ms]),
            "instead_of_pct":   100.0 * np.mean([m["has_instead_of"] for m in ms]),
            "override_pct":     100.0 * np.mean([m["has_override"] for m in ms]),
            "pred_sents_mean":  np.mean([m["pred_num_sentences"] for m in ms]),
        }
        format_summary[v] = s
        print(f"{v:<15} {s['chars_total_mean']:>10.1f} {s['chars_pred_mean']:>11.1f} "
              f"{s['bulleted_mean']:>9.2f} {s['worst_future_pct']:>12.1f} "
              f"{s['instead_of_pct']:>13.1f} {s['override_pct']:>11.1f} "
              f"{s['pred_sents_mean']:>11.2f}", flush=True)

    # ------- verdict
    print("\n=== v2 verdict ===", flush=True)
    for v in ("die_v2", "adv_v2"):
        s = format_summary[v]
        cos = np.mean(cos_vs_concise[v])
        self_noise = np.mean(cos_vs_concise["concise_rerun"])
        checks = [
            ("cos_vs_concise > 0.90",       cos > 0.90),
            ("cos_vs_concise > 0.85",       cos > 0.85),
            ("close to self-noise (w/in .05)", abs(cos - self_noise) < 0.05),
            ("instead_of_% == 0",           s["instead_of_pct"] == 0),
            ("override_% == 0",             s["override_pct"] == 0),
            ("worst_future_% == 0",         s["worst_future_pct"] == 0),
            ("bulleted_mean <= 0.5",        s["bulleted_mean"] <= 0.5),
        ]
        print(f"\n-- {v} (cos={cos:.3f}, self-noise={self_noise:.3f}) --", flush=True)
        for msg, ok in checks:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {msg}", flush=True)

    # ------- dump
    out_path = args.out_dir / "probe_results.json"
    dumpable = {
        "variant_names": list(variants.keys()),
        "samples_t": [s["t"] for s in samples],
        "cos_vs_concise": {v: cos_vs_concise[v] for v in variants},
        "format_summary": format_summary,
        "per_variant_metrics": {v: per_variant[v]["metrics"] for v in variants},
        "per_variant_texts":   {v: per_variant[v]["texts"]   for v in variants},
        "per_variant_preds":   {v: per_variant[v]["preds"]   for v in variants},
    }
    with out_path.open("w") as f:
        json.dump(dumpable, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
