"""Iterative probe harness for thinking / grounded prompt design.

For N samples from each of {PSF, golden} datasets, run Gemini with several
prompt variants, embed the `Prediction:` suffix via gemini-embedding-001, and
report cosine-similarity matrices. Goal: find a grounded prompt whose
embeddings sit close to the natural (concise) distribution, so the downstream
fidelity test is not confounded by distribution shift.

Outputs to stdout + JSON to --out-dir for post-hoc analysis.

Criteria (from discussion):
  mean_cos(concise, grounded) > 0.85  -> clean
  < 0.70                             -> distribution shift, iterate
  in between                         -> report, judge case-by-case

Usage:
  PYTHONPATH=. python tools/probe_grounded_prompts.py \
      --num-samples 10 --out-dir probe_results/iter01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from labelling.obs_to_text import obs_to_text
from pipeline.text_utils import filter_text_obs
from pipeline.gemini_label import call_gemini, decode_obs_from_bitpacked
from pipeline.embed import _embed_one_gemini, extract_prediction_suffix
from pipeline.config import ACTION_NAMES

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
GOLDEN_FILE = Path(
    "/data/group_data/rl/geney/oracle_pipeline"
    "/predict_only_final_v2_cadence5_gemini_emb/trajectories_000000.npz"
)

TEMPLATE_DIR = Path("configs/training/templates")
CONCISE_TMPL = TEMPLATE_DIR / "predict_state_only_prompt_concise.txt"
THINKING_TMPL = TEMPLATE_DIR / "predict_only_thinking_prompt.txt"

THINK_BUDGET = 512
THINK_MAX_OUT = 1024
NORMAL_MAX_OUT = 512

# Gemini call cadence used to build datasets: future_offset=5 env steps forward.
FUTURE_OFFSET = 5


# ---------------------------------------------------------------------------
# Grounded prompt variants — iterate these
# ---------------------------------------------------------------------------
GROUNDED_V1 = """You are forecasting what a Craftax player is about to do.

Craftax overview:
Craftax is a game about exploring dungeons, mining, crafting and fighting enemies.

Game rules:
1) Coordinates are (Row, Column), centered on the player at (0,0).
   - Negative Row = UP, Positive Row = DOWN.
   - Negative Column = LEFT, Positive Column = RIGHT.
2) Intrinsics: Health, Food, Drink, Energy, Mana. All out of 9.
   - Food/Drink/Energy naturally decay. If they reach 0, Health decays.

Output ONLY a single "Prediction:" line. Use relative directions (up, down, left,
right, up-left, etc.) and qualitative distances (adjacent, nearby, a few tiles,
far) — do NOT reference specific numerical (row, column) coordinates. Keep the
prediction to 1-2 sentences describing the player's high-level behavior.

Example outputs:
Prediction: Move right to the cluster of trees and chop wood for basic tools.
Prediction: Chase and kill the cow directly above to restore food.
Prediction: Move down-left to the nearby water tiles to drink, then look for a safe stone cluster to sleep in.
Prediction: Move up and left to the visible open ladder and descend to the next floor.

Now, predict the future of the following state.

Current state:
{current_state_filtered}

(Internal context, do not mention: the player's next few actions and the state
five steps ahead are provided below. Use these to inform your forecast, but
phrase the Prediction exactly as if you were forecasting from the current state
alone, in the same voice as the examples.)
Next actions: {actions_csv}
State five steps ahead:
{future_state_filtered}
"""

GROUNDED_V2 = """Predict the next few seconds of Craftax gameplay for the state below.

Craftax overview: dungeon/mine/craft/fight. Use relative directions (up, down,
left, right, up-left, ...) and qualitative distances (adjacent, nearby, a few
tiles, far). Do NOT reference numerical (row, col) coords.

Output ONLY a single "Prediction:" line, 1-2 sentences, in the voice of these
examples:
Prediction: Move right to the cluster of trees and chop wood for basic tools.
Prediction: Chase and kill the cow directly above to restore food.
Prediction: Move down-left to the nearby water tiles to drink, then look for a safe stone cluster to sleep in.
Prediction: Move up and left to the visible open ladder and descend to the next floor.

Current state:
{current_state_filtered}

For your reference (do not copy or mention verbatim): after the next five
actions ({actions_csv}) the player's state will be:
{future_state_filtered}

Now write the Prediction line.
"""

GROUNDED_V3 = """You are a Craftax forecaster. Emit a single "Prediction:" line that
describes what the player is about to do over the next few steps, in the voice
below.

Format rules: 1-2 sentences, relative directions only (up/down/left/right/up-
left/...), qualitative distances (adjacent/nearby/a few tiles/far). Never use
numerical (row, col) coordinates. Never write the words "will happen", "future",
"prediction of", "forecast"; just describe the action, e.g. "Move up to the
tree and chop it."

Examples:
Prediction: Move right to the cluster of trees and chop wood for basic tools.
Prediction: Chase and kill the cow directly above to restore food.
Prediction: Move down-left to the nearby water tiles to drink, then look for a safe stone cluster to sleep in.
Prediction: Move up and left to the visible open ladder and descend to the next floor.

State right now:
{current_state_filtered}

(For grounding: in five steps the state becomes --
{future_state_filtered}
)

Prediction:"""


GROUNDED_VARIANTS: Dict[str, str] = {
    "grounded_v1": GROUNDED_V1,
    "grounded_v2": GROUNDED_V2,
    "grounded_v3": GROUNDED_V3,
}


# Iter02 designs: match concise's output structure (State Understanding +
# Prediction) so pure-style diffs shrink; only real content divergence remains.

GROUNDED_V4 = """You are forecasting a plausible future for a Craftax state.

Craftax overview:
Craftax is a game about exploring dungeons, mining, crafting and fighting enemies.

Use these game rules:
1) Coordinates are (Row, Column), centered on the player at (0,0).
   - Negative Row = UP, Positive Row = DOWN.
   - Negative Column = LEFT, Positive Column = RIGHT.
2) Intrinsics: Health, Food, Drink, Energy, Mana. All out of 9.
   - Food/Drink/Energy naturally decay.
   - If Food/Drink/Energy reaches 0, Health will decay.
   - If Food/Drink/Energy are maintained, Health can recover.
3) Floor progression uses ladders.

Here is a good algorithm the player will play the game by:
At every step, the player should act with the goal of staying alive and progressing
down floors. Priorities in order:
1. Survive (restore low Food/Drink/Energy, avoid damage).
2. Take the ladder if it is open and on-screen.
3. Upgrade equipment if survival is stable.
4. Explore for resources, troops, and the ladder.

Output two sections:

State Understanding: briefly describe the current situation and what matters most
(1-3 sentences). Use relative directions (up/down/left/right/up-left/...) and
qualitative distances (adjacent/nearby/a few tiles/far). Do NOT reference
numerical (row, col) coordinates.

Prediction: a single 1-2 sentence forecast of the next few steps, in the voice of
these examples:
Prediction: Move right to the cluster of trees and chop wood for basic tools.
Prediction: Chase and kill the cow directly above to restore food.
Prediction: Move down-left to the nearby water tiles to drink, then look for a safe stone cluster to sleep in.
Prediction: Move up and left to the visible open ladder and descend to the next floor.

Now, predict the future of the following state.

Current state:
{current_state_filtered}

(Helpful context: in five steps the state becomes --
{future_state_filtered}
--. Use this to inform your Prediction, but phrase it as a forecast in the same
voice as the examples.)
"""

# V5: identical wrapper to V4, but hide the actions (only future state), so
# Gemini must phrase as forecast without being told the literal action sequence.
GROUNDED_V5 = """You are forecasting a plausible future for a Craftax state.

Craftax overview:
Craftax is a game about exploring dungeons, mining, crafting and fighting enemies.

Rules: Coords (Row, Col) with player at (0,0). Intrinsics Health/Food/Drink/
Energy/Mana out of 9. Floor progression uses ladders. Survive, ladder, upgrade,
explore.

Output two sections.

State Understanding: briefly describe the current situation. Use relative
directions and qualitative distances; no (row, col) coordinates.

Prediction: a single 1-2 sentence forecast, in the voice of:
Prediction: Move right to the cluster of trees and chop wood for basic tools.
Prediction: Chase and kill the cow directly above to restore food.
Prediction: Move down-left to the nearby water tiles to drink, then look for a safe stone cluster to sleep in.
Prediction: Move up and left to the visible open ladder and descend to the next floor.

Current state:
{current_state_filtered}

(Context only, do not mention: the state five steps ahead will be --
{future_state_filtered}
--. Let this inform your Prediction, but phrase it as a forecast in the voice
of the examples.)
"""

# V6: start from the EXACT concise template text and bolt the future context on
# without otherwise modifying voice. This is the minimal-diff variant.
V6_PREFIX_READS_CONCISE = True  # filled in at load time
GROUNDED_V6_TEMPLATE_MARKER = "__USE_CONCISE_AS_PREFIX__"


GROUNDED_VARIANTS.update({
    "grounded_v4": GROUNDED_V4,
    "grounded_v5": GROUNDED_V5,
    "grounded_v6": GROUNDED_V6_TEMPLATE_MARKER,  # resolved later
})


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_episodes_with_future(
    npz_path: Path,
    num_samples: int,
    future_offset: int = FUTURE_OFFSET,
    rng_seed: int = 0,
) -> List[Dict]:
    """Pick `num_samples` (t, t+k) pairs that stay within an episode.

    Returns list of dicts: {t, obs_t, obs_future, actions_csv}.
    """
    print(f"Loading {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    obs = decode_obs_from_bitpacked(d)
    actions = d["action"].astype(int)
    done = d["done"].astype(bool)
    T = len(obs)

    # Only pick t where t+future_offset is still in the same episode.
    # Episode break = done[i] = 1, so the next episode starts at i+1.
    # For simplicity: valid if none of done[t:t+future_offset] is True.
    valid_t = []
    for t in range(T - future_offset):
        if not done[t : t + future_offset].any():
            valid_t.append(t)
    print(f"  {len(valid_t)} / {T - future_offset} candidate indices have a clean future window")

    rng = np.random.default_rng(rng_seed)
    picks = rng.choice(len(valid_t), size=min(num_samples, len(valid_t)), replace=False)
    ts = sorted(int(valid_t[p]) for p in picks)

    samples = []
    for t in ts:
        actions_csv = ",".join(
            ACTION_NAMES[int(a)] if 0 <= int(a) < len(ACTION_NAMES) else str(int(a))
            for a in actions[t : t + future_offset]
        )
        samples.append({
            "t": t,
            "obs_t_text": filter_text_obs(obs_to_text(obs[t])),
            "obs_future_text": filter_text_obs(obs_to_text(obs[t + future_offset])),
            "actions_csv": actions_csv,
        })
    return samples


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
def render_prompt(template: str, sample: Dict) -> str:
    return (
        template
        .replace("{current_state_filtered}", sample["obs_t_text"])
        .replace("{future_state_filtered}", sample["obs_future_text"])
        .replace("{actions_csv}", sample["actions_csv"])
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_text(text: str, api_key: str) -> np.ndarray:
    url = f"{EMBED_URL}?key={api_key}"
    return _embed_one_gemini(text, url, EMBED_DIM).astype(np.float32)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-golden", action="store_true",
                    help="Only probe the PSF (AWR) set")
    ap.add_argument("--skip-thinking", action="store_true",
                    help="Skip the thinking-variant probe")
    ap.add_argument("--grounded-only", nargs="+", default=None,
                    help="Limit grounded variants to these names")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset sample sets.
    datasets: List[Tuple[str, Path]] = [("psf", PSF_FILE)]
    if not args.skip_golden:
        datasets.append(("golden", GOLDEN_FILE))

    # Which prompts to run.
    variants: Dict[str, Dict] = {
        "concise":        {"template": CONCISE_TMPL.read_text(), "thinking_budget": 0,
                           "max_out": NORMAL_MAX_OUT},
        "concise_rerun":  {"template": CONCISE_TMPL.read_text(), "thinking_budget": 0,
                           "max_out": NORMAL_MAX_OUT},  # re-run to measure self-noise
    }
    if not args.skip_thinking:
        variants["thinking"] = {
            "template": THINKING_TMPL.read_text(),
            "thinking_budget": THINK_BUDGET,
            "max_out": THINK_MAX_OUT,
        }
    active_grounded = GROUNDED_VARIANTS
    if args.grounded_only:
        active_grounded = {k: v for k, v in GROUNDED_VARIANTS.items()
                           if k in args.grounded_only}
    concise_src = CONCISE_TMPL.read_text()
    for name, tmpl in active_grounded.items():
        if tmpl == GROUNDED_V6_TEMPLATE_MARKER:
            # V6 = concise template verbatim, then a trailing future-context block.
            tmpl = (
                concise_src.rstrip()
                + "\n\n(Helpful context only -- do not mention this block in your"
                " output: in five steps the state becomes --\n"
                "{future_state_filtered}\n"
                "--. Let this inform your Prediction, but phrase it in the same voice"
                " as the examples.)\n"
            )
        variants[name] = {"template": tmpl, "thinking_budget": 0,
                          "max_out": NORMAL_MAX_OUT}

    results_by_dataset: Dict[str, Dict] = {}

    for ds_name, ds_path in datasets:
        print(f"\n{'=' * 80}\nDATASET: {ds_name}\n{'=' * 80}")
        samples = sample_episodes_with_future(ds_path, args.num_samples, FUTURE_OFFSET,
                                              rng_seed=args.seed)

        per_variant_texts: Dict[str, List[str]] = {k: [] for k in variants}
        per_variant_preds: Dict[str, List[str]] = {k: [] for k in variants}
        per_variant_embeds: Dict[str, List[np.ndarray]] = {k: [] for k in variants}
        per_variant_tokens: Dict[str, List[Tuple[int, int, int]]] = {k: [] for k in variants}

        for i, sample in enumerate(samples):
            print(f"\n-- {ds_name} sample {i + 1}/{len(samples)} (t={sample['t']}) --")
            for vname, vcfg in variants.items():
                prompt = render_prompt(vcfg["template"], sample)
                r = call_gemini(
                    prompt, api_key, model=MODEL,
                    thinking_budget=vcfg["thinking_budget"],
                    max_output_tokens=vcfg["max_out"],
                )
                text = r.get("text", "") if r.get("ok", True) else ""
                if not text:
                    # retry once
                    print(f"    {vname}: empty output, retrying")
                    time.sleep(1)
                    r = call_gemini(
                        prompt, api_key, model=MODEL,
                        thinking_budget=vcfg["thinking_budget"],
                        max_output_tokens=vcfg["max_out"],
                    )
                    text = r.get("text", "") if r.get("ok", True) else ""
                pred_suffix, status = extract_prediction_suffix(text)
                emb = embed_text(pred_suffix, api_key)
                per_variant_texts[vname].append(text)
                per_variant_preds[vname].append(pred_suffix)
                per_variant_embeds[vname].append(emb)
                # token counts: prompt, completion, thoughts (if exposed)
                tok = (
                    int(r.get("prompt_tokens", 0) or 0),
                    int(r.get("completion_tokens", 0) or 0),
                    int(r.get("thoughts_tokens", 0) or 0),
                )
                per_variant_tokens[vname].append(tok)
                head = pred_suffix.split("\n")[0][:90]
                print(f"    {vname:16s}  tok(p/c/t)={tok}  status={status}  pred: {head}")

        # Cosine-sim analysis.
        print(f"\n--- {ds_name}: pairwise cosine-sim of predonly embeddings ---")
        vnames = list(variants.keys())
        cos_matrix = np.zeros((len(vnames), len(vnames)), dtype=np.float32)
        per_pair_samples: Dict[Tuple[str, str], List[float]] = {}
        for a_i, a in enumerate(vnames):
            for b_i, b in enumerate(vnames):
                sims = [cos_sim(per_variant_embeds[a][k], per_variant_embeds[b][k])
                        for k in range(len(samples))]
                per_pair_samples[(a, b)] = sims
                cos_matrix[a_i, b_i] = float(np.mean(sims))

        # pretty print
        w = max(len(v) for v in vnames) + 1
        print(" " * w + " | " + "  ".join(f"{v:>8s}" for v in vnames))
        print("-" * (w + 3 + len(vnames) * 10))
        for i, a in enumerate(vnames):
            row = f"{a:<{w}s} | " + "  ".join(f"{cos_matrix[i, j]:>8.4f}"
                                              for j in range(len(vnames)))
            print(row)

        # Distances to natural baseline (concise).
        print(f"\n--- {ds_name}: mean / std / min cosine sim vs concise (per variant) ---")
        header = f"{'variant':<16}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}  pass(>.85)"
        print(header)
        for v in vnames:
            sims = np.array(per_pair_samples[("concise", v)], dtype=np.float32)
            flag = "Y" if sims.mean() > 0.85 else ("~" if sims.mean() > 0.70 else "N")
            print(f"{v:<16}  {sims.mean():>7.4f}  {sims.std():>7.4f}  "
                  f"{sims.min():>7.4f}  {sims.max():>7.4f}  {flag}")

        results_by_dataset[ds_name] = {
            "samples_t": [s["t"] for s in samples],
            "variant_names": vnames,
            "cos_matrix": cos_matrix.tolist(),
            "per_pair_samples": {f"{a}|{b}": v for (a, b), v in per_pair_samples.items()},
            "token_counts": per_variant_tokens,
            "predictions": {v: per_variant_preds[v] for v in vnames},
            "raw_texts":   {v: per_variant_texts[v] for v in vnames},
        }

    out_path = args.out_dir / "probe_results.json"
    with out_path.open("w") as f:
        json.dump(results_by_dataset, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
