"""
Multistep counterfactual direction-match analyzer.

Reads {policy}/episodes.jsonl produced by eval.eval_direction_counterfactual_multistep,
groups records by (policy, step), prints per-step:
  - obs_flip_changes_action  (fraction where A→B action differs)
  - emb_flip_changes_action  (fraction where A→C action differs)
  - match_vs_orig / match_vs_flipped for each condition A/B/C
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

COORD_RE = re.compile(r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)")
DIR_NEG_Y, DIR_POS_Y = 3, 4   # UP, DOWN
DIR_NEG_X, DIR_POS_X = 1, 2   # LEFT, RIGHT


def parse_first_coord(text: str):
    for m in COORD_RE.finditer(text or ""):
        dy, dx = int(m.group(1)), int(m.group(2))
        if dy == 0 and dx == 0:
            continue
        return dy, dx
    return None


def valid_dirs(dy: int, dx: int):
    d = set()
    if dy < 0: d.add(DIR_NEG_Y)
    elif dy > 0: d.add(DIR_POS_Y)
    if dx < 0: d.add(DIR_NEG_X)
    elif dx > 0: d.add(DIR_POS_X)
    return d


def rate(rows, action_key, coord_key):
    hits = n = 0
    for r in rows:
        c = r.get(coord_key)
        if c is None:
            continue
        if r.get(action_key) in valid_dirs(*c):
            hits += 1
        n += 1
    return (hits / n if n else None, n)


def flip_rate(rows, a_key, b_key):
    n = hits = 0
    for r in rows:
        a, b = r.get(a_key), r.get(b_key)
        if a is None or b is None:
            continue
        n += 1
        if a != b:
            hits += 1
    return hits / n if n else None


def chance(rows, coord_key):
    vals = [len(valid_dirs(*r[coord_key])) / 4.0 for r in rows if r.get(coord_key) is not None]
    return mean(vals) if vals else None


def fmt(v):
    return f"{v*100:5.1f}%" if v is not None else "   n/a"


def analyze(jsonl_path: Path):
    by_step = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            r["coord_orig"] = parse_first_coord(r.get("narrative_orig", ""))
            r["coord_flipped"] = parse_first_coord(r.get("narrative_flipped", ""))
            by_step[r["step"]].append(r)
    return by_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path,
                    default=Path("/data/group_data/rl/geney/eval_results"))
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--dump-json", type=Path, default=None)
    args = ap.parse_args()

    all_results = {}
    print("\n" + "=" * 116)
    print("MULTISTEP COUNTERFACTUAL DIRECTION MATCH")
    print("  A = (orig_obs, hidden_orig)       B = (flipped_obs, hidden_orig)       C = (orig_obs, hidden_flipped)")
    print("=" * 116)

    for name in args.policies:
        p = args.eval_root / name / "episodes.jsonl"
        if not p.exists():
            print(f"[warn] missing: {p}")
            continue
        by_step = analyze(p)
        per_step = {}

        print(f"\n── {name} " + "─" * (113 - len(name)))
        for step in sorted(by_step.keys()):
            rows = by_step[step]
            n = len(rows)
            no = sum(1 for r in rows if r["coord_orig"] is not None)
            nf = sum(1 for r in rows if r["coord_flipped"] is not None)
            cho = chance(rows, "coord_orig")
            chf = chance(rows, "coord_flipped")
            obs_change = flip_rate(rows, "action_A", "action_B")
            emb_change = flip_rate(rows, "action_A", "action_C")

            print(f"\n  step={step:3d}  n={n}  coord_orig:{no}  coord_flipped:{nf}  "
                  f"chance_orig={fmt(cho)}  chance_flipped={fmt(chf)}")
            print(f"    obs flip (A→B) changes chosen action in: {fmt(obs_change)}")
            print(f"    emb flip (A→C) changes chosen action in: {fmt(emb_change)}")
            print(f"                         match_vs_orig   match_vs_flipped")
            per_cond = {}
            for lbl, akey in [("A", "action_A"), ("B", "action_B"), ("C", "action_C")]:
                mo, _ = rate(rows, akey, "coord_orig")
                mf, _ = rate(rows, akey, "coord_flipped")
                per_cond[lbl] = {"match_vs_orig": mo, "match_vs_flipped": mf}
                print(f"    {lbl}  {akey:<10s}  {fmt(mo):>10s}          {fmt(mf):>10s}")

            per_step[step] = {
                "n": n, "n_coord_orig": no, "n_coord_flipped": nf,
                "chance_orig": cho, "chance_flipped": chf,
                "obs_flip_changes_action": obs_change,
                "emb_flip_changes_action": emb_change,
                "conditions": per_cond,
            }
        all_results[name] = per_step

    if args.dump_json:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {args.dump_json}")


if __name__ == "__main__":
    main()
