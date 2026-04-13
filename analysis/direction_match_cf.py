"""
Counterfactual direction-match analyzer.

Reads {policy}/episodes.jsonl produced by eval.eval_direction_counterfactual,
parses first non-origin (dy, dx) from each of (narrative_orig, narrative_flipped),
then for each condition {A=orig+orig, B=flip+orig, C=orig+flip} computes match
rate against BOTH candidate targets:

  match_vs_orig     = policy action ∈ valid_dirs(coord_orig)
  match_vs_flipped  = policy action ∈ valid_dirs(coord_flipped)

Interpretation:
  obs drives       → in B, match flips (orig↓, flipped↑); in C, match stays on orig
  embedding drives → in B, match stays on orig; in C, match flips
  neither          → all ≈ chance
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

COORD_RE = re.compile(r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)")
DIR_NEG_Y = 3   # UP
DIR_POS_Y = 4   # DOWN
DIR_NEG_X = 1   # LEFT
DIR_POS_X = 2   # RIGHT
ACTION_NAMES = {1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN"}


def parse_first_coord(text: str) -> tuple[int, int] | None:
    for m in COORD_RE.finditer(text or ""):
        dy, dx = int(m.group(1)), int(m.group(2))
        if dy == 0 and dx == 0:
            continue
        return dy, dx
    return None


def valid_dirs(dy: int, dx: int) -> set[int]:
    d: set[int] = set()
    if dy < 0: d.add(DIR_NEG_Y)
    elif dy > 0: d.add(DIR_POS_Y)
    if dx < 0: d.add(DIR_NEG_X)
    elif dx > 0: d.add(DIR_POS_X)
    return d


def analyze_policy(jsonl_path: Path) -> dict:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            coord_o = parse_first_coord(r.get("narrative_orig", ""))
            coord_f = parse_first_coord(r.get("narrative_flipped", ""))
            rows.append({**r, "coord_orig": coord_o, "coord_flipped": coord_f})

    def _rate(rows, action_key, coord_key):
        hits, n = 0, 0
        for r in rows:
            c = r.get(coord_key)
            if c is None:
                continue
            valid = valid_dirs(*c)
            if r.get(action_key) in valid:
                hits += 1
            n += 1
        return (hits / n if n else None, n)

    def _chance(rows, coord_key):
        vals = [len(valid_dirs(*r[coord_key])) / 4.0 for r in rows if r.get(coord_key) is not None]
        return mean(vals) if vals else None

    summary = {
        "policy": jsonl_path.parent.name,
        "n_total": len(rows),
        "n_coord_orig": sum(1 for r in rows if r["coord_orig"] is not None),
        "n_coord_flipped": sum(1 for r in rows if r["coord_flipped"] is not None),
        "chance_orig": _chance(rows, "coord_orig"),
        "chance_flipped": _chance(rows, "coord_flipped"),
        "conditions": {},
        "rows": rows,
    }
    for label, action_key in [("A", "action_A"), ("B", "action_B"), ("C", "action_C")]:
        mo, no = _rate(rows, action_key, "coord_orig")
        mf, nf = _rate(rows, action_key, "coord_flipped")
        summary["conditions"][label] = {
            "action_key": action_key,
            "match_vs_orig": mo, "n_orig": no,
            "match_vs_flipped": mf, "n_flipped": nf,
        }

    # Flip-action agreement (does the policy's action itself flip between B and A, or C and A?)
    def _flip_rate(rows, a_key, b_key):
        n = hits = 0
        for r in rows:
            a, b = r.get(a_key), r.get(b_key)
            if a is None or b is None:
                continue
            n += 1
            if a != b:
                hits += 1
        return hits / n if n else None
    summary["obs_flip_changes_action"] = _flip_rate(rows, "action_A", "action_B")
    summary["emb_flip_changes_action"] = _flip_rate(rows, "action_A", "action_C")
    return summary


def fmt(v):
    return f"{v*100:5.1f}%" if v is not None else "   n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path,
                    default=Path("/data/group_data/rl/geney/eval_results"))
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--dump-json", type=Path, default=None)
    args = ap.parse_args()

    results = []
    for name in args.policies:
        p = args.eval_root / name / "episodes.jsonl"
        if not p.exists():
            print(f"[warn] missing: {p}")
            continue
        results.append(analyze_policy(p))

    print("\n" + "=" * 110)
    print("COUNTERFACTUAL DIRECTION MATCH")
    print(f"  Condition A = (orig_obs,    hidden_orig)    # baseline")
    print(f"  Condition B = (flipped_obs, hidden_orig)    # obs counterfactual")
    print(f"  Condition C = (orig_obs,    hidden_flipped) # embedding counterfactual")
    print("=" * 110)
    for r in results:
        print(f"\n{r['policy']}   n={r['n_total']}  "
              f"coord_orig:{r['n_coord_orig']}  coord_flipped:{r['n_coord_flipped']}  "
              f"chance_orig={fmt(r['chance_orig'])}  chance_flipped={fmt(r['chance_flipped'])}")
        print(f"  obs flip  (A→B) changes chosen action in: {fmt(r['obs_flip_changes_action'])}")
        print(f"  emb flip  (A→C) changes chosen action in: {fmt(r['emb_flip_changes_action'])}")
        print(f"                       match_vs_orig   match_vs_flipped")
        for lbl in ["A", "B", "C"]:
            c = r["conditions"][lbl]
            print(f"  {lbl}  {c['action_key']:<10s}  {fmt(c['match_vs_orig']):>10s}          "
                  f"{fmt(c['match_vs_flipped']):>10s}")

    if args.dump_json:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.dump_json}")


if __name__ == "__main__":
    main()
