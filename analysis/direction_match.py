"""
Direction-match analyzer.

For each episode in each policy's eval_results/<policy>/episode_XX/ dir:
  1. Parse the step-0 Gemini narrative for the first (dy, dx) coord
  2. Compute the valid direction set from coord signs
  3. Find the first movement action in summary.json["actions"]
  4. Record match + chance baseline (valid_dirs / 4)

Coord semantics (Craftax):   (Row, Column) = (dY, dX)
  Row:    -=UP   +=DOWN
  Column: -=LEFT +=RIGHT
Actions: NOOP=0, LEFT=1, RIGHT=2, UP=3, DOWN=4
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

MOVE_ACTIONS = {1, 2, 3, 4}
DIR_FROM_COORD = {
    "dy_neg": 3,  # UP
    "dy_pos": 4,  # DOWN
    "dx_neg": 1,  # LEFT
    "dx_pos": 2,  # RIGHT
}
ACTION_NAMES = {0: "NOOP", 1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN"}

# Coord pattern: "(-3, 0)", "(1,1)", "(+2, -1)". Reject single numbers or 3-tuples.
COORD_RE = re.compile(r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)")


def valid_dirs_from_coord(dy: int, dx: int) -> set[int]:
    dirs: set[int] = set()
    if dy < 0:
        dirs.add(DIR_FROM_COORD["dy_neg"])
    elif dy > 0:
        dirs.add(DIR_FROM_COORD["dy_pos"])
    if dx < 0:
        dirs.add(DIR_FROM_COORD["dx_neg"])
    elif dx > 0:
        dirs.add(DIR_FROM_COORD["dx_pos"])
    return dirs


def parse_first_coord(text: str) -> tuple[int, int] | None:
    """Return the first (dy, dx) coord in Gemini's narrative, or None.

    Skips the origin (0, 0) since it carries no direction signal.
    """
    for m in COORD_RE.finditer(text):
        dy, dx = int(m.group(1)), int(m.group(2))
        if dy == 0 and dx == 0:
            continue
        return dy, dx
    return None


def first_move_action(actions: list[int]) -> int | None:
    for a in actions:
        if a in MOVE_ACTIONS:
            return a
    return None


def analyze_policy(policy_dir: Path) -> dict:
    episode_dirs = sorted(policy_dir.glob("episode_*"))
    rows = []
    for ep_dir in episode_dirs:
        summary_path = ep_dir / "summary.json"
        gemini_path = ep_dir / "gemini_log.jsonl"
        if not summary_path.exists() or not gemini_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        actions = summary.get("actions")
        if not actions:
            continue

        # Read step-0 Gemini entry
        step0_text = None
        with open(gemini_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("step") == 0:
                    step0_text = entry.get("gemini_text", "")
                    break
        if step0_text is None:
            continue

        coord = parse_first_coord(step0_text)
        first_mv = first_move_action(actions)

        row = {
            "episode": summary.get("episode"),
            "return": summary.get("return"),
            "coord": coord,
            "first_move": first_mv,
        }
        if coord is not None and first_mv is not None:
            dy, dx = coord
            valid = valid_dirs_from_coord(dy, dx)
            row["valid_dirs"] = sorted(valid)
            row["match"] = first_mv in valid
            row["chance"] = len(valid) / 4.0
        rows.append(row)

    scored = [r for r in rows if "match" in r]
    summary: dict = {
        "policy": policy_dir.name,
        "n_episodes": len(rows),
        "n_scored": len(scored),
        "n_coord_missing": sum(1 for r in rows if r["coord"] is None),
        "n_no_movement": sum(
            1 for r in rows if r["first_move"] is None and r["coord"] is not None
        ),
        "match_rate": mean(1.0 if r["match"] else 0.0 for r in scored) if scored else None,
        "chance_rate": mean(r["chance"] for r in scored) if scored else None,
        "rows": rows,
    }
    return summary


def fmt_row(p: dict) -> str:
    mr = p["match_rate"]
    cr = p["chance_rate"]
    if mr is None:
        return f"{p['policy']:<40s}  n={p['n_scored']:>2d}/{p['n_episodes']:<2d}  (no scored eps)"
    lift = mr - cr
    return (
        f"{p['policy']:<40s}  "
        f"n={p['n_scored']:>2d}/{p['n_episodes']:<2d}  "
        f"match={mr*100:5.1f}%  chance={cr*100:5.1f}%  lift={lift*100:+5.1f}pp"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path,
                    default=Path("/data/group_data/rl/geney/eval_results"))
    ap.add_argument("--policies", nargs="+", required=True,
                    help="Subdir names under eval-root to analyze")
    ap.add_argument("--dump-json", type=Path, default=None,
                    help="Write detailed per-episode rows here")
    ap.add_argument("--print-episodes", action="store_true",
                    help="Print per-episode rows too")
    args = ap.parse_args()

    results = []
    for name in args.policies:
        pdir = args.eval_root / name
        if not pdir.exists():
            print(f"[warn] missing: {pdir}")
            continue
        results.append(analyze_policy(pdir))

    print(f"\n{'='*96}")
    print("DIRECTION-MATCH SUMMARY  (first policy movement vs first Gemini (dy,dx))")
    print(f"{'='*96}")
    for r in results:
        print(fmt_row(r))

    if args.print_episodes:
        for r in results:
            print(f"\n--- {r['policy']} ---")
            for row in r["rows"]:
                print(f"  ep{row.get('episode'):>2}  "
                      f"coord={row['coord']!s:<12}  "
                      f"first_move={ACTION_NAMES.get(row.get('first_move'), row.get('first_move'))!s:<6}  "
                      f"match={row.get('match')}  "
                      f"return={row.get('return'):.2f}"
                      if row.get('return') is not None else "")

    if args.dump_json:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.dump_json}")


if __name__ == "__main__":
    main()
