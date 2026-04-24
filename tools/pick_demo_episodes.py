"""Pick representative episodes from each eval condition for video review.

For each condition dir, pick:
  - highest-return episode  ("best case")
  - lowest-return episode   ("worst case")
  - episode closest to median return

Print video paths and gemini-text excerpts so the user can quickly find
the videos that demonstrate each behavior.

Usage:
  python tools/pick_demo_episodes.py --condition-dir <dir> [--condition-dir ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_eps(eval_dir: Path) -> List[dict]:
    eps = []
    for ep_dir in sorted(eval_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        sf = ep_dir / "summary.json"
        if not sf.exists():
            continue
        try:
            with sf.open() as f:
                d = json.load(f)
                d["_ep_dir"] = str(ep_dir)
            eps.append(d)
        except Exception:
            continue
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-dir", type=Path, action="append", required=True)
    args = ap.parse_args()

    for cdir in args.condition_dir:
        eps = load_eps(cdir)
        if not eps:
            print(f"\n=== {cdir} : NO EPISODES ===")
            continue
        eps_by_ret = sorted(eps, key=lambda e: e["return"])
        n = len(eps)
        picks = [
            ("LOW",  eps_by_ret[0]),
            ("MED",  eps_by_ret[n // 2]),
            ("HIGH", eps_by_ret[-1]),
        ]
        print(f"\n=== {cdir} (n={n}) ===")
        for tag, ep in picks:
            ep_dir = Path(ep["_ep_dir"])
            video = ep_dir / "gameplay.mp4"
            print(f"  [{tag}] ep{ep['episode']:02d}  return={ep['return']:.2f}  "
                  f"len={ep['length']}  num_ach={ep['num_achievements']}")
            print(f"        achievements: {sorted(ep.get('achievements', {}).keys())[:8]}{'...' if len(ep.get('achievements', {}))>8 else ''}")
            print(f"        video: {video}")
            # First few Gemini predictions for context
            log = ep_dir / "gemini_log.jsonl"
            if log.exists():
                with log.open() as f:
                    lines = list(f)
                if lines:
                    try:
                        first = json.loads(lines[0])
                        text = first.get("gemini_text", "").split("Prediction:")[-1].strip().split("\n")[0]
                        print(f"        first gemini pred: {text[:120]}")
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
