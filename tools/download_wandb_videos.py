"""Download specific episode videos from wandb runs.

Usage:
    python tools/download_wandb_videos.py --list          # list all mp4s in target runs
    python tools/download_wandb_videos.py --download      # download the configured targets
    python tools/download_wandb_videos.py --download-all  # download every mp4 in target runs

Output dir: ./wandb_videos/<run_id>/
"""
import argparse
import pathlib
import re

import wandb

ENTITY_PROJECT = "iris-sobolmark/craftax-offline-awr"

TARGETS = [
    ("w21fwecj", 50),
    ("w21fwecj", 48),
    ("y09770mm", 45),
    ("y09770mm", 39),
]


def episode_in_name(name: str, ep: int) -> bool:
    """Match common wandb video naming patterns for an episode index."""
    nums = re.findall(r"\d+", name)
    return str(ep) in nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="just list mp4s, no download")
    ap.add_argument("--download", action="store_true", help="download configured TARGETS")
    ap.add_argument("--download-all", action="store_true", help="download every mp4 found")
    ap.add_argument("--out", default="./wandb_videos", help="output root dir")
    args = ap.parse_args()

    if not (args.list or args.download or args.download_all):
        ap.error("pass --list, --download, or --download-all")

    api = wandb.Api()
    out_root = pathlib.Path(args.out)

    run_ids = sorted({r for r, _ in TARGETS})
    for run_id in run_ids:
        run = api.run(f"{ENTITY_PROJECT}/{run_id}")
        print(f"\n=== run {run_id} ({run.name}) ===")
        mp4s = [f for f in run.files() if f.name.endswith(".mp4")]
        for f in mp4s:
            print(f"  {f.name}  ({f.size} bytes)")

        if args.download_all:
            for f in mp4s:
                f.download(root=str(out_root / run_id), replace=True)
                print(f"  downloaded {f.name}")
        elif args.download:
            wanted_eps = [ep for r, ep in TARGETS if r == run_id]
            for ep in wanted_eps:
                matches = [f for f in mp4s if episode_in_name(f.name, ep)]
                if not matches:
                    print(f"  [WARN] no match for ep {ep}")
                    continue
                if len(matches) > 1:
                    print(f"  [WARN] {len(matches)} matches for ep {ep}: {[m.name for m in matches]}")
                for f in matches:
                    f.download(root=str(out_root / run_id), replace=True)
                    print(f"  downloaded ep {ep}: {f.name}")


if __name__ == "__main__":
    main()
