"""Retroactive wandb upload for a gemini_play run whose wandb.init failed.

Usage:
    python -m tools.wandb_sync_gemini_play \
        --run-dir /data/group_data/rl/geney/eval_results/gemini_play_survivefix_3flash \
        --wandb-name gemini_play_survivefix_3flash
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--wandb-name", required=True)
    p.add_argument("--project", default="craftax-llm-harnessed")
    p.add_argument("--entity", default="iris-sobolmark")
    args = p.parse_args()

    run_dir = args.run_dir
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No summary.json at {summary_path}")
    overall = json.loads(summary_path.read_text())

    ep_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    if not ep_dirs:
        raise SystemExit(f"No episode dirs in {run_dir}")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.wandb_name,
        config={
            "eval_type": "gemini_plays_craftax",
            "model": overall.get("model"),
            "n_episodes": overall.get("n_episodes"),
            "retroactive_sync": True,
        },
    )

    for i, ep_dir in enumerate(ep_dirs, start=1):
        ep_summary = json.loads((ep_dir / "summary.json").read_text())
        log = {
            "episode/return": ep_summary.get("return"),
            "episode/length": ep_summary.get("length"),
            "episode/parse_fail": ep_summary.get("parse_fail"),
            "episode/api_fail": ep_summary.get("api_fail"),
            "episode/mean_latency_s": ep_summary.get("mean_latency"),
        }
        video_path = ep_dir / "gameplay.mp4"
        if video_path.exists():
            try:
                log[f"video/episode_{i:02d}"] = wandb.Video(str(video_path), fps=15, format="mp4")
            except Exception as e:
                print(f"  video upload failed for {ep_dir.name}: {e}")
        wandb.log(log, step=i)
        print(f"  logged {ep_dir.name}: return={ep_summary.get('return'):+.2f}")

    wandb.log({
        "summary/return_mean": overall["return_mean"],
        "summary/return_std": overall["return_std"],
        "summary/return_min": overall["return_min"],
        "summary/return_max": overall["return_max"],
        "summary/length_mean": overall["length_mean"],
    })
    wandb.finish()
    print(f"Done: {run.url}")


if __name__ == "__main__":
    main()
