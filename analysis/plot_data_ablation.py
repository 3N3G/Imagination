#!/usr/bin/env python3
"""
Plot data quantity ablation: training samples vs online performance.

Reads results from eval_results/data_ablation/{aug,unaug}_f{N}_s{seed}/results.json
and generates a publication-quality figure.

Usage:
    python plot_data_ablation.py [--output data_ablation_results.png]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

EVAL_DIR = "/data/group_data/rl/geney/eval_results/data_ablation"
FILE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 126]
SAMPLES_PER_FILE = 100_682  # approximate average
SEEDS = [42, 123]


def load_results():
    """Load all results, grouped by (model_type, file_count)."""
    data = defaultdict(list)  # key: (type, file_count) -> list of per-episode returns

    for model_type in ["aug", "unaug"]:
        for fc in FILE_COUNTS:
            for seed in SEEDS:
                rpath = os.path.join(EVAL_DIR, f"{model_type}_f{fc}_s{seed}", "results.json")
                if not os.path.exists(rpath):
                    continue
                with open(rpath) as f:
                    res = json.load(f)
                episode_returns = [ep["return"] for ep in res["episodes"]]
                data[(model_type, fc)].append({
                    "seed": seed,
                    "mean_return": res["mean_return"],
                    "std_return": res["std_return"],
                    "episode_returns": episode_returns,
                    "mean_achievements": res["mean_achievements"],
                })

    return data


def compute_stats(data):
    """Compute per-file-count statistics across seeds."""
    stats = {}
    for (model_type, fc), seed_results in data.items():
        # Pool all episode returns across seeds
        all_returns = []
        all_achievements = []
        seed_means = []
        for sr in seed_results:
            all_returns.extend(sr["episode_returns"])
            all_achievements.append(sr["mean_achievements"])
            seed_means.append(sr["mean_return"])

        n_seeds = len(seed_results)
        n_episodes = len(all_returns)
        samples = fc * SAMPLES_PER_FILE

        stats[(model_type, fc)] = {
            "samples": samples,
            "n_seeds": n_seeds,
            "n_episodes": n_episodes,
            "mean_return": np.mean(all_returns),
            "std_return": np.std(all_returns),
            "sem_return": np.std(seed_means) / np.sqrt(n_seeds) if n_seeds > 1 else np.std(all_returns) / np.sqrt(n_episodes),
            "seed_means": seed_means,
            "mean_achievements": np.mean(all_achievements),
            "std_achievements": np.std(all_achievements) if n_seeds > 1 else 0,
        }

    return stats


def make_plot(stats, output_path):
    """Generate the data ablation figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"aug": "#2196F3", "unaug": "#FF9800"}
    labels = {"aug": "Augmented (w/ imagination)", "unaug": "Unaugmented (obs-only)"}
    markers = {"aug": "o", "unaug": "s"}

    for model_type in ["aug", "unaug"]:
        xs, ys_return, errs_return = [], [], []
        ys_ach, errs_ach = [], []
        seed_xs, seed_ys = [], []

        for fc in FILE_COUNTS:
            key = (model_type, fc)
            if key not in stats:
                continue
            s = stats[key]
            xs.append(s["samples"])
            ys_return.append(s["mean_return"])
            errs_return.append(s["sem_return"])
            ys_ach.append(s["mean_achievements"])
            errs_ach.append(s["std_achievements"])

            # Individual seed points
            for sm in s["seed_means"]:
                seed_xs.append(s["samples"])
                seed_ys.append(sm)

        if not xs:
            continue

        xs = np.array(xs)
        ys_return = np.array(ys_return)
        errs_return = np.array(errs_return)
        ys_ach = np.array(ys_ach)
        errs_ach = np.array(errs_ach)

        c = colors[model_type]
        label = labels[model_type]
        marker = markers[model_type]

        # Plot 1: Return
        ax1.plot(xs, ys_return, color=c, marker=marker, markersize=7,
                 linewidth=2, label=label, zorder=3)
        ax1.fill_between(xs, ys_return - errs_return, ys_return + errs_return,
                         alpha=0.2, color=c, zorder=2)
        # Individual seed means as faint dots
        ax1.scatter(seed_xs, seed_ys, color=c, alpha=0.3, s=20, zorder=1)

        # Plot 2: Achievements
        ax2.plot(xs, ys_ach, color=c, marker=marker, markersize=7,
                 linewidth=2, label=label, zorder=3)
        if np.any(errs_ach > 0):
            ax2.fill_between(xs, ys_ach - errs_ach, ys_ach + errs_ach,
                             alpha=0.2, color=c, zorder=2)

    # Format axes
    for ax in [ax1, ax2]:
        ax.set_xscale("log")
        ax.set_xlabel("Training Samples", fontsize=12)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

        # Custom tick labels for clarity
        tick_positions = [fc * SAMPLES_PER_FILE for fc in FILE_COUNTS]
        tick_labels = []
        for fc in FILE_COUNTS:
            n = fc * SAMPLES_PER_FILE
            if n >= 1_000_000:
                tick_labels.append(f"{n / 1_000_000:.1f}M")
            else:
                tick_labels.append(f"{n / 1_000:.0f}K")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)

    ax1.set_ylabel("Mean Episodic Return", fontsize=12)
    ax1.set_title("Online Return vs Training Data Quantity", fontsize=13)

    ax2.set_ylabel("Mean Achievements", fontsize=12)
    ax2.set_title("Achievements vs Training Data Quantity", fontsize=13)

    fig.suptitle("Data Quantity Ablation — AWR Offline Policy (w512, 100K steps)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    # Also save as PDF
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved plot to {pdf_path}")


def print_table(stats):
    """Print a summary table to stdout."""
    print("\n" + "=" * 90)
    print(f"{'Type':<8} {'Files':>5} {'Samples':>10} {'Seeds':>5} {'Episodes':>8} "
          f"{'Return':>10} {'SEM':>8} {'Achiev':>8}")
    print("-" * 90)
    for model_type in ["aug", "unaug"]:
        for fc in FILE_COUNTS:
            key = (model_type, fc)
            if key not in stats:
                continue
            s = stats[key]
            print(f"{model_type:<8} {fc:>5} {s['samples']:>10,} {s['n_seeds']:>5} "
                  f"{s['n_episodes']:>8} {s['mean_return']:>10.2f} "
                  f"{s['sem_return']:>8.2f} {s['mean_achievements']:>8.1f}")
    print("=" * 90)


def main():
    global EVAL_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data_ablation_results.png")
    parser.add_argument("--eval-dir", default=EVAL_DIR)
    args = parser.parse_args()

    EVAL_DIR = args.eval_dir

    data = load_results()
    if not data:
        print(f"No results found in {EVAL_DIR}/")
        print("Expected structure: {aug,unaug}_f{N}_s{seed}/results.json")
        sys.exit(1)

    stats = compute_stats(data)
    print_table(stats)
    make_plot(stats, args.output)


if __name__ == "__main__":
    main()
