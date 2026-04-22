"""Build PSF size-ablation subsets.

Rank episodes (delimited by done=True) in the merged PSF cadence=5 dir by
return_to_go[ep_start] descending, then extract rows from the top episodes
until each target row-count is reached. Produces one subset dir per target.

We drop `text_generated` (not needed for AWR training) but KEEP `hidden_state`
in fp16 so the same subsets work for the augmented policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(
    "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
    "/final_trajectories_psf_v2_cadence5_gemini_emb"
)
DEFAULT_OUTPUT_BASE = Path(
    "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
    "/psf_size_ablation_subsets"
)

# Keys to carry forward. We skip text_generated (variable-length strings).
KEYS_KEEP = [
    "obs_map_bits",
    "obs_aux",
    "action",
    "reward",
    "done",
    "log_prob",
    "return_to_go",
    "hidden_state",
    "gemini_step_idx",
]
# Scalars to propagate (same value across every output file).
SCALAR_KEYS = ["obs_map_dim"]


def find_episodes(done: np.ndarray):
    done_idx = np.where(done)[0]
    eps = []
    start = 0
    for di in done_idx:
        eps.append((start, di + 1))
        start = di + 1
    return eps


def scan_episodes(files):
    """Return list of (file_idx, ep_start, ep_end, return_at_start)."""
    t0 = time.time()
    meta = []
    total_rows = 0
    for fi, f in enumerate(files):
        d = np.load(f, allow_pickle=False)
        done = np.asarray(d["done"]).reshape(-1).astype(bool)
        rtg = np.asarray(d["return_to_go"]).reshape(-1).astype(np.float32)
        total_rows += len(done)
        d.close()
        for s, e in find_episodes(done):
            meta.append((fi, s, e, float(rtg[s])))
        if (fi + 1) % 20 == 0 or fi == 0:
            print(f"  scanned {fi + 1}/{len(files)}   "
                  f"episodes={len(meta)}   rows={total_rows}   "
                  f"elapsed={time.time() - t0:.1f}s")
    print(f"  done scan: {len(meta)} episodes, {total_rows} rows, "
          f"{time.time() - t0:.1f}s")
    return meta, total_rows


def pick_subset(meta_sorted, target_rows):
    """Walk sorted episode meta and collect entries until cumulative rows >= target."""
    picked = []
    rows = 0
    for entry in meta_sorted:
        fi, s, e, r = entry
        picked.append(entry)
        rows += (e - s)
        if rows >= target_rows:
            break
    return picked, rows


def write_subset(picked, files, out_dir: Path, per_file_rows: int = 100_000):
    """Group picked episodes by file, extract row-slices, repack into fresh npz files.

    We aim for ~per_file_rows per output file to keep sizes similar to source.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_file: dict[int, list[tuple[int, int]]] = {}
    for fi, s, e, _ in picked:
        by_file.setdefault(fi, []).append((s, e))

    # Read a sample source file for scalar propagation.
    sample = np.load(files[0], allow_pickle=False)
    scalars = {k: sample[k].copy() for k in SCALAR_KEYS if k in sample.files}
    sample.close()

    # Iterate source files in order, stream rows into output shards.
    buf: dict[str, list[np.ndarray]] = {k: [] for k in KEYS_KEEP}
    buf_rows = 0
    out_idx = 0
    total_written = 0

    def flush():
        nonlocal buf, buf_rows, out_idx, total_written
        if buf_rows == 0:
            return
        payload = {k: np.concatenate(buf[k], axis=0) for k in KEYS_KEEP}
        payload.update(scalars)
        out_path = out_dir / f"trajectories_{out_idx:06d}.npz"
        np.savez(out_path, **payload)
        total_written += buf_rows
        print(f"    wrote {out_path.name}  rows={buf_rows}  "
              f"cum={total_written}")
        buf = {k: [] for k in KEYS_KEEP}
        buf_rows = 0
        out_idx += 1

    for fi in sorted(by_file.keys()):
        d = np.load(files[fi], allow_pickle=False)
        arrs = {k: d[k] for k in KEYS_KEEP}
        d.close()
        for s, e in by_file[fi]:
            for k in KEYS_KEEP:
                buf[k].append(arrs[k][s:e])
            buf_rows += (e - s)
            if buf_rows >= per_file_rows:
                flush()
    flush()
    print(f"  finished {out_dir.name}  total_rows={total_written}")
    return total_written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    p.add_argument("--targets", type=int, nargs="+",
                   default=[1_000_000, 2_000_000, 4_000_000, 8_000_000])
    p.add_argument("--per-file-rows", type=int, default=100_000)
    args = p.parse_args()

    files = sorted(args.input_dir.glob("trajectories_*.npz"))
    print(f"Input: {args.input_dir}  files={len(files)}")
    print(f"Output base: {args.output_base}")
    print(f"Targets: {args.targets}")

    args.output_base.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 1: scan episodes ---")
    meta, total_rows = scan_episodes(files)
    meta_sorted = sorted(meta, key=lambda x: x[3], reverse=True)
    returns = np.array([m[3] for m in meta_sorted], dtype=np.float32)
    print(f"  return distribution: min={returns.min():.2f} "
          f"max={returns.max():.2f} mean={returns.mean():.2f} "
          f"median={float(np.median(returns)):.2f}")
    print(f"  top-1% cutoff: {float(np.percentile(returns, 99)):.2f}")

    manifest = {
        "input_dir": str(args.input_dir),
        "total_rows": int(total_rows),
        "total_episodes": len(meta),
        "return_stats": {
            "min": float(returns.min()),
            "max": float(returns.max()),
            "mean": float(returns.mean()),
            "median": float(np.median(returns)),
        },
        "subsets": {},
    }

    for target in sorted(args.targets):
        tag = f"top{target // 1_000_000}M"
        out_dir = args.output_base / f"final_trajectories_psf_v2_cadence5_gemini_emb_{tag}"
        print(f"\n--- Phase 2 ({tag}): pick + write subset to {out_dir} ---")
        picked, actual_rows = pick_subset(meta_sorted, target)
        kept_returns = np.array([e[3] for e in picked], dtype=np.float32)
        print(f"  picked {len(picked)} episodes, "
              f"target={target} actual_rows={actual_rows}   "
              f"return range kept: [{kept_returns.min():.2f}, "
              f"{kept_returns.max():.2f}]   "
              f"return cutoff (min kept): {kept_returns.min():.2f}")
        written = write_subset(picked, files, out_dir,
                               per_file_rows=args.per_file_rows)
        manifest["subsets"][tag] = {
            "target_rows": target,
            "actual_rows": int(written),
            "n_episodes": len(picked),
            "return_min_kept": float(kept_returns.min()),
            "return_max_kept": float(kept_returns.max()),
            "output_dir": str(out_dir),
        }

    manifest_path = args.output_base / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
