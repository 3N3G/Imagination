#!/usr/bin/env python3
"""
Main orchestrator for the imagination-augmented offline RL pipeline.

Pipeline phases (run via SLURM chain or manually):
  Phase 3: Filter raw trajectories by episode return, bitpack, compute RTG
  Phase 4: Generate Gemini oracle labels on filtered data
  Phase 5: Extract Qwen3-8B structured embeddings from Gemini text
  Phase 6: Merge embeddings + text into final trajectory files

Each phase is independently resumable. Run individually:
    python -m pipeline.filter_and_repack --input_dir ... --output_dir ...
    python -m pipeline.gemini_label --api-key YOUR_KEY
    python -m pipeline.embed --device cuda
    python -m pipeline.merge

Usage:
    python -m pipeline.run --api-key YOUR_GEMINI_KEY [--steps 4,5,6]
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Imagination-augmented offline RL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  4  Generate Gemini oracle labels (requires --api-key or GEMINI_API_KEY)
  5  Extract Qwen3-8B structured embeddings (requires GPU)
  6  Merge embeddings + text into final trajectory files

Examples:
  # Run full pipeline (phases 4-6)
  python -m pipeline.run --api-key YOUR_KEY

  # Just Gemini labelling
  python -m pipeline.run --steps 4 --api-key YOUR_KEY

  # Embed + merge after Gemini is done
  python -m pipeline.run --steps 5,6
""",
    )
    parser.add_argument("--steps", type=str, default="4,5,6",
                        help="Comma-separated list of phases to run (default: 4,5,6)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for embedding extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for embedding model")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max files to process per step (for testing)")
    args = parser.parse_args()

    steps = set(int(s.strip()) for s in args.steps.split(","))
    t0 = time.time()

    print("=" * 70)
    print("  Imagination-Augmented Offline RL Pipeline")
    print("  Gemini Oracle Labels + Qwen3-8B Structured Embeddings")
    print("=" * 70)
    print(f"  Phases to run: {sorted(steps)}")
    print()

    # Phase 4: Gemini labelling
    if 4 in steps:
        api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("ERROR: Phase 4 requires Gemini API key.")
            print("  Pass --api-key or set GEMINI_API_KEY environment variable.")
            sys.exit(1)
        from pipeline.gemini_label import run as run_gemini
        run_gemini(
            api_key=api_key,
            max_files=args.max_files,
        )
        print()

    # Phase 5: Embedding extraction
    if 5 in steps:
        from pipeline.embed import run as run_embed
        run_embed(
            batch_size=args.batch_size,
            device=args.device,
            max_files=args.max_files,
        )
        print()

    # Phase 6: Merge
    if 6 in steps:
        from pipeline.merge import run as run_merge
        run_merge(max_files=args.max_files)
        print()

    elapsed = time.time() - t0
    print(f"\nPipeline complete. Total time: {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
