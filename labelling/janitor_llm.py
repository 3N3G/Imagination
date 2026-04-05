#!/usr/bin/env python3
"""
Janitor script for LLM labelling jobs.

Monitors for incomplete jobs (have progress files but no final output) and re-queues them.
Run this alongside workers to handle preempted jobs.

Usage:
    python janitor_llm.py --host $(cat /data/group_data/rl/geney/redis_host.txt) --interval 60
"""

import redis
import os
import glob
import argparse
import time
import json
from datetime import datetime

# Directories
RESULTS_DIR = "/data/group_data/rl/geney/vllm_craftax_labelled_results/"
PROGRESS_DIR = os.path.join(RESULTS_DIR, "progress")
TEMP_NPY_DIR = os.path.join(RESULTS_DIR, "temp_npy")
UNLABELLED_DIR = "/data/group_data/rl/geney/craftax_unlabelled_symbolic/"
QUEUE_NAME = "craftax_llm_job_queue"

# How long a job must be stale before re-queuing (seconds)
# If a progress file hasn't been updated in this time, assume the worker died
STALE_THRESHOLD = 600  # 10 minutes


def get_progress_age(progress_path):
    """Get how many seconds since the progress file was last modified."""
    try:
        mtime = os.path.getmtime(progress_path)
        return time.time() - mtime
    except:
        return float('inf')


def get_completed_jobs():
    """Get set of completed job basenames (without .npz extension)."""
    completed = set()
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.npz")):
        basename = os.path.basename(f)
        completed.add(basename)
    return completed


def get_queued_jobs(r):
    """Get set of jobs currently in the queue."""
    queued = set()
    # Use LRANGE to peek at queue without modifying it
    jobs = r.lrange(QUEUE_NAME, 0, -1)
    for job in jobs:
        if isinstance(job, bytes):
            job = job.decode()
        queued.add(os.path.basename(job))
    return queued


def find_incomplete_jobs():
    """Find jobs that have progress files but no final output."""
    incomplete = []
    
    progress_files = glob.glob(os.path.join(PROGRESS_DIR, "*_progress.json"))
    completed = get_completed_jobs()
    
    for pf in progress_files:
        # Extract the original filename: trajectories_batch_XXXXXX.npz
        basename = os.path.basename(pf).replace("_progress.json", "")
        
        # Check if already completed
        if basename in completed:
            continue
        
        # Check if the corresponding source file exists
        source_file = os.path.join(UNLABELLED_DIR, basename)
        if not os.path.exists(source_file):
            print(f"[WARN] Progress file for {basename} but source not found")
            continue
        
        # Get age of progress file
        age = get_progress_age(pf)
        
        incomplete.append({
            "basename": basename,
            "source_file": source_file,
            "progress_file": pf,
            "age_seconds": age
        })
    
    return incomplete


def run_janitor(r, interval, dry_run=False):
    """Main janitor loop."""
    print(f"[JANITOR] Starting. Checking every {interval}s. Stale threshold: {STALE_THRESHOLD}s")
    print(f"[JANITOR] Queue: {QUEUE_NAME}")
    print(f"[JANITOR] Results dir: {RESULTS_DIR}")
    print("-" * 60)
    
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get current state
        queue_size = r.llen(QUEUE_NAME)
        completed = get_completed_jobs()
        queued = get_queued_jobs(r)
        incomplete = find_incomplete_jobs()
        
        # Filter to stale jobs not already in queue
        stale_jobs = [j for j in incomplete 
                      if j["age_seconds"] > STALE_THRESHOLD 
                      and j["basename"] not in queued]
        
        print(f"[{now}] Queue: {queue_size} | Completed: {len(completed)} | "
              f"Incomplete: {len(incomplete)} | Stale (re-queue): {len(stale_jobs)}")
        
        # Re-queue stale jobs
        for job in stale_jobs:
            if dry_run:
                print(f"  [DRY-RUN] Would re-queue: {job['basename']} (stale {job['age_seconds']:.0f}s)")
            else:
                r.lpush(QUEUE_NAME, job["source_file"])
                print(f"  [RE-QUEUED] {job['basename']} (was stale for {job['age_seconds']:.0f}s)")
        
        # Sleep
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Janitor for LLM labelling jobs")
    parser.add_argument("--host", required=True, help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually re-queue, just print")
    parser.add_argument("--once", action="store_true", help="Run once and exit (don't loop)")
    args = parser.parse_args()
    
    # Connect to Redis
    try:
        r = redis.Redis(host=args.host, port=args.port, decode_responses=False)
        r.ping()
        print(f"[JANITOR] Connected to Redis at {args.host}:{args.port}")
    except redis.ConnectionError as e:
        print(f"[ERROR] Could not connect to Redis: {e}")
        return 1
    
    if args.once:
        # Run once
        incomplete = find_incomplete_jobs()
        queued = get_queued_jobs(r)
        stale = [j for j in incomplete 
                 if j["age_seconds"] > STALE_THRESHOLD 
                 and j["basename"] not in queued]
        
        print(f"Found {len(incomplete)} incomplete jobs, {len(stale)} stale")
        for job in stale:
            if args.dry_run:
                print(f"  [DRY-RUN] Would re-queue: {job['basename']}")
            else:
                r.lpush(QUEUE_NAME, job["source_file"])
                print(f"  [RE-QUEUED] {job['basename']}")
        return 0
    
    # Run continuous loop
    try:
        run_janitor(r, args.interval, args.dry_run)
    except KeyboardInterrupt:
        print("\n[JANITOR] Stopped by user")
        return 0


if __name__ == "__main__":
    exit(main())
