import redis
import glob
import os
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Push Craftax trajectory files to a Redis queue for LLM labeling."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Optional subdirectory name (e.g., 'gene'). Leave empty for root of data dir.",
    )

    # Redis Connection flags
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="Which login node are you on? (e.g. login2)",
    )
    parser.add_argument(
        "--port", type=int, default=6379, help="Redis port (default: 6379)"
    )
    parser.add_argument(
        "--queue", type=str, default="craftax_llm_job_queue", help="Name of the Redis queue"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print found files without pushing to Redis",
    )
    parser.add_argument(
        "--symbolic",
        action="store_true",
        help="Use symbolic trajectory data directory",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Configuration Constants
    if args.symbolic:
        BASE_DATA_DIR = "/data/group_data/rl/geney/craftax_unlabelled_symbolic/"
        RESULTS_DIR = "/data/group_data/rl/geney/vllm_craftax_labelled_results/"
    else:
        # Fallback to original directory structure
        BASE_DATA_DIR = "/data/group_data/rl/geney/craftax_unlabelled_new/"
        RESULTS_DIR = "/data/group_data/rl/geney/vllm_craftax_labelled_results/"
    
    FILE_PATTERN = "trajectories_batch_*.npz"

    # Construct the full path
    data_dir = os.path.join(BASE_DATA_DIR, args.name)

    # Validation
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Find files
    search_path = os.path.join(data_dir, FILE_PATTERN)
    print(f"Searching for files in: {search_path}")
    file_paths = glob.glob(search_path)
    file_paths.sort()

    total_found = len(file_paths)

    if total_found == 0:
        print("No files found matching the pattern.")
        sys.exit(0)

    # Filter out already-completed files
    completed_files = set()
    if os.path.exists(RESULTS_DIR):
        completed_files = set(os.path.basename(f) for f in glob.glob(os.path.join(RESULTS_DIR, "*.npz")))
    
    pending_files = [f for f in file_paths if os.path.basename(f) not in completed_files]
    
    print(f"Already completed: {len(completed_files)} files")
    print(f"Pending: {len(pending_files)} files")

    if len(pending_files) == 0:
        print("All files already completed!")
        sys.exit(0)

    if args.dry_run:
        print(
            "Dry run enabled. No jobs pushed. Example file:",
            pending_files[0] if pending_files else "None",
        )
        return

    # Redis Connection
    try:
        r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
        r.ping()
        print(f"Connected to Redis at {args.host}:{args.port}")
    except redis.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        sys.exit(1)

    # Push to Queue
    print(f"Pushing jobs to queue: '{args.queue}'...")
    try:
        for path in pending_files:
            r.lpush(args.queue, path)
        print(f"Successfully pushed {len(pending_files)} jobs.")
    except Exception as e:
        print(f"Error pushing to Redis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
