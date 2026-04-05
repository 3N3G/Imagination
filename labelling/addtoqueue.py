import redis
import glob
import os
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Push Craftax trajectory files to a Redis queue."
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the subdirectory to look in (e.g., 'gene', 'max', or 'vansh')",
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
        "--queue", type=str, default="craftax_job_queue", help="Name of the Redis queue"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print found files without pushing to Redis",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Configuration Constants
    BASE_DATA_DIR = "/data/group_data/rl/craftax_unlabelled_new/"
    RESULTS_DIR = "/data/group_data/rl/craftax_resume_labelled_results/"
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

    print(f"Found {total_found} files. ")

    if args.dry_run:
        print(
            "Dry run enabled. No jobs pushed. Example file:",
            file_paths[0] if file_paths else "None",
        )
        return

    # Redis Connection
    try:
        r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
        r.ping()  # Test connection
        print(f"Connected to Redis at {args.host}:{args.port}")
    except redis.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        sys.exit(1)

    # Push to Queue
    print(f"Pushing jobs to queue: '{args.queue}'...")

    try:
        with r.pipeline() as pipe:
            for path in file_paths:
                pipe.lpush(args.queue, path)
            pipe.execute()
        print(f"Successfully pushed {len(file_paths)} jobs.")
    except Exception as e:
        print(f"An error occurred while pushing to Redis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
