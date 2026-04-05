import redis
import os
import time
import logging
import argparse

parser = argparse.ArgumentParser(description="Janitor for Craftax")
parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Name of the subdirectory to look in (e.g., 'gene', 'max', or 'vansh')",
)
parser.add_argument(
    "--host", type=str, required=True, help="Which login node are you on? (e.g. login2)"
)
parser.add_argument(
    "--port", type=int, required=True, help="Port number of the Redis server"
)
args = parser.parse_args()

# --- Configuration ---
QUEUE_NAME = "craftax_job_queue"
UNLABELLED_DIR = f"/data/group_data/rl/geney/craftax_unlabelled_new/{args.name}/"
RESULTS_DIR = f"/data/group_data/rl/geney/craftax_labelled_results/{args.name}/"
SLEEP_INTERVAL = 300  # 5 minutes

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("Janitor")

# --- Redis Connection ---
r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
r.ping()
logger.info("Janitor successfully connected to Redis.")


def find_missing_jobs():
    unlabelled_files = {f for f in os.listdir(UNLABELLED_DIR) if f.endswith(".npz")}
    if not unlabelled_files:
        logger.warning(f"No files found in UNLABELLED_DIR: {UNLABELLED_DIR}")
        return set(), set()

    results_files = {f for f in os.listdir(RESULTS_DIR) if f.endswith(".npz")}
    missing_job_names = unlabelled_files - results_files

    in_queue_paths = set(r.lrange(QUEUE_NAME, 0, -1))
    in_queue_names = {os.path.basename(p) for p in in_queue_paths}

    jobs_to_queue = missing_job_names - in_queue_names
    job_paths_to_queue = {os.path.join(UNLABELLED_DIR, f) for f in jobs_to_queue}

    return job_paths_to_queue, missing_job_names


# --- Main Janitor Loop ---
logger.info("Janitor started. Will check for orphaned jobs...")
while True:
    try:
        job_paths_to_queue, total_missing = find_missing_jobs()

        if job_paths_to_queue:
            logger.info(f"Found {len(job_paths_to_queue)} orphaned jobs to re-queue.")
            for job_path in job_paths_to_queue:
                r.lpush(QUEUE_NAME, job_path)
                logger.info(f"  Re-queued: {job_path}")
        else:
            logger.info(
                f"No orphaned jobs found. Total missing/pending: {len(total_missing)}"
            )

        logger.info(f"Sleeping for {SLEEP_INTERVAL} seconds...")
        time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Janitor stopping.")
        break
    except Exception as e:
        logger.error(f"Janitor loop failed: {e}", exc_info=True)
        time.sleep(60)
