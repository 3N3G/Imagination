"""
LLM Worker for Craftax Data Labelling

Processes trajectory data through vLLM server to extract last-token hidden state
representations from game observations.

Supports two modes (controlled by GENERATE_TEXT flag):
1. Direct extraction (GENERATE_TEXT=False, default):
   - Extracts last-token hidden states directly from prompts
   - ~34x faster, no text generation
   - Suitable for training policies

2. Generation mode (GENERATE_TEXT=True):
   - Generates text reasoning first, then takes last-token hidden state
   - Slower but provides both text and hidden states
   - Useful for analysis or when text is needed

Hidden state output format: (N, hidden_size) — last-token hidden state only.
See docs/progress_journal.md for rationale.

Requires vLLM server running:
  bash scripts/start_vllm_hidden.sh --mode last_token
"""

import redis
import numpy as np
import os
import time
import logging
import socket
import wandb
import sys
import json
import requests
import shutil
from contextlib import suppress
from obs_to_text import obs_to_text  # Decode symbolic observations to text

from llm.extractor import VLLMHiddenStateExtractor
from llm.prompts import filter_text_obs

# --- Constants ---
QUEUE_NAME = os.environ.get("QUEUE_NAME", "craftax_llm_job_queue")
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR", "/data/group_data/rl/geney/vllm_craftax_labelled_results/"
)
LOCAL_WORK_DIR = os.environ.get(
    "WORKER_LOCAL_DIR",
    os.environ.get("TMPDIR", f"/tmp/{os.environ.get('USER', 'user')}/craftax_llm_worker"),
)
LOGS_DIR = os.environ.get("LOGS_DIR", os.path.join(LOCAL_WORK_DIR, "worker_logs"))
TEMP_NPY_DIR = os.path.join(LOCAL_WORK_DIR, "temp_npy")
PROGRESS_DIR = os.path.join(LOCAL_WORK_DIR, "progress")

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
BATCH_SIZE = 16  # 32 OOMed, 16 should be safe with ~50% headroom
TOKENS_GENERATED = 256  # Token budget for thinking + answer
GENERATE_TEXT = False  # Set to True to generate text before extracting hidden states
RETRY_HASH_NAME = os.environ.get("RETRY_HASH_NAME", "craftax_llm_job_retry_counts")
COMPLETED_COUNTER_NAME = os.environ.get(
    "COMPLETED_COUNTER_NAME", "craftax_llm_completed_count"
)
MAX_JOB_RETRIES = int(os.environ.get("MAX_JOB_RETRIES", "2"))
FLUSH_EVERY_BATCHES = max(1, int(os.environ.get("FLUSH_EVERY_BATCHES", "8")))
WANDB_LOG_EVERY_BATCHES = max(1, int(os.environ.get("WANDB_LOG_EVERY_BATCHES", "16")))
PROGRESS_LOG_EVERY_BATCHES = max(1, int(os.environ.get("PROGRESS_LOG_EVERY_BATCHES", "16")))
RESULT_SAVE_RETRIES = max(1, int(os.environ.get("RESULT_SAVE_RETRIES", "5")))
RESULT_SAVE_RETRY_SEC = float(os.environ.get("RESULT_SAVE_RETRY_SEC", "1.5"))
REDIS_RETRY_INITIAL_SLEEP_SEC = float(os.environ.get("REDIS_RETRY_INITIAL_SLEEP_SEC", "2.0"))
REDIS_RETRY_MAX_SLEEP_SEC = float(os.environ.get("REDIS_RETRY_MAX_SLEEP_SEC", "30.0"))
REDIS_RETRY_LOG_EVERY = max(1, int(os.environ.get("REDIS_RETRY_LOG_EVERY", "10")))
REDIS_MAX_CONSECUTIVE_ERRORS = int(os.environ.get("REDIS_MAX_CONSECUTIVE_ERRORS", "60"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "craftax_offline_llm_labelling")
WANDB_PER_FILE = os.environ.get("WANDB_PER_FILE", "0") == "1"
WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "0") == "1"

# --- mmap/save Constants ---
MAX_TEXT_LEN = 2048
TEXT_DTYPE = f'<U{MAX_TEXT_LEN}'
# Hidden states are saved as (N, hidden_size) — last-token only
# This matches what the policy network consumes and is ~256x smaller than per-token

# --- Standard Logging Setup ---
os.makedirs(LOCAL_WORK_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_NPY_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)
pid = os.getpid()
hostname = socket.gethostname()
logger = logging.getLogger(f"worker_{pid}")
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

file_handler_added = False
candidate_log_dirs = []
if LOGS_DIR:
    candidate_log_dirs.append(LOGS_DIR)
fallback_log_dir = os.path.join(LOCAL_WORK_DIR, "worker_logs")
if fallback_log_dir not in candidate_log_dirs:
    candidate_log_dirs.append(fallback_log_dir)

for log_dir in candidate_log_dirs:
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"worker_{hostname}_{pid}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Worker file log path: {log_filename}")
        file_handler_added = True
        break
    except OSError as log_exc:
        logger.warning(f"Failed to initialize file logging at {log_dir}: {log_exc}")

if not file_handler_added:
    logger.warning("Proceeding without file log due to filesystem/quota constraints.")

# Keep run artifacts off quota-limited home paths when invoked directly.
os.environ.setdefault("WANDB_DIR", os.path.join(LOCAL_WORK_DIR, "wandb"))
os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(LOCAL_WORK_DIR, "wandb_cache"))
os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(LOCAL_WORK_DIR, "wandb_config"))
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["WANDB_CONFIG_DIR"], exist_ok=True)

# --- Redis Connection ---
# Allow one-shot jobs to override Redis endpoint and avoid clobbering
# long-running coordinator jobs that use /data/.../redis_host.txt.
REDIS_HOST_FILE = "/data/group_data/rl/geney/redis_host.txt"
REDIS_HOST = os.environ.get("REDIS_HOST", "").strip()
if REDIS_HOST:
    logger.info(f"Using Redis host from env: {REDIS_HOST}")
else:
    try:
        with open(REDIS_HOST_FILE, 'r') as f:
            REDIS_HOST = f.read().strip()
        logger.info(f"Read Redis host from file: {REDIS_HOST}")
    except FileNotFoundError:
        REDIS_HOST = "login1"  # Fallback
        logger.warning(f"Redis host file not found, using fallback: {REDIS_HOST}")

REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
r.ping()
logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

# --- vLLM Server Check ---
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
logger.info(f"Checking for vLLM server at {VLLM_URL}...")
try:
    resp = requests.get(f"{VLLM_URL}/health", timeout=5)
    if resp.status_code != 200:
        raise Exception(f"Server returned status {resp.status_code}")
    logger.info(f"✅ vLLM server ready at {VLLM_URL}")
except Exception as e:
    logger.error(f"❌ vLLM server not available: {e}")
    logger.error(f"Please start the server first:")
    logger.error(f"  vllm serve configs/vllm_hidden_last --max-model-len 8192 --gpu-memory-utilization 0.95 \\")
    logger.error(f"    --kv-transfer-config '{{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{{\"shared_storage_path\":\"/tmp/hidden_states\",\"mode\":\"last_token\"}}}}'")
    sys.exit(1)

# --- Initialize vLLM Extractor ---
logger.info(f"Initializing VLLMHiddenStateExtractor...")
# Use the model that the server actually loads
model_name = "./configs/vllm_hidden_qwen4b"
target_layer = -1  # Last of 4 extracted layers (layer 35)

extractor = VLLMHiddenStateExtractor(
    server_url=VLLM_URL,
    model_name=model_name,
    model_id=MODEL_ID,  # For tokenizer
    target_layer=target_layer,
    max_workers=BATCH_SIZE,  # Concurrent requests
)
logger.info(f"VLLMHiddenStateExtractor initialized.")

HIDDEN_SIZE = extractor.hidden_size
logger.info(f"Hidden size: {HIDDEN_SIZE}")


def wandb_log_safe(payload):
    if WANDB_DISABLED or wandb.run is None:
        return
    try:
        wandb.log(payload)
    except Exception as wandb_exc:
        logger.warning(f"wandb.log failed: {wandb_exc}")


def wandb_finish_safe(exit_code=None):
    if wandb.run is None:
        return
    try:
        if exit_code is None:
            wandb.finish()
        else:
            wandb.finish(exit_code=exit_code)
    except Exception as wandb_exc:
        logger.warning(f"wandb.finish failed: {wandb_exc}")


WORKER_WANDB_ACTIVE = False
if not WANDB_DISABLED and not WANDB_PER_FILE:
    worker_run_name = (
        f"labelling_worker_"
        f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'nojid'))}_"
        f"{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{hostname}"
    )
    try:
        wandb.init(
            project=WANDB_PROJECT,
            name=worker_run_name,
            resume="allow",
            config={
                "batch_size": BATCH_SIZE,
                "flush_every_batches": FLUSH_EVERY_BATCHES,
                "wandb_log_every_batches": WANDB_LOG_EVERY_BATCHES,
                "max_job_retries": MAX_JOB_RETRIES,
            },
            reinit=True,
        )
        WORKER_WANDB_ACTIVE = True
        logger.info(f"Initialized worker-level wandb run: {worker_run_name}")
    except Exception as wandb_exc:
        logger.warning(f"Worker-level wandb init failed; continuing without wandb: {wandb_exc}")



def write_progress(progress_path, batch_idx):
    """Atomically writes the last completed batch index."""
    tmp_progress_path = f"{progress_path}.tmp"
    with open(tmp_progress_path, 'w') as f:
        json.dump({"last_completed_batch": batch_idx}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_progress_path, progress_path)

def read_progress(progress_path):
    """Reads the progress file. Returns -1 if not found."""
    if not os.path.exists(progress_path):
        return -1
    try:
        with open(progress_path, 'r') as f:
            data = json.load(f)
            return data.get("last_completed_batch", -1)
    except (json.JSONDecodeError, IOError):
        logger.warning(f"Could not read progress file {progress_path}. Starting from scratch.")
        return -1


def save_result_atomically(save_data, result_path, job_basename):
    """
    Save .npz to a unique temp file in RESULTS_DIR, then atomically rename.
    The explicit ".npz" suffix avoids numpy's implicit extension behavior.
    """
    for attempt in range(1, RESULT_SAVE_RETRIES + 1):
        tmp_result_path = os.path.join(
            RESULTS_DIR, f".{job_basename}.tmp.{pid}.{int(time.time() * 1e6)}.npz"
        )
        try:
            with open(tmp_result_path, "wb") as f:
                np.savez_compressed(f, **save_data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_result_path, result_path)
            return
        except Exception:
            with suppress(OSError):
                if os.path.exists(tmp_result_path):
                    os.remove(tmp_result_path)
            if attempt == RESULT_SAVE_RETRIES:
                raise
            sleep_s = RESULT_SAVE_RETRY_SEC * attempt
            logger.warning(
                "Final save attempt %d/%d failed for %s; retrying in %.1fs",
                attempt,
                RESULT_SAVE_RETRIES,
                job_basename,
                sleep_s,
            )
            time.sleep(sleep_s)

# --- Main Worker Loop ---
completed_jobs_worker = 0
redis_error_count = 0
redis_backoff_sec = REDIS_RETRY_INITIAL_SLEEP_SEC
while True:
    file_path = None
    job_basename = None
    result_path = None
    temp_hidden_path = None
    temp_text_path = None
    progress_path = None
    local_input_path = None
    data = None
    hidden_states_memmap = None
    text_outputs_memmap = None
    file_wandb_active = False
    file_start_time = None
    try:
        # 1. GET JOB
        try:
            file_path = r.rpop(QUEUE_NAME)
            redis_error_count = 0
            redis_backoff_sec = REDIS_RETRY_INITIAL_SLEEP_SEC
        except redis.exceptions.RedisError as redis_exc:
            redis_error_count += 1
            if (
                redis_error_count == 1
                or redis_error_count % REDIS_RETRY_LOG_EVERY == 0
            ):
                logger.warning(
                    "Redis unavailable while polling queue (%d/%s): %s. Retrying in %.1fs.",
                    redis_error_count,
                    "inf" if REDIS_MAX_CONSECUTIVE_ERRORS <= 0 else REDIS_MAX_CONSECUTIVE_ERRORS,
                    redis_exc,
                    redis_backoff_sec,
                )

            if (
                REDIS_MAX_CONSECUTIVE_ERRORS > 0
                and redis_error_count >= REDIS_MAX_CONSECUTIVE_ERRORS
            ):
                logger.error(
                    "Exceeded Redis retry budget (%d consecutive failures). Exiting worker.",
                    REDIS_MAX_CONSECUTIVE_ERRORS,
                )
                break

            time.sleep(redis_backoff_sec)
            redis_backoff_sec = min(
                REDIS_RETRY_MAX_SLEEP_SEC,
                max(REDIS_RETRY_INITIAL_SLEEP_SEC, redis_backoff_sec * 1.5),
            )
            continue

        if file_path is None:
            logger.info("No more jobs! Exiting.")
            break

        logger.info(f"Processing job: {file_path}")
        file_start_time = time.time()
        job_basename = os.path.basename(file_path)
        result_path = os.path.join(RESULTS_DIR, job_basename)
        if os.path.exists(result_path):
            logger.info(f"Result already exists, skipping: {result_path}")
            with suppress(Exception):
                r.hdel(RETRY_HASH_NAME, file_path)
            continue
        if not WANDB_DISABLED and WANDB_PER_FILE:
            try:
                wandb.init(
                    project=WANDB_PROJECT,
                    name=f"labelling_{job_basename}",
                    resume="allow",
                    reinit=True,
                )
                file_wandb_active = True
            except Exception as wandb_exc:
                logger.warning(f"Per-file wandb init failed for {job_basename}: {wandb_exc}")
                file_wandb_active = False

        # Define paths for mmap "save state" files
        temp_hidden_path = os.path.join(TEMP_NPY_DIR, f"{job_basename}_temp_hidden.npy")
        temp_text_path = os.path.join(TEMP_NPY_DIR, f"{job_basename}_temp_text.npy")
        progress_path = os.path.join(PROGRESS_DIR, f"{job_basename}_progress.json")
        local_input_path = os.path.join(TEMP_NPY_DIR, f"{job_basename}_input.npz")

        # Stage source file to local scratch to reduce NFS metadata churn.
        shutil.copyfile(file_path, local_input_path)
        
        data = np.load(local_input_path, allow_pickle=True)
        num_samples = len(data["obs"])
        total_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

        # 2. CHECK FOR SAVED PROGRESS
        last_completed_batch = read_progress(progress_path)
        start_batch = last_completed_batch + 1
        start_index = start_batch * BATCH_SIZE
        
        # Hidden states saved as (N, hidden_size) — last-token only
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch} (sample index {start_index})")
            hidden_states_memmap = np.memmap(
                temp_hidden_path, dtype=np.float16, mode='r+',
                shape=(num_samples, HIDDEN_SIZE)
            )
            text_outputs_memmap = np.memmap(
                temp_text_path, dtype=TEXT_DTYPE, mode='r+', shape=(num_samples,)
            )
        else:
            logger.info("Starting new job, creating temp files.")
            hidden_states_memmap = np.memmap(
                temp_hidden_path, dtype=np.float16, mode='w+',
                shape=(num_samples, HIDDEN_SIZE)
            )
            text_outputs_memmap = np.memmap(
                temp_text_path, dtype=TEXT_DTYPE, mode='w+', shape=(num_samples,)
            )
            write_progress(progress_path, -1)

        # 3. RUN INFERENCE (THE LONG PART)
        logger.info(f"Beginning inference from index {start_index}...")
        start_time = time.time()

        for i in range(start_index, num_samples, BATCH_SIZE):
            current_batch_idx = i // BATCH_SIZE
            current_batch_indices = range(i, min(i + BATCH_SIZE, num_samples))
            current_batch_size = len(current_batch_indices)

            # Collect filtered observations for this batch
            batch_observations = []
            for idx in current_batch_indices:
                # Use pre-saved text_obs if available, otherwise decode from obs
                if "text_obs" in data and data["text_obs"][idx]:
                    raw_text_obs = str(data["text_obs"][idx])
                else:
                    # Decode symbolic observation to text
                    raw_text_obs = obs_to_text(data["obs"][idx])

                # Filter to show only interesting tiles (remove background)
                filtered_text_obs = filter_text_obs(raw_text_obs)
                batch_observations.append(filtered_text_obs)

            if GENERATE_TEXT:
                # Mode 1: Generate text first, then take last-token hidden state
                # Slower but provides both text and hidden states
                batch_hidden_vectors, generated_texts, metrics = extractor.extract_hidden_states(
                    batch_observations,
                    batch_size=BATCH_SIZE
                )
                # extract_hidden_states returns (N, hidden_size) last-token hidden states
                batch_hidden_state = batch_hidden_vectors.astype(np.float16)
                batch_text_fixed = np.array(generated_texts, dtype=TEXT_DTYPE)

            else:
                # Mode 2: Direct hidden state extraction (no text generation)
                # ~34x faster, returns (N, hidden_size) last-token hidden states
                batch_hidden_vectors, metrics = extractor.extract_hidden_states_no_cot(
                    batch_observations
                )
                batch_hidden_state = batch_hidden_vectors.astype(np.float16)
                batch_text_fixed = np.array(["" for _ in range(current_batch_size)], dtype=TEXT_DTYPE)

            # 4. SAVE PROGRESS TO DISK
            hidden_states_memmap[current_batch_indices, :] = batch_hidden_state
            text_outputs_memmap[current_batch_indices] = batch_text_fixed

            is_flush_batch = (
                ((current_batch_idx + 1) % FLUSH_EVERY_BATCHES == 0)
                or (i + BATCH_SIZE >= num_samples)
            )
            if is_flush_batch:
                hidden_states_memmap.flush()
                text_outputs_memmap.flush()
                write_progress(progress_path, current_batch_idx)

            if (
                ((current_batch_idx + 1) % PROGRESS_LOG_EVERY_BATCHES == 0)
                or (i + BATCH_SIZE >= num_samples)
            ):
                logger.info(f"  ... completed batch {current_batch_idx + 1} / {total_batches}")
            if (
                ((current_batch_idx + 1) % WANDB_LOG_EVERY_BATCHES == 0)
                or (i + BATCH_SIZE >= num_samples)
            ):
                wandb_log_safe(
                    {
                        "progress_batches": current_batch_idx + 1,
                        "total_batches": total_batches,
                    }
                )

        end_time = time.time()
        logger.info(f"Finished inference in {end_time - start_time:.2f}s for {file_path}")

        # 5. FINAL PACKAGING AND CLEANUP
        hidden_states_memmap.flush()
        text_outputs_memmap.flush()
        del hidden_states_memmap
        del text_outputs_memmap
        hidden_states_memmap = None
        text_outputs_memmap = None
        
        logger.info("Loading temporary .npy files for final save...")
        hidden_states_numpy = np.memmap(
            temp_hidden_path, dtype=np.float16, mode='r',
            shape=(num_samples, HIDDEN_SIZE)
        )
        all_outputs_numpy = np.memmap(
            temp_text_path, dtype=TEXT_DTYPE, mode='r', shape=(num_samples,)
        ).astype(object)  # Convert back to object

        save_data = {
            "obs": data["obs"], "next_obs": data["next_obs"],
            "action": data["action"], "reward": data["reward"],
            "done": data["done"], "log_prob": data["log_prob"],
            "hidden_state": hidden_states_numpy,
            "text_generated": all_outputs_numpy
        }

        logger.info(f"Saving final augmented data to: {result_path}")
        save_result_atomically(save_data, result_path, job_basename)
        logger.info(f"Job {file_path} completed and saved.")
        with suppress(Exception):
            r.hdel(RETRY_HASH_NAME, file_path)
        with suppress(Exception):
            r.incr(COMPLETED_COUNTER_NAME)
        completed_jobs_worker += 1
        if file_start_time is not None:
            wandb_log_safe(
                {
                    "worker_completed_jobs": completed_jobs_worker,
                    "file_seconds": float(time.time() - file_start_time),
                }
            )

        # Clean up ALL temporary files on success.
        for local_path in [temp_hidden_path, temp_text_path, progress_path, local_input_path]:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
        if file_wandb_active:
            wandb_finish_safe()

    except Exception as e:
        # 6. HANDLE ERRORS
        log_with_traceback = file_path is not None
        logger.error(
            f"Failed to process {file_path}: {e}",
            exc_info=log_with_traceback,
        )
        wandb_log_safe({"worker_errors": 1})
        if file_path is not None:
            try:
                retry_count = r.hincrby(RETRY_HASH_NAME, file_path, 1)
                if retry_count <= MAX_JOB_RETRIES:
                    r.lpush(QUEUE_NAME, file_path)
                    logger.warning(
                        "Re-queued failed job (%d/%d): %s",
                        retry_count,
                        MAX_JOB_RETRIES,
                        file_path,
                    )
                else:
                    logger.error(
                        "Not re-queuing %s; exceeded retry budget (%d).",
                        file_path,
                        MAX_JOB_RETRIES,
                    )
            except Exception as requeue_exc:
                logger.error(f"Failed to re-queue {file_path}: {requeue_exc}")

        if data is not None:
            with suppress(Exception):
                data.close()
            data = None
        if hidden_states_memmap is not None:
            with suppress(Exception):
                del hidden_states_memmap
            hidden_states_memmap = None
        if text_outputs_memmap is not None:
            with suppress(Exception):
                del text_outputs_memmap
            text_outputs_memmap = None

        # Temp files are local scratch only; clean them to avoid filling node disk.
        for local_path in [temp_hidden_path, temp_text_path, progress_path, local_input_path]:
            try:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
            except OSError:
                pass
        if file_wandb_active:
            wandb_finish_safe(exit_code=1)
        if file_path is None:
            time.sleep(REDIS_RETRY_INITIAL_SLEEP_SEC)
    finally:
        if data is not None:
            with suppress(Exception):
                data.close()

if WORKER_WANDB_ACTIVE:
    wandb_finish_safe()
logger.info("Worker finished.")
