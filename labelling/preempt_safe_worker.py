import redis
import numpy as np
import os
import time
import logging
import socket
import wandb
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import flash_attn
import sys
import json  # <-- ADDED FOR PROGRESS FILE

# --- Constants ---
QUEUE_NAME = "craftax_job_queue"
RESULTS_DIR = "/data/group_data/rl/craftax_resuming_labelled_results/"
LOGS_DIR = "/data/group_data/rl/craftax_resuming_job_logs/"
TEMP_NPY_DIR = os.path.join(RESULTS_DIR, "temp_npy") # For mmap files
PROGRESS_DIR = os.path.join(RESULTS_DIR, "progress") # For progress files

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BATCH_SIZE = 32
TOKENS_GENERATED = 256

# --- mmap Constants ---
# We still need a fixed text size, but we DON'T use it for resumption
MAX_TEXT_LEN = 2048 
TEXT_DTYPE = f'<U{MAX_TEXT_LEN}'

# --- Standard Logging Setup ---
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_NPY_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True) 
pid = os.getpid()
hostname = socket.gethostname()
log_filename = os.path.join(LOGS_DIR, f"worker_{hostname}_{pid}.log")
logger = logging.getLogger(f"worker_{pid}")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_filename)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler(sys.stdout))


# --- Redis Connection ---
r = redis.Redis(host='login2', port=6379, decode_responses=True)
r.ping()
logger.info("Successfully connected to Redis.")

# --- Model Initialization ---
logger.info("Initializing qwen3")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float16, quantization_config=None,
    attn_implementation="flash_attention_2", device_map="auto", trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
logger.info("Qwen3 initialized.")

HIDDEN_SIZE = model.config.hidden_size
DOWNSAMPLED_SEQ_LEN = TOKENS_GENERATED // 8
logger.info(f"Model hidden size: {HIDDEN_SIZE}, Downsampled seq len: {DOWNSAMPLED_SEQ_LEN}")

# --- Game Description and Prompt ---
gamedesc = """... (your game description) ..."""
question = """... (your question) ..."""

def create_consolidated_prompt(obs):
    img = Image.fromarray((obs * 255).astype(np.uint8))
    msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": gamedesc + "\n" + question}]}]
    return msg

def write_progress(progress_path, batch_idx):
    """Atomically writes the last completed batch index."""
    with open(progress_path, 'w') as f:
        json.dump({"last_completed_batch": batch_idx}, f)

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

# --- Main Worker Loop ---
while True:
    file_path = None
    try:
        # 1. GET JOB
        file_path = r.rpop(QUEUE_NAME)
        if file_path is None:
            logger.info("No more jobs! Exiting.")
            break

        logger.info(f"Processing job: {file_path}")
        job_basename = os.path.basename(file_path)
        wandb.init(project="craftax_offline_qwen3vl4b_labelling", name=f"labelling_{job_basename}", resume="allow")

        # Define paths for mmap "save state" files
        temp_hidden_path = os.path.join(TEMP_NPY_DIR, f"{job_basename}_temp_hidden.npy")
        temp_text_path = os.path.join(TEMP_NPY_DIR, f"{job_basename}_temp_text.npy")
        progress_path = os.path.join(PROGRESS_DIR, f"{job_basename}_progress.json") 
        
        data = np.load(file_path)
        num_samples = len(data["obs"])

        # 2. CHECK FOR SAVED PROGRESS
        last_completed_batch = read_progress(progress_path)
        start_batch = last_completed_batch + 1
        start_index = start_batch * BATCH_SIZE
        
        if start_batch > 0:
            logger.info(f"Resuming from batch {start_batch} (sample index {start_index})")
            # Open existing files in read/write ('r+') mode
            hidden_states_memmap = np.memmap(
                temp_hidden_path, dtype=np.float16, mode='r+', 
                shape=(num_samples, DOWNSAMPLED_SEQ_LEN, HIDDEN_SIZE)
            )
            text_outputs_memmap = np.memmap(
                temp_text_path, dtype=TEXT_DTYPE, mode='r+', shape=(num_samples,)
            )
        else:
            logger.info("Starting new job, creating temp files.")
            # Create new files in write ('w+') mode
            hidden_states_memmap = np.memmap(
                temp_hidden_path, dtype=np.float16, mode='w+', 
                shape=(num_samples, DOWNSAMPLED_SEQ_LEN, HIDDEN_SIZE)
            )
            text_outputs_memmap = np.memmap(
                temp_text_path, dtype=TEXT_DTYPE, mode='w+', shape=(num_samples,)
            )
            # Write initial progress file
            write_progress(progress_path, -1)

        # 3. RUN INFERENCE (THE LONG PART)
        logger.info(f"Beginning inference from index {start_index}...")
        start_time = time.time()

        for i in range(start_index, num_samples, BATCH_SIZE):
            current_batch_idx = i // BATCH_SIZE
            batch_prompts = []
            current_batch_indices = range(i, min(i + BATCH_SIZE, num_samples))
            current_batch_size = len(current_batch_indices)

            for idx in current_batch_indices:
                prompt = create_consolidated_prompt(data["obs"][idx])
                batch_prompts.append(prompt)

            inputs = processor.apply_chat_template(
                batch_prompts, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=TOKENS_GENERATED,
                    output_hidden_states=True, return_dict_in_generate=True,
                )
            
            last_layer_states_list = [s[-1] for s in outputs.hidden_states]
            generated_hidden_states = torch.cat(last_layer_states_list, dim=1)
            
            seq_len = generated_hidden_states.shape[1]
            if seq_len > TOKENS_GENERATED:
                 generated_hidden_states = generated_hidden_states[:, :TOKENS_GENERATED, :]
            elif seq_len < TOKENS_GENERATED:
                padding = torch.zeros(
                    (current_batch_size, TOKENS_GENERATED - seq_len, HIDDEN_SIZE),
                    device=generated_hidden_states.device, dtype=generated_hidden_states.dtype
                )
                generated_hidden_states = torch.cat([generated_hidden_states, padding], dim=1)
            
            indices = torch.arange(TOKENS_GENERATED - 1, -1, -8, device=model.device)
            batch_hidden_state = generated_hidden_states[:, indices[::-1], :].cpu().to(torch.float16).numpy()

            # --- Text Output Processing (same as before) ---
            prompt_length = inputs['input_ids'].shape[1]
            generated_token_ids = outputs.sequences[:, prompt_length:]
            generated_text_list = processor.batch_decode(generated_token_ids, skip_special_tokens=True)
            batch_text_fixed = np.array(generated_text_list, dtype=TEXT_DTYPE)

            # 4. SAVE PROGRESS TO DISK
            hidden_states_memmap[current_batch_indices, :, :] = batch_hidden_state
            text_outputs_memmap[current_batch_indices] = batch_text_fixed

            # Flush mmap files
            hidden_states_memmap.flush()
            text_outputs_memmap.flush()
            
            # Update the progress file
            write_progress(progress_path, current_batch_idx)

            logger.info(f"  ... completed batch {current_batch_idx} / {num_samples // BATCH_SIZE}")
            wandb.log({"progress_batches": current_batch_idx, "total_batches": num_samples // BATCH_SIZE})

        end_time = time.time()
        logger.info(f"Finished inference in {end_time - start_time:.2f}s for {file_path}")

        # 5. FINAL PACKAGING AND CLEANUP
        del hidden_states_memmap
        del text_outputs_memmap
        
        logger.info("Loading temporary .npy files for final save...")
        hidden_states_numpy = np.load(temp_hidden_path)
        all_outputs_numpy = np.load(temp_text_path).astype(object) # Convert back to object

        save_data = {
            "obs": data["obs"], "next_obs": data["next_obs"],
            "action": data["action"], "reward": data["reward"],
            "done": data["done"], "log_prob": data["log_prob"],
            "hidden_state": hidden_states_numpy,
            "text_generated": all_outputs_numpy
        }

        result_path = os.path.join(RESULTS_DIR, job_basename)

        logger.info(f"Saving final augmented data to: {result_path}")
        np.savez_compressed(result_path, **save_data)
        logger.info(f"Job {file_path} completed and saved.")

        # Clean up ALL temporary files on success
        os.remove(temp_hidden_path)
        os.remove(temp_text_path)
        os.remove(progress_path) # <-- CLEAN UP PROGRESS FILE
        
        wandb.finish()

    except Exception as e:
        # 6. HANDLE ERRORS
        # We DO NOT re-queue. The Janitor will handle it.
        # We DO NOT delete the temp files. The next worker needs them.
        logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
        if wandb.run:
            wandb.finish(exit_code=1) 

logger.info("Worker finished.")
