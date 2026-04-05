import redis
import numpy as np
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoProcessor,
)
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration
import time
import logging
import socket
import wandb
import argparse

parser = argparse.ArgumentParser(description="Worker for Craftax")
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

# gemini logging stuff
LOGS_DIR = "/data/group_data/rl/geney/craftax_job_logs/"  # Make this directory
os.makedirs(LOGS_DIR, exist_ok=True)
pid = os.getpid()
hostname = socket.gethostname()
log_filename = os.path.join(LOGS_DIR, f"worker_{hostname}_{pid}.log")
logger = logging.getLogger(f"worker_{pid}")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_filename)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
# gemini logging stuff


QUEUE_NAME = "craftax_job_queue"

RESULTS_DIR = f"/data/group_data/rl/geney/craftax_labelled_results/{args.name}/"
logger.info(f"Results will be saved in: {RESULTS_DIR}")
os.makedirs(RESULTS_DIR, exist_ok=True)

r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
r.ping()

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
BATCH_SIZE = 32
TOKENS_GENERATED = 256

logger.info("Initializing qwen3")
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_type=torch.float16
# )

import flash_attn

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    quantization_config=None,  # or quantization_config
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
logger.info("Qwen3 initialized.")

gamedesc = """Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions using WASD and can interact using SPACE. Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest.

The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. Mana is used for casting spells or enchanting items and will naturally recover. Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. If the players health falls beneath 0 they will die and the game will restart.

To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). There are 9 levels in total.
"""

question = """
"Think about what the character should do next, keeping in mind the intrinsics displayed on the screen. Provide a detailed action plan for the next steps the character should take to ensure survival and progress in the game."
"""


def create_consolidated_prompt(obs):
    img = Image.fromarray((obs * 255).astype(np.uint8))
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": gamedesc + "\n" + question},
            ],
        }
    ]
    return msg


while True:
    file_path = r.rpop(QUEUE_NAME)
    if file_path is None:
        logger.info("No more jobs! Exiting.")
        break

    logger.info(f"Processing job: {file_path}")
    wandb.init(
        project="craftax_offline_qwen3vl4b_labelling", name="labelling" + file_path
    )
    try:
        data = np.load(file_path)
        num_samples = len(data["obs"])  # Should be 8192
        all_hidden_states = []
        all_outputs = []

        logger.info(
            f"Beginning inference on {num_samples} samples in batches of {BATCH_SIZE}..."
        )
        start_time = time.time()

        for i in range(0, num_samples, BATCH_SIZE):
            batch_prompts = []
            indices = range(i, min(i + BATCH_SIZE, num_samples))

            for idx in indices:
                prompt = create_consolidated_prompt(
                    data["obs"][idx],
                    # data['next_obs'][idx],
                    # data['action'][idx],
                    # data['reward'][idx],
                    # data['done'][idx]
                )
                batch_prompts.append(prompt)

            inputs = processor.apply_chat_template(
                batch_prompts,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=TOKENS_GENERATED,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            last_layer_states_list = [
                step_hidden_states[-1] for step_hidden_states in outputs.hidden_states
            ]
            generated_hidden_states = torch.cat(last_layer_states_list, dim=1)
            seq_len = generated_hidden_states.shape[1]
            indices = torch.arange(
                seq_len - 1, -1, -8, device=generated_hidden_states.device
            )
            last_layer_hidden_state = (
                generated_hidden_states[:, indices, :].cpu().numpy()
            )
            # last_layer_hidden_state = outputs.hidden_states[:, ::-8, -1] # should be (batch_size, TOKENS_GENERATED, hidden_size)

            all_hidden_states.append(last_layer_hidden_state)
            prompt_length = inputs["input_ids"].shape[1]
            generated_token_ids = outputs.sequences[:, prompt_length:]
            generated_text_list = processor.batch_decode(
                generated_token_ids, skip_special_tokens=True
            )
            np_text = np.array(generated_text_list, dtype=object)
            all_outputs.append(np_text)

            if (i // BATCH_SIZE) % 10 == 0:  # Log progress
                logger.info(
                    f"  ... processed batch {i // BATCH_SIZE} / {num_samples // BATCH_SIZE}"
                )

        end_time = time.time()
        logger.info(
            f"Finished inference in {end_time - start_time:.2f}s of {file_path}"
        )
        hidden_states_numpy = np.concatenate(all_hidden_states, axis=0)
        all_outputs_numpy = np.concatenate(all_outputs, axis=0)

        save_data = {  # idk if necessary to remake
            "obs": data["obs"],
            "next_obs": data["next_obs"],
            "action": data["action"],
            "reward": data["reward"],
            "done": data["done"],
            "log_prob": data["log_prob"],
            "hidden_state": hidden_states_numpy,
            "text_generated": all_outputs_numpy,
        }

        result_file_name = os.path.basename(file_path)
        result_path = os.path.join(RESULTS_DIR, result_file_name)

        # Save the new .npz file
        logger.info(f"  Saving augmented data to: {result_path}")
        np.savez_compressed(result_path, **save_data)
        logger.info(f"  Job {file_path} completed and saved.")

    except Exception as e:
        logger.info(f"Failed to process {file_path}: {e}", exc_info=True)
        # r.lpush(QUEUE_NAME, file_path) # gemini uncertain if this is needed

logger.info("Worker finished.")
