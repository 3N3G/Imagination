"""
VLM Server for Craftax Augmented Evaluation
Hosts Qwen3-VL model and provides API for generating hidden states
"""
import os
import argparse
from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import io
import base64
import json
from datetime import datetime
from image_utils import obs_to_pil_image, get_obs_stats

# VLM Configuration
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
TOKENS_GENERATED = 256
ONLY_GENERATION = False # Whether to take the last 32 tokens instead of all 80 (includes the prompt)

gamedesc = """Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions using WASD and can interact using SPACE. Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest.

The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. Mana is used for casting spells or enchanting items and will naturally recover. Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. If the players health falls beneath 0 they will die and the game will restart.

To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). There are 9 levels in total.
"""

question = """
"Think about what the character should do next, keeping in mind the intrinsics displayed on the screen. Provide a detailed action plan for the next steps the character should take to ensure survival and progress in the game."
"""

app = Flask(__name__)

# Global variables for model
vlm_model = None
processor = None

# Debug mode variables
DEBUG_MODE = False
DEBUG_DIR = "./vlm_debug"
request_counter = 0


def create_consolidated_prompt(img_pil):
    """Create VLM prompt from PIL Image"""
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text", "text": gamedesc + "\n" + question},
            ],
        }
    ]
    return msg


def load_vlm_model():
    """Load the Qwen3-VL model"""
    print(f"Loading VLM model: {MODEL_ID}...")
    import flash_attn

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        quantization_config=None,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("VLM model loaded successfully!")

    return model, proc


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ready", "model": MODEL_ID})


@app.route("/get_hidden_state", methods=["POST"])
def get_hidden_state():
    """
    Generate hidden state from observation image

    Expected JSON:
    {
        "obs": base64-encoded PNG image (or numpy array as list)
    }

    Returns JSON:
    {
        "hidden_state": list of floats (length 2560),
        "shape": [2560],
        "stats": {"mean": float, "std": float, "min": float, "max": float}  # if debug mode
    }
    """
    global request_counter
    try:
        request_counter += 1
        current_request_id = request_counter

        data = request.get_json()

        # Handle base64 image or numpy array
        if "obs" in data:
            # Assume obs is a list representing numpy array
            obs_np = np.array(data["obs"], dtype=np.float32)

            # DEBUG: Log observation statistics
            if DEBUG_MODE:
                obs_stats = get_obs_stats(obs_np)
                print(f"\n[DEBUG Request {current_request_id}] Observation received:")
                print(f"  Shape: {obs_stats['shape']}")
                print(f"  Dtype: {obs_stats['dtype']}")
                print(
                    f"  Min: {obs_stats['min']:.4f}, Max: {obs_stats['max']:.4f}, Mean: {obs_stats['mean']:.4f}"
                )

            # Use image helper to convert to PIL Image
            img_pil = obs_to_pil_image(obs_np)
        else:
            return jsonify({"error": "Missing obs"}), 400

        # DEBUG: Save first query
        if DEBUG_MODE and current_request_id <= 3:
            os.makedirs(DEBUG_DIR, exist_ok=True)

            # Save image
            img_path = os.path.join(
                DEBUG_DIR, f"request_{current_request_id:04d}_image.png"
            )
            img_pil.save(img_path)

            # Save observation array if available
            if obs_np is not None:
                obs_path = os.path.join(
                    DEBUG_DIR, f"request_{current_request_id:04d}_obs.npz"
                )
                np.savez(obs_path, obs=obs_np)

            # Save prompt
            prompt = create_consolidated_prompt(img_pil)
            prompt_path = os.path.join(
                DEBUG_DIR, f"request_{current_request_id:04d}_prompt.json"
            )
            with open(prompt_path, "w") as f:
                json.dump(
                    {"prompt": str(prompt), "gamedesc": gamedesc, "question": question},
                    f,
                    indent=2,
                )

            print(f"[DEBUG] Saved request {current_request_id} to {DEBUG_DIR}/")

        # Create prompt
        prompt = create_consolidated_prompt(img_pil)

        # Process through VLM
        inputs = processor.apply_chat_template(
            [prompt],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(vlm_model.device)

        with torch.no_grad():
            outputs = vlm_model.generate(
                **inputs,
                max_new_tokens=TOKENS_GENERATED,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs.sequences[0, input_len:]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)

        # Extract hidden states from last layer - ALL generated tokens, no subsampling
        # This matches online_rl_hidden.py and extract_hidden_states.py
        last_layer_states_list = [
            step_hidden_states[-1] for step_hidden_states in outputs.hidden_states
        ]
        generated_hidden_states = torch.cat(last_layer_states_list, dim=1)
        seq_len = generated_hidden_states.shape[1]

        # DIAGNOSTIC: Log token counts to debug distribution shift
        if current_request_id <= 3 or DEBUG_MODE:
            print(
                f"[DIAG] Request {current_request_id}: seq_len={seq_len}, "
                f"num_hidden_state_steps={len(outputs.hidden_states)}, "
                f"generated_hidden_states.shape={generated_hidden_states.shape}"
                f"generated_text={generated_text}"
            )

        # Truncate to TOKENS_GENERATED if longer (matching other scripts)
        if seq_len > TOKENS_GENERATED:
            generated_hidden_states = generated_hidden_states[:, :TOKENS_GENERATED, :]
            seq_len = TOKENS_GENERATED

        # DIAGNOSTIC: Log shape before pooling
        if current_request_id <= 3 or DEBUG_MODE:
            print(
                f"[DIAG] Before pooling: {generated_hidden_states.shape} "
                f"(using all {seq_len} tokens, no subsampling)"
            )
            raw_l2_before_pool = torch.norm(generated_hidden_states, p=2).item()
            print(f"[DIAG] Raw L2 norm before pooling: {raw_l2_before_pool:.2f}")

        # Mean pool across all tokens to (1, 2560)
        pooled_hidden = torch.mean(generated_hidden_states, dim=1)  # (1, 2560)

        # DIAGNOSTIC: Log pooled L2 norm (training data ~279, eval currently ~230)
        if current_request_id <= 3 or DEBUG_MODE:
            pooled_l2 = torch.norm(pooled_hidden, p=2).item()
            print(
                f"[DIAG] After mean pooling: shape={pooled_hidden.shape}, L2 norm={pooled_l2:.2f}"
            )

        # Convert to list for JSON
        hidden_np = pooled_hidden.cpu().numpy()[0]  # (2560,)
        hidden_list = hidden_np.tolist()

        # DEBUG: Log hidden state statistics
        if DEBUG_MODE:
            hidden_stats = {
                "mean": float(hidden_np.mean()),
                "std": float(hidden_np.std()),
                "min": float(hidden_np.min()),
                "max": float(hidden_np.max()),
            }
            print(f"[DEBUG Request {current_request_id}] Hidden state stats:")
            print(f"  Mean: {hidden_stats['mean']:.4f}, Std: {hidden_stats['std']:.4f}")
            print(f"  Min: {hidden_stats['min']:.4f}, Max: {hidden_stats['max']:.4f}")

            # Save hidden state for first few requests
            if current_request_id <= 3:
                hidden_path = os.path.join(
                    DEBUG_DIR, f"request_{current_request_id:04d}_hidden.npz"
                )
                np.savez(hidden_path, hidden_state=hidden_np, stats=hidden_stats)

        response_data = {"hidden_state": hidden_list, "shape": list(hidden_np.shape), "generated_text": generated_text}

        if DEBUG_MODE:
            response_data["stats"] = hidden_stats
            response_data["request_id"] = current_request_id

        return jsonify(response_data)

    except Exception as e:
        import traceback

        print(f"[ERROR Request {request_counter}] Exception occurred:")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (saves first 3 queries)"
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="./vlm_debug",
        help="Directory to save debug files",
    )
    args = parser.parse_args()

    # Set debug mode globally
    global vlm_model, processor, DEBUG_MODE, DEBUG_DIR
    DEBUG_MODE = args.debug
    DEBUG_DIR = args.debug_dir

    if DEBUG_MODE:
        print(f"\n{'='*60}")
        print(f"DEBUG MODE ENABLED")
        print(f"Debug files will be saved to: {DEBUG_DIR}")
        print(f"{'='*60}\n")

    # Load model globally
    vlm_model, processor = load_vlm_model()

    print(f"\nStarting VLM server on {args.host}:{args.port}")
    print(f"Model: {MODEL_ID}")
    print(f"Debug mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
    print("Endpoints:")
    print(f"  - GET  {args.host}:{args.port}/health")
    print(f"  - POST {args.host}:{args.port}/get_hidden_state")
    print("\nServer ready!\n")

    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
