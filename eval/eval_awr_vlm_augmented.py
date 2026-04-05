"""
Evaluation script for AWR augmented model using VLM server
"""
import os
import argparse
import torch
import numpy as np
import jax
import wandb
import cv2
import requests
import time
from pathlib import Path

from offline_rl.awr_augmented import Config, ActorCriticConvAug
from envs.image_utils import obs_to_01_range, obs_to_255_range, get_obs_stats


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_platform_name", "cpu")


def wait_for_server(server_url, timeout=60):
    """Wait for VLM server to be ready"""
    print(f"Waiting for VLM server at {server_url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"  Server ready! Model: {data['model']}")
                return True
        except:
            pass
        time.sleep(2)
    raise TimeoutError(f"VLM server at {server_url} not ready after {timeout}s")


def get_hidden_state_from_server(server_url, obs_np, debug=False):
    """
    Get hidden state from VLM server

    Args:
        server_url: Base URL of VLM server (e.g., "http://babel-v5-16:5000")
        obs_np: Observation as numpy array (H, W, C) in any valid range
        debug: If True, print debug info

    Returns:
        numpy array of shape (2560,) - raw hidden state (not normalized)
    """
    # Ensure observation is in 0-1 range for server
    obs_01 = obs_to_01_range(obs_np)

    # Send observation to server
    obs_list = obs_01.tolist()
    response = requests.post(
        f"{server_url}/get_hidden_state", json={"obs": obs_list}, timeout=30
    )

    if response.status_code != 200:
        raise RuntimeError(f"Server error: {response.text}")

    data = response.json()
    hidden_state = np.array(data["hidden_state"], dtype=np.float32)
    generated_text = data.get("generated_text", "NO TEXT RETURNED SOMEHOW")

    if debug and "stats" in data:
        print(
            f"  [Server Stats] Mean: {data['stats']['mean']:.4f}, Std: {data['stats']['std']:.4f}"
        )
        print(
            f"                 Min: {data['stats']['min']:.4f}, Max: {data['stats']['max']:.4f}"
        )

    return hidden_state, generated_text


def load_model(checkpoint_path, device):
    """Load augmented policy model"""
    print(f"Loading checkpoint: {checkpoint_path}")
    model = ActorCriticConvAug(
        action_dim=Config.ACTION_DIM,
        layer_width=Config.LAYER_WIDTH,
        hidden_state_dim=Config.HIDDEN_STATE_DIM,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Policy loaded successfully!")
    return model


def load_normalization_stats(stats_path):
    """Load hidden state normalization statistics"""
    print(f"Loading normalization stats from: {stats_path}")
    data = np.load(stats_path)
    hidden_mean = data["mean"]
    hidden_std = data["std"]
    print(
        f"  Mean shape: {hidden_mean.shape}, range: [{hidden_mean.min():.3f}, {hidden_mean.max():.3f}]"
    )
    print(
        f"  Std shape: {hidden_std.shape}, range: [{hidden_std.min():.3f}, {hidden_std.max():.3f}]"
    )
    return hidden_mean, hidden_std


def wrap_text(text, font, font_scale, thickness, max_width):
    """Helper to wrap text into multiple lines"""
    if not text:
        return []
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        # Try adding word to current line
        test_line = ' '.join(current_line + [word])
        (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if w <= max_width:
            current_line.append(word)
        else:
            # Line full, push it and start new one
            lines.append(' '.join(current_line))
            current_line = [word]
            
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def draw_dual_line_graph(frame, values, rtgs, text=None, v_min=-1.0, v_max=10.0):
    """
    Draw dual line graph AND generated text with FIXED dimensions.
    Resizes frame to width 600.
    Adds a fixed 450px footer for graph + text to ensure consistent video size.
    """
    # 1. Resize frame for visualization (so text is readable)
    target_w = 600
    h, w, c = frame.shape
    scale = target_w / w
    target_h = int(h * scale)
    
    # 2. Define Fixed Footer Size (Graph + Text area)
    # This prevents the "inhomogeneous shape" error
    FOOTER_H = 450 
    
    # Graph Configuration
    graph_h = 60
    
    # Text Configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_spacing = 20
    text_padding = 10
    
    # Create Canvas of Fixed Size
    total_h = target_h + FOOTER_H
    canvas = np.zeros((total_h, target_w, c), dtype=frame.dtype)
    
    # Draw Game Frame
    canvas[0:target_h, 0:target_w] = viz_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # --- Draw Graph (at top of footer) ---
    graph_y_start = target_h + 10
    graph_w = target_w - 20
    x_start = 10
    
    # Background for graph
    cv2.rectangle(
        canvas,
        (x_start, graph_y_start),
        (x_start + graph_w, graph_y_start + graph_h - 10),
        (30, 30, 30),
        -1,
    )
    
    def value_to_y(val):
        clamped = max(v_min, min(v_max, val))
        ratio = (clamped - v_min) / (v_max - v_min)
        # Invert Y (0 is top)
        return int((graph_y_start + graph_h - 10) - ratio * (graph_h - 20))

    # Draw Value line (green)
    if len(values) > 1:
        for i in range(len(values) - 1):
            x1 = int(x_start + (i / len(values)) * graph_w)
            x2 = int(x_start + ((i + 1) / len(values)) * graph_w)
            y1 = value_to_y(values[i])
            y2 = value_to_y(values[i + 1])
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw RTG line (blue)
    if rtgs is not None and len(rtgs) > 1:
        for i in range(len(rtgs) - 1):
            x1 = int(x_start + (i / len(rtgs)) * graph_w)
            x2 = int(x_start + ((i + 1) / len(rtgs)) * graph_w)
            y1 = value_to_y(rtgs[i])
            y2 = value_to_y(rtgs[i + 1])
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
    # Legend
    cv2.putText(canvas, f"V: {values[-1]:.2f}", (x_start, graph_y_start - 2), 
                font, 0.4, (0, 255, 0), 1)
    if rtgs:
        cv2.putText(canvas, f"RTG: {rtgs[-1]:.2f}", (x_start + 80, graph_y_start - 2), 
                    font, 0.4, (255, 0, 0), 1)

    # --- Draw Text (below graph) ---
    if text:
        # Wrap text
        wrapped_lines = wrap_text(text, font, font_scale, thickness, target_w - 2 * text_padding)
        
        text_y_start = graph_y_start + graph_h + text_padding
        max_y = total_h - line_spacing  # Margin at bottom
        
        for i, line in enumerate(wrapped_lines):
            y_pos = text_y_start + (i * line_spacing)
            
            # Stop drawing if we run out of space in the fixed footer
            if y_pos > max_y:
                break
                
            cv2.putText(canvas, line, (text_padding, y_pos), 
                        font, font_scale, (200, 200, 200), thickness)

    return canvas

def run_eval(args):
    """Run evaluation"""
    # Initialize WandB
    run_name = f"eval-aug-{os.path.basename(args.checkpoint).replace('.pth', '')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
    )
    print("[DEBUG] WandB initialized")

    # Wait for VLM server
    wait_for_server(args.server_url)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Load normalization stats
    hidden_mean, hidden_std = load_normalization_stats(args.stats_path)
    hidden_mean_tensor = torch.from_numpy(hidden_mean).float().to(device)
    hidden_std_tensor = torch.from_numpy(hidden_std).float().to(device)

    # Initialize environment
    print(f"Initializing {args.env_name}...")
    
    import craftax.craftax.envs.craftax_pixels_env as pixels_env_module
    from craftax.craftax_env import make_craftax_env_from_name

    Achievement = pixels_env_module.Achievement
    def log_achievements_to_info_always(state, done):
        # ORIGINAL: achievements = state.achievements * done * 100.0
        # NEW: Returns achievements regardless of 'done' state
        achievements = state.achievements * 100.0

        info = {}
        for achievement in Achievement:
            name = f"Achievements/{achievement.name.lower()}"
            info[name] = achievements[achievement.value]
        return info
    pixels_env_module.log_achievements_to_info = log_achievements_to_info_always

    
    env = make_craftax_env_from_name(args.env_name, auto_reset=False)
    env_params = env.default_params

    rng = jax.random.PRNGKey(args.seed)

    # Run episodes
    all_returns = []
    all_lengths = []

    # Track hidden state statistics across all episodes
    all_hidden_raw_stats = []  # Per-episode stats of raw hidden states
    all_hidden_norm_stats = []  # Per-episode stats of normalized hidden states

    print(f"\n=== Episode 1/{args.num_episodes} ===")

    for ep in range(args.num_episodes):
        if ep > 0:
            print(f"\n=== Episode {ep+1}/{args.num_episodes} ===")

        # Reset environment
        print("[DEBUG] Resetting environment...")
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)
        print(f"[DEBUG] Environment reset. Obs shape: {obs.shape}")

        # Log observation stats
        obs_stats = get_obs_stats(np.array(obs))
        print(
            f"[DEBUG] Initial obs: min={obs_stats['min']:.3f}, max={obs_stats['max']:.3f}, mean={obs_stats['mean']:.3f}"
        )

        done = False
        ep_return = 0.0
        ep_length = 0
        ep_values = []
        ep_rewards = []
        ep_frames = []

        # Track hidden states for this episode
        ep_hidden_raw = []  # Raw hidden states from VLM
        ep_hidden_norm = []  # Normalized hidden states

        # Get first hidden state
        print("[DEBUG] Getting first hidden state from VLM server...")
        obs_np = np.array(obs)

        hidden_raw, current_text = get_hidden_state_from_server(
            args.server_url, obs_np, debug=(ep == 0)
        )
        print(f"[DEBUG] Received hidden state. Shape: {hidden_raw.shape}")
        print(
            f"[DEBUG] Raw hidden state: mean={hidden_raw.mean():.4f}, std={hidden_raw.std():.4f}, min={hidden_raw.min():.4f}, max={hidden_raw.max():.4f}"
        )

        while not done and ep_length < 10000:
            # Convert observation to proper formats
            obs_np = np.array(obs)
            obs_01 = obs_to_01_range(obs_np)  # For model input
            frame_raw = obs_to_255_range(obs_np)  # For video

            obs_tensor = torch.from_numpy(obs_01).float().to(device).unsqueeze(0)

            # Normalize hidden state
            hidden_tensor = torch.from_numpy(hidden_raw).float().to(device).unsqueeze(0)
            hidden_normalized = (hidden_tensor - hidden_mean_tensor) / hidden_std_tensor

            # Track hidden states
            ep_hidden_raw.append(hidden_raw.copy())
            ep_hidden_norm.append(hidden_normalized.cpu().numpy()[0].copy())

            # Log first step details
            if ep_length == 0:
                print(
                    f"[DEBUG] Normalized hidden state: mean={hidden_normalized.mean().item():.4f}, std={hidden_normalized.std().item():.4f}"
                )
                print(
                    f"[DEBUG]                          min={hidden_normalized.min().item():.4f}, max={hidden_normalized.max().item():.4f}"
                )

            # Get action and value
            with torch.no_grad():
                pi, v_tensor = model(obs_tensor, hidden_normalized)
                action_tensor = pi.sample()
                action = action_tensor.item()
                value = v_tensor.item()

            if ep_length == 0:
                print(f"[DEBUG] Got action from policy: {action}, value: {value:.4f}")
                print(f"[DEBUG] Action distribution entropy: {pi.entropy().item():.4f}")

            ep_values.append(value)

            # Step environment
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                step_key, env_state, action, env_params
            )

            step_log = {
                k: v for k, v in info.items() 
                if k.startswith("Achievements/")
            }
            step_log["step_reward"] = float(reward)
            step_log["step_value"] = value
            wandb.log(step_log)

            ep_return += float(reward)
            ep_rewards.append(float(reward))
            ep_length += 1

            if ep_length == 1:
                print(f"[DEBUG] Environment stepped. Reward: {reward}, Done: {done}")

            # Save frame with visualization
            if args.save_video:
                # Compute RTG for visualization
                rtgs = [sum(ep_rewards[i:]) for i in range(len(ep_rewards))]
                frame_with_viz = draw_dual_line_graph(
                    frame_raw.copy(),
                    ep_values,
                    rtgs,
                    text=current_text,
                    v_min=args.v_min,
                    v_max=args.v_max,
                )
                ep_frames.append(frame_with_viz)

            # Get next hidden state
            if not done:
                obs_np = np.array(obs)
                hidden_raw, current_text = get_hidden_state_from_server(args.server_url, obs_np)

            # Progress updates
            if ep_length % 100 == 0:
                print(f"  Step {ep_length}: return={ep_return:.2f}")

        all_returns.append(ep_return)
        all_lengths.append(ep_length)

        # Compute episode hidden state statistics
        ep_hidden_raw_arr = np.array(ep_hidden_raw)  # (T, 2560)
        ep_hidden_norm_arr = np.array(ep_hidden_norm)  # (T, 2560)

        raw_stats = {
            "mean": float(ep_hidden_raw_arr.mean()),
            "std": float(ep_hidden_raw_arr.std()),
            "min": float(ep_hidden_raw_arr.min()),
            "max": float(ep_hidden_raw_arr.max()),
        }

        norm_stats = {
            "mean": float(ep_hidden_norm_arr.mean()),
            "std": float(ep_hidden_norm_arr.std()),
            "min": float(ep_hidden_norm_arr.min()),
            "max": float(ep_hidden_norm_arr.max()),
        }

        all_hidden_raw_stats.append(raw_stats)
        all_hidden_norm_stats.append(norm_stats)

        print(f"\nEpisode {ep+1} finished: return={ep_return:.2f}, length={ep_length}")
        print(
            f"  Raw hidden states:  mean={raw_stats['mean']:.4f}, std={raw_stats['std']:.4f}, min={raw_stats['min']:.4f}, max={raw_stats['max']:.4f}"
        )
        print(
            f"  Norm hidden states: mean={norm_stats['mean']:.4f}, std={norm_stats['std']:.4f}, min={norm_stats['min']:.4f}, max={norm_stats['max']:.4f}"
        )
        print(
            f"  Training stats:     mean={hidden_mean.mean():.4f}, std={hidden_std.mean():.4f}"
        )

        # Log to WandB
        wandb.log(
            {
                f"eval/ep{ep+1}/hidden_raw_mean": raw_stats["mean"],
                f"eval/ep{ep+1}/hidden_raw_std": raw_stats["std"],
                f"eval/ep{ep+1}/hidden_raw_min": raw_stats["min"],
                f"eval/ep{ep+1}/hidden_raw_max": raw_stats["max"],
                f"eval/ep{ep+1}/hidden_norm_mean": norm_stats["mean"],
                f"eval/ep{ep+1}/hidden_norm_std": norm_stats["std"],
                f"eval/ep{ep+1}/hidden_norm_min": norm_stats["min"],
                f"eval/ep{ep+1}/hidden_norm_max": norm_stats["max"],
                f"eval/ep{ep+1}/return": ep_return,
                f"eval/ep{ep+1}/length": ep_length,
                f"eval/ep{ep+1}/mean_value": np.mean(ep_values),
            }
        )

        # Save video
        if args.save_video and len(ep_frames) > 0:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{ep+1}.mp4"

            # Write video using cv2
            h, w = ep_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
            for frame in ep_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"  Video saved: {video_path}")

            # Log to WandB
            video_array = np.array(ep_frames)  # (T, H, W, C)
            video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
            wandb.log(
                {
                    f"eval/episode_{ep+1}_video": wandb.Video(
                        video_array, fps=15, format="mp4"
                    )
                }
            )

    # Summary
    print(f"\n{'='*70}")
    print(f"=== Evaluation Results ===")
    print(f"{'='*70}")
    print(f"Mean Return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"Mean Length: {np.mean(all_lengths):.2f}")
    print(f"All Returns: {all_returns}")

    # Aggregate hidden state statistics across all episodes
    print(f"\n{'='*70}")
    print(f"=== Hidden State Statistics (Across All Episodes) ===")
    print(f"{'='*70}")

    avg_raw_stats = {
        "mean": np.mean([s["mean"] for s in all_hidden_raw_stats]),
        "std": np.mean([s["std"] for s in all_hidden_raw_stats]),
        "min": np.min([s["min"] for s in all_hidden_raw_stats]),
        "max": np.max([s["max"] for s in all_hidden_raw_stats]),
    }

    avg_norm_stats = {
        "mean": np.mean([s["mean"] for s in all_hidden_norm_stats]),
        "std": np.mean([s["std"] for s in all_hidden_norm_stats]),
        "min": np.min([s["min"] for s in all_hidden_norm_stats]),
        "max": np.max([s["max"] for s in all_hidden_norm_stats]),
    }

    print(f"\nRaw Hidden States (from VLM):")
    print(f"  Mean: {avg_raw_stats['mean']:.4f}")
    print(f"  Std:  {avg_raw_stats['std']:.4f}")
    print(f"  Min:  {avg_raw_stats['min']:.4f}")
    print(f"  Max:  {avg_raw_stats['max']:.4f}")

    print(f"\nNormalized Hidden States (fed to policy):")
    print(f"  Mean: {avg_norm_stats['mean']:.4f}  (expected: ~0.0)")
    print(f"  Std:  {avg_norm_stats['std']:.4f}  (expected: ~1.0)")
    print(f"  Min:  {avg_norm_stats['min']:.4f}")
    print(f"  Max:  {avg_norm_stats['max']:.4f}")

    print(f"\nTraining Normalization Parameters:")
    print(
        f"  Mean range: [{hidden_mean.min():.4f}, {hidden_mean.max():.4f}], avg: {hidden_mean.mean():.4f}"
    )
    print(
        f"  Std range:  [{hidden_std.min():.4f}, {hidden_std.max():.4f}], avg: {hidden_std.mean():.4f}"
    )

    # Check if normalized stats are reasonable
    if abs(avg_norm_stats["mean"]) > 0.5:
        print(
            f"\n⚠️  WARNING: Normalized mean is {avg_norm_stats['mean']:.4f}, expected ~0.0!"
        )
        print(f"   This suggests VLM outputs during eval differ from training data.")

    if abs(avg_norm_stats["std"] - 1.0) > 0.5:
        print(
            f"\n⚠️  WARNING: Normalized std is {avg_norm_stats['std']:.4f}, expected ~1.0!"
        )
        print(f"   This suggests VLM outputs during eval differ from training data.")

    print(f"{'='*70}\n")

    wandb.log(
        {
            "eval/mean_return": np.mean(all_returns),
            "eval/std_return": np.std(all_returns),
            "eval/mean_length": np.mean(all_lengths),
            "eval/hidden_raw_mean": avg_raw_stats["mean"],
            "eval/hidden_raw_std": avg_raw_stats["std"],
            "eval/hidden_norm_mean": avg_norm_stats["mean"],
            "eval/hidden_norm_std": avg_norm_stats["std"],
        }
    )

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="VLM server URL (e.g., http://babel-v5-16:5000)",
    )
    parser.add_argument(
        "--stats_path", type=str, required=True, help="Path to hidden_state_stats.npz"
    )
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_video", action="store_true", help="Save episode videos")
    parser.add_argument("--video_dir", type=str, default="./eval_videos")
    parser.add_argument("--wandb_project", type=str, default="craftax-offline-awr")
    parser.add_argument("--wandb_entity", type=str, default="iris-sobolmark")
    parser.add_argument("--v_min", type=float, default=-1.0)
    parser.add_argument("--v_max", type=float, default=10.0)

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
