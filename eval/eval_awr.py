import os
import argparse
import torch
import numpy as np
import jax
import wandb
import cv2  # Required for drawing video overlay
import matplotlib.pyplot as plt

from offline_rl.awr import Config, ActorCriticConv
from envs.image_utils import obs_to_01_range, obs_to_255_range


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax.config.update("jax_platform_name", "cpu")


def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    model = ActorCriticConv(
        action_dim=Config.ACTION_DIM, layer_width=Config.LAYER_WIDTH
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def draw_dual_line_graph(frame, values, rtgs, v_min=-1.0, v_max=10.0):
    """
    Draw dual line graph with FIXED dimensions.
    Resizes frame to width 600.
    Adds a footer for the graph.
    """
    # 1. Resize frame for visualization
    target_w = 600
    h, w, c = frame.shape
    scale = target_w / w
    target_h = int(h * scale)

    # 2. Define Footer Size
    # Reference used 450 for text; we use 100 since we only have the graph.
    FOOTER_H = 100

    # Graph Configuration
    graph_h = 60

    # Text Configuration for Legend
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Create Canvas
    total_h = target_h + FOOTER_H
    canvas = np.zeros((total_h, target_w, c), dtype=frame.dtype)

    # Draw Game Frame
    # Ensure resizing doesn't change dtype (usually uint8)
    viz_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas[0:target_h, 0:target_w] = viz_frame

    # --- Draw Graph (at top of footer) ---
    graph_y_start = target_h + 20
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
    val_disp = values[-1] if values else 0.0
    rtg_disp = rtgs[-1] if rtgs else 0.0

    cv2.putText(
        canvas,
        f"Pred V: {val_disp:.2f}",
        (x_start, graph_y_start - 5),
        font,
        0.5,
        (0, 255, 0),
        1,
    )

    if rtgs:
        cv2.putText(
            canvas,
            f"True G: {rtg_disp:.2f}",
            (x_start + 150, graph_y_start - 5),
            font,
            0.5,
            (255, 0, 0),
            1,
        )

    return canvas


def run_eval(args):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"eval-{os.path.basename(args.checkpoint)}",
        config=vars(args),
    )

    print(f"Initializing {args.env_name}...")
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(args.env_name, auto_reset=False)
    env_params = env.default_params

    rng = jax.random.PRNGKey(args.seed)
    model = load_model(args.checkpoint, args.device)

    total_rewards = []

    print(f"Starting evaluation for {args.num_episodes} episodes...")

    for ep in range(args.num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        done = False
        ep_reward = 0
        step_count = 0

        # --- Buffers for Two-Pass Rendering ---
        raw_frames_buffer = []  # Store raw images here
        ep_values = []  # Store V(s)
        ep_rewards = []  # Store r

        while not done:
            # 1. Handle Observation Conversion
            obs_np = np.array(obs)

            # Convert to proper formats
            frame_raw = obs_to_255_range(obs_np)  # For video (uint8 0-255)
            obs_01 = obs_to_01_range(obs_np)  # For model (float 0-1)

            obs_tensor = torch.from_numpy(obs_01).float().to(args.device).unsqueeze(0)

            # 2. Model Forward
            with torch.no_grad():
                pi, v_tensor = model(obs_tensor)
                action_tensor = pi.sample()
                action = action_tensor.item()
                value_scalar = v_tensor.item()

            # 3. Buffer Data (Do NOT draw yet)
            ep_values.append(value_scalar)
            raw_frames_buffer.append(frame_raw)

            # 4. Step Env
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                step_key, env_state, action, env_params
            )

            ep_rewards.append(float(reward))
            ep_reward += float(reward)
            step_count += 1

            if done or step_count > 10000:
                break

        total_rewards.append(ep_reward)
        print(
            f"Episode {ep+1}: Reward = {ep_reward:.2f} (Steps: {step_count}, Mean V: {np.mean(ep_values):.3f})"
        )

        # --- POST-PROCESSING: Calculate Returns ---
        # Calculate Discounted Return to Go (G_t)
        gamma = 0.99
        returns_to_go = []
        G = 0
        for r in reversed(ep_rewards):
            G = r + gamma * G
            returns_to_go.insert(0, G)

        # --- POST-PROCESSING: Generate Video ---
        # Now we have both V(s) and G_t, we can draw the frames
        ep_frames_with_overlay = []

        # Determine loop length (handles rare off-by-one edge cases in buffers)
        loop_len = min(len(raw_frames_buffer), len(ep_values), len(returns_to_go))

        for i in range(loop_len):
            # Pass history up to current step i to create the "evolving" effect
            current_values = ep_values[: i + 1]
            current_rtgs = returns_to_go[: i + 1]

            frame_overlay = draw_dual_line_graph(
                frame=raw_frames_buffer[i],
                values=current_values,
                rtgs=current_rtgs,
                v_min=args.v_min,
                v_max=args.v_max,
            )
            ep_frames_with_overlay.append(frame_overlay)

        # --- LOGGING: WandB ---
        # 1. Plot Line Series (Overlay V vs Return)
        steps_axis = [i for i in range(loop_len)]
        wandb.log(
            {
                f"eval/episode_{ep}_value_curve": wandb.plot.line_series(
                    xs=[steps_axis, steps_axis],
                    ys=[ep_values[:loop_len], returns_to_go[:loop_len]],
                    keys=["Predicted Value", "Empirical Return"],
                    title=f"Ep {ep} Value vs Return",
                    xname="step",
                )
            }
        )

        # 2. Upload Video
        if len(ep_frames_with_overlay) > 0:
            print("Processing video...")
            video_array = np.array(ep_frames_with_overlay)  # (T, H, W, C)
            video_array = np.transpose(video_array, (0, 3, 1, 2))  # (T, C, H, W)
            wandb.log(
                {
                    "eval/trajectory_video": wandb.Video(
                        video_array, fps=15, format="mp4"
                    )
                }
            )
            print("Video uploaded to WandB.")

    avg_reward = np.mean(total_rewards)
    print(f"Evaluation Complete. Average Reward: {avg_reward:.2f}")

    wandb.log({"eval/avg_reward": avg_reward})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pth file"
    )
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--wandb_project", type=str, default="awr-rich-logs-v1")
    parser.add_argument("--wandb_entity", type=str, default="iris-sobolmark")
    # Visual config
    parser.add_argument(
        "--v_min", type=float, default=-1.0, help="Min value for video bar scaling"
    )
    parser.add_argument(
        "--v_max", type=float, default=10.0, help="Max value for video bar scaling"
    )

    run_eval(parser.parse_args())
