import os
import glob
import argparse
import numpy as np
import cv2
import concurrent.futures
import wandb
import time
import gc

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from models.actor_critic_aug import orthogonal_init

# Only needed during eval
import jax

# ==============================================================================
# 1. Configuration
# ==============================================================================
class Config:
    # Data
    DATA_DIR = "/data/group_data/rl/geney/craftax_labelled_results_with_returns"
    DATA_GLOB = "trajectories_batch_*.npz"

    # Model
    ACTION_DIM = 43
    LAYER_WIDTH = 512

    # AWR Hyperparameters
    GAMMA = 0.99
    AWR_BETA = 10.0
    AWR_MAX_WEIGHT = 20.0
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    TOTAL_STEPS = 100_000
    BATCH_SIZE = 256
    LOG_FREQ = 100
    SAVE_FREQ = 25000
    EVAL_FREQ = 5000
    SAVE_DIR = "/data/group_data/rl/geney/checkpoints/awr_noaug/"
    SEED = 42

    # Eval
    EVAL_EPISODES = 3

    # Wandb
    WANDB_PROJECT = "craftax-offline-awr"
    WANDB_ENTITY = "iris-sobolmark"


# ==============================================================================
# 2. Model Architecture
# ==============================================================================
class ActorCriticConv(nn.Module):
    def __init__(self, action_dim, layer_width):
        super().__init__()
        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        self.flatten_dim = 512

        # Actor Head
        self.actor_fc1 = nn.Linear(self.flatten_dim, layer_width)
        self.actor_fc2 = nn.Linear(layer_width, action_dim)
        self.actor_fc3 = nn.Linear(action_dim, action_dim)

        # Critic Head
        self.critic_fc1 = nn.Linear(self.flatten_dim, layer_width)
        self.critic_fc2 = nn.Linear(layer_width, 1)

        self.apply_init()

    def apply_init(self):
        orthogonal_init(self.conv1, gain=nn.init.calculate_gain("relu"))
        orthogonal_init(self.conv2, gain=nn.init.calculate_gain("relu"))
        orthogonal_init(self.conv3, gain=nn.init.calculate_gain("relu"))

        orthogonal_init(self.actor_fc1, gain=2.0)
        orthogonal_init(self.actor_fc2, gain=0.01)
        orthogonal_init(self.actor_fc3, gain=0.01)

        orthogonal_init(self.critic_fc1, gain=2.0)
        orthogonal_init(self.critic_fc2, gain=1.0)

    def forward(self, obs):
        x = obs.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        embedding = x.reshape(x.size(0), -1)

        # Actor
        actor_x = F.relu(self.actor_fc1(embedding))
        actor_x = F.relu(self.actor_fc2(actor_x))
        actor_logits = self.actor_fc3(actor_x)
        pi = Categorical(logits=actor_logits)

        # Critic
        critic_x = F.relu(self.critic_fc1(embedding))
        value = self.critic_fc2(critic_x)

        return pi, value.squeeze(-1)


# ==============================================================================
# 3. Dataset Loader
# ==============================================================================
class OfflineDataset:
    def __init__(self, data_dir, file_pattern):
        search_path = os.path.join(data_dir, file_pattern)
        files = glob.glob(search_path)
        if not files:
            raise ValueError(f"No files found at {search_path}")

        files = sorted(files)
        print(f"Found {len(files)} files. Calculating total size...")

        total_samples = 0
        file_info = []

        for f in files:
            try:
                with np.load(f, mmap_mode="r") as d:
                    n = d["reward"].shape[0]
                    total_samples += n
                    file_info.append((f, n))
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")

        print(f"Allocating buffers for {total_samples} samples...")
        self.obs = np.zeros((total_samples, 130, 110, 3), dtype=np.uint8)
        self.next_obs = np.zeros((total_samples, 130, 110, 3), dtype=np.uint8)
        self.action = np.zeros((total_samples,), dtype=np.int32)
        self.reward = np.zeros((total_samples,), dtype=np.float32)
        self.done = np.zeros((total_samples,), dtype=np.float32)
        self.return_to_go = np.zeros((total_samples,), dtype=np.float32)

        def load_single_file(args):
            fpath, expected_n = args
            try:
                with np.load(fpath) as data:
                    raw_obs = data["obs"]
                    raw_next = data["next_obs"]

                    if raw_obs.dtype == np.float32 and raw_obs.max() <= 1.1:
                        np.multiply(raw_obs, 255, out=raw_obs)
                        np.multiply(raw_next, 255, out=raw_next)

                    obs_uint8 = raw_obs.astype(np.uint8)
                    next_uint8 = raw_next.astype(np.uint8)

                    return {
                        "obs": obs_uint8,
                        "next_obs": next_uint8,
                        "action": data["action"],
                        "reward": data["reward"].astype(np.float32),
                        "done": data["done"],
                        "return_to_go": data["return_to_go"].astype(np.float32),
                        "count": len(raw_obs),
                    }
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                return None

        print("Starting parallel load (Max Workers: 8)...")
        loaded_count = 0
        idx = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {
                executor.submit(load_single_file, info): info for info in file_info
            }

            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                if result is None:
                    continue

                n = result["count"]
                if idx + n > total_samples:
                    break

                self.obs[idx : idx + n] = result["obs"]
                self.next_obs[idx : idx + n] = result["next_obs"]
                self.action[idx : idx + n] = result["action"]
                self.reward[idx : idx + n] = result["reward"]
                self.done[idx : idx + n] = result["done"]
                self.return_to_go[idx : idx + n] = result["return_to_go"]

                idx += n
                loaded_count += 1
                if loaded_count % 10 == 0:
                    print(f"Loaded {loaded_count}/{len(file_info)} files...")

        self.size = idx
        if self.size < total_samples:
            self.obs = self.obs[:idx]
            self.next_obs = self.next_obs[:idx]
            self.action = self.action[:idx]
            self.reward = self.reward[:idx]
            self.done = self.done[:idx]
            self.return_to_go = self.return_to_go[:idx]

        print(f"Dataset loaded. Total samples: {self.size}")

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        obs_t = (
            torch.tensor(self.obs[idx], dtype=torch.float32, device=Config.DEVICE)
            / 255.0
        )
        next_obs_t = (
            torch.tensor(self.next_obs[idx], dtype=torch.float32, device=Config.DEVICE)
            / 255.0
        )
        action_t = torch.tensor(
            self.action[idx], dtype=torch.long, device=Config.DEVICE
        )
        reward_t = torch.tensor(
            self.reward[idx], dtype=torch.float32, device=Config.DEVICE
        )
        done_t = torch.tensor(self.done[idx], dtype=torch.float32, device=Config.DEVICE)
        rtg_t = torch.tensor(
            self.return_to_go[idx], dtype=torch.float32, device=Config.DEVICE
        )

        return {
            "obs": obs_t,
            "next_obs": next_obs_t,
            "action": action_t,
            "reward": reward_t,
            "done": done_t,
            "return_to_go": rtg_t,
        }


# ==============================================================================
# 4. Training Logic
# ==============================================================================
def train_step(model, optimizer, batch):
    pi, current_v = model(batch["obs"])

    td_target = batch["return_to_go"]
    advantage = td_target - current_v

    critic_loss = 0.5 * torch.mean(advantage.pow(2))

    log_probs = pi.log_prob(batch["action"])
    adv_detached = advantage.detach()
    weights = torch.exp(adv_detached / Config.AWR_BETA)
    weights_clipped = torch.clamp(weights, max=Config.AWR_MAX_WEIGHT)
    actor_loss = -torch.mean(log_probs * weights_clipped)

    total_loss = critic_loss + actor_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        clip_frac = (weights >= Config.AWR_MAX_WEIGHT).float().mean().item()
        var_diff = torch.var(advantage)
        var_return = torch.var(batch["return_to_go"])
        explained_var = 1.0 - (var_diff / (var_return + 1e-8))

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": pi.entropy().mean().item(),
        "mean_weight": weights_clipped.mean().item(),
        "weight_clip_frac": clip_frac,
        "mean_value": current_v.detach().mean().item(),
        "mean_return": batch["return_to_go"].mean().item(),  # For comparison
        "explained_variance": explained_var.item(),
    }


# ==============================================================================
# 5. Evaluation Logic
# ==============================================================================


def draw_dual_line_graph(frame, val_hist, ret_hist, y_min, y_max, total_steps):
    """
    Draws a graph showing the return to go and the predicted value
    """
    frame = np.ascontiguousarray(frame)
    h, w, c = frame.shape

    footer_h = 60
    new_h = h + footer_h
    new_frame = np.zeros((new_h, w, c), dtype=frame.dtype)
    new_frame[0:h, 0:w] = frame

    graph_y_start = h + 5
    graph_y_end = h + footer_h - 5
    graph_h = graph_y_end - graph_y_start

    # Avoid divide by zero
    if y_max <= y_min:
        y_max = y_min + 1.0
    if total_steps < 1:
        total_steps = 1

    # Helper to project value to Y pixel
    def get_y(val):
        val = max(y_min, min(y_max, val))
        ratio = (val - y_min) / (y_max - y_min)
        return int(graph_y_end - (ratio * graph_h))

    # Helper to project step index to X pixel
    def get_x(step_idx):
        ratio = step_idx / total_steps
        return int(ratio * w)

    # Draw Lines (From step 0 up to current step)
    # We iterate through the history we have SO FAR (val_hist)
    # But we calculate X positions based on total_steps (GLOBAL scaling)
    if len(val_hist) > 1:
        for i in range(1, len(val_hist)):
            x_prev = get_x(i - 1)
            x_curr = get_x(i)

            # Draw Critic (Green)
            cv2.line(
                new_frame,
                (x_prev, get_y(val_hist[i - 1])),
                (x_curr, get_y(val_hist[i])),
                (0, 255, 0),
                1,
            )

            # Draw Return (Cyan)
            cv2.line(
                new_frame,
                (x_prev, get_y(ret_hist[i - 1])),
                (x_curr, get_y(ret_hist[i])),
                (255, 255, 0),
                1,
            )

    # Draw Legend / Current Vals
    curr_v = val_hist[-1] if len(val_hist) > 0 else 0
    curr_r = ret_hist[-1] if len(ret_hist) > 0 else 0

    # Text Legend
    cv2.putText(
        new_frame,
        f"Pred: {curr_v:.1f}",
        (10, h + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        new_frame,
        f"True: {curr_r:.1f}",
        (10, h + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 0),
        1,
    )

    # Optional: Draw a faint grey line at the "End" (Right side)
    cv2.line(new_frame, (w - 1, graph_y_start), (w - 1, graph_y_end), (50, 50, 50), 1)

    return new_frame


def run_eval(model, step):
    print("\n--- Starting Evaluation ---")

    # Clear JAX compilation cache and Python garbage to free memory
    # This prevents "Cannot allocate memory" errors during evaluation
    try:
        jax.clear_caches()
    except Exception:
        pass  # If version too old
    gc.collect()

    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(42 + step)

    model.eval()

    video_frames_final = []
    total_rewards = []

    for ep in range(Config.EVAL_EPISODES):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        done = False

        # Buffers for Pass 1 (Record Data)
        ep_raw_frames = []
        ep_values = []
        ep_rewards = []

        capture_video = ep == 0  # Only video for first ep

        step_count = 0
        while not done and step_count < 2000:
            obs_np = np.array(obs)

            # Store Frame
            if capture_video:
                if obs_np.max() <= 1.1:
                    f = (obs_np * 255).astype(np.uint8)
                else:
                    f = obs_np.astype(np.uint8)
                ep_raw_frames.append(f)

            # Get Value
            obs_tensor = torch.from_numpy(obs_np).float().to(Config.DEVICE)
            if obs_tensor.max() > 1.1:
                obs_tensor /= 255.0
            obs_tensor = obs_tensor.unsqueeze(0)

            with torch.no_grad():
                pi, v_tensor = model(obs_tensor)
                action = pi.sample().item()
                value = v_tensor.item()

            ep_values.append(value)

            # Step
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                step_key, env_state, action, env_params
            )
            ep_rewards.append(float(reward))

            step_count += 1

        # --- End of Episode: Pass 2 (Compute & Draw) ---

        # 1. Calculate Cumulative Score & True RTG
        cum_scores = []
        r_score = 0.0
        for r in ep_rewards:
            r_score += r
            cum_scores.append(r_score)

        true_rtgs = np.zeros_like(cum_scores)
        run_rtg = 0.0
        for i in reversed(range(len(cum_scores))):
            run_rtg = cum_scores[i] + (Config.GAMMA * run_rtg)
            true_rtgs[i] = run_rtg

        total_rewards.append(sum(ep_rewards))

        if capture_video:
            # Determine Ranges
            all_vals = ep_values + true_rtgs.tolist()
            y_min = min(all_vals)
            y_max = max(all_vals)
            total_steps = len(ep_raw_frames)  # This locks the X-Axis

            print(
                f"Generating video. Steps: {total_steps}, Y-Range: [{y_min:.1f}, {y_max:.1f}]"
            )

            for i in range(len(ep_raw_frames)):
                # We pass 'total_steps' so the graph knows the final width
                vf = draw_dual_line_graph(
                    ep_raw_frames[i],
                    ep_values[: i + 1],
                    true_rtgs[: i + 1],
                    y_min,
                    y_max,
                    total_steps,
                )  # <--- NEW ARGUMENT
                video_frames_final.append(vf)

    avg_reward = np.mean(total_rewards)
    print(f"Eval Done. Avg Reward (Raw): {avg_reward:.2f}")

    log_dict = {"eval/avg_reward": avg_reward}

    if len(video_frames_final) > 0:
        video_array = np.array(video_frames_final)
        video_array = np.transpose(video_array, (0, 3, 1, 2))
        log_dict["eval/video"] = wandb.Video(video_array, fps=15, format="mp4")

    wandb.log(log_dict, step=step)
    model.train()

    # Clean up memory after evaluation
    gc.collect()


# ==============================================================================
# 6. Main
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advantage Weighted Regression for Craftax"
    )
    parser.add_argument(
        "--data_dir", type=str, default=Config.DATA_DIR, help="Path to training data"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=Config.SAVE_DIR,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=Config.TOTAL_STEPS,
        help="Total training steps",
    )
    parser.add_argument(
        "--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=Config.LR, help="Learning rate")
    parser.add_argument(
        "--awr_beta",
        type=float,
        default=Config.AWR_BETA,
        help="AWR temperature parameter",
    )
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed")
    parser.add_argument(
        "--wandb_name", type=str, default=Config.WANDB_NAME, help="WandB run name"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=Config.EVAL_FREQ, help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=Config.SAVE_FREQ,
        help="Checkpoint save frequency",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # Update config with CLI args
    Config.DATA_DIR = args.data_dir
    Config.SAVE_DIR = args.save_dir
    Config.TOTAL_STEPS = args.total_steps
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.AWR_BETA = args.awr_beta
    Config.SEED = args.seed
    Config.WANDB_NAME = args.wandb_name
    Config.EVAL_FREQ = args.eval_freq
    Config.SAVE_FREQ = args.save_freq

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    if not args.no_wandb:
        wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=Config.WANDB_NAME,
            config={k: v for k, v in vars(Config).items() if not k.startswith("_")},
            settings=wandb.Settings(init_timeout=300),
        )
        print(f"WandB initialized: {Config.WANDB_PROJECT}/{Config.WANDB_NAME}")

    print(f"Starting AWR training with {Config.TOTAL_STEPS} steps")
    print(f"Data: {Config.DATA_DIR}")
    print(f"Checkpoints: {Config.SAVE_DIR}")
    print(f"AWR Beta: {Config.AWR_BETA}")

    model = ActorCriticConv(
        action_dim=Config.ACTION_DIM, layer_width=Config.LAYER_WIDTH
    ).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    dataset = OfflineDataset(Config.DATA_DIR, Config.DATA_GLOB)

    model.train()

    for step in range(1, Config.TOTAL_STEPS + 1):
        batch = dataset.sample(Config.BATCH_SIZE)
        metrics = train_step(model, optimizer, batch)

        if step % Config.LOG_FREQ == 0:
            log_dict = {
                "train/actor_loss": metrics["actor_loss"],
                "train/critic_loss": metrics["critic_loss"],
                "train/mean_weight": metrics["mean_weight"],
                "train/weight_clip_frac": metrics["weight_clip_frac"],
                "train/explained_variance": metrics["explained_variance"],
                "value_debug/predicted_value": metrics["mean_value"],
                "value_debug/actual_return": metrics["mean_return"],
            }
            if not args.no_wandb:
                wandb.log(log_dict, step=step)
            if step % (Config.LOG_FREQ * 10) == 0:
                print(
                    f"Step {step}/{Config.TOTAL_STEPS}: actor_loss={metrics['actor_loss']:.4f}, "
                    f"critic_loss={metrics['critic_loss']:.4f}, expl_var={metrics['explained_variance']:.3f}"
                )

        if step % Config.EVAL_FREQ == 0:
            run_eval(model, step)

        if step % Config.SAVE_FREQ == 0:
            ckpt_path = os.path.join(Config.SAVE_DIR, f"awr_checkpoint_{step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at step {step}")

    final_path = os.path.join(Config.SAVE_DIR, "awr_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
