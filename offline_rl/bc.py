import os
import glob
import argparse
import numpy as np
import cv2
import concurrent.futures
import wandb
import time

# --- FIX: Prevent Import Hangs / Deadlocks on Clusters ---
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

    # BC Hyperparameters
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    TOTAL_STEPS = 100_000
    BATCH_SIZE = 256
    LOG_FREQ = 100
    SAVE_FREQ = 25000
    EVAL_FREQ = 5000
    SAVE_DIR = "/data/group_data/rl/geney/checkpoints/bc_noaug/"
    SEED = 42

    # Eval
    EVAL_EPISODES = 3
    V_MIN_GRAPH = -1.0
    V_MAX_GRAPH = 15.0

    # Wandb
    WANDB_PROJECT = "craftax-offline-bc"
    WANDB_ENTITY = "iris-sobolmark"
    WANDB_NAME = "bc-baseline-v1"


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

        # Critic Head (Kept for compatibility/eval visualization, but not trained in BC)
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
# 3. Dataset Loader (Unchanged)
# ==============================================================================
class OfflineDataset:
    def __init__(self, data_dir, file_pattern):
        search_path = os.path.join(data_dir, file_pattern)
        files = glob.glob(search_path)
        if not files:
            raise ValueError(f"No files found at {search_path}")

        files = sorted(files)
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

        self.obs = np.zeros((total_samples, 130, 110, 3), dtype=np.uint8)
        self.action = np.zeros((total_samples,), dtype=np.int32)
        # Note: reward/done/rtg kept for eval compatibility but not used in BC training
        self.return_to_go = np.zeros((total_samples,), dtype=np.float32)

        def load_single_file(args):
            fpath, expected_n = args
            try:
                with np.load(fpath) as data:
                    raw_obs = data["obs"]
                    if raw_obs.dtype == np.float32 and raw_obs.max() <= 1.1:
                        np.multiply(raw_obs, 255, out=raw_obs)
                    obs_uint8 = raw_obs.astype(np.uint8)

                    rtg = (
                        data["return_to_go"]
                        if "return_to_go" in data
                        else np.zeros_like(data["action"])
                    )

                    return {
                        "obs": obs_uint8,
                        "action": data["action"],
                        "return_to_go": rtg,
                        "count": len(raw_obs),
                    }
            except Exception as e:
                return None

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
                self.obs[idx : idx + n] = result["obs"]
                self.action[idx : idx + n] = result["action"]
                self.return_to_go[idx : idx + n] = result["return_to_go"]
                idx += n

        self.size = idx

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs_t = (
            torch.tensor(self.obs[idx], dtype=torch.float32, device=Config.DEVICE)
            / 255.0
        )
        action_t = torch.tensor(
            self.action[idx], dtype=torch.long, device=Config.DEVICE
        )
        rtg_t = torch.tensor(
            self.return_to_go[idx], dtype=torch.float32, device=Config.DEVICE
        )
        return {"obs": obs_t, "action": action_t, "return_to_go": rtg_t}


# ==============================================================================
# 4. Training Logic (BC Simplified)
# ==============================================================================
def train_step(model, optimizer, batch):
    # 1. Forward pass
    pi, current_v = model(batch["obs"])

    # 2. Behavioral Cloning Loss (Cross Entropy)
    # We want to maximize the log probability of the action taken in the dataset
    log_probs = pi.log_prob(batch["action"])
    bc_loss = -log_probs.mean()

    # 3. Optional: Still update critic if you want to see the value graph during eval
    # Even in BC, a value head can be trained as a "passive observer" of the data
    td_target = batch["return_to_go"]
    critic_loss = 0.5 * F.mse_loss(current_v, td_target)

    # In pure BC, we usually only care about bc_loss.
    # Here we sum them so the value graph remains meaningful.
    total_loss = bc_loss + critic_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # --- Metrics ---
    with torch.no_grad():
        # Accuracy: How often the model picks the same action as the dataset
        preds = torch.argmax(pi.logits, dim=-1)
        accuracy = (preds == batch["action"]).float().mean().item()

    return {
        "bc_loss": bc_loss.item(),
        "critic_loss": critic_loss.item(),
        "accuracy": accuracy,
        "entropy": pi.entropy().mean().item(),
        "mean_value": current_v.mean().item(),
    }


# ==============================================================================
# 5. Evaluation Logic (Unchanged)
# ==============================================================================
def draw_value_graph(frame, value_history, v_min=-1.0, v_max=15.0):
    frame = np.ascontiguousarray(frame)
    h, w, c = frame.shape
    footer_h = 50
    new_frame = np.zeros((h + footer_h, w, c), dtype=frame.dtype)
    new_frame[0:h, 0:w] = frame
    if len(value_history) < 2:
        return new_frame

    points = value_history[-w:]
    graph_y_end = h + footer_h - 5
    graph_h = footer_h - 10

    for i in range(1, len(points)):
        y_prev = int(
            graph_y_end
            - (
                (max(v_min, min(v_max, points[i - 1])) - v_min)
                / (v_max - v_min)
                * graph_h
            )
        )
        y_curr = int(
            graph_y_end
            - ((max(v_min, min(v_max, points[i])) - v_min) / (v_max - v_min) * graph_h)
        )
        cv2.line(
            new_frame,
            (int((i - 1) / len(points) * w), y_prev),
            (int(i / len(points) * w), y_curr),
            (0, 255, 0),
            1,
        )
    return new_frame


def run_eval(model, step):
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
    rng = jax.random.PRNGKey(Config.SEED + step)
    model.eval()
    total_rewards = []
    video_frames = []

    for ep in range(Config.EVAL_EPISODES):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env.default_params)
        done, ep_reward, ep_values = False, 0, []

        while not done and len(ep_values) < 1000:
            obs_tensor = (
                torch.from_numpy(np.array(obs)).float().to(Config.DEVICE).unsqueeze(0)
            )
            if obs_tensor.max() > 1.1:
                obs_tensor /= 255.0
            with torch.no_grad():
                pi, v = model(obs_tensor)
                action = torch.argmax(pi.logits, dim=-1).item()  # Greedy for BC eval
                ep_values.append(v.item())

            if ep == 0:
                video_frames.append(
                    draw_value_graph((np.array(obs) * 255).astype(np.uint8), ep_values)
                )

            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, _ = env.step(
                step_key, env_state, action, env.default_params
            )
            ep_reward += float(reward)
        total_rewards.append(ep_reward)

    wandb.log(
        {
            "eval/avg_reward": np.mean(total_rewards),
            "eval/video": wandb.Video(
                np.transpose(np.array(video_frames), (0, 3, 1, 2)), fps=15
            )
            if video_frames
            else None,
        },
        step=step,
    )
    model.train()


# ==============================================================================
# 6. Main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Behavioral Cloning for Craftax")
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
    Config.SEED = args.seed
    Config.WANDB_NAME = args.wandb_name
    Config.EVAL_FREQ = args.eval_freq
    Config.SAVE_FREQ = args.save_freq

    torch.manual_seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    if not args.no_wandb:
        wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            name=Config.WANDB_NAME,
            config={k: v for k, v in vars(Config).items() if not k.startswith("_")},
            settings=wandb.Settings(_service_wait=300, start_method="fork"),
        )
        print(f"WandB initialized: {Config.WANDB_PROJECT}/{Config.WANDB_NAME}")

    print(f"Starting BC training with {Config.TOTAL_STEPS} steps")
    print(f"Data: {Config.DATA_DIR}")
    print(f"Checkpoints: {Config.SAVE_DIR}")

    model = ActorCriticConv(Config.ACTION_DIM, Config.LAYER_WIDTH).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    dataset = OfflineDataset(Config.DATA_DIR, Config.DATA_GLOB)

    for step in range(1, Config.TOTAL_STEPS + 1):
        batch = dataset.sample(Config.BATCH_SIZE)
        metrics = train_step(model, optimizer, batch)

        if step % Config.LOG_FREQ == 0:
            if not args.no_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            if step % (Config.LOG_FREQ * 10) == 0:
                print(
                    f"Step {step}/{Config.TOTAL_STEPS}: loss={metrics['bc_loss']:.4f}, acc={metrics['accuracy']:.3f}"
                )

        if step % Config.EVAL_FREQ == 0:
            run_eval(model, step)
        if step % Config.SAVE_FREQ == 0:
            torch.save(
                model.state_dict(), os.path.join(Config.SAVE_DIR, f"bc_step_{step}.pth")
            )
            print(f"Saved checkpoint at step {step}")

    if not args.no_wandb:
        wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
