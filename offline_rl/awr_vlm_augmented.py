import os
import glob
import argparse
import numpy as np
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
    DATA_DIR = "/data/group_data/rl/geney/craftax_labelled_results_with_returns"
    DATA_GLOB = "trajectories_batch_*.npz"

    # Model
    ACTION_DIM = 43
    LAYER_WIDTH = 512
    HIDDEN_STATE_DIM = 2560  # After pooling (80, 2560) → (2560,)

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
    SAVE_DIR = "/data/group_data/rl/geney/checkpoints/awr_augmented/"
    SEED = 42

    # Wandb
    WANDB_PROJECT = "craftax-offline-awr"
    WANDB_ENTITY = "iris-sobolmark"


# ==============================================================================
# 2. Model Architecture (Augmented with Hidden States)
# ==============================================================================
class ActorCriticConvAug(nn.Module):
    def __init__(self, action_dim, layer_width, hidden_state_dim):
        super().__init__()

        # CNN encoder for images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.img_feat_dim = 512

        # Fusion
        self.combined_dim = self.img_feat_dim + hidden_state_dim  # 512 + 2560 = 3072

        # Actor head
        self.actor_fc1 = nn.Linear(self.combined_dim, layer_width)
        self.actor_fc2 = nn.Linear(layer_width, action_dim)
        self.actor_fc3 = nn.Linear(action_dim, action_dim)

        # Critic head
        self.critic_fc1 = nn.Linear(self.combined_dim, layer_width)
        self.critic_fc2 = nn.Linear(layer_width, 1)

        self.apply_init()

    def apply_init(self):
        # Orthogonal initialization
        orthogonal_init(self.conv1, gain=nn.init.calculate_gain("relu"))
        orthogonal_init(self.conv2, gain=nn.init.calculate_gain("relu"))
        orthogonal_init(self.conv3, gain=nn.init.calculate_gain("relu"))
        orthogonal_init(self.actor_fc1, gain=2.0)
        orthogonal_init(self.actor_fc2, gain=0.01)
        orthogonal_init(self.actor_fc3, gain=0.01)
        orthogonal_init(self.critic_fc1, gain=2.0)
        orthogonal_init(self.critic_fc2, gain=1.0)

    def forward(self, obs, hidden_state):
        # obs: (B, H, W, C) -> (B, C, H, W)
        x = obs.permute(0, 3, 1, 2)

        # CNN
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten
        img_embed = x.reshape(x.size(0), -1)  # (B, 512)

        # Concatenate with hidden state
        combined = torch.cat([img_embed, hidden_state], dim=1)  # (B, 3072)

        # Actor
        actor_x = F.relu(self.actor_fc1(combined))
        actor_x = F.relu(self.actor_fc2(actor_x))
        actor_logits = self.actor_fc3(actor_x)
        pi = Categorical(logits=actor_logits)

        # Critic
        critic_x = F.relu(self.critic_fc1(combined))
        value = self.critic_fc2(critic_x)

        return pi, value.squeeze(-1)


# ==============================================================================
# 3. Dataset Loader (Augmented with Hidden States)
# ==============================================================================
class OfflineDatasetAugmented:
    def __init__(self, data_dir, file_pattern):
        search_path = os.path.join(data_dir, file_pattern)
        files = glob.glob(search_path)
        if not files:
            raise ValueError(f"No files found at {search_path}")

        files = sorted(files)
        print(f"Found {len(files)} VLM-labelled files.")

        # Count total samples
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

        # Allocate arrays
        self.obs = np.zeros((total_samples, 130, 110, 3), dtype=np.uint8)
        self.action = np.zeros((total_samples,), dtype=np.int32)
        self.reward = np.zeros((total_samples,), dtype=np.float32)
        self.done = np.zeros((total_samples,), dtype=np.float32)
        self.hidden_state = np.zeros(
            (total_samples, Config.HIDDEN_STATE_DIM), dtype=np.float32
        )
        self.return_to_go = np.zeros((total_samples,), dtype=np.float32)

        def load_single_file(args):
            fpath, expected_n = args
            try:
                with np.load(fpath) as data:
                    # Load images
                    raw_obs = data["obs"]
                    if raw_obs.dtype == np.float32 and raw_obs.max() <= 1.1:
                        np.multiply(raw_obs, 255, out=raw_obs)
                    obs_uint8 = raw_obs.astype(np.uint8)

                    # Load hidden states (N, 80, 2560) -> mean pool to (N, 2560)
                    raw_hidden = data["hidden_state"]  # (N, 80, 2560)
                    pooled_hidden = np.mean(raw_hidden, axis=1).astype(
                        np.float32
                    )  # (N, 2560)

                    # Use pre-computed return_to_go from labelled data
                    return {
                        "obs": obs_uint8,
                        "action": data["action"],
                        "reward": data["reward"].astype(np.float32),
                        "done": data["done"],
                        "hidden_state": pooled_hidden,
                        "return_to_go": data["return_to_go"].astype(np.float32),
                        "count": len(raw_obs),
                    }
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                return None

        # Parallel loading
        print("Starting parallel load (8 workers)...")
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
                self.reward[idx : idx + n] = result["reward"]
                self.done[idx : idx + n] = result["done"]
                self.hidden_state[idx : idx + n] = result["hidden_state"]
                self.return_to_go[idx : idx + n] = result["return_to_go"]
                idx += n

        self.size = idx
        print(f"Dataset loaded. Total samples: {self.size}")

        # Compute statistics for monitoring (hidden states might benefit from normalization)
        self.hidden_mean = np.mean(self.hidden_state[: self.size], axis=0)
        self.hidden_std = np.std(self.hidden_state[: self.size], axis=0)
        self.hidden_std = np.where(self.hidden_std < 1e-6, 1.0, self.hidden_std)
        print(
            f"Hidden state stats - Mean range: [{self.hidden_mean.min():.3f}, {self.hidden_mean.max():.3f}], "
            f"Std range: [{self.hidden_std.min():.3f}, {self.hidden_std.max():.3f}]"
        )

        # Save normalization statistics for evaluation
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        stats_path = os.path.join(Config.SAVE_DIR, "hidden_state_stats.npz")
        np.savez(stats_path, mean=self.hidden_mean, std=self.hidden_std)
        print(f"Saved normalization statistics to {stats_path}")

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        obs_t = (
            torch.tensor(self.obs[idx], dtype=torch.float32, device=Config.DEVICE)
            / 255.0
        )
        action_t = torch.tensor(
            self.action[idx], dtype=torch.long, device=Config.DEVICE
        )

        # Normalize hidden states (VLM outputs can have unusual scales) TODO CHECK IF THIS IS BAD
        hidden_normalized = (
            self.hidden_state[idx] - self.hidden_mean
        ) / self.hidden_std
        hidden_t = torch.tensor(
            hidden_normalized, dtype=torch.float32, device=Config.DEVICE
        )

        # Use return_to_go as-is (same scale as baseline data)
        rtg_t = torch.tensor(
            self.return_to_go[idx], dtype=torch.float32, device=Config.DEVICE
        )

        return {
            "obs": obs_t,
            "action": action_t,
            "hidden_state": hidden_t,
            "return_to_go": rtg_t,
        }


# ==============================================================================
# 4. Training Step
# ==============================================================================
def train_step(model, optimizer, batch):
    pi, current_v = model(batch["obs"], batch["hidden_state"])

    td_target = batch["return_to_go"]
    advantage = td_target - current_v

    # Critic loss
    critic_loss = 0.5 * torch.mean(advantage.pow(2))

    # Actor loss (AWR)
    log_probs = pi.log_prob(batch["action"])
    weights = torch.exp(advantage.detach() / Config.AWR_BETA)
    weights_clipped = torch.clamp(weights, max=Config.AWR_MAX_WEIGHT)
    actor_loss = -torch.mean(log_probs * weights_clipped)

    total_loss = critic_loss + actor_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Metrics
    with torch.no_grad():
        clip_frac = (weights >= Config.AWR_MAX_WEIGHT).float().mean().item()
        var_diff = torch.var(advantage)
        var_return = torch.var(batch["return_to_go"])
        explained_var = 1.0 - (var_diff / (var_return + 1e-8))

        # Hidden state statistics (normalized, as fed to model)
        hidden_mean = batch["hidden_state"].mean().item()
        hidden_std = batch["hidden_state"].std().item()
        hidden_min = batch["hidden_state"].min().item()
        hidden_max = batch["hidden_state"].max().item()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": pi.entropy().mean().item(),
        "mean_weight": weights_clipped.mean().item(),
        "weight_clip_frac": clip_frac,
        "mean_value": current_v.detach().mean().item(),
        "mean_return": batch["return_to_go"].mean().item(),
        "explained_variance": explained_var.item(),
        "hidden_mean": hidden_mean,
        "hidden_std": hidden_std,
        "hidden_min": hidden_min,
        "hidden_max": hidden_max,
    }


# ==============================================================================
# 5. CLI Arguments
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="VLM-Augmented AWR for Craftax")
    parser.add_argument("--data_dir", type=str, default=Config.DATA_DIR)
    parser.add_argument("--save_dir", type=str, default=Config.SAVE_DIR)
    parser.add_argument("--total_steps", type=int, default=Config.TOTAL_STEPS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--awr_beta", type=float, default=Config.AWR_BETA)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Custom WandB run name (default: auto-generated with timestamp)",
    )
    parser.add_argument("--save_freq", type=int, default=Config.SAVE_FREQ)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


# ==============================================================================
# 6. Main Training Loop
# ==============================================================================
def main():
    args = parse_args()

    # Update config
    Config.DATA_DIR = args.data_dir
    Config.SAVE_DIR = args.save_dir
    Config.TOTAL_STEPS = args.total_steps
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.AWR_BETA = args.awr_beta
    Config.SEED = args.seed
    Config.SAVE_FREQ = args.save_freq

    # Set unique wandb name with timestamp
    if args.wandb_name:
        Config.WANDB_NAME = args.wandb_name
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        Config.WANDB_NAME = f"awr-augmented-vlm-{timestamp}"

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

    print(f"Starting VLM-Augmented AWR training with {Config.TOTAL_STEPS} steps")
    print(f"Data: {Config.DATA_DIR}")
    print(f"Checkpoints: {Config.SAVE_DIR}")
    print(f"AWR Beta: {Config.AWR_BETA}")
    print("\nInitializing model...")

    model = ActorCriticConvAug(
        action_dim=Config.ACTION_DIM,
        layer_width=Config.LAYER_WIDTH,
        hidden_state_dim=Config.HIDDEN_STATE_DIM,
    ).to(Config.DEVICE)
    print("Model initialized successfully!")

    print("Creating optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    print("Optimizer created!")

    print("\n" + "=" * 60)
    print("Loading dataset (this may take several minutes)...")
    print("=" * 60)
    dataset = OfflineDatasetAugmented(Config.DATA_DIR, Config.DATA_GLOB)
    print("\n" + "=" * 60)
    print("Dataset loaded successfully!")
    print("=" * 60)

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
                "hidden_states/mean": metrics["hidden_mean"],
                "hidden_states/std": metrics["hidden_std"],
                "hidden_states/min": metrics["hidden_min"],
                "hidden_states/max": metrics["hidden_max"],
            }
            if not args.no_wandb:
                wandb.log(log_dict, step=step)
            if step % (Config.LOG_FREQ * 10) == 0:
                print(
                    f"Step {step}/{Config.TOTAL_STEPS}: actor={metrics['actor_loss']:.4f}, "
                    f"critic={metrics['critic_loss']:.4f}, expl_var={metrics['explained_variance']:.3f}, "
                    f"hidden_mean={metrics['hidden_mean']:.4f}, hidden_std={metrics['hidden_std']:.4f}"
                )

        if step % Config.SAVE_FREQ == 0:
            ckpt_path = os.path.join(Config.SAVE_DIR, f"awr_aug_checkpoint_{step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint at step {step}")

    final_path = os.path.join(Config.SAVE_DIR, "awr_aug_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
