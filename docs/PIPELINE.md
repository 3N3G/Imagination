# Craftax Trajectory Pipeline: Complete Documentation

## Table of Contents
1. [Codebase Understanding](#1-codebase-understanding)
2. [Current Data Format](#2-current-data-format)
3. [Observation Structure](#3-observation-structure)
4. [Existing LLM Labelling Pipeline](#4-existing-llm-labelling-pipeline)
5. [Prompt Templates](#5-prompt-templates)
6. [Downstream Consumers](#6-downstream-consumers)
7. [Planned Changes: Gemini Golden Labels + Compressed Storage](#7-planned-changes)
8. [Implementation Plan](#8-implementation-plan)

---

## 1. Codebase Understanding

### Key Directories & Files

| Path | Purpose |
|------|---------|
| `~/Craftax_Baselines/ppo.py` | PPO training + trajectory saving (save_callback at line 630) |
| `~/Craftax_Baselines/labelling/` | All labelling scripts |
| `~/Craftax_Baselines/labelling/obs_to_text.py` | Decodes 8268-dim obs vector → human-readable text |
| `~/Craftax_Baselines/labelling/add_text_obs.py` | Batch adds text_obs to NPZ files |
| `~/Craftax_Baselines/labelling/vllm_labeller.py` | vLLM-based text generation (Qwen3-4B) |
| `~/Craftax_Baselines/labelling/llm_worker.py` | vLLM hidden state extraction worker |
| `~/Craftax_Baselines/labelling/extract_hidden_states.py` | Prefill-based hidden state extraction (10-30x faster) |
| `~/Craftax_Baselines/labelling/addtoqueue_llm.py` | Redis queue job dispatcher |
| `~/Craftax_Baselines/utils/llm_prompts.py` | Shared prompts, filter_text_obs(), SYSTEM_PROMPT |
| `~/Craftax_Baselines/utils/llm_extractor.py` | HuggingFace-based hidden state extractor class |
| `~/Craftax_Baselines/utils/vllm_hidden_connector.py` | Custom vLLM connector for hidden states (EAGLE3-based) |
| `~/Craftax_Baselines/utils/vllm_batch_extractor.py` | Optimized batch inference wrapper |
| `~/Craftax_Baselines/offline_rl/awr_llm_augmented.py` | AWR offline RL with LLM hidden states |
| `~/Craftax_Baselines/scripts/future_imagination_eval.py` | Prompt-template driven evaluation framework |
| `~/Craftax_Baselines/configs/future_imagination/templates/` | Prompt template files |
| `/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/` | Sharded tar.gz trajectory storage |

### Craftax Environment

- **Package location:** `/data/user_data/geney/.conda/envs/craftax_fast_llm/lib/python3.10/site-packages/craftax/`
- **Symbolic renderer:** `craftax/craftax/renderer.py` → `render_craftax_symbolic()` (line 9)
- **Constants:** `craftax/craftax/constants.py` → OBS_DIM=(9,11), BlockType (37 types), ItemType (5 types), 43 actions
- **Environment:** Uses JAX, produces symbolic observations of shape (8268,)

---

## 2. Current Data Format

### Storage Structure
- **Location:** `/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/`
- **Format:** `shard_XXXXXX.tar.gz` + `shard_XXXXXX.files` (manifest)
- **Shard count:** 50+ shards
- Each shard contains 2-4 NPZ files from `new_craftax_llm_labelled_results/`
- Each NPZ is named `trajectories_batch_NNNNNN.npz`

### NPZ Keys (after full pipeline)

| Key | dtype | Shape per sample | Bytes/sample | Notes |
|-----|-------|-----------------|-------------|-------|
| `obs` | float32 | (8268,) | 33,072 | Symbolic observation |
| `next_obs` | float32 | (8268,) | 33,072 | Next step observation |
| `action` | int/float | (1,) | 4 | Action taken (0-42) |
| `reward` | float32 | (1,) | 4 | Reward signal |
| `done` | bool/float | (1,) | 4 | Episode termination |
| `log_prob` | float32 | (1,) | 4 | Policy log probability |
| `text_obs` | object/str | variable | ~1-2 KB | Human-readable obs text |
| `text_generated` | `<U2048` | variable | ~2 KB | LLM chain-of-thought output |
| `hidden_state` | float16 | (256, 2560) or (2560,) | 1,310,720 or 5,120 | LLM hidden states |

### Data Pipeline Stages

```
Stage 1: ppo.py save_callback
  → obs, next_obs, action, reward, done, log_prob (all float32)

Stage 2: add_text_obs.py
  → adds text_obs (string) via obs_to_text() decoder

Stage 3: vllm_labeller.py (or llm_worker.py)
  → adds text_generated (LLM output string)
  → llm_worker.py: adds hidden_state (N, 2560) — last-token only
  → OR extract_hidden_states.py: adds hidden_state (N, 256, 2560) — all generated tokens

Stage 4: Sharding into tar.gz for storage
```

### Current LLM Model
- **Model:** Qwen/Qwen3-4B-Thinking-2507
- **Hidden size:** 2560
- **Tokens generated:** 256
- **Max sequence length:** 8448 (prompt ~8192 + generation 256)

### Current Prompts (Gameplay/Action)
The current labelling prompts ask the LLM to play the game: "Given the game state, what action should you take?" with chain-of-thought reasoning in `<think>` blocks.

---

## 3. Observation Structure (8268 dimensions)

### Source: `render_craftax_symbolic()` in `craftax/craftax/renderer.py`

The observation is a flat float32 vector constructed by concatenating:

```python
all_flattened = jnp.concatenate([
    all_map.flatten(),    # 8217 dims (map blocks + items + mobs + light)
    inventory,            # 16 dims
    potions,              # 6 dims
    intrinsics,           # 9 dims
    direction,            # 4 dims
    armour,               # 4 dims
    armour_enchantments,  # 4 dims
    special_values,       # 8 dims
])  # Total: 8268
```

### Detailed Breakdown

#### Map Section (8217 dims) — ALL BINARY {0, 1}

| Component | Shape | Dims | Type | Encoding |
|-----------|-------|------|------|----------|
| Block types | (9, 11, 37) | 3663 | Binary | One-hot over 37 BlockType values |
| Item types | (9, 11, 5) | 495 | Binary | One-hot over 5 ItemType values |
| Mob indicators | (9, 11, 40) | 3960 | Binary | 5 mob classes × 8 types, presence flags |
| Light/visibility | (9, 11, 1) | 99 | Binary | Thresholded boolean (> 0.05) |

#### Inventory Section (16 dims) — MIXED

| Dim Index | Field | Encoding | Range | Type |
|-----------|-------|----------|-------|------|
| 0-9 | wood, stone, coal, iron, diamond, sapphire, ruby, sapling, torches, arrows | `sqrt(count)/10` | [0, ~3.16] | Continuous float |
| 10 | books | `count/2` | {0, 0.5, 1.0} | Discrete float |
| 11 | pickaxe | `level/4` | {0, 0.25, 0.5, 0.75, 1.0} | Discrete float |
| 12 | sword | `level/4` | {0, 0.25, 0.5, 0.75, 1.0} | Discrete float |
| 13 | sword_enchantment | raw | {0, 1} | Binary |
| 14 | bow_enchantment | raw | {0, 1} | Binary |
| 15 | bow | raw | {0, 1} | Binary |

#### Potions (6 dims) — Continuous float
- `sqrt(count)/10` for each of 6 potion types

#### Intrinsics (9 dims) — Continuous float
- `value/10` for: health, food, drink, energy, mana, xp, dexterity, strength, intelligence

#### Direction (4 dims) — Binary one-hot
- One-hot encoding of direction (up/down/left/right)

#### Armour (4 dims) — Discrete float
- `value/2` for 4 armour slots → values in {0, 0.5, 1.0}

#### Armour Enchantments (4 dims) — Binary
- Boolean flags for 4 armour enchantment slots

#### Special Values (8 dims) — Mixed

| Dim | Field | Type | Range |
|-----|-------|------|-------|
| 0 | light_level | Continuous float | [0, 1] |
| 1 | is_sleeping | Binary | {0, 1} |
| 2 | is_resting | Binary | {0, 1} |
| 3 | learned_fireball | Binary | {0, 1} |
| 4 | learned_iceball | Binary | {0, 1} |
| 5 | floor | Discrete float | `level/10`, {0, 0.1, ..., 0.9} |
| 6 | ladder_open | Binary | {0, 1} (from `>=` comparison) |
| 7 | boss_vulnerable | Binary | {0, 1} |

### Summary: Binary vs Float

| Category | Dims | Binary? |
|----------|------|---------|
| Map (blocks+items+mobs+light) | 8217 | YES — all binary |
| Inventory (enchantments, bow) | 3 | YES — binary |
| Direction | 4 | YES — one-hot binary |
| Armour enchantments | 4 | YES — binary |
| Special (sleeping, resting, spells, ladder, boss) | 5 | YES — binary |
| **Total binary** | **8233** | |
| Inventory (counts, levels) | 13 | NO — continuous/discrete float |
| Potions | 6 | NO — continuous float |
| Intrinsics | 9 | NO — continuous float |
| Armour | 4 | NO — discrete float |
| Special (light_level, floor) | 2 | NO — continuous/discrete float |
| **Total float** | **34** | |
| Books (count/2) | 1 | Borderline — only {0, 0.5, 1.0} |
| **GRAND TOTAL** | **8268** | **99.6% binary** |

### obs_to_text() Decoder

Located at `~/Craftax_Baselines/labelling/obs_to_text.py`. Key functions:
- `obs_to_text(obs)` — single observation → text string
- `obs_to_text_batch(obs_array)` — batch version
- `decode_map_section(map_flat)` — decodes 8217-dim map → block_types, item_types, mob_map, light_map
- `decode_inventory_section(inv_flat)` — decodes 51-dim inventory section

The decoder reconstructs counts via inverse encoding:
- `sqrt(count)/10` → `count = round((val * 10)^2)`
- `level/4` → `level = round(val * 4)`
- etc.

### BlockType Enum (37 values)
INVALID=0, OUT_OF_BOUNDS=1, GRASS=2, WATER=3, STONE=4, TREE=5, WOOD=6, PATH=7, COAL=8, IRON=9, DIAMOND=10, CRAFTING_TABLE=11, FURNACE=12, SAND=13, LAVA=14, PLANT=15, RIPE_PLANT=16, WALL=17, DARKNESS=18, WALL_MOSS=19, STALAGMITE=20, SAPPHIRE=21, RUBY=22, CHEST=23, FOUNTAIN=24, FIRE_GRASS=25, ICE_GRASS=26, GRAVEL=27, FIRE_TREE=28, ICE_SHRUB=29, ENCHANTMENT_TABLE_FIRE=30, ENCHANTMENT_TABLE_ICE=31, NECROMANCER=32, GRAVE=33, GRAVE2=34, GRAVE3=35, NECROMANCER_VULNERABLE=36

### ItemType Enum (5 values)
NONE=0, TORCH=1, LADDER_DOWN=2, LADDER_UP=3, LADDER_DOWN_BLOCKED=4

### Mob Classes (5 × 8 = 40 types)
- Class 0 (Melee): Zombie, Gnome Warrior, Orc Soldier, Lizard, Knight, Troll, Fire Elemental, Ice Elemental
- Class 1 (Passive): Cow, Bat, Snail, Frog, Deer, Golem, Imp, Penguin
- Class 2 (Ranged): Archer, Gnome Archer, Orc Archer, Crocodile, Archer Knight, Troll Archer, Fire Mage, Ice Mage
- Class 3 (Mob Projectiles): Arrow, Dagger, Fireball, Iceball, Arrow, Slimeball, Fireball, Iceball
- Class 4 (Player Projectiles): Same as class 3 but player-owned

---

## 4. Existing LLM Labelling Pipeline

### Stage 1: Raw Data Collection (ppo.py)
- `save_callback` (line 630) triggers every `SAVE_TRAJ_EVERY` updates
- Converts JAX arrays to numpy, reshapes from (NUM_STEPS, NUM_ENVS, ...) to (N, ...)
- Saves as compressed NPZ via `save_trajectory_batch()`
- Default save path: `/data/group_data/rl/geney/craftax_unlabelled_symbolic_with_text/`

### Stage 2: Text Observation Addition (add_text_obs.py)
- Post-processing step after data collection
- Runs `obs_to_text()` on each observation
- Adds `text_obs` key to existing NPZ files (overwrites in place)
- Can be parallelized with `--num_workers`

### Stage 3a: LLM Text Generation (vllm_labeller.py)
- Uses vLLM with Qwen3-4B-Thinking-2507
- Applies `filter_text_obs()` to remove background tiles (grass, sand, gravel, etc.)
- Uses SYSTEM_PROMPT (gameplay instructions) + FEW_SHOT_EXAMPLES + user prompt
- Generates 256 tokens per observation
- Stores in `text_generated` field
- Can run via Redis queue for distributed processing

### Stage 3b: Hidden State Extraction
Two approaches:
1. **llm_worker.py** — extracts last-token hidden state `(N, 2560)` via vLLM server
   - Uses `VLLMHiddenStateExtractor` from `utils/llm_extractor.py`
   - Supports GENERATE_TEXT=True (generate then extract) or False (direct extraction, ~34x faster)
   - Saves progress via memmap files for crash recovery

2. **extract_hidden_states.py** — extracts all 256 generated token hidden states `(N, 256, 2560)` via HuggingFace prefill
   - Requires text_generated to already exist in the NPZ
   - Reconstructs full sequences (prompt + generated text) and does single forward pass
   - Stores hidden states from model.model (base model, not lm_head)
   - Uses Flash Attention 2

### Stage 4: Sharding
Files are tar.gz'd into shards with .files manifests for the storage directory.

### vLLM Hidden State Infrastructure
- `utils/vllm_hidden_connector.py` — custom connector inheriting from EAGLE3 infrastructure
- Supports `mode="all"` (all tokens) and `mode="last_token"`
- Uses vLLM's KV transfer mechanism to extract hidden states during inference
- **Proven to work with Qwen3-8B** (the EAGLE3 plugin was literally built for this model)

---

## 5. Prompt Templates

### Location: `~/Craftax_Baselines/configs/future_imagination/templates/`

| File | Description |
|------|-------------|
| `oracle_next15_prompt.txt` | Oracle with privileged 15-step future. **This is the one we'll use for Gemini.** |
| `oracle_privileged_prompt.txt` | Oracle with full privileged future (variable horizon) |
| `predict_history_k_prompt.txt` | Prediction from state + k-step history (verbose) |
| `predict_state_only_prompt.txt` | Prediction from state only (verbose) |
| `predict_history_k_prompt_concise.txt` | Prediction from state + history (concise, detailed game rules) |
| `predict_state_only_prompt_concise.txt` | Prediction from state only (concise, detailed game rules) |

### oracle_next15_prompt.txt (to be used with Gemini)
```
Headline: <one line>

Meaningful events (ordered):
1. [t+<offset>] <event>
2. [t+<offset>] <event>
3. ...

Trajectory summary:
<Narrative summary of what happens over approximately t+1 through t+15.>

Predicted Next Action: <Action that led the player from t+0 to t+1>
```

Template variables:
- `{current_state_compact}` — compact state representation
- `{future_state_block}` — privileged future trajectory (stride=1, 15 steps)

### predict_*_concise.txt (for eventual online inference)
These are the prompts that would be used at inference time without privileged information. The concise versions include detailed Craftax game rules, algorithm, and 2 example outputs. They ask the model to predict the next 15 steps based on current state (and optionally history).

---

## 6. Downstream Consumers

### AWR with LLM Hidden States (awr_llm_augmented.py)

**Architecture: `ActorCriticAug`**
```
actor:  obs → 512 → ‖ → 512 → 512 → action_logits
        hidden → 512 ↗
critic: obs → 512 → ‖ → 512 → 512 → value
        hidden → 512 ↗
```

**Key configurations:**
- `Config.OBS_DIM = 8268`
- `Config.HIDDEN_STATE_DIM = 2560` (will change to 4096 with Qwen3-8B)
- `Config.ACTION_DIM = 43`
- `Config.GAMMA = 0.99`
- `Config.NUM_ENVS = 128`
- `Config.AWR_BETA = 10.0`
- `Config.AWR_MAX_WEIGHT = 20.0`

**Data loading (OfflineDatasetLLMAugmented):**
- Reads `obs`, `action`, `reward`, `done`, `hidden_state` from NPZ files
- **Does NOT use `next_obs`** — confirmed
- Computes `return_to_go` from `reward` + `done` if not pre-stored
- Already supports 3 observation formats via `decode_obs_array()`:
  1. `obs` — standard float32 (8268,)
  2. `obs_map_bits` + `obs_aux` — **bitpacked binary + float16 aux** ← already implemented!
  3. `obs_map` + `obs_aux` — uint8 map + float aux
- Supports `hidden_skip_n` — holds hidden state for N steps (important for 15-step cadence)
- Supports `min_rtg_quantile` — filters by return-to-go quantile
- If hidden_state is 3D (N, T, H), it mean-pools to (N, H) automatically
- Computes running hidden state normalization statistics (mean/std)

**AWR Training:**
- Uses advantage-weighted regression: `weight = exp(advantage / beta)` clamped to AWR_MAX_WEIGHT
- `advantage = return_to_go - value_baseline`
- Actor loss: weighted negative log-likelihood
- Critic loss: MSE on return-to-go
- **Needs `return_to_go`** — computed from `reward` + `done` (gamma=0.99). Can be precomputed.

**Evaluation results from progress_journal.md (for context on return scale):**
- PPO baseline: mean_return=18.63
- Unaugmented offline: mean_return=18.04
- LLM-augmented skip25: mean_return=17.37
- LLM-augmented skip5: mean_return=15.7
- LLM-augmented offline skip25: mean_return=15.01
- LLM-augmented offline skip1: mean_return=13.23

---

## 7. Planned Changes: Gemini Golden Labels + Compressed Storage

### Overview

Replace the current LLM labels (Qwen3-4B gameplay chain-of-thought) with:
1. **Gemini-generated oracle summaries** using `oracle_next15_prompt.txt` with privileged future info
2. **Qwen3-8B embeddings** at 10 structured token positions from the Gemini text
3. **Compressed observation storage** (drop next_obs, bitpack obs)

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Gemini prompt | `oracle_next15_prompt.txt` | Oracle with ground-truth future, stride=1 |
| Embedding model | **Qwen3-8B** (not 3.5-9B) | Proven vLLM hidden state support, standard transformer arch |
| Embedding hidden size | **4096** | Qwen3-8B hidden_size (vs 2560 for Qwen3-4B) |
| Embedding layer | **Layer 30** of 36 (~83% depth) | High-level semantic features; easy to change via constant |
| Embedding positions | **10 structured positions** | Headline (2), events 1-3 (6), summary (2); **NOT** predicted action |
| Embedding shape | **(10, 4096) float16** | 80 KB/sample |
| Trajectory filtering | **Episode return >= 15** | Keeps "decent gameplay" episodes; configurable threshold |
| Gemini call frequency | **Every 15 steps** | Matches the "next 15 steps" prediction window |
| Training conditioning | **Every step**, using nearest prior Gemini label | Dense training via `hidden_skip_n=15`; sparse Gemini calls |
| Obs format | `obs_map_bits` + `obs_aux` | Bitpacked binary (8233 dims) + float16 aux (35 dims); already supported by AWR loader |
| Drop next_obs | **YES** | Not used by awr_llm_augmented.py; derivable as obs[t+1] if ever needed |
| Keep text | **YES** | Store Gemini output text for inspection/debugging/BC distillation |
| Pre-compute return_to_go | **YES** | Avoids recomputation; needed for AWR and episode filtering |
| Predicted Next Action | **Generated by Gemini** but **not embedded** | Useful for diagnostics, but not relevant for 15-step conditioning |

### Why return >= 15 (not percentile-based filtering)

- Trained PPO achieves mean_return ~18.6; offline augmented methods get 13-18
- A threshold of 15 keeps "decent" episodes — those performing at or above offline-augmented level
- Percentile-based filtering depends on the (unknown) training data distribution; absolute threshold is stable and interpretable
- **The script should print a histogram of episode returns** before filtering so you can verify the threshold is reasonable and adjust

### 10 Embedding Extraction Points

When Qwen3-8B encodes the Gemini output text via prefill, extract layer-30 hidden states at:

| # | Position | Description |
|---|----------|-------------|
| 1 | After `Headline:` | Start of headline content |
| 2 | End of headline line (`\n`) | Complete headline representation |
| 3 | Start of `1. [t+...]` | Beginning of event 1 |
| 4 | End of event 1 (before `\n2.`) | Complete event 1 representation |
| 5 | Start of `2. [t+...]` | Beginning of event 2 |
| 6 | End of event 2 | Complete event 2 representation |
| 7 | Start of `3. [t+...]` | Beginning of event 3 |
| 8 | End of event 3 | Complete event 3 representation |
| 9 | After `Trajectory summary:` or `\n` | Start of summary content |
| 10 | End of trajectory summary (before `\nPredicted`) | Complete summary representation |

**Handling missing events:** If Gemini produces fewer than 3 events, pad missing event positions with zeros.

**Note:** `Predicted Next Action:` is still generated by Gemini (for diagnostics) but positions 11-12 are **NOT** extracted. The action prediction is short-horizon and redundant when conditioning on the 15-step summary.

### Storage Format (New NPZ Keys)

| Key | dtype | Shape | Bytes/sample | Notes |
|-----|-------|-------|-------------|-------|
| `obs_map_bits` | uint8 | (N, ceil(8233/8)) = (N, 1030) | 1,030 | Bitpacked binary obs dims |
| `obs_map_dim` | int | scalar | — | = 8233 (for np.unpackbits count) |
| `obs_aux` | float16 | (N, 35) | 70 | Non-binary obs dims |
| `action` | int32 | (N,) | 4 | |
| `reward` | float32 | (N,) | 4 | |
| `done` | uint8 | (N,) | 1 | |
| `log_prob` | float32 | (N,) | 4 | |
| `return_to_go` | float32 | (N,) | 4 | Precomputed; gamma=0.99 |
| `text_generated` | object | (N,) | ~500-1500 | Gemini oracle output text |
| `hidden_state` | float16 | (N, 10, 4096) | 81,920 | Qwen3-8B layer-30 embeddings |
| `gemini_step_idx` | int32 | (N,) | 4 | Index of the Gemini label this step uses (t // 15 * 15) |

**Total per sample:** ~83 KB (vs ~1,379 KB current = **94% savings**)

After filtering (keeping only return >= 15 episodes), total dataset size shrinks further.

### Suboptimal Data Discussion

The training data comes from a learning PPO agent. Early episodes are near-random; later ones are good. Considerations:

1. **Filtering by return (>= 15)** is the primary mitigation. This keeps only episodes where the agent played decently.
2. **Gemini labels are ground-truth descriptions** of what actually happens (oracle). Even for mediocre play, the label accurately describes the trajectory. The question is whether BC on mediocre-but-accurately-described behavior is useful.
3. **AWR naturally handles this** via advantage weighting: timesteps with high return-to-go get higher BC weight. So even within a "decent" episode, the best moments are emphasized.
4. **The value of the Gemini embedding** is primarily for the *critic* (value estimation). Knowing "the next 15 steps involve taking damage and losing resources" helps the critic predict low value, regardless of whether the behavior is optimal.
5. **For the actor** (action prediction), the Gemini embedding provides context like "we're about to fight an orc" which can inform better action selection even if the training data's actions weren't optimal.

**Recommendation:** Start with return >= 15. If the resulting dataset is too small or results are poor, lower to >= 10. The script prints distribution stats so you can make an informed choice.

---

## 8. Implementation Plan

### Step 1: Untar and Index Shards
```bash
# Extract all shards to a working directory
for f in /data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/shard_*.tar.gz; do
    tar xzf "$f" -C /data/group_data/rl/geney/
done
```
- Build index: list all NPZ files, record (path, num_samples)

### Step 2: Compute Episode Returns and Filter
- For each NPZ file:
  - Load `reward`, `done` arrays
  - Compute `return_to_go` using `compute_return_to_go()` from awr_llm_augmented.py (gamma=0.99, num_envs=128)
  - Identify episode boundaries from `done` flags
  - Compute per-episode total return (= return_to_go at episode start)
  - Print histogram of episode returns
- Apply filter: keep only samples from episodes with return >= 15
- Record which (file, sample_indices) survive the filter

### Step 3: Build Gemini Input Batches
- For each surviving timestep at t where `t % 15 == 0`:
  - Load obs[t] through obs[t+15] (or fewer if episode ends)
  - Convert each to text via `obs_to_text()` + `filter_text_obs()`
  - Build `{current_state_compact}` from obs[t] using `_build_compact_state()` from future_imagination_eval.py
  - Build `{future_state_block}` from obs[t+1:t+16] (stride=1, each step as a snapshot)
  - Fill `oracle_next15_prompt.txt` template
- Batch requests to Gemini API (use async/concurrent for throughput)
- Store results in a sidecar file keyed by (npz_file, sample_index)
- **Retry/resume logic**: track which (file, idx) pairs have been completed

### Step 4: Embed Gemini Outputs with Qwen3-8B
- Load `Qwen/Qwen3-8B` with HuggingFace transformers + Flash Attention 2
- For each Gemini output text:
  - Tokenize the full text
  - Find the 10 extraction positions by searching for marker strings in tokens:
    - `"Headline:"` → token after this
    - First `"\n"` after headline → token before this
    - `"1."`, `"2."`, `"3."` → tokens for event starts
    - `"\n"` after each event → tokens for event ends
    - `"Trajectory summary:"` → token after this (or after `\n`)
    - `"\n"` or `"\nPredicted"` → token before this for summary end
  - Run forward pass with `output_hidden_states=True`
  - Extract hidden states from layer index 30 at the 10 positions
  - Store as (10, 4096) float16
- Batch processing: process multiple texts per forward pass with padding

### Step 5: Repack NPZ Files
- For each original NPZ, create new compressed NPZ:
  - **Compress obs**: split into binary (8233 dims) → `np.packbits()` and float (35 dims) → float16
  - **Drop next_obs**
  - **Add**: `return_to_go`, `hidden_state` (10, 4096), `text_generated`, `gemini_step_idx`
  - Save with `np.savez_compressed()`
- Only include samples that passed the return filter

### Step 6: Update AWR Configuration
- `decode_obs_array()` already supports `obs_map_bits` + `obs_aux` — no changes needed for obs loading
- Change `Config.HIDDEN_STATE_DIM = 4096` (the AWR code already mean-pools 3D hidden states via `np.mean(raw_hidden, axis=1)` at line 479)
- So (N, 10, 4096) → mean-pool → (N, 4096) happens automatically
- Set `hidden_skip_n = 15` to match Gemini generation cadence
- Set `hidden_skip_reset_on_done = True` (refresh on episode boundaries)

### Notes on obs_to_text from Compressed Obs

`obs_to_text()` expects a float32 array of shape (8268,). When obs is stored as `obs_map_bits` + `obs_aux`, use `decode_obs_array()` from awr_llm_augmented.py to reconstruct the full float32 vector first. This is already implemented.

### Gemini API Cost Estimate

Rough calculation (will refine after seeing actual data size):
- Assume ~5M total transitions in dataset
- After return >= 15 filter: keep ~30-50% → ~1.5-2.5M transitions
- Gemini called every 15 steps: ~100-170K calls
- Gemini 2.5 Flash pricing: ~$0.15/M input tokens, ~$0.60/M output tokens
- Input per call: ~2-3K tokens (current state + 15 future states)
- Output per call: ~200-400 tokens
- **Estimated cost: $20-50** (very rough; depends on actual data size and response lengths)

### HuggingFace Qwen3-8B for Structured Embedding

We use HuggingFace (not vLLM) for the embedding step because:
- vLLM's hidden state connector only supports "all tokens" or "last token" extraction
- We need hidden states at **specific token positions** (the 10 marker positions)
- This is a batch offline job; vLLM's speed advantage matters less than extraction flexibility
- The existing `extract_hidden_states.py` already demonstrates this approach with HuggingFace

Key code pattern:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

EXTRACT_LAYER = 30  # Easy to change

with torch.no_grad():
    outputs = model.model(  # .model to get base model (not lm_head)
        input_ids=...,
        attention_mask=...,
        output_hidden_states=True,
    )
    layer_hidden = outputs.hidden_states[EXTRACT_LAYER]  # (batch, seq, 4096)
    # Index into specific token positions
    embeddings = layer_hidden[:, extraction_positions, :]  # (batch, 10, 4096)
```

---

## Appendix: Key Code References

### obs_to_text.py constants
```python
OBS_DIM = (9, 11)
NUM_BLOCK_TYPES = 37
NUM_ITEM_TYPES = 5
NUM_MOB_CLASSES = 5
NUM_MOB_TYPES = 8
MOB_CHANNELS = 40
MAP_CHANNELS = 37 + 5 + 40 + 1 = 83
MAP_OBS_SIZE = 9 * 11 * 83 = 8217
INVENTORY_OBS_SIZE = 51
TOTAL_OBS_SIZE = 8268
```

### Binary dimension indices (for bitpacking)
- Indices 0–8216: Map section (all binary after one-hot + indicator encoding)
- Within inventory (indices 8217–8267):
  - Binary: indices 8230 (sword_enchantment), 8231 (bow_enchantment), 8232 (bow)
  - Binary: indices 8247–8250 (direction one-hot)
  - Binary: indices 8255–8258 (armour enchantments)
  - Binary: indices 8260 (is_sleeping), 8261 (is_resting), 8262 (learned_fireball), 8263 (learned_iceball)
  - Binary: indices 8265 (ladder_open), 8266 (boss_vulnerable)

### Non-binary dimension indices (35 total)
- 8217–8226: inventory counts (10 items, sqrt-normalized)
- 8227: books (count/2)
- 8228: pickaxe (level/4)
- 8229: sword (level/4)
- 8233–8238: potions (6 types, sqrt-normalized)
- 8239–8247: intrinsics (9 stats, /10 normalized)
- 8251–8254: armour (4 slots, /2 normalized)
- 8259: light_level (continuous)
- 8264: floor (level/10)

### TMPDIR Setup for Claude Code
```bash
export TMPDIR=/data/user_data/geney/tmp_claude
mkdir -p $TMPDIR
claude
```
