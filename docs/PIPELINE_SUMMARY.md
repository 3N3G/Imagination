# Imagination-Augmented Offline RL Pipeline — Full Summary

**Status: PAUSED** (as of 2026-03-23, awaiting group design decisions)

---

## Goal

Train an offline RL agent (AWR) for Craftax that conditions on LLM "imagination" of future states. The pipeline uses Gemini 2.5 Flash as an oracle — given the current game state and the next 15 privileged future states, it writes a narrative summary of what happens. That text is embedded by Qwen3-8B into a dense vector that the AWR policy conditions on at training time.

---

## Pipeline Architecture (6 Phases)

### Phase 1: Environment Setup

- Conda env `craftax` (JAX-based) for PPO + data processing
- Conda env `craftax_fast_llm` (PyTorch) for Qwen3-8B embedding
- Working directory: `/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/`
- Pipeline code: `pipeline/` subdirectory (config.py, filter_and_repack.py, gemini_label.py, text_utils.py, embed.py, merge.py)

### Phase 2: PPO Training — COMPLETED

- **Job:** 6721647, partition `rl` (1 GPU), 7h52m runtime
- **Config:** `Craftax-Symbolic-v1`, 200M timesteps, 128 parallel envs, 64 steps per update
- **Script:** `~/Craftax_Baselines/online_rl/ppo.py`
- **Output:** 12,207 raw trajectory NPZ files in `raw_trajectories/` (~16GB total)
- **Each file:** shape `(64, 128, ...)` = 8,192 transitions (64 steps × 128 envs)
- **Observation format:** 8,268 floats = 8,217 binary map dims + 51 inventory/stats dims
- **Logged to:** W&B project "Unaugmented PPO", entity iris-sobolmark
- **Note:** This is an unaugmented baseline PPO — no LLM labels during training. The trajectories are the input dataset for offline RL.

### Phase 3: Filter & Repack — COMPLETED

- **Job:** 6723088, partition `cpu`, 1h55m runtime
- **Script:** `pipeline/filter_and_repack.py`

**Filtering criteria:**
- Reconstructs per-environment episode timelines from interleaved parallel rollouts
- Computes discounted return per episode (gamma=0.99)
- **Keeps only episodes with total return >= 20.0** (top ~48%)
- Discards incomplete episodes (93,357 discarded)

**Results:**
- 129,382 total episodes → 62,199 kept (48.1%)
- 99.9M total transitions → 63.7M kept (63.7%)
- 632 output files, ~100K samples each, 2.4GB total

**Output format per sample:**
- `obs_map_bits` (uint8, bitpacked) — 8,217 binary map dims packed to 1,028 bytes
- `obs_aux` (float16, 51 dims) — inventory/stats
- `action` (int32), `reward` (float32), `done` (uint8), `log_prob` (float32)
- `return_to_go` (float32) — computed backwards per episode with gamma=0.99

**Return stats:** min=-0.9, max=42.1, mean=19.0, median=19.1, std=3.2

**raw_trajectories/ deleted** after verification (16GB freed)

### Phase 4: Gemini Oracle Labelling — PAUSED (no labels saved)

- **Script:** `pipeline/gemini_label.py`
- **Model:** Gemini 2.5 Flash, temperature=0.3, max_output=512 tokens, thinking disabled
- **Call cadence:** Every 15 steps within each episode
- **Concurrency:** 20 threads, token-bucket rate limiter at 500 req/min
- **Resumable:** tracks completed `sample_idx` per JSONL file, skips on restart

**Current config: `--max-files 158`** (quarter of 632 shards):
- 158 files = **1,068,752 Gemini API calls**
- Estimated cost: **~$1,454** (based on actual avg 7,694 prompt tokens + 344 completion tokens per call)
- Estimated runtime: **~44.5 hours** at ~400 req/min (fits in one 48h SLURM run)

**Prompt template** (`~/Craftax_Baselines/configs/future_imagination/templates/oracle_next15_prompt.txt`):

Input:
- Compact current state: filtered map (interesting tiles only), stats (Health/Food/Drink/Energy/Mana/XP), direction, floor, ladder, inventory, equipment, action/reward/done
- Future state block: same format for t+1 through t+15, each wrapped in `[FUTURE STATE t+N]`

Requested output format:
```
Headline: <one line summarizing the 15-step window>

Meaningful events (exactly 3, summarizing the full window):
1. [t+<start>–t+<end>] <event summarizing a group of related steps>
2. [t+<start>–t+<end>] <event summarizing a group of related steps>
3. [t+<start>–t+<end>] <event summarizing a group of related steps>

Trajectory summary:
<Narrative summary of what happens over approximately t+1 through t+15.>

Predicted Next Action: <Action that led the player from t+0 to t+1>
```

**Observation pipeline:** raw obs (8268 floats) → `obs_to_text()` (Craftax_Baselines) → `filter_text_obs()` (removes background tiles) → `_parse_features()` → `build_compact_state()` (structured string)

**Minor format difference vs Craftax_Baselines reports:** Pipeline omits `| abs_t=Y` from future state headers (relative offsets only). Functionally equivalent — Gemini doesn't need absolute timestamps.

### Phase 5: Qwen3-8B Embedding — NOT STARTED

- **Script:** `pipeline/embed.py`
- **Model:** Qwen/Qwen3-8B
- **Extraction:** Layer 30 of 36 (~83% depth), hidden dim 4096
- **SLURM config:** partition `rl`, 1 GPU, 64GB RAM, 24h time limit, conda env `craftax_fast_llm`

**10 structured extraction positions per text:**

| Positions | Section |
|-----------|---------|
| 0–1 | Headline start/end |
| 2–3 | Event 1 start/end |
| 4–5 | Event 2 start/end |
| 6–7 | Event 3 start/end |
| 8–9 | Trajectory summary start / before Predicted Next Action |

- **Output per sample:** `(10, 4096)` float16 = ~80KB per sample
- **Batch size:** 16, max sequence length 2048
- Position finding uses regex patterns + tokenizer offset_mapping to map char positions to token indices
- Missing sections get zero vectors (robust to variable Gemini output)

### Phase 6: Merge — NOT STARTED

- **Script:** `pipeline/merge.py`
- **Logic:** For each trajectory file, loads filtered data + Gemini JSONL + embedding NPZ

**`gemini_step_idx` mapping:** Each sample is assigned to its nearest prior Gemini label within the same episode (labels at episode steps 0, 15, 30, ...). All samples between two Gemini calls share the same label/embedding.

**Final output per file (NPZ in `final_trajectories/`):**
- All original fields: obs_map_bits, obs_aux, action, reward, done, log_prob, return_to_go
- `hidden_state` (float16, `(N, 10, 4096)`) — Qwen embeddings (zeros if no match)
- `text_generated` (str array) — Gemini oracle text per sample (empty if no match)
- `gemini_step_idx` (int32) — pointer to associated Gemini label

---

## Data Scale Summary

| Metric | Full (632 shards) | Current config (158 shards) |
|--------|--------------------|-----------------------------|
| Samples | 63.7M | ~15.9M |
| Episodes | ~62K | ~15.5K |
| Gemini calls | ~4.27M | ~1.07M |
| Estimated Gemini cost | ~$5,800 | ~$1,454 |
| Estimated Gemini runtime | ~178h | ~44.5h |

---

## Downstream Consumer

`~/Craftax_Baselines/offline_rl/awr_llm_augmented.py` — AWR training that loads `final_trajectories/`, conditions policy on the `hidden_state` embeddings.

---

## Bugs Fixed Along the Way

1. `int64` not JSON serializable in gemini_label.py → cast to `int()`
2. `torch` missing in `craftax` conda env → embed script uses `craftax_fast_llm` env instead
3. SLURM scripts lacked `set -e` → Python errors weren't propagating as job failures
4. Gemini was serial (~23 req/min) → added ThreadPoolExecutor with 20 workers (~400 req/min)
5. Prompt didn't constrain event count → Gemini listed 15 events (one per step) instead of 3 summary events → Fixed prompt to specify "exactly 3" + stale labels deleted

---

## Infrastructure Notes

- NFS home (`/home/geney`) is 97% full — all SLURM scripts use `TMPDIR=/tmp`
- `/data/user_data/geney` is 100% full
- Working data lives on `/data/group_data/rl/geney/` (~1.2TB free)
- Claude Code launched with `TMPDIR=/data/group_data/rl/geney/tmp_claude`

---

## How to Resume

When ready, from the working directory:

```bash
GEMINI_JOB=$(sbatch --parsable slurm_gemini_label.sh) && \
EMBED_JOB=$(sbatch --parsable --dependency=afterok:$GEMINI_JOB slurm_embed.sh) && \
sbatch --dependency=afterok:$EMBED_JOB slurm_merge.sh
```

To adjust shard count, edit `--max-files N` in `slurm_gemini_label.sh`.

---

## Key Config Values

| Parameter | Value | Source |
|-----------|-------|--------|
| PPO total timesteps | 200M | slurm_ppo_200m.sh |
| PPO num_envs | 128 | slurm_ppo_200m.sh |
| PPO num_steps | 64 | slurm_ppo_200m.sh |
| Filter min_return | 20.0 | slurm_filter_repack.sh |
| RTG gamma | 0.99 | pipeline/config.py |
| Obs total dim | 8268 (8217 map + 51 aux) | pipeline/config.py |
| Gemini model | gemini-2.5-flash | pipeline/config.py |
| Gemini temperature | 0.3 | pipeline/config.py |
| Gemini max output tokens | 512 | pipeline/config.py |
| Gemini step cadence | 15 | pipeline/config.py |
| Gemini concurrent requests | 20 | pipeline/config.py |
| Gemini rate limit | 500 req/min | pipeline/config.py |
| Embed model | Qwen/Qwen3-8B | pipeline/config.py |
| Embed layer | 30 of 36 | pipeline/config.py |
| Embed hidden dim | 4096 | pipeline/config.py |
| Embed positions | 10 | pipeline/config.py |
| Samples per output file | 100,000 | slurm_filter_repack.sh |

---

## File Layout

```
new_craftax_llm_labelled_results_shards/
├── pipeline/
│   ├── config.py              # All shared constants/paths
│   ├── filter_and_repack.py   # Phase 3
│   ├── gemini_label.py        # Phase 4
│   ├── text_utils.py          # Obs→text conversion
│   ├── embed.py               # Phase 5
│   └── merge.py               # Phase 6
├── filtered_trajectories/     # Phase 3 output (632 files, 2.4GB)
├── gemini_labels/             # Phase 4 output (empty, ready)
├── embeddings/                # Phase 5 output (not yet created)
├── final_trajectories/        # Phase 6 output (not yet created)
├── slurm_gemini_label.sh      # --max-files 158
├── slurm_embed.sh
├── slurm_merge.sh
├── logs/
└── PIPELINE_SUMMARY.md        # This file
```
