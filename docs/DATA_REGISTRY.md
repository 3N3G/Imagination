# Data Registry

Canonical record of data directories, their provenance, and which checkpoints
consume them. Updated whenever a new labelling/embedding/subset variant is
created. Stored on `/data/group_data/rl/geney/` (not in this repo).

**Naming convention (v2 and later):**
```
<stage>_psf_v2_[cadenceN_][<prompt-variant>_][<encoder>]
```
- `stage` ∈ {gemini_labels, embeddings, final_trajectories, predict_only_*}
- `cadenceN` = Gemini call cadence (only set when ≠ default; currently always 5)
- `prompt-variant` ∈ {<none>=concise, predonly (extract Prediction suffix at embed time), nocoord, think, grounded}
- `encoder` ∈ {gemini_emb, qwen3gen, qwen3emb}

---

## Current (v2 cadence=5) lineage

### Base pipeline (concise prompt, 3-flash, cadence=5)
| Stage | Path | Notes |
|-------|------|-------|
| Labels | `gemini_labels_psf_v2_cadence5_3flash/` | 158 files, concise prompt, gemini-3-flash-preview |
| Embeds | `embeddings_psf_v2_cadence5_gemini_emb/` | Full response → gemini-embedding-001 (3072-dim) |
| Merged | `final_trajectories_psf_v2_cadence5_gemini_emb/` | 158 files, ~20 GB fp16, baseline training data |

### Predonly variant (same labels, Prediction-suffix-only at embed)
| Stage | Path | Notes |
|-------|------|-------|
| Embeds | `embeddings_psf_v2_cadence5_predonly_gemini_emb/` | Calls `extract_prediction_suffix()` before embed |
| Merged | `final_trajectories_psf_v2_cadence5_predonly_gemini_emb/` | 158 files, 19.39 GB |

### Oracle set (for val / oracle_loss)
| Stage | Path | Notes |
|-------|------|-------|
| Labels | `oracle_pipeline/predict_only_gemini_labels_v2_cadence5_3flash/` | Oracle sees future, predict-only prompt |
| Embeds (full) | `oracle_pipeline/predict_only_embeddings_v2_cadence5_gemini_emb/` | For base lineage |
| Embeds (predonly) | `oracle_pipeline/predict_only_embeddings_v2_cadence5_predonly_gemini_emb/` | For predonly lineage |
| Merged (full) | `oracle_pipeline/predict_only_final_v2_cadence5_gemini_emb/trajectories_000000.npz` | Single file, used as `--oracle-data` + `--val-data` |
| Merged (predonly) | `oracle_pipeline/predict_only_final_v2_cadence5_predonly_gemini_emb/trajectories_000000.npz` | Predonly-lineage oracle |

### PSF size-ablation subsets (built from base lineage)
Ranked by `return_to_go[ep_start]` desc, walked until target rows.
Drops `text_generated`; keeps `hidden_state` fp16.
| Path | Rows (target) | Episodes | Return cutoff |
|------|---------------|----------|---------------|
| `psf_size_ablation_subsets/final_trajectories_psf_v2_cadence5_gemini_emb_top1M/` | 1M | ~3k | top-1% episodes |
| `psf_size_ablation_subsets/final_trajectories_psf_v2_cadence5_gemini_emb_top2M/` | 2M | ~6k | top-4% |
| `psf_size_ablation_subsets/final_trajectories_psf_v2_cadence5_gemini_emb_top4M/` | 4M | ~12k | top-8% |
| `psf_size_ablation_subsets/final_trajectories_psf_v2_cadence5_gemini_emb_top8M/` | 8M | ~24k | top-16% |
| (full: `final_trajectories_psf_v2_cadence5_gemini_emb/`, ~12.7M effective, all episodes) | 12.7M | ~40k | — |
| Manifest | `psf_size_ablation_subsets/manifest.json` | | |

---

## Checkpoints for v2 cadence=5 lineage

### Base-lineage AWR
| Checkpoint dir | Data | Recipe |
|----------------|------|--------|
| `awr_psf_v2_cadence5_gemini_emb/` | full | AWR pretrain, 100k steps, β=10 |
| `v2_cadence5_gemini_emb_freezenone/` | full | freezenone BC+AWR, 50k steps, β=30 ofrac=0.05 ow=0.5 |

### Base-lineage size ablation
| Checkpoint dir | Data | Recipe |
|----------------|------|--------|
| `psf_size_ablation_cadence5/awr_full/` | full (12.7M) | AWR pretrain, 100k steps |
| `psf_size_ablation_cadence5/awr_top8M/` | top8M | same |
| `psf_size_ablation_cadence5/awr_top4M/` | top4M | same |
| `psf_size_ablation_cadence5/awr_top2M/` | top2M | same |
| `psf_size_ablation_cadence5/awr_top1M/` | top1M | same |

### Predonly lineage
| Checkpoint dir | Data | Recipe |
|----------------|------|--------|
| `psf_v2_cadence5_predonly/awr/` | predonly | AWR pretrain, 100k steps |
| `psf_v2_cadence5_predonly/freezenone/` | predonly | freezenone BC+AWR, 50k steps |

---

## Eval results dirs

| Path | Source |
|------|--------|
| `eval_results/psf_size_ablation_cadence5/{full,top8M,top4M,top2M,top1M}_50ep/` | AWR-only, 50-ep live |
| `eval_results/psf_v2_cadence5_predonly/freezenone_50ep/` | Track A freezenone, 50-ep live (predonly inference) |

---

## Legacy dirs (pre-v2, kept for archaeology — do not use for new runs)

- `final_trajectories/` — initial pipeline, small
- `final_trajectories_psf_gemini_emb/` — PSF v1 (before obs_to_text equipment fix)
- `final_trajectories_psf_qwen3emb/` — Qwen3 embedder variant
- `final_trajectories_psf_qwen3gen/` — Qwen3 generation variant
- `final_trajectories_psf_v2_gemini_emb/` — v2 without cadence=5 fix (every-step calls)
- Old checkpoints: anything under `awr_psf_*` / `psf_freeze_*` not starting with `v2_cadence5` or `psf_v2_cadence5` or `psf_size_ablation_cadence5`.

---

## SCALING_C lineage (v3, PPO-RNN-derived data)

Goal: train a higher-return C-style policy on trajectories sourced from
PPO-RNN 1e8 (training-time mean episode return ~27.87) instead of the
PPO-symbolic-derived top-2M PSF v2 data (mean ~21.21). All data here is
on `/data/user_data/geney/scaling_c_data/` because the group_data quota
is exhausted (deletions reclaimed by other rl_data users).

| Stage | Path | Notes |
|-------|------|-------|
| **Raw trajectories** (Phase 1) | `/data/group_data/rl/geney/raw_trajectories/ppo_rnn_1e8_save_traj_continuous/` | Job 7507785 (DONE 2026-04-26 14:07). PPO-RNN 1e8 with `--save_traj_every 1 --save_traj_start_step 1000`. 525 batches, 17 GB compressed, ~34 M transitions. Replaces the buggy `ppo_rnn_1e8_save_traj/` (every-10 cadence; deleted) which had unrecoverable episode-attribution. |
| **Bitpacked filtered** (Phase 2a) | `/data/user_data/geney/scaling_c_data/filtered_trajectories_pporn_1e8/` | Job 7519127 (RUNNING). `filter_and_repack` with `--min_return 15 --num_envs 1024`. |
| **Top-4M-rows subset** (Phase 2b) | `/data/user_data/geney/scaling_c_data/filtered_trajectories_psf_v3_pporn_1e8_top4M/` | Job 7519127 stage 2. `build_bitpacked_top_subset` with `--target-rows 4000000`. Picks highest-RTG episodes until ≥ 4 M rows. |
| **Gemini grounded labels** (Phase 3) | `/data/user_data/geney/scaling_c_data/gemini_labels_psf_v3_cadence5_grounded_3flash/` | NOT YET BUILT. Same recipe as `C_grounded_2M`: grounded prompt + cadence=5 + 3-flash + future_offset=5. Cost ~$400. |
| **Gemini embeddings** (Phase 4a) | `/data/user_data/geney/scaling_c_data/embeddings_psf_v3_cadence5_grounded_predonly_gemini_emb/` | NOT YET BUILT. Predonly extraction + gemini-embedding-001 (3072-dim). |
| **Final merged** (Phase 4b) | `/data/user_data/geney/scaling_c_data/final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M/` | NOT YET BUILT. Training-ready data for AWR pretrain + BC+AWR finetune. |

Submit-driver scripts: `slurm/jobs/scaling_c_phase{2,3,4,5}*.sh`. Phase 2
auto-fires (job 7519127); Phases 3-5 manual after their dependencies land.

**Notable changes:**
- `pipeline/filter_and_repack.py` gained `--num_envs` flag (default 128
  for legacy 2D NPZs) so it can handle PPO-RNN's 1024-env layout.
- `online_rl/ppo_rnn.py` gained `--save_traj/--save_traj_every/--save_traj_start_step/--traj_save_path` to support trajectory dumping.

## Planned / in-flight (not yet built)

- **Thinking prompt + top2M relabel**: will live at
  `gemini_labels_psf_v2_cadence5_think_3flash/` → `embeddings_psf_v2_cadence5_think_gemini_emb/` → `final_trajectories_psf_v2_cadence5_think_gemini_emb_top2M/`
  (labels only top2M episodes to cut cost).
- **Grounded-as-if-predicting + top2M relabel**: `gemini_labels_psf_v2_cadence5_grounded_3flash/` → `..._grounded_gemini_emb/` → top2M merged.
  Oracle future given to Gemini, output phrased as forward "Prediction: ...".

**Rule (per `feedback_labelling_isolation.md`):** every new prompt/cadence/model
change writes to its OWN output dir; never reuse a prior label dir + resume.
