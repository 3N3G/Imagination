# SCALING_C — train a higher-return C-style policy on PPO-RNN-derived data

## Motivation

`C_grounded_2M` was trained on the **top 2M rows** of an *older, smaller* PPO
training run. That source PPO had mean episode return ≈ 9.5 across 17,462
episodes; the top-2M-rows subset spans returns 11.09 → 14.42. The trained
augmented policy lands at ≈ 14.7 — i.e. it **matches the top of its training
data** but the training data itself caps the policy's quality.

Meanwhile our PPO-RNN baseline at 1e8 timesteps reaches **27.87** mean return.
If we generate trajectories with PPO-RNN at scale, we should be able to feed
the offline-RL pipeline data with episode returns >> 14, keep the same
{AWR pretrain → BC+AWR finetune} recipe, and end up with a steerable C-style
policy whose floor is much higher. Steerability should remain because the
mechanism (Gemini predict→embed→hidden branch reads content) is identical.

The user's question: can we save trajectories from PPO-RNN training in the
**exact same schema** as the existing PSF data, then re-run the existing
`scan_streaming → filter_top → gemini_label → embed → merge → AWR + BC+AWR`
pipeline against this new data?

**Answer: yes, with a small code change to `ppo_rnn.py`.** The existing
schema is what `online_rl/ppo.py` (the symbolic PPO) writes via
`save_trajectory_batch`. `ppo_rnn.py` did not have that path; this doc
records the change and the recipe.

## Code change (already landed)

`online_rl/ppo_rnn.py`:

1. `Transition` now carries `next_obs` and `next_done` (post-step
   observation + post-step done flag), in addition to the existing fields
   used by GAE/PPO. Existing fields (`done = last_done`, `value`, `reward`,
   `log_prob`, `obs`, `info`) are unchanged; the PPO update still uses them
   exactly as before.
2. The rollout step populates the new fields from `obsv` and `done` (the
   post-step values returned by `env.step`).
3. After the wandb logging callback, a new `save_callback` is wired in via
   `jax.experimental.io_callback` (gated on Python-level
   `if config["SAVE_TRAJ"]`, so no `vmap-of-cond` pitfall). The callback
   writes `trajectories_batch_NNNNNN.npz` with the same keys
   `pipeline/scan_streaming.py` expects: `obs, next_obs, action, reward,
   done, log_prob`.
4. New CLI args:
   - `--save_traj` — enable saving (default off).
   - `--save_traj_every <N>` — save one batch every N update steps
     (default 10). One update step = `NUM_STEPS × NUM_ENVS = 16 384`
     transitions.
   - `--save_traj_start_step <N>` — skip the first N update steps before
     saving (default 0). Use this to drop the early-training low-return
     trajectories.
   - `--traj_save_path <dir>` — output dir.

## Disk-size budget for PPO-RNN at 1e8 timesteps

- One update step writes `NUM_STEPS × NUM_ENVS ≈ 16 384` transitions.
- Total updates = `1e8 / 16 384 ≈ 6 103`.
- Each transition's `obs + next_obs` is `2 × 8 268 × 4 bytes = 66.1 KB`
  raw; `np.savez_compressed` typically gets ~5× → ~13 KB on disk.

| `--save_traj_every` | Updates saved | Transitions saved | Disk ≈ |
|---|---|---|---|
| 10 | 610 | 10 M | ~130 GB |
| 20 | 305 | 5 M | ~65 GB |
| 50 | 122 | 2 M | ~26 GB |

For the `top-4M` target, `--save_traj_every 20 --save_traj_start_step 1500`
gives ~5M transitions from the second half of training — plenty of overhead
to filter top-4M and keep only the highest-return episodes.

## Recipe to produce a `C_v2` (higher-return-data) policy

**Phase 1 — generate trajectories** (~24 h on L40S):

```bash
./slurm/jobs/ppo_rnn_save.sh 1e8 \
    --extra "--save_traj --save_traj_every 20 --save_traj_start_step 1500 \
             --traj_save_path /data/group_data/rl/geney/raw_trajectories/ppo_rnn_1e8_save"
```

(needs `--extra` plumbing in submit.sh, or just run `python -m
online_rl.ppo_rnn` with the new args directly — the slurm wrapper passes
extra args through `"$@"`.)

**Phase 2 — scan + filter** (~30 min on CPU):

```bash
PYTHONPATH=. python pipeline/scan_streaming.py \
    --input-dir /data/group_data/rl/geney/raw_trajectories/ppo_rnn_1e8_save \
    --output-dir /data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/raw_psf_v3_pporn_1e8

PYTHONPATH=. python pipeline/filter_and_repack.py \
    --input-dir <above raw> \
    --top-rows 4000000 \
    --output-dir /data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/raw_psf_v3_pporn_1e8_top4M
```

**Phase 3 — Gemini grounded label** (the expensive step, ~5–10 h API
time, big spend):

```bash
PYTHONPATH=. python pipeline/gemini_label.py \
    --input-dir <top4M raw> \
    --output-dir gemini_labels_psf_v3_cadence5_grounded_3flash \
    --prompt-template configs/training/templates/predict_state_only_prompt_concise_grounded.txt \
    --cadence 5
```

This is the budget item the user must approve. Order-of-magnitude:
~5 000 episodes × ~150 cadence-5 calls/ep ≈ 750 k Gemini calls. Cost
depends on rate-card, but hours rather than minutes and not free.

**Phase 4 — embed + merge** (~few hours on a GPU):

```bash
PYTHONPATH=. python pipeline/embed.py \
    --labels-dir <gemini_labels...> \
    --raw-trajectories-dir <top4M raw> \
    --output-dir embeddings_psf_v3_cadence5_grounded_predonly_gemini_emb \
    --backend gemini-embedding-001 \
    --extract-prediction-only

PYTHONPATH=. python pipeline/merge.py \
    --raw-dir <top4M raw> \
    --labels-dir <gemini_labels...> \
    --embeddings-dir <embeddings...> \
    --output-dir final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M
```

**Phase 5 — train AWR + BC+AWR** (same hyperparams as `C_grounded_2M`,
~6 h):

```bash
# AWR pretrain (100k steps, β=10)
PYTHONPATH=. python -m offline_rl.train_awr \
    --data-dir final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M \
    --total-steps 100000 --batch-size 256 --lr 3e-4 --awr-beta 10 \
    --hidden-dim 3072 --layer-width 512 \
    --oracle-data <existing oracle> --oracle-fraction 0.05 --oracle-loss-weight 0.0 \
    --save-dir /data/group_data/rl/geney/checkpoints/psf_v3_pporn_1e8_grounded_top4M/awr

# BC+AWR finetune (50k steps, β=30, freezenone)
PYTHONPATH=. python -m offline_rl.train_awr_weighted_v2 \
    --data-dir <same> \
    --total-steps 50000 --batch-size 256 --lr 1e-4 --awr-beta 30 \
    --pretrained-checkpoint <awr/final.pth> \
    --freeze-mode none --oracle-fraction 0.05 --oracle-loss-weight 0.5 --entropy-coeff 0.01 \
    --save-dir /data/group_data/rl/geney/checkpoints/psf_v3_pporn_1e8_grounded_top4M/freezenone
```

**Phase 6 — eval** (same prompt suite as `C_grounded_2M`, ~3 h):

Re-run the specificity matrix's 21 cells to confirm steerability survives
the data swap, plus the score-max + survive_long iteration on the new
policy to see how much higher the ceiling moves.

## Expected outcome

**Optimistic case (steerability + return both lift):**
- Baseline return on the new policy: ~22–26 (between full-data PPO 9.5
  source and PPO-RNN 27.87 source ceiling, reduced by AWR/BC fidelity loss
  similar to the current 14.7 / 14.4 ratio).
- Specificity matrix WIN rate similar to current 12/21 (steerability is a
  property of the embedding pathway, not the trajectory quality).
- `target_descend_v2` boosts return further; survive-extending may now be
  possible because the policy actually has the underlying skill.

**Pessimistic case:**
- The hidden branch reads PPO-RNN-style hidden activations differently
  from PPO-symbolic-style → fidelity drops → return drops.
- Or: AWR-on-RNN-data fails because the RNN trajectories presuppose
  recurrent state the BC student doesn't have. Mitigation: PPO-symbolic
  1e8 (already running, 7492202) gives a backup higher-return data source.

## Status

- [x] Code change to `ppo_rnn.py` to support `--save_traj` (this doc).
- [ ] User approval to run the Gemini labelling spend.
- [ ] Resubmit PPO-RNN 1e8 with `--save_traj` (the in-flight 7492203
      job runs WITHOUT it; we'd need a fresh job).
- [ ] Phases 2–6.
