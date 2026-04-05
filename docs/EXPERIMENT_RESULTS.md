# AWR Model Size & Augmentation Experiment Results

## Experiment Overview

All models trained with AWR (Advantage Weighted Regression) on the same offline data:
- **Data**: 126 files from `final_trajectories/` (80% train split, ~2.1M samples per shard, 6 shards)
- **Training steps**: 100,000
- **Batch size**: 256
- **Learning rate**: 3e-4
- **AWR beta**: 10.0
- **Evaluation**: 10 live episodes per model in Craftax-Symbolic-v1 (max 10,000 steps/episode)

### Architecture

**Augmented** (`ActorCriticAug`): Dual-branch. obs(8268)->W and hidden(4096)->W projected separately, concatenated to 2W, then two W-width layers to output. Hidden states are 4096-dim Qwen3-8B layer-30 mean-pooled embeddings of Gemini 2.5 Flash future-state predictions, generated every 15 environment steps.

**Unaugmented** (`ActorCritic`): Single-branch. obs(8268)->W, then two more W-width layers to output. Same depth, no hidden state input.

Both use tanh activations and orthogonal initialization. Separate actor/critic heads.

## Results

| Model | Width | Augmented | Params | Avg Return (10 ep) | Std | Mean Achievements | Checkpoint |
|-------|-------|-----------|--------|--------------------:|----:|------------------:|------------|
| aug-w512 | 512 | Yes | 14,260,268 | 16.20 | 5.11 | 16.9 | `checkpoints/awr_imagination/final.pth` |
| **unaug-w512** | 512 | No | 9,540,652 | **17.70** | 2.69 | 18.2 | `checkpoints/awr_unaug_w512/final.pth` |
| aug-w1024 | 1024 | Yes | 31,666,220 | 14.50 | 4.72 | 15.2 | `checkpoints/awr_aug_w1024/final.pth` |
| **unaug-w1024** | 1024 | No | 21,178,412 | **19.10** | 1.90 | 19.2 | `checkpoints/awr_unaug_w1024/final.pth` |
| **aug-w2048** | 2048 | Yes | 75,915,308 | **18.90** | 2.14 | 19.2 | `checkpoints/awr_aug_w2048/final.pth` |
| unaug-w2048 | 2048 | No | 50,745,388 | 17.40 | 3.80 | 18.1 | `checkpoints/awr_unaug_w2048/final.pth` |

### Key Observations

1. **Unaugmented outperforms augmented at small/medium scale**: At width 512 and 1024, the unaugmented policy achieves higher returns than the augmented one. The augmented model's hidden state branch may be noisy or underfitting at these sizes.

2. **Augmentation helps at large scale**: At width 2048 (76M params), the augmented model (18.90) outperforms the unaugmented (17.40) by 1.5 points. The larger model has enough capacity to usefully integrate the LLM-derived hidden states.

3. **Best overall model**: unaug-w1024 (19.10) and aug-w2048 (18.90) are statistically similar given their std deviations.

4. **Stability**: unaug-w1024 has the lowest variance (std=1.90), while aug-w512 has the highest (std=5.11).

## Training Summary

All training completed 2026-03-28.

| Model | Training Time | Steps/sec | GPU |
|-------|--------------|-----------|-----|
| unaug-w512 | 11.7 min | ~143 | A100 80GB |
| unaug-w1024 | 13.7 min | ~122 | A100 80GB |
| unaug-w2048 | 19.4 min | ~86 | A100 80GB |
| aug-w512 | ~17 min (previous session) | ~100 | A100 80GB |
| aug-w1024 | 20.7 min | ~80 | A100 80GB |
| aug-w2048 | 29.5 min | ~57 | A100 80GB |

## Notes

- All checkpoints stored under `/data/group_data/rl/geney/checkpoints/`
- Eval results (JSON + videos) stored under `/data/group_data/rl/geney/eval_results/{aug,unaug}_w{512,1024,2048}/`
- wandb project: `craftax-offline-awr`, entity: `iris-sobolmark`
- Augmented eval uses ~660 Gemini API calls per episode (every 15 steps)
- Training code: `pipeline/train_awr.py` (supports `--no-augmentation`, `--layer-width`)
- Eval code: `pipeline/eval_online.py` (augmented), `pipeline/eval_unaugmented.py` (unaugmented)
