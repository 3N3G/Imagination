# Imagination-Augmented RL for Craftax — Research Log

**Project:** Can LLM-generated "imagination" of future states improve offline RL in Craftax?

**Core idea:** Embed Gemini's text predictions of what will happen next into a dense vector (Qwen3-8B layer 30), and condition the offline RL policy on this "hidden state" alongside raw observations.

---

## Phase 1: Pipeline & Infrastructure (completed 2026-03-23)

Built the full pipeline: PPO data collection → filter top episodes → Gemini labeling → Qwen embedding → merge into training data. See [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md) for architecture details.

**Key numbers:**
- 12,207 raw trajectory files from PPO (200M steps, 128 envs)
- Top 250 episodes filtered by return-to-go
- Gemini 2.5 Flash for labeling, Qwen3-8B layer-30 for 4096-dim embeddings
- Labels generated every 15 env steps

---

## Phase 2: Model Size Ablation (completed 2026-03-28)

Trained 6 AWR models: augmented vs unaugmented × {w512, w1024, w2048}. 10-episode online evals. See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

**Result:** Mixed — augmentation helped at w2048 (18.90 vs 17.40) but hurt at w512 and w1024. Unaugmented w1024 was the overall best (19.10). High variance in augmented results suggested the embedding branch might be noisy.

**Open question:** Does the model actually *use* the embeddings, or is the hidden state branch just adding parameters/noise?

---

## Phase 3: Embedding Ablation & Validation (completed 2026-03-30)

Ran a comprehensive set of experiments to understand whether and how models use their embeddings. See [EXPERIMENT_RESULTS_2.md](EXPERIMENT_RESULTS_2.md).

### 3a. Validation ablation (real/zero/shuffled)

| Model family | Uses embeddings? | Real−Zero gap |
|-------------|-----------------|---------------|
| Oracle (ground-truth future labels) | **Yes** | +5pp |
| PSF (predict-state-only labels) | No | ±0.2pp noise |
| Unaugmented (no embeddings) | N/A | N/A |

**Insight:** The architecture *can* learn to use embeddings when they carry real signal (oracle). But predict-state-only labels are too noisy — models trained on them learn to completely ignore the hidden state branch.

### 3b. Cross-distribution test

Fed PSF embeddings to the oracle-trained model: **36.32% accuracy — worse than zero (36.96%)**. This proves PSF embeddings are out-of-distribution for the oracle model and actively harmful.

### 3c. Online embedding ablation (100 episodes × 5 conditions)

Tested Gemini, constant string, random gibberish, adversarial futures, and death-seeking futures on oracle w512, oracle w2048, PSF w2048, and unaugmented w2048.

**Result:** Embedding content has zero effect on online returns. All conditions overlap within noise. "Die" embeddings don't make the agent die. Unaugmented (17.66) beats all augmented models.

**Root cause:** Distribution mismatch. Oracle model learned to use oracle embeddings (which encode ground truth), but at eval time every embedding source is OOD. The hidden branch processes OOD noise and injects confusion, making augmented models strictly worse than unaugmented ones online.

### 3d. Weighted BC+AWR with oracle demonstrations

Mixed training: 25% oracle trajectories upweighted 5× + 75% normal AWR.

**Result: Collapsed.** 5.05 return (vs 17.66 unaugmented). Model memorized oracle actions with near-zero entropy (0.06), losing ability to handle novel states. Classic BC distribution shift failure.

---

## Key Takeaways (as of 2026-03-31)

1. **The architecture works** — oracle training proves it can learn a 5pp dependency on hidden state embeddings
2. **Predict-state labels don't carry enough signal** — PSF models ignore their embeddings entirely
3. **Distribution mismatch kills gains at eval time** — even the oracle model can't use any available eval-time embedding source
4. **The augmentation branch is net negative at inference** — it adds OOD noise; unaugmented models win
5. **BC with oracle demos doesn't work** — too narrow, causes catastrophic overfitting

## Open Directions

- **Better eval-time embeddings:** Learned world model? Better Gemini prompts? Chain-of-thought strategy reasoning?
- **Bridge the distribution gap:** Train on embeddings that are closer to what's available at eval time
- **Abandon augmentation:** Focus on obs-only architectures and invest the compute budget elsewhere
- **Online augmented RL:** Train with live Gemini calls during PPO (job 6816399 was attempted but paused)
