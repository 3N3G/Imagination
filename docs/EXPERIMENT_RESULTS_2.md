# Imagination-Augmented RL: Embedding Ablation & Validation Results

**Date:** 2026-03-30

---

## TL;DR

The oracle-trained model genuinely learns to use ground-truth future embeddings
(+5pp validation accuracy). But at eval time, no available embedding source
(predict-state Gemini, constant text, random gibberish, or adversarial prompts)
is close enough to the oracle training distribution to help. A cross-distribution
test confirms that predict-state embeddings are **worse than zero** for the oracle
model. PSF-trained models never learn to use their embeddings at all. The
augmentation branch is a net negative at inference time.

---

## 1. Validation: Does the Model Use Embeddings?

Three conditions on held-out data (same obs/actions, different hidden states):
- **Real**: actual training-distribution embeddings, correctly paired
- **Zero**: zero vector (no embedding information)
- **Shuffled**: training-distribution embeddings, randomly mis-paired

### Oracle models (trained on ground-truth future summaries)

| Model | Real Acc | Zero Acc | Shuffled Acc | Real−Zero |
|-------|----------|----------|--------------|-----------|
| Oracle w512 | **41.93%** | 36.96% | 32.65% | **+4.97pp** |
| Oracle w2048 | **40.91%** | 35.70% | 31.44% | **+5.21pp** |

**Yes, the oracle model uses embeddings.** Real > Zero > Shuffled, with large gaps.
Wrong oracle embeddings (shuffled) are worse than no embeddings at all — the model
actively relies on embedding content and is misled by incorrect signal.

### PSF models (trained on predicted/guessed future summaries)

| Model | Real Acc | Zero Acc | Shuffled Acc | Real−Zero |
|-------|----------|----------|--------------|-----------|
| PSF w512 | 38.57% | 38.77% | 38.51% | −0.20pp |
| PSF w1024 | 38.22% | 38.41% | 38.12% | −0.19pp |
| PSF w2048 | 37.64% | 37.71% | 37.51% | −0.07pp |

**No. PSF models completely ignore their embeddings.** All three conditions are within
noise (< 0.3pp). The predict-state-only Gemini labels don't carry enough signal for
the model to learn a dependency.

### Unaugmented models (obs-only, no hidden state branch)

| Model | Accuracy |
|-------|----------|
| Unaug w512 | 38.80% |
| Unaug w1024 | 38.48% |
| Unaug w2048 | 38.54% |

PSF models ≈ unaugmented models. The extra hidden-state branch in PSF models
adds parameters but contributes nothing.

---

## 2. Cross-Distribution Test: What Happens When You Feed the Wrong Embeddings?

Key experiment: take the oracle w512 model and evaluate it with PSF embeddings
(same obs/actions from the same trajectories, but hidden states come from
predict-state-only labels instead of oracle labels).

| Condition | NLL | Accuracy | What it tells us |
|-----------|-----|----------|------------------|
| Real oracle | 2.08 | **41.93%** | Correct training-distribution signal |
| Zero | 2.53 | 36.96% | No signal; obs-only fallback |
| **PSF cross** | **2.72** | **36.32%** | **OOD embeddings → worse than zero** |
| Shuffled oracle | 2.93 | 32.65% | In-distribution but wrong-paired; most harmful |

**PSF embeddings actively hurt the oracle model** (36.32% < 36.96%). The model
tries to interpret them as oracle embeddings, finds different semantic content,
and gets confused. This is the key to understanding the online eval results.

---

## 3. Online Evaluation: Embedding Ablation (100 episodes each)

At inference time, what happens when we replace the hidden state with different
embedding sources?

### Oracle augmented w512

| Embedding source | Return | ± Std | Achievements |
|-----------------|--------|-------|-------------|
| Gemini (predict-state) | 15.36 | 5.71 | 16.0 |
| Constant string | 14.97 | 5.30 | 15.5 |
| Random gibberish | 15.01 | 5.23 | 15.7 |
| Adversarial (bad futures) | 16.20 | 3.96 | 16.9 |
| Die (death-seeking) | 16.65 | 4.00 | 17.2 |

### Oracle augmented w2048

| Embedding source | Return | ± Std | Achievements |
|-----------------|--------|-------|-------------|
| Gemini (predict-state) | 14.65 | 4.54 | 15.2 |
| Constant string | 14.19 | 5.06 | 14.9 |
| Random gibberish | 13.02 | 5.15 | 13.8 |
| Adversarial (bad futures) | 14.58 | 4.57 | 15.3 |
| Die (death-seeking) | 14.70 | 4.84 | 15.4 |

### PSF augmented w2048

| Embedding source | Return | ± Std | Achievements |
|-----------------|--------|-------|-------------|
| Gemini (predict-state) | 17.27 | 3.56 | 17.9 |
| Constant string | 16.99 | 4.02 | 17.6 |
| Random gibberish | 16.42 | 4.17 | 17.1 |
| Adversarial (bad futures) | 17.60 | 3.70 | 18.1 |
| Die (death-seeking) | 17.18 | 3.67 | 17.8 |

### Unaugmented w2048 (no embeddings at all)

| Return | ± Std | Achievements |
|--------|-------|-------------|
| **17.66** | 3.89 | **18.2** |

### What this shows

1. **Embedding content does not meaningfully affect online returns.** For every
   model, all five conditions (Gemini, constant, random, adversarial, die) produce
   statistically overlapping results. "Die" embeddings don't make the agent die.
   Adversarial embeddings don't make it play worse. The model treats all eval-time
   embeddings as uninformative.

2. **Unaugmented consistently wins.** Unaug w2048 (17.66) beats every oracle
   augmented condition (13–17 range) and matches or beats PSF augmented conditions.

3. **PSF w2048 ≈ unaugmented w2048.** Both get ~17–18 returns across conditions,
   confirming the validation finding that PSF models ignore their embeddings.

---

## 4. Explaining the Paradox

**Q: The oracle model uses embeddings (validation proves it with a 5pp gap). So
why doesn't embedding quality matter during online play?**

**A: Distribution mismatch.** The oracle model was trained on embeddings of
ground-truth future summaries. At eval time, every embedding source is
out-of-distribution (OOD):

```
Training:  "You are writing a narrative summary of what ACTUALLY HAPPENS"
           → Oracle sees the real next 15 steps, then summarizes them
           → Qwen embeds that summary → embedding encodes ground truth

Eval:      "You are FORECASTING a plausible future"
           → Gemini guesses what might happen (no access to real future)
           → Qwen embeds that guess → embedding encodes speculation
```

The cross-distribution test (§2) confirms this: PSF embeddings fed to the oracle
model are **worse than zero** (36.32% vs 36.96%). All eval-time embeddings —
Gemini predictions, constant strings, random text, adversarial prompts — are
equally OOD from the oracle training distribution. They all cause the same mild
confusion.

**The ordering, from most to least helpful:**

```
Real oracle embeddings     41.93%  ← only available at training time
Zero (no embedding)        36.96%  ← what an unaugmented model effectively sees
PSF/eval-time embeddings   36.32%  ← what the model actually gets at eval time
Shuffled oracle embeddings 32.65%  ← wrong in-distribution signal (worst case)
```

The hidden state branch was trained to extract signal from oracle embeddings.
When fed OOD inputs, it produces noisy activations that corrupt the obs-only
signal. This is why augmented models underperform unaugmented ones: the
unaugmented model doesn't have a noise-injecting hidden branch.

---

## 5. Weighted BC+AWR with Oracle Demonstrations

Trained an augmented w512 model with 25% of each batch from oracle (golden)
trajectories, upweighted 5× in the loss. Oracle trajectories were labeled
through the full PSF pipeline (Gemini predict-state-only → Qwen embed).

**Training:** 100K steps, 19.7 min. Oracle BC loss → 0.03 (near-perfect
imitation), oracle entropy → 0.06 (very confident on oracle actions).

**Online eval (100 episodes):**

| Metric | Value |
|--------|-------|
| Return | **5.05 ± 3.52** |
| Achievements | 5.8 ± 3.3 |
| Min/Max return | 0.10 / 18.10 |

**The model collapsed.** Despite near-perfect imitation of oracle actions in
training (BC loss 0.03), online performance is catastrophic — 5.05 return vs
17.66 for unaugmented. The model became overconfident on oracle actions
(entropy 0.06) and lost the ability to generalize. This is classic behavioral
cloning failure: the oracle trajectories are too narrow a distribution, and
the 5× upweighting caused the model to overfit to oracle-specific behavior
that doesn't transfer to novel states encountered during online play.

---

## 6. Complete Results Table

### Validation accuracy (held-out offline data)

| Model | Best Acc | Embedding signal? |
|-------|----------|-------------------|
| Oracle aug w512 | 41.93% (real) | **Yes** (+4.97pp over zero) |
| Oracle aug w2048 | 40.91% (real) | **Yes** (+5.21pp over zero) |
| PSF aug w512 | 38.57% | No (±0.2pp noise) |
| PSF aug w1024 | 38.22% | No (±0.2pp noise) |
| PSF aug w2048 | 37.64% | No (±0.1pp noise) |
| Unaug w512 | 38.80% | N/A |
| Unaug w1024 | 38.48% | N/A |
| Unaug w2048 | 38.54% | N/A |

### Online returns (100-episode evals)

| Model | Gemini | Constant | Random | Adversarial | Die | Unaug baseline |
|-------|--------|----------|--------|-------------|-----|----------------|
| Oracle w512 | 15.36 | 14.97 | 15.01 | 16.20 | 16.65 | — |
| Oracle w2048 | 14.65 | 14.19 | 13.02 | 14.58 | 14.70 | — |
| PSF w2048 | 17.27 | 16.99 | 16.42 | 17.60 | 17.18 | — |
| **Weighted BC+AWR w512** | **5.05** | — | — | — | — | — |
| Unaug w2048 | — | — | — | — | — | **17.66** |

### Prior 10-episode evals (for reference)

| Model | Return | ± Std |
|-------|--------|-------|
| Unaug w1024 | **19.10** | 1.90 |
| PSF aug w2048 | 18.90 | 1.40 |
| Oracle aug w2048 | 18.90 | 2.14 |
| Unaug w512 | 17.70 | 2.69 |
| Unaug w2048 | 17.40 | 3.80 |
| PSF aug w1024 | 16.90 | 4.87 |
| PSF aug w512 | 16.70 | 4.80 |
| Oracle aug w512 | 16.20 | 5.11 |
| Oracle aug w1024 | 14.50 | 4.72 |

---

## 7. Conclusions

1. **Oracle embeddings work — at training time.** The oracle model learns a real
   5pp dependency on ground-truth future embeddings. This proves the architecture
   *can* use hidden state information when it's informative.

2. **Predict-state-only embeddings are uninformative.** PSF models trained on
   guessed-future embeddings learn to completely ignore them (0pp real/zero gap).
   The guessed futures don't carry enough signal about optimal actions.

3. **Distribution mismatch kills the oracle advantage.** At eval time, even the
   oracle model can't use predict-state embeddings — they're OOD and slightly
   worse than zero. All eval-time embedding sources (good or bad) produce the
   same result.

4. **The augmentation branch is a net negative at inference time.** It adds
   parameters that process OOD noise and inject confusion. Unaugmented models
   consistently outperform or match augmented ones online.

5. **Weighted BC with oracle demonstrations collapses.** Upweighting oracle
   trajectories 5× in a mixed BC+AWR loss causes catastrophic overfitting
   (5.05 return vs 17.66 unaugmented). The model memorizes oracle-specific
   actions with near-zero entropy (0.06), losing the ability to handle novel
   states. Oracle demonstrations are too narrow to serve as a useful prior.

6. **The path forward requires closing the distribution gap.** Either:
   - Make eval-time embeddings match oracle quality (learned world model?)
   - Train on predict-state embeddings that actually carry signal (better prompts?
     different LLM? chain-of-thought reasoning about strategy?)
   - Remove the hidden state branch entirely and focus on obs-only architectures
