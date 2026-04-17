# State of the Project (2026-03-22 → 2026-04-17)

**One-line:** We have mapped the regime for LLM-imagination-augmented RL on Craftax. The imagination pathway can be made content-sensitive and correct-directional, but the current training recipe cannot convert that signal into online returns above the obs-only baseline (~18.4). Bottleneck is not the imagination signal — it's how the policy integrates it.

## Details

### Training

| Recipe | Data | What happens | Result |
|---|---|---|---|
| **AWR** on imagination-augmented shards | Filtered PPO rollouts, PSF Gemini labels, Qwen3-8B embeddings | Policy's value function shows ~0 accuracy gap between real / zero / shuffled embeddings → **ignores imagination entirely** | ~18 online, matches obs-only |
| **AWR + BC** on human (golden) trajectories, oracle-labelled | Top-N human eps, Gemini sees future 15 steps | BC forces the policy to **listen** to imagination, but memorizes the oracle-specific embedding distribution → online collapse | ~5 online |
| **AWR + BC, PSF-labelled**, freeze obs_branch from pretrained | PSF golden + PSF shards, matched embedder | First non-collapsing BC+AWR recipe; policy reads some content and survives online | qwen3gen 17.58, qwen3emb 16.60, gemini_emb 14.96 |
| + **β tuning** on qwen3emb / gemini_emb | same | Encoder-dependent sharpness preferences (qwen3emb wants soft β=3, gemini_emb wants sharp β=30) | qwen3emb β=3 = **17.88**, gemini_emb β=30 = 15.68 |
| + **freezenone** (let obs-branch keep adapting) for gemini_emb | same | Obs-branch improves alongside hidden; only config to beat the gemini_emb β=30 baseline | **16.20** |

### Evaluation

**Game returns (50-ep online).** The top policies, plus the unaug baseline:

| Policy | Return |
|---|---:|
| unaug (obs only, no imagination) | **18.38 ± 2.69** |
| qwen3emb β=3 freeze_obs_bcawr | 17.88 ± 5.36 |
| qwen3gen freeze_obs_bcawr | 17.58 ± 3.56 |
| gemini_emb freezenone | 16.20 ± 6.73 |
| gemini_emb β=30 freeze_obs_bcawr | 15.68 ± 5.14 |

No imagination-augmented config meaningfully exceeds the obs-only ceiling. Best imagination policy (qwen3emb β=3) is within noise of unaug.

**How much does the policy listen to imagination?** Three complementary probes, all run per encoder on the PSF freeze_obs_bcawr family:

1. **Direction counterfactuals** (multistep, flip obs vs. flip embedding, measure argmax action change):
   - qwen3gen: 1.7% emb-flip vs 66% obs-flip → ignores imagination
   - qwen3emb: 11.6% emb-flip → reads it
   - gemini_emb: 14.2% emb-flip → reads it most
   - All three are obs-dominated (≥60% obs-flip).

2. **Value counterfactuals** (new; Health/Food field perturbation on the obs fed to Gemini, measure ΔV on the policy's value head while symbolic obs stays real):
   - qwen3gen: mean \|ΔV\| = 0.08, arg-change 4% → **content-blind**
   - qwen3emb β=3: mean \|ΔV\| = 0.63, but 3 of 4 probe signs are *wrong* (low HP → V goes up, high HP → V goes down) → **content-reactive, miscalibrated**
   - gemini_emb β=30: mean \|ΔV\| = 0.95, all 4 signs *correct*, food_low ΔV = −0.33 is the strongest single signal in the grid → **content-reactive, correct-direction**
   - freezenone strengthens 3 of 4 gemini_emb signals (e.g. food_high 10× stronger) → the +0.52 online lift comes from more content-reading, not memorization.

3. **Gameplay with bad online prompts** (replace the future narrative with an "adversarial" or "die" instruction at every Gemini call):
   - qwen3gen: flat (+0.3, −0.2 vs baseline) → content doesn't matter
   - qwen3emb: **rises +2.0 pp** when content is sabotaged → its real content is net-harmful
   - gemini_emb: **drops −1.3 pp** → its real content is mildly helpful

The three probes agree on the regime each policy falls into. Notably: **the highest-returning policy (qwen3gen) is the one that reads the imagination the least**. More content-sensitivity has *not* translated into higher return so far.

### High level

**Why Craftax is hard.** Long episodes (500+ steps), 42 discrete actions, sparse achievement rewards, simultaneous attention required to food/drink/energy/health/inventory/positioning/combat across 9 floors. Credit assignment over this horizon is the core difficulty. The original motivation for imagination-augmentation was to inject short-horizon semantic context that could shortcut some of that credit-assignment work.

**What regime we're in.** The obs-only baseline (~18.4) is strong because the symbolic observation already contains nearly everything the policy needs within a small neighborhood. Our best imagination-augmented policies match this ceiling but do not exceed it. The policy either:
- ignores imagination (qwen3gen) and coasts on the obs branch,
- listens to imagination but the content is miscalibrated (qwen3emb), or
- listens and content is correct-signed but underpowered (gemini_emb).

The **central finding** of the month is that correct imagination reading ≠ better policy: gemini_emb has the cleanest content-to-value mapping but underperforms content-blind qwen3gen by ~2 pp online. That means the bottleneck is obs-branch integration, not the imagination signal itself. The freezenone result (allowing the obs-branch to keep training alongside a content-aware hidden) is our first evidence that improving the *integration* is what unlocks modest gains.

**Open questions carried forward:**
- Is there a recipe where the obs-branch and a content-correct hidden both train to strength, without the hidden pathway being ignored or net-negative?
- Does a stronger training-time text generator (e.g. 3.1-pro) change the picture? The 3.1-pro swap at inference was null (needs retrain for OOD; deferred).
- Is the ~18.4 ceiling itself an artifact of the obs encoding? A richer obs representation (pixel, or deeper symbolic MLP) might move it.
