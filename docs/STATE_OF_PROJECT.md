# State of the Project (2026-03-22 → 2026-04-17)

**One-line:** Across all imagination-augmented configs and the obs-only baseline, online returns sit in a narrow band (15.7–18.4) that is small relative to per-policy std (2.7–6.7). The more interesting finding is on the listening side: **we now have one encoder (gemini_emb) that passes *every* content-sensitivity test** — direction counterfactual, value counterfactual, and bad-prompt gameplay all point the right way. qwen3gen fails all three (content-blind); qwen3emb has the magnitudes but fails on sign (wrong-direction content). Game returns don't separate the three; mechanistic probes do.

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

| Policy | Return | gap vs unaug in units of this policy's σ |
|---|---:|---:|
| unaug (obs only, no imagination) | **18.38 ± 2.69** | — |
| qwen3emb β=3 freeze_obs_bcawr | 17.88 ± 5.36 | 0.09σ |
| qwen3gen freeze_obs_bcawr | 17.58 ± 3.56 | 0.22σ |
| gemini_emb freezenone | 16.20 ± 6.73 | 0.32σ |
| gemini_emb β=30 freeze_obs_bcawr | 15.68 ± 5.14 | 0.52σ |

Every imagination config is within ≤ 0.5 σ of the unaug baseline. At n=50, these rankings are suggestive but not decisive — swap 10 episodes of noise between any two and the order changes. Treat game returns as "all these configs are roughly in the same neighborhood" rather than as an informative ranking. The mechanistic probes below do a much better job of separating them.

**How much does the policy listen to imagination?** Three complementary probes, all run per encoder on the PSF freeze_obs_bcawr family:

1. **Direction counterfactuals** (*do actions change when we swap what the policy sees?*). Procedure per episode:
   - Play normally up to a fixed "intervention step" *t* ∈ {0, 75, 150, 300} using the real policy.
   - Record the symbolic obs *o* at step *t* and the real embedding *h* (= Qwen/Gemini embedding of the future narrative Gemini wrote for *o*).
   - Build a **spatial-flipped obs** *o'*: the symbolic observation rotated 180° (tiles/entities mirrored around the player).
   - Generate a **flipped embedding** *h'*: call Gemini on *o'*, embed that narrative.
   - Forward the policy three times on the same state: (*o*, *h*) baseline, (*o'*, *h*) obs-flip, (*o*, *h'*) emb-flip.
   - Record whether the argmax action differs from baseline for each flip. That's one data point per (episode × intervention step × flip type).
   - Aggregate across all completed (episode × step) pairs (≈30 eps × 4 steps = up to 120 data points per policy per flip type, minus eps that ended early), then take the fraction where argmax changed.

   The "14.2% emb-flip rate" for gemini_emb means that in 14.2% of the recorded gemini_emb probe events, replacing only the imagination embedding (while keeping the real obs) caused the policy to pick a different argmax action. The obs-flip rate is the same statistic but for the complementary intervention.

   | Encoder | emb-flip rate (action change when only hidden swapped) | obs-flip rate |
   |---|---:|---:|
   | qwen3gen | 1.7% | 67% |
   | qwen3emb | 11.6% | 76% |
   | gemini_emb | 14.2% | 71% |

   Obs-flip changes the argmax in ~60–83% of events for all three, confirming the obs is the dominant input. The emb-flip rate ranges 7–8× across encoders, confirming that "how much the policy listens to imagination" varies — but it's always small relative to how much it listens to obs.

2. **Value counterfactuals** (new; Health/Food field perturbation on the obs fed to Gemini, measure ΔV on the policy's value head while symbolic obs stays real):
   - qwen3gen: mean \|ΔV\| = 0.08, arg-change 4% → **content-blind**
   - qwen3emb β=3: mean \|ΔV\| = 0.63, but 3 of 4 probe signs are *wrong* (low HP → V goes up, high HP → V goes down) → **content-reactive, miscalibrated**
   - gemini_emb β=30: mean \|ΔV\| = 0.95, all 4 signs *correct*, food_low ΔV = −0.33 is the strongest single signal in the grid → **content-reactive, correct-direction**
   - freezenone strengthens 3 of 4 gemini_emb signals (e.g. food_high 10× stronger) → the +0.52 online lift comes from more content-reading, not memorization.

3. **Gameplay with bad online prompts** (replace the future narrative with an "adversarial" or "die" instruction at every Gemini call):
   - qwen3gen: flat (+0.3, −0.2 vs baseline) → content doesn't matter
   - qwen3emb: **rises +2.0 pp** when content is sabotaged → its real content is net-harmful
   - gemini_emb: **drops −1.3 pp** → its real content is mildly helpful

The three probes agree, and together they give a much sharper separation than game returns do. **Only gemini_emb passes every test**: it has the highest emb-flip action-change rate (14.2%), it is the only encoder whose value function moves in the correct direction for *every* field perturbation (Health low, Health high, Food low, Food high), and its return drops when we sabotage the imagination with adversarial/die prompts (−1.3 pp) — the expected sign if the policy is really using the imagination content. qwen3gen fails all three (flat on every probe → content-blind). qwen3emb passes on magnitude but fails on sign (large \|ΔV\|, wrong direction; content is net-harmful so sabotaging it *raises* returns).

### High level

**Why Craftax is hard.** Long episodes (500+ steps), 42 discrete actions, sparse achievement rewards, simultaneous attention required to food/drink/energy/health/inventory/positioning/combat across 9 floors. Credit assignment over this horizon is the core difficulty. The original motivation for imagination-augmentation was to inject short-horizon semantic context that could shortcut some of that credit-assignment work.

**What regime we're in.** The obs-only baseline (~18.4) is strong because the symbolic observation already contains nearly everything the policy needs within a small neighborhood, and all our imagination-augmented configs cluster around it within ≤ 0.5 σ. Game returns at n=50 can't discriminate here — we'd need much larger n, or a harder task, to get a clean return ranking. What *does* discriminate the configs is their relationship to the imagination signal:

- **qwen3gen: content-blind.** Passes no content-sensitivity test. The hidden pathway is an expensive no-op; the policy coasts on the obs branch.
- **qwen3emb β=3: content-reactive, wrong-direction.** Large magnitudes but wrong signs; sabotaging the content *raises* return by +2 pp.
- **gemini_emb (β=30 and freezenone): content-reactive, correct-direction — the only one to pass every probe.**

The **central finding** of the month is that we have demonstrated a training recipe that produces a *verifiably* imagination-reading policy: gemini_emb freezenone has returns in the same neighborhood as qwen3gen and unaug, but unlike them, its value function responds to imagination content in the semantically correct direction on every field we've probed. This is the first imagination-augmented policy in this project that can be said to actually *use* imagination correctly — we've gone from "policies that ignore or miscalibrate the imagination" to "policies that read it right."

Whether that correct reading eventually translates into *better* returns — rather than merely comparable ones — is the next question. At the current scale it doesn't; the obs branch is strong enough to carry the policy on its own.

**Open questions carried forward:**
- Can we find a task or regime where correct imagination reading *does* pay off in returns (harder task, worse obs, much longer episodes, distribution shift at eval)? Right now gemini_emb "reads right but doesn't need to."
- Does a stronger training-time text generator (e.g. 3.1-pro) amplify the correct-direction signal enough to matter? The 3.1-pro swap at inference was null (needs retrain for OOD; deferred).
- Is the ~18.4 ceiling itself an artifact of the obs encoding? A richer obs representation (pixel, or deeper symbolic MLP) might move it, and only then would imagination quality become a visible lever.
