# Path 1 — C's train/eval distribution mismatch

**Status**: deferred per user request 2026-04-24 ("save for a future
discussion, find all options not super satisfying"). This doc captures
the diagnosis, the five mitigation options I considered, the trade-offs
of each, and the open questions.

## The mismatch

Track C (`psf_v2_cadence5_grounded_predonly_top2M`) was trained on
the V6 grounded prompt, where Gemini's input includes:

1. The current obs.
2. The actual obs at `t+5` (the future ground-truth from the saved
   trajectory) embedded as a "future:" block.

Gemini's prediction at training time was therefore a *near-oracle*
description of what the player would do — it could see the answer.
The Qwen3 / gemini-embedding-001 vector encoded this oracle-quality
description. The actor-critic policy learned to read those
oracle-quality embeddings.

At eval time we drop the future block (we don't have a real future when
the policy is rolling out fresh). Gemini gets only the current obs and
must guess. The resulting embedding is structurally similar (same
prompt template skeleton, same Qwen3 backbone) but the *content
distribution* shifts: predictions are now genuine forecasts rather than
recalls. The policy was never trained on these.

## Direct evidence the mismatch is load-bearing

| Symptom | Number | Interpretation |
|---|---|---|
| Regular-eval return | C=14.66 vs A_full=18.98 | C is the lowest of the three tracks despite having the highest-quality training labels. |
| Held-out Δ(shuf−real) NLL | C=+0.195 vs A=+0.026 | C's policy reads embedding content 7× as much as A's. |
| HP/food perturbation ΔV | food_low ΔV=+0.098 (10× A and WRONG SIGN) | C learned "low food" embedding context as positive value because in training, low-food states always paired with surviving-future obs blocks. |
| Synthetic embedding direction at α=+2 along d_die | Δret=−5.13 (matches direct-prompt die_v2 at −4.90) | Embedding-space content axis IS the policy's reading axis. |
| `v2_long_tail` patch prompt | Δret=+2.14, wake_up 0.52→0.90 (+38pp) | A clearer prompt at inference partially closes the deficit. |

C's high content-sensitivity and its low return are coupled phenomena —
both follow from "C learned to trust embedding content" + "the eval
embeddings have lower content quality than training embeddings".

## Five mitigation options (with reasoning)

### Option A — Inference-time prompt format alignment (quick smoke check)

**What**: at eval time, prepend an empty or placeholder "future:" block
to the regular concise prompt, so the eval prompt has the same
*structural* shape as the grounded training prompt.

**Mechanism hypothesis**: even with placeholder content, the prompt
structure may push Gemini's generation closer to the training-output
distribution simply because the `predict_state_only_prompt_concise.txt`
template differs from `predict_state_only_prompt_concise_grounded.txt`
in how the obs block is laid out — adding the future-shaped scaffolding
might recover a fraction of the embedding distribution.

**Cost**: ~1h work + 1 cell of 30-ep eval (~$1.50, ~3h walltime).

**Why I'm not satisfied with this**: the key mismatch is *content*
quality (Gemini doesn't actually have future info), not template shape.
The placeholder content can't fool Gemini into generating oracle-quality
predictions. Most likely outcome: embedding cos-sim to grounded-train-
distribution rises a few hundredths but the actual predictive content
stays the same as regular concise. Worth running as a 1-cell smoke
check before committing to anything bigger, but not a real fix.

### Option B — Mixed-prompt re-labelling and fine-tuning

**What**: Take the existing 2M PSF training states. Re-label a subset
(say 100k–500k states, ~5–25%) using the regular concise prompt instead
of the grounded prompt. Mix that into the training data
(grounded-only:regular-mix at e.g. 70:30). Fine-tune C's encoder on
the mixed dataset.

**Mechanism hypothesis**: by training on both grounded and regular
embeddings the policy learns to handle the eval distribution natively.
It should retain content sensitivity (the grounded portion teaches it
to read embeddings) and gain robustness (the regular portion teaches it
that "no future info" is a normal embedding flavor too).

**Cost**: 100k Gemini calls ≈ $30; 1–2h training; ~1h eval. Total
maybe $35 + 4h.

**Why I'm not satisfied**: the mixing ratio is a free hyperparameter
with no principled answer. We'd need to sweep a few (e.g. 90/10, 70/30,
50/50, 30/70, 0/100) to find the sweet spot, and 0/100 is just
re-deriving track A. The extreme ratio that maximizes return might
disagree with the one that maximizes content sensitivity, and the
"sweet spot" depends on which one we care about. There's also a risk
that mixing dilutes the grounded-quality signal so much that we lose
the steerability win along with the train/eval gap.

### Option C — Self-play futures at inference (the most "scientific" close)

**What**: at eval time, when Gemini needs to predict from a state s_t,
roll out the env forward 5 steps using the *current policy itself* (or
a shadow copy) to produce a fake "future obs" block. Feed that to
Gemini in the grounded prompt template. The result is: Gemini sees the
actual current state + an estimate of what the policy is about to do.
The embedding is now structurally and content-wise aligned with
training.

**Mechanism hypothesis**: this is the closest possible match to the
training distribution that we can construct online. The grounded
training futures came from saved trajectories generated by an earlier
policy; the self-play futures come from the current policy. There's a
slight closed-loop concern (policy reading its own forecast back) but
the embedding-level distance to training should be much smaller than
either option A or current vanilla eval.

**Cost**: each Gemini call now needs 5 extra env steps + a forward pass
of the current policy. Wall-time per episode roughly 2–3× current.
Code change is moderate (~50 lines in eval_online.py). One eval cell
at n=30 is feasible at ~$3 / ~6h walltime.

**Why I'm not satisfied**: it's a one-track-pony. The self-play future
will be biased toward what the policy is going to do anyway — Gemini
becomes a slightly-amplified mirror of the policy's current preferences
rather than an independent advisor. We'd lose part of what makes the
imagination useful (introducing new behaviors). It's also a pretty
heavy implementation for what may turn out to be a small effect.

### Option D — Translator head (architectural, expensive)

**What**: Train a small MLP that takes a regular-concise embedding as
input and produces an embedding closer to the grounded-distribution
manifold. Train it on paired (regular_embed, grounded_embed) data —
which we'd need to collect by running both prompts on the same set of
states.

**Mechanism hypothesis**: a learned linear (or shallow MLP) translation
between the two embedding spaces should be enough to bridge the gap if
the gap is mostly an affine shift + scale.

**Cost**: ~10k Gemini calls for paired data ($10), ~1h training a
small head, ~1h eval. Plus architectural complexity at inference (an
extra forward pass).

**Why I'm not satisfied**: this is a workaround, not a fix, and it
adds a moving part that obscures the underlying behavior. If it works,
we've shown the gap is approximately linear; if it doesn't, we've spent
the cost without diagnostic value. Better to spend the same effort on
B (which directly addresses the gap by retraining).

### Option E — Accept and frame the mismatch as the mechanism, don't fix

**What**: do nothing. Document that C's high content-sensitivity is
caused BY the train/eval distribution mismatch, not despite it. The
mismatch *makes* C steerable; closing the mismatch may close
steerability too.

**Mechanism hypothesis**: the steerability we observe on C is real but
its root cause is the policy's over-reliance on embedding content,
which in turn is caused by training on oracle-quality embeddings. If
we close the train/eval gap (any of A–D), the policy may stop reading
embeddings as carefully and thus become *less steerable*.

**Cost**: zero.

**Why I'm not satisfied**: this gives up on the central pitch of the
project — that better LLMs + online RL would beat current SOTA via
steerability. If C's steerability requires a permanent train/eval
distribution gap to maintain, it's not a recipe for better policies in
general.

## Cross-option trade-off summary

| Option | Cost | Likelihood return improves | Likelihood steerability preserved | Diagnostic value |
|---|---|---|---|---|
| A. Placeholder-future eval | tiny | low | high (no training change) | low |
| B. Mixed re-label + fine-tune | medium | high | moderate (depends on ratio) | high |
| C. Self-play futures | medium | high | unclear (closed-loop bias) | moderate |
| D. Translator head | medium | unclear | high (no policy change) | low (workaround) |
| E. Don't fix, frame as mechanism | zero | zero | high (status quo) | high (forces honest writeup) |

## Caveat: prompt-quality vs policy-fidelity confound on the wrong-sign result

The "C's HP/food sign is WRONG (food_low ΔV=+0.098)" finding is being
used here as evidence that the train/eval mismatch corrupts C's value
head (because the grounded-train pairing of low-food states with
surviving-future blocks taught the policy that "low food + this
embedding = good"). That story may be only half-right.

The other half: the wrong sign could partially be a **prompting-logic
issue**, not just a representation issue. If at training time Gemini's
grounded prediction for a low-food state often said something like
"the player will eat the cow at (1,2) and recover" — i.e., used the
oracle future to give a clean "no need to worry" prediction — then the
embedding for low-food states encoded that resolution. At eval time
without the future block, Gemini's regular concise prediction for the
same low-food state might say "the player should seek food" (correctly
worried) but the embedding shifted away from the trained "resolution"
embedding, so the policy reads it incorrectly.

To distinguish:
- Read 5–10 wandb runs' `gemini_log.jsonl` files for low-food states
  paired with the C policy. Compare:
  - What did the grounded-prompt prediction look like when the policy
    was trained?
  - What does the regular-prompt prediction look like at eval?
- If the eval prediction is *correctly* describing the worry but the
  policy's V is wrong, the issue is at the policy/embedding level.
- If the eval prediction itself fails to mention food / panics
  inappropriately, the wrong sign is a **Gemini prompt-quality**
  problem and would also affect any other policy that reads such
  embeddings.

This matters for which mitigation makes sense:
- If the issue is prompt quality at eval time: Option B (mixed re-label
  + fine-tune) is overkill. A targeted prompt revision to the regular
  concise prompt — making it ALWAYS mention intrinsics and risk —
  could close the wrong-sign gap without retraining.
- If the issue is the trained policy: Option B is necessary; prompt
  revisions can't fix it.

Action item when this is revisited: **manually inspect 5–10 low-food
states' gemini outputs at train-time vs eval-time** before committing
to any of A–E.

## Open questions before doing any of this

1. Is C's steerability conditional on the mismatch? If we ran B at any
   mixing ratio and steerability dropped to A's level, that would tell
   us the mismatch IS the mechanism — and that "fixing" it loses the
   property we cared about. The test for this is a B variant at
   say 70/30 and check the synthetic-die-direction sensitivity. If
   C-with-70/30-fix shows die-direction Δ −2 (vs C's −5.13), the
   mismatch was load-bearing on steerability. If it stays at −5, the
   mismatch is independent of steerability.
2. Does the "wrong sign on food_low ΔV" trace specifically to the
   future-block content being correlated with high-survival outcomes?
   A targeted intervention: re-label only the low-food states and see
   if the wrong sign goes away.
3. If we close the mismatch and steerability stays, can we then *also*
   apply the patch-by-prompt mechanism (`v2_long_tail`) on top to
   compound? That would be the dream: closed mismatch + clearer prompt
   = best of both.

## Recommendation when this is revisited

If the user wants the cheapest information per dollar: run Option A
(placeholder eval) FIRST as a smoke check. It's the only experiment
that runs in <4 hours and ~$2. Use the result to decide between
Option B (if smoke check shows promise) or Option E (if smoke check is
null and we accept the mismatch).

Skip Option D unless we explicitly want to study the translation
geometry — it's a workaround with low diagnostic value.

Option C is the most scientifically interesting but also the heaviest
to implement; defer until A and B have ruled themselves in or out.
