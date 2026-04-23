# Plan — online base → BC/AWR on Gemini-conditioned dataset

**Motivation.** The current stack trains an augmented policy from a small
PSF dataset via offline BC/AWR with Gemini-conditioned hidden inputs.
Observed ceiling at top-2M is ~18 return (A_full) to ~15 (C_grounded).
Every augmented policy we have is **stateless per-step** — a feedforward
MLP on `(current_obs, current_gemini_hidden) → action`, no trailing
trajectory. Meanwhile every Craftax scoreboard entry worth comparing
against — PPO-RNN, PPO-GTrXL, and the better model-based methods —
carries a recurrent/transformer state across steps. Efficient MBRL
(arXiv:2502.01591) and Simulus (arXiv:2502.11537) are included as
**inspiration** for what Craftax-competitive architectures look like;
they are not candidate base models here.

The hypothesis is that the BC/AWR optimization + small dataset + stateless
architecture jointly cap what the model can learn, and that a policy
initialized from a same-scale online RL run (which has matching
recurrence) and then fine-tuned on Gemini-conditioned data can (a) match
the online base on raw return and (b) still pick up Gemini-content-
following from the fine-tune step.

## Research question

**Q.** If we start from a non-augmented policy trained online at a
data/compute scale matched to our BC/AWR top-2M set, and fine-tune on
the Gemini-conditioned dataset, can we get a policy that is both (a)
at parity with its online base on raw return and (b) responsive to
Gemini content in the same way our BC/AWR policies are?

Anchor axes: **performance**, **steerability**, **robustness**,
**generalization**. (`feedback_research_goals_anchor.md`.)

## Baselines — picking a *comparable-scale* online base

1e8-step PPO-RNN (27.87 return) is **not** a fair base: it consumed 100×
the interactions we gave BC/AWR. A fair base should be matched to the
~2M offline transitions our augmented policies see.

Options, in rough priority order:

1. **PPO-RNN at 2-5M env steps** — the "small-data" PPO-RNN, which based
   on the training curves should sit close to our augmented policies'
   return (the user's recall: PPO-RNN at 1M env steps was "terrible").
   Cheap to produce (~15 min on L40S). Gives an RNN-carrying base with
   return near our own. **This is the right pilot baseline.**
2. **PPO-MLP stateless** at matched steps — feedforward PPO trained on
   the same 2-5M env steps, no recurrence. Matches our architecture
   exactly. Use this as a *second* base to isolate what the RNN state
   buys vs what the online interaction buys.
3. **PPO-RNN at scale** (1e8 at 27.87, or smaller e.g. 1e7 at some
   intermediate return). Kept as an upper reference but not the pilot
   base; the goal isn't to match scoreboard, it's to measure the
   online→BC transfer delta.
4. **PPO-GTrXL** — only if PPO-RNN pilot shows the online init pays off
   and we want to push the base higher. Not part of the pilot.

Note on statefulness: every scoreboard number we've cited (PPO-RNN,
GTrXL, MBRL approaches) uses trailing state. If we keep our augmented
policies stateless, we're handicapped by construction against those
reference numbers. The online→BC experiment is where that handicap
shows up — a stateful base *could* transfer its sequential reasoning
into the fine-tuned policy, but only if our architecture and eval loop
preserve RNN state.

## Architecture plan

### Stage 1 — pilot base policies

Produce two checkpoints:

- **PPO-RNN small**: 2e6 and 5e6 env steps. Saves to
  `/data/group_data/rl/geney/checkpoints/ppo_rnn_{2M,5M}_baseline/policies`.
  Quick to run (< 1h each on L40S).
- **PPO-MLP small**: same 2e6 and 5e6 steps, stateless backbone. If
  `online_rl/ppo.py` supports MLP config, use it; else write a minimal
  feedforward variant.

Report raw 50-ep return for each on fresh seeds. Expected: both sit
near our BC/AWR range (12-18), with RNN ≥ MLP at matched steps.

(The already-queued 1e8-save run — job 7429017 — stays useful as an
*upper* reference. Not the pilot base.)

### Stage 2 — RNN-aware eval_online

Before Stage 3, `eval_online.py` must be able to run a stateful policy.
Add an `--rnn-arch {none, gru, lstm}` flag plus a per-episode hidden-state
buffer that is carried across steps and reset on episode boundaries.
This is a small addition (~30 lines) — the eval loop already iterates
per-step.

Deliverable: a run-smoke test showing PPO-RNN-small checkpoint loaded,
rolled out for 50 eps, matches within noise the training-time
episode_return curve on the same seed distribution.

### Stage 3 — fine-tune on Gemini-conditioned dataset

From each pilot base (PPO-RNN-small, PPO-MLP-small):

**(A) Port backbone to PyTorch** (see "Porting" below), add a
`HiddenBranch` head that zero-inits its final projection so the initial
fine-tuned model = the base, and fine-tune with:

- **BC** on PSF-top-2M `(obs, hidden) → action`.
- **BC + AWR** using PSF advantages (already computed).
- **BC + KL-to-base**, **λ ∈ {0.01, 0.1, 1.0}** — user-confirmed sweep.
  Lower λ = faster adaptation, higher drift. Higher λ = stays close to
  base, less Gemini pickup.

**Incremental rollout.** Run {BC-only, BC+KL λ=0.1} first on PPO-RNN-2M
base → eval → monitor → only then run the λ sweep and the PPO-MLP arm.
"Do some and monitor them and then do others" — keep the sweep tight
until one cell clearly beats regular BC/AWR.

### Stage 4 — probe battery on fine-tuned policies

Incremental again. Start with a **small** probe set per fine-tuned
policy:

- 50-ep regular
- die_v2 (content-robustness)
- direction-CF step-0 (steerability)

Expand to the full battery (constant, random, adversarial_v2,
avoid_water_v2, avoid_animals_v2, multistep CF, HP perturbation,
held-out Δ_shuf) only if the small set shows an interesting delta vs
the plain BC/AWR policies.

## Porting ActorCriticRNN (Flax → PyTorch)

Realistic effort: **a few hours**, not a day. The inference graph is:

```
obs → Linear(obs_dim, 512) → tanh
      → GRUCell(512)
      → Linear(512, 512) → tanh
      → [ LinearHead(512, action_dim)   # actor logits
          LinearHead(512, 1) ]           # critic
```

Steps:
1. Read the Flax module and weight shapes from `online_rl/ppo_rnn.py`.
2. Mirror in PyTorch (`torch.nn.GRUCell`, `torch.nn.Linear`).
3. Weight transfer: Flax GRU weight layout (`update/reset/candidate`
   stacks) into PyTorch's fused `W_ir/W_iz/W_in + W_hr/W_hz/W_hn`. One
   one-off permutation; verify with 100 obs that action logits match
   Flax within 1e-4 L1.
4. Wrap in a `torch.nn.Module` that exposes `forward(obs, h) -> logits,
   value, h'` so `eval_online.py`'s new `--rnn-arch` path can call it.

## Concrete task list (revised, user-confirmed incremental plan)

- [x] PPO-RNN 1e8 save-enabled rerun queued (job 7429017) — kept as
      upper reference only.
- [ ] PPO-RNN 2M and 5M save-enabled runs (new, matched-scale pilot bases).
- [ ] PPO-MLP 2M and 5M save-enabled runs (stateless comparison).
- [ ] Port ActorCriticRNN + weight transfer to PyTorch. Unit test.
- [ ] Extend `eval_online.py` with `--rnn-arch` flag + per-step hidden
      state carry.
- [ ] Smoke-test: PPO-RNN-2M 50-ep eval from the PyTorch port.
- [ ] Fine-tune PPO-RNN-2M: BC-only, then BC+KL λ=0.1. **Monitor.**
- [ ] If interesting, run λ sweep {0.01, 0.1, 1.0} and PPO-MLP arm.
- [ ] Small probe set (regular / die_v2 / dir-CF step-0) on every
      fine-tuned policy.
- [ ] Expand to full battery only on promising cells.
- [ ] Journal.

## Sizing & cost

- 2M+5M PPO-{RNN,MLP} training: 4× ≤1h ≈ 3-4 GPU-hours total.
- Port + eval extension: a few hours human time; no compute.
- Fine-tune on PSF-top-2M: ≤ 4h per config. Pilot (2 configs) = ~8 GPU-h.
  Full λ×arch sweep (6 configs) = ~24 GPU-h.
- Small probe set on pilot: ~3 GPU-h per policy × 2 = ~6 GPU-h.
- Full probe battery (if triggered): ~15 GPU-h per policy.

Pilot-only end-to-end: ≤ 20 GPU-h. Fits in one overnight.

## Risks and caveats

- **Porting fidelity.** GRU weight layouts differ. Mitigation: compare
  JAX vs PyTorch logits on 100 obs; fail loudly if L1 > 1e-4.
- **RNN state handoff in eval.** `eval_online.py` is stateless per step
  today. The `--rnn-arch` addition must reset state on done/truncated.
- **PSF-top-2M data OODness for an online-trained policy.** The online
  base has never seen these obs distributions directly; BC on them may
  push it away from the trajectories that earned it return. KL-to-base
  is the lever; sweep λ.
- **Checkpoint format.** Orbax checkpoint ↔ Flax model class at load
  time. One `eval/load_ppo_rnn_ckpt.py` helper.
- **Gemini cost.** Reuses existing embedded PSF-top-2M shards. No new
  Gemini calls at training time; same eval-time cost as current stack.

## What I am NOT planning yet

- **PPO-GTrXL / MBRL / Simulus ports.** Only as upper references if the
  PPO-RNN-small pilot pays off and we want to push base return higher.
- **End-to-end joint BC + online PPO fine-tune.** Only if (BC + KL-to-
  base) fails to preserve raw return at every λ.

---

## User-confirmed decisions

- λ sweep: yes, confirmed.
- Incremental rollout: yes — run pilot fine-tunes (BC-only + one λ),
  monitor, then expand.
- Base-model papers: kept as **inspiration** in motivation; removed
  from base-model candidate list.
- Porting effort: revised down to a few hours.
- Stateless baseline acceptance: not yet decided — plan covers both
  (PPO-MLP stateless *and* PPO-RNN with RNN-aware eval) so we can
  measure the stateful delta directly.
