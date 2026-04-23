# Plan — strong online base → BC/AWR on Gemini-conditioned dataset

**Motivation.** The current stack trains an augmented policy from a small
PSF (predict-state-filtered) dataset via offline BC/AWR with Gemini-conditioned
hidden inputs. Observed ceiling at top-2M is ~18 return (A_full) to ~15
(C_grounded). Meanwhile a plain PPO-RNN at 1e8 steps reaches 27.87 raw
return and 1B-scale PPO-GTrXL hits 18.3% of max (~41 return). The
augmentation work has been mechanistic — content-reading probes improve
with fidelity, but absolute performance doesn't.

The hypothesis is that the BC/AWR optimization + small dataset caps what
the model can learn, and that a policy initialized from strong online RL
and then *fine-tuned* on Gemini-conditioned data can (a) inherit the
online-RL performance floor and (b) still pick up Gemini-content-following
from the fine-tune step.

## Research question

**Q.** If we start from a strong non-augmented policy (online RL) and
fine-tune on the Gemini-conditioned dataset, can we get a policy that is
both (a) competitive with the online base on raw return and (b)
responsive to Gemini content in the same way our BC/AWR policies are?

Anchor axes: **performance**, **steerability**, **robustness**,
**generalization**. (See `feedback_research_goals_anchor.md`.)

## Architecture plan

### Stage 1 — strong base policy (cheap online RL)

Target: raw return ≥ 25 on Craftax-Symbolic-v1 in ≤ 12h of 1×L40S.

Candidates, in priority order:

1. **PPO-RNN 1e8** (`online_rl/ppo_rnn.py`). We just completed one run at
   27.87 raw (job 7397006, no saved policy). Checkpoint-saving rerun is
   now queued as **job 7429017**. Pre-baked in repo, known to reach scale.
2. **PPO-GTrXL** (GRU-replaced-with-transformer). Needs implementation
   port (`online_rl/ppo_gtrxl.py` doesn't exist). Craftax scoreboard has
   this at 18.3% / ~41 return at 1B. Not needed for stage 1 if PPO-RNN
   suffices.
3. **Efficient MBRL** ([arXiv:2502.01591](https://arxiv.org/abs/2502.01591))
   — sample-efficient world-model-based. Good if we want a base that
   matches published ≥35 return for fewer env steps. Higher integration
   cost (needs a dream-rollout infrastructure).
4. **Simulus** ([arXiv:2502.11537](https://arxiv.org/abs/2502.11537)) —
   Craftax-specific architecture. Also needs port. Defer.

**Plan.** Use PPO-RNN as stage 1 for the pilot. The save-enabled rerun
(7429017) gives us a checkpoint at `/data/group_data/rl/geney/checkpoints/
ppo_rnn_1e8_baseline/policies`.

**Deliverable:** one JAX/Flax orbax checkpoint + a JAX eval script that
reports 50-ep raw return on fresh seeds, matched-protocol to our
`eval_online.py` sampling.

### Stage 2 — fine-tune on Gemini-conditioned dataset

Two sub-variants, both start from Stage 1's weights:

**(A) PyTorch port + BC head on Gemini hidden.** Port the Flax ActorCritic
with GRU/LSTM state into PyTorch (just the inference graph — no backward
pass on JAX), initialize the obs-only trunk with the PPO-RNN weights,
**add** the hidden-branch MLP (dim-4096 Gemini hidden) used by
`ActorCriticAug`, and BC on PSF-top-2M `(obs, hidden) → action` pairs.
Freeze options: {none, obs trunk, partial}. Loss options: {BC, BC + AWR,
BC + PPO online fine-tune, BC + KL to base}.

Quickest: **BC + KL-to-base** — minimizes BC loss on Gemini-conditioned
data subject to the policy not drifting far from the PPO-RNN prior.
Prevents catastrophic forgetting of the strong base behavior.

**(B) JAX continued-training with hidden branch.** Modify
`online_rl/ppo_rnn.py` (or write `online_rl/ppo_rnn_aug.py`) to accept a
hidden_state input channel, initialize hidden-branch weights to zeros
(so initial policy = PPO-RNN base), and run a short online+offline mixed
pass: replay-buffer BC on PSF data *and* on-policy PPO rollouts with
Gemini hidden plugged in during rollout. More complex but avoids the
port step.

**Plan.** Go with (A). PyTorch BC+AWR is where our existing imagination
code lives, so we can reuse the training loop, eval framework, probe
suite, and all six-axes measurements. The port is limited to the
ActorCriticRNN inference graph; we keep the PyTorch model as the trained
policy.

### Stage 3 — eval on the full probe battery

Once Stage 2 gives us a fine-tuned policy:

- 50-ep raw return (regular, constant, random, die_v2, adversarial_v2,
  avoid_water_v2, avoid_animals_v2).
- Direction-CF step-0 and multistep.
- HP/Food perturbation (check food_low ΔV sign).
- Held-out Δ_shuf NLL on the track's own val data.

Expected outcomes to differentiate:

| Outcome | Interpretation |
|---|---|
| Raw ≈ 25, die_v2 ≈ 20 | Strong base + weak content-reading — obs-only is doing the work; Gemini hidden is ignored. |
| Raw ≈ 25, die_v2 ≈ 10 | Strong base + *strong* content-reading — best case. The policy follows Gemini content into failure when prompted to, and otherwise plays near the online baseline. Means online init + BC on conditioned data can combine both axes. |
| Raw ≈ 18, die_v2 ≈ 10 | Strong content-reading but forgot the base — BC overwrote the online prior. KL-to-base was too weak. |
| Raw ≈ 25, die_v2 ≈ 25 | No content-reading at all — BC didn't integrate Gemini. Architecture or loss issue. |

## Concrete task list

1. (In flight) PPO-RNN 1e8 rerun with `--save_policy` (job 7429017).
2. (NEW) Port `online_rl/ppo_rnn.py::ActorCriticRNN` inference graph to
   PyTorch. Single GRU cell + MLP head. Write a weight-transfer function
   `load_jax_into_torch(jax_params) -> torch_state_dict`.
3. (NEW) Add a `HiddenBranch` head module (reuse `ActorCriticAug`'s
   hidden branch) on top of the ported backbone: `logits = head_obs(obs
   features) + head_hidden(gemini_hidden)` (additive — zero-init on hidden
   side so initial behavior matches PPO-RNN).
4. (NEW) Training loop variants, all in `offline_rl/train_online_to_bc.py`:
   - BC on PSF-top-2M `(obs, hidden) → action`.
   - BC + AWR using PSF advantage estimates (already computed in the
     offline stack).
   - BC + KL-to-base λ sweep {0.01, 0.1, 1.0}.
5. Eval on full probe battery (Stage 3).
6. Journal the results with the six-axes table.

## Sizing & cost

- Stage 1: PPO-RNN 1e8, 10h L40S — already in flight, free.
- Stage 2: fine-tune = BC on top-2M dataset, should be ≤ 4h on A100/L40S.
- Stage 3: 50 eps × (7 probe types) × (PPO-RNN-init policy) ≈ 2-3h per
  probe × 7 = 14-20 GPU-hours. Comparable to the v2 die/adv array
  (7412177) we already ran overnight — fits in one 18h window.

Total: ≤ 40 GPU-hours for a well-powered pilot answer. Acceptable.

## Risks and caveats

- **JAX→PyTorch port fidelity.** GRU weight layouts differ (JAX uses
  stacked W_z/W_r/W_h, PyTorch uses fused W_ir/W_iz/W_in + W_hr/W_hz/W_hn).
  Unit-test the port by comparing action distributions on held-out obs
  between JAX and PyTorch implementations. Target: KL < 1e-4 per step on
  100 obs.
- **RNN state management.** `eval_online.py` is stateless per step.
  RNN-based policies need per-step hidden state. Either: (a) keep RNN
  state in the eval loop (already done for ppo_rnn during training); (b)
  squash RNN state into the hidden embedding pathway. Easier: (a), add
  RNN-state support to `eval_online.py`.
- **BC on PSF-top-2M overwrites base.** The PSF data is ~half noise and
  ~half fragment of "random exploration"; BC on it could push a strong
  base back toward the weaker BC-level policy. KL-to-base is the guard,
  and we should sweep λ.
- **Checkpoint format.** Orbax checkpoints are not trivially
  self-contained; we'll need the Flax model class available at load time
  to restore. Wrap the load in `eval/load_ppo_rnn_ckpt.py` once.
- **Gemini cost.** Nothing new on the training-data side (reuse existing
  PSF-top-2M embedded shards). Gemini costs only at eval-time, same as
  existing stack.

## What I am NOT planning yet

- Simulus / efficient MBRL / PPO-GTrXL ports. We pick these up only if
  PPO-RNN-init pilot shows the online init pays off and we want to push
  the base higher.
- End-to-end joint BC + online PPO fine-tune. Too much moving. Only
  consider if the simpler (BC + KL-to-base) variant fails to preserve
  raw return.

---

## Open for user decision before I start coding

- **GO / NO-GO on the port.** Porting ActorCriticRNN from Flax to
  PyTorch is ~1 day of careful work. Alternative is JAX-native
  offline_rl pipeline, but our probe infrastructure is PyTorch.
- **λ sweep scope.** Do we want the sweep in Stage 2, or just pick the
  middle value and iterate?
- **Which probes at Stage 3.** The v2 probe battery is ~7 items; if you
  want a quick pilot, maybe just {regular, die_v2, direction-CF-step0}
  first and expand only if the pilot is interesting.
