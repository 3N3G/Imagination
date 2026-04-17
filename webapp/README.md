# Craftax prompt iterator

Streamlit webapp for iterating on action-selection and future-prediction prompts
with Gemini on a hardcoded sample of 10 × 6-step Craftax trajectories.

## Setup (local, not cluster)

```bash
cd webapp/
pip install -r requirements.txt
export GEMINI_API_KEY=...        # or paste into the sidebar
streamlit run app.py
```

The browser opens at `http://localhost:8501`.

## Data

- `data/trajectories.json` — 10 trajectories (6 from PSF shards, 4 from PSF
  golden). Each has 6 consecutive env steps.
- `data/images/traj_{i}_step_{j}.png` — local tile view rendered from the
  symbolic obs. Covers the player's 11×9 egocentric window; player is the red
  dot in the center, with a white tick indicating facing direction. Not the
  full Craftax texture rendering — reconstructing that from obs alone is hard
  and not worth it for prompt iteration.

## How the webapp is laid out

- **Sidebar:** trajectory + step pickers, Gemini API key, temperature, max
  tokens, model selection (any subset of 2.5-flash / 3.1-flash-lite / 3.1-flash
  / 3.1-pro preview).
- **Top:** tile image + action-taken + reward + expandable obs text for the
  currently-selected step, and a strip of all 6 steps in the trajectory.
- **"Future prediction" tab:** editable prompt that Gemini receives on step 0.
  Placeholder `{current_state_filtered}` is replaced by the filtered obs text.
  Hitting "Run" fires the selected models in parallel and shows each response
  side-by-side, followed by the ground-truth next 5 steps for comparison.
- **"Action selection" tab:** same structure but with a one-shot action prompt.
  The predicted action is extracted from `ACTION: <NAME>` and compared against
  the action actually taken in the trajectory.

## Regenerating the data (on cluster)

```bash
cd ~/Imagination
/data/user_data/geney/.conda/envs/craftax_fast_llm/bin/python tools/sample_webapp_trajectories.py
```

This rewrites `webapp/data/trajectories.json` and all PNGs. Use a different
seed by editing `sample_trajectories(seed=...)` in the script.
