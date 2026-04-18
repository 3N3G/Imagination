"""Prompt-iteration webapp for Craftax imagination + action selection.

Run locally (from repo root):
    pip install -r webapp/requirements.txt
    export GEMINI_API_KEY=...
    streamlit run webapp/app.py

Data + images are committed to webapp/data/ — no cluster access needed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

import streamlit as st

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))           # so `from llm.gameplay import ...` works
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"

# Source of truth for both prompts (and the algorithm they share).
from llm.gameplay import FUTURE_PREDICT_PROMPT, ACTION_SELECT_PROMPT

MODELS = [
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]
# Models that require thinking mode (cannot set thinkingBudget=0). For these
# we leave thinking on by default and increase max_output_tokens, since
# thinking tokens count toward the output budget and were causing mid-sentence
# truncation at 2048.
THINKING_REQUIRED = {"gemini-3.1-pro-preview"}

# Defaults for the editable prompt textareas (single source of truth in
# llm/gameplay.py — bring the algorithm changes there and webapp picks them up).
DEFAULT_FUTURE_PREDICT_PROMPT = FUTURE_PREDICT_PROMPT
DEFAULT_ACTION_SELECT_PROMPT = ACTION_SELECT_PROMPT


# ---------------------------------------------------------------------------
# Gemini HTTP call (with MAX_TOKENS auto-retry for thinking-mode models like
# 3.1-pro, where thinking tokens silently consume the output budget and cause
# mid-sentence truncation at 2048).
# ---------------------------------------------------------------------------
def _single_call(prompt: str, model: str, api_key: str,
                 temperature: float, max_output_tokens: int):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    gen_config = {"maxOutputTokens": max_output_tokens, "temperature": temperature}
    # Disable thinking for speed where the model supports budget=0.
    # 2.5-flash/pro allow it; 3.x flash-lite/flash allow it; 3.1-pro REQUIRES
    # thinking mode (errors with INVALID_ARGUMENT on budget=0).
    if model.startswith("gemini-2.5") or "flash" in model:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }).encode("utf-8")

    t0 = time.perf_counter()
    req = urlrequest.Request(url, data=body,
                             headers={"Content-Type": "application/json"},
                             method="POST")
    try:
        with urlrequest.urlopen(req, timeout=120.0) as resp:
            raw = resp.read().decode("utf-8")
    except urlerror.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        return {"text": "", "error": f"HTTP {e.code}: {err}",
                "latency_s": time.perf_counter() - t0,
                "finish_reason": "ERROR",
                "max_output_tokens": max_output_tokens}

    parsed = json.loads(raw)
    cands = parsed.get("candidates", [])
    text = ""
    finish = "UNKNOWN"
    if cands:
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        finish = cands[0].get("finishReason", "UNKNOWN")
    usage = parsed.get("usageMetadata", {})
    return {
        "text": text,
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
        "thoughts_tokens": usage.get("thoughtsTokenCount", 0),
        "latency_s": time.perf_counter() - t0,
        "error": None,
        "finish_reason": finish,
        "max_output_tokens": max_output_tokens,
    }


# Cap on the auto-retry budget so a runaway thinking model can't burn cost.
MAX_TOKENS_AUTO_CAP = 16384


def call_gemini(prompt: str, model: str, api_key: str,
                temperature: float = 0.2, max_output_tokens: int = 2048):
    """Single Gemini call with up to two auto-retries on MAX_TOKENS finish.

    For thinking-required models (3.1-pro), thinking tokens count toward
    `maxOutputTokens`. If the response is cut mid-sentence with
    `finishReason: MAX_TOKENS`, we retry with the budget doubled (and again
    if needed), up to MAX_TOKENS_AUTO_CAP. The final `max_output_tokens`
    actually used is reported in the result so the UI can show it.
    """
    # Bump starting budget for thinking-required models so the first try
    # already has room.
    budget = max_output_tokens
    if model in THINKING_REQUIRED:
        budget = max(budget, 4096)

    attempts = []
    for _ in range(3):
        result = _single_call(prompt, model, api_key, temperature, budget)
        attempts.append({
            "max_output_tokens": budget,
            "finish_reason": result.get("finish_reason"),
            "completion_tokens": result.get("completion_tokens", 0),
            "thoughts_tokens": result.get("thoughts_tokens", 0),
        })
        if result.get("error"):
            result["attempts"] = attempts
            return result
        if result.get("finish_reason") != "MAX_TOKENS":
            result["attempts"] = attempts
            return result
        if budget >= MAX_TOKENS_AUTO_CAP:
            break
        budget = min(budget * 2, MAX_TOKENS_AUTO_CAP)

    result["attempts"] = attempts
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_trajectories():
    with (DATA_DIR / "trajectories.json").open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Craftax prompt iterator", layout="wide")

trajs = load_trajectories()

st.sidebar.markdown("### Trajectory picker")
traj_id = st.sidebar.selectbox(
    "Trajectory",
    options=list(range(len(trajs))),
    format_func=lambda i: f"{i}: {trajs[i]['source']} / {trajs[i]['source_file']} @ {trajs[i]['source_start_idx']}",
)
traj = trajs[traj_id]
step = st.sidebar.slider("Step in trajectory (0 = prompt source)", 0, 5, 0)
current_step = traj["steps"][step]

api_key = st.sidebar.text_input(
    "GEMINI_API_KEY",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password",
    help="Falls back to $GEMINI_API_KEY env var.",
)

temperature = st.sidebar.slider("Gemini temperature", 0.0, 1.0, 0.2, 0.05)
max_tokens = st.sidebar.number_input(
    "Max output tokens (initial)", 256, MAX_TOKENS_AUTO_CAP, 4096, 256,
    help=(
        "Starting budget per call. If a thinking-required model "
        "(e.g. 3.1-pro) returns finish=MAX_TOKENS, the call is auto-retried "
        f"with the budget doubled, up to {MAX_TOKENS_AUTO_CAP} tokens."
    ),
)
model_choices = st.sidebar.multiselect(
    "Models to run", MODELS, default=["gemini-2.5-flash", "gemini-3.1-pro-preview"]
)

st.title("Craftax prompt iterator")
st.caption(
    f"Trajectory {traj_id} · source **{traj['source']}** ({traj['source_file']}) · "
    f"start idx {traj['source_start_idx']} · viewing step {step} of 5. "
    "Gemini is called on step 0 only; steps 1–5 are the ground-truth future."
)

# -- Top: state visualization --
col_img, col_meta = st.columns([1, 1])
with col_img:
    st.image(str(DATA_DIR / current_step["image"]),
             caption=f"Step {step} local tile view",
             width="stretch")
with col_meta:
    st.markdown(f"**Action taken:** `{current_step['action_name']}` "
                f"(id={current_step['action']})")
    st.markdown(f"**Reward:** {current_step['reward']:.2f}"
                f" &nbsp; **Done:** {current_step['done']}",
                unsafe_allow_html=True)
    with st.expander("Filtered obs text (what Gemini sees)", expanded=False):
        st.text(current_step["obs_text_filtered"])
    with st.expander("Full obs text (unfiltered)"):
        st.text(current_step["obs_text_full"])

# -- Strip of all 6 steps for context --
st.markdown("### Trajectory strip (all 6 steps)")
strip_cols = st.columns(6)
for j, sc in enumerate(strip_cols):
    with sc:
        st.image(str(DATA_DIR / traj["steps"][j]["image"]),
                 caption=f"t+{j}: {traj['steps'][j]['action_name']}",
                 width="stretch")

st.divider()

# -- Prompt iteration --
tab_future, tab_action = st.tabs(["Future prediction", "Action selection"])

anchor_step = traj["steps"][0]
filtered_state = anchor_step["obs_text_filtered"]

with tab_future:
    st.markdown(
        "**Task:** Given the step-0 state, generate a 15-step future narrative. "
        "Goal is text whose embedding remains useful for action selection across the "
        "next 5 steps (i.e., doesn't go stale immediately). Steps 1–5 below are the "
        "ground-truth future."
    )
    prompt_future = st.text_area(
        "Prompt (use `{current_state_filtered}` placeholder):",
        value=DEFAULT_FUTURE_PREDICT_PROMPT,
        height=320,
        key="prompt_future",
    )
    run_future = st.button("Run future prediction on selected models", key="run_future")

    if run_future:
        if not api_key:
            st.error("Set GEMINI_API_KEY first.")
        elif not model_choices:
            st.warning("Pick at least one model.")
        else:
            rendered_prompt = prompt_future.replace(
                "{current_state_filtered}", filtered_state
            )
            with st.expander("Final rendered prompt"):
                st.text(rendered_prompt)

            cols = st.columns(len(model_choices))
            for c, mdl in zip(cols, model_choices):
                with c:
                    with st.spinner(f"{mdl}..."):
                        res = call_gemini(rendered_prompt, mdl, api_key,
                                          temperature=temperature,
                                          max_output_tokens=int(max_tokens))
                    st.markdown(f"**{mdl}**")
                    if res.get("error"):
                        st.error(res["error"])
                    else:
                        finish = res.get("finish_reason", "?")
                        thoughts = res.get("thoughts_tokens", 0) or 0
                        budget = res.get("max_output_tokens", "?")
                        cap = " (hit auto-retry cap)" if (
                            finish == "MAX_TOKENS" and budget == MAX_TOKENS_AUTO_CAP
                        ) else ""
                        if finish == "MAX_TOKENS":
                            st.warning(
                                f"finish=MAX_TOKENS at budget={budget}{cap} — output may be truncated. "
                                f"Try larger 'Max output tokens' in the sidebar."
                            )
                        elif finish != "STOP":
                            st.caption(f"finish={finish}")
                        st.caption(
                            f"{res['latency_s']:.1f}s · "
                            f"{res['prompt_tokens']}→{res['completion_tokens']} tok"
                            + (f" · thoughts={thoughts}" if thoughts else "")
                            + f" · budget={budget}"
                        )
                    st.text_area("Response", res["text"], height=400, key=f"fut_{mdl}")

    st.markdown("#### Ground-truth future (t+1 .. t+5)")
    for j in range(1, 6):
        s = traj["steps"][j]
        c_img, c_txt = st.columns([1, 3])
        with c_img:
            st.image(str(DATA_DIR / s["image"]),
                     caption=f"t+{j}: {s['action_name']} · r={s['reward']:.2f}",
                     width="stretch")
        with c_txt:
            with st.expander(f"t+{j} filtered obs text"):
                st.text(s["obs_text_filtered"])


with tab_action:
    st.markdown(
        "**Task:** Given the step-0 state, pick a single action. "
        f"The action actually taken in this trajectory was "
        f"`{anchor_step['action_name']}` (id={anchor_step['action']}). "
        "Compare what each model picks."
    )
    prompt_action = st.text_area(
        "Prompt (use `{current_state_filtered}` placeholder):",
        value=DEFAULT_ACTION_SELECT_PROMPT,
        height=320,
        key="prompt_action",
    )
    run_action = st.button("Run action selection on selected models", key="run_action")

    if run_action:
        if not api_key:
            st.error("Set GEMINI_API_KEY first.")
        elif not model_choices:
            st.warning("Pick at least one model.")
        else:
            rendered_prompt = prompt_action.replace(
                "{current_state_filtered}", filtered_state
            )
            with st.expander("Final rendered prompt"):
                st.text(rendered_prompt)

            cols = st.columns(len(model_choices))
            for c, mdl in zip(cols, model_choices):
                with c:
                    with st.spinner(f"{mdl}..."):
                        res = call_gemini(rendered_prompt, mdl, api_key,
                                          temperature=temperature,
                                          max_output_tokens=int(max_tokens))
                    st.markdown(f"**{mdl}**")
                    if res.get("error"):
                        st.error(res["error"])
                    else:
                        finish = res.get("finish_reason", "?")
                        thoughts = res.get("thoughts_tokens", 0) or 0
                        budget = res.get("max_output_tokens", "?")
                        cap = " (hit auto-retry cap)" if (
                            finish == "MAX_TOKENS" and budget == MAX_TOKENS_AUTO_CAP
                        ) else ""
                        if finish == "MAX_TOKENS":
                            st.warning(
                                f"finish=MAX_TOKENS at budget={budget}{cap} — output may be truncated."
                            )
                        elif finish != "STOP":
                            st.caption(f"finish={finish}")
                        st.caption(
                            f"{res['latency_s']:.1f}s · "
                            f"{res['prompt_tokens']}→{res['completion_tokens']} tok"
                            + (f" · thoughts={thoughts}" if thoughts else "")
                            + f" · budget={budget}"
                        )
                        # Extract predicted action name
                        text = res["text"]
                        pred = None
                        for line in text.splitlines():
                            if line.strip().upper().startswith("ACTION:"):
                                pred = line.split(":", 1)[1].strip()
                                break
                        if pred:
                            match = pred == anchor_step["action_name"]
                            st.markdown(
                                f"Predicted: `{pred}`  "
                                f"{'✅ matches' if match else '❌ differs from'} ground-truth "
                                f"`{anchor_step['action_name']}`"
                            )
                    st.text_area("Response", res["text"], height=400, key=f"act_{mdl}")
