#!/usr/bin/env python3
"""Streamlit prompt-iteration UI for Craftax Qwen/vLLM experiments."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prompt_iter_backend as backend


st.set_page_config(page_title="Craftax Prompt Iteration", layout="wide")


def _init_defaults() -> None:
    if "manifest_path" not in st.session_state:
        st.session_state["manifest_path"] = str(backend.DEFAULT_MANIFEST)
    if "server_url" not in st.session_state:
        st.session_state["server_url"] = os.getenv("PROMPT_ITER_VLLM_URL", backend.DEFAULT_VLLM_URL)
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = backend.DEFAULT_VLLM_MODEL
    if "model_id" not in st.session_state:
        st.session_state["model_id"] = backend.DEFAULT_MODEL_ID
    if "prompt_variant" not in st.session_state:
        st.session_state["prompt_variant"] = "default"
    if "max_tokens" not in st.session_state:
        st.session_state["max_tokens"] = 256
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.7
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "notes" not in st.session_state:
        st.session_state["notes"] = ""
    if "prefer_chat_completions" not in st.session_state:
        st.session_state["prefer_chat_completions"] = True


def _set_sections_from_variant(variant: str) -> None:
    sections = backend.default_prompt_sections(variant)
    st.session_state["system_prompt"] = sections.system_prompt
    st.session_state["few_shot_examples"] = sections.few_shot_examples
    st.session_state["task_instruction"] = sections.task_instruction
    st.session_state["generation_prefix"] = sections.generation_prefix
    st.session_state["stop_sequences"] = "\n".join(sections.stop_sequences)


@st.cache_data(show_spinner=False)
def _load_states(manifest_path: str):
    return backend.load_fixed_states(Path(manifest_path))


def _current_sections() -> backend.PromptSections:
    stop_sequences = [
        line.strip()
        for line in st.session_state.get("stop_sequences", "").splitlines()
        if line.strip()
    ]
    return backend.PromptSections(
        system_prompt=st.session_state.get("system_prompt", ""),
        few_shot_examples=st.session_state.get("few_shot_examples", ""),
        task_instruction=st.session_state.get("task_instruction", ""),
        generation_prefix=st.session_state.get("generation_prefix", ""),
        stop_sequences=stop_sequences,
    )


def _append_history(mode: str, result_payload) -> None:
    st.session_state["history"].append(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "result": result_payload,
            "notes": st.session_state.get("notes", ""),
        }
    )


def _render_result(result: dict) -> None:
    st.markdown(f"### Result: {result['label']} (`{result['state_id']}`)")
    mode = result.get("request_mode", "unknown")
    st.write(f"Latency: {result['latency_s']:.2f}s | Request mode: `{mode}`")

    tabs = st.tabs(["State", "Prompt", "Response", "Raw JSON"])
    with tabs[0]:
        st.code(result["filtered_text_obs"], language="text")
    with tabs[1]:
        st.code(result["prompt"], language="text")
    with tabs[2]:
        st.code(result["response_text"], language="text")
    with tabs[3]:
        st.code(json.dumps(result["response_json"], indent=2, ensure_ascii=True), language="json")


def _render_batch_results(results: List[dict]) -> None:
    if not results:
        return
    rows = []
    for r in results:
        rows.append(
            {
                "state_id": r["state_id"],
                "label": r["label"],
                "latency_s": round(float(r["latency_s"]), 3),
                "response_chars": len(r.get("response_text", "")),
            }
        )
    st.dataframe(rows, use_container_width=True)

    selected = st.selectbox(
        "Inspect batch item",
        options=[r["state_id"] for r in results],
        index=0,
        key="batch_inspect_state_id",
    )
    for r in results:
        if r["state_id"] == selected:
            _render_result(r)
            break


def main() -> None:
    _init_defaults()

    st.title("Craftax Prompt Iteration Webapp")
    st.caption("Edit prompt sections, run Qwen on a fixed 10-state set, and inspect full prompts/responses.")

    if "system_prompt" not in st.session_state:
        _set_sections_from_variant(st.session_state["prompt_variant"])

    with st.sidebar:
        st.header("Runtime")
        st.text_input("Fixed state manifest", key="manifest_path")
        st.text_input("vLLM URL", key="server_url")
        st.text_input("Served model name", key="model_name")
        st.text_input("Tokenizer model id", key="model_id")
        st.number_input("Max new tokens", min_value=1, max_value=2048, key="max_tokens")
        st.number_input("Temperature", min_value=0.0, max_value=2.0, step=0.05, key="temperature")
        st.checkbox(
            "Prefer /v1/chat/completions (works without local transformers)",
            key="prefer_chat_completions",
        )
        if st.button("Check vLLM health", use_container_width=True):
            health = backend.check_vllm_health(st.session_state["server_url"])
            if health["ok"]:
                st.success(f"vLLM healthy: {health['url']} [{health['status_code']}]")
            else:
                st.error(
                    f"vLLM unhealthy: {health['url']} "
                    f"(status={health['status_code']}, error={health['error']})"
                )

        st.header("Prompt Variant")
        st.selectbox(
            "Base variant",
            options=["default", "future_based", "future_based_opt"],
            key="prompt_variant",
        )
        if st.button("Reset sections from variant defaults", use_container_width=True):
            _set_sections_from_variant(st.session_state["prompt_variant"])

    fixed_states = None
    try:
        fixed_states = _load_states(st.session_state["manifest_path"])
        st.session_state["last_good_fixed_states"] = fixed_states
    except Exception as exc:
        cached_states = st.session_state.get("last_good_fixed_states")
        if cached_states:
            fixed_states = cached_states
            st.warning(
                "Failed to reload fixed states from the selected manifest; "
                f"showing last successful state set instead. Error: {exc}"
            )
        else:
            st.error(f"Failed to load fixed states: {exc}")
            return

    by_id = backend.states_by_id(fixed_states)

    st.subheader("Representative Fixed States (10)")
    preview_rows = []
    for state in fixed_states:
        preview_rows.append(
            {
                "state_id": state.state_id,
                "label": state.label,
                "source": state.source_kind,
                "t": state.t,
                "tags": ", ".join(state.tags),
                "map_preview": state.map_preview(8),
            }
        )
    st.dataframe(preview_rows, use_container_width=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Prompt Sections")
        st.text_area("System prompt", key="system_prompt", height=220)
        st.text_area("Few-shot examples", key="few_shot_examples", height=260)
    with col_right:
        st.subheader("Task + Generation")
        st.text_area("Task instruction", key="task_instruction", height=220)
        st.text_input("Generation prefix", key="generation_prefix")
        st.text_area("Stop sequences (one per line)", key="stop_sequences", height=120)
        st.text_area("Manual notes", key="notes", height=120)

    state_ids = [s.state_id for s in fixed_states]
    st.subheader("Execution")
    single_state_id = st.selectbox("Single state", options=state_ids, index=0)
    batch_state_ids = st.multiselect("Batch states", options=state_ids, default=state_ids)

    run_col_1, run_col_2 = st.columns([1, 1])
    single_run_clicked = run_col_1.button("Run single state", use_container_width=True)
    batch_run_clicked = run_col_2.button("Run batch", use_container_width=True)

    sections = _current_sections()

    if single_run_clicked:
        with st.spinner("Running completion..."):
            try:
                result = backend.run_state(
                    by_id[single_state_id],
                    sections,
                    server_url=st.session_state["server_url"],
                    model_name=st.session_state["model_name"],
                    model_id=st.session_state["model_id"],
                    max_tokens=int(st.session_state["max_tokens"]),
                    temperature=float(st.session_state["temperature"]),
                    stop_sequences=sections.stop_sequences,
                    prefer_chat_completions=bool(st.session_state["prefer_chat_completions"]),
                )
                _append_history("single", result)
                st.success("Single-state run completed.")
                _render_result(result)
            except Exception as exc:
                st.error(f"Single-state run failed: {exc}")

    if batch_run_clicked:
        selected_states = [by_id[state_id] for state_id in batch_state_ids]
        with st.spinner(f"Running batch over {len(selected_states)} states..."):
            try:
                results = backend.run_batch(
                    selected_states,
                    sections,
                    server_url=st.session_state["server_url"],
                    model_name=st.session_state["model_name"],
                    model_id=st.session_state["model_id"],
                    max_tokens=int(st.session_state["max_tokens"]),
                    temperature=float(st.session_state["temperature"]),
                    stop_sequences=sections.stop_sequences,
                    prefer_chat_completions=bool(st.session_state["prefer_chat_completions"]),
                )
                _append_history("batch", results)
                st.success("Batch run completed.")
                _render_batch_results(results)
            except Exception as exc:
                st.error(f"Batch run failed: {exc}")

    st.subheader("Export")
    export_payload = {
        "manifest_path": st.session_state["manifest_path"],
        "server_url": st.session_state["server_url"],
        "model_name": st.session_state["model_name"],
        "model_id": st.session_state["model_id"],
        "prompt_variant": st.session_state["prompt_variant"],
        "sections": asdict(sections),
        "notes": st.session_state.get("notes", ""),
        "history": st.session_state.get("history", []),
    }
    export_json = json.dumps(export_payload, indent=2, ensure_ascii=True)
    st.download_button(
        "Download session JSON",
        data=export_json,
        file_name=f"prompt_iter_session_{time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    if st.session_state.get("history"):
        st.caption(f"Stored runs in session history: {len(st.session_state['history'])}")


if __name__ == "__main__":
    main()
