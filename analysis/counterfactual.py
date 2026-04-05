#!/usr/bin/env python3
"""Manual qualitative text counterfactual analysis for fixed Craftax observations.

This script evaluates policies on:
- base text observations from bundle step_* directories
- manually specified modified text counterfactuals

It keeps the same obs_before vector for each step and changes only text inputs.
Hidden states are cached by (llm_layer, filtered_text) so policies sharing a
layer reuse extracted embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import jax
except ImportError:  # pragma: no cover
    jax = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]


MAP_PREFIX = "Map (interesting tiles only): "
EXTRACTED_LAYERS = [8, 16, 24, 35]


def _import_filter_text_obs():
    from llm.prompts import filter_text_obs  # local import to keep preview lightweight

    return filter_text_obs


def _import_obs_to_text():
    from labelling.obs_to_text import obs_to_text  # local import to keep preview lightweight

    return obs_to_text


@dataclass
class Scenario:
    step_dir: Path
    step: int
    reasoning: str
    obs: Optional["np.ndarray"]
    text_original: str


@dataclass
class Variant:
    step: int
    variant_id: str
    description: str
    text_modified: str


class HiddenStatePool:
    """Caches hidden states for (requested_layer, filtered_text)."""

    def __init__(self, server_url: str, served_model_name: str, model_id: str = "Qwen/Qwen3-4B"):
        self.server_url = server_url.rstrip("/")
        self.served_model_name = served_model_name
        self.model_id = model_id
        self.extractors: Dict[int, VLLMHiddenStateExtractor] = {}
        self.cache: Dict[Tuple[int, str], np.ndarray] = {}

    @staticmethod
    def requested_layer_to_index(requested_layer: int) -> int:
        if requested_layer == -1:
            return -1
        if requested_layer not in EXTRACTED_LAYERS:
            raise ValueError(
                f"Requested layer {requested_layer} not in extracted layers {EXTRACTED_LAYERS}."
            )
        return EXTRACTED_LAYERS.index(requested_layer)

    def _get_extractor(self, requested_layer: int) -> VLLMHiddenStateExtractor:
        from llm.extractor import VLLMHiddenStateExtractor

        if requested_layer not in self.extractors:
            target_index = self.requested_layer_to_index(requested_layer)
            self.extractors[requested_layer] = VLLMHiddenStateExtractor(
                server_url=self.server_url,
                model_name=self.served_model_name,
                model_id=self.model_id,
                target_layer=target_index,
            )
        return self.extractors[requested_layer]

    def encode_many(
        self,
        requested_layer: int,
        text_obs_list: Sequence[str],
        batch_size: int = 16,
    ) -> None:
        pending: List[str] = []
        filtered_pending: List[str] = []
        filter_text_obs = _import_filter_text_obs()

        for text_obs in text_obs_list:
            filtered = filter_text_obs(text_obs)
            key = (requested_layer, filtered)
            if key in self.cache:
                continue
            pending.append(text_obs)
            filtered_pending.append(filtered)

        if not pending:
            return

        extractor = self._get_extractor(requested_layer)
        hidden, _ = extractor.extract_hidden_states_no_cot(
            filtered_pending,
            batch_size=max(1, min(batch_size, len(filtered_pending))),
        )
        hidden = np.asarray(hidden, dtype=np.float32)

        for filtered, h in zip(filtered_pending, hidden):
            self.cache[(requested_layer, filtered)] = h.astype(np.float32, copy=False)

    def get(self, requested_layer: int, text_obs: str) -> np.ndarray:
        filter_text_obs = _import_filter_text_obs()
        filtered = filter_text_obs(text_obs)
        key = (requested_layer, filtered)
        if key not in self.cache:
            self.encode_many(requested_layer, [text_obs], batch_size=1)
        vec = self.cache[key]
        return vec.reshape(1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual qualitative text counterfactual analysis")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--policy_ids", type=str, default="")
    parser.add_argument("--bundle_dir", type=str, required=True)
    parser.add_argument("--counterfactuals_yaml", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--default_llm_layer", type=int, default=-1)
    parser.add_argument("--hidden_batch_size", type=int, default=16)
    parser.add_argument("--preview_only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def parse_policy_id_filter(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_scenarios(bundle_dir: Path, require_obs: bool = True) -> List[Scenario]:
    filter_text_obs = _import_filter_text_obs()
    obs_to_text = _import_obs_to_text()

    scenario_root = bundle_dir
    step_dirs = sorted(p.resolve() for p in scenario_root.glob("step_*") if p.is_dir())
    if not step_dirs and (bundle_dir / "bundles").is_dir():
        scenario_root = bundle_dir / "bundles"
        step_dirs = sorted(p.resolve() for p in scenario_root.glob("step_*") if p.is_dir())

    scenarios: List[Scenario] = []
    for step_dir in step_dirs:
        meta_path = step_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        step_num = int(meta.get("step", step_dir.name.replace("step_", "")))
        reasoning = str(meta.get("reasoning", "")).strip()

        obs = None
        obs_path = step_dir / "obs_before.npy"
        if require_obs:
            if np is None:
                raise ImportError("numpy is required for evaluation mode (not preview-only).")
            if not obs_path.exists():
                continue
            obs = np.load(obs_path).astype(np.float32).reshape(1, -1)

        text_path = step_dir / "before_state_filtered.txt"
        if text_path.exists():
            text_original = text_path.read_text().strip()
        else:
            if obs is None:
                raise ValueError(
                    f"Missing before_state_filtered.txt and obs unavailable for text fallback in {step_dir}"
                )
            text_original = filter_text_obs(obs_to_text(obs[0])).strip()

        scenarios.append(
            Scenario(
                step_dir=step_dir,
                step=step_num,
                reasoning=reasoning,
                obs=obs,
                text_original=text_original,
            )
        )

    if not scenarios:
        raise ValueError(f"No step_* scenarios found under {bundle_dir}")
    scenarios.sort(key=lambda s: s.step)
    return scenarios


def normalize_map_entry(entry: str) -> str:
    m = re.match(r"\s*(-?\d+)\s*,\s*(-?\d+)\s*:(.+)\s*$", entry)
    if not m:
        return entry.strip()
    x, y, rest = m.groups()
    return f"{int(x)}, {int(y)}:{rest.strip()}"


def split_map_entries(payload: str) -> List[str]:
    payload = payload.strip()
    if not payload:
        return []

    starts = list(re.finditer(r"-?\d+\s*,\s*-?\d+\s*:", payload))
    if not starts:
        return [payload]

    entries: List[str] = []
    for i, m in enumerate(starts):
        start = m.start()
        end = starts[i + 1].start() - 2 if i + 1 < len(starts) else len(payload)
        token = payload[start:end].strip().rstrip(",")
        if token:
            entries.append(normalize_map_entry(token))
    return entries


def update_map_line(text: str, map_remove: Sequence[str], map_add: Sequence[str]) -> str:
    lines = text.splitlines()
    map_idx = next((i for i, line in enumerate(lines) if line.startswith(MAP_PREFIX)), None)
    if map_idx is None:
        raise ValueError("Could not find map line to update")

    payload = lines[map_idx][len(MAP_PREFIX) :]
    entries = split_map_entries(payload)

    remove_set = {normalize_map_entry(x) for x in map_remove}
    add_norm = [normalize_map_entry(x) for x in map_add]

    entries = [e for e in entries if e not in remove_set]
    for token in add_norm:
        if token not in entries:
            entries.append(token)

    lines[map_idx] = MAP_PREFIX + ", ".join(entries)
    return "\n".join(lines)


def apply_variant_to_text(base_text: str, variant_cfg: Dict) -> str:
    text = str(base_text).strip()

    map_remove = [str(x) for x in variant_cfg.get("map_remove", [])]
    map_add = [str(x) for x in variant_cfg.get("map_add", [])]
    if map_remove or map_add:
        text = update_map_line(text, map_remove=map_remove, map_add=map_add)

    remove_lines = [str(x) for x in variant_cfg.get("remove_lines", [])]
    if remove_lines:
        line_set = {x.strip() for x in remove_lines}
        lines = [ln for ln in text.splitlines() if ln.strip() not in line_set]
        text = "\n".join(lines)

    replacements = variant_cfg.get("line_replacements", [])
    if replacements:
        lines = text.splitlines()
        for rep in replacements:
            src = str(rep["from"])
            dst = str(rep["to"])
            replaced = False
            for i, line in enumerate(lines):
                if line == src:
                    lines[i] = dst
                    replaced = True
                    break
            if not replaced:
                raise ValueError(f"Line replacement source not found: {src}")
        text = "\n".join(lines)

    return text.strip()


def load_variants(counterfactuals_yaml: Path, scenarios_by_step: Dict[int, Scenario]) -> List[Variant]:
    raw = yaml.safe_load(counterfactuals_yaml.read_text())
    cfg_variants = raw.get("variants", [])

    variants: List[Variant] = []
    for item in cfg_variants:
        step = int(item["step"])
        if step not in scenarios_by_step:
            raise ValueError(f"Variant references missing step: {step}")

        variant_id = str(item["variant_id"])
        description = str(item.get("description", "")).strip()
        base_text = scenarios_by_step[step].text_original
        text_modified = apply_variant_to_text(base_text, item)

        variants.append(
            Variant(
                step=step,
                variant_id=variant_id,
                description=description,
                text_modified=text_modified,
            )
        )

    expected_total = raw.get("expected_total_variants")
    if expected_total is not None and int(expected_total) != len(variants):
        raise ValueError(
            f"Expected {expected_total} variants from YAML, got {len(variants)}"
        )

    expected_per_step = raw.get("expected_variants_per_step", {}) or {}
    if expected_per_step:
        actual: Dict[int, int] = {}
        for v in variants:
            actual[v.step] = actual.get(v.step, 0) + 1
        for k, val in expected_per_step.items():
            step = int(k)
            want = int(val)
            got = int(actual.get(step, 0))
            if got != want:
                raise ValueError(
                    f"Expected {want} variants for step {step}, got {got}"
                )

    variants.sort(key=lambda v: (v.step, v.variant_id))
    return variants


def load_manifest_specs_and_layers(
    manifest_path: Path,
    policy_ids: List[str],
    default_llm_layer: int,
):
    from scripts.eval_policy_wave import resolve_manifest_policies

    manifest = yaml.safe_load(manifest_path.read_text())
    specs = resolve_manifest_policies(manifest, include_slices=False, slice_count=0)

    if policy_ids:
        wanted = set(policy_ids)
        specs = [s for s in specs if s.policy_id in wanted]
        missing = sorted(wanted - {s.policy_id for s in specs})
        if missing:
            raise ValueError(f"Missing requested policy_ids in manifest resolution: {missing}")

    if not specs:
        raise ValueError("No policies resolved from manifest/policy filter")

    policy_cfg = {str(p["id"]): p for p in manifest.get("policies", [])}
    policy_layers: Dict[str, int] = {}
    for spec in specs:
        cfg = policy_cfg.get(spec.policy_id, {})
        policy_layers[spec.policy_id] = int(cfg.get("llm_layer", default_llm_layer))

    return specs, policy_layers


def action_name(action_id: int) -> str:
    try:
        from craftax.craftax.constants import Action

        return Action(int(action_id)).name
    except Exception:
        return f"UNKNOWN_{int(action_id)}"


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plots(output_dir: Path, summary_rows: List[Dict], value_deltas: Dict[str, List[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["policy_id"] for r in summary_rows]
    rates = [float(r["action_change_rate"]) for r in summary_rows]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, rates)
    plt.ylabel("Action Change Rate")
    plt.xlabel("Policy")
    plt.title("Action Change Rate: Base vs Modified Text")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "action_change_rate.png", dpi=150)
    plt.close()

    box_labels = []
    box_data = []
    for label in labels:
        box_labels.append(label)
        box_data.append(value_deltas.get(label, [0.0]))

    plt.figure(figsize=(12, 5))
    plt.boxplot(box_data, labels=box_labels, showfliers=True)
    plt.ylabel("Value Delta (modified - base)")
    plt.xlabel("Policy")
    plt.title("Value Delta Distribution Across Counterfactuals")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "value_delta_boxplot.png", dpi=150)
    plt.close()


def preview_variants(variants: Sequence[Variant], scenarios_by_step: Dict[int, Scenario]) -> str:
    chunks: List[str] = []
    for v in variants:
        scenario = scenarios_by_step[v.step]
        chunks.append(
            "\n".join(
                [
                    f"=== step_{v.step:04d} | {v.variant_id} ===",
                    f"reasoning: {scenario.reasoning}",
                    f"description: {v.description}",
                    "--- modified_text ---",
                    v.text_modified,
                    "",
                ]
            )
        )
    return "\n".join(chunks).strip() + "\n"


def evaluate_policy(
    policy: Dict,
    obs: np.ndarray,
    hidden_in: np.ndarray,
    rng: jax.Array,
) -> Tuple[int, str, float]:
    action_id = int(policy["act_fn"](obs, hidden_in, True, rng)[0])
    value = float(policy["value_fn"](obs, hidden_in)[0])
    return action_id, action_name(action_id), value


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    counterfactuals_yaml = Path(args.counterfactuals_yaml).expanduser().resolve()

    scenarios = load_scenarios(bundle_dir, require_obs=not args.preview_only)
    scenarios_by_step = {s.step: s for s in scenarios}
    variants = load_variants(counterfactuals_yaml, scenarios_by_step)

    preview_text = preview_variants(variants, scenarios_by_step)
    preview_path = output_dir / "counterfactual_preview.txt"
    preview_path.write_text(preview_text, encoding="utf-8")
    print(preview_text)
    print(f"Wrote preview: {preview_path}")

    if args.preview_only:
        return

    if np is None:
        raise ImportError("numpy is required for evaluation mode (not preview-only).")
    if jax is None:
        raise ImportError("jax is required for evaluation mode (not preview-only).")
    if torch is None:
        raise ImportError("torch is required for evaluation mode (not preview-only).")

    from scripts.eval_policy_wave import load_policy

    manifest_path = Path(args.manifest).expanduser().resolve()
    policy_filter = parse_policy_id_filter(args.policy_ids)
    specs, policy_layers = load_manifest_specs_and_layers(
        manifest_path=manifest_path,
        policy_ids=policy_filter,
        default_llm_layer=args.default_llm_layer,
    )

    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
    served_model = os.environ.get("VLLM_SERVED_MODEL_NAME", "./configs/vllm_hidden_qwen4b")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_policies: Dict[str, Dict] = {}
    for spec in specs:
        loaded_policies[spec.policy_id] = load_policy(spec, device=device)

    hidden_pool = HiddenStatePool(server_url=vllm_url, served_model_name=served_model)

    required_layers = sorted(
        {
            int(policy_layers[spec.policy_id])
            for spec in specs
            if loaded_policies[spec.policy_id].get("uses_hidden", False)
        }
    )

    text_universe = sorted(
        {
            s.text_original for s in scenarios
        }
        | {v.text_modified for v in variants}
    )

    for layer in required_layers:
        hidden_pool.encode_many(layer, text_universe, batch_size=args.hidden_batch_size)

    base_results: Dict[Tuple[str, int], Dict] = {}
    records: List[Dict] = []

    fixed_rng = jax.random.PRNGKey(0)

    for spec in specs:
        policy = loaded_policies[spec.policy_id]
        uses_hidden = bool(policy.get("uses_hidden", False))
        layer = int(policy_layers[spec.policy_id])

        for scenario in scenarios:
            if uses_hidden:
                hidden_raw = hidden_pool.get(layer, scenario.text_original)
                hm = policy.get("hidden_mean")
                hs = policy.get("hidden_std")
                hidden_in = (hidden_raw - hm[None, :]) / hs[None, :] if hm is not None and hs is not None else hidden_raw
            else:
                hidden_in = np.zeros((1, 1), dtype=np.float32)

            action_id, action_label, value = evaluate_policy(policy, scenario.obs, hidden_in, fixed_rng)
            base_results[(spec.policy_id, scenario.step)] = {
                "action_id": action_id,
                "action_name": action_label,
                "value": value,
                "llm_layer": layer,
            }

            records.append(
                {
                    "policy_id": spec.policy_id,
                    "policy_name": spec.policy_name,
                    "variant_name": spec.variant_name,
                    "checkpoint_path": spec.checkpoint_path,
                    "step": scenario.step,
                    "counterfactual_id": "base",
                    "counterfactual_description": "base_text",
                    "text_variant": "base",
                    "action_id": action_id,
                    "action_name": action_label,
                    "value": value,
                    "llm_layer": layer,
                }
            )

    comparisons: List[Dict] = []
    summary_by_policy: Dict[str, Dict[str, object]] = {
        spec.policy_id: {
            "policy_id": spec.policy_id,
            "policy_name": spec.policy_name,
            "variant_name": spec.variant_name,
            "checkpoint_path": spec.checkpoint_path,
            "num_counterfactuals": 0,
            "action_change_count": 0,
            "value_deltas": [],
        }
        for spec in specs
    }

    for spec in specs:
        policy = loaded_policies[spec.policy_id]
        uses_hidden = bool(policy.get("uses_hidden", False))
        layer = int(policy_layers[spec.policy_id])

        for variant in variants:
            scenario = scenarios_by_step[variant.step]
            base = base_results[(spec.policy_id, variant.step)]

            if uses_hidden:
                hidden_raw = hidden_pool.get(layer, variant.text_modified)
                hm = policy.get("hidden_mean")
                hs = policy.get("hidden_std")
                hidden_in = (hidden_raw - hm[None, :]) / hs[None, :] if hm is not None and hs is not None else hidden_raw
            else:
                hidden_in = np.zeros((1, 1), dtype=np.float32)

            mod_action_id, mod_action_name, mod_value = evaluate_policy(policy, scenario.obs, hidden_in, fixed_rng)

            records.append(
                {
                    "policy_id": spec.policy_id,
                    "policy_name": spec.policy_name,
                    "variant_name": spec.variant_name,
                    "checkpoint_path": spec.checkpoint_path,
                    "step": scenario.step,
                    "counterfactual_id": variant.variant_id,
                    "counterfactual_description": variant.description,
                    "text_variant": "modified",
                    "action_id": mod_action_id,
                    "action_name": mod_action_name,
                    "value": mod_value,
                    "llm_layer": layer,
                }
            )

            action_changed = int(base["action_id"] != mod_action_id)
            value_delta = float(mod_value - base["value"])
            comparisons.append(
                {
                    "policy_id": spec.policy_id,
                    "policy_name": spec.policy_name,
                    "variant_name": spec.variant_name,
                    "checkpoint_path": spec.checkpoint_path,
                    "step": variant.step,
                    "counterfactual_id": variant.variant_id,
                    "counterfactual_description": variant.description,
                    "llm_layer": layer,
                    "base_action_id": base["action_id"],
                    "base_action_name": base["action_name"],
                    "modified_action_id": mod_action_id,
                    "modified_action_name": mod_action_name,
                    "action_changed": action_changed,
                    "base_value": base["value"],
                    "modified_value": mod_value,
                    "value_delta": value_delta,
                }
            )

            summary = summary_by_policy[spec.policy_id]
            summary["num_counterfactuals"] = int(summary["num_counterfactuals"]) + 1
            summary["action_change_count"] = int(summary["action_change_count"]) + action_changed
            summary["value_deltas"].append(value_delta)

    summary_rows: List[Dict] = []
    value_deltas_for_plot: Dict[str, List[float]] = {}
    for spec in specs:
        s = summary_by_policy[spec.policy_id]
        deltas = list(s["value_deltas"])
        n = max(1, int(s["num_counterfactuals"]))
        summary_rows.append(
            {
                "policy_id": spec.policy_id,
                "policy_name": spec.policy_name,
                "variant_name": spec.variant_name,
                "checkpoint_path": spec.checkpoint_path,
                "llm_layer": int(policy_layers[spec.policy_id]),
                "num_counterfactuals": int(s["num_counterfactuals"]),
                "action_change_count": int(s["action_change_count"]),
                "action_change_rate": float(s["action_change_count"]) / float(n),
                "value_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
                "value_delta_std": float(np.std(deltas)) if deltas else 0.0,
                "value_delta_abs_mean": float(np.mean(np.abs(np.asarray(deltas)))) if deltas else 0.0,
            }
        )
        value_deltas_for_plot[spec.policy_id] = deltas

    write_jsonl(output_dir / "records.jsonl", records)
    write_csv(
        output_dir / "comparisons.csv",
        comparisons,
        [
            "policy_id",
            "policy_name",
            "variant_name",
            "checkpoint_path",
            "step",
            "counterfactual_id",
            "counterfactual_description",
            "llm_layer",
            "base_action_id",
            "base_action_name",
            "modified_action_id",
            "modified_action_name",
            "action_changed",
            "base_value",
            "modified_value",
            "value_delta",
        ],
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "policy_id",
            "policy_name",
            "variant_name",
            "checkpoint_path",
            "llm_layer",
            "num_counterfactuals",
            "action_change_count",
            "action_change_rate",
            "value_delta_mean",
            "value_delta_std",
            "value_delta_abs_mean",
        ],
    )

    make_plots(output_dir, summary_rows, value_deltas_for_plot)

    run_meta = {
        "manifest": str(manifest_path),
        "bundle_dir": str(bundle_dir),
        "counterfactuals_yaml": str(counterfactuals_yaml),
        "policy_ids": [s.policy_id for s in specs],
        "policy_layers": policy_layers,
        "num_scenarios": len(scenarios),
        "num_counterfactuals": len(variants),
        "output_dir": str(output_dir),
        "vllm_url": vllm_url,
        "served_model": served_model,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Wrote records: {output_dir / 'records.jsonl'}")
    print(f"Wrote comparisons: {output_dir / 'comparisons.csv'}")
    print(f"Wrote summary: {output_dir / 'summary.csv'}")
    print(
        f"Wrote plots: {output_dir / 'action_change_rate.png'}, "
        f"{output_dir / 'value_delta_boxplot.png'}"
    )


if __name__ == "__main__":
    main()
