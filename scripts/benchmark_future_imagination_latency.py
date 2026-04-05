#!/usr/bin/env python3
"""
Benchmark future-imagination prompt latency across providers/models.

This script benchmarks two prompt variants:
  - predict_state_only
  - predict_history_k5

It can run against:
  - Gemini REST API (generateContent)
  - OpenAI-compatible endpoints (e.g., vLLM /v1/completions)
  - Local HuggingFace models via transformers (hf_local)
  - Native vLLM offline API with hidden-state output (vllm_offline)

For OpenAI-compatible endpoints with vLLM hidden-state plugin enabled, it also:
  - extracts hidden_states_path from the response (or completion-id fallback),
  - loads the safetensors file,
  - reports hidden-state load latency and shape.

For hf_local, it:
  - runs generation locally via transformers,
  - runs an explicit forward pass to extract last-token hidden state,
  - reports hidden extraction latency and shape without plugin sidecar files.

For vllm_offline, it:
  - uses vLLM's native speculative method "extract_hidden_states",
  - follows the official offline example with ExampleHiddenStatesConnector,
  - reads prompt hidden states from connector-written safetensors files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


DEFAULT_GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
)


@dataclass
class PromptItem:
    variant: str
    timestep: int
    prompt_path: str
    prompt_text: str
    repeat_idx: int


def _effective_call_params(
    benchmark_mode: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[int, float, float]:
    mode = (benchmark_mode or "generate_and_hidden").strip().lower()
    if mode == "hidden_only":
        return 1, 0.0, 1.0
    return int(max_tokens), float(temperature), float(top_p)


def _safe_int(v: object) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _load_prompt_variant(
    variant: str,
    prompt_dir: str,
    repeats: int,
) -> List[PromptItem]:
    files = sorted(glob.glob(os.path.join(prompt_dir, "t_*.txt")))
    items: List[PromptItem] = []
    for path in files:
        name = os.path.basename(path)
        # expected format t_00000.txt
        t_raw = name.replace("t_", "").replace(".txt", "")
        try:
            timestep = int(t_raw)
        except Exception:
            continue
        text = Path(path).read_text(encoding="utf-8")
        for r in range(repeats):
            items.append(
                PromptItem(
                    variant=variant,
                    timestep=timestep,
                    prompt_path=path,
                    prompt_text=text,
                    repeat_idx=r,
                )
            )
    return items


def _load_raw_config_dict(model_id: str) -> Dict[str, object]:
    local_path = Path(model_id)
    candidate = local_path / "config.json" if local_path.is_dir() else local_path
    if candidate.exists():
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data

    try:
        quoted = urlparse.quote(model_id, safe="/")
        url = f"https://huggingface.co/{quoted}/raw/main/config.json"
        with urlrequest.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
        data = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    raise RuntimeError(f"Could not load config.json for model/config path: {model_id}")


def _cfg_lookup(cfg: Dict[str, object], key: str, default=None):
    if key in cfg and cfg.get(key) is not None:
        return cfg.get(key)
    for nested_key in ("text_config", "llm_config"):
        nested = cfg.get(nested_key)
        if isinstance(nested, dict) and nested.get(key) is not None:
            return nested.get(key)
    return default


def _extract_openai_text(resp_json: Dict[str, object]) -> str:
    choices = resp_json.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    if "text" in first:
        return str(first.get("text", ""))
    msg = first.get("message")
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    return ""


def _extract_openai_usage(resp_json: Dict[str, object]) -> Dict[str, Optional[int]]:
    usage = resp_json.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        "prompt_tokens": _safe_int(usage.get("prompt_tokens")),
        "completion_tokens": _safe_int(usage.get("completion_tokens")),
        "total_tokens": _safe_int(usage.get("total_tokens")),
    }


def _extract_gemini_text(resp_json: Dict[str, object]) -> str:
    candidates = resp_json.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return ""
    c0 = candidates[0]
    if not isinstance(c0, dict):
        return ""
    content = c0.get("content", {})
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        return ""
    chunks: List[str] = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            chunks.append(str(part["text"]))
    return "".join(chunks)


def _extract_gemini_usage(resp_json: Dict[str, object]) -> Dict[str, Optional[int]]:
    usage = resp_json.get("usageMetadata", {})
    if not isinstance(usage, dict):
        usage = {}
    return {
        "prompt_tokens": _safe_int(usage.get("promptTokenCount")),
        "completion_tokens": _safe_int(usage.get("candidatesTokenCount")),
        "total_tokens": _safe_int(usage.get("totalTokenCount")),
        "thoughts_tokens": _safe_int(usage.get("thoughtsTokenCount")),
    }


def _usage_int(usage: Dict[str, object], key: str) -> Optional[int]:
    if not isinstance(usage, dict):
        return None
    return _safe_int(usage.get(key))


def _extract_hidden_path(
    result_json: Dict[str, object],
    hidden_states_path: Optional[str],
) -> Optional[str]:
    # Choices first
    choices = result_json.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            kv = choice.get("kv_transfer_params", {})
            if isinstance(kv, dict):
                p = kv.get("hidden_states_path")
                if isinstance(p, str) and p:
                    return p

    # Top-level fallback
    kv_top = result_json.get("kv_transfer_params", {})
    if isinstance(kv_top, dict):
        p = kv_top.get("hidden_states_path")
        if isinstance(p, str) and p:
            return p

    # Completion id fallback
    if hidden_states_path:
        completion_id = result_json.get("id")
        if isinstance(completion_id, str) and completion_id:
            pattern = os.path.join(hidden_states_path, f"{completion_id}-*.safetensors")
            for attempt in range(8):
                matches = glob.glob(pattern)
                if matches:
                    return max(matches, key=os.path.getmtime)
                time.sleep(0.05 * (attempt + 1))
    return None


def _load_hidden(
    hidden_file: str,
    target_layer: int,
    expected_num_layers: Optional[int],
    delete_after_read: bool,
) -> Tuple[Optional[List[int]], Optional[float], Optional[float], Optional[float]]:
    import numpy as np
    import torch
    from safetensors import safe_open

    started = time.perf_counter()
    if not hidden_file or not os.path.exists(hidden_file):
        return None, None, None, None

    tensor = None
    last_error = None
    for attempt in range(12):
        try:
            with safe_open(hidden_file, framework="pt") as f:
                keys = list(f.keys())
                if "hidden_states" in keys:
                    tensor = f.get_tensor("hidden_states")
                elif "aux_hidden_states" in keys:
                    tensor = f.get_tensor("aux_hidden_states")
                else:
                    tensor = None
            break
        except Exception as exc:  # pragma: no cover
            last_error = exc
            time.sleep(0.05 * (attempt + 1))

    if tensor is None:
        _ = last_error
        return None, None, None, None

    if tensor.ndim == 3 and expected_num_layers is not None:
        expected = int(expected_num_layers)
        if expected > 0 and tensor.shape[0] != expected and tensor.shape[1] == expected:
            # Some connector/plugin paths emit [seq, layers, hidden].
            tensor = tensor.transpose(0, 1).contiguous()
        if expected > 0 and tensor.shape[0] != expected:
            raise ValueError(
                "Unexpected hidden tensor layout: "
                f"shape={tuple(tensor.shape)}, expected_num_layers={expected}"
            )

    shape = list(tensor.shape)

    vec = None
    try:
        if tensor.ndim == 3:
            layer_idx = int(target_layer)
            if layer_idx < 0:
                layer_idx = tensor.shape[0] + layer_idx
            if layer_idx < 0 or layer_idx >= tensor.shape[0]:
                layer_idx = tensor.shape[0] - 1
            layer = tensor[layer_idx]  # [seq, hidden]
            vec = layer[-1].to(torch.float32).cpu().numpy()
        elif tensor.ndim == 2:
            vec = tensor[-1].to(torch.float32).cpu().numpy()
        else:
            vec = tensor.reshape(-1).to(torch.float32).cpu().numpy()
        norm = float(np.linalg.norm(vec))
        mean = float(np.mean(vec))
    except Exception as exc:  # pragma: no cover
        norm = None
        mean = None
        if delete_after_read:
            try:
                os.remove(hidden_file)
            except OSError:
                pass
        _ = exc
        return shape, None, None, None

    load_s = time.perf_counter() - started
    if delete_after_read:
        try:
            os.remove(hidden_file)
        except OSError:
            pass

    return shape, load_s, norm, mean


def _resolve_torch_dtype(name: str):
    import torch

    key = (name or "").strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    if key in {"auto", ""}:
        return "auto"
    raise ValueError(f"Unsupported --hf-dtype: {name}")


class _HFLocalRunner:
    def __init__(
        self,
        *,
        model_id: str,
        device_map: str,
        dtype_name: str,
        trust_remote_code: bool,
        attn_implementation: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        model_kwargs: Dict[str, object] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        resolved_dtype = _resolve_torch_dtype(dtype_name)
        model_kwargs["torch_dtype"] = resolved_dtype

        if device_map and device_map.lower() != "none":
            model_kwargs["device_map"] = device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model.eval()
        self.input_device = self._infer_input_device()

    def _infer_input_device(self):
        if hasattr(self.model, "hf_device_map"):
            hf_map = getattr(self.model, "hf_device_map", None)
            if isinstance(hf_map, dict):
                for dev in hf_map.values():
                    if isinstance(dev, int):
                        return self._torch.device(f"cuda:{dev}")
                    if isinstance(dev, str):
                        if dev in {"cpu", "disk"}:
                            continue
                        return self._torch.device(dev)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self._torch.device("cpu")

    def _maybe_sync_cuda(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def call(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        target_layer: int,
        benchmark_mode: str,
    ) -> Dict[str, object]:
        import numpy as np

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.input_device) for k, v in encoded.items()}
        input_len = int(encoded["input_ids"].shape[-1])

        # Hidden extraction pass: last token at the target layer.
        self._maybe_sync_cuda()
        hidden_started = time.perf_counter()
        with self._torch.inference_mode():
            hidden_out = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                output_hidden_states=True,
                use_cache=False,
            )
        self._maybe_sync_cuda()
        hidden_s = time.perf_counter() - hidden_started

        if not getattr(hidden_out, "hidden_states", None):
            raise RuntimeError("HF model call returned no hidden_states.")
        hs = hidden_out.hidden_states[target_layer]
        hs_shape = list(hs.shape)
        vec = hs[0, -1, :].to(self._torch.float32).detach().cpu().numpy()
        hidden_norm = float(np.linalg.norm(vec))
        hidden_mean = float(np.mean(vec))
        del hidden_out

        mode = (benchmark_mode or "generate_and_hidden").strip().lower()
        if mode == "hidden_only":
            usage = {
                "prompt_tokens": input_len,
                "completion_tokens": 0,
                "total_tokens": input_len,
            }
            return {
                "request_s": hidden_s,
                "total_s": hidden_s,
                "text": "",
                "usage": usage,
                "hidden": {
                    "shape": hs_shape,
                    "hidden_extract_s": hidden_s,
                    "hidden_norm": hidden_norm,
                    "hidden_mean": hidden_mean,
                },
            }

        # Completion pass.
        do_sample = float(temperature) > 0.0
        gen_kwargs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask"),
            "max_new_tokens": int(max_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        self._maybe_sync_cuda()
        req_started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self.model.generate(**gen_kwargs)
        self._maybe_sync_cuda()
        request_s = time.perf_counter() - req_started

        out_tokens = generated[:, input_len:]
        completion_tokens = int(out_tokens.shape[-1])
        text = (
            self.tokenizer.decode(out_tokens[0], skip_special_tokens=True)
            if completion_tokens > 0
            else ""
        )

        usage = {
            "prompt_tokens": input_len,
            "completion_tokens": completion_tokens,
            "total_tokens": input_len + completion_tokens,
        }

        return {
            "request_s": request_s,
            "total_s": hidden_s + request_s,
            "text": text,
            "usage": usage,
            "hidden": {
                "shape": hs_shape,
                "hidden_extract_s": hidden_s,
                "hidden_norm": hidden_norm,
                "hidden_mean": hidden_mean,
            },
        }


class _VLLMOfflineRunner:
    def __init__(
        self,
        *,
        model_id: str,
        target_layer: int,
        tensor_parallel_size: int,
        dtype_name: str,
        trust_remote_code: bool,
        max_model_len: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        spec_num_tokens: int,
        enable_prefix_caching: bool,
        enable_chunked_prefill: bool,
        enable_hybrid_kv_cache_manager: bool,
        kv_connector_module_path: str,
        kv_connector_mode: str,
    ) -> None:
        import tempfile
        from vllm import LLM, SamplingParams
 
        self._SamplingParams = SamplingParams
        self._target_layer = int(target_layer)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="vllm_hidden_states_")
        self._hidden_dir = self._tmpdir.name

        cfg = _load_raw_config_dict(model_id)
        num_hidden_layers = _cfg_lookup(cfg, "num_hidden_layers")
        if num_hidden_layers is None:
            raise RuntimeError(
                "Could not infer num_hidden_layers from model config for vLLM "
                f"extract_hidden_states: model={model_id}"
            )

        resolved_layer = self._target_layer
        if resolved_layer < 0:
            resolved_layer = int(num_hidden_layers) + resolved_layer
        if resolved_layer < 0 or resolved_layer >= int(num_hidden_layers):
            raise ValueError(
                "hidden target layer out of range: "
                f"target={self._target_layer}, resolved={resolved_layer}, "
                f"num_hidden_layers={num_hidden_layers}"
            )
        self._resolved_layer = int(resolved_layer)
        self._expected_num_layers = 1

        llm_kwargs: Dict[str, object] = {
            "model": model_id,
            "trust_remote_code": bool(trust_remote_code),
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_model_len": int(max_model_len),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "dtype": (dtype_name or "auto"),
            "enable_prefix_caching": bool(enable_prefix_caching),
            "enable_chunked_prefill": bool(enable_chunked_prefill),
            "speculative_config": {
                "method": "extract_hidden_states",
                "num_speculative_tokens": int(spec_num_tokens),
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": [self._resolved_layer],
                    },
                },
            },
            "kv_transfer_config": {
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_module_path": (
                    kv_connector_module_path.strip() if kv_connector_module_path else None
                ),
                "kv_connector_extra_config": {
                    "shared_storage_path": self._hidden_dir,
                    "mode": kv_connector_mode,
                },
            },
        }
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if enable_hybrid_kv_cache_manager:
            llm_kwargs["disable_hybrid_kv_cache_manager"] = False

        self.llm = LLM(**llm_kwargs)

    def _fallback_hidden_file(self, before_files: set[str]) -> Optional[str]:
        all_files = set(glob.glob(os.path.join(self._hidden_dir, "*.safetensors")))
        created = sorted(all_files - before_files)
        if created:
            return max(created, key=os.path.getmtime)
        return None

    def call(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        target_layer: int,
        benchmark_mode: str,
    ) -> Dict[str, object]:
        mode = (benchmark_mode or "generate_and_hidden").strip().lower()
        effective_max_tokens, effective_temperature, effective_top_p = _effective_call_params(
            mode,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        sampling = self._SamplingParams(
            temperature=float(effective_temperature),
            top_p=float(effective_top_p),
            max_tokens=int(effective_max_tokens),
        )

        before_files = set(glob.glob(os.path.join(self._hidden_dir, "*.safetensors")))
        started = time.perf_counter()
        outputs = self.llm.generate([prompt], sampling_params=sampling, use_tqdm=False)
        request_s = time.perf_counter() - started

        if not outputs:
            raise RuntimeError("vLLM generate returned no outputs.")
        out0 = outputs[0]
        gen0 = out0.outputs[0] if getattr(out0, "outputs", None) else None
        if gen0 is None:
            raise RuntimeError("vLLM output had no completion entries.")

        hidden_path = None
        kv = getattr(out0, "kv_transfer_params", None)
        if isinstance(kv, dict):
            p = kv.get("hidden_states_path")
            if isinstance(p, str) and p:
                hidden_path = p
        if hidden_path is None:
            hidden_path = self._fallback_hidden_file(before_files)
        if hidden_path is None:
            raise RuntimeError("vLLM output missing hidden_states_path.")

        hs_shape, hidden_s, hidden_norm, hidden_mean = _load_hidden(
            hidden_path,
            target_layer=target_layer,
            expected_num_layers=self._expected_num_layers,
            delete_after_read=True,
        )
        if hs_shape is None:
            raise RuntimeError(f"Failed to load hidden states file: {hidden_path}")

        usage = {
            "prompt_tokens": len(getattr(out0, "prompt_token_ids", []) or []),
            "completion_tokens": len(getattr(gen0, "token_ids", []) or []),
            "total_tokens": (
                len(getattr(out0, "prompt_token_ids", []) or [])
                + len(getattr(gen0, "token_ids", []) or [])
            ),
        }

        return {
            "request_s": request_s,
            "total_s": request_s + (hidden_s or 0.0),
            "text": str(getattr(gen0, "text", "") or ""),
            "usage": usage,
            "hidden": {
                "shape": hs_shape,
                "hidden_extract_s": hidden_s,
                "hidden_norm": hidden_norm,
                "hidden_mean": hidden_mean,
            },
        }


def _call_openai_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    api_key: str,
) -> Dict[str, object]:
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlrequest.Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    with urlrequest.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - started
    parsed = json.loads(raw)
    return {
        "request_s": elapsed,
        "raw": parsed,
        "text": _extract_openai_text(parsed),
        "usage": _extract_openai_usage(parsed),
    }


def _call_gemini(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
) -> Dict[str, object]:
    url = f"{base_url.rstrip('/')}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
            "topP": float(top_p),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urlrequest.Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    with urlrequest.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - started
    parsed = json.loads(raw)
    return {
        "request_s": elapsed,
        "raw": parsed,
        "text": _extract_gemini_text(parsed),
        "usage": _extract_gemini_usage(parsed),
    }


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for r in records:
        key = (str(r.get("model", "")), str(r.get("variant", "")))
        buckets.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for (model, variant), rows in sorted(buckets.items()):
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        req_vals = [_safe_float(r.get("request_s")) for r in ok_rows]
        req_vals = [x for x in req_vals if x is not None]
        hid_vals = [_safe_float(r.get("hidden_load_s")) for r in ok_rows]
        hid_vals = [x for x in hid_vals if x is not None]
        tot_vals = [_safe_float(r.get("total_with_hidden_s")) for r in ok_rows]
        tot_vals = [x for x in tot_vals if x is not None]
        comp_toks = [
            _safe_int((r.get("usage") or {}).get("completion_tokens")) for r in ok_rows
        ]
        comp_toks = [x for x in comp_toks if x is not None]
        prompt_toks = [
            _safe_int((r.get("usage") or {}).get("prompt_tokens")) for r in ok_rows
        ]
        prompt_toks = [x for x in prompt_toks if x is not None]
        total_toks = [
            _safe_int((r.get("usage") or {}).get("total_tokens")) for r in ok_rows
        ]
        total_toks = [x for x in total_toks if x is not None]
        req_total_s = sum(req_vals) if req_vals else None
        tot_total_s = sum(tot_vals) if tot_vals else None
        req_rps = (
            (len(req_vals) / req_total_s) if req_total_s and req_total_s > 0 else None
        )
        tot_rps = (
            (len(tot_vals) / tot_total_s) if tot_total_s and tot_total_s > 0 else None
        )

        out.append(
            {
                "model": model,
                "variant": variant,
                "calls": len(rows),
                "ok_calls": len(ok_rows),
                "error_calls": len(rows) - len(ok_rows),
                "avg_request_latency_seconds": (
                    statistics.mean(req_vals) if req_vals else None
                ),
                "avg_requests_per_second": req_rps,
                "avg_end_to_end_latency_seconds": (
                    statistics.mean(tot_vals) if tot_vals else None
                ),
                "avg_end_to_end_responses_per_second": tot_rps,
                "avg_hidden_file_load_seconds": (
                    statistics.mean(hid_vals) if hid_vals else None
                ),
                "avg_completion_tokens": (
                    statistics.mean(comp_toks) if comp_toks else None
                ),
                "avg_prompt_tokens": (
                    statistics.mean(prompt_toks) if prompt_toks else None
                ),
                "avg_total_tokens": (
                    statistics.mean(total_toks) if total_toks else None
                ),
            }
        )
    return out


def _write_markdown(
    out_path: Path,
    *,
    run_meta: Dict[str, object],
    summary: List[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# Future Imagination Latency Benchmark")
    lines.append("")
    lines.append("## Run Meta")
    lines.append("")
    for k, v in run_meta.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| model | prompt_variant | total_calls | successful_calls | error_calls | average_request_latency_seconds | average_request_calls_per_second | average_hidden_ready_latency_seconds | average_hidden_ready_calls_per_second | average_prompt_tokens | average_completion_tokens | average_total_tokens |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in summary:
        lines.append(
            "| {model} | {variant} | {calls} | {ok_calls} | {error_calls} | {avg_request_latency_seconds} | {avg_requests_per_second} | {avg_end_to_end_latency_seconds} | {avg_end_to_end_responses_per_second} | {avg_prompt_tokens} | {avg_completion_tokens} | {avg_total_tokens} |".format(
                **{
                    **row,
                    "avg_request_latency_seconds": (
                        f"{row['avg_request_latency_seconds']:.4f}"
                        if row.get("avg_request_latency_seconds") is not None
                        else ""
                    ),
                    "avg_requests_per_second": (
                        f"{row['avg_requests_per_second']:.4f}"
                        if row.get("avg_requests_per_second") is not None
                        else ""
                    ),
                    "avg_end_to_end_latency_seconds": (
                        f"{row['avg_end_to_end_latency_seconds']:.4f}"
                        if row.get("avg_end_to_end_latency_seconds") is not None
                        else ""
                    ),
                    "avg_end_to_end_responses_per_second": (
                        f"{row['avg_end_to_end_responses_per_second']:.4f}"
                        if row.get("avg_end_to_end_responses_per_second") is not None
                        else ""
                    ),
                    "avg_completion_tokens": (
                        f"{row['avg_completion_tokens']:.1f}"
                        if row.get("avg_completion_tokens") is not None
                        else ""
                    ),
                    "avg_prompt_tokens": (
                        f"{row['avg_prompt_tokens']:.1f}"
                        if row.get("avg_prompt_tokens") is not None
                        else ""
                    ),
                    "avg_total_tokens": (
                        f"{row['avg_total_tokens']:.1f}"
                        if row.get("avg_total_tokens") is not None
                        else ""
                    ),
                }
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _warmup_subset(
    prompts: List[PromptItem],
    warmup_prompts_per_variant: int,
) -> List[PromptItem]:
    if warmup_prompts_per_variant <= 0:
        return []
    by_variant: Dict[str, List[PromptItem]] = {}
    for item in prompts:
        by_variant.setdefault(item.variant, []).append(item)
    warmup: List[PromptItem] = []
    for variant in sorted(by_variant):
        warmup.extend(by_variant[variant][:warmup_prompts_per_variant])
    return warmup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark future-imagination prompts for a single provider/model."
    )
    parser.add_argument(
        "--provider",
        choices=["openai_compatible", "gemini", "hf_local", "vllm_offline"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--prompt-dir-state", required=True)
    parser.add_argument("--prompt-dir-history", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--hidden-states-path", default="")
    parser.add_argument("--hidden-target-layer", type=int, default=-1)
    parser.add_argument("--delete-hidden-after-read", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--min-request-interval-s", type=float, default=0.0)
    parser.add_argument(
        "--benchmark-mode",
        choices=["generate_and_hidden", "hidden_only"],
        default="generate_and_hidden",
        help=(
            "generate_and_hidden: generation latency plus hidden extraction. "
            "hidden_only: fastest hidden-state path; forces minimal generation on "
            "API/vLLM backends and skips generation entirely for HF."
        ),
    )
    parser.add_argument(
        "--warmup-prompts-per-variant",
        type=int,
        default=0,
        help="Number of prompt calls to warm up per variant before measurement.",
    )
    parser.add_argument(
        "--gemini-base-url",
        default=DEFAULT_GEMINI_BASE_URL,
    )
    parser.add_argument(
        "--hf-model-id",
        default="",
        help="HF model id/path for --provider hf_local (defaults to --model).",
    )
    parser.add_argument(
        "--hf-device-map",
        default="auto",
        help="transformers device_map for --provider hf_local (e.g., auto, cuda:0, none).",
    )
    parser.add_argument(
        "--hf-dtype",
        default="bfloat16",
        help="Torch dtype for --provider hf_local: auto|bfloat16|float16|float32.",
    )
    parser.add_argument(
        "--hf-attn-implementation",
        default="",
        help="Optional transformers attn_implementation for --provider hf_local.",
    )
    parser.add_argument(
        "--hf-trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for --provider hf_local.",
    )
    parser.add_argument(
        "--vllm-model-id",
        default="",
        help="Model id/path for --provider vllm_offline (defaults to --model).",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor-parallel size for --provider vllm_offline.",
    )
    parser.add_argument(
        "--vllm-dtype",
        default="bfloat16",
        help="vLLM dtype for --provider vllm_offline.",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=8192,
        help="vLLM max_model_len for --provider vllm_offline.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM gpu_memory_utilization for --provider vllm_offline.",
    )
    parser.add_argument(
        "--vllm-trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for --provider vllm_offline.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Enable enforce_eager for --provider vllm_offline.",
    )
    parser.add_argument(
        "--vllm-spec-num-tokens",
        type=int,
        default=1,
        help="Speculative token count for extract_hidden_states in --provider vllm_offline.",
    )
    parser.add_argument(
        "--vllm-enable-prefix-caching",
        action="store_true",
        help=(
            "Enable prefix caching for --provider vllm_offline. "
            "Default is disabled for connector stability."
        ),
    )
    parser.add_argument(
        "--vllm-enable-chunked-prefill",
        action="store_true",
        help=(
            "Enable chunked prefill for --provider vllm_offline. "
            "Default is disabled for connector stability."
        ),
    )
    parser.add_argument(
        "--vllm-enable-hybrid-kv-cache-manager",
        action="store_true",
        help=(
            "Enable hybrid KV cache manager for --provider vllm_offline. "
            "For the patched Qwen3.5 offline connector path this should be "
            "enabled; without the companion runtime patches, engine startup "
            "can still fail either at the HMA support gate or at hybrid-KV "
            "cache unification."
        ),
    )
    parser.add_argument(
        "--vllm-kv-connector-module-path",
        default="",
        help=(
            "Optional Python module path providing ExampleHiddenStatesConnector "
            "for --provider vllm_offline."
        ),
    )
    parser.add_argument(
        "--vllm-kv-connector-mode",
        default="last_token",
        choices=["all", "last_token"],
        help=(
            "Connector export mode for --provider vllm_offline. "
            "Use last_token for the fastest hidden extraction path."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key
    if not api_key and args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")

    prompts = []
    prompts.extend(
        _load_prompt_variant("predict_state_only", args.prompt_dir_state, args.repeats)
    )
    prompts.extend(
        _load_prompt_variant("predict_history_k5", args.prompt_dir_history, args.repeats)
    )
    prompts.sort(key=lambda x: (x.variant, x.timestep, x.repeat_idx))
    warmup_prompts = _warmup_subset(prompts, args.warmup_prompts_per_variant)

    hf_runner: Optional[_HFLocalRunner] = None
    vllm_runner: Optional[_VLLMOfflineRunner] = None
    if args.provider == "hf_local":
        hf_model_id = args.hf_model_id or args.model
        hf_runner = _HFLocalRunner(
            model_id=hf_model_id,
            device_map=args.hf_device_map,
            dtype_name=args.hf_dtype,
            trust_remote_code=bool(args.hf_trust_remote_code),
            attn_implementation=args.hf_attn_implementation,
        )
    elif args.provider == "vllm_offline":
        vllm_model_id = args.vllm_model_id or args.model
        vllm_runner = _VLLMOfflineRunner(
            model_id=vllm_model_id,
            target_layer=args.hidden_target_layer,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            dtype_name=args.vllm_dtype,
            trust_remote_code=bool(args.vllm_trust_remote_code),
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            enforce_eager=bool(args.vllm_enforce_eager),
            spec_num_tokens=args.vllm_spec_num_tokens,
            enable_prefix_caching=bool(args.vllm_enable_prefix_caching),
            enable_chunked_prefill=bool(args.vllm_enable_chunked_prefill),
            enable_hybrid_kv_cache_manager=bool(
                args.vllm_enable_hybrid_kv_cache_manager
            ),
            kv_connector_module_path=args.vllm_kv_connector_module_path,
            kv_connector_mode=args.vllm_kv_connector_mode,
        )

    effective_max_tokens, effective_temperature, effective_top_p = _effective_call_params(
        args.benchmark_mode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for item in warmup_prompts:
        try:
            if args.provider == "gemini":
                if not api_key:
                    raise RuntimeError(
                        "Gemini provider selected but no API key set "
                        f"(use --api-key or --api-key-env)."
                    )
                _call_gemini(
                    base_url=args.gemini_base_url,
                    model=args.model,
                    api_key=api_key,
                    prompt=item.prompt_text,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    timeout_s=args.timeout_s,
                )
            elif args.provider == "hf_local":
                assert hf_runner is not None
                hf_runner.call(
                    prompt=item.prompt_text,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    target_layer=args.hidden_target_layer,
                    benchmark_mode=args.benchmark_mode,
                )
            elif args.provider == "vllm_offline":
                assert vllm_runner is not None
                vllm_runner.call(
                    prompt=item.prompt_text,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    target_layer=args.hidden_target_layer,
                    benchmark_mode=args.benchmark_mode,
                )
            else:
                _call_openai_completion(
                    base_url=args.base_url,
                    model=args.model,
                    api_key=api_key,
                    prompt=item.prompt_text,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    timeout_s=args.timeout_s,
                )
        except Exception as exc:  # pragma: no cover
            print(
                "warmup_warning="
                f"{args.provider}:{item.variant}:t={item.timestep}:{exc.__class__.__name__}:{exc}"
            )

    records: List[Dict[str, object]] = []
    for i, item in enumerate(prompts, start=1):
        record: Dict[str, object] = {
            "idx": i,
            "provider": args.provider,
            "model": args.model,
            "benchmark_mode": args.benchmark_mode,
            "variant": item.variant,
            "timestep": item.timestep,
            "repeat_idx": item.repeat_idx,
            "prompt_path": item.prompt_path,
            "prompt_chars": len(item.prompt_text),
            "status": "ok",
            "error": "",
            "request_s": None,
            "hidden_load_s": None,
            "total_with_hidden_s": None,
            "hidden_shape": None,
            "hidden_norm": None,
            "hidden_mean": None,
            "response_chars": 0,
            "usage": {},
        }

        try:
            if args.provider == "gemini":
                if not api_key:
                    raise RuntimeError(
                        "Gemini provider selected but no API key set "
                        f"(use --api-key or --api-key-env)."
                    )
                call = _call_gemini(
                    base_url=args.gemini_base_url,
                    model=args.model,
                    api_key=api_key,
                    prompt=item.prompt_text,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    timeout_s=args.timeout_s,
                )
            elif args.provider == "hf_local":
                assert hf_runner is not None
                call = hf_runner.call(
                    prompt=item.prompt_text,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    target_layer=args.hidden_target_layer,
                    benchmark_mode=args.benchmark_mode,
                )
            elif args.provider == "vllm_offline":
                assert vllm_runner is not None
                call = vllm_runner.call(
                    prompt=item.prompt_text,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    target_layer=args.hidden_target_layer,
                    benchmark_mode=args.benchmark_mode,
                )
            else:
                call = _call_openai_completion(
                    base_url=args.base_url,
                    model=args.model,
                    api_key=api_key,
                    prompt=item.prompt_text,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    top_p=effective_top_p,
                    timeout_s=args.timeout_s,
                )

            response_text = str(call.get("text", ""))
            usage = call.get("usage", {}) or {}

            record["request_s"] = round(float(call["request_s"]), 6)
            record["response_chars"] = len(response_text)
            record["usage"] = usage

            # Hidden-state load path for vLLM/openai compatible.
            hidden_s = None
            if args.provider in {"hf_local", "vllm_offline"}:
                hidden = call.get("hidden", {})
                if isinstance(hidden, dict):
                    record["hidden_shape"] = hidden.get("shape")
                    record["hidden_norm"] = _safe_float(hidden.get("hidden_norm"))
                    record["hidden_mean"] = _safe_float(hidden.get("hidden_mean"))
                    hidden_s = _safe_float(hidden.get("hidden_extract_s"))
            elif args.provider == "openai_compatible" and args.hidden_states_path:
                hidden_path = _extract_hidden_path(
                    call.get("raw", {}) if isinstance(call.get("raw"), dict) else {},
                    args.hidden_states_path,
                )
                if hidden_path:
                    shape, hidden_s, hidden_norm, hidden_mean = _load_hidden(
                        hidden_path,
                        target_layer=args.hidden_target_layer,
                        expected_num_layers=None,
                        delete_after_read=args.delete_hidden_after_read,
                    )
                    record["hidden_shape"] = shape
                    record["hidden_norm"] = hidden_norm
                    record["hidden_mean"] = hidden_mean
                else:
                    record["error"] = "hidden_path_missing"

            record["hidden_load_s"] = (
                round(float(hidden_s), 6) if hidden_s is not None else None
            )
            req_s = _safe_float(record["request_s"])
            total_s = _safe_float(call.get("total_s"))
            if total_s is not None:
                record["total_with_hidden_s"] = round(total_s, 6)
            elif req_s is not None:
                if hidden_s is not None:
                    record["total_with_hidden_s"] = round(req_s + hidden_s, 6)
                else:
                    record["total_with_hidden_s"] = round(req_s, 6)

        except urlerror.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = str(exc)
            record["status"] = "error"
            record["error"] = f"HTTPError {exc.code}: {body}"
        except Exception as exc:  # pragma: no cover
            record["status"] = "error"
            record["error"] = f"{exc.__class__.__name__}: {exc}"

        records.append(record)
        if args.min_request_interval_s > 0:
            time.sleep(args.min_request_interval_s)

    summary = _summarize(records)
    meta = {
        "provider": args.provider,
        "model": args.model,
        "run_label": args.run_label,
        "num_prompts": len(prompts),
        "warmup_prompts_per_variant": args.warmup_prompts_per_variant,
        "benchmark_mode": args.benchmark_mode,
        "max_tokens": args.max_tokens,
        "effective_max_tokens": effective_max_tokens,
        "temperature": args.temperature,
        "effective_temperature": effective_temperature,
        "top_p": args.top_p,
        "effective_top_p": effective_top_p,
        "repeats": args.repeats,
        "prompt_dir_state": args.prompt_dir_state,
        "prompt_dir_history": args.prompt_dir_history,
        "hidden_states_path": args.hidden_states_path,
        "hidden_target_layer": args.hidden_target_layer,
        "hf_model_id": (args.hf_model_id or args.model) if args.provider == "hf_local" else "",
        "hf_device_map": args.hf_device_map if args.provider == "hf_local" else "",
        "hf_dtype": args.hf_dtype if args.provider == "hf_local" else "",
        "hf_attn_implementation": (
            args.hf_attn_implementation if args.provider == "hf_local" else ""
        ),
        "hf_trust_remote_code": bool(args.hf_trust_remote_code) if args.provider == "hf_local" else "",
        "vllm_model_id": (args.vllm_model_id or args.model) if args.provider == "vllm_offline" else "",
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size if args.provider == "vllm_offline" else "",
        "vllm_dtype": args.vllm_dtype if args.provider == "vllm_offline" else "",
        "vllm_max_model_len": args.vllm_max_model_len if args.provider == "vllm_offline" else "",
        "vllm_gpu_memory_utilization": (
            args.vllm_gpu_memory_utilization if args.provider == "vllm_offline" else ""
        ),
        "vllm_trust_remote_code": (
            bool(args.vllm_trust_remote_code) if args.provider == "vllm_offline" else ""
        ),
        "vllm_enforce_eager": (
            bool(args.vllm_enforce_eager) if args.provider == "vllm_offline" else ""
        ),
        "vllm_spec_num_tokens": args.vllm_spec_num_tokens if args.provider == "vllm_offline" else "",
        "vllm_enable_prefix_caching": (
            bool(args.vllm_enable_prefix_caching) if args.provider == "vllm_offline" else ""
        ),
        "vllm_enable_chunked_prefill": (
            bool(args.vllm_enable_chunked_prefill) if args.provider == "vllm_offline" else ""
        ),
        "vllm_enable_hybrid_kv_cache_manager": (
            bool(args.vllm_enable_hybrid_kv_cache_manager)
            if args.provider == "vllm_offline"
            else ""
        ),
        "vllm_kv_connector_module_path": (
            args.vllm_kv_connector_module_path if args.provider == "vllm_offline" else ""
        ),
        "vllm_kv_connector_mode": (
            args.vllm_kv_connector_mode if args.provider == "vllm_offline" else ""
        ),
    }

    _write_jsonl(out_dir / "records.jsonl", records)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    _write_markdown(out_dir / "summary.md", run_meta=meta, summary=summary)

    print(f"records_jsonl={out_dir / 'records.jsonl'}")
    print(f"summary_json={out_dir / 'summary.json'}")
    print(f"summary_md={out_dir / 'summary.md'}")
    print(f"calls={len(records)}")
    print(f"ok={sum(1 for r in records if r.get('status') == 'ok')}")
    print(f"errors={sum(1 for r in records if r.get('status') != 'ok')}")


if __name__ == "__main__":
    main()
