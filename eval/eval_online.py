#!/usr/bin/env python3
"""
Online evaluation of the imagination-augmented AWR policy.

Each step:
  1. obs_to_text(obs) → filter_text_obs → Gemini prediction prompt
  2. Gemini 2.5 Flash generates a 15-step future narrative (every 15 steps)
  3. Qwen3-8B embeds the narrative → 4096-dim hidden state
  4. Policy(obs, hidden) → action

Logs: Gemini responses, video recordings, per-step rewards, achievements,
      hidden state stats, value predictions.

Usage:
    python -m pipeline.eval_online --num-episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import jax
import torch

from labelling.obs_to_text import obs_to_text
from llm.prompts import filter_text_obs
from models.actor_critic_aug import (
    ActorCriticAug as ActorCriticAugBase,
    ActorCriticAugLN,
    ActorCriticAugV2,
    ActorCriticAugGated,
    ActorCriticHiddenOnly,
)

from pipeline.config import (
    ACTION_NAMES,
    EMBED_HIDDEN_DIM,
    EMBED_LAYER,
    EMBED_MODEL,
    GEMINI_BASE_URL,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
)
from pipeline.embed import extract_prediction_suffix

# --- Constants ---
CKPT_DIR = Path("/data/group_data/rl/geney/checkpoints/awr_imagination")
PREDICT_TEMPLATE_PATH = (
    Path.home()
    / "Craftax_Baselines/configs/future_imagination/templates/predict_state_only_prompt_concise.txt"
)
STEP_CADENCE = 5  # matches prompt's 5-step forecast horizon
ACTION_DIM = 43
OBS_DIM = 8268
DEFAULT_LAYER_WIDTH = 512

# Few-shot examples from real Gemini oracle responses (training distribution).
# These are actual outputs from Phase 4 (gemini_label.py) so the predict-only
# model produces in-distribution text at inference time.
CONSTANT_EMBED_TEXT = "Nothing happens. The player stands still and observes the environment."

ADVERSARIAL_PROMPT_SUFFIX = """

IMPORTANT OVERRIDE: Instead of predicting what the player WILL do, predict a future
where the player makes the WORST possible decisions. The player should waste resources,
craft useless items, ignore threats, walk into lava, let hunger/thirst deplete,
and generally play as poorly as possible. Make the future plausible but terrible."""

DIE_PROMPT_SUFFIX = """

IMPORTANT OVERRIDE: Instead of predicting what the player WILL do, predict a future
where the player actively tries to DIE as quickly as possible. The player should
seek out enemies, walk into lava or water when thirsty/hungry, refuse to eat or drink,
engage monsters without weapons, descend floors without preparation, and take every
possible action that leads to rapid death."""

# v2: full-template swap instead of suffix. The v2 templates replace the
# "Here is a good algorithm…" section with an alternative algorithm.
# Same voice, same sections, no "Instead of" / "override" phrasing, no
# bulleted Worst Possible Future structure.
# Two flavors per mode: concise-base (for Track A / Track C inference) and
# thinking-base (for Track B inference). eval_online.py picks the matching
# flavor based on the caller's --prompt-template-path basename.
# See journals/log_2026-04-22.md and log_2026-04-23.md for probe justification.
_TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent / "configs/training/templates"
)
_V2_TEMPLATES = {
    "die_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_die_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_die_v2.txt",
    },
    "adversarial_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_adversarial_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_adversarial_v2.txt",
    },
    "avoid_water_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_avoid_water_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_avoid_water_v2.txt",
    },
    "avoid_animals_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_avoid_animals_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_avoid_animals_v2.txt",
    },
    # Positive achievement steering — thinking variants now exist for the 5
    # priority modes (collect_stone, descend, eat_cow, direction_left/right);
    # drink_water/place_stone/up/down keep concise-only fallback.
    "target_collect_stone_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_collect_stone_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_target_collect_stone_v2.txt",
    },
    "target_descend_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_descend_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_target_descend_v2.txt",
    },
    "target_eat_cow_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_eat_cow_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_target_eat_cow_v2.txt",
    },
    "target_drink_water_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_drink_water_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_drink_water_v2.txt",
    },
    "target_place_stone_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_place_stone_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_place_stone_v2.txt",
    },
    "target_hunt_animals_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_hunt_animals_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_target_hunt_animals_v2.txt",
    },
    # Patch-by-prompt experiments (better baseline algorithms, not steering-away
    # variants). Same shape as base concise prompt but with explicit checklists.
    "v2_basic_coverage": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_v2_basic_coverage.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_v2_basic_coverage.txt",
    },
    "v2_long_tail": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_v2_long_tail.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_v2_long_tail.txt",
    },
    # Pure direction steering
    "direction_left_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_direction_left_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_direction_left_v2.txt",
    },
    "direction_right_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_direction_right_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_only_thinking_prompt_direction_right_v2.txt",
    },
    "direction_up_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_direction_up_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_direction_up_v2.txt",
    },
    "direction_down_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_direction_down_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_direction_down_v2.txt",
    },
    # Replacements for die_v2 / adversarial_v2 — phrased positively as
    # "move toward enemies (die_fast)" and "stay alive long via survival loop".
    "die_fast_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_die_fast_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_die_fast_v2.txt",
    },
    "survive_long_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_survive_long_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_survive_long_v2.txt",
    },
    "survive_long_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_survive_long_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_survive_long_v3.txt",
    },
    "achievement_max_v1": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v1.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v1.txt",
    },
    "achievement_max_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v2.txt",
    },
    "achievement_max_v2_thresh6": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v2_thresh6.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v2_thresh6.txt",
    },
    "explore_ood_v1": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_explore_ood_v1.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_explore_ood_v1.txt",
    },
    "v2_eat_bat": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_v2_eat_bat.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_v2_eat_bat.txt",
    },
    "v2_enter_mines": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_v2_enter_mines.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_v2_enter_mines.txt",
    },
    "achievement_max_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v3.txt",
    },
    "achievement_max_v4": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v4.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_achievement_max_v4.txt",
    },
    # v3 iterations of the NULL specificity-matrix cells
    "target_eat_cow_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_eat_cow_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_eat_cow_v3.txt",
    },
    "target_drink_water_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_drink_water_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_drink_water_v3.txt",
    },
    "target_stay_overworld_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_stay_overworld_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_stay_overworld_v3.txt",
    },
    "target_place_plant_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_place_plant_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_place_plant_v3.txt",
    },
    "target_defeat_zombie_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_defeat_zombie_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_defeat_zombie_v3.txt",
    },
    "target_collect_sapling_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_collect_sapling_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_collect_sapling_v3.txt",
    },
    # Specificity matrix: rare-behavior upward steering + opposite-direction
    # mirrors. All concise-only fallback (no thinking variants yet).
    "target_collect_diamond_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_collect_diamond_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_collect_diamond_v2.txt",
    },
    "target_make_iron_pickaxe_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_make_iron_pickaxe_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_make_iron_pickaxe_v2.txt",
    },
    "target_collect_sapling_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_collect_sapling_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_collect_sapling_v2.txt",
    },
    "target_defeat_zombie_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_defeat_zombie_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_defeat_zombie_v2.txt",
    },
    "target_place_plant_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_place_plant_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_place_plant_v2.txt",
    },
    "target_avoid_stone_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_avoid_stone_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_avoid_stone_v2.txt",
    },
    "target_avoid_stone_v3": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_avoid_stone_v3.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_avoid_stone_v3.txt",
    },
    "target_stay_overworld_v2": {
        "concise":  _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_stay_overworld_v2.txt",
        "thinking": _TEMPLATE_DIR / "predict_state_only_prompt_concise_target_stay_overworld_v2.txt",
    },
}


def _pick_v2_template(base_template_path: Path, mode: str) -> Path:
    """Pick thinking- or concise-flavored v2 template to match caller's base."""
    base = base_template_path.name
    flavor = "thinking" if "thinking" in base else "concise"
    if mode not in _V2_TEMPLATES:
        raise ValueError(f"unexpected v2 mode: {mode}")
    return _V2_TEMPLATES[mode][flavor]


FEW_SHOT_EXAMPLES = [
    {
        "label": "A",
        "response": """Headline: Crafting a wood pickaxe and gathering stone on floor 0.

Meaningful events (ordered):
1. [t+1 to t+2] The player moves right, chops a tree (Wood: 3->4), then places a crafting table (Wood: 4->2).
2. [t+3 to t+4] The player crafts a wood pickaxe (Wood: 2->1) and moves one step south.
3. [t+5 to t+15] The player moves further south, mining stone three times (Stone: 0->3). Drink and Food decrease slightly (Drink: 9->8, Food: 9->8).

Trajectory summary:
The player prioritizes early game progression by crafting a wood pickaxe on floor 0. After crafting, the focus shifts to gathering stone, likely for further tool upgrades.""",
    },
    {
        "label": "B",
        "response": """Headline: Resource gathering and drink replenishment on floor 0.

Meaningful events (ordered):
1. [t+1 to t+2] The player moves right and down, recovering 1 health (Health: 3->4).
2. [t+3 to t+5] The player mines a stone tile (Stone: 2->3) and then an iron tile (Iron: 0->1).
3. [t+6 to t+12] The player moves up and mines four more stone tiles (Stone: 3->7).
4. [t+13 to t+15] The player moves to a water tile and drinks twice (Drink: 1->3).

Trajectory summary:
The player prioritizes gathering resources, mining stone and iron, and then replenishes their drink before continuing.""",
    },
    {
        "label": "C",
        "response": """Headline: Crafting a stone sword and moving towards the ladder on floor 0.

Meaningful events (ordered):
1. [t+1 to t+6] The player moves north, then west, chopping a tree (Wood: 3->4). Food and Energy decrease slightly (Food: 9->8, Energy: 8->7).
2. [t+7 to t+8] The player places a crafting table and crafts a stone sword (Wood: 4->2).
3. [t+9 to t+15] The player moves south towards the open ladder, with Drink decreasing (Drink: 3->2).

Trajectory summary:
The player prioritizes crafting a stone sword on the first floor, gathering wood to do so, then immediately begins moving towards the open ladder to descend to the next floor.""",
    },
]


# ======================================================================
# Gemini API
# ======================================================================
def call_gemini(prompt: str, api_key: str, model: str = GEMINI_MODEL,
                use_thinking: bool = True,
                thinking_budget: int = None,
                max_output_tokens: int = None) -> dict:
    """Call Gemini and return text + token counts.

    thinking_budget overrides use_thinking when not None (same semantics as
    pipeline.gemini_label.call_gemini): 0 disables thinking, N>0 caps, -1
    dynamic. max_output_tokens overrides the default when set (thinking models
    need a larger budget since thoughts count against output).
    """
    from urllib import request as urlrequest, error as urlerror

    url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
    _max_out = max_output_tokens if max_output_tokens is not None else GEMINI_MAX_OUTPUT_TOKENS
    gen_config = {
        "maxOutputTokens": _max_out,
        "temperature": GEMINI_TEMPERATURE,
    }
    if thinking_budget is not None:
        gen_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    elif use_thinking:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    t0 = time.perf_counter()
    req = urlrequest.Request(url, data=body, headers=headers, method="POST")
    with urlrequest.urlopen(req, timeout=60.0) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)

    candidates = parsed.get("candidates", [])
    text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))

    usage = parsed.get("usageMetadata", {})
    return {
        "text": text,
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
        "latency_s": time.perf_counter() - t0,
    }


# ======================================================================
# Embedding backends
# ======================================================================
class QwenEmbedder:
    """Loads Qwen3-8B truncated to 31 layers for layer-30 mean-pool extraction."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        print("Loading Qwen3-8B (31 layers, SDPA)...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        config.num_hidden_layers = EMBED_LAYER + 1  # 31 layers: 0..30
        self.model = AutoModelForCausalLM.from_pretrained(
            EMBED_MODEL, config=config, dtype=torch.float16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.device = device
        print(f"  Loaded in {time.time() - t0:.1f}s ({config.num_hidden_layers} layers)")

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text → (4096,) float32."""
        enc = self.tokenizer(
            [text], return_tensors="pt", truncation=True,
            max_length=2048, padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        out = self.model.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hs = out.last_hidden_state  # (1, seq, 4096)

        mask = attention_mask.unsqueeze(-1).float()
        mean_pooled = ((last_hs.float() * mask).sum(1) / mask.sum(1).clamp(min=1))
        return mean_pooled[0].cpu().numpy().astype(np.float32)


class Qwen3EmbedEmbedder:
    """Qwen3-Embedding model with last-token pooling. Produces 4096-dim vectors."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        MODEL_ID = "Qwen/Qwen3-Embedding-8B"
        print(f"Loading {MODEL_ID} (last-token pooling)...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModel.from_pretrained(
            MODEL_ID, dtype=torch.float16,
            attn_implementation="sdpa", trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.device = device
        self.hidden_dim = self.model.config.hidden_size
        print(f"  Loaded in {time.time() - t0:.1f}s, hidden_dim={self.hidden_dim}")

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text → (hidden_dim,) float32 via last-token pooling."""
        enc = self.tokenizer(
            [text], return_tensors="pt", truncation=True,
            max_length=2048, padding=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Last non-padding token
        seq_len = attention_mask.sum(dim=1) - 1
        last_hs = out.last_hidden_state[0, seq_len[0]]  # (H,)
        return last_hs.float().cpu().numpy()


class GeminiEmbedder:
    """Calls Gemini text-embedding-004 API for each text. Output dim configurable."""

    def __init__(self, api_key: str, output_dim: int = 3072):
        import os
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GeminiEmbedder requires api_key or GEMINI_API_KEY env var")
        self.output_dim = output_dim
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models"
            f"/gemini-embedding-001:embedContent?key={self.api_key}"
        )
        print(f"GeminiEmbedder: gemini-embedding-001, output_dim={output_dim}")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text → (output_dim,) float32."""
        import json
        from urllib import request as urlrequest, error as urlerror

        payload = json.dumps({
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.output_dim,
        }).encode()
        for attempt in range(4):
            try:
                req = urlrequest.Request(
                    self.url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlrequest.urlopen(req, timeout=30) as resp:
                    d = json.loads(resp.read())
                return np.array(d["embedding"]["values"], dtype=np.float32)
            except urlerror.HTTPError as e:
                if e.code == 429 and attempt < 3:
                    import time as _t; _t.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Gemini embed HTTP {e.code}: {e.read()[:200]}")
        raise RuntimeError("GeminiEmbedder: max retries exceeded")


def make_embedder(backend: str, device: str, api_key: str = "", output_dim: int = 3072):
    """Factory: returns the right embedder for the given backend string."""
    if backend == "qwen3_gen":
        return QwenEmbedder(device=device)
    elif backend == "qwen3_embed":
        return Qwen3EmbedEmbedder(device=device)
    elif backend == "gemini_embed":
        return GeminiEmbedder(api_key=api_key, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown embed_backend: {backend!r}")


# ======================================================================
# Video rendering helpers
# ======================================================================
def render_frame(env_state) -> np.ndarray:
    """Render Craftax state to RGB pixel array."""
    from craftax.craftax.renderer import render_craftax_pixels
    pixels = render_craftax_pixels(env_state, block_pixel_size=16, do_night_noise=False)
    return np.array(pixels, dtype=np.uint8)


def make_video_frame(game_frame, values, rewards, gemini_text, step):
    """Compose game frame + value graph + full Gemini text overlay."""
    target_w = 600
    h, w = game_frame.shape[:2]
    scale = target_w / w
    target_h = int(h * scale)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 16
    graph_h = 50
    graph_overhead = 10 + graph_h + 20  # padding + graph + gap

    # Wrap gemini text to fit width (~90 chars per line at font scale 0.35)
    max_chars = 90
    text_lines = []
    if gemini_text:
        for raw_line in gemini_text.split("\n"):
            if not raw_line.strip():
                text_lines.append("")
                continue
            while len(raw_line) > max_chars:
                text_lines.append(raw_line[:max_chars])
                raw_line = raw_line[max_chars:]
            text_lines.append(raw_line)

    text_region_h = max(len(text_lines) * line_h + 20, 100)
    footer_h = graph_overhead + text_region_h

    canvas = np.zeros((target_h + footer_h, target_w, 3), dtype=np.uint8)
    resized = cv2.resize(game_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas[:target_h, :target_w] = resized

    # Value graph
    graph_y = target_h + 10
    cv2.rectangle(canvas, (10, graph_y), (target_w - 10, graph_y + graph_h), (30, 30, 30), -1)
    if len(values) > 1:
        v_min, v_max = min(min(values), 0), max(max(values), 1)
        for i in range(len(values) - 1):
            x1 = int(10 + i / len(values) * (target_w - 20))
            x2 = int(10 + (i + 1) / len(values) * (target_w - 20))
            y1 = int(graph_y + graph_h - (values[i] - v_min) / (v_max - v_min + 1e-8) * graph_h)
            y2 = int(graph_y + graph_h - (values[i + 1] - v_min) / (v_max - v_min + 1e-8) * graph_h)
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.putText(canvas, f"Step {step}  V={values[-1]:.2f}  R={sum(rewards):.2f}",
                (10, graph_y - 3), font, 0.4, (200, 200, 200), 1)

    # Full Gemini text
    text_y = graph_y + graph_h + 20
    for line in text_lines:
        cv2.putText(canvas, line, (10, text_y), font, 0.35, (180, 180, 180), 1)
        text_y += line_h

    return canvas


# ======================================================================
# Main evaluation
# ======================================================================
def run_eval(args):
    device = args.device
    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key and _needs_api_key:
        raise ValueError("Set GEMINI_API_KEY env var or pass --gemini-api-key")

    _gemini_model = args.gemini_model or GEMINI_MODEL
    _use_thinking = (
        _gemini_model.startswith("gemini-2.5")
        or _gemini_model.startswith("gemini-3")
    )
    _embedding_mode = args.embedding_mode
    _v2_modes = tuple(_V2_TEMPLATES.keys())
    _needs_gemini = _embedding_mode in ("gemini", "adversarial", "die", "embed_arith") + _v2_modes
    _needs_api_key = _needs_gemini

    # Load synthetic embedding direction if in embed_arith mode.
    # The direction is added to the regular embedding before normalization:
    #   hidden_modified = hidden_raw + alpha * direction
    arith_direction = None
    arith_alpha = float(getattr(args, "embed_arith_alpha", 0.0))
    if _embedding_mode == "embed_arith":
        if not args.embed_arith_direction:
            raise ValueError("--embedding-mode embed_arith requires --embed-arith-direction <path.npy>")
        arith_direction = np.load(args.embed_arith_direction).astype(np.float32)
        print(f"embed_arith: direction loaded from {args.embed_arith_direction}, "
              f"norm={float(np.linalg.norm(arith_direction)):.4f}, alpha={arith_alpha}")

    # Load prompt template. v2 modes pick their template based on the caller's
    # base template (thinking vs concise); for other modes --prompt-template-path
    # is honored (else the default PREDICT_TEMPLATE_PATH).
    base_template = Path(args.prompt_template_path) if args.prompt_template_path else PREDICT_TEMPLATE_PATH
    if _embedding_mode in _v2_modes:
        template_path = _pick_v2_template(base_template, _embedding_mode)
        print(f"v2 mode: base={base_template.name} -> {template_path.name}")
    else:
        template_path = base_template
    raw_template = template_path.read_text()
    template = raw_template
    print(f"Prompt template: {template_path.name} (with {len(FEW_SHOT_EXAMPLES)} real few-shot examples)")

    # Mid-episode switch: if --switch-mode is set, after --switch-step env steps,
    # the embedding mode (and template) flip to the switch_mode. Both templates
    # are pre-loaded.
    _switch_mode = getattr(args, "switch_mode", None)
    _switch_step = int(getattr(args, "switch_step", 0))
    _switch_template = None
    if _switch_mode:
        if _switch_mode in _v2_modes:
            _switch_template_path = _pick_v2_template(base_template, _switch_mode)
        else:
            _switch_template_path = base_template
        _switch_template = _switch_template_path.read_text()
        print(f"Switch enabled: at step >= {_switch_step}, mode -> {_switch_mode} "
              f"(template {_switch_template_path.name})")
    _initial_mode = _embedding_mode

    # Load policy
    layer_width = args.layer_width
    if args.arch_hidden_only:
        ModelClass = ActorCriticHiddenOnly
    elif args.arch_gated:
        ModelClass = ActorCriticAugGated
    elif args.arch_v2:
        ModelClass = ActorCriticAugV2
    elif args.no_layernorm:
        ModelClass = ActorCriticAugBase
    else:
        ModelClass = ActorCriticAugLN
    _hidden_dim = args.hidden_dim if args.hidden_dim > 0 else EMBED_HIDDEN_DIM
    model_kwargs = dict(obs_dim=OBS_DIM, action_dim=ACTION_DIM, layer_width=layer_width, hidden_state_dim=_hidden_dim)
    if ModelClass != ActorCriticAugBase:
        model_kwargs["dropout"] = args.dropout
    model = ModelClass(**model_kwargs).to(device)
    ckpt_path = args.checkpoint or str(CKPT_DIR / "final.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Policy loaded: {ckpt_path}")

    # Load normalization stats
    stats_path = args.hidden_stats or str(CKPT_DIR / "hidden_state_stats.npz")
    stats = np.load(stats_path)
    hidden_mean = stats["mean"].astype(np.float32)
    hidden_std = stats["std"].astype(np.float32)
    print(f"Hidden stats loaded: mean=[{hidden_mean.min():.1f},{hidden_mean.max():.1f}]")

    # Load embedder (backend selectable)
    embedder = make_embedder(
        backend=args.embed_backend,
        device=device,
        api_key=api_key or "",
        output_dim=args.hidden_dim if args.hidden_dim > 0 else EMBED_HIDDEN_DIM,
    )

    # Init Craftax environment
    import craftax.craftax.envs.craftax_pixels_env as pxmod
    from craftax.craftax_env import make_craftax_env_from_name
    Achievement = pxmod.Achievement

    # Patch achievement logging to always report (not just on done)
    def log_achievements_always(state, done):
        achievements = state.achievements * 100.0
        return {
            f"Achievements/{a.name.lower()}": achievements[a.value]
            for a in Achievement
        }
    pxmod.log_achievements_to_info = log_achievements_always

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    rng = jax.random.PRNGKey(args.seed)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Init wandb ---
    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        try:
            wandb.init(
                project="craftax-offline-awr",
                entity="iris-sobolmark",
                name=args.wandb_name or f"eval-imagination-{time.strftime('%Y%m%d-%H%M%S')}",
                config={
                    "eval_type": "online_imagination",
                    "num_episodes": args.num_episodes,
                    "checkpoint": str(ckpt_path),
                    "seed": args.seed,
                    "step_cadence": STEP_CADENCE,
                    "embed_model": EMBED_MODEL,
                    "embed_layer": EMBED_LAYER,
                    "embed_dim": EMBED_HIDDEN_DIM,
                    "gemini_model": _gemini_model,
                    "gemini_temperature": GEMINI_TEMPERATURE,
                    "predict_template": PREDICT_TEMPLATE_PATH.name,
                    "num_fewshot_examples": len(FEW_SHOT_EXAMPLES),
                    "embedding_mode": _embedding_mode,
                },
                settings=wandb.Settings(init_timeout=600),
            )
            print("wandb initialized")
        except Exception as e:
            print(f"WARNING: wandb.init failed ({type(e).__name__}: {e}); continuing without wandb.")
            use_wandb = False
    elif wandb is None:
        print("WARNING: wandb not installed, logging to files only")

    # Global step counter for wandb x-axis (monotonic across episodes)
    global_step = 0

    # --- Run episodes ---
    all_results = []
    total_gemini_calls = 0
    total_gemini_cost = 0.0

    for ep in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{args.num_episodes}")
        print(f"{'='*60}")

        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, env_params)

        done = False
        step = 0
        ep_return = 0.0
        ep_rewards = []
        ep_values = []
        ep_frames = []
        ep_gemini_log = []
        ep_achievements = {}
        ep_action_counts = np.zeros(ACTION_DIM, dtype=np.int32)
        ep_actions: list[int] = []
        ep_gemini_errors = 0

        # Hidden state (initialized to zero until first Gemini call)
        current_hidden = np.zeros(_hidden_dim, dtype=np.float32)
        current_gemini_text = ""

        while not done and step < 10000:
            obs_np = np.array(obs, dtype=np.float32)

            # --- Every STEP_CADENCE steps: generate hidden state embedding ---
            if step % STEP_CADENCE == 0:
                # Apply mid-episode switch if configured
                if _switch_mode and step >= _switch_step:
                    _embedding_mode = _switch_mode
                    if _switch_template is not None:
                        template = _switch_template
                else:
                    _embedding_mode = _initial_mode
                    template = raw_template
                try:
                    if _embedding_mode == "constant":
                        # Embed a fixed constant string (no Gemini needed)
                        current_gemini_text = CONSTANT_EMBED_TEXT
                        current_hidden = embedder.embed(current_gemini_text)
                        hidden_norm = float(np.linalg.norm(current_hidden))
                        ep_gemini_log.append({
                            "step": step,
                            "gemini_text": current_gemini_text,
                            "prompt_tokens": 0, "completion_tokens": 0,
                            "latency_s": 0.0, "cost_usd": 0.0,
                            "hidden_norm": hidden_norm,
                            "embedding_mode": "constant",
                        })
                        if step == 0:
                            print(f"  [step {step}] Constant embed: {current_gemini_text[:60]}")

                    elif _embedding_mode == "random":
                        # Embed random gibberish text (no Gemini needed)
                        import random as _rng
                        words = ["zyx", "quantum", "banana", "13579", "!!??",
                                 "flurb", "xkcd", "aaaa", "wumpus", "foo",
                                 "splork", "99red", "balloons", "quux", "nope"]
                        random_text = " ".join(_rng.choices(words, k=_rng.randint(20, 60)))
                        current_gemini_text = random_text
                        current_hidden = embedder.embed(current_gemini_text)
                        hidden_norm = float(np.linalg.norm(current_hidden))
                        ep_gemini_log.append({
                            "step": step,
                            "gemini_text": current_gemini_text,
                            "prompt_tokens": 0, "completion_tokens": 0,
                            "latency_s": 0.0, "cost_usd": 0.0,
                            "hidden_norm": hidden_norm,
                            "embedding_mode": "random",
                        })
                        if step == 0 or step % (STEP_CADENCE * 10) == 0:
                            print(f"  [step {step}] Random embed: {random_text[:60]}")

                    else:
                        # Gemini-based modes: gemini, adversarial, die
                        text_obs = obs_to_text(obs_np)
                        filtered = filter_text_obs(text_obs)
                        prompt = template.replace("{current_state_filtered}", filtered)

                        # Oracle future: if template has {future_state_filtered} and
                        # --oracle-future-embed is set, roll the env forward 5 steps
                        # with the trained policy to get the actual t+5 obs and inject
                        # it. This emulates the training-time grounded prompt.
                        if (getattr(args, "oracle_future_embed", False)
                                and "{future_state_filtered}" in prompt):
                            fut_env_state = env_state
                            fut_obs = obs
                            fut_hidden = current_hidden  # use the same hidden as a stand-in
                            fut_rng = rng
                            for _ in range(5):
                                # Pick action with the current policy (greedy)
                                _fut_obs_np = np.asarray(fut_obs, dtype=np.float32)
                                _fut_hidden_normed = (fut_hidden - hidden_mean) / hidden_std
                                _f_obs_t = torch.tensor(_fut_obs_np, dtype=torch.float32, device=device).unsqueeze(0)
                                _f_hid_t = torch.tensor(_fut_hidden_normed, dtype=torch.float32, device=device).unsqueeze(0)
                                with torch.no_grad():
                                    _pi, _ = model(_f_obs_t, _f_hid_t)
                                    _fut_a = int(_pi.probs.argmax(dim=-1).item())
                                fut_rng, _f_step_key = jax.random.split(fut_rng)
                                fut_obs, fut_env_state, _r, _d, _i = env.step(
                                    _f_step_key, fut_env_state, _fut_a, env_params
                                )
                                if bool(_d):
                                    break
                            _fut_text = obs_to_text(np.asarray(fut_obs, dtype=np.float32))
                            _fut_filtered = filter_text_obs(_fut_text)
                            prompt = prompt.replace("{future_state_filtered}", _fut_filtered)

                        # Append adversarial/die suffix if needed
                        if _embedding_mode == "adversarial":
                            prompt += ADVERSARIAL_PROMPT_SUFFIX
                        elif _embedding_mode == "die":
                            prompt += DIE_PROMPT_SUFFIX

                        _tb = getattr(args, "gemini_thinking_budget", None)
                        _mo = 1024 if (_tb is not None and _tb > 0) else None
                        gemini_result = call_gemini(prompt, api_key,
                                                    model=_gemini_model,
                                                    use_thinking=_use_thinking,
                                                    thinking_budget=_tb,
                                                    max_output_tokens=_mo)
                        current_gemini_text = gemini_result["text"]
                        total_gemini_calls += 1
                        cost = (gemini_result["prompt_tokens"] * 0.15e-6
                                + gemini_result["completion_tokens"] * 0.60e-6)
                        total_gemini_cost += cost

                        # Optionally slice to the Prediction: suffix before embedding.
                        text_for_embed = current_gemini_text
                        if getattr(args, "extract_prediction_only", False):
                            suffix, _status = extract_prediction_suffix(current_gemini_text)
                            text_for_embed = suffix

                        # Embed Gemini text
                        current_hidden = embedder.embed(text_for_embed)
                        hidden_norm_pre = float(np.linalg.norm(current_hidden))

                        # Synthetic embedding arithmetic: add alpha * direction
                        # in raw embedding space, before policy normalization.
                        if _embedding_mode == "embed_arith" and arith_direction is not None:
                            current_hidden = current_hidden + arith_alpha * arith_direction
                        hidden_norm = float(np.linalg.norm(current_hidden))

                        ep_gemini_log.append({
                            "step": step,
                            "gemini_text": current_gemini_text,
                            "prompt_tokens": gemini_result["prompt_tokens"],
                            "completion_tokens": gemini_result["completion_tokens"],
                            "latency_s": gemini_result["latency_s"],
                            "cost_usd": cost,
                            "hidden_norm": hidden_norm,
                            "hidden_norm_pre": hidden_norm_pre,
                            "embedding_mode": _embedding_mode,
                            "embed_arith_alpha": arith_alpha if _embedding_mode == "embed_arith" else None,
                        })

                        # wandb: log Gemini call details
                        if use_wandb:
                            headline = current_gemini_text.split("\n")[0] if current_gemini_text else ""
                            wandb.log({
                                "gemini/prompt_tokens": gemini_result["prompt_tokens"],
                                "gemini/completion_tokens": gemini_result["completion_tokens"],
                                "gemini/latency_s": gemini_result["latency_s"],
                                "gemini/cost_usd": cost,
                                "gemini/hidden_norm": hidden_norm,
                                "gemini/response_len": len(current_gemini_text),
                                "gemini/headline": wandb.Html(f"<pre>{headline[:120]}</pre>"),
                            }, step=global_step)

                        if step == 0 or step % (STEP_CADENCE * 5) == 0:
                            headline = current_gemini_text.split("\n")[0] if current_gemini_text else ""
                            print(f"  [step {step}] {_embedding_mode}: {headline[:80]}")

                except Exception as e:
                    ep_gemini_errors += 1
                    print(f"  [step {step}] Embedding error ({_embedding_mode}): {e}")
                    # Keep previous hidden state on failure

            # --- Normalize hidden state and get action ---
            hidden_normed = (current_hidden - hidden_mean) / hidden_std
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            hid_t = torch.tensor(hidden_normed, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                pi, value = model(obs_t, hid_t)
                action = pi.sample().item()
                v = value.item()
                entropy = pi.entropy().item()

            ep_values.append(v)
            ep_action_counts[action] += 1
            ep_actions.append(int(action))

            # --- Step environment ---
            rng, step_key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                step_key, env_state, action, env_params
            )

            reward_f = float(reward)
            ep_return += reward_f
            ep_rewards.append(reward_f)

            # Track achievements
            for k, val in info.items():
                if k.startswith("Achievements/") and float(val) > 0:
                    name = k.replace("Achievements/", "")
                    if name not in ep_achievements:
                        ep_achievements[name] = step
                        print(f"  [step {step}] Achievement unlocked: {name}")

            # wandb: per-step logging (every 15 steps to avoid excessive data)
            if use_wandb and step % STEP_CADENCE == 0:
                step_log = {
                    "step/value": v,
                    "step/entropy": entropy,
                    "step/reward": reward_f,
                    "step/cumulative_return": ep_return,
                    "step/action": action,
                    "step/action_name": ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action),
                    "step/hidden_norm": float(np.linalg.norm(hidden_normed)),
                    "step/episode": ep + 1,
                    "step/ep_step": step,
                    "step/num_achievements": len(ep_achievements),
                }
                wandb.log(step_log, step=global_step)

            # --- Record video frame ---
            if args.save_video:
                try:
                    game_frame = render_frame(env_state)
                    frame = make_video_frame(
                        game_frame, ep_values, ep_rewards,
                        current_gemini_text, step,
                    )
                    ep_frames.append(frame)
                except Exception:
                    pass  # rendering can fail on some states

            step += 1
            global_step += 1
            if step % 200 == 0:
                print(f"  Step {step}: return={ep_return:.2f}, value={v:.2f}, "
                      f"action={ACTION_NAMES[action] if action < len(ACTION_NAMES) else action}")

        # --- Episode summary ---
        result = {
            "episode": ep + 1,
            "return": ep_return,
            "length": step,
            "achievements": ep_achievements,
            "num_achievements": len(ep_achievements),
            "gemini_calls": len(ep_gemini_log),
            "gemini_errors": ep_gemini_errors,
            "mean_value": float(np.mean(ep_values)) if ep_values else 0,
            "actions": ep_actions,
        }
        all_results.append(result)

        print(f"\n  Return: {ep_return:.2f}")
        print(f"  Length: {step}")
        print(f"  Achievements ({len(ep_achievements)}): {list(ep_achievements.keys())}")
        print(f"  Gemini calls: {len(ep_gemini_log)}, errors: {ep_gemini_errors}")

        # Save episode log
        ep_dir = out_dir / f"episode_{ep + 1:02d}"
        ep_dir.mkdir(exist_ok=True)

        with open(ep_dir / "gemini_log.jsonl", "w") as f:
            for entry in ep_gemini_log:
                f.write(json.dumps(entry) + "\n")

        with open(ep_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)

        # Save video
        video_path = None
        if args.save_video and ep_frames:
            video_path = ep_dir / "gameplay.mp4"
            # Pad all frames to the max height (variable due to Gemini text length)
            max_h = max(f.shape[0] for f in ep_frames)
            w = ep_frames[0].shape[1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, max_h))
            for frame in ep_frames:
                if frame.shape[0] < max_h:
                    pad = np.zeros((max_h - frame.shape[0], w, 3), dtype=np.uint8)
                    frame = np.vstack([frame, pad])
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            # Re-encode to H.264 for browser compatibility (wandb)
            import subprocess, shutil
            if shutil.which("ffmpeg"):
                h264_path = ep_dir / "gameplay_h264.mp4"
                ret = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video_path),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                     "-pix_fmt", "yuv420p", str(h264_path)],
                    capture_output=True, timeout=120,
                )
                if ret.returncode == 0 and h264_path.exists():
                    h264_path.rename(video_path)
                    print(f"  Video saved (H.264): {video_path}")
                else:
                    print(f"  Video saved (mp4v fallback): {video_path}")
            else:
                print(f"  Video saved (mp4v): {video_path}")

        # wandb: per-episode logging
        if use_wandb:
            ep_log = {
                "episode/return": ep_return,
                "episode/length": step,
                "episode/num_achievements": len(ep_achievements),
                "episode/gemini_calls": len(ep_gemini_log),
                "episode/gemini_errors": ep_gemini_errors,
                "episode/mean_value": float(np.mean(ep_values)) if ep_values else 0,
                "episode/std_value": float(np.std(ep_values)) if ep_values else 0,
                "episode/mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0,
                "episode/total_gemini_cost_usd": total_gemini_cost,
            }
            # Log per-achievement timing
            for ach_name, ach_step in ep_achievements.items():
                ep_log[f"achievements/{ach_name}_step"] = ach_step
            # Log action distribution
            top_actions = np.argsort(-ep_action_counts)[:10]
            for rank, ai in enumerate(top_actions):
                if ep_action_counts[ai] > 0:
                    aname = ACTION_NAMES[ai] if ai < len(ACTION_NAMES) else str(ai)
                    ep_log[f"actions/{aname}"] = int(ep_action_counts[ai])

            wandb.log(ep_log, step=global_step)

            # Log video to wandb
            if video_path and video_path.exists():
                try:
                    wandb.log({
                        f"video/episode_{ep+1:02d}": wandb.Video(
                            str(video_path), fps=15, format="mp4",
                        ),
                    }, step=global_step)
                except Exception as e:
                    print(f"  wandb video upload failed: {e}")

            # Log a sample Gemini response as a text table
            if ep_gemini_log:
                sample = ep_gemini_log[0]
                table = wandb.Table(columns=["step", "headline", "full_text", "tokens", "latency_s"])
                for entry in ep_gemini_log[:5]:
                    hl = entry["gemini_text"].split("\n")[0] if entry["gemini_text"] else ""
                    table.add_data(
                        entry["step"], hl, entry["gemini_text"][:500],
                        entry.get("prompt_tokens", 0) + entry.get("completion_tokens", 0),
                        round(entry.get("latency_s", 0), 2),
                    )
                wandb.log({f"gemini_samples/episode_{ep+1:02d}": table}, step=global_step)

    # --- Overall summary ---
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    returns = [r["return"] for r in all_results]
    lengths = [r["length"] for r in all_results]
    n_ach = [r["num_achievements"] for r in all_results]
    print(f"Episodes: {args.num_episodes}")
    print(f"Return:       {np.mean(returns):.2f} +/- {np.std(returns):.2f}  "
          f"(min={min(returns):.2f}, max={max(returns):.2f})")
    print(f"Length:       {np.mean(lengths):.0f} +/- {np.std(lengths):.0f}")
    print(f"Achievements: {np.mean(n_ach):.1f} +/- {np.std(n_ach):.1f}")
    print(f"Gemini calls: {total_gemini_calls} (${total_gemini_cost:.2f})")

    # Achievement frequency across episodes
    all_ach = {}
    for r in all_results:
        for name in r["achievements"]:
            all_ach[name] = all_ach.get(name, 0) + 1
    if all_ach:
        print(f"\nAchievement frequency (out of {args.num_episodes} episodes):")
        for name, count in sorted(all_ach.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}/{args.num_episodes}")

    # Save overall results
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "episodes": all_results,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "mean_achievements": float(np.mean(n_ach)),
            "total_gemini_calls": total_gemini_calls,
            "total_gemini_cost_usd": total_gemini_cost,
            "achievement_frequency": all_ach,
        }, f, indent=2)

    # wandb: final summary
    if use_wandb:
        wandb.summary["mean_return"] = float(np.mean(returns))
        wandb.summary["std_return"] = float(np.std(returns))
        wandb.summary["mean_length"] = float(np.mean(lengths))
        wandb.summary["mean_achievements"] = float(np.mean(n_ach))
        wandb.summary["total_gemini_calls"] = total_gemini_calls
        wandb.summary["total_gemini_cost_usd"] = total_gemini_cost
        for name, count in all_ach.items():
            wandb.summary[f"achievement_freq/{name}"] = count / args.num_episodes

        # Summary table of all episodes
        ep_table = wandb.Table(
            columns=["episode", "return", "length", "achievements", "gemini_calls"],
        )
        for r in all_results:
            ep_table.add_data(
                r["episode"], r["return"], r["length"],
                r["num_achievements"], r["gemini_calls"],
            )
        wandb.log({"summary/episodes": ep_table})
        wandb.finish()
        print("wandb run finished")

    print(f"\nResults saved to {out_dir}")


# ======================================================================
# CLI
# ======================================================================
def main():
    p = argparse.ArgumentParser(description="Online eval of imagination-augmented policy")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to policy checkpoint (default: CKPT_DIR/final.pth)")
    p.add_argument("--hidden-stats", type=str, default=None,
                    help="Path to hidden_state_stats.npz (default: CKPT_DIR/)")
    p.add_argument("--gemini-api-key", type=str, default=None,
                    help="Gemini API key (or set GEMINI_API_KEY env var)")
    p.add_argument("--output-dir", type=str,
                    default="/data/group_data/rl/geney/eval_imagination/")
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-video", dest="save_video", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable wandb logging")
    p.add_argument("--wandb-name", type=str, default=None,
                    help="Custom wandb run name")
    p.add_argument("--layer-width", type=int, default=DEFAULT_LAYER_WIDTH,
                    help="Width of hidden layers (must match training)")
    p.add_argument("--gemini-model", type=str, default=None,
                    help="Override Gemini model (default: config GEMINI_MODEL)")
    p.add_argument("--embedding-mode", type=str, default="gemini",
                    choices=["gemini", "constant", "random", "adversarial", "die",
                             "die_v2", "adversarial_v2",
                             "avoid_water_v2", "avoid_animals_v2",
                             "target_collect_stone_v2", "target_descend_v2",
                             "target_eat_cow_v2", "target_drink_water_v2",
                             "target_place_stone_v2",
                             "target_hunt_animals_v2",
                             "v2_basic_coverage", "v2_long_tail",
                             "direction_left_v2", "direction_right_v2",
                             "direction_up_v2", "direction_down_v2",
                             "die_fast_v2", "survive_long_v2",
                             "target_collect_diamond_v2",
                             "target_make_iron_pickaxe_v2",
                             "target_collect_sapling_v2",
                             "target_defeat_zombie_v2",
                             "target_place_plant_v2",
                             "target_avoid_stone_v2",
                             "target_avoid_stone_v3",
                             "target_stay_overworld_v2",
                             "survive_long_v3",
                             "achievement_max_v1", "achievement_max_v2",
                             "achievement_max_v2_thresh6",
                             "explore_ood_v1",
                             "v2_eat_bat",
                             "v2_enter_mines",
                             "achievement_max_v3", "achievement_max_v4",
                             "target_eat_cow_v3", "target_drink_water_v3",
                             "target_stay_overworld_v3", "target_place_plant_v3",
                             "target_defeat_zombie_v3", "target_collect_sapling_v3",
                             "embed_arith"],
                    help="How to generate hidden state embeddings: "
                         "gemini=normal Gemini+Qwen, constant=embed fixed string, "
                         "random=embed random text, adversarial=Gemini bad futures, "
                         "die=Gemini death-seeking futures, target_X_v2/direction_X_v2 "
                         "= positive/directional steering via algorithm-section swap")
    p.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate (must match training architecture)")
    p.add_argument("--embed-arith-direction", type=str, default=None,
                    help="(embed_arith mode) Path to .npy direction vector to add to embedding.")
    p.add_argument("--embed-arith-alpha", type=float, default=0.0,
                    help="(embed_arith mode) Scalar multiplier for direction vector.")
    p.add_argument("--switch-mode", type=str, default=None,
                    help="If set, after --switch-step env steps the embedding "
                         "mode flips to this mode for the rest of each episode.")
    p.add_argument("--switch-step", type=int, default=0,
                    help="Env step at which to switch mode (only if --switch-mode set).")
    p.add_argument("--no-layernorm", action="store_true",
                    help="Use ActorCriticAug (no LayerNorm) instead of ActorCriticAugLN")
    p.add_argument("--arch-v2", action="store_true",
                    help="Use ActorCriticAugV2 architecture")
    p.add_argument("--arch-gated", action="store_true",
                    help="Use ActorCriticAugGated architecture")
    p.add_argument("--arch-hidden-only", action="store_true",
                    help="Use ActorCriticHiddenOnly (hidden/imagination input only)")
    p.add_argument("--embed-backend", type=str, default="qwen3_gen",
                    choices=["qwen3_gen", "qwen3_embed", "gemini_embed"],
                    help="Which model embeds the Gemini text (default: qwen3_gen)")
    p.add_argument("--hidden-dim", type=int, default=0,
                    help="Override embedding hidden dim (0 = use EMBED_HIDDEN_DIM=4096)")
    p.add_argument("--extract-prediction-only", action="store_true",
                    help="At eval time, embed only the Prediction: suffix of Gemini output "
                         "(matches predonly-trained policies).")
    p.add_argument("--oracle-future-embed", action="store_true",
                    help="If the prompt template has a {future_state_filtered} block, "
                         "fill it by rolling the env forward 5 steps with the current "
                         "policy (greedy). Emulates training-time grounded prompt.")
    p.add_argument("--prompt-template-path", type=str, default=None,
                    help="Override the Gemini prompt template (default: built-in concise).")
    p.add_argument("--gemini-thinking-budget", type=int, default=None,
                    help="If set, override Gemini thinking budget at inference (0=off, "
                         "N>0=cap, -1=dynamic). Required for policies trained with a "
                         "thinking prompt.")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
