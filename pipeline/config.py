"""
Shared configuration for the imagination-augmented offline RL pipeline.

All paths, constants, and hyperparameters live here so every pipeline step
uses the same values and nothing is duplicated.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parent.parent  # the shards directory

# Phase 3 output: filtered + bitpacked trajectory files
FILTERED_DIR = PIPELINE_ROOT / "filtered_trajectories"

# Phase 4 output: Gemini oracle labels (one JSONL per trajectory file)
GEMINI_OUTPUT_DIR = PIPELINE_ROOT / "gemini_labels"

# Phase 5 output: Qwen3-8B embeddings (one NPZ per trajectory file)
EMBED_OUTPUT_DIR = PIPELINE_ROOT / "embeddings"

# Phase 6 output: final merged files (trajectories + embeddings + text)
FINAL_DIR = PIPELINE_ROOT / "final_trajectories"

# Craftax_Baselines repo (for importing obs_to_text, filter_text_obs, etc.)
CRAFTAX_BASELINES = Path.home() / "Craftax_Baselines"

# Prompt template
ORACLE_TEMPLATE_PATH = (
    CRAFTAX_BASELINES
    / "configs"
    / "future_imagination"
    / "templates"
    / "oracle_next15_prompt_concise.txt"
)

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------
GAMMA = 0.99  # discount factor for return-to-go
GEMINI_STEP_CADENCE = 15  # Gemini label every N env-steps within an episode

# Observation layout
# The obs vector is (8268,) = map (8217 binary) + inventory/stats (51 mixed).
# Bitpacking convention matches decode_obs_array() in awr_llm_augmented.py:
# first MAP_OBS_DIM dims are binary → bitpacked, rest stored as float16.
TOTAL_OBS_DIM = 8268
MAP_OBS_DIM = 8217  # contiguous binary map section (indices 0-8216)
INV_OBS_DIM = 51  # inventory + stats section (indices 8217-8267, mixed types)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBED_MODEL = "Qwen/Qwen3-8B"
EMBED_HIDDEN_DIM = 4096
EMBED_LAYER = 30  # of 36 total (~83% depth)

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MAX_OUTPUT_TOKENS = 512
GEMINI_TEMPERATURE = 0.3  # low temp for consistent oracle labels
GEMINI_CONCURRENT_REQUESTS = 50  # parallel threads (sized to approach RPM limit)
GEMINI_REQUESTS_PER_MINUTE = 2000  # rate limit (Flash paid tier)

# ---------------------------------------------------------------------------
# Action names (for compact state rendering)
# ---------------------------------------------------------------------------
ACTION_NAMES = [
    "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
    "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE", "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD", "MAKE_IRON_SWORD", "REST", "DESCEND", "ASCEND",
    "MAKE_DIAMOND_PICKAXE", "MAKE_DIAMOND_SWORD", "MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOUR", "SHOOT_ARROW", "MAKE_ARROW", "CAST_FIREBALL",
    "CAST_ICEBALL", "PLACE_TORCH", "DRINK_POTION_RED", "DRINK_POTION_GREEN",
    "DRINK_POTION_BLUE", "DRINK_POTION_PINK", "DRINK_POTION_CYAN",
    "DRINK_POTION_YELLOW", "READ_BOOK", "ENCHANT_SWORD", "ENCHANT_ARMOUR",
    "MAKE_TORCH", "LEVEL_UP_DEX", "LEVEL_UP_STR", "LEVEL_UP_INT",
    "ENCHANT_BOW",
]


def ensure_imports():
    """No-op: imports now resolved via the Imagination package structure."""
    pass
