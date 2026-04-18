#!/usr/bin/env python3
"""Regenerate disk-resident prompt templates from llm/gameplay.py.

The labelling / eval / online-RL pipelines load a static .txt template
from disk for historical reasons. This script writes the current value of
`FUTURE_PREDICT_PROMPT` out to that .txt so the disk version stays in
sync with the Python source of truth.

Run after editing llm/gameplay.py:
    python tools/regenerate_prompt_templates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm.gameplay import FUTURE_PREDICT_PROMPT

TARGETS = [
    Path.home() / "Craftax_Baselines" / "configs" / "future_imagination"
    / "templates" / "predict_state_only_prompt_concise.txt",
]


def main():
    for path in TARGETS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(FUTURE_PREDICT_PROMPT)
        print(f"wrote {len(FUTURE_PREDICT_PROMPT)} chars → {path}")


if __name__ == "__main__":
    main()
