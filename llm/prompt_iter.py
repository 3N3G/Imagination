"""Prompt-iteration loop for pure Gemini-plays Craftax.

GEPA-style: each iteration runs N rollouts with the current prompt, then
asks a stronger Gemini model to propose a refined prompt given the
trajectories' returns + observed failure modes.

Usage:
    python -m llm.prompt_iter \\
        --start-prompt configs/training/templates/predict_state_only_prompt_concise.txt \\
        --output-dir /data/user_data/geney/prompt_iter_runs/run_$(date +%s) \\
        --num-iters 10 \\
        --eps-per-iter 5 \\
        --player-model gemini-3-flash-preview \\
        --proposer-model gemini-3-pro-preview

Each iteration writes:
    iter_NN/
      prompt.txt              — the prompt used in this iter
      eval_results/           — gemini_play output (results.json, episode_*/summary.json)
      proposal_meta.json      — proposer call payload + raw response
The next iter's prompt.txt is the proposer's output.

Outer summary: scores.csv (iter, mean_return, std_return).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Direct Gemini API call for the proposer (we want a different model + prompt
# style than the gameplay caller).
import urllib.request
import urllib.error


META_PROMPT_TEMPLATE = """\
You are optimizing a system prompt that tells Gemini how to play the Craftax
game (a 2D survival/crafting game with achievements). The prompt is fed to a
gameplay LLM (Gemini Flash) at every step; the LLM responds with a single
ACTION token. The score is "raw episode return" = sum of achievement rewards
(BASIC=1, INTERMEDIATE=3, ADVANCED=5, VERY_ADVANCED=8). Episodes end on
player death or after 600 steps (truncation).

Below is the CURRENT prompt and the results of running it for {n} episodes.

Your task:
1. Identify recurring failure modes from the episodes (early deaths, missed
   achievements, stuck loops, etc.)
2. Propose a REFINED full prompt that (a) keeps the working behaviors and
   (b) fixes the failure modes.
3. The new prompt MUST contain exactly one `{{current_state_filtered}}`
   placeholder where the current observation will be substituted.
4. The new prompt MUST instruct the model to output one line of the form
   `ACTION: <NAME>` where <NAME> is one of the legal Craftax action names
   (NOOP, LEFT, RIGHT, UP, DOWN, DO, SLEEP, PLACE_STONE, PLACE_TABLE,
   PLACE_FURNACE, PLACE_PLANT, MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE,
   MAKE_IRON_PICKAXE, MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD,
   REST, DESCEND, ASCEND, MAKE_DIAMOND_PICKAXE, MAKE_DIAMOND_SWORD,
   MAKE_IRON_ARMOUR, MAKE_DIAMOND_ARMOUR, SHOOT_ARROW, MAKE_ARROW,
   CAST_FIREBALL, CAST_ICEBALL, PLACE_TORCH, DRINK_POTION_*,
   READ_BOOK, ENCHANT_SWORD, ENCHANT_ARMOUR, MAKE_TORCH, LEVEL_UP_DEX,
   LEVEL_UP_STR, LEVEL_UP_INT, ENCHANT_BOW).
5. Keep the prompt UNDER 6000 characters.

==== CURRENT PROMPT ====
{prompt}

==== EPISODE RESULTS (n={n}) ====
mean return = {mean_ret:.2f}
std  return = {std_ret:.2f}
mean length = {mean_len:.1f}
mean achievements/episode = {mean_ach:.2f}

==== PER-EPISODE DETAIL ====
{episode_blocks}

==== INSTRUCTIONS ====
Output ONLY the new prompt text (no preamble, no commentary, no code fence).
Begin with the first line of the new prompt directly.
"""


def call_gemini_proposer(text: str, api_key: str, model: str,
                         max_output_tokens: int = 8192) -> str:
    """Call Gemini API for a single completion. Returns the response text."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    payload = json.dumps({
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": max_output_tokens,
        },
    }).encode()
    headers = {"Content-Type": "application/json"}
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
            cands = data.get("candidates", [])
            if not cands:
                raise RuntimeError(f"No candidates in response: {data}")
            parts = cands[0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
            if attempt == 3:
                raise
            wait = 5 * (attempt + 1)
            print(f"  proposer attempt {attempt+1} failed: {e}; retry in {wait}s")
            time.sleep(wait)


def summarize_episode(ep_dir: Path) -> dict:
    """Read summary.json and produce a compact dict for the proposer."""
    sf = ep_dir / "summary.json"
    if not sf.exists():
        return {"return": 0.0, "length": 0, "achievements": [], "last_actions": []}
    d = json.load(open(sf))
    actions = d.get("actions", []) or []
    return {
        "return": float(d.get("return", 0.0)),
        "length": int(d.get("length", 0)),
        "achievements": list(d.get("achievements", {}).keys()) if isinstance(d.get("achievements"), dict) else list(d.get("achievements", [])),
        "last_actions": actions[-15:] if actions else [],  # last 15 actions before episode end
    }


ACTION_NAMES_FOR_DECODE = [
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


def format_episode_block(idx: int, ep: dict) -> str:
    last = ep["last_actions"]
    last_str = ",".join(ACTION_NAMES_FOR_DECODE[a] if 0 <= a < len(ACTION_NAMES_FOR_DECODE) else str(a) for a in last)
    return (
        f"Episode {idx}: return={ep['return']:.1f}, length={ep['length']}, "
        f"achievements ({len(ep['achievements'])}): {ep['achievements']}\n"
        f"  last 15 actions: [{last_str}]"
    )


def run_iteration(iter_dir: Path, prompt_path: Path, api_key: str,
                  player_model: str, proposer_model: str, eps_per_iter: int,
                  max_steps: int) -> dict:
    """Run one iteration: eval current prompt, then call proposer."""
    eval_dir = iter_dir / "eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Iter {iter_dir.name}: running {eps_per_iter} eps with prompt={prompt_path} ===")
    cmd = [
        sys.executable, "-m", "llm.gemini_play",
        "--num-episodes", str(eps_per_iter),
        "--max-steps", str(max_steps),
        "--seed", "42",
        "--model", player_model,
        "--api-key", api_key,
        "--output-dir", str(eval_dir),
        "--no-wandb",
        "--prompt-file", str(prompt_path),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, env={**os.environ, "GEMINI_API_KEY": api_key},
                          capture_output=True, text=True)
    elapsed = time.time() - t0
    (iter_dir / "gemini_play.stdout").write_text(proc.stdout)
    (iter_dir / "gemini_play.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  gemini_play exited with code {proc.returncode}; stderr tail:")
        print(proc.stderr[-1000:])
        raise RuntimeError("gemini_play failed")

    # Aggregate episode results.
    episodes = []
    for ep_d in sorted(eval_dir.iterdir()):
        if ep_d.is_dir() and ep_d.name.startswith("episode_"):
            episodes.append(summarize_episode(ep_d))
    if not episodes:
        raise RuntimeError(f"No episode_* dirs in {eval_dir}")

    returns = [e["return"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    aches = [len(e["achievements"]) for e in episodes]
    summary = {
        "n": len(episodes),
        "mean_return": statistics.mean(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "mean_length": statistics.mean(lengths),
        "mean_achievements": statistics.mean(aches),
        "elapsed_sec": elapsed,
        "episodes": episodes,
    }
    (iter_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  iter eval done in {elapsed:.0f}s: mean_return={summary['mean_return']:.2f} +/- {summary['std_return']:.2f}, mean_length={summary['mean_length']:.1f}")

    # Build meta-prompt + call proposer.
    cur_prompt = prompt_path.read_text()
    blocks = "\n\n".join(format_episode_block(i, e) for i, e in enumerate(episodes))
    meta_prompt = META_PROMPT_TEMPLATE.format(
        n=len(episodes),
        prompt=cur_prompt,
        mean_ret=summary["mean_return"],
        std_ret=summary["std_return"],
        mean_len=summary["mean_length"],
        mean_ach=summary["mean_achievements"],
        episode_blocks=blocks,
    )
    (iter_dir / "proposer_input.txt").write_text(meta_prompt)
    print(f"  calling proposer ({proposer_model})...")
    t0 = time.time()
    new_prompt = call_gemini_proposer(meta_prompt, api_key, proposer_model)
    proposer_elapsed = time.time() - t0
    (iter_dir / "proposer_response.txt").write_text(new_prompt)

    if "{current_state_filtered}" not in new_prompt:
        print(f"  WARNING: proposer output missing {{current_state_filtered}} placeholder; "
              f"keeping previous prompt for next iter")
        next_prompt_text = cur_prompt
    else:
        next_prompt_text = new_prompt

    summary["proposer_elapsed_sec"] = proposer_elapsed
    summary["proposer_chars"] = len(new_prompt)
    return summary, next_prompt_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-prompt", type=str, required=True,
                   help="Path to the seed prompt with {current_state_filtered} placeholder")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-iters", type=int, default=10)
    p.add_argument("--eps-per-iter", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--player-model", type=str, default="gemini-3-flash-preview")
    p.add_argument("--proposer-model", type=str, default="gemini-3-pro-preview")
    p.add_argument("--api-key", type=str, default="")
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY env var or pass --api-key")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== prompt_iter run ===")
    print(f"  start prompt: {args.start_prompt}")
    print(f"  output dir:   {out_dir}")
    print(f"  num iters:    {args.num_iters}")
    print(f"  eps per iter: {args.eps_per_iter}")
    print(f"  player:       {args.player_model}")
    print(f"  proposer:     {args.proposer_model}")

    scores_path = out_dir / "scores.csv"
    if not scores_path.exists():
        scores_path.write_text("iter,mean_return,std_return,mean_length,mean_ach,proposer_chars\n")

    # Seed the iter_00 prompt by copying the start prompt.
    iter0_dir = out_dir / "iter_00"
    iter0_dir.mkdir(parents=True, exist_ok=True)
    iter0_prompt_path = iter0_dir / "prompt.txt"
    if not iter0_prompt_path.exists():
        shutil.copy(args.start_prompt, iter0_prompt_path)

    best_return = float("-inf")
    best_iter = -1

    for i in range(args.num_iters):
        iter_dir = out_dir / f"iter_{i:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = iter_dir / "prompt.txt"
        if not prompt_path.exists():
            raise RuntimeError(f"Missing {prompt_path}; outer loop bug")

        try:
            summary, next_prompt_text = run_iteration(
                iter_dir, prompt_path, api_key,
                args.player_model, args.proposer_model,
                args.eps_per_iter, args.max_steps)
        except Exception as e:
            print(f"  iter {i} FAILED: {e}")
            continue

        with scores_path.open("a") as f:
            f.write(f"{i},{summary['mean_return']:.3f},{summary['std_return']:.3f},"
                    f"{summary['mean_length']:.1f},{summary['mean_achievements']:.2f},"
                    f"{summary['proposer_chars']}\n")

        if summary["mean_return"] > best_return:
            best_return = summary["mean_return"]
            best_iter = i
            shutil.copy(prompt_path, out_dir / "best_prompt.txt")
            print(f"  *** new best: iter {i}, return {best_return:.2f}")

        # Seed the next iter's prompt.
        if i + 1 < args.num_iters:
            next_dir = out_dir / f"iter_{i+1:02d}"
            next_dir.mkdir(parents=True, exist_ok=True)
            (next_dir / "prompt.txt").write_text(next_prompt_text)

    print(f"\n=== prompt_iter complete ===")
    print(f"  best iter: {best_iter}, return: {best_return:.2f}")
    print(f"  best prompt saved to: {out_dir / 'best_prompt.txt'}")


if __name__ == "__main__":
    main()
