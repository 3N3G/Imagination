"""Prompt-iteration webapp for Craftax imagination + action selection.

Run locally:
    cd webapp/
    pip install -r requirements.txt
    export GEMINI_API_KEY=...
    streamlit run app.py

Data + images are committed to webapp/data/ — no cluster access needed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"

MODELS = [
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]

DEFAULT_FUTURE_PREDICT_PROMPT = """\
You are forecasting a plausible future for a Craftax state.

Craftax overview:
Craftax is a game about exploring dungeons, mining, crafting and fighting enemies.

Use these game rules:
1) Coordinates are (Row, Column), centered on the player at (0,0).
   - Negative Row = UP, Positive Row = DOWN.
   - Negative Column = LEFT, Positive Column = RIGHT.
2) Intrinsics: Health, Food, Drink, Energy, Mana. All out of 9.
   - Food/Drink/Energy naturally decay.
   - If Food/Drink/Energy reaches 0, Health will decay.
   - If Food/Drink/Energy are maintained, Health can recover.
3) Floor progression uses ladders.
   - Descending requires reaching the ladder on the current floor.
   - On non-overworld floors, the ladder is generally closed until enough mobs are killed.
4) Actions: NOOP, LEFT, RIGHT, UP, DOWN, DO (interact/attack/mine/drink/eat),
   SLEEP, PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT,
   MAKE_{WOOD,STONE,IRON,DIAMOND}_{PICKAXE,SWORD}, REST, DESCEND, ASCEND,
   MAKE_ARROW, SHOOT_ARROW, CAST_FIREBALL, CAST_ICEBALL, PLACE_TORCH,
   DRINK_POTION_*, READ_BOOK, ENCHANT_*, LEVEL_UP_*.

Here is a good algorithm the player will play the game by:
At every step, the player should act with the goal of staying alive and progressing down floors.
This means the player will choose the highest-priority active goal in this order:
1. Survive
2. Take the ladder if it is open and visible
3. Upgrade equipment if survival is stable
4. Explore to find resources, troops, and the ladder
1. Survive
The player must track health, food, drink, and energy.  If food is <= 4, get food immediately by killing animals and eating them.  If drink is <= 4, get drink immediately from water tiles.  If energy is <= 4, make a safe enclosure and sleep.  If health is <= 4, restore food, drink, and energy before doing anything risky.  The player should never sleep in the open. Before sleeping, block enemies out, for example with stone walls. An easy way for the player to become safe is to mine a tunnel into a cluster of stone and place a stone behind blocking off the tunnel.
2. Take the ladder if it is open and visible
If the ladder is already open, the player should prioritize finding it and using it.  On the overworld, progression is just finding the ladder. Note that open and visible are not the same. A ladder opens so that a player who finds it can descend using it, but the player still needs to find the tile labelled as down_ladder. On later floors, the ladder opens only after 8 troops have been killed.  If the ladder is open, the player should stop focusing on upgrades unless the player is on the first floor, where most of the important early resources are found, such as wood, stone, and coal. Each time and only each time the player descends to a new floor, they will gain one player_xp, which can be used to upgrade one of three attributes:
1. Strength: increases max health and physical melee damage
2. Dexterity: increases max food, drink, and energy and slows their decrease
2. Intelligence: increases max mana, mana regeneration, and spell damage
3. Upgrade equipment
The player should upgrade only when survival is stable and the ladder is not already the main priority.
Upgrade order:
Pickaxe: wood -> stone -> iron -> diamond
Sword: wood -> stone -> iron -> diamond
Armor: iron -> diamond
Upgrade rules:
If the player has no useful tools and less than 10 wood, gather wood first.
If crafting is needed and there is no crafting table nearby, craft a crafting table.
If the player has wood tools, mine 10 stone.
If the player has stone but no stone tools, craft a stone pickaxe and stone sword.
Mine coal whenever it is seen.
Mine iron whenever it is seen if the player has an iron pickaxe.
If the player has iron, coal, and wood, and is next to a furnace and crafting table, craft iron tools.
If the player has extra iron, craft iron armor.
If the player has diamonds and is next to a furnace and crafting table, craft diamond equipment.
Diamond tools require diamond, coal, and wood.
Diamond armor requires diamond and coal.
4. Explore
If the player is not in immediate danger, the ladder is not in sight, and no immediate upgrade is available, the player should explore.
While exploring, the player should:
- look for the ladder
- kill troops if the ladder is still closed
- gather useful nearby resources, especially wood, stone, coal, iron, and diamonds

Predict at a high level what the next five steps for the player will look like, given that they are following the algorithm. Do not forecast beyond five time steps! In particular, the player can move at most five tiles during these five steps. Reason during your state understanding about the most immediate next step according to the algorithm and then predict the player's immediate behavior.

State Understanding: <A few sentences analyzing the current scene. Focus on careful spatial reasoning of the relevant tiles or tiles near the player. >

Prediction: <1 sentence description of the high-level behavior of the player in the next five steps. E.g. "move right to the cluster of trees", or "chase and kill the cow above", or "move down to look for water", or "move up and left to the visible open ladder". >

Now, predict the future of the following state.

Current state:
{current_state_filtered}
"""


DEFAULT_ACTION_SELECT_PROMPT = """\
You are playing Craftax. At every step, choose the single action that best follows this algorithm:

At every step, the player should act with the goal of staying alive and progressing down floors.
This means the player will choose the highest-priority active goal in this order:
1. Survive
2. Take the ladder if it is open and visible
3. Upgrade equipment if survival is stable
4. Explore to find resources, troops, and the ladder
1. Survive
The player must track health, food, drink, and energy.  If food is <= 4, get food immediately by killing animals and eating them.  If drink is <= 4, get drink immediately from water tiles.  If energy is <= 4, make a safe enclosure and sleep.  If health is <= 4, restore food, drink, and energy before doing anything risky.  The player should never sleep in the open. Before sleeping, block enemies out, for example with stone walls. An easy way for the player to become safe is to mine a tunnel into a cluster of stone and place a stone behind blocking off the tunnel.
2. Take the ladder if it is open and visible
If the ladder is already open, the player should prioritize finding it and using it.  On the overworld, progression is just finding the ladder. Note that open and visible are not the same. A ladder opens so that a player who finds it can descend using it, but the player still needs to find the tile labelled as down_ladder. On later floors, the ladder opens only after 8 troops have been killed.  If the ladder is open, the player should stop focusing on upgrades unless the player is on the first floor, where most of the important early resources are found, such as wood, stone, and coal. Each time and only each time the player descends to a new floor, they will gain one player_xp, which can be used to upgrade one of three attributes:
1. Strength: increases max health and physical melee damage
2. Dexterity: increases max food, drink, and energy and slows their decrease
2. Intelligence: increases max mana, mana regeneration, and spell damage
3. Upgrade equipment
The player should upgrade only when survival is stable and the ladder is not already the main priority.
Upgrade order:
Pickaxe: wood -> stone -> iron -> diamond
Sword: wood -> stone -> iron -> diamond
Armor: iron -> diamond
Upgrade rules:
If the player has no useful tools and less than 10 wood, gather wood first.
If crafting is needed and there is no crafting table nearby, craft a crafting table.
If the player has wood tools, mine 10 stone.
If the player has stone but no stone tools, craft a stone pickaxe and stone sword.
Mine coal whenever it is seen.
Mine iron whenever it is seen if the player has an iron pickaxe.
If the player has iron, coal, and wood, and is next to a furnace and crafting table, craft iron tools.
If the player has extra iron, craft iron armor.
If the player has diamonds and is next to a furnace and crafting table, craft diamond equipment.
Diamond tools require diamond, coal, and wood.
Diamond armor requires diamond and coal.
4. Explore
If the player is not in immediate danger, the ladder is not in sight, and no immediate upgrade is available, the player should explore.
While exploring, the player should:
- look for the ladder
- kill troops if the ladder is still closed
- gather useful nearby resources, especially wood, stone, coal, iron, and diamonds

Coordinates: (Row, Column) relative to player at (0,0).
  Negative Row = UP, Positive Row = DOWN.
  Negative Column = LEFT, Positive Column = RIGHT.

Available actions (only use these exact names):
NOOP, LEFT, RIGHT, UP, DOWN, DO, SLEEP, PLACE_STONE, PLACE_TABLE,
PLACE_FURNACE, PLACE_PLANT, MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE,
MAKE_IRON_PICKAXE, MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD,
REST, DESCEND, ASCEND, MAKE_DIAMOND_PICKAXE, MAKE_DIAMOND_SWORD,
MAKE_IRON_ARMOUR, MAKE_DIAMOND_ARMOUR, SHOOT_ARROW, MAKE_ARROW,
CAST_FIREBALL, CAST_ICEBALL, PLACE_TORCH, DRINK_POTION_RED,
DRINK_POTION_GREEN, DRINK_POTION_BLUE, DRINK_POTION_PINK,
DRINK_POTION_CYAN, DRINK_POTION_YELLOW, READ_BOOK, ENCHANT_SWORD,
ENCHANT_ARMOUR, MAKE_TORCH, LEVEL_UP_DEXTERITY, LEVEL_UP_STRENGTH,
LEVEL_UP_INTELLIGENCE, ENCHANT_BOW.

Output format (strict):
REASONING: <couple sentences of rationale>
ACTION: <single action name>

Do not output anything else.

Current state:
{current_state_filtered}
"""


# ---------------------------------------------------------------------------
# Gemini HTTP call
# ---------------------------------------------------------------------------
def call_gemini(prompt: str, model: str, api_key: str,
                temperature: float = 0.2, max_output_tokens: int = 2048):
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
        with urlrequest.urlopen(req, timeout=90.0) as resp:
            raw = resp.read().decode("utf-8")
    except urlerror.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        return {"text": "", "error": f"HTTP {e.code}: {err}",
                "latency_s": time.perf_counter() - t0}

    parsed = json.loads(raw)
    cands = parsed.get("candidates", [])
    text = ""
    if cands:
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    usage = parsed.get("usageMetadata", {})
    return {
        "text": text,
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
        "latency_s": time.perf_counter() - t0,
        "error": None,
    }


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
max_tokens = st.sidebar.number_input("Max output tokens", 256, 8192, 2048, 256)
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
                        st.caption(
                            f"{res['latency_s']:.1f}s · "
                            f"{res['prompt_tokens']}→{res['completion_tokens']} tok"
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
                        st.caption(
                            f"{res['latency_s']:.1f}s · "
                            f"{res['prompt_tokens']}→{res['completion_tokens']} tok"
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
