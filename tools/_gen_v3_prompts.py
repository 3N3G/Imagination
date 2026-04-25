"""One-shot script to generate v3 versions of the NULL-cell prompts.

Each v3 prompt follows the target_descend_v2 pattern that succeeded:
  - Tilt the high-level goal to elevate the target behavior
  - Add Target X as priority 1 in the algorithm priority list (above Survive)
  - Add a "1. Target X" section describing the behavior in concrete terms
  - Keep the rest of the base prompt intact (upgrade tree, etc.)

Output files are written to the templates dir.
"""
from pathlib import Path

BASE = Path(
    "/home/geney/Imagination/configs/training/templates/"
    "predict_state_only_prompt_concise.txt"
).read_text()
OUT_DIR = Path("/home/geney/Imagination/configs/training/templates")

TARGETS = {
    "target_eat_cow_v3": {
        "tilt": "with hunting any visible cow and EATING the meat as the top priority",
        "section_title": "Target eat cow",
        "section_body": (
            "The player walks directly toward any visible cow and uses DO to attack. "
            "After 2-3 hits the cow drops a meat tile; the player steps onto the meat "
            "and uses DO again to eat it (this both raises Food and unlocks +1 eat_cow). "
            "If multiple cows are visible the player routes to the closest first and "
            "moves on to the next once the current cow is dead. If no cow is visible "
            "the player walks toward the most cow-likely unexplored direction (open "
            "grass tiles, since cows spawn on grass). The player hunts cows even when "
            "Food is at 9, since the +1 achievement is one-time and only counts if it "
            "happens at all."
        ),
    },
    "target_drink_water_v3": {
        "tilt": "with walking to the nearest water tile and DRINKING from it as the top priority, repeated as often as water is available",
        "section_title": "Target drink water",
        "section_body": (
            "The player walks directly to the nearest visible water tile and uses DO "
            "while adjacent to drink. The player keeps drinking even when Drink is "
            "already 9 — every adjacent water tile triggers another DO. Once the "
            "current water tile no longer raises Drink, the player walks to the next "
            "visible water tile and repeats. If no water is visible, the player walks "
            "toward the most water-likely unexplored direction (typically toward darker "
            "blue tiles or low-lying terrain). Drinking is the player's primary "
            "objective; mining and crafting only happen when no water tile is in sight."
        ),
    },
    "target_stay_overworld_v3": {
        "tilt": "while NEVER descending a ladder and staying entirely on Floor 0 (the overworld)",
        "section_title": "Stay on the overworld",
        "section_body": (
            "The player STAYS on Floor 0 for the entire episode. The player NEVER uses "
            "the DESCEND action. If a down-ladder is visible, the player walks past it "
            "without stopping. The player treats the overworld as the only valid play "
            "area and keeps doing useful overworld activities (gather wood, mine stone "
            "and iron, craft, eat cows, drink water, sleep) for as long as the episode "
            "lasts. The player explores horizontally to find more resources, never "
            "vertically. If the player is somehow already on a non-overworld floor the "
            "player walks to the up-ladder and ASCENDs back to the overworld."
        ),
    },
    "target_place_plant_v3": {
        "tilt": "with PLACING a sapling on every reachable grass tile as the top priority once the player is carrying any sapling",
        "section_title": "Target place plant",
        "section_body": (
            "If the player is carrying at least 1 sapling, the player walks to the "
            "nearest grass tile (within 5 tiles) and uses PLACE_PLANT to place the "
            "sapling. The player keeps placing saplings on grass tiles until the "
            "sapling inventory is empty. After placing, the player notes the location "
            "(plant ripens after about 30-60 steps and can be eaten via DO for +1 "
            "eat_plant). If the player has no sapling, the player walks to the nearest "
            "visible grass-with-sapling-icon tile and uses DO to harvest a sapling, "
            "then resumes placing. PLACE_PLANT is one-time +1 achievement and the "
            "player should use it at the very first opportunity."
        ),
    },
    "target_defeat_zombie_v3": {
        "tilt": "with seeking out and killing zombies as the top priority whenever any zombie is visible on screen",
        "section_title": "Target defeat zombie",
        "section_body": (
            "Zombies are MELEE mobs that look like green humanoids. The player actively "
            "seeks them out: if a zombie is visible the player walks directly to it and "
            "uses DO to attack until the zombie dies. If the player has a sword "
            "(wood/stone/iron) it kills in fewer hits — the player crafts a wood sword "
            "as soon as possible to fight zombies more efficiently. The player attacks "
            "zombies even at full Health and Food, since defeat_zombie is a one-time "
            "+1 achievement. After killing a zombie the player scans for the next one. "
            "The player does NOT run from zombies under any circumstance."
        ),
    },
    "target_collect_sapling_v3": {
        "tilt": "with searching for and harvesting saplings off grass tiles as the top priority",
        "section_title": "Target collect sapling",
        "section_body": (
            "Saplings appear as small green icons on grass tiles. The player walks to "
            "every visible sapling within 5 tiles and uses DO to harvest it (this "
            "unlocks +1 collect_sapling and adds 1 sapling to inventory). Once the "
            "first sapling is collected, the player should aim to maintain at least 3 "
            "saplings in inventory — saplings respawn on grass tiles every ~50 steps so "
            "the player makes a habit of revisiting grassy areas. If no sapling is "
            "visible, the player walks toward the nearest grass cluster (since "
            "saplings only spawn on grass) and waits or explores there for one to "
            "appear. Harvesting saplings is the primary goal; everything else is "
            "secondary."
        ),
    },
}


def _render(name, tilt, section_title, section_body):
    out = BASE
    # Replace the high-level goal line
    out = out.replace(
        "At every step, the player should act with the goal of staying alive and progressing down floors.",
        f"At every step, the player should act with the goal of staying alive and progressing down floors, {tilt}.",
    )
    # Insert "Target X" as priority 1 (renumber existing priorities)
    out = out.replace(
        "This means the player will choose the highest-priority active goal in this order:\n"
        "1. Survive\n"
        "2. Take the ladder if it is open and on-screen\n"
        "3. Upgrade equipment if survival is stable. This takes priority over taking the ladder if the player is in the overworld (floor 0) and has a sword or pickaxe worse than stone or missing.\n"
        "4. Explore to find resources, troops, and the ladder",
        "This means the player will choose the highest-priority active goal in this order:\n"
        f"1. {section_title}\n"
        "2. Survive\n"
        "3. Take the ladder if it is open and on-screen\n"
        "4. Upgrade equipment if survival is stable. This takes priority over taking the ladder if the player is in the overworld (floor 0) and has a sword or pickaxe worse than stone or missing.\n"
        "5. Explore to find resources, troops, and the ladder",
    )
    # Insert the target section before "1. Survive" body (keep the body number-renumbered later)
    out = out.replace(
        "1. Survive\n",
        f"1. {section_title}\n{section_body}\n\n2. Survive\n",
        1,
    )
    # Renumber the section headers that follow
    out = out.replace("\n2. Take the ladder if it is open and visible\n",
                      "\n3. Take the ladder if it is open and visible\n")
    out = out.replace("\n3. Upgrade equipment\n", "\n4. Upgrade equipment\n")
    out = out.replace("\n4. Explore\n", "\n5. Explore\n")

    path = OUT_DIR / f"predict_state_only_prompt_concise_{name}.txt"
    path.write_text(out)
    return path


for name, spec in TARGETS.items():
    p = _render(name, **spec)
    print(f"wrote {p}")
