"""
Observation to Text Decoder for Craftax Symbolic Observations

This module provides obs_to_text() which converts the 8268-dimensional symbolic
observation from Craftax back into human-readable text format, matching the
output of render_craftax_text().

The symbolic observation is structured as:
1. Map view (9x11 grid) - one-hot encoded block types
2. Item map view - one-hot encoded item types  
3. Mob map - binary indicators for mob positions
4. Light map - visibility mask
5. Inventory - sqrt-normalized counts
6. Intrinsics - player stats normalized by /10
7. Direction - one-hot (4 classes)
8. Armour and enchantments
9. Special values (light level, flags, etc.)

Based on craftax v1.5.0 render_craftax_symbolic encoding.
"""

import numpy as np
from enum import Enum
from typing import List, Tuple


# ============================================================================
# Constants (matching craftax/craftax/constants.py)
# ============================================================================

OBS_DIM = (9, 11)  # Visible map dimensions


class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36


class ItemType(Enum):
    NONE = 0
    TORCH = 1
    LADDER_DOWN = 2
    LADDER_UP = 3
    LADDER_DOWN_BLOCKED = 4


class Action(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


# Mob class names for the 5 mob classes × 8 types
MOB_NAMES = {
    # Class 0: Melee mobs (8 types per floor)
    0: ["Zombie", "Gnome Warrior", "Orc Soldier", "Lizard", "Knight", "Troll", "Fire Elemental", "Ice Elemental"],
    # Class 1: Passive mobs (8 types per floor)
    1: ["Cow", "Bat", "Snail", "Frog", "Deer", "Golem", "Imp", "Penguin"],
    # Class 2: Ranged mobs
    2: ["Archer", "Gnome Archer", "Orc Archer", "Crocodile", "Archer Knight", "Troll Archer", "Fire Mage", "Ice Mage"],
    # Class 3: Mob projectiles
    3: ["Arrow", "Dagger", "Fireball", "Iceball", "Arrow", "Slimeball", "Fireball", "Iceball"],
    # Class 4: Player projectiles
    4: ["Arrow (Player)", "Dagger (Player)", "Fireball (Player)", "Iceball (Player)", 
        "Arrow (Player)", "Slimeball (Player)", "Fireball (Player)", "Iceball (Player)"],
}

NUM_BLOCK_TYPES = len(BlockType)  # 37
NUM_ITEM_TYPES = len(ItemType)    # 5
NUM_MOB_CLASSES = 5
NUM_MOB_TYPES = 8
MOB_CHANNELS = NUM_MOB_CLASSES * NUM_MOB_TYPES  # 40

# Observation structure sizes
MAP_CHANNELS = NUM_BLOCK_TYPES + NUM_ITEM_TYPES + MOB_CHANNELS + 1  # blocks + items + mobs + light
MAP_OBS_SIZE = OBS_DIM[0] * OBS_DIM[1] * MAP_CHANNELS  # 9 * 11 * 83 = 8217
INVENTORY_OBS_SIZE = 51  # From get_inventory_obs_shape()
TOTAL_OBS_SIZE = MAP_OBS_SIZE + INVENTORY_OBS_SIZE  # 8268


# ============================================================================
# Decoder Functions
# ============================================================================

def decode_map_section(map_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode the flattened map observation into block types, items, mobs, and light.
    
    Args:
        map_flat: Flattened map observation of shape (9*11*83,) = (8217,)
    
    Returns:
        block_types: (9, 11) array of BlockType values
        item_types: (9, 11) array of ItemType values
        mob_map: (9, 11, 40) array of mob presence indicators
        light_map: (9, 11) boolean visibility mask
    """
    # Reshape to (9, 11, 83)
    map_3d = map_flat.reshape(OBS_DIM[0], OBS_DIM[1], MAP_CHANNELS)
    
    # Split channels
    block_onehot = map_3d[:, :, :NUM_BLOCK_TYPES]                    # (9, 11, 37)
    item_onehot = map_3d[:, :, NUM_BLOCK_TYPES:NUM_BLOCK_TYPES + NUM_ITEM_TYPES]  # (9, 11, 5)
    mob_indicators = map_3d[:, :, NUM_BLOCK_TYPES + NUM_ITEM_TYPES:-1]  # (9, 11, 40)
    light_mask = map_3d[:, :, -1]                                    # (9, 11)
    
    # Decode one-hot to indices
    block_types = np.argmax(block_onehot, axis=-1)  # (9, 11)
    item_types = np.argmax(item_onehot, axis=-1)    # (9, 11)
    
    return block_types, item_types, mob_indicators, light_mask


def decode_inventory_section(inv_flat: np.ndarray) -> dict:
    """
    Decode the inventory section of the observation.
    
    The inventory section is 51 floats structured as:
    - 16 inventory counts (sqrt-normalized by /10)
    - 6 potions (sqrt-normalized)
    - 9 intrinsics (/10 normalized)
    - 4 direction one-hot
    - 4 armour (/2 normalized)
    - 4 armour enchantments
    - 8 special values
    
    Args:
        inv_flat: Flattened inventory observation of shape (51,)
    
    Returns:
        Dictionary with decoded inventory and player stats
    """
    idx = 0
    
    # Helper to decode sqrt-normalized counts: obs = sqrt(count)/10 → count = (obs*10)^2
    def decode_sqrt_count(val):
        return int(round((val * 10) ** 2))
    
    # Inventory counts (16 values, sqrt-normalized)
    inventory = {
        "wood": decode_sqrt_count(inv_flat[0]),
        "stone": decode_sqrt_count(inv_flat[1]),
        "coal": decode_sqrt_count(inv_flat[2]),
        "iron": decode_sqrt_count(inv_flat[3]),
        "diamond": decode_sqrt_count(inv_flat[4]),
        "sapphire": decode_sqrt_count(inv_flat[5]),
        "ruby": decode_sqrt_count(inv_flat[6]),
        "sapling": decode_sqrt_count(inv_flat[7]),
        "torches": decode_sqrt_count(inv_flat[8]),
        "arrows": decode_sqrt_count(inv_flat[9]),
        "books": int(round(inv_flat[10] * 2)),  # /2 normalized
        "pickaxe": int(round(inv_flat[11] * 4)),  # /4 normalized
        "sword": int(round(inv_flat[12] * 4)),    # /4 normalized
        "sword_enchantment": int(inv_flat[13]),
        "bow_enchantment": int(inv_flat[14]),
        "bow": int(inv_flat[15]),
    }
    idx = 16
    
    # Potions (6 values, sqrt-normalized)
    potions = [decode_sqrt_count(inv_flat[idx + i]) for i in range(6)]
    idx += 6
    
    # Intrinsics (9 values, /10 normalized)
    intrinsics = {
        "health": inv_flat[idx] * 10,
        "food": int(round(inv_flat[idx + 1] * 10)),
        "drink": int(round(inv_flat[idx + 2] * 10)),
        "energy": int(round(inv_flat[idx + 3] * 10)),
        "mana": int(round(inv_flat[idx + 4] * 10)),
        "xp": int(round(inv_flat[idx + 5] * 10)),
        "dexterity": int(round(inv_flat[idx + 6] * 10)),
        "strength": int(round(inv_flat[idx + 7] * 10)),
        "intelligence": int(round(inv_flat[idx + 8] * 10)),
    }
    idx += 9
    
    # Direction (4 one-hot values)
    direction_onehot = inv_flat[idx:idx + 4]
    direction = int(np.argmax(direction_onehot)) + 1  # Add 1 because it's encoded as direction-1
    idx += 4
    
    # Armour (4 values, /2 normalized)
    armour = [int(round(inv_flat[idx + i] * 2)) for i in range(4)]
    idx += 4
    
    # Armour enchantments (4 values)
    armour_enchantments = [int(inv_flat[idx + i]) for i in range(4)]
    idx += 4
    
    # Special values (8 values)
    special = {
        "light_level": inv_flat[idx],
        "is_sleeping": bool(inv_flat[idx + 1] > 0.5),
        "is_resting": bool(inv_flat[idx + 2] > 0.5),
        "learned_fireball": bool(inv_flat[idx + 3] > 0.5),
        "learned_iceball": bool(inv_flat[idx + 4] > 0.5),
        "floor": int(round(inv_flat[idx + 5] * 10)),
        "ladder_open": bool(inv_flat[idx + 6] > 0.5),
        "boss_vulnerable": bool(inv_flat[idx + 7] > 0.5),
    }
    
    return {
        "inventory": inventory,
        "potions": potions,
        "intrinsics": intrinsics,
        "direction": direction,
        "armour": armour,
        "armour_enchantments": armour_enchantments,
        "special": special,
    }


def get_mob_name(mob_idx: int) -> str:
    """Get mob name from mob index (0-39)."""
    mob_class = mob_idx // NUM_MOB_TYPES
    mob_type = mob_idx % NUM_MOB_TYPES
    if mob_class in MOB_NAMES:
        return MOB_NAMES[mob_class][mob_type]
    return f"Unknown({mob_idx})"


def obs_to_text(obs: np.ndarray) -> str:
    """
    Convert a Craftax symbolic observation to human-readable text.
    
    This reverses the encoding done by render_craftax_symbolic() to produce
    text output matching the format of render_craftax_text().
    
    Args:
        obs: Symbolic observation array of shape (8268,)
    
    Returns:
        Human-readable text description of the game state
    """
    obs = np.asarray(obs, dtype=np.float32).flatten()
    
    if len(obs) != TOTAL_OBS_SIZE:
        raise ValueError(f"Expected observation of size {TOTAL_OBS_SIZE}, got {len(obs)}")
    
    # Split into map and inventory sections
    map_obs = obs[:MAP_OBS_SIZE]
    inv_obs = obs[MAP_OBS_SIZE:]
    
    # Decode sections
    block_types, item_types, mob_map, light_map = decode_map_section(map_obs)
    decoded = decode_inventory_section(inv_obs)
    
    # Build text output (matching render_craftax_text format)
    # Coordinate system: (Row, Col) where:
    #   - Negative Row is UP, Positive Row is DOWN
    #   - Negative Col is LEFT, Positive Col is RIGHT
    #   - (0, 0) is the player position (center of 9x11 grid)
    text = "Map: "  # Tiles on same line for easier parsing
    
    # x iterates over rows (0-8), y iterates over columns (0-10)
    # OBS_DIM = (9, 11) meaning 9 rows, 11 columns
    # Player is at center: row 4, col 5
    for row_idx in range(OBS_DIM[0]):
        for col_idx in range(OBS_DIM[1]):
            rel_row = row_idx - OBS_DIM[0] // 2  # -4 to +4
            rel_col = col_idx - OBS_DIM[1] // 2  # -5 to +5
            text += f"{rel_row},{rel_col}:"
            
            if light_map[row_idx, col_idx] > 0.5:  # Visible
                # Check for mobs
                mob_present = mob_map[row_idx, col_idx].max() > 0.5
                if mob_present:
                    mob_idx = int(np.argmax(mob_map[row_idx, col_idx]))
                    text += f"{get_mob_name(mob_idx)} on "
                
                # Check for items
                item = item_types[row_idx, col_idx]
                if item != ItemType.NONE.value:
                    text += f"{ItemType(item).name.lower().replace('_', ' ')} on "
                
                # Block type
                block = block_types[row_idx, col_idx]
                text += BlockType(block).name.lower().replace('_', ' ')
            else:
                text += "darkness"
            
            text += ", " if not (row_idx == OBS_DIM[0] - 1 and col_idx == OBS_DIM[1] - 1) else ""
    
    text += "\n"
    
    # Inventory section
    inv = decoded["inventory"]
    text += f"\nInventory: "
    text += f"Wood:{inv['wood']}, Stone:{inv['stone']}, Coal:{inv['coal']}, "
    text += f"Iron:{inv['iron']}, Diamond:{inv['diamond']}, Sapphire:{inv['sapphire']}, "
    text += f"Ruby:{inv['ruby']}, Sapling:{inv['sapling']}, Torch:{inv['torches']}, "
    text += f"Arrow:{inv['arrows']}, Book:{inv['books']}, "
    
    # Potions
    pots = decoded["potions"]
    text += f"Red potion:{pots[0]}, Green potion:{pots[1]}, Blue potion:{pots[2]}, "
    text += f"Pink potion:{pots[3]}, Cyan potion:{pots[4]}, Yellow potion:{pots[5]}, "
    
    # Intrinsics
    intr = decoded["intrinsics"]
    text += f"Health:{intr['health']:.1f}, Food:{intr['food']}, Drink:{intr['drink']}, "
    text += f"Energy:{intr['energy']}, Mana:{intr['mana']}, XP:{intr['xp']}, "
    text += f"Dexterity:{intr['dexterity']}, Strength:{intr['strength']}, Intelligence:{intr['intelligence']}, "
    
    # Direction
    dir_name = Action(decoded["direction"]).name.lower()
    text += f"Direction:{dir_name}, "
    
    # Special values
    sp = decoded["special"]
    text += f"Light:{sp['light_level']:.3f}, "
    text += f"Is Sleeping:{sp['is_sleeping']}, Is Resting:{sp['is_resting']}, "
    text += f"Learned Fireball:{sp['learned_fireball']}, Learned Iceball:{sp['learned_iceball']}, "
    text += f"Floor:{sp['floor']}, Ladder Open:{sp['ladder_open']}, Is Boss Vulnerable:{sp['boss_vulnerable']}"
    
    return text


def obs_to_text_batch(obs_batch: np.ndarray) -> List[str]:
    """
    Convert a batch of observations to text.
    
    Args:
        obs_batch: Array of shape (batch_size, 8268)
    
    Returns:
        List of text strings
    """
    return [obs_to_text(obs) for obs in obs_batch]


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Test with a sample file if provided
    if len(sys.argv) > 1:
        import numpy as np
        data = np.load(sys.argv[1], allow_pickle=True)
        print(f"Loaded {sys.argv[1]}")
        print(f"Keys: {data.files}")
        print(f"obs shape: {data['obs'].shape}")
        
        # Convert first observation
        text = obs_to_text(data['obs'][0])
        print(f"\nFirst observation as text:\n{text}")
    else:
        print("Usage: python obs_to_text.py <npz_file>")
        print("\nTesting with random observation...")
        
        # Create random observation for structure test
        random_obs = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        try:
            text = obs_to_text(random_obs)
            print(f"Random observation decoded (length: {len(text)} chars)")
            print(text[:500] + "...")
        except Exception as e:
            print(f"Error: {e}")
