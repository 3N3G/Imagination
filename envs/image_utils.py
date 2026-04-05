"""
Image format utilities for consistent observation handling across training and evaluation.

All functions handle observations consistently:
- Input observations can be in either 0-1 float or 0-255 uint8 range
- Output is always in the specified format
- Functions include assertions to catch format errors early
"""
import numpy as np
from PIL import Image


def obs_to_01_range(obs_np):
    """
    Convert observation to 0-1 float range, regardless of input format.

    Args:
        obs_np: numpy array, can be:
            - float32/float64 in [0, 1] range
            - uint8 in [0, 255] range
            - float32/float64 in [0, 255] range

    Returns:
        numpy array of float32 in [0, 1] range

    Raises:
        AssertionError: if input values are outside expected ranges
    """
    obs_np = np.array(obs_np)

    # Check if already in 0-1 range
    if obs_np.max() <= 1.0 and obs_np.min() >= 0.0:
        return obs_np.astype(np.float32)

    # Check if in 0-255 range
    if obs_np.max() <= 255.0 and obs_np.min() >= 0.0:
        return (obs_np / 255.0).astype(np.float32)

    raise AssertionError(
        f"Observation values outside expected range [0, 255]: "
        f"min={obs_np.min():.2f}, max={obs_np.max():.2f}"
    )


def obs_to_255_range(obs_np):
    """
    Convert observation to 0-255 uint8 range, regardless of input format.

    Args:
        obs_np: numpy array, can be:
            - float32/float64 in [0, 1] range
            - uint8 in [0, 255] range
            - float32/float64 in [0, 255] range

    Returns:
        numpy array of uint8 in [0, 255] range

    Raises:
        AssertionError: if input values are outside expected ranges
    """
    obs_np = np.array(obs_np)

    # Check if in 0-1 range (needs scaling)
    if obs_np.max() <= 1.0 and obs_np.min() >= 0.0:
        return (obs_np * 255.0).astype(np.uint8)

    # Check if in 0-255 range (just convert dtype)
    if obs_np.max() <= 255.0 and obs_np.min() >= 0.0:
        return obs_np.astype(np.uint8)

    raise AssertionError(
        f"Observation values outside expected range [0, 255]: "
        f"min={obs_np.min():.2f}, max={obs_np.max():.2f}"
    )


def obs_to_pil_image(obs_np):
    """
    Convert observation to PIL Image, regardless of input format.

    Args:
        obs_np: numpy array (H, W, C) in either [0, 1] or [0, 255] range

    Returns:
        PIL Image in RGB mode
    """
    # Convert to 0-255 uint8 first
    obs_uint8 = obs_to_255_range(obs_np)

    # Ensure shape is (H, W, C)
    assert len(obs_uint8.shape) == 3, f"Expected 3D array, got shape {obs_uint8.shape}"
    assert obs_uint8.shape[2] == 3, f"Expected 3 channels, got {obs_uint8.shape[2]}"

    return Image.fromarray(obs_uint8, mode="RGB")


def get_obs_stats(obs_np):
    """
    Get statistics about an observation array for debugging.

    Args:
        obs_np: numpy array

    Returns:
        dict with keys: shape, dtype, min, max, mean, std
    """
    return {
        "shape": obs_np.shape,
        "dtype": str(obs_np.dtype),
        "min": float(obs_np.min()),
        "max": float(obs_np.max()),
        "mean": float(obs_np.mean()),
        "std": float(obs_np.std()),
    }


def verify_obs_format(obs_np, expected_range="auto", name="observation"):
    """
    Verify observation is in expected format and print warning if not.

    Args:
        obs_np: numpy array to verify
        expected_range: '01', '255', or 'auto' (default)
        name: name of the observation for error messages

    Returns:
        bool: True if format is valid

    Raises:
        Warning if format doesn't match expectations
    """
    stats = get_obs_stats(obs_np)

    if expected_range == "01":
        if stats["max"] > 1.0 or stats["min"] < 0.0:
            print(
                f"WARNING: {name} expected in [0,1] but got [{stats['min']:.3f}, {stats['max']:.3f}]"
            )
            return False
    elif expected_range == "255":
        if stats["max"] > 255.0 or stats["min"] < 0.0:
            print(
                f"WARNING: {name} expected in [0,255] but got [{stats['min']:.3f}, {stats['max']:.3f}]"
            )
            return False
    elif expected_range == "auto":
        if not (0.0 <= stats["min"] <= stats["max"] <= 255.0):
            print(
                f"WARNING: {name} outside valid range [0,255]: [{stats['min']:.3f}, {stats['max']:.3f}]"
            )
            return False

    return True
