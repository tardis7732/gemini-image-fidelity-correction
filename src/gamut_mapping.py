from __future__ import annotations

import numpy as np


def hard_clip(base: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = base + delta
    clipped = np.clip(projected, 0.0, 255.0)
    scale = np.ones(base.shape[:2], dtype=np.float32)
    return clipped, scale


def rgb_vector_limit(
    base: np.ndarray,
    delta: np.ndarray,
    shoulder: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale each RGB correction vector so its direction is preserved in gamut."""
    base64 = base.astype(np.float64, copy=False)
    delta64 = delta.astype(np.float64, copy=False)
    positive = delta64 > 1e-12
    negative = delta64 < -1e-12
    channel_scale = np.full(delta64.shape, np.inf, dtype=np.float64)
    np.divide(255.0 - base64, delta64, out=channel_scale, where=positive)
    negative_scale = np.full(delta64.shape, np.inf, dtype=np.float64)
    np.divide(-base64, delta64, out=negative_scale, where=negative)
    channel_scale = np.minimum(channel_scale, negative_scale)
    scale = np.clip(np.min(channel_scale, axis=-1), 0.0, 1.0)
    if shoulder > 0.0:
        limited = scale < 1.0
        scale[limited] *= 1.0 - shoulder * (1.0 - scale[limited])
    mapped = base64 + delta64 * scale[..., None]
    return np.clip(mapped, 0.0, 255.0), scale.astype(np.float32)


def channel_soft_knee(
    base: np.ndarray,
    delta: np.ndarray,
    knee_start: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress outward channel corrections smoothly into the available headroom."""
    base64 = base.astype(np.float64, copy=False)
    delta64 = delta.astype(np.float64, copy=False)
    magnitude = np.abs(delta64)
    room = np.where(delta64 >= 0.0, 255.0 - base64, base64)
    start = knee_start * room
    span = np.maximum(room - start, 1e-12)
    compressed = start + span * np.tanh((magnitude - start) / span)
    mapped_magnitude = np.where(magnitude <= start, magnitude, compressed)
    mapped_magnitude = np.minimum(mapped_magnitude, room)
    mapped_delta = np.copysign(mapped_magnitude, delta64)
    scale = np.ones_like(magnitude)
    np.divide(mapped_magnitude, magnitude, out=scale, where=magnitude > 1e-12)
    mapped = base64 + mapped_delta
    return np.clip(mapped, 0.0, 255.0), scale.astype(np.float32)


def apply_gamut_mapping(
    base: np.ndarray,
    delta: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "hard_clip":
        return hard_clip(base, delta)
    if mode == "rgb_vector":
        return rgb_vector_limit(base, delta, 0.0)
    if mode.startswith("rgb_vector_knee_"):
        shoulder = float(mode.rsplit("_", 1)[1])
        return rgb_vector_limit(base, delta, shoulder)
    if mode.startswith("channel_soft_knee_"):
        knee_start = float(mode.rsplit("_", 1)[1])
        return channel_soft_knee(base, delta, knee_start)
    raise ValueError(f"Unknown gamut mapping mode: {mode}")
