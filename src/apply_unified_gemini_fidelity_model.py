from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from build_unified_gemini_fidelity_model import global_features
from gamut_mapping import apply_gamut_mapping
from reference_free_fidelity_features import (
    deterministic_sample_coords,
    reference_free_features,
)


PERIODS = [2, 4, 8, 16]
EXTRA_AXIS_PERIODS = [(8.0 / 3.0, "8o3")]
DIAGONAL_PERIODS = [(8.0 / 3.0, "8o3"), (4.0, "4"), (8.0, "8"), (16.0, "16")]
COLOR_AXIS_PERIODS = EXTRA_AXIS_PERIODS + [(4.0, "4"), (8.0, "8"), (16.0, "16")]


def lap_at(rgb: np.ndarray, radius: int) -> np.ndarray:
    up = np.roll(rgb, -radius, axis=0)
    down = np.roll(rgb, radius, axis=0)
    left = np.roll(rgb, radius, axis=1)
    right = np.roll(rgb, -radius, axis=1)
    return (up + down + left + right) * 0.25 - rgb


def grad_xy(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.roll(rgb, 1, axis=1)
    right = np.roll(rgb, -1, axis=1)
    up = np.roll(rgb, -1, axis=0)
    down = np.roll(rgb, 1, axis=0)
    return (right - left) * 0.5, (down - up) * 0.5


def coefficient_map(names: np.ndarray, beta: np.ndarray) -> dict[str, np.ndarray]:
    return {str(name): beta[index].astype(np.float32) for index, name in enumerate(names)}


def effective_coefficient(
    coefficients: dict[str, np.ndarray],
    name: str,
    size_log: float,
    aspect_log: float,
) -> np.ndarray:
    value = coefficients[name].copy()
    size_name = f"{name}_x_size_log"
    aspect_name = f"{name}_x_aspect_log"
    if size_name in coefficients:
        value += np.float32(size_log) * coefficients[size_name]
    if aspect_name in coefficients:
        value += np.float32(aspect_log) * coefficients[aspect_name]
    return value


def add_feature(
    correction: np.ndarray,
    feature: np.ndarray,
    coefficients: dict[str, np.ndarray],
    name: str,
    size_log: float,
    aspect_log: float,
) -> None:
    coeff = effective_coefficient(coefficients, name, size_log, aspect_log)
    correction += feature[:, :, None] * coeff[None, None, :]


def broad_prediction(
    output: np.ndarray,
    names: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    height, width = output.shape[:2]
    size_log = math.log(max(width, height) / 1024.0)
    aspect_log = math.log(width / height)
    coefficients = coefficient_map(names, beta)
    correction = np.zeros((height, width, 3), dtype=np.float32)
    correction += coefficients["bias"][None, None, :]
    correction += np.float32(size_log) * coefficients["size_log"][None, None, :]
    correction += np.float32(aspect_log) * coefficients["aspect_log"][None, None, :]

    rgb = output.astype(np.float32) / 255.0
    for channel_index, channel in enumerate("rgb"):
        add_feature(
            correction,
            rgb[:, :, channel_index],
            coefficients,
            f"rgb_{channel}",
            size_log,
            aspect_log,
        )

    lap1 = lap_at(rgb, 1)
    edge1 = np.mean(np.abs(lap1), axis=2).astype(np.float32)
    add_feature(correction, edge1, coefficients, "edge1", size_log, aspect_log)
    for channel_index, channel in enumerate("rgb"):
        add_feature(
            correction,
            lap1[:, :, channel_index],
            coefficients,
            f"lap1_{channel}",
            size_log,
            aspect_log,
        )
        add_feature(
            correction,
            rgb[:, :, channel_index] * edge1,
            coefficients,
            f"rgb_edge1_{channel}",
            size_log,
            aspect_log,
        )
        add_feature(
            correction,
            lap1[:, :, channel_index] * edge1,
            coefficients,
            f"lap1_edge1_{channel}",
            size_log,
            aspect_log,
        )

    lap2 = lap_at(rgb, 2)
    edge2 = np.mean(np.abs(lap2), axis=2).astype(np.float32)
    add_feature(correction, edge2, coefficients, "edge2", size_log, aspect_log)
    for channel_index, channel in enumerate("rgb"):
        add_feature(
            correction,
            lap2[:, :, channel_index],
            coefficients,
            f"lap2_{channel}",
            size_log,
            aspect_log,
        )
    del lap2, edge2

    lap4 = lap_at(rgb, 4)
    for channel_index, channel in enumerate("rgb"):
        add_feature(
            correction,
            lap4[:, :, channel_index],
            coefficients,
            f"lap4_{channel}",
            size_log,
            aspect_log,
        )
    del lap4

    grad_x, grad_y = grad_xy(rgb)
    texture = np.mean(np.abs(grad_x) + np.abs(grad_y), axis=2).astype(np.float32)
    add_feature(correction, texture, coefficients, "texture", size_log, aspect_log)
    for channel_index, channel in enumerate("rgb"):
        add_feature(
            correction,
            grad_x[:, :, channel_index],
            coefficients,
            f"grad_x_{channel}",
            size_log,
            aspect_log,
        )
        add_feature(
            correction,
            grad_y[:, :, channel_index],
            coefficients,
            f"grad_y_{channel}",
            size_log,
            aspect_log,
        )
    del grad_x, grad_y, texture

    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    for period in PERIODS:
        for name, feature in [
            (f"cos_x_{period}", np.broadcast_to(np.cos(2.0 * np.pi * x / period), (height, width))),
            (f"sin_x_{period}", np.broadcast_to(np.sin(2.0 * np.pi * x / period), (height, width))),
            (f"cos_y_{period}", np.broadcast_to(np.cos(2.0 * np.pi * y / period), (height, width))),
            (f"sin_y_{period}", np.broadcast_to(np.sin(2.0 * np.pi * y / period), (height, width))),
        ]:
            add_feature(correction, feature, coefficients, name, size_log, aspect_log)
    for period, label in EXTRA_AXIS_PERIODS:
        for name, feature in [
            (f"cos_x_{label}", np.broadcast_to(np.cos(2.0 * np.pi * x / period), (height, width))),
            (f"sin_x_{label}", np.broadcast_to(np.sin(2.0 * np.pi * x / period), (height, width))),
            (f"cos_y_{label}", np.broadcast_to(np.cos(2.0 * np.pi * y / period), (height, width))),
            (f"sin_y_{label}", np.broadcast_to(np.sin(2.0 * np.pi * y / period), (height, width))),
        ]:
            if name in coefficients:
                add_feature(correction, feature, coefficients, name, size_log, aspect_log)
    if any(name.startswith("cos_diag_") for name in coefficients):
        yy, xx = np.indices((height, width), dtype=np.float32)
        for period, label in DIAGONAL_PERIODS:
            plus = 2.0 * np.pi * (xx + yy) / period
            minus = 2.0 * np.pi * (xx - yy) / period
            for name, feature in [
                (f"cos_diag_plus_{label}", np.cos(plus)),
                (f"sin_diag_plus_{label}", np.sin(plus)),
                (f"cos_diag_minus_{label}", np.cos(minus)),
                (f"sin_diag_minus_{label}", np.sin(minus)),
            ]:
                if name in coefficients:
                    add_feature(
                        correction,
                        feature.astype(np.float32),
                        coefficients,
                        name,
                        size_log,
                        aspect_log,
                    )
    if any("_x_rgb_" in name and ("cos_" in name or "sin_" in name) for name in coefficients):
        yy, xx = np.indices((height, width), dtype=np.float32)
        color_waves: list[tuple[str, np.ndarray]] = []
        for period, label in COLOR_AXIS_PERIODS:
            color_waves.extend(
                [
                    (f"cos_x_{label}", np.cos(2.0 * np.pi * xx / period)),
                    (f"sin_x_{label}", np.sin(2.0 * np.pi * xx / period)),
                    (f"cos_y_{label}", np.cos(2.0 * np.pi * yy / period)),
                    (f"sin_y_{label}", np.sin(2.0 * np.pi * yy / period)),
                ]
            )
        for period, label in DIAGONAL_PERIODS:
            plus = 2.0 * np.pi * (xx + yy) / period
            minus = 2.0 * np.pi * (xx - yy) / period
            color_waves.extend(
                [
                    (f"cos_diag_plus_{label}", np.cos(plus)),
                    (f"sin_diag_plus_{label}", np.sin(plus)),
                    (f"cos_diag_minus_{label}", np.cos(minus)),
                    (f"sin_diag_minus_{label}", np.sin(minus)),
                ]
            )
        for wave_name, wave in color_waves:
            for channel_index, channel in enumerate("rgb"):
                name = f"{wave_name}_x_rgb_{channel}"
                if name in coefficients:
                    add_feature(
                        correction,
                        wave.astype(np.float32) * rgb[:, :, channel_index],
                        coefficients,
                        name,
                        size_log,
                        aspect_log,
                    )
    return correction


def deterministic_samples(output: np.ndarray, count: int = 12000) -> np.ndarray:
    height, width = output.shape[:2]
    ys, xs = deterministic_sample_coords(width, height, count)
    return output[ys, xs]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("air_downloads/reference_free_unified_fidelity_model_20260829/reference_free_unified_fidelity_model_soft_knee075.npz"),
    )
    args = parser.parse_args()

    package = np.load(args.model.resolve(), allow_pickle=False)
    with Image.open(args.input.resolve()) as image:
        output = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = output.shape[:2]

    broad = broad_prediction(
        output,
        package["broad_feature_names"],
        package["broad_beta"].astype(np.float32),
    )
    ys, xs = deterministic_sample_coords(width, height)
    if "zero_mean_broad" in package.files and bool(package["zero_mean_broad"]):
        broad -= broad[ys, xs].mean(axis=0, keepdims=True)[None, :, :]
    output_samples = output[ys, xs]
    global_x = global_features(output_samples, width, height)
    global_dc = global_x @ package["global_beta"].astype(np.float64)
    correction = (
        broad * float(package["broad_strength"])
        + global_dc[None, None, :] * float(package["global_strength"])
    )
    gate_alpha = 1.0
    if "gate_beta" in package.files:
        gate_x = reference_free_features(
            global_x,
            output_samples,
            broad[ys, xs],
            global_dc,
            width,
            height,
            float(package["broad_strength"]),
            float(package["global_strength"]),
        )
        normalized = (
            gate_x - package["gate_feature_mean"].astype(np.float64)
        ) / package["gate_feature_scale"].astype(np.float64)
        raw_alpha = float(
            np.r_[1.0, normalized] @ package["gate_beta"].astype(np.float64)
        )
        gate_alpha = float(
            np.clip(
                1.0 + float(package["gate_strength"]) * (raw_alpha - 1.0),
                0.0,
                float(package["gate_max_alpha"]),
            )
        )
    raw_projected = output + correction * gate_alpha
    raw_overflow_fraction = float(
        np.mean((raw_projected < 0.0) | (raw_projected > 255.0))
    )
    gamut_mode = str(package["gamut_mode"]) if "gamut_mode" in package.files else "hard_clip"
    mapped, mapping_scale = apply_gamut_mapping(
        output,
        correction * gate_alpha,
        gamut_mode,
    )
    mapped_range_violation_fraction = float(
        np.mean((mapped < 0.0) | (mapped > 255.0))
    )
    limited_fraction = float(np.mean(mapping_scale < 1.0))
    corrected = np.clip(
        np.rint(mapped),
        0,
        255,
    ).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrected, "RGB").save(args.output.resolve(), "PNG", optimize=True)
    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "size": f"{width}x{height}",
                "broad_strength": float(package["broad_strength"]),
                "global_strength": float(package["global_strength"]),
                "global_dc_rgb": [round(float(value), 6) for value in global_dc],
                "reference_free_gate": "gate_beta" in package.files,
                "predicted_correction_alpha": round(gate_alpha, 6),
                "gamut_mode": gamut_mode,
                "raw_overflow_fraction": round(raw_overflow_fraction, 8),
                "limited_fraction": round(limited_fraction, 8),
                "mapped_range_violation_fraction": round(mapped_range_violation_fraction, 8),
                "same_model_for_all_images": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
