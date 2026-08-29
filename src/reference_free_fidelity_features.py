from __future__ import annotations

import math

import numpy as np


SAMPLE_COUNT = 12000


def deterministic_sample_coords(
    width: int,
    height: int,
    count: int = SAMPLE_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(count, width * height)
    rng = np.random.default_rng(20260829 + width)
    indices = rng.choice(width * height, size=count, replace=False)
    return indices // width, indices % width


def reference_free_feature_names(base_names: list[str]) -> list[str]:
    names = list(base_names)
    for prefix in ["output", "broad", "correction"]:
        for statistic in ["mean", "std", "mean_abs", "rms", "p50_abs", "p90_abs", "p99_abs"]:
            names.extend(f"{prefix}_{statistic}_{channel}" for channel in "rgb")
    names.extend(f"output_p{percentile}_{channel}" for percentile in [1, 5, 25, 50, 75, 95, 99] for channel in "rgb")
    names.extend(f"correction_positive_{channel}" for channel in "rgb")
    names.extend(f"correction_negative_{channel}" for channel in "rgb")
    names.extend(f"correction_output_corr_{channel}" for channel in "rgb")
    names.extend(f"projected_clip_{channel}" for channel in "rgb")
    names.extend(
        [
            "projected_clip_all",
            "output_luma_mean",
            "output_luma_std",
            "output_chroma_mean",
            "output_chroma_std",
            "output_near_black",
            "output_near_white",
            "correction_luma_mean_abs",
            "correction_chroma_mean_abs",
            "size_log_repeat",
            "aspect_log_repeat",
        ]
    )
    return names


def _channel_statistics(values: np.ndarray) -> list[float]:
    return [
        *values.mean(axis=0),
        *values.std(axis=0),
        *np.mean(np.abs(values), axis=0),
        *np.sqrt(np.mean(values**2, axis=0)),
        *np.percentile(np.abs(values), 50, axis=0),
        *np.percentile(np.abs(values), 90, axis=0),
        *np.percentile(np.abs(values), 99, axis=0),
    ]


def reference_free_features(
    base_features: np.ndarray,
    output_samples: np.ndarray,
    broad_samples: np.ndarray,
    global_dc: np.ndarray,
    width: int,
    height: int,
    broad_strength: float,
    global_strength: float,
) -> np.ndarray:
    output = output_samples.astype(np.float64)
    broad = broad_samples.astype(np.float64) * broad_strength
    correction = broad + global_dc[None, :] * global_strength
    values: list[float] = [*base_features.astype(np.float64)]
    values.extend(_channel_statistics(output / 255.0))
    values.extend(_channel_statistics(broad / 16.0))
    values.extend(_channel_statistics(correction / 16.0))

    for percentile in [1, 5, 25, 50, 75, 95, 99]:
        values.extend(np.percentile(output / 255.0, percentile, axis=0))
    values.extend(np.mean(correction > 0, axis=0))
    values.extend(np.mean(correction < 0, axis=0))
    for channel in range(3):
        x = output[:, channel]
        y = correction[:, channel]
        if float(x.std()) < 1e-8 or float(y.std()) < 1e-8:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(x, y)[0, 1]))

    projected = output + correction
    projected_clip = (projected < 0.0) | (projected > 255.0)
    values.extend(projected_clip.mean(axis=0))
    values.append(float(projected_clip.mean()))

    normalized = output / 255.0
    luma = normalized @ np.asarray([0.2126, 0.7152, 0.0722])
    chroma = normalized.max(axis=1) - normalized.min(axis=1)
    correction_luma = correction @ np.asarray([0.2126, 0.7152, 0.0722])
    correction_chroma = correction.max(axis=1) - correction.min(axis=1)
    values.extend(
        [
            float(luma.mean()),
            float(luma.std()),
            float(chroma.mean()),
            float(chroma.std()),
            float(np.mean(luma < 4.0 / 255.0)),
            float(np.mean(luma > 251.0 / 255.0)),
            float(np.mean(np.abs(correction_luma)) / 16.0),
            float(np.mean(np.abs(correction_chroma)) / 16.0),
            math.log(max(width, height) / 1024.0),
            math.log(width / height),
        ]
    )
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Reference-free features contain non-finite values")
    return result
