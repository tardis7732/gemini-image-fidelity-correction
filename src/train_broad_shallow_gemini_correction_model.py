from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SAMPLES_PER_IMAGE = 12000
PERIODS = [2, 4, 8, 16]
EXTRA_AXIS_PERIODS = [(8.0 / 3.0, "8o3")]
DIAGONAL_PERIODS = [(8.0 / 3.0, "8o3"), (4.0, "4"), (8.0, "8"), (16.0, "16")]
COLOR_AXIS_PERIODS = EXTRA_AXIS_PERIODS + [(4.0, "4"), (8.0, "8"), (16.0, "16")]
STRENGTHS = [0.25, 0.5, 0.75, 1.0, 1.25]
MODELS = [
    "legacy_local_periodic",
    "multiscale_edge_aspect_periodic",
    "multiscale_edge_aspect_diagonal_periodic",
    "multiscale_edge_aspect_color_diagonal_periodic",
]
MULTISCALE_MODELS = {
    "multiscale_edge_aspect_periodic",
    "multiscale_edge_aspect_diagonal_periodic",
    "multiscale_edge_aspect_color_diagonal_periodic",
}


@dataclass
class Pair:
    index: int
    dataset: str
    kind: str
    family: str
    shape: str
    size_label: str
    group_id: int
    width: int
    height: int
    base_similarity: float
    reference_path: Path
    output_path: Path
    filename: str


def read_pairs(root: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with (root / "pairs.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            pairs.append(
                Pair(
                    index=i,
                    dataset=row["dataset"],
                    kind=row["kind"],
                    family=row["family"],
                    shape=row["shape"],
                    size_label=row["size_label"],
                    group_id=int(row["group_id"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    base_similarity=float(row["base_similarity"]),
                    reference_path=root / row["reference_path"],
                    output_path=root / row["output_path"],
                    filename=row.get("filename", ""),
                )
            )
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def similarity(ref: np.ndarray, img: np.ndarray) -> float:
    residual = np.clip(np.rint(img), 0, 255).astype(np.float32) - ref.astype(np.float32)
    return float(100.0 * (1.0 - np.mean(np.abs(residual)) / 255.0))


def sampled_similarity(ref: np.ndarray, img_samples: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> float:
    residual = np.clip(np.rint(img_samples), 0, 255).astype(np.float32) - ref[ys, xs].astype(np.float32)
    return float(100.0 * (1.0 - np.mean(np.abs(residual)) / 255.0))


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


def base_feature_names(model: str) -> list[str]:
    names = ["bias"]
    if model == "legacy_local_periodic":
        groups = ["rgb", "lap1", "grad_x", "grad_y"]
        for group in groups:
            names.extend(f"{group}_{channel}" for channel in "rgb")
    elif model in MULTISCALE_MODELS:
        groups = ["rgb", "lap1", "lap2", "lap4", "grad_x", "grad_y"]
        for group in groups:
            names.extend(f"{group}_{channel}" for channel in "rgb")
        names.extend(["edge1", "edge2", "texture"])
        names.extend(f"rgb_edge1_{channel}" for channel in "rgb")
        names.extend(f"lap1_edge1_{channel}" for channel in "rgb")
    else:
        raise ValueError(model)
    for period in PERIODS:
        names.extend([f"cos_x_{period}", f"sin_x_{period}", f"cos_y_{period}", f"sin_y_{period}"])
    if model in {
        "multiscale_edge_aspect_diagonal_periodic",
        "multiscale_edge_aspect_color_diagonal_periodic",
    }:
        for _, label in EXTRA_AXIS_PERIODS:
            names.extend([f"cos_x_{label}", f"sin_x_{label}", f"cos_y_{label}", f"sin_y_{label}"])
        for _, label in DIAGONAL_PERIODS:
            names.extend(
                [
                    f"cos_diag_plus_{label}",
                    f"sin_diag_plus_{label}",
                    f"cos_diag_minus_{label}",
                    f"sin_diag_minus_{label}",
                ]
            )
    if model == "multiscale_edge_aspect_color_diagonal_periodic":
        color_wave_names: list[str] = []
        for _, label in COLOR_AXIS_PERIODS:
            color_wave_names.extend(
                [f"cos_x_{label}", f"sin_x_{label}", f"cos_y_{label}", f"sin_y_{label}"]
            )
        for _, label in DIAGONAL_PERIODS:
            color_wave_names.extend(
                [
                    f"cos_diag_plus_{label}",
                    f"sin_diag_plus_{label}",
                    f"cos_diag_minus_{label}",
                    f"sin_diag_minus_{label}",
                ]
            )
        for wave_name in color_wave_names:
            names.extend(f"{wave_name}_x_rgb_{channel}" for channel in "rgb")
    return names


def feature_names(model: str) -> list[str]:
    base = base_feature_names(model)
    names = list(base)
    names.append("size_log")
    names.extend(f"{name}_x_size_log" for name in base[1:])
    if model in MULTISCALE_MODELS:
        names.append("aspect_log")
        names.extend(f"{name}_x_aspect_log" for name in base[1:])
    return names


def feature_arrays(output: np.ndarray, pair: Pair, model: str) -> list[np.ndarray]:
    rgb = output / 255.0
    lap1 = lap_at(rgb, 1)
    gx, gy = grad_xy(rgb)
    arrays: list[np.ndarray] = []
    if model == "legacy_local_periodic":
        groups = [rgb, lap1, gx, gy]
        for arr in groups:
            for channel_i in range(3):
                arrays.append(arr[:, :, channel_i : channel_i + 1].astype(np.float32))
    elif model in MULTISCALE_MODELS:
        lap2 = lap_at(rgb, 2)
        lap4 = lap_at(rgb, 4)
        groups = [rgb, lap1, lap2, lap4, gx, gy]
        for arr in groups:
            for channel_i in range(3):
                arrays.append(arr[:, :, channel_i : channel_i + 1].astype(np.float32))
        edge1 = np.mean(np.abs(lap1), axis=2, keepdims=True).astype(np.float32)
        edge2 = np.mean(np.abs(lap2), axis=2, keepdims=True).astype(np.float32)
        texture = np.mean(np.abs(gx) + np.abs(gy), axis=2, keepdims=True).astype(np.float32)
        arrays.extend([edge1, edge2, texture])
        for channel_i in range(3):
            arrays.append((rgb[:, :, channel_i : channel_i + 1] * edge1).astype(np.float32))
        for channel_i in range(3):
            arrays.append((lap1[:, :, channel_i : channel_i + 1] * edge1).astype(np.float32))
    else:
        raise ValueError(model)

    yy, xx = np.indices(output.shape[:2], dtype=np.float32)
    for period in PERIODS:
        arrays.extend(
            [
                np.cos(2.0 * np.pi * xx / period)[:, :, None].astype(np.float32),
                np.sin(2.0 * np.pi * xx / period)[:, :, None].astype(np.float32),
                np.cos(2.0 * np.pi * yy / period)[:, :, None].astype(np.float32),
                np.sin(2.0 * np.pi * yy / period)[:, :, None].astype(np.float32),
            ]
        )
    if model in {
        "multiscale_edge_aspect_diagonal_periodic",
        "multiscale_edge_aspect_color_diagonal_periodic",
    }:
        for period, _ in EXTRA_AXIS_PERIODS:
            arrays.extend(
                [
                    np.cos(2.0 * np.pi * xx / period)[:, :, None].astype(np.float32),
                    np.sin(2.0 * np.pi * xx / period)[:, :, None].astype(np.float32),
                    np.cos(2.0 * np.pi * yy / period)[:, :, None].astype(np.float32),
                    np.sin(2.0 * np.pi * yy / period)[:, :, None].astype(np.float32),
                ]
            )
        for period, _ in DIAGONAL_PERIODS:
            plus = 2.0 * np.pi * (xx + yy) / period
            minus = 2.0 * np.pi * (xx - yy) / period
            arrays.extend(
                [
                    np.cos(plus)[:, :, None].astype(np.float32),
                    np.sin(plus)[:, :, None].astype(np.float32),
                    np.cos(minus)[:, :, None].astype(np.float32),
                    np.sin(minus)[:, :, None].astype(np.float32),
                ]
            )
    if model == "multiscale_edge_aspect_color_diagonal_periodic":
        color_waves: list[np.ndarray] = []
        for period, _ in COLOR_AXIS_PERIODS:
            color_waves.extend(
                [
                    np.cos(2.0 * np.pi * xx / period)[:, :, None].astype(np.float32),
                    np.sin(2.0 * np.pi * xx / period)[:, :, None].astype(np.float32),
                    np.cos(2.0 * np.pi * yy / period)[:, :, None].astype(np.float32),
                    np.sin(2.0 * np.pi * yy / period)[:, :, None].astype(np.float32),
                ]
            )
        for period, _ in DIAGONAL_PERIODS:
            plus = 2.0 * np.pi * (xx + yy) / period
            minus = 2.0 * np.pi * (xx - yy) / period
            color_waves.extend(
                [
                    np.cos(plus)[:, :, None].astype(np.float32),
                    np.sin(plus)[:, :, None].astype(np.float32),
                    np.cos(minus)[:, :, None].astype(np.float32),
                    np.sin(minus)[:, :, None].astype(np.float32),
                ]
            )
        for wave in color_waves:
            for channel_i in range(3):
                arrays.append((wave * rgb[:, :, channel_i : channel_i + 1]).astype(np.float32))
    return arrays


def assemble_features(arrays: list[np.ndarray], pair: Pair, ys: np.ndarray | None = None, xs: np.ndarray | None = None, *, model: str) -> np.ndarray:
    if ys is None or xs is None:
        h, w = arrays[0].shape[:2]
        cols = [np.ones((h, w, 1), dtype=np.float32)]
        cols.extend(arrays)
        size_log = np.float32(math.log(max(pair.width, pair.height) / 1024.0))
        base_cols = list(cols)
        cols.append(np.ones((h, w, 1), dtype=np.float32) * size_log)
        cols.extend([col * size_log for col in base_cols[1:]])
        if model in MULTISCALE_MODELS:
            aspect_log = np.float32(math.log(pair.width / pair.height))
            cols.append(np.ones((h, w, 1), dtype=np.float32) * aspect_log)
            cols.extend([col * aspect_log for col in base_cols[1:]])
        return np.concatenate(cols, axis=2)

    cols_2d: list[np.ndarray] = [np.ones(len(xs), dtype=np.float32)]
    for arr in arrays:
        cols_2d.append(arr[ys, xs, 0].astype(np.float32))
    size_log = np.float32(math.log(max(pair.width, pair.height) / 1024.0))
    base_cols = list(cols_2d)
    cols_2d.append(np.ones(len(xs), dtype=np.float32) * size_log)
    cols_2d.extend([col * size_log for col in base_cols[1:]])
    if model in MULTISCALE_MODELS:
        aspect_log = np.float32(math.log(pair.width / pair.height))
        cols_2d.append(np.ones(len(xs), dtype=np.float32) * aspect_log)
        cols_2d.extend([col * aspect_log for col in base_cols[1:]])
    return np.stack(cols_2d, axis=1)


def sample_feature_matrix(output: np.ndarray, pair: Pair, ys: np.ndarray, xs: np.ndarray, model: str) -> np.ndarray:
    rgb = output / 255.0
    lap1 = lap_at(rgb, 1)
    gx, gy = grad_xy(rgb)
    cols: list[np.ndarray] = [np.ones(len(xs), dtype=np.float32)]

    if model == "legacy_local_periodic":
        groups = [rgb, lap1, gx, gy]
        for arr in groups:
            for channel_i in range(3):
                cols.append(arr[ys, xs, channel_i].astype(np.float32))
    elif model in MULTISCALE_MODELS:
        lap2 = lap_at(rgb, 2)
        lap4 = lap_at(rgb, 4)
        groups = [rgb, lap1, lap2, lap4, gx, gy]
        for arr in groups:
            for channel_i in range(3):
                cols.append(arr[ys, xs, channel_i].astype(np.float32))
        edge1 = np.mean(np.abs(lap1), axis=2).astype(np.float32)
        edge2 = np.mean(np.abs(lap2), axis=2).astype(np.float32)
        texture = np.mean(np.abs(gx) + np.abs(gy), axis=2).astype(np.float32)
        cols.extend([edge1[ys, xs], edge2[ys, xs], texture[ys, xs]])
        for channel_i in range(3):
            cols.append((rgb[:, :, channel_i] * edge1)[ys, xs].astype(np.float32))
        for channel_i in range(3):
            cols.append((lap1[:, :, channel_i] * edge1)[ys, xs].astype(np.float32))
    else:
        raise ValueError(model)

    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    for period in PERIODS:
        cols.extend(
            [
                np.cos(2.0 * np.pi * xs_f / period).astype(np.float32),
                np.sin(2.0 * np.pi * xs_f / period).astype(np.float32),
                np.cos(2.0 * np.pi * ys_f / period).astype(np.float32),
                np.sin(2.0 * np.pi * ys_f / period).astype(np.float32),
            ]
        )

    if model in {
        "multiscale_edge_aspect_diagonal_periodic",
        "multiscale_edge_aspect_color_diagonal_periodic",
    }:
        for period, _ in EXTRA_AXIS_PERIODS:
            cols.extend(
                [
                    np.cos(2.0 * np.pi * xs_f / period).astype(np.float32),
                    np.sin(2.0 * np.pi * xs_f / period).astype(np.float32),
                    np.cos(2.0 * np.pi * ys_f / period).astype(np.float32),
                    np.sin(2.0 * np.pi * ys_f / period).astype(np.float32),
                ]
            )
        for period, _ in DIAGONAL_PERIODS:
            plus = 2.0 * np.pi * (xs_f + ys_f) / period
            minus = 2.0 * np.pi * (xs_f - ys_f) / period
            cols.extend(
                [
                    np.cos(plus).astype(np.float32),
                    np.sin(plus).astype(np.float32),
                    np.cos(minus).astype(np.float32),
                    np.sin(minus).astype(np.float32),
                ]
            )
    if model == "multiscale_edge_aspect_color_diagonal_periodic":
        color_waves_1d: list[np.ndarray] = []
        for period, _ in COLOR_AXIS_PERIODS:
            color_waves_1d.extend(
                [
                    np.cos(2.0 * np.pi * xs_f / period).astype(np.float32),
                    np.sin(2.0 * np.pi * xs_f / period).astype(np.float32),
                    np.cos(2.0 * np.pi * ys_f / period).astype(np.float32),
                    np.sin(2.0 * np.pi * ys_f / period).astype(np.float32),
                ]
            )
        for period, _ in DIAGONAL_PERIODS:
            plus = 2.0 * np.pi * (xs_f + ys_f) / period
            minus = 2.0 * np.pi * (xs_f - ys_f) / period
            color_waves_1d.extend(
                [
                    np.cos(plus).astype(np.float32),
                    np.sin(plus).astype(np.float32),
                    np.cos(minus).astype(np.float32),
                    np.sin(minus).astype(np.float32),
                ]
            )
        sampled_rgb = rgb[ys, xs]
        for wave in color_waves_1d:
            for channel_i in range(3):
                cols.append((wave * sampled_rgb[:, channel_i]).astype(np.float32))

    size_log = np.float32(math.log(max(pair.width, pair.height) / 1024.0))
    base_cols = list(cols)
    cols.append(np.ones(len(xs), dtype=np.float32) * size_log)
    cols.extend([col * size_log for col in base_cols[1:]])
    if model in MULTISCALE_MODELS:
        aspect_log = np.float32(math.log(pair.width / pair.height))
        cols.append(np.ones(len(xs), dtype=np.float32) * aspect_log)
        cols.extend([col * aspect_log for col in base_cols[1:]])
    return np.stack(cols, axis=1)


def sample_coords(pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    n = min(SAMPLES_PER_IMAGE, pair.width * pair.height)
    rng = np.random.default_rng(20260829 + pair.index * 31 + pair.width)
    idx = rng.choice(pair.width * pair.height, size=n, replace=False)
    return (idx // pair.width).astype(np.int64), (idx % pair.width).astype(np.int64)


def pair_stats(pair: Pair, ref: np.ndarray, out: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = sample_coords(pair)
    x = sample_feature_matrix(out, pair, ys, xs, model).astype(np.float64)
    y = (ref[ys, xs] - out[ys, xs]).astype(np.float64)
    return x.T @ x, x.T @ y


def fit_beta(xtx: np.ndarray, xty: np.ndarray, alpha: float) -> np.ndarray:
    reg = np.eye(xtx.shape[0], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, xty).astype(np.float32)


def apply_beta(pair: Pair, out: np.ndarray, beta: np.ndarray, model: str) -> np.ndarray:
    arrays = feature_arrays(out, pair, model)
    features = assemble_features(arrays, pair, model=model)
    return features @ beta


def apply_beta_samples(pair: Pair, out: np.ndarray, beta: np.ndarray, model: str, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    features = sample_feature_matrix(out, pair, ys, xs, model)
    return features @ beta


def train_filter(pairs: list[Pair], source: str) -> list[Pair]:
    if source == "solid_all":
        return [p for p in pairs if p.kind == "solid"]
    if source == "square_car_only":
        return [p for p in pairs if p.kind == "car" and p.shape == "1x1"]
    if source == "wide_car_only":
        return [p for p in pairs if p.kind == "car" and p.shape == "wide"]
    if source == "car_all":
        return [p for p in pairs if p.kind == "car"]
    if source == "car_plus_solid_all":
        return [p for p in pairs if p.kind in {"car", "solid"}]
    raise ValueError(source)


def eval_filter(pairs: list[Pair], split: str) -> list[Pair]:
    if split == "car_all":
        return [p for p in pairs if p.kind == "car"]
    if split == "car_wide":
        return [p for p in pairs if p.kind == "car" and p.shape == "wide"]
    if split == "car_square":
        return [p for p in pairs if p.kind == "car" and p.shape == "1x1"]
    if split == "car_wide_pass97":
        return [p for p in pairs if p.kind == "car" and p.shape == "wide" and p.base_similarity >= 97.0]
    if split == "car_wide_high985":
        return [p for p in pairs if p.kind == "car" and p.shape == "wide" and p.base_similarity >= 98.5]
    raise ValueError(split)


def split_label(pair: Pair) -> str:
    if pair.kind == "car" and pair.shape == "wide":
        return "car_wide"
    if pair.kind == "car":
        return "car_square"
    return "solid"


def summarize(rows: list[dict[str, object]], split: str, strength: float) -> dict[str, object]:
    subset = [r for r in rows if r["eval_split"] == split and abs(float(r["strength"]) - strength) < 1e-9]
    if not subset:
        return {}
    base = np.asarray([float(r["base_similarity"]) for r in subset], dtype=np.float64)
    corr = np.asarray([float(r["corrected_similarity"]) for r in subset], dtype=np.float64)
    delta = corr - base
    return {
        "eval_split": split,
        "strength": strength,
        "samples": len(subset),
        "base_mean": round(float(base.mean()), 6),
        "corrected_mean": round(float(corr.mean()), 6),
        "mean_delta": round(float(delta.mean()), 6),
        "median_delta": round(float(np.median(delta)), 6),
        "min_delta": round(float(delta.min()), 6),
        "max_delta": round(float(delta.max()), 6),
        "improved_images": int(np.sum(delta > 0)),
        "worse_images": int(np.sum(delta < 0)),
    }


def evaluate_predictions(
    pairs: list[Pair],
    loaded: dict[int, tuple[np.ndarray, np.ndarray]],
    betas: dict[int, np.ndarray] | np.ndarray,
    model: str,
    scenario: str,
    eval_pairs: list[Pair],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in eval_pairs:
        ref, out = loaded[pair.index]
        beta = betas[pair.index] if isinstance(betas, dict) else betas
        ys, xs = sample_coords(pair)
        pred = apply_beta_samples(pair, out, beta, model, ys, xs)
        out_samples = out[ys, xs]
        base = sampled_similarity(ref, out_samples, ys, xs)
        for strength in STRENGTHS:
            corrected_samples = np.clip(np.rint(out_samples + pred * strength), 0, 255).astype(np.float32)
            score = sampled_similarity(ref, corrected_samples, ys, xs)
            rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "eval_split": split_label(pair),
                    "dataset": pair.dataset,
                    "shape": pair.shape,
                    "size_label": pair.size_label,
                    "group_id": pair.group_id,
                    "strength": strength,
                    "base_similarity": round(base, 6),
                    "csv_base_similarity": round(pair.base_similarity, 6),
                    "corrected_similarity": round(score, 6),
                    "delta": round(score - base, 6),
                }
            )
    return rows


def save_examples(
    out_dir: Path,
    rows: list[dict[str, object]],
    pairs: list[Pair],
    loaded: dict[int, tuple[np.ndarray, np.ndarray]],
    betas: dict[int, np.ndarray] | np.ndarray,
    model: str,
    scenario: str,
    strength: float,
    limit: int = 18,
) -> None:
    by_key = {(r["dataset"], int(r["group_id"]), float(r["strength"])): r for r in rows}
    pair_map = {(p.dataset, p.group_id): p for p in pairs}
    selected = [
        r for r in rows
        if r["scenario"] == scenario and r["model"] == model and r["eval_split"] == "car_wide" and abs(float(r["strength"]) - strength) < 1e-9
    ]
    if not selected:
        return
    selected = sorted(selected, key=lambda r: float(r["delta"]))[: limit // 2] + sorted(selected, key=lambda r: float(r["delta"]), reverse=True)[: limit // 2]
    example_dir = out_dir / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[Image.Image] = []
    for row in selected:
        pair = pair_map[(row["dataset"], int(row["group_id"]))]
        ref, out = loaded[pair.index]
        beta = betas[pair.index] if isinstance(betas, dict) else betas
        pred = apply_beta(pair, out, beta, model)
        corrected = np.clip(np.rint(out + pred * strength), 0, 255).astype(np.uint8)
        panels = []
        for arr, label in [
            (ref.astype(np.uint8), f"ref {pair.group_id}"),
            (out.astype(np.uint8), f"gemini {float(row['base_similarity']):.4f}"),
            (corrected, f"corrected {float(row['corrected_similarity']):.4f} ({float(row['delta']):+.4f})"),
        ]:
            image = Image.fromarray(arr, "RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
            draw.text((9, 9), label, fill=(255, 255, 255))
            panels.append(image)
        sheet = Image.new("RGB", (panels[0].width * 3, panels[0].height), (255, 255, 255))
        for i, panel in enumerate(panels):
            sheet.paste(panel, (i * panel.width, 0))
        path = example_dir / f"{scenario}_{model}_s{strength:g}_{pair.group_id:03d}_{float(row['delta']):+.4f}.png"
        sheet.save(path)
        thumbs.append(sheet.resize((900, int(round(sheet.height * 900 / sheet.width))), Image.Resampling.LANCZOS))
    if thumbs:
        cols = 2
        tw, th = thumbs[0].size
        contact = Image.new("RGB", (tw * cols, th * math.ceil(len(thumbs) / cols)), (255, 255, 255))
        for i, thumb in enumerate(thumbs):
            contact.paste(thumb, ((i % cols) * tw, (i // cols) * th))
        contact.save(out_dir / f"{scenario}_{model}_s{strength:g}_wide_extremes_contact.png")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--out-dir", default="outputs/broad_shallow_model")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--examples-per-best", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
        help="Model variants to evaluate.",
    )
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(root)
    loaded = {pair.index: (load_rgb(pair.reference_path), load_rgb(pair.output_path)) for pair in pairs}
    all_rows: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    saved_models: list[dict[str, object]] = []

    scenarios = [
        ("solid_all_fit", "solid_all", False, "car_all"),
        ("square_car_fit", "square_car_only", False, "car_wide"),
        ("car_all_loo", "car_all", True, "car_all"),
        ("car_plus_solid_all_loo", "car_plus_solid_all", True, "car_all"),
        ("wide_car_loo", "wide_car_only", True, "car_wide"),
    ]

    for model in args.models:
        names = feature_names(model)
        print(f"model={model} features={len(names)}", flush=True)
        stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, pair in enumerate(pairs, start=1):
            ref, out = loaded[pair.index]
            stats[pair.index] = pair_stats(pair, ref, out, model)
            if i % 25 == 0 or i == len(pairs):
                print(f"stats {model} {i}/{len(pairs)}", flush=True)

        for scenario, source, loo, eval_split_name in scenarios:
            train_pairs = train_filter(pairs, source)
            eval_pairs = eval_filter(pairs, eval_split_name)
            train_indices = {p.index for p in train_pairs}
            total_xtx = sum((stats[i][0] for i in train_indices), np.zeros((len(names), len(names)), dtype=np.float64))
            total_xty = sum((stats[i][1] for i in train_indices), np.zeros((len(names), 3), dtype=np.float64))
            if loo:
                betas: dict[int, np.ndarray] = {}
                for pair in eval_pairs:
                    if pair.index in train_indices:
                        xtx = total_xtx - stats[pair.index][0]
                        xty = total_xty - stats[pair.index][1]
                    else:
                        xtx = total_xtx
                        xty = total_xty
                    betas[pair.index] = fit_beta(xtx, xty, args.alpha)
            else:
                betas = fit_beta(total_xtx, total_xty, args.alpha)
            rows = evaluate_predictions(pairs, loaded, betas, model, scenario, eval_pairs)
            all_rows.extend(rows)

            for split in ["car_square", "car_wide"]:
                for strength in STRENGTHS:
                    summary = summarize(rows, split, strength)
                    if summary:
                        summary.update({"scenario": scenario, "model": model, "train_source": source, "loo": loo})
                        all_summary.append(summary)
                        print(json.dumps(summary), flush=True)

            best_wide = max(
                [s for s in all_summary if s["scenario"] == scenario and s["model"] == model and s["eval_split"] == "car_wide"],
                key=lambda r: float(r["mean_delta"]),
                default=None,
            )
            if best_wide and args.examples_per_best > 0:
                save_examples(out_dir, rows, pairs, loaded, betas, model, scenario, float(best_wide["strength"]), limit=args.examples_per_best)

            if not loo:
                model_path = out_dir / f"{scenario}_{model}.npz"
                np.savez(model_path, beta=betas, feature_names=np.asarray(names), model=model, scenario=scenario, source=source)
                saved_models.append({"scenario": scenario, "model": model, "path": str(model_path)})

    write_csv(out_dir / "prediction_details.csv", all_rows)
    write_csv(out_dir / "summary_metrics.csv", all_summary)
    best_by_split = {}
    for split in sorted({row["eval_split"] for row in all_summary}):
        candidates = [row for row in all_summary if row["eval_split"] == split]
        best_by_split[split] = max(candidates, key=lambda row: float(row["mean_delta"]))
    summary_doc = {
        "data_root": str(root),
        "pairs": len(pairs),
        "counts": {
            "solid": sum(1 for p in pairs if p.kind == "solid"),
            "car": sum(1 for p in pairs if p.kind == "car"),
            "car_square": sum(1 for p in pairs if p.kind == "car" and p.shape == "1x1"),
            "car_wide": sum(1 for p in pairs if p.kind == "car" and p.shape == "wide"),
        },
        "best_by_split": best_by_split,
        "saved_models": saved_models,
        "files": {
            "prediction_details": str(out_dir / "prediction_details.csv"),
            "summary_metrics": str(out_dir / "summary_metrics.csv"),
            "examples": str(out_dir / "examples"),
        },
        "note": "Metrics use deterministic sampled pixels per image for speed; csv_base_similarity keeps the original full-image baseline.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_doc, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
