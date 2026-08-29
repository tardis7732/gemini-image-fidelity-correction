from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from train_broad_shallow_gemini_correction_model import (
    Pair,
    apply_beta_samples,
    load_rgb,
    read_pairs,
    sample_coords,
    sampled_similarity,
)


BROAD_STRENGTHS = [0.25, 0.5, 0.75, 1.0]
HEAD_ALPHAS = [0.01, 0.1, 1.0, 10.0]
HEAD_STRENGTHS = [0.5, 0.75, 1.0]
FOLDS = 5
OUTPUT_RE = re.compile(
    r"^(?P<source>.+)_(?:r\d+|extra\d+_for_r\d+)_native_(?:1024|2048)\.png$",
    re.IGNORECASE,
)
HEX_RE = re.compile(r"([0-9A-Fa-f]{6})(?:\.[^.]+)?$")
CANDIDATE_RE = re.compile(r"candidate(\d+)", re.IGNORECASE)


@dataclass
class CachedPair:
    pair: Pair
    category: str
    fold: int
    features: np.ndarray
    ref_samples: np.ndarray
    output_samples: np.ndarray
    broad_prediction: np.ndarray
    base_similarity: float


def global_feature_names() -> list[str]:
    names = ["bias"]
    names.extend(f"mean_{channel}" for channel in "rgb")
    names.extend(f"mean2_{channel}" for channel in "rgb")
    names.extend(f"mean3_{channel}" for channel in "rgb")
    names.extend(f"std_{channel}" for channel in "rgb")
    names.extend(f"near_zero_{channel}" for channel in "rgb")
    names.extend(f"near_one_{channel}" for channel in "rgb")
    names.extend(["flatness", "size_log", "aspect_log"])
    names.extend(f"flat_mean_{channel}" for channel in "rgb")
    names.extend(f"flat_mean2_{channel}" for channel in "rgb")
    names.extend(f"flat_near_zero_{channel}" for channel in "rgb")
    names.extend(f"flat_near_one_{channel}" for channel in "rgb")
    names.extend(["flat_size_log", "flat_aspect_log"])
    return names


def global_features(output_samples: np.ndarray, width: int, height: int) -> np.ndarray:
    rgb = output_samples.astype(np.float64) / 255.0
    mean = rgb.mean(axis=0)
    std = rgb.std(axis=0)
    near_zero = np.exp(-mean / 0.04)
    near_one = np.exp(-(1.0 - mean) / 0.04)
    flatness = float(np.exp(-float(std.mean()) / 0.035))
    size_log = math.log(max(width, height) / 1024.0)
    aspect_log = math.log(width / height)
    values = np.concatenate(
        [
            np.ones(1),
            mean,
            mean**2,
            mean**3,
            std,
            near_zero,
            near_one,
            np.asarray([flatness, size_log, aspect_log]),
            flatness * mean,
            flatness * mean**2,
            flatness * near_zero,
            flatness * near_one,
            np.asarray([flatness * size_log, flatness * aspect_log]),
        ]
    )
    return values.astype(np.float64)


def group_key(pair: Pair) -> str:
    if pair.kind == "solid":
        output_match = OUTPUT_RE.match(pair.filename)
        solid_name = output_match.group("source") if output_match else pair.filename
        match = HEX_RE.search(solid_name)
        if match:
            return f"solid:{match.group(1).lower()}"
    if pair.kind == "car":
        match = CANDIDATE_RE.search(pair.filename)
        if match:
            return f"car:candidate{int(match.group(1)):04d}"
    return f"{pair.kind}:{pair.dataset}:{pair.group_id}:{pair.filename.lower()}"


def fold_for_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % FOLDS


def load_extreme_pairs(root: Path, start_index: int) -> list[Pair]:
    specs = [
        ("1k_1x1", "1K", 1024, root / "sources" / "1k_1x1", root / "1k_1x1" / "normalized_1024"),
        ("2k_1x1", "2K", 2048, root / "sources" / "2k_1x1", root / "2k_1x1" / "normalized_2048"),
    ]
    pairs: list[Pair] = []
    next_index = start_index
    for run_id, size_label, expected_size, source_dir, output_dir in specs:
        for output_path in sorted(output_dir.glob("*.png")):
            match = OUTPUT_RE.match(output_path.name)
            if not match:
                continue
            source_stem = match.group("source")
            source_path = source_dir / f"{source_stem}.png"
            if not source_path.exists():
                continue
            with Image.open(output_path) as image:
                width, height = image.size
            if (width, height) != (expected_size, expected_size):
                continue
            pairs.append(
                Pair(
                    index=next_index,
                    dataset=f"extreme_{run_id}",
                    kind="solid",
                    family="extreme_solid",
                    shape="1x1",
                    size_label=size_label,
                    group_id=next_index - start_index + 1,
                    width=width,
                    height=height,
                    base_similarity=0.0,
                    reference_path=source_path,
                    output_path=output_path,
                    filename=output_path.name,
                )
            )
            next_index += 1
    return pairs


def category_for(pair: Pair) -> str:
    if pair.kind == "car":
        return "car"
    if pair.family == "extreme_solid":
        return "solid_extreme"
    return "solid_random"


def cache_pair(pair: Pair, broad_beta: np.ndarray, model_name: str) -> CachedPair:
    ref = load_rgb(pair.reference_path)
    output = load_rgb(pair.output_path)
    ys, xs = sample_coords(pair)
    ref_samples = ref[ys, xs].astype(np.float32)
    output_samples = output[ys, xs].astype(np.float32)
    broad_prediction = apply_beta_samples(pair, output, broad_beta, model_name, ys, xs).astype(np.float32)
    return CachedPair(
        pair=pair,
        category=category_for(pair),
        fold=fold_for_key(group_key(pair)),
        features=global_features(output_samples, pair.width, pair.height),
        ref_samples=ref_samples,
        output_samples=output_samples,
        broad_prediction=broad_prediction,
        base_similarity=sampled_similarity(ref, output_samples, ys, xs),
    )


def target_dc(item: CachedPair, broad_strength: float) -> np.ndarray:
    broad = np.clip(
        np.rint(item.output_samples + item.broad_prediction * broad_strength), 0, 255
    )
    return (item.ref_samples - broad).mean(axis=0).astype(np.float64)


def fit_head(
    items: list[CachedPair],
    broad_strength: float,
    alpha: float,
) -> np.ndarray:
    x = np.stack([item.features for item in items], axis=0)
    y = np.stack([target_dc(item, broad_strength) for item in items], axis=0)
    categories = sorted({item.category for item in items})
    counts = {category: sum(item.category == category for item in items) for category in categories}
    weights = np.asarray(
        [len(items) / (len(categories) * counts[item.category]) for item in items],
        dtype=np.float64,
    )
    xw = x * np.sqrt(weights[:, None])
    yw = y * np.sqrt(weights[:, None])
    reg = np.eye(x.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)


def evaluate_item(
    item: CachedPair,
    broad_strength: float,
    head_beta: np.ndarray,
    head_strength: float,
) -> tuple[float, float]:
    dc = item.features @ head_beta
    corrected = np.clip(
        np.rint(
            item.output_samples
            + item.broad_prediction * broad_strength
            + dc[None, :] * head_strength
        ),
        0,
        255,
    )
    residual = corrected.astype(np.float32) - item.ref_samples
    score = float(100.0 * (1.0 - np.mean(np.abs(residual)) / 255.0))
    return score, score - item.base_similarity


def summarize_details(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for category in ["car", "solid_random", "solid_extreme", "all"]:
        subset = rows if category == "all" else [row for row in rows if row["category"] == category]
        base = np.asarray([float(row["base_similarity"]) for row in subset], dtype=np.float64)
        corrected = np.asarray([float(row["corrected_similarity"]) for row in subset], dtype=np.float64)
        delta = corrected - base
        summaries.append(
            {
                "category": category,
                "samples": len(subset),
                "base_mean": round(float(base.mean()), 6),
                "corrected_mean": round(float(corrected.mean()), 6),
                "mean_delta": round(float(delta.mean()), 6),
                "median_delta": round(float(np.median(delta)), 6),
                "min_delta": round(float(delta.min()), 6),
                "max_delta": round(float(delta.max()), 6),
                "improved": int(np.sum(delta > 0)),
                "worse": int(np.sum(delta < 0)),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="air_upload/gemini_correction_broad_bundle_20260829")
    parser.add_argument(
        "--extreme-root",
        default="air_downloads/extreme_solid_repeat_gemini_flash_1k_2k_20260829",
    )
    parser.add_argument(
        "--broad-model",
        default="air_downloads/gemini_correction_broad_model_20260829/broad_deploy_models/car_plus_solid_all_fit_multiscale_edge_aspect_periodic.npz",
    )
    parser.add_argument(
        "--out-dir",
        default="air_downloads/unified_gemini_fidelity_model_20260829",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    extreme_root = Path(args.extreme_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    broad_package = np.load(Path(args.broad_model).resolve(), allow_pickle=False)
    broad_beta = broad_package["beta"].astype(np.float32)
    model_name = str(broad_package["model"])

    broad_pairs = read_pairs(data_root)
    extreme_pairs = load_extreme_pairs(extreme_root, len(broad_pairs))
    broad_cached: list[CachedPair] = []
    for i, pair in enumerate(broad_pairs, start=1):
        broad_cached.append(cache_pair(pair, broad_beta, model_name))
        if i % 20 == 0 or i == len(broad_pairs):
            print(f"cached broad {i}/{len(broad_pairs)}", flush=True)

    extreme_cached: list[CachedPair] = []
    for i, pair in enumerate(extreme_pairs, start=1):
        extreme_cached.append(cache_pair(pair, broad_beta, model_name))
        if i % 10 == 0 or i == len(extreme_pairs):
            print(f"cached extreme {i}/{len(extreme_pairs)}", flush=True)

    all_cached = broad_cached + extreme_cached
    candidate_rows: list[dict[str, object]] = []
    candidate_details: dict[tuple[float, float, float], list[dict[str, object]]] = {}
    for broad_strength in BROAD_STRENGTHS:
        for alpha in HEAD_ALPHAS:
            fold_betas = {
                fold: fit_head(
                    [item for item in all_cached if item.fold != fold],
                    broad_strength,
                    alpha,
                )
                for fold in range(FOLDS)
            }
            for head_strength in HEAD_STRENGTHS:
                details: list[dict[str, object]] = []
                for item in all_cached:
                    score, delta = evaluate_item(
                        item,
                        broad_strength,
                        fold_betas[item.fold],
                        head_strength,
                    )
                    details.append(
                        {
                            "dataset": item.pair.dataset,
                            "filename": item.pair.filename,
                            "category": item.category,
                            "evaluation": "grouped_5fold",
                            "base_similarity": round(item.base_similarity, 6),
                            "corrected_similarity": round(score, 6),
                            "delta": round(delta, 6),
                        }
                    )
                summaries = summarize_details(details)
                by_category = {row["category"]: row for row in summaries}
                category_deltas = [
                    float(by_category[name]["mean_delta"])
                    for name in ["car", "solid_random", "solid_extreme"]
                ]
                balanced_delta = float(np.mean(category_deltas))
                worst_category_delta = float(min(category_deltas))
                improved_fraction = float(
                    sum(float(row["delta"]) > 0 for row in details) / len(details)
                )
                candidate_rows.append(
                    {
                        "broad_strength": broad_strength,
                        "head_alpha": alpha,
                        "head_strength": head_strength,
                        "car_delta": category_deltas[0],
                        "solid_random_delta": category_deltas[1],
                        "solid_extreme_delta": category_deltas[2],
                        "balanced_delta": round(balanced_delta, 6),
                        "worst_category_delta": round(worst_category_delta, 6),
                        "improved_fraction": round(improved_fraction, 6),
                    }
                )
                candidate_details[(broad_strength, alpha, head_strength)] = details
                print(json.dumps(candidate_rows[-1]), flush=True)

    viable = [
        row
        for row in candidate_rows
        if min(
            float(row["car_delta"]),
            float(row["solid_random_delta"]),
            float(row["solid_extreme_delta"]),
        )
        > 0
    ]
    pool = viable or candidate_rows
    best = max(
        pool,
        key=lambda row: (
            float(row["balanced_delta"]),
            float(row["worst_category_delta"]),
            float(row["improved_fraction"]),
        ),
    )
    key = (
        float(best["broad_strength"]),
        float(best["head_alpha"]),
        float(best["head_strength"]),
    )
    selected_details = candidate_details[key]
    selected_summary = summarize_details(selected_details)
    deploy_head_beta = fit_head(all_cached, key[0], key[1])

    package_path = out_dir / "unified_fidelity_model.npz"
    np.savez(
        package_path,
        broad_beta=broad_beta,
        broad_feature_names=broad_package["feature_names"],
        broad_model=np.asarray(model_name),
        broad_strength=np.asarray(key[0], dtype=np.float32),
        global_beta=deploy_head_beta.astype(np.float32),
        global_feature_names=np.asarray(global_feature_names()),
        global_strength=np.asarray(key[2], dtype=np.float32),
        global_alpha=np.asarray(key[1], dtype=np.float32),
        same_model_for_all_images=np.asarray(True),
    )

    write_csv(out_dir / "candidate_search.csv", candidate_rows)
    write_csv(out_dir / "selected_per_image_metrics.csv", selected_details)
    write_csv(out_dir / "selected_summary.csv", selected_summary)
    report = {
        "model_path": str(package_path),
        "same_model_for_all_images": True,
        "broad_model": model_name,
        "conditional_routing": False,
        "training_pairs": len(all_cached),
        "extreme_solid_pairs": len(extreme_cached),
        "selected": best,
        "selected_summary": selected_summary,
        "evaluation_notes": {
            "all_categories": "grouped 5-fold evaluation for the global residual head; solid colors are grouped by RGB hex across repeats and resolutions",
            "broad_base": "the broad local model is the existing car-plus-random-solid all-fit model",
            "metric": "deterministic 12000-pixel-per-image MAE similarity",
        },
        "components": [
            "one fixed multiscale local/edge/periodic linear correction",
            "one fixed global RGB residual head using output statistics",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
