from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from apply_unified_gemini_fidelity_model import broad_prediction
from build_unified_gemini_fidelity_model import global_features
from gamut_mapping import apply_gamut_mapping
from reference_free_fidelity_features import (
    deterministic_sample_coords,
    reference_free_features,
)
from advanced_components import (
    PhaseResidualCNN,
    PixelMLP,
    SampleItem,
    hybrid_features,
    infer_cnn_full,
    infer_mlp_samples,
    local_maps,
    nonlinear_features,
    predict_weights,
)
from train_broad_shallow_gemini_correction_model import Pair


def linear_correction(
    output: np.ndarray,
    package: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    height, width = output.shape[:2]
    broad = broad_prediction(
        output,
        package["broad_feature_names"],
        package["broad_beta"].astype(np.float32),
    )
    ys, xs = deterministic_sample_coords(width, height)
    output_samples = output[ys, xs]
    base_x = global_features(output_samples, width, height)
    global_dc = base_x @ package["global_beta"].astype(np.float64)
    correction = (
        broad * float(package["broad_strength"])
        + global_dc[None, None, :] * float(package["global_strength"])
    )
    gate_alpha = 1.0
    if "gate_beta" in package.files:
        gate_x = reference_free_features(
            base_x,
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
        raw_alpha = float(np.r_[1.0, normalized] @ package["gate_beta"].astype(np.float64))
        gate_alpha = float(
            np.clip(
                1.0 + float(package["gate_strength"]) * (raw_alpha - 1.0),
                0.0,
                float(package["gate_max_alpha"]),
            )
        )
        correction *= gate_alpha
    return correction.astype(np.float32), broad[ys, xs].astype(np.float32), base_x, gate_alpha


def make_item(
    output: np.ndarray,
    pair: Pair,
    ys: np.ndarray,
    xs: np.ndarray,
    broad_samples: np.ndarray,
    current_samples: np.ndarray,
    base_x: np.ndarray,
) -> SampleItem:
    samples = output[ys, xs].astype(np.float32)
    return SampleItem(
        pair=pair,
        category="inference",
        fold=0,
        ref=output.astype(np.uint8),
        output=output.astype(np.uint8),
        ys=ys,
        xs=xs,
        ref_samples=samples,
        output_samples=samples,
        broad_samples=broad_samples,
        current_correction=current_samples,
        current_score=0.0,
        base_score=0.0,
        global_x=base_x,
    )


def infer_mlp_full(
    model: PixelMLP,
    template: SampleItem,
    current_full: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    device: torch.device,
    chunk: int = 65536,
) -> np.ndarray:
    height, width = template.output.shape[:2]
    edge_texture = local_maps(template.output)
    result = np.empty((height * width, 3), dtype=np.float32)
    for start in range(0, height * width, chunk):
        stop = min(height * width, start + chunk)
        indices = np.arange(start, stop, dtype=np.int64)
        ys = indices // width
        xs = indices % width
        item = replace(
            template,
            ys=ys,
            xs=xs,
            output_samples=template.output[ys, xs].astype(np.float32),
            ref_samples=template.output[ys, xs].astype(np.float32),
            broad_samples=np.zeros((len(indices), 3), dtype=np.float32),
            current_correction=current_full[ys, xs].astype(np.float32),
        )
        features = nonlinear_features(item, edge_texture)
        result[start:stop] = infer_mlp_samples(
            model, features, feature_mean, feature_scale, device
        )
        if stop == height * width or stop % (chunk * 8) == 0:
            print(f"mlp inferred {stop}/{height * width}", flush=True)
    return result.reshape(height, width, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Image.open(args.input.resolve()) as image:
        output = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = output.shape[:2]
    pair = Pair(
        index=0,
        dataset="inference",
        kind="car",
        family="inference",
        shape="wide" if width > height else "1x1",
        size_label="custom",
        group_id=0,
        width=width,
        height=height,
        base_similarity=0.0,
        reference_path=args.input.resolve(),
        output_path=args.input.resolve(),
        filename=args.input.name,
    )

    linear_package = np.load(model_dir / "current_linear_model.npz", allow_pickle=False)
    linear_full, broad_samples, base_x, linear_alpha = linear_correction(output, linear_package)
    ys, xs = deterministic_sample_coords(width, height)
    template = make_item(
        output,
        pair,
        ys,
        xs,
        broad_samples,
        linear_full[ys, xs],
        base_x,
    )

    mlp_package = torch.load(
        model_dir / "nonlinear_color_texture_frequency_mlp.pt",
        map_location=device,
        weights_only=False,
    )
    mlp = PixelMLP(int(mlp_package["inputs"])).to(device)
    mlp.load_state_dict(mlp_package["state_dict"])
    mlp.eval()
    mlp_full = infer_mlp_full(
        mlp,
        template,
        linear_full,
        np.asarray(mlp_package["feature_mean"], dtype=np.float32),
        np.asarray(mlp_package["feature_scale"], dtype=np.float32),
        device,
    )

    cnn_package = torch.load(
        model_dir / "phase_residual_cnn.pt",
        map_location=device,
        weights_only=False,
    )
    cnn = PhaseResidualCNN(inputs=int(cnn_package["inputs"])).to(device)
    cnn.load_state_dict(cnn_package["state_dict"])
    cnn.eval()
    cnn_full = infer_cnn_full(cnn, output, device, context_pad=128)

    gate = np.load(model_dir / "hybrid_linear_mlp_cnn_gate.npz", allow_pickle=False)
    feature = hybrid_features(
        template,
        linear_full[ys, xs],
        mlp_full[ys, xs],
        cnn_full[ys, xs],
    )
    fit = (
        gate["gate_beta"].astype(np.float64),
        gate["gate_feature_mean"].astype(np.float64),
        gate["gate_feature_scale"].astype(np.float64),
    )
    weights = predict_weights(feature, fit, float(gate["gate_strength"]))
    extra_correction = mlp_full * weights[1] + cnn_full * weights[2]
    correction = linear_full * weights[0] + extra_correction
    mapped, mapping_scale = apply_gamut_mapping(
        output,
        correction,
        "channel_soft_knee_0.75",
    )
    corrected = np.clip(np.rint(mapped), 0, 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrected, "RGB").save(args.output.resolve(), "PNG", optimize=True)
    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "size": [width, height],
                "device": str(device),
                "linear_reference_free_alpha": linear_alpha,
                "hybrid_weights": {
                    "linear": float(weights[0]),
                    "nonlinear_mlp": float(weights[1]),
                    "phase_cnn": float(weights[2]),
                },
                "mean_abs_correction_rgb": [
                    float(value) for value in np.mean(np.abs(correction), axis=(0, 1))
                ],
                "spatial_taper": False,
                "gamut_limited_fraction": float(np.mean(mapping_scale < 1.0)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
