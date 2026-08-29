from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_broad_shallow_gemini_correction_model import Pair


TARGET_SCALE = 12.0
PERIODS = (8.0 / 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)


@dataclass
class SampleItem:
    pair: Pair
    category: str
    fold: int
    ref: np.ndarray
    output: np.ndarray
    ys: np.ndarray
    xs: np.ndarray
    ref_samples: np.ndarray
    output_samples: np.ndarray
    broad_samples: np.ndarray
    current_correction: np.ndarray
    current_score: float
    base_score: float
    global_x: np.ndarray


def local_maps(output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = output.astype(np.float32) / 255.0
    left = np.roll(rgb, 1, axis=1)
    right = np.roll(rgb, -1, axis=1)
    up = np.roll(rgb, -1, axis=0)
    down = np.roll(rgb, 1, axis=0)
    lap = (left + right + up + down) * 0.25 - rgb
    edge = np.mean(np.abs(lap), axis=2)
    texture = np.mean(np.abs(right - left) + np.abs(down - up), axis=2) * 0.5
    return edge.astype(np.float32), texture.astype(np.float32)


def nonlinear_features(
    item: SampleItem,
    maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    rgb = item.output_samples.astype(np.float32) / 255.0
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    saturation = rgb.max(axis=1) - rgb.min(axis=1)
    edge, texture = maps if maps is not None else local_maps(item.output)
    e = edge[item.ys, item.xs]
    t = texture[item.ys, item.xs]
    features = [
        rgb,
        luma[:, None],
        (luma**2)[:, None],
        saturation[:, None],
        np.exp(-luma[:, None] / 0.08),
        np.exp(-(1.0 - luma[:, None]) / 0.08),
        e[:, None],
        t[:, None],
        np.tanh(e[:, None] * 24.0),
        np.tanh(t[:, None] * 16.0),
        (rgb[:, 0] - rgb[:, 1])[:, None],
        (rgb[:, 2] - 0.5 * (rgb[:, 0] + rgb[:, 1]))[:, None],
        item.current_correction.astype(np.float32) / TARGET_SCALE,
    ]
    x = item.xs.astype(np.float32)
    y = item.ys.astype(np.float32)
    waves = []
    for period in PERIODS:
        for phase in (x, y, x + y, x - y):
            waves.append(np.sin(2.0 * np.pi * phase / period)[:, None])
            waves.append(np.cos(2.0 * np.pi * phase / period)[:, None])
    wave_matrix = np.concatenate(waves, axis=1).astype(np.float32)
    modulation = np.stack(
        [luma, luma**2, saturation, np.tanh(e * 24.0), np.tanh(t * 16.0)],
        axis=1,
    ).astype(np.float32)
    wave_modulation = (wave_matrix[:, :, None] * modulation[:, None, :]).reshape(len(x), -1)
    features.extend([wave_matrix, wave_modulation])
    features.append(
        np.broadcast_to(
            np.asarray(
                [
                    math.log(max(item.pair.width, item.pair.height) / 1024.0),
                    math.log(item.pair.width / item.pair.height),
                ],
                dtype=np.float32,
            ),
            (len(x), 2),
        )
    )
    return np.concatenate(features, axis=1).astype(np.float32)


class PixelMLP(nn.Module):
    def __init__(self, inputs: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


@torch.no_grad()
def infer_mlp_samples(
    model: PixelMLP,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), 131072):
        normalized = ((features[start : start + 131072] - mean) / scale).astype(np.float32)
        x = torch.from_numpy(normalized).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs.append(model(x).float().cpu().numpy())
    return np.concatenate(outputs, axis=0) * TARGET_SCALE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.silu(self.conv1(x))) * 0.2


class PhaseResidualCNN(nn.Module):
    def __init__(self, inputs: int = 77, width: int = 48) -> None:
        super().__init__()
        self.head = nn.Conv2d(inputs, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8, 16, 32)]
        )
        self.tail = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.silu(self.head(x))
        for block in self.blocks:
            value = block(value)
        return torch.tanh(self.tail(F.silu(value)))


def phase_channels(
    batch: int,
    height: int,
    width: int,
    x0: torch.Tensor,
    y0: torch.Tensor,
    full_width: torch.Tensor,
    full_height: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    xx = torch.arange(width, device=device, dtype=torch.float32)[None, None, None, :] + x0[:, None, None, None]
    yy = torch.arange(height, device=device, dtype=torch.float32)[None, None, :, None] + y0[:, None, None, None]
    channels = []
    for period in PERIODS:
        for phase in (xx, yy, xx + yy, xx - yy):
            channels.append(torch.sin(2.0 * math.pi * phase / period).expand(batch, 1, height, width))
            channels.append(torch.cos(2.0 * math.pi * phase / period).expand(batch, 1, height, width))
    size = torch.log(torch.maximum(full_width, full_height) / 1024.0)[:, None, None, None]
    aspect = torch.log(full_width / full_height)[:, None, None, None]
    channels.extend(
        [size.expand(batch, 1, height, width), aspect.expand(batch, 1, height, width)]
    )
    return torch.cat(channels, dim=1)


def cnn_input(
    rgb: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    full_width: torch.Tensor,
    full_height: torch.Tensor,
) -> torch.Tensor:
    luma = rgb[:, 0:1] * 0.2126 + rgb[:, 1:2] * 0.7152 + rgb[:, 2:3] * 0.0722
    saturation = rgb.amax(dim=1, keepdim=True) - rgb.amin(dim=1, keepdim=True)
    phases = phase_channels(
        rgb.shape[0], rgb.shape[2], rgb.shape[3], x0, y0, full_width, full_height, rgb.device
    )
    low31 = F.avg_pool2d(F.pad(rgb, (15, 15, 15, 15), mode="reflect"), 31, stride=1)
    low63 = F.avg_pool2d(F.pad(rgb, (31, 31, 31, 31), mode="reflect"), 63, stride=1)
    return torch.cat([rgb, luma, saturation, low31, low63, phases], dim=1)


@torch.no_grad()
def infer_cnn_full(
    model: PhaseResidualCNN,
    output: np.ndarray,
    device: torch.device,
    context_pad: int = 0,
) -> np.ndarray:
    height, width = output.shape[:2]
    padded = (
        np.pad(
            output,
            ((context_pad, context_pad), (context_pad, context_pad), (0, 0)),
            mode="reflect",
        )
        if context_pad
        else output
    )
    rgb = torch.from_numpy(
        (padded.astype(np.float32).transpose(2, 0, 1) / 255.0)[None]
    ).to(device)
    x0 = torch.tensor([-context_pad], dtype=torch.float32, device=device)
    y0 = torch.tensor([-context_pad], dtype=torch.float32, device=device)
    full_width = torch.tensor([width], dtype=torch.float32, device=device)
    full_height = torch.tensor([height], dtype=torch.float32, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model(cnn_input(rgb, x0, y0, full_width, full_height))[0]
    result = prediction.float().cpu().numpy().transpose(1, 2, 0) * TARGET_SCALE
    if context_pad:
        result = result[context_pad : context_pad + height, context_pad : context_pad + width]
    return result


def hybrid_features(
    item: SampleItem,
    linear: np.ndarray,
    mlp: np.ndarray,
    cnn: np.ndarray,
) -> np.ndarray:
    values = [item.global_x.astype(np.float64)]
    for correction in (linear, mlp, cnn, mlp - linear, cnn - linear, cnn - mlp):
        values.extend(
            [
                correction.mean(axis=0),
                correction.std(axis=0),
                np.mean(np.abs(correction), axis=0),
                np.max(np.abs(correction), axis=0),
            ]
        )
    values.append(
        np.asarray(
            [
                np.mean(np.abs(mlp - linear)),
                np.mean(np.abs(cnn - linear)),
                np.mean(np.abs(cnn - mlp)),
                np.mean(np.abs(mlp)),
                np.mean(np.abs(cnn)),
            ],
            dtype=np.float64,
        )
    )
    return np.concatenate(values)


def predict_weights(
    feature: np.ndarray,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray],
    strength: float,
) -> np.ndarray:
    beta, mean, scale = fit
    raw = np.r_[1.0, (feature - mean) / scale] @ beta
    baseline = np.asarray([1.0, 0.0, 0.0])
    weights = baseline + strength * (raw - baseline)
    weights = np.clip(weights, 0.0, 1.25)
    if weights.sum() > 1.25:
        weights *= 1.25 / weights.sum()
    return weights
