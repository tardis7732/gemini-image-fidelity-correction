# 1024 real-photo recurrence comparison

## Protocol

- Input: one real automobile image, center-cropped without distortion to 1024x1024
- Gemini model: `gemini-3.1-flash-image`
- Request: 1K, 1:1, fixed no-change prompt
- Shared first pass: the exact same `G1` file is used by both chains
- Corrected recurrence: each corrected `Cn` becomes the input to the next Gemini pass
- Metric: MAE similarity to the original 1024 image

```text
G1 = Gemini(source)  # generated once

raw:       G1 -> G2 -> G3 -> G4 -> G5
corrected: G1 -> C1 -> G2' -> C2 -> G3' -> C3 -> G4' -> C4 -> G5' -> C5
```

| Stage | Raw chain | Corrected branch before | Corrected result | Immediate gain | Corrected vs raw | Next Gemini input |
|---|---:|---:|---:|---:|---:|---|
| Shared `G1` | `G1` 98.9252 | same `G1` 98.9252 | `C1` 99.1659 | +0.2407 | +0.2407 | `C1` |
| Pass 2 | `G2` 98.2969 | `G2'` 97.4502 | `C2` 97.6904 | +0.2402 | -0.6065 | `C2` |
| Pass 3 | `G3` 97.6380 | `G3'` 97.4638 | `C3` 97.5927 | +0.1289 | -0.0453 | `C3` |
| Pass 4 | `G4` 97.0164 | `G4'` 97.3195 | `C4` 97.3370 | +0.0175 | +0.3207 | `C4` |
| Pass 5 | `G5` 96.4261 | `G5'` 97.1415 | `C5` 97.1766 | +0.0351 | +0.7505 | final |

![Paired metrics table](paired_metrics_table.png)

![Paired recurrence contact sheet](comparison_contact_sheet.png)

The correction improved the current image at every pass. The later chain-to-chain comparison also contains Gemini sampling variance because the corrected image changes the next request input.
