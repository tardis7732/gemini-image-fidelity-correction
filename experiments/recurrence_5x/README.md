# 1024 real-photo recurrence comparison

## Protocol

- Input: one real automobile image, center-cropped without distortion to 1024x1024
- Gemini model: `gemini-3.1-flash-image`
- Request: 1K, 1:1, fixed no-change prompt
- Shared first pass: yes
- Correction: `advanced_multiband_all_20260829`, no spatial taper
- Metric: MAE similarity to the original 1024 image

| Pass | Gemini only | Corrected-chain before | Corrected-chain after | Immediate gain | Chain gain vs raw |
|---:|---:|---:|---:|---:|---:|
| 1 | 98.9252 | 98.9252 | 99.1659 | +0.2407 | +0.2407 |
| 2 | 98.2969 | 97.4502 | 97.6904 | +0.2402 | -0.6065 |
| 3 | 97.6380 | 97.4638 | 97.5927 | +0.1289 | -0.0453 |
| 4 | 97.0164 | 97.3195 | 97.3370 | +0.0175 | +0.3207 |
| 5 | 96.4261 | 97.1415 | 97.1766 | +0.0351 | +0.7505 |

The correction improved the current image at every pass. The later chain-to-chain comparison also contains Gemini sampling variance because the corrected image changes the next request input.
