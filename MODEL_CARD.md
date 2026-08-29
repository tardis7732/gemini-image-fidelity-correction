# Model card

## Intended use

이 가중치는 Gemini no-change 이미지 편집에서 관찰된 작은 색상/질감 드리프트를 생성 이미지 한 장만으로 완화하는 연구용 후보입니다. 입력 이미지의 내용이나 구도를 재생성하지 않고 작은 RGB correction field를 예측합니다.

## Training and evaluation snapshot

- Generator: Gemini 3.1 Flash Image (`gemini-3.1-flash-image`)
- Covered formats: `1:1` and `16:9` at both `1K` and `2K`
- Training/evaluation pairs: 342
- Validation: source identity로 묶은 grouped 5-fold
- Sampling: 이미지당 deterministic 12,000 pixels
- External holdout: 학습에 사용하지 않은 자동차 실사 10장
- Metric: MAE similarity percentage

## Components

- `current_linear_model.npz`: color-adaptive linear/DC correction
- `nonlinear_color_texture_frequency_mlp.pt`: nonlinear color/texture/frequency correction
- `phase_residual_cnn.pt`: spatial phase residual network
- `hybrid_linear_mlp_cnn_gate.npz`: reference-free blend gate

## Risks

- OOF 342장 중 17장은 보정 후 점수가 하락했습니다.
- 외부 실사 10장 중 1장은 `-0.0600%p` 하락했습니다.
- 원본 이미지가 없는 실제 추론에서는 개선 여부를 직접 확인할 수 없습니다.
- 기하 변화나 강한 재생성은 이 모델의 보정 범위를 벗어납니다.
