# data-volume

Rebuttal용 data-volume ablation 작업 공간이다.

목표:
- 기존 450k mix (`300k Pro + 150k Reasoning`)를 그대로 쓰지 않고,
- 두 데이터를 합친 pool에서 random subset을 뽑아
- `50k`, `150k` 규모로 각각 학습하고 평가한다.
- 학습 budget은 full-data `1 epoch`와 유사한 `4654` optimizer steps로 맞춘다.

핵심 원칙:
- 새 파일은 이 디렉터리 아래에만 둔다.
- 기존 코드와 기존 결과물은 수정하지 않는다.
- 새 학습 산출물은 `/workspace/data2/jaehee/AlienLM/outputs/icml2026-rebuttal/data-volume` 아래에 저장한다.

현재 비교 기준:
- full-data `1 epoch` 기준점은 기존 모델의 `checkpoint-4654`를 사용한다.
- full-data `2 epoch` 최종점은 필요 시 `checkpoint-9306`를 참고한다.

## Directory Layout

- `configs/`: 50k / 150k 학습 config
- `scripts/build_data_subsets.py`: random subset 생성
- `scripts/train_subset.sh`: subset 전처리 + 학습
- `scripts/evaluate_model.sh`: main/code 평가 래퍼
- `data/`: 생성된 jsonl subset
- `data-prepared/`: axolotl prepared dataset cache
- `logs/`: 실행 로그

## Default Experiment Settings

- backbone: local cached `Meta-Llama-3-8B-Instruct` snapshot under `/workspace/CACHE/MODELS`
- alien tokenizer: `/workspace/data2/jaehee/AlienLM/outputs/Llama3-8B-Instruct-AlienLM-50-all-tokenizer-v3-32-qwenv2/checkpoint-9306`
- subset seed: `42`
- subset sampling ratio: `Magpie-Pro : Magpie-Reasoning = 2 : 1`
- training budget: `4654` optimizer steps (`max_steps`)
- in-training eval: enabled (`val_set_size: 0.001`, `eval_steps: 1000`)
- training GPUs: fixed to `0,1,2,3`
- training runner: root submission env at `/workspace/codes/AlienLMv2/.venv/bin/axolotl`
- model cache default: `/workspace/CACHE/MODELS`
- dataset cache default: `/workspace/data2/jaehee/AlienLM/HF_DATASET`
- wandb project: `magpie-alienlmv2`
- attention backend for rebuttal env: `sdp_attention` (`flash_attention: false`, `sample_packing: true`, `bf16: auto`)

## Typical Workflow

```bash
cd /workspace/codes/AlienLMv2

# 1) Build 50k / 150k subsets
python icml2026-rebuttal/data-volume/scripts/build_data_subsets.py

# 2) Train subset models
export WANDB_API_KEY=...
bash icml2026-rebuttal/data-volume/scripts/train_subset.sh 50k
bash icml2026-rebuttal/data-volume/scripts/train_subset.sh 150k

# 3) Evaluate
bash icml2026-rebuttal/data-volume/scripts/evaluate_model.sh full-1epoch --suite all
bash icml2026-rebuttal/data-volume/scripts/evaluate_model.sh 50k --suite all
bash icml2026-rebuttal/data-volume/scripts/evaluate_model.sh 150k --suite all
```

## Notes

- 이 setup은 reviewer의 `data volume / data efficiency` 질문을 답하기 위한 최소 구성이다.
- 기존 full-data intermediate checkpoint는 `training progress`에 대한 근거로는 쓸 수 있지만, `data size ablation`의 직접 증거로 쓰지 않는다.
