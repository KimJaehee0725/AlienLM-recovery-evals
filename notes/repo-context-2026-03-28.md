# Repo Context

Date: 2026-03-28

## Working Constraints

- 기존 코드 경로는 건드리지 않는다.
- rebuttal 관련 새 산출물은 `icml2026-rebuttal/` 아래에서만 만든다.
- 학습된 모델, 체크포인트, 일부 평가 결과의 실제 source of truth는 `/workspace/data2/jaehee/AlienLM/outputs`다.

## Repository Shape

- 루트 `pyproject.toml`은 얇은 workspace wrapper다.
- 실제 구현 축은 대략 아래 네 묶음으로 보인다.
  - `alien_tokenizer/`: alien tokenizer 자산과 token init 실험
  - `icml2026-submition/`: tokenizer, translator, training, eval만 남긴 제출용 최소 셋
  - `icml-2026/`: rot13, SentinelLM, tenant-alienlm, OpenAI finetuning 실험
  - `iclr_review/`: alignment, ByteT5, attack scenario 등 별도 리뷰용 실험
- `axolotl/`, `lm-evaluation-harness/`, `lm-evaluation-harness-original/`는 외부 레포 사본을 같이 들고 있는 구조다.

## Output Path Notes

확인된 외부 출력 루트:

`/workspace/data2/jaehee/AlienLM/outputs`

눈에 띄는 하위 구조:
- `Llama3-8B-Instruct-AlienLM-*`
- `Qwen25-7b-Instruct-*`
- `Qwen25-14b-Instruct-*`
- `Gemma2-9b-it-*`
- `data-finetuning/`
- `iclr_2026/`
- `icml2026/`

이 경로에는 다음이 함께 섞여 있다.
- 학습 checkpoint 디렉터리
- 태스크별 평가 결과 디렉터리
- 일부 tokenizer/translation 관련 파생 산출물

즉, rebuttal 작업 시에는 레포 내부 `evaluation_result/`만 보는 것이 아니라 외부 출력 경로와 같이 봐야 한다.

## Practical Rule

앞으로 rebuttal용 스크립트는 가능하면 다음 원칙을 따른다.

- 입력 기본값은 `/workspace/data2/jaehee/AlienLM/outputs`
- 출력 기본값은 `icml2026-rebuttal/` 내부
- 기존 경로의 파일은 읽기 전용으로 취급
