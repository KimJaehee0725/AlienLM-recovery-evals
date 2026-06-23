# Working Agenda

Date: 2026-03-28

이 메모는 현재 rebuttal 작업에서 바로 써야 하는 우선순위와 문서 수정 포인트를 분리해서 적어둔 것이다.

## Current Ground Truth

- rebuttal 관련 새 파일은 `icml2026-rebuttal/` 아래에만 둔다.
- 실험 결과의 실제 source of truth는 레포 내부 `evaluation_result/`만이 아니라 `/workspace/data2/jaehee/AlienLM/outputs`도 함께 봐야 한다.
- `icml 답변`이라고 적힌 문서는 draft로 취급하고, final wording보다 실험 축과 논리 구조를 우선 재사용한다.

## Immediate Paper Edit TODO

### Section 3.4 inference example ref missing

현재 PDF 기준으로 Appendix G 도입 문단은 qualitative examples가 **Section 3.4**와 **Section 4.5**를 보완한다고 명시한다.

즉, Section 3.4 마지막 inference 설명에서 qualitative example을 언급한다면 최소한 아래 ref 중 하나는 붙는 것이 맞다.

- `Appendix G`
- `Figure 6` for GSM8K reasoning example
- `Figure 7` for MBPP code example

가장 안전한 수정 방향:

- inference-time human opacity와 model-side consistency를 설명하는 문장 끝에 `See Appendix G and Figures 6–7 for qualitative GSM8K/MBPP examples.` 같은 sign-post를 추가
- 만약 정량 robustness와 함께 연결하고 싶으면 `Appendix G`와 별도로 `Section 4.5`도 함께 언급

가능한 문장 초안:

`Qualitative inference examples illustrating both human opacity and model-side consistency are provided in Appendix G (Figures 6 and 7).`

또는

`See Appendix G (Figures 6 and 7) for qualitative GSM8K and MBPP examples of inference-time alienized inputs/outputs.`

## Priority Items From Current Reviews

### P0: directly answerable with existing or near-existing evidence

- stronger attack suite
  - known-plaintext
  - parallel-corpus alignment
  - LM-guided decipherment
- serving overhead
  - translator latency
  - token count / API cost impact
  - end-to-end latency/throughput
- code/math degradation analysis
  - bucketed pairing and GSM8K behavior
  - tokenizer/bijection differences on hard samples
- Halawi et al. positioning
  - overlap in setup
  - substantive methodological differences
- data-efficiency ablation
  - `50k / 150k / 300k` or equivalent compact sweep

### P1: wording and framing fixes

- replace strong crypto framing with narrower exposure-reduction framing
- explicitly state adversary model / leakage model / non-goals
- move scope limitation from appendix into intro + early method
- separate privacy claim from safety claim
- explicitly acknowledge behavioral profiling and metadata leakage limitations

## Reviewer-linked Notes

### Reviewer ZP5r

- strongest direct concern is code/math degradation
- best support currently comes from:
  - Section 4.3 / Appendix D.4 style domain-adaptation evidence
  - qualitative code/math examples in Appendix G
- useful additional angle:
  - hard-example analysis showing different bijections correlate with large sample-level variance

### Reviewer SoMh

- token inflation / API cost must be answered quantitatively
- if tokenizer-overlap experiment already exists, tie it to actual dataset-level token counts
- user behavior inference answer should be framed as:
  - semantic signals reduced
  - structural/profile signals remain because mapping is deterministic

### Reviewer 3piF

- Halawi comparison must be explicit, not implicit
- longer generation concern can be partially covered with MBPP/HumanEval if full new generation benchmark is too expensive
- safety concern should be answered as:
  - limitation acknowledged
  - AlienLM is not a replacement for safety layers
  - encrypted-domain / system-level safety remains open work

### Reviewer g_Q9F

- the key demand is not rhetoric but stronger scope definition plus stress tests
- best response structure:
  - narrow claim
  - explicit threat model
  - empirical stress tests
  - deployment overhead
  - safety limitation

## Reusable Evidence Already Mentioned In Prior Notes

- Appendix G already provides the qualitative examples that Section 3.4 should point to.
- prior ICLR response files already contain reusable numbers for:
  - stronger attacks
  - serving overhead
  - white-box SentinelLM upper bound
  - ByT5 sanity check
  - safety regression

## Practical Next Steps

1. Fix paper sign-posting first.
2. Rebuild or verify the easiest quantitative additions from existing outputs.
3. Only then refine rebuttal prose.
