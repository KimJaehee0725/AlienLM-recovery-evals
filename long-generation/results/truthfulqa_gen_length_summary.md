# TruthfulQA Generation Length Summary

All token lengths are measured with the tokenizer actually used by the evaluated model variant.


| Backbone            | Variant  | BLEU Acc | ROUGE-L Acc | Mean Input Tok | Var Input Tok | Mean Output Tok | Var Output Tok |
| ------------------- | -------- | -------- | ----------- | -------------- | ------------- | --------------- | -------------- |
| Llama3-8B-Instruct  | original | 0.4627   | 0.4749      | 157.81         | 49.82         | 21.54           | 290.81         |
| Llama3-8B-Instruct  | alienlm  | 0.3929   | 0.4039      | 157.81         | 49.82         | 22.33           | 358.18         |
| Qwen2.5-7B-Instruct | original | 0.5055   | 0.5018      | 166.89         | 51.49         | 77.82           | 2235.34        |
| Qwen2.5-7B-Instruct | alienlm  | 0.3819   | 0.3929      | 166.89         | 51.49         | 33.72           | 2662.09        |
| Gemma2-9B-it        | original | 0.4896   | 0.5153      | 177.89         | 52.34         | 22.83           | 134.27         |
| Gemma2-9B-it        | alienlm  | 0.4162   | 0.4406      | 177.89         | 52.34         | 26.51           | 247.99         |


## Detailed Notes

### Llama3-8B-Instruct / original

- Sample count: 817
- BLEU Acc: 0.4627
- ROUGE-L Acc: 0.4749
- Mean input tokens: 157.81
- Input token variance: 49.82
- Mean output tokens: 21.54
- Output token variance: 290.81
- Mean input chars: 637.15
- Input char variance: 1195.12
- Mean output chars: 89.17
- Output char variance: 5096.89

### Llama3-8B-Instruct / alienlm

- Sample count: 817
- BLEU Acc: 0.3929
- ROUGE-L Acc: 0.4039
- Mean input tokens: 157.81
- Input token variance: 49.82
- Mean output tokens: 22.33
- Output token variance: 358.18
- Mean input chars: 637.15
- Input char variance: 1195.12
- Mean output chars: 94.22
- Output char variance: 8079.61

### Qwen2.5-7B-Instruct / original

- Sample count: 817
- BLEU Acc: 0.5055
- ROUGE-L Acc: 0.5018
- Mean input tokens: 166.89
- Input token variance: 51.49
- Mean output tokens: 77.82
- Output token variance: 2235.34
- Mean input chars: 637.15
- Input char variance: 1195.12
- Mean output chars: 379.97
- Output char variance: 59295.30

### Qwen2.5-7B-Instruct / alienlm

- Sample count: 817
- BLEU Acc: 0.3819
- ROUGE-L Acc: 0.3929
- Mean input tokens: 166.89
- Input token variance: 51.49
- Mean output tokens: 33.72
- Output token variance: 2662.09
- Mean input chars: 637.15
- Input char variance: 1195.12
- Mean output chars: 131.39
- Output char variance: 40621.47

### Gemma2-9B-it / original

- Sample count: 817
- BLEU Acc: 0.4896
- ROUGE-L Acc: 0.5153
- Mean input tokens: 177.89
- Input token variance: 52.34
- Mean output tokens: 22.83
- Output token variance: 134.27
- Mean input chars: 637.15
- Input char variance: 1195.12
- Mean output chars: 100.38
- Output char variance: 3498.23

### Gemma2-9B-it / alienlm

- Sample count: 817
- BLEU Acc: 0.4162
- ROUGE-L Acc: 0.4406
- Mean input tokens: 177.89
- Input token variance: 52.34
- Mean output tokens: 26.51
- Output token variance: 247.99
- Mean input chars: 637.15
- Input char variance: 1195.12
- Mean output chars: 112.21
- Output char variance: 5345.57

