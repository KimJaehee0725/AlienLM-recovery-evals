# Data-Volume Rebuttal Notes

We compared three training budgets for the Llama AlienLM setting using the same main evaluation suite:

- `full-0.5epoch`: the original full-data model at `checkpoint-2327`
- `50k-ga2`: the rebuttal 50k subset run
- `150k-ga2`: the rebuttal 150k subset run

## Main Takeaways

- `150k-ga2` is consistently stronger than `50k-ga2` and also slightly outperforms the original `full-0.5epoch` checkpoint on aggregate.
- The largest gains from `150k-ga2` over `full-0.5epoch` appear on `GSM8K CoT` (`+5.23` points), `MMLU` (`+3.59`), and `ARC-Challenge` (`+2.99`).
- `50k-ga2` underperforms `full-0.5epoch` on every reported task, with the largest drop on `HellaSwag` (`-6.73`) and a noticeable drop on `GSM8K CoT` (`-4.55`).
- The macro average is `45.43` for `50k-ga2`, `48.58` for `full-0.5epoch`, and `50.64` for `150k-ga2`.

## Suggested Rebuttal Framing

These results support a clear monotonic data-volume trend: reducing the adaptation data to `50k` causes a broad degradation, while increasing the subset to `150k` recovers most of the gap and slightly surpasses the full-data `0.5 epoch` reference under the same evaluation suite. This suggests that a moderate amount of adaptation data is sufficient to recover much of AlienLM's downstream performance, whereas `50k` samples are not yet enough for stable task transfer.
