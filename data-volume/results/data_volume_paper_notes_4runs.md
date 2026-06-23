# Data-Volume Rebuttal Notes (4 Runs)

We now compare four Llama AlienLM training budgets under the same main evaluation suite:

- `full-0.5epoch`: original full-data model at `checkpoint-2327`
- `full-1epoch`: original full-data model at `checkpoint-4654`
- `50k-ga2`: rebuttal 50k subset run
- `150k-ga2`: rebuttal 150k subset run

## Main Takeaways

- The aggregate ordering is `full-1epoch (51.80) > 150k-ga2 (50.64) > full-0.5epoch (48.58) > 50k-ga2 (45.43)`.
- `150k-ga2` closes most of the gap to `full-1epoch`, trailing by only `1.16` macro-average points.
- `150k-ga2` is substantially better than `50k-ga2` on every task except `TruthfulQA MC1`, where they are effectively tied with the full-data runs.
- The strongest `full-1epoch` gain over `full-0.5epoch` appears on `GSM8K CoT` (`+10.54` points), suggesting that math performance remains especially sensitive to training budget.
- `150k-ga2` slightly exceeds `full-1epoch` on `MMLU` (`39.56` vs. `39.20`) and `ARC-Easy` (`65.03` vs. `64.23`), but remains lower on `ARC-Challenge`, `HellaSwag`, `WinoGrande`, and especially `GSM8K CoT`.

## Suggested Rebuttal Framing

These results show a clear data-volume trend. Reducing the adaptation set to `50k` causes a broad degradation across all main tasks. Increasing the subset to `150k` recovers most of the lost performance and reaches within `1.16` macro-average points of the full-data `1 epoch` checkpoint. However, the remaining gap on `GSM8K CoT` indicates that mathematically structured tasks require a larger adaptation budget than general language understanding tasks.
