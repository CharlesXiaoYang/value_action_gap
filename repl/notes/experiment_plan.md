# Experiment Extension Plan

## Recommended next experiment
Compare 2-3 models on the same `country/topic` slice and report results at both:
- individual value level
- Schwartz family level

## Minimal high-value setup
- Country: `United States`
- Topic: `Politics`
- Models:
  - `groq:qwen/qwen3-32b`
  - `groq:llama-3.1-8b-instant`
  - `groq:llama-3.3-70b-versatile` if token budget permits

## Why this is strong
- Same pipeline for all models
- Same value/action pairs
- Fair same-model Task 1 vs Task 2 comparison
- Family aggregation reduces noise and makes the paper extension easier to interpret
- Prompt sensitivity shows whether conclusions are robust or prompt-fragile

## Scripts added
- `repl/scripts/aggregate_value_families.py`
- `repl/scripts/analyze_prompt_sensitivity.py`

## Suggested workflow per model
1. Run Task 1 for one `(country, topic)` slice.
2. Parse Task 1.
3. Run Task 2 on the same `(country, topic)` slice.
4. Summarize Task 2.
5. Compare Task 1 vs Task 2.
6. Aggregate to Schwartz families.
7. Analyze prompt sensitivity.
8. Plot value-level and family-level figures.

## Questions this can answer
- Which value families are most aligned in action?
- Which families show the largest value-action gaps?
- Are the conclusions stable across prompt variants?
- Do different models show the same gap profile or different ones?
