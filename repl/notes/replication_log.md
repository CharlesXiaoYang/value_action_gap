# Replication Log (Value Action Gap)

## Goal
Recreate the end-to-end pipeline (Task1 -> Task2 -> parsing -> aggregation -> metrics -> figures)
without OpenAI API, using a free/local model.

## What stays unchanged
- Original repository files under src/tasks, outputs/data_release, etc.

## What I add
- repl/ folder containing full replication pipeline implementation.

## Key decisions
- Model backend: `aisuite` client so the same runner can target OpenAI, Groq, or other supported providers.
- Subset configuration: runners support small test slices before any full run.
- Prompt variants: reuse the original 8 Task 1 and 8 Task 2 prompt variants.
- Parsing rules: repair JSON-like outputs, normalize Task 1 Likert ratings, and map Task 2 choices back to canonical positive vs negative polarity.
- Metric definitions: summarize positive-choice rate, negative-choice rate, parse-failure rate, and later compare Task 1 value scores against Task 2 action choices for the same model.
