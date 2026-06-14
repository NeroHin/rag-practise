# rag-practise

Query Transformation experiments for compact RAG evaluation.

This repository compares several query transformation methods across multiple LLM
providers, then evaluates retrieval and answer-support quality with deterministic
metrics and LLM-as-a-Judge.

## Scope

Implemented methods:

- `baseline`
- `rewrite`
- `expand`
- `multi_query`
- `child_query`
- `hyde`
- `step_back`

Main experiment setup:

- Dataset: CRUD-RAG compact
- Cases: 20 QA cases
- Distractors: 100 documents
- Retrieval: FAISS flat inner product + Reciprocal Rank Fusion
- Embedding: local OMLX OpenAI-compatible endpoint
- Judge: OpenAI structured output with `gpt-5-mini-2025-08-07`

## Setup

```bash
uv sync
```

Required environment variables are listed in `.env.local`:

```bash
OPENAI_API_KEY
OPENROUTER_API_KEY
GEMINI_API_KEY
NVIDIA_API_KEY
HUGGINGFACE_API_KEY
OMLX_API_KEY
OMLX_HOST_URL
```

Use `.env` for local secrets. `.env` is ignored by git.

## Dataset

Prepared files are already under `dataset/`:

- `crud_rag_20_cases.jsonl`
- `crud_rag_100_distractors.jsonl`
- `documents_dup_part_10_part_1`
- `crud_split_merged.json`

Regenerate the compact dataset:

```bash
rag-practise datasets prepare crud-rag
```

## Run

Build retrieval index metadata:

```bash
rag-practise index build \
  --config configs/experiment.crud-rag.yaml \
  --embedding-provider omlx \
  --embedding-model Qwen3-Embedding-0.6B-4bit-DWQ
```

Run the E2E matrix with answer generation and judge:

```bash
rag-practise experiment run-matrix-async \
  --config configs/experiment.crud-rag.yaml \
  --models-config configs/models.yaml \
  --output-dir reports/model-matrix-crud-rag-compact-e2e \
  --embedding-provider omlx \
  --embedding-model Qwen3-Embedding-0.6B-4bit-DWQ \
  --max-concurrency 4 \
  --generate \
  --generation-model-id openai_fast \
  --generation-max-concurrency 4 \
  --judge \
  --judge-model-id openai_judge_gpt5_mini \
  --judge-max-concurrency 4 \
  --judge-retry-failed \
  --judge-max-attempts 3
```

Generate article charts from an existing judged report:

```bash
python scripts/plot_query_transform_results.py \
  --report-dir reports/model-matrix-crud-rag-compact-e2e-structured-judge \
  --output-dir docs/assets/query-transformations
```

`reports/` and `docs/` are ignored by git.

## Outputs

Experiment runs write:

- `records.jsonl`
- `records.csv`
- `summary.csv`
- `report.md`
- optional judge checkpoint files

Chart generation writes PNG/SVG figures and CSV comparison tables.

## Validate

```bash
ruff check .
python -m pytest -q
```

