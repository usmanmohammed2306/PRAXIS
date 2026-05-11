# PRAXIS — Procedural Retrieval-Augmented eXperience-Informed System

**Continual Procedural Memory for Tool-Calling LLM Agents**

> *An agent that learns from every interaction — not just its own.*

PRAXIS wraps any frozen LLM with a growing, leakage-safe procedural memory —
distilled from all agents' experience — so each run starts smarter than the last.

## Overview

This prototype compares four controllers on the same fixed base model:

| # | Controller | Description |
|---|---|---|
| 1 | `baseline` | Vanilla native tool-calling with a minimal policy prompt. |
| 2 | `act` | Act-only ablation: tool calls without reasoning prose. |
| 3 | `react` | ReAct ablation: one short `Thought:` before tool calls. |
| 4 | `praxis` | **PRAXIS**: audited procedural experience retrieval, TacticalPlaybook injection, and write-only SABER-style mutation reflection. |

The active contribution is **PRAXIS**. All four controllers share the same model, temperature, tool schemas, max steps, and truncation budget — the only varying axis is the controller. Baselines run first; PRAXIS distils their experience into ProcessMemoryCards and retrieves the top-5 most relevant lessons at each step.

## Research Idea

**PRAXIS: Continual Procedural Memory for Tool-Calling LLM Agents.**

Small open models often fail tool benchmarks because they lack a confident procedure for the task. PRAXIS gives any frozen model procedural memory without benchmark-answer leakage:

- Retail experience cards come from the official τ-bench retail train/dev splits.
- Airline cards are generated from policy/tool structure (local airline env exposes only a test split).
- BFCL cards are generated from schema/task metadata only; `possible_answer/` is never used.
- Prior eval trajectories are regression evidence only — never prompt examples.

Each card is PII-stripped and quality-scored before use. Cards teach **process**, not arguments.

## The Memory Loop

```
  Act ─┐
ReAct ─┼─▶  trajectories.jsonl
  TC  ─┘
             │
             ▼
        DISTILLER         strips PII · extracts procedure
             │             tool order · failure patterns
             ▼             recovery hints · anti-patterns
        MEMORY BANK        ProcessMemoryCards
        retail.jsonl       grows across every run
        airline.jsonl      deduplicated · quality-scored
        bfcl.jsonl         max 4,096 cards / domain
             │
             │  top-5 cards (refreshed every 2 steps)
             ▼
        PRAXIS AGENT       system prompt + TacticalPlaybook
                           SABER reflection on writes
             │
             └──── completed tasks ───▶ back into the loop ↑
```

## Architecture

**Two memory banks, one hybrid retriever:**

- **Seed memory** (`outputs/experience_bank/{retail,airline,bfcl}.jsonl`) — built from allowed support data only. Never includes prior eval trajectories or test-split answers.
- **Runtime memory** (`outputs/experience_runtime/`) — append-only, deduplicated, persisted across runs. Populated automatically at the end of every non-test run.

**HybridRetriever** at each step:

- BM25 × 0.60 + TF-IDF × 0.40
- Domain boost +0.05, quality boost +0.05 × score
- Diversity cap: 3 cards per category
- Returns top-5 cards → rendered as ≤ 2,400-char TacticalPlaybook

**SABER mutation reflection** guards every write tool:

- READ tool → execute immediately, no delay.
- WRITE tool → pause, ask same model: *"Do observations confirm this action is safe & requested?"* → ALLOW or BLOCK.

**Leakage boundary.** Cards store *process* (intent, sequence, evidence required, confirmation point, common trap) — never IDs, emails, payment methods, dates, or argument values. Promotion is gated: test-split runs are blocked from writing to runtime memory by default.

## Train / Test Split (Evaluation Fairness)

All four controllers are evaluated on identical held-out test sets. PRAXIS's
memory bank is only ever populated from **train** trajectories, never from the
tasks used to score it. Three layers enforce this:

1. **Run-level gate** — `tau_runner._should_promote()` returns False for
   pure-test runs unless explicitly overridden.
2. **Record-level gate** — `pipeline._is_test_split_record()` drops any record
   tagged `info.task_split == "test"` before it can be distilled. Both
   runners stamp every record with its split label.
3. **Pre-warm separation** — PRAXIS pre-warm runs (`--praxis-prewarm`) write
   only on train indices; the subsequent four-way evaluation runs only on
   test indices.

| Benchmark | Train source | Test source |
|---|---|---|
| τ-bench retail  | Upstream `train` split (500 tasks) | Upstream `test` split (115 tasks) |
| τ-bench airline | Internal split: indices `[0, AIRLINE_TRAIN_END)` of the upstream `test` split (default `AIRLINE_TRAIN_END=20`) | Indices `[AIRLINE_TRAIN_END, end)` of the upstream `test` split |
| BFCL V4         | Per-category first `BFCL_TRAIN_FRACTION` of each category (default `0.25`, deterministic sort by task id) | Remaining `1 − BFCL_TRAIN_FRACTION` per category |

Override via env vars: `AIRLINE_TRAIN_END=N`, `BFCL_TRAIN_FRACTION=F`. Setting
either to `0` disables that split (not recommended for paper-grade runs).

## Key Numbers

| Property | Value |
|---|---|
| Model fine-tuning steps | **0** |
| Agents compared | **4** |
| Cards retrieved per query | **5** |
| Steps between memory refreshes | **2** |
| Max chars in TacticalPlaybook | **2,400** |
| Max memory cards per domain | **4,096** |
| Benchmarks evaluated | **3** |
| Distillation overhead per run | **< 30 s** |

## Run

```bash
bash setup_env.sh
bash run_project.sh --dry-run
bash run_project.sh --profile smoke --controllers baseline,act,react,praxis
```

For quick signal, use at least `50x1`:

```bash
bash run_project.sh --tau-tasks 50 --tau-trials 1 --skip-bfcl --controllers baseline,act,react,praxis
```

For a full evaluation pass:

```bash
bash run_project.sh --profile full --skip-bfcl --controllers baseline,act,react,praxis
```

## Shell Script Invariant

Exactly two shell scripts are allowed:

- `setup_env.sh`
- `run_project.sh`

## Outputs

```text
outputs/
  experience_bank/
    manifest.json
    retail.jsonl
    airline.jsonl
    bfcl.jsonl
  tau_retail_baseline/
  tau_retail_act/
  tau_retail_react/
  tau_retail_praxis/
  tau_airline_baseline/
  tau_airline_act/
  tau_airline_react/
  tau_airline_praxis/
  bfcl_praxis/
  summary/
    summary.json
    summary.md
```

## Verification

```bash
python3 -m unittest tests.test_rex -q
python3 -m compileall src tests scripts -q
bash run_project.sh --dry-run
python3 -m pip check
git diff --check
```

## What Is New

| Standard approach | PRAXIS |
|---|---|
| Standard RAG → retrieves documents to answer questions | PRAXIS → retrieves **procedures** to execute multi-step tasks |
| In-context learning → static examples in every prompt | PRAXIS → **growing compact lessons** across all runs |
| Fine-tuning → expensive, slow, risks forgetting | PRAXIS → **zero training**, works on any frozen model |

## Method Boundaries

- No SFT, DPO, RL, tree search, judge model, or benchmark-answer retrieval.
- Same fixed base model for all controllers in a comparison run.
- No manual in-context examples.
- No previous test/eval trajectories as in-context examples.
- Mutation reflection uses the same model and only guards writes.
