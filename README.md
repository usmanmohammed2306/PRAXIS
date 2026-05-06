# REx-RPE — Leakage-Safe Experience Retrieval for Tool Agents

This prototype compares four controllers on the same fixed base model:

| # | Controller | Description |
|---|---|---|
| 1 | `baseline` | Vanilla native tool-calling with a minimal policy prompt. |
| 2 | `act` | Act-only ablation: tool calls without reasoning prose. |
| 3 | `react` | ReAct ablation: one short `Thought:` before tool calls. |
| 4 | `rex` | REx-RPE: audited procedural experience retrieval, brief startup analysis, native tool-calling, and write-only SABER-style reflection. |

The active contribution is now **REx-RPE**, not CARGO. The old CARGO package is retained only as legacy code for rollback and historical tests; runners, smoke tests, summaries, and `run_project.sh` use `rex`.

## Research Idea

**REx: Leakage-Safe Experience Retrieval for Training-Free Multi-Turn Tool Agents.**

Small open models often fail tool benchmarks because they do not have a confident procedure for the task. REx gives the model procedural memory without benchmark-answer leakage:

- Retail experience cards come from the official τ-bench retail train/dev splits.
- Airline cards are generated from policy/tool structure because the local airline env exposes only a test split.
- ACEBench cards are generated from schema/task metadata only; `possible_answer/` is never used.
- Prior eval trajectories are only regression evidence, never prompt examples.

Each card is redacted and audited before use. Cards teach process, not arguments.

## Architecture

At task start, REx:

1. Builds or loads `outputs/experience_bank/{retail,airline,ace}.jsonl`.
2. Retrieves the top 3 similar process cards with a tiny stdlib BM25-style retriever.
3. Renders an experience brief with an explicit “never copy IDs” guard.
4. Runs one short no-tools startup analysis.
5. Executes the task with native OpenAI-compatible function calling.
6. Runs same-model SABER-style reflection only before mutating tau-bench tools.

Reads are never blocked by REx. If a write lacks policy support or user confirmation, REx asks a precise question instead of executing the write.

## Run

```bash
bash setup_env.sh
bash run_project.sh --dry-run
bash run_project.sh --profile smoke --controllers baseline,act,react,rex
```

For quick signal, use at least `50x1`:

```bash
bash run_project.sh --tau-tasks 50 --tau-trials 1 --skip-acebench --controllers baseline,act,react,rex
```

For the previous comparable tau setup:

```bash
bash run_project.sh --profile full --skip-acebench --controllers baseline,act,react,rex
```

## Shell Script Invariant

Exactly two shell scripts are allowed:

- `setup_env.sh`
- `run_project.sh`

## Outputs

Typical output layout:

```text
outputs/
  experience_bank/
    manifest.json
    retail.jsonl
    airline.jsonl
    ace.jsonl
  tau_retail_baseline/
  tau_retail_act/
  tau_retail_react/
  tau_retail_rex/
  tau_airline_baseline/
  tau_airline_act/
  tau_airline_react/
  tau_airline_rex/
  acebench_agent_rex/
  summary/
    summary.json
    summary.md
```

## Verification

Offline checks:

```bash
python3 -m unittest tests.test_rex -q
python3 -m compileall src tests scripts -q
bash run_project.sh --dry-run
python3 -m pip check
git diff --check
```

Smoke:

```bash
python3 scripts/run_smoke.py --target synthetic
python3 scripts/run_smoke.py --target tau
python3 scripts/run_smoke.py --target ace
```

Live tau/ACE smoke requires a working OpenAI-compatible model endpoint.

## Method Boundaries

- No SFT, DPO, RL, tree search, judge model, or benchmark-answer retrieval.
- Same fixed base model for all controllers in a comparison run.
- No manual in-context examples.
- No previous test/eval trajectories as in-context examples.
- Mutation reflection uses the same model and only guards writes.
