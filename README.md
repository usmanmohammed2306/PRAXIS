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

REx is a continually-improving process-memory agent. Each run produces
trajectories; each trajectory is distilled into a procedural lesson; the next
run retrieves those lessons and uses them to choose the next tool call.

**Two memory banks, one retriever:**

* **Seed memory** (`outputs/experience_bank/{retail,airline,ace}.jsonl`) — built
  from allowed support data only (retail train/dev splits, policy-derived
  airline cards, ACE schema cards). Never includes prior eval trajectories or
  test-split answers.
* **Runtime memory** (`$REX_RUNTIME_DIR`, default `outputs/experience_runtime/`)
  — append-only, deduplicated, persisted across runs. Populated automatically
  at the end of every non-test run by `promote_trajectories`.

**The full learning loop:**

```
saved trajectories
  → process memory distillation (distill_trajectory)
  → persistent memory bank (outputs/experience_runtime/)
  → retrieval at runtime (load_experience_cards merges seed + runtime)
  → short playbook synthesis (render_experience_brief)
  → next tool call decision
  → tool execution
  → new trajectory
  → distill again
```

**At each step in a trajectory, REx:**

1. Builds a stateful retrieval query from `(initial_user, latest_user_reply,
   latest_tool_name, latest_tool_observation)` — retrieval evolves with the
   conversation, it is not a one-shot lookup at task start.
2. Refreshes the experience brief in the system prompt every
   `REX_RETRIEVAL_REFRESH_EVERY` (default 2) effective steps.
3. Executes the next tool call with native OpenAI-compatible function calling.
4. Runs same-model SABER-style reflection only before mutating tau-bench tools.

**Leakage boundary.** Cards store *process* (intent, sequence, evidence
required, confirmation point, common trap) — never IDs, emails, payment
methods, dates, or argument values. Promotion is gated: by default, test-split
runs are blocked from writing to runtime memory.

Reads are never blocked by REx. If a write lacks policy support or user
confirmation, REx asks a precise question instead of executing the write.

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
