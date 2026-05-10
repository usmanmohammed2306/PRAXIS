# PRAXIS — Project Instructions

## Primary Goal

Build and validate a prototype that compares four agents on the same fixed base model:

1. **baseline** — Vanilla tool-calling with a minimal policy prompt.
2. **act** — Act-only ablation (tool calls without reasoning prose).
3. **react** — ReAct ablation (one short Thought before each tool call).
4. **praxis** — **PRAXIS**: continual procedural memory via experience distillation, HybridRetriever, TacticalPlaybook injection, and SABER mutation reflection.

The active contribution is **PRAXIS**. Baselines run first so PRAXIS can distil their experience into ProcessMemoryCards.

## Core Rules

- Optimize for practical success, not elegance.
- Prefer the fastest honest path to a runnable prototype on 1× A100.
- Minimize manual debugging, unnecessary abstraction, and moving parts.
- Keep the same fixed base model for all controllers unless explicitly changed.
- Do not fabricate benchmark results, compatibility, or repo details.
- If something is uncertain, say so clearly and choose the strongest practical fallback.

## Repository Preferences

- Prefer `python -m venv` + pip.
- Prefer one local OpenAI-compatible model server (vLLM).
- Keep upstream benchmark modifications small.
- Prefer smoke-test runs first, then optional full runs.
- Ensure outputs, logs, and summaries are clearly named.

## Hard Constraints

- Exactly two shell scripts total:
  - `setup_env.sh`
  - `run_project.sh`
- No third shell script anywhere in the repository.
- Same fixed model for all four controllers.
- Baseline must remain vanilla tool-calling.
- PRAXIS is the improved system (procedural memory retrieval).
- `run_project.sh` must run all four controllers and generate a comparison summary.

## Execution Style

- Inspect uploaded files first.
- Verify instead of guessing when possible.
- Create files directly in the repo instead of printing huge inline outputs.
- Keep answers concise unless more detail is requested.
- Prioritize consistency, complete imports, correct paths, and runnable scripts.
- Emphasize version checks, compatibility checks, and easy execution flow.
