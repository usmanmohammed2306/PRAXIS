# Project instructions

Primary goal:
Build and validate a prototype that compares four controllers on the same fixed base model:
1. baseline — vanilla tool-calling
2. act — Act-only ablation
3. react — ReAct ablation
4. praxis — PRAXIS (Procedural Retrieval-Augmented eXperience-Informed System)

The active contribution is PRAXIS: continual procedural memory via experience distillation,
HybridRetriever, TacticalPlaybook injection, and SABER mutation reflection.

Core rules:
- Optimize for practical success, not elegance.
- Prefer the fastest honest path to a runnable prototype on 1x A100.
- Minimize manual debugging, unnecessary abstraction, and moving parts.
- Keep the same fixed base model for all four controllers unless explicitly changed.
- Do not fabricate benchmark results, compatibility, or repo details.
- If something is uncertain, say so clearly and choose the strongest practical fallback.

How to use uploaded materials:
- Treat the uploaded project PDF as the source of truth for goals and constraints.
- Treat the uploaded working shell file only as a compatibility and environment reference.
- Do not copy the shell file blindly.
- Extract useful assumptions, commands, paths, and version expectations, then implement a cleaner version.

Repository preferences:
- Prefer python -m venv + pip.
- Prefer one local OpenAI-compatible model server if feasible.
- Keep upstream benchmark modifications small.
- Prefer smoke-test runs first, then optional full runs.
- Ensure outputs, logs, and summaries are clearly named.

Hard constraints:
- Exactly two shell scripts total:
  - setup_env.sh
  - run_project.sh
- No third shell script anywhere in the repository.
- Same fixed model for all four controllers.
- Baseline must remain vanilla tool-calling.
- Improved system must be PRAXIS (procedural experience retrieval).
- run_project.sh must run all four controllers and generate a comparison summary.

Execution style:
- Inspect uploaded files first.
- Verify instead of guessing when possible.
- Create files directly in the repo instead of printing huge inline outputs.
- Keep answers concise unless more detail is requested.
- Prioritize consistency, complete imports, correct paths, and runnable scripts.
- Emphasize version checks, compatibility checks, and easy execution flow.