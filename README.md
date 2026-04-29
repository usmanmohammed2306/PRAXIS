# CARGO — Calibrated Action-Risk Gating with Outcome-rollouts

Lightweight prototype that compares **four** agent controllers **on the
same fixed base model** across **τ-bench retail**, **τ-bench airline**,
and **ACEBench Agent**:

| # | Controller | What it adds over the layer above |
|---|---|---|
| 1 | **Vanilla TC** (`baseline`) | Native function-calling, minimal system prompt. |
| 2 | **Act** (`act`)             | Yao et al. 2022 ablation: action-only, no reasoning prose. |
| 3 | **ReAct** (`react`)         | Yao et al. 2022: one-line `Thought:` before each Action. |
| 4 | **CARGO** (`cargo`, *ours*) | A JSON-emitting proposer declares a risk class + pre/post-conditions for each step; deterministic gates check argument grounding and pre-conditions, and a calibrated self-consistency vote (+ counterfactual rollout for IRREVERSIBLE / FINAL) decides whether to commit, retry, ask, or finalize. Mutations execute only after every gate passes. |

All four conditions share the **same in-process loop, model,
temperature, max-steps, and truncation budget**. The only varying axis is
the controller — for CARGO, the risk-typed gate stack. The `respond`
contract that tau-bench / ACEBench use as the user-facing channel is
unchanged.

## The architecture

```
                    ┌────────────────────────────────────────────┐
                    │             TASK ENVIRONMENT (τ-bench)     │
                    │    user simulator  +  tool sandbox  +  DB  │
                    └──────────────────▲─────────────────────┬───┘
                                       │ obs                 │ action
                                       │                     ▼
            ┌──────────────────────────┴──────────────────────────────┐
            │                  CARGO AGENT LOOP                       │
            │                                                         │
            │  (0)  Tool-effect schema cache (built once at init)     │
            │       per tool t → {class, pre, post, irreversible}     │
            │       Rule-based by name prefix; LLM augmentation       │
            │       optional via CARGO_INDUCE_VIA_LLM=1.              │
            │                                                         │
            │  (1)  Typed Working Memory  (deterministic)             │
            │       slots: goal, user_revealed_facts,                 │
            │              db_confirmed_facts, assumptions,           │
            │              pending_obligations, last_obs,             │
            │              last_error, budget_steps                   │
            │                                                         │
            │  (2)  Proposer  (1 LLM call, JSON output)               │
            │       Output: {thought, action:{name, args,             │
            │                declared_class, declared_pre,            │
            │                declared_post, informational_intent,     │
            │                user_text}}                              │
            │                                                         │
            │  (3)  Risk Router  (deterministic, O(1))                │
            │       READ  → fast path → execute                       │
            │       WRITE / IRREV / FINAL → calibrated gate           │
            │                                                         │
            │  (4)  Calibrated Gate                                   │
            │       (4-) repeat-loop check (cheap; runs always)       │
            │       (4a) declared-pre ⊆ user_facts ∪ db_facts         │
            │       (4b) ID-typed args grounded in evidence (regex)   │
            │       (4c) self-consistency: k=3 samples at T=0.7,      │
            │            agreement ≥ τ_c                              │
            │       (4d) counterfactual rollout (IRREV/FINAL only)    │
            │                                                         │
            │  (5)  Repair on ABSTAIN  (deterministic)                │
            │       grounding/precond  → ASK_USER                     │
            │       low SC / CF block  → RETRY w/ critique (≤2)       │
            │       repeat-loop / out-of-budget → FINALIZE_GENERIC    │
            │                                                         │
            │  (6)  Execute → obs                                     │
            │  (7)  Post-condition check (advisory)                   │
            │  (8)  WM update (db_facts ← scalar keys in obs)         │
            │  (9)  Loop until FINAL passes the gate.                 │
            └─────────────────────────────────────────────────────────┘
```

## The five risk classes

| Class | Examples | Treatment |
|---|---|---|
| `READ` | `get_*`, `list_*`, `find_*`, `search_*`, `lookup_*`, `view_*` | Fast path. Repeat-loop check only. |
| `WRITE` | `update_*`, `modify_*`, `add_*`, `edit_*`, `set_*`, `place_*`, `book_*` | Pre-cond + arg-grounding + SC. |
| `IRREVERSIBLE` | `cancel_*`, `delete_*`, `refund_*`, `charge_*`, `send_*`, `transfer_*` | Pre-cond + arg-grounding + SC + CF rollout. |
| `FINAL` | `respond` (when committing the user's task) | Pre-cond + SC + CF rollout. |
| `ASK_USER` | `respond` (when asking a clarifying question) | Pass-through. |

Auto-induction is rule-based (name prefix + parameter shape) by default —
identical across τ-bench retail, τ-bench airline, and ACEBench Agent.
Setting `CARGO_INDUCE_VIA_LLM=1` runs an additional one-shot LLM
classification per ambiguous tool at startup.

## Calibrated thresholds

Defaults (overridable via env vars):

| Class | k samples | SC threshold τ_c | Counterfactual? |
|---|---:|---:|:---:|
| WRITE | 3 | 0.66 | no |
| IRREVERSIBLE | 3 | 0.66 | yes |
| FINAL | 3 | 0.66 | yes |

Override per class with `CARGO_SC_TAU_WRITE`, `CARGO_SC_TAU_IRREV`,
`CARGO_SC_TAU_FINAL`, `CARGO_SC_K_*`, and `CARGO_CF_IRREV` /
`CARGO_CF_FINAL`. Calibration on logged baseline rollouts is the v2 step
(`src/cargo/calibration.py:fit_thresholds`).

## Diagnostics (per controller, per cell)

`info.cargo_stats` records:

- `steps_total`, `steps_fast_path`, `steps_gated`
- `gate_runs` and `gate_fails` per gate name
- `abstain_total`, `repair_retry`, `repair_ask_user`, `repair_finalize`
- `json_parse_failures`
- `actions_executed`, `executed_by_class`
- `per_step` (capped at 200) — per-step proposer thought, declared class,
  gates run / failed, SC agreement, CF blocking flag

The summary builder aggregates these across trajectories.

## Fixed base model (priority order)

| Priority | Hugging Face ID | Notes |
|---|---|---|
| 1 | `Qwen/Qwen2.5-7B-Instruct` | Primary. 32 K context, stable tool-calling. |
| 2 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | FP8 fallback. |
| 3 | `Qwen/Qwen3-4B-Instruct-2507` | Non-FP8 last-resort fallback. |

`run_project.sh` tries the candidates in order and runs **the first one
that serves successfully** for *all four* controllers — guaranteeing the
same model across every condition.

## Two shell scripts (and only two)

- `setup_env.sh`   — creates `.venv`, installs requirements, clones
  `tau-bench` and `ACEBench` into `external/`, runs version checks.
- `run_project.sh` — launches vLLM with the model fallback chain, runs
  the **twelve** evaluations (4 controllers × 3 benchmarks), shuts vLLM
  down, and writes the summary.

## Quickstart

```bash
bash setup_env.sh
bash run_project.sh
```

Outputs:

```
outputs/
  active_model.txt
  vllm.log
  tau_retail_baseline/   tau_retail_act/   tau_retail_react/   tau_retail_cargo/
  tau_airline_baseline/  tau_airline_act/  tau_airline_react/  tau_airline_cargo/
  acebench_agent_baseline/ acebench_agent_act/ acebench_agent_react/ acebench_agent_cargo/
  summary/
    summary.json   # 4-way comparison + CARGO-vs-best-baseline + CARGO-vs-baseline deltas
    summary.md     # rendered table
```

## Tests

The architecture is covered by offline unit tests + an integration smoke
test (no live model needed):

```bash
python3 -m unittest tests.test_cargo -v
```

The suite checks: rule-based risk classification; tool schema caching;
working-memory absorption (user text + observation); precondition
matching (positive and negative); argument-grounding regex coverage;
repeat-loop detection; self-consistency vote (mock client with `n>1`);
counterfactual rollout (mock client); post-condition error detection;
proposer JSON parsing (clean / fenced / malformed / nested); repair
policy decisions; full agent loop on a mock env that returns scripted
observations.

## What CARGO is — and isn't

CARGO is **lightweight on purpose**:

- ~1 LLM call per step on the fast path (READ).
- 1 + 1 (with `n=3`) = 2 LLM calls on a gated WRITE step.
- 1 + 1 + 1 = 3 LLM calls on a gated IRREVERSIBLE / FINAL step (proposer
  + SC + CF). On a 12-turn τ-retail trajectory with ~2 IRREV/FINAL turns,
  that's ~16 calls vs ReAct's ~12.

It uses **no** tree search, **no** multi-agent debate, **no**
fine-tuning, **no** external LLM judge, **no** memory / fingerprint
retrieval (slot reserved for v2).

The novelty story is the *composition*: risk-class typing of tools
(auto-induced) + class-specific calibrated abstention + selective
self-consistency only on irreversible/final actions + deterministic
argument-grounding + named deterministic repair. None of the parts is
unprecedented; the integration as a coherent training-free architecture
for parameterized tool-using LLM agents, with **per-class** calibrated
abstention rather than syntactic guardrails, is what differentiates it
from NeMo-Guardrails (syntactic, uncalibrated), AdaPlanner (LLM-binary
in-plan/out-of-plan judge, no class typing), KnowNo (single-skill
robotics, no parameterized tool calls), AgentPRM/AgentRM (single-head
PRM, not abstention), and ToolGate (contract-grounded but uncalibrated).
