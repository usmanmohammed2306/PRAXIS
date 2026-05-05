# CARGO — Calibrated Action-Risk Gating with Outcome-rollouts

Lightweight prototype that compares **four** agent controllers **on the
same fixed base model** across **τ-bench retail**, **τ-bench airline**,
and **ACEBench Agent**:

| # | Controller | What it adds over the layer above |
|---|---|---|
| 1 | **Vanilla TC** (`baseline`) | Native function-calling, minimal system prompt. |
| 2 | **Act** (`act`)             | Yao et al. 2022 ablation: action-only, no reasoning prose. |
| 3 | **ReAct** (`react`)         | Yao et al. 2022: one-line `Thought:` before each Action. |
| 4 | **CARGO** (`cargo`, *ours*) | A JSON-emitting proposer declares a risk class + pre/post-conditions for each step; deterministic gates check argument grounding, task-state validity, semantic completeness, and pre-conditions, and a calibrated self-consistency vote (+ counterfactual rollout for IRREVERSIBLE / FINAL) decides whether to commit, retry, ask, or finalize. Mutations execute only after every gate passes. |

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
            │              semantic task slots, phase locks,          │
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
            │       (4a) state-validity check vs semantic WM          │
            │       (4b) declared-pre ⊆ user_facts ∪ db_facts         │
            │       (4c) ID-typed args grounded in evidence           │
            │       (4d) completeness / confirmation for writes       │
            │       (4e) self-consistency: k=3 samples at T=0.7,      │
            │            agreement ≥ τ_c                              │
            │       (4f) counterfactual rollout (IRREV/FINAL only)    │
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

## CARGO-v2: Generic Semantic State-Constraint Risk Gating

CARGO remains a lightweight, training-free controller.  The v2 hardening
does not add fine-tuning, a separate judge, benchmark answer retrieval, or a
tree-search planner.  It extends the original gate stack so the verifier asks
two questions before acting:

1. Is the action grounded and well-formed?
2. Is the action correct, complete, policy-compliant, and appropriate for the
   current semantic task state?

CARGO-v2 is split into a generic core plus pluggable adapters:

- `src/cargo/core.py` defines the domain-neutral kernel: typed task state,
  facts, constraints, preferences, fallback rules, candidate objects,
  obligations, failed signatures, executed writes, semantic validation hooks,
  completeness hooks, and terminal state.
- `src/cargo/adapters/` contains benchmark/domain adapters.  The core does
  not name retail products, flights, reservations, or ACEBench answer
  patterns.  Adapters own tool schema enrichment, ID fields, non-ID semantic
  fields, user-message binding, observation absorption, policy hooks,
  semantic validators, and completion criteria.
- Current adapters: `tau_retail`, `tau_airline`, `acebench`, and
  `synthetic_generic`.

The deterministic working memory now separates:

- opaque typed IDs such as `user_id`, `order_id`, `item_id`, `product_id`,
  `payment_method_id`, `reservation_id`, and `flight_number`
- semantic task slots such as date, route, cabin, trip type, baggage count,
  insurance choice, payment preferences, intent, and product-option
  constraints
- durable DB-confirmed caches such as orders, products, profiles, and
  reservations

The gate stack uses that state to block completed-phase re-entry, repeated
dead-end actions without new evidence, semantic mismatches between action
arguments and state, partial writes, missing booking slots, and replacement
candidate choices that violate hard constraints.  Preferences only rank
candidates after every hard filter has passed; fallbacks apply only when the
strict constraint set is exhausted and the user allowed the fallback.

The recovery ledger for uploaded failures is tracked in
[`docs/known_issue_ledger.md`](docs/known_issue_ledger.md), with a
machine-readable companion at [`docs/known_issues.json`](docs/known_issues.json).
It maps each observed failure class to the invariant and regression test that
now protects it.

## The five risk classes

| Class | Examples | Treatment |
|---|---|---|
| `READ` | `get_*`, `list_*`, `find_*`, `search_*`, `lookup_*`, `view_*` | Fast path with repeat-loop, state-validity, and ID grounding. |
| `WRITE` | `update_*`, `modify_*`, `add_*`, `edit_*`, `set_*`, `place_*`, `book_*` | State-validity + confirmation + completeness + pre-cond + arg-grounding + SC. |
| `IRREVERSIBLE` | `cancel_*`, `delete_*`, `refund_*`, `charge_*`, `send_*`, `transfer_*` | State-validity + confirmation + completeness + pre-cond + arg-grounding + SC + CF rollout. |
| `FINAL` | `respond` (when committing the user's task) | Final-completeness + pre-cond + SC + CF rollout. |
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

The candidate chain varies by `--model` tier. The first that serves
successfully is used for **all four** controllers — guaranteeing the same
model across every condition.

| Tier | Priority | Hugging Face ID | Notes |
|---|---|---|---|
| `14b` (default, 1×80 GB GPU) | 1 | `Qwen/Qwen2.5-14B-Instruct` | Primary. Better JSON compliance + SC quality than 7B; fits 80 GB at max_model_len=32768. |
| `14b` | 2 | `Qwen/Qwen2.5-7B-Instruct` | Fallback if 14B download or serving fails. |
| `14b` | 3 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | FP8 last-resort. |
| `7b` (40 GB GPU or speed preference) | 1 | `Qwen/Qwen2.5-7B-Instruct` | 32 K context at max_model_len=32768; stable tool-calling, fastest. |
| `7b` | 2 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | FP8 fallback. |
| `7b` | 3 | `Qwen/Qwen3-4B-Instruct-2507` | Non-FP8 last-resort. |
| `32b` (≥2 GPUs, TP=2) | 1 | `Qwen/Qwen2.5-32B-Instruct` | Best accuracy; TP=2 on 80 GB GPUs; max_model_len=32768. |
| `32b` | 2 | `Qwen/Qwen2.5-7B-Instruct` | Fallback to 7B tier. |
| `32b` | 3 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | FP8 last-resort. |

## Two shell scripts (and only two)

- `setup_env.sh`   — creates `.venv`, installs requirements, clones
  `tau-bench` and `ACEBench` into `external/`, runs version checks.
- `run_project.sh` — launches vLLM with the model fallback chain, runs
  the **twelve** evaluations (4 controllers × 3 benchmarks), shuts vLLM
  down, and writes the summary.

No other `.sh` files are added.  Benchmark setup and smoke automation live
as Python helpers:

- `python3 scripts/benchmark_setup.py --bench all --install`
- `python3 scripts/run_smoke.py --target all`
- `python3 scripts/parse_smoke_results.py`

## Quickstart

```bash
bash setup_env.sh
bash run_project.sh                       # auto-detect GPUs, pick sensible defaults
```

By default, `run_project.sh` detects available GPUs and picks a model
tier and a workload profile to match. Override anything you like via
flags (also accepted as environment variables — run `bash run_project.sh
--help` for the full reference):

```bash
# Just sanity-check the resolved configuration without running anything:
bash run_project.sh --dry-run

# 1 GPU, smaller model, tiny sweep (~10–15 min):
bash run_project.sh --gpus 0 --model 7b --profile smoke

# 2 GPUs, 32B with TP=2, full 15h-budget sweep:
bash run_project.sh --gpus 0,1 --model 32b --profile full

# Run only τ-bench retail with just CARGO + ReAct, skip ACEBench:
bash run_project.sh --tau-only retail --controllers react,cargo --skip-acebench

# Fine overrides (take precedence over the profile):
bash run_project.sh --gpus 0,1 --tau-tasks 20 --tau-trials 2 --max-concurrency 16
```

### Benchmark Setup And Smoke Tests

Classic tau-bench and ACEBench are cloned into `external/`:

```bash
python3 scripts/benchmark_setup.py --bench all --install
```

By default the ACEBench helper installs the data/evaluation dependencies and
skips ACEBench's pinned `vllm==0.6.1.post1`; `setup_env.sh` owns the model
serving stack and filters upstream requirements so the CARGO vLLM build is not
clobbered.  To reproduce upstream ACEBench exactly in an isolated environment,
run:

```bash
python3 scripts/benchmark_setup.py --bench ace --install --include-ace-vllm
```

Run local smoke checks:

```bash
python3 scripts/run_smoke.py --target synthetic
python3 scripts/run_smoke.py --target all
python3 scripts/parse_smoke_results.py
```

If no `OPENAI_API_KEY` or `OPENAI_BASE_URL` is present, live tau/ACE smoke
runs are marked `blocked` with rerun commands instead of being faked.  The
synthetic smoke remains fully offline and exercises the generic core and
adapter invariants.

### Auto-selected defaults

| Detected GPUs | `--model auto` picks | TP | `max_model_len` | `gpu_mem_util` |
|---:|---|---:|---:|---:|
| 1 (80 GB A100/H100) | `14b` (Qwen2.5-14B-Instruct) | 1 | 32768 | 0.85 |
| 1 (40 GB GPU) | use `--model 7b` explicitly | 1 | 32768 | 0.80 |
| ≥ 2 | `32b` (Qwen2.5-32B-Instruct) → 7B fallback | count | 32768 | 0.85 |

> **Why 14B over 7B on a single 80 GB GPU?** The CARGO proposer must emit
> structured JSON reliably — `declared_class`, `declared_pre`, and `args` all
> have to parse correctly. Qwen2.5-14B has measurably better JSON compliance
> and instruction-following than 7B, which means fewer `json_parse_failures`,
> more meaningful self-consistency votes, and cleaner argument-grounding hits.
> At BF16, 14B weights ≈ 28 GB + KV cache at `max_model_len=32768` with
> concurrency=4 ≈ 24 GB → ~52 GB total, comfortably inside 80 GB at 0.85
> mem util. The ~1.8–2× slower throughput vs 7B still fits the `medium`
> profile within ~5–9 h on one H100.

> **Why 32K context?** τ-bench trajectories with 30 tasks × up to 30 steps
> each accumulate tool-call JSON + DB observations quickly. At 12K tokens,
> late-trajectory turns get truncated and the agent loses earlier
> DB-confirmed facts — critical for CARGO's precondition and grounding gates.
> 32K fits all three tiers within their respective GPU memory budgets:
> 7B (~32 GB on 40 GB), 14B (~52 GB on 80 GB), 32B (~68 GB/GPU on 2×80 GB).
> Override with `--max-model-len` if you hit OOM on unusual hardware.

### Workload profiles

| `--profile` | tasks/env | trials | ACE limit | concurrency | rough wall-clock |
|---|---:|---:|---:|---:|---|
| `smoke` | 5 | 1 | 5 | 2 | ~10–15 min |
| `small` | 15 | 3 | 15 | 4 | ~1–2 h |
| `medium` (default) | 30 | 3 | 20 | 4 | ~3–6 h |
| `full` | 50 | 4 | 40 | 8 | ~8–14 h |

`--max-concurrency` is doubled automatically when ≥ 2 GPUs are present
(vLLM batches across replicas, so higher concurrency fills the GPUs
without extra Python overhead). Per-knob CLI flags (`--tau-tasks`,
`--tau-trials`, `--ace-limit`, `--max-concurrency`, `--max-model-len`,
`--gpu-mem-util`) take precedence over the profile.

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
working-memory absorption (user text + observation); typed task-state conflict
handling; generic adapter schema enrichment; retail hard-constraint vs
preference separation; ACEBench-style local-pass/global-fail decoy rejection;
precondition matching; argument-grounding regex coverage; repeat-loop
detection; self-consistency vote (mock client with `n>1`); counterfactual
rollout (mock client); post-condition error detection; proposer JSON parsing;
repair policy decisions; and full agent loop behavior on mock environments.

## What CARGO is — and isn't

CARGO is **lightweight on purpose**:

- ~1 LLM call per step on the fast path (READ).
- 1 + 1 (with `n=3`) = 2 LLM calls on a gated WRITE step.
- 1 + 1 + 1 = 3 LLM calls on a gated IRREVERSIBLE / FINAL step (proposer
  + SC + CF). On a 12-turn τ-retail trajectory with ~2 IRREV/FINAL turns,
  that's ~16 calls vs ReAct's ~12.

It uses **no** tree search, **no** multi-agent debate, **no**
fine-tuning, **no** external LLM judge, and **no** memory / fingerprint
retrieval.

The novelty story is the *composition*: risk-class typing of tools
(auto-induced) + class-specific calibrated abstention + selective
self-consistency only on risky actions + deterministic argument grounding +
generic semantic state/constraint gates + named deterministic repair. None of the parts is
unprecedented; the integration as a coherent training-free architecture
for parameterized tool-using LLM agents, with **per-class** calibrated
abstention rather than syntactic guardrails, is what differentiates it
from NeMo-Guardrails (syntactic, uncalibrated), AdaPlanner (LLM-binary
in-plan/out-of-plan judge, no class typing), KnowNo (single-skill
robotics, no parameterized tool calls), AgentPRM/AgentRM (single-head
PRM, not abstention), and ToolGate (contract-grounded but uncalibrated).
