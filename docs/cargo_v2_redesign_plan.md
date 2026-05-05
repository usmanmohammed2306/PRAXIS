# CARGO v2 Training-Free Redesign Plan

Date: 2026-05-05

This note records the redesign direction after the latest full tau-bench runs:

- Retail `metrics (56)`: 200 trajectories, success `0.01`, `2966` abstains, `2044` retry repairs, `752` finalize repairs, `1602` actions executed.
- Airline `metrics (57)`: 200 trajectories, success `0.04`, `4107` abstains, `2614` retry repairs, `1313` finalize repairs, `1612` actions executed.
- ACEBench remains healthy in prior smoke runs, so the fix should target tau-bench continuity without making the generic core domain-specific.

The dominant failure is not basic tool availability. The traces show useful retrieval followed by unsafe or unproductive control transitions:

- Retail can assemble a plausible exchange, but it can skip the required authentication/confirmation spine and then fails the benchmark policy path.
- Airline can retrieve the profile and correct search results, but then repeats searches, asks from hallucinated itinerary details, or executes a pseudo-write such as `calculate(total_cost + taxes_and_fees)`.
- Repeated READ failures were being turned into retry/finalize spirals, inflating abstains even though non-mutating deviations are much less harmful than wrong writes.

## Architecture

CARGO v2 keeps the training-free research identity:

- same base model across controllers
- risk-typed tool actions
- deterministic gates
- no fine-tuning
- no answer retrieval
- no judge model
- no tree search
- no multi-agent debate

The spine is now:

1. Perceive user/tool evidence into working memory.
2. Maintain compact task state and candidate sets.
3. Use a phase-aware deterministic controller before gates:
   `AUTHENTICATE -> DISCOVER -> CONFIRM -> COMMIT -> WRAP`.
4. Keep the soft goal-field router as a candidate scorer, not as a planner.
5. Run a cheap deterministic pre-commit verifier only on `WRITE`/`IRREVERSIBLE`.
6. Keep commit certificates as final safety evidence for now, not as the planner.

## Implemented In This Patch

- Added a generic `PhaseEngine`, `PhaseDecision`, `PreCommitVerifier`, and `PreCommitVerdict` in `src/cargo/core.py`.
- Increased the repeat-signature window to `8`.
- Added exact pending-commit confirmation state in `WorkingMemory`.
- Added a pre-commit verifier gate to reject:
  - unsupported pseudo-write tools such as `calculate`
  - placeholder arguments such as `latest_search_result`
  - missing required write arguments
  - wrong-typed opaque IDs
- Added retail auth-phase routing so account/order mutation goals authenticate before order reads, while preserving order-id recovery after failed identity lookup.
- Added nested airline itinerary candidate-set support so one-stop search results are represented as grounded itinerary candidates.
- Added deterministic airline booking progression:
  - filters direct flights by time/cabin availability
  - falls back to one-stop candidates when direct options violate hard constraints
  - chooses the cheapest viable itinerary
  - builds passengers from profile evidence
  - computes certificate/card payment split from profile payment methods
  - computes free/non-free baggage from membership and cabin
  - asks for exact user confirmation before `book_reservation`

## Test Coverage Added

- repeat window is exactly eight signatures
- pre-commit verifier blocks pseudo writes and placeholders
- retail account task authenticates before direct order lookup
- retail order-id recovery remains live after failed identity
- nested airline one-stop candidates are recorded with individual flight IDs
- airline presents grounded itinerary details before booking
- airline builds a complete `book_reservation` after confirmation
- pre-commit verifier appears in the write gate diagnostics

Existing tests for read-permissive behavior, write completeness, active task-frame isolation, commit certificates, soft goal-field routing, and synthetic smoke remain part of the suite.

## Suggested Hyperparameters

Do not change `setup_env.sh` or `run_project.sh` until live smoke confirms the new spine. Suggested values for manual runs:

| Parameter | Qwen2.5-14B | Qwen2.5-32B |
|---|---:|---:|
| proposer temperature, single sample | `0.2` | `0.2` |
| proposer temperature, write sampling | `0.6` | `0.5` |
| max proposer tokens | `256` | `384` |
| concise reasoning budget | `64` tokens | `96` tokens |
| self-consistency `k` for READ/ASK | `1` | `1` |
| self-consistency `k` for WRITE | `3` | `2` |
| self-consistency `k` for IRREVERSIBLE | `3` | `3` |
| loop window | `8` | `8` |
| repeat forced-advance threshold | `2` | `2` |
| ask retries per missing slot | `2` | `2` |
| max steps | `30` | `30` |

The highest-value next hyperparameter change is to disable expensive verification on `FINAL` and reserve extra samples for actual writes, but that should be changed only after a controlled smoke comparison.

## Remaining Bottlenecks

- Retail still needs a cleaner exact confirmation barrier for every mutation, not just policy-shaped intent text.
- Airline modify/cancel/update flows need deterministic write builders similar to the new booking builder.
- The proposer prompt still carries older proof/verifier wording; it should be shortened after the new phase spine proves out in live smoke.
- Live tau/ACE smoke still requires `OPENAI_API_KEY` or `OPENAI_BASE_URL`.

## Verification Commands

Run:

```bash
python3 -m unittest tests.test_cargo -q
python3 -m compileall src tests scripts -q
bash run_project.sh --dry-run
python3 scripts/run_smoke.py --target all --json-out outputs/smoke/smoke_summary_v2_spine.json
```

Live tau/ACE reruns require a model endpoint. The shell scripts are intentionally unchanged in this patch.
