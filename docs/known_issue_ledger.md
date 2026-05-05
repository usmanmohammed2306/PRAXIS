# CARGO Known-Issue Ledger

This ledger is the recovery record for the uploaded CARGO runs from
`metrics (24).json` through `metrics (33).json`, their paired trajectories,
and the local regression suite.  It is intentionally phrased by failure
class rather than benchmark task answer, so it remains a test and design
artifact rather than an answer key.

Machine-readable companion: `docs/known_issues.json`.

## Metric Timeline

| Run | Domain | Success | Avg reward | Avg trajectory | Abstains | Retries | Ask repairs | Finalize repairs | Executed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metrics (24) | retail | 0.0 | 0.0 | 35.0 | 71 | 40 | 17 | 14 | 21 |
| metrics (25) | retail | 0.0 | 0.0 | 37.8 | 76 | 45 | 13 | 18 | 24 |
| metrics (26) | retail | 0.0 | 0.0 | 38.0 | 75 | 45 | 7 | 23 | 25 |
| metrics (27) | retail | 0.0 | 0.0 | 40.6 | 73 | 34 | 18 | 21 | 27 |
| metrics (28) | retail | 0.0 | 0.0 | 26.8 | 36 | 20 | 9 | 7 | 24 |
| metrics (29) | retail | 0.0 | 0.0 | 16.2 | 6 | 4 | 1 | 1 | 21 |
| metrics (30) | airline | 0.0 | 0.0 | 39.8 | 94 | 17 | 69 | 8 | 6 |
| metrics (31) | retail | 0.0 | 0.0 | 16.6 | 1 | 1 | 0 | 0 | 24 |
| metrics (32) | airline | 0.0 | 0.0 | 32.8 | 94 | 52 | 14 | 28 | 6 |
| metrics (33) | retail | 0.0 | 0.0 | 22.6 | 31 | 19 | 2 | 10 | 20 |

## Benchmark Context Used

- tau-bench evaluates multi-turn tool-agent-user interaction in retail and
  airline domains, with policy adherence and final database state compared
  against an annotated goal state.  This makes partial task completion and
  post-write continuation correctness-critical.
- tau-style pass^k/reliability framing penalizes inconsistent behavior across
  repeated trials, so deterministic repeat suppression and state carryover are
  part of the CARGO thesis rather than convenience patches.
- ACE/AgentCE-style evaluations contain local constraints, global constraints,
  and decoy candidates that can look valid locally but break the full task.
  This motivates semantic constraint validation as a gate, not a scorer.

## CARGO-v2 Architecture Mapping

- Generic core: `src/cargo/core.py` owns typed task state, fact precedence,
  constraints, preferences, fallback rules, candidate objects, obligations,
  semantic validation hooks, completeness hooks, and diagnostics.
- Adapters: `src/cargo/adapters/` owns benchmark/domain knowledge.  Current
  adapters are `tau_retail`, `tau_airline`, `acebench`, and
  `synthetic_generic`.
- Shell-script invariant: no new `.sh` files were added.  Benchmark setup and
  smoke commands are Python helpers in `scripts/`.

## Ledger

| Failure class | Evidence | Root cause | Invariant | Regression coverage | Status |
|---|---|---|---|---|---|
| Placeholder `user_id` used | Airline runs repeatedly proposed `user_id="user_id"` before a valid user id was reused | User-provided IDs were loose text, not typed state | Opaque IDs must bind immediately and placeholders resolve only from one grounded value | `test_user_supplied_user_id_is_bound_to_typed_state`, `test_grounded_placeholder_resolver_uses_user_provided_id`, `test_resolved_placeholder_passes_arg_grounding` | Fixed, verified |
| Ambiguous ID guessing | Adjacent risk after placeholder resolution | Resolver could choose the wrong ID if multiple were present | Ambiguity must block, not guess | `test_grounded_placeholder_resolver_does_not_guess_ambiguous_ids` | Fixed, verified |
| Valid airline dates blocked | Airline run blocked `date=YYYY-MM-DD` as ungrounded ID | ID regex treated semantic date literals as opaque IDs | Date/time fields are semantic literals unless explicitly ID-typed | `test_iso_date_argument_is_not_treated_as_opaque_id` | Fixed, verified |
| Origin/destination/date state loss | Airline searches depended on prompt text only | Non-ID slots were not represented in persistent state | User facts bind into semantic task slots | `test_absorb_user_message_binds_airline_semantic_slots`, `test_i2_state_gate_blocks_search_conflicting_with_bound_date` | Fixed, verified |
| Ask-user loop despite answer | Airline trajectories showed repeated generic asks after users answered | Completed/known slots were not surfaced as state | Known slots should suppress re-asking and guide next action | Semantic slots in `render_compact`; `test_absorb_user_message_binds_airline_semantic_slots` | Fixed, verified |
| Repeated profile lookup | Airline trajectories repeated `get_user_details` after profile was cached | No durable user-profile cache or post-profile transition | Completed profile phase must advance to reservation retrieval | `test_profile_reservation_list_is_bound_to_typed_state`, `test_airline_profile_repetition_advances_to_reservation_scan` | Fixed, verified |
| Reservation IDs not typed | Airline profile observations returned reservation lists that were not reused | `reservations[]` was not mapped to `reservation_id` | Tool-returned reservation IDs must become typed evidence | `test_profile_reservation_list_is_bound_to_typed_state` | Fixed, verified |
| Completed auth re-entered | Retail/airline loops returned to auth tools after auth | Auth was treated as prompt state, not hard phase state | Locked phases cannot be re-entered | `test_i1_completed_auth_phase_blocks_auth_tool_reentry`, existing auth-lock tests | Fixed, verified |
| Repeated flight search with same args | Airline direct/onestop searches repeated with no new evidence | Repeat suppression needed to catch identical READs | Same signature cannot repeat without new evidence | `test_repeat_signature_detected`, `test_failed_signature_blocks_until_new_evidence` | Fixed, verified |
| Payment preferences not represented | Airline booking tasks mention certificates/cards | Payment text was not bound outside raw prose | Payment preferences are semantic slots, not opaque IDs | `test_absorb_user_message_binds_airline_semantic_slots`, `test_i3_booking_write_requires_complete_slots` | Fixed, verified |
| Baggage and insurance not represented | Airline tasks mention bags/insurance | Non-ID preferences were not persisted | Baggage/insurance bind as semantic slots | `test_absorb_user_message_binds_airline_semantic_slots` | Fixed, verified |
| Booking before complete slots | Adjacent airline failure after search succeeds | Generic write completeness only covered retail | Booking writes require user, flights, passenger, payment, cabin slots | `test_i3_booking_write_requires_complete_slots`, `test_i4_booking_write_passes_slot_completeness_when_filled` | Fixed, verified |
| Retail wrong keyboard variant selected | Retail runs selected variants violating requested options | Candidate scoring treated constraints as preferences | Hard constraints filter before ranking | `test_g1_exchange_constraints_are_hard_filters`, `test_g2_exchange_rejects_decoy_when_constraint_missing` | Fixed, verified |
| Clicky/full-size ignored | Retail keyboard failures selected wrong switch/size | Selector lacked strict semantic validation | Explicit option constraints are hard filters | `test_g1_exchange_constraints_are_hard_filters`, `test_g2_exchange_rejects_decoy_when_constraint_missing` | Fixed, verified |
| Backlight fallback too early | Retail fallback chose a locally attractive but invalid candidate | Fallback could rank before strict exhaustion | Fallback applies only after strict candidates fail | `test_e3_exchange_respects_only_exchange_fallback`, `test_g1_exchange_constraints_are_hard_filters` | Fixed, verified |
| Thermostat compatibility trap | Retail thermostat must map Google Home/Assistant equivalently | Compatibility strings were not treated as hard constraints | Compatibility must be verified against variant options | `test_g1_exchange_constraints_are_hard_filters`, product-match regressions | Fixed, verified |
| Partial multi-item exchange | Retail writes executed with only part of requested operation | Completeness was syntactic, not semantic | All requested items must resolve before write | `test_h1_partial_write_is_blocked_by_completeness_gate`, `test_h2_complete_canonical_write_passes_completeness_gate` | Fixed, verified |
| Missing target options guessed | Retail exchange with no desired replacement options picked arbitrary alternative | “Any different item” was treated as enough | Replacement writes require explicit target constraints | `test_h6_exchange_without_target_options_asks_instead_of_guessing` | Fixed, verified |
| User-provided wrong IDs outrank DB | Adjacent retail risk | User IDs/items could override grounded DB state | DB-confirmed facts outrank user claims | `test_db_confirmed_semantic_slot_outranks_later_user_claim`, typed ID grounding tests | Fixed, verified |
| Duplicate writes after success | Retail policy says exchange/modify once | Successful mutation signature/status was not durable enough | Successful writes lock mutation and block duplicates | `test_f5_completed_mutation_gate_blocks_reexecution`, `test_g4_distinct_multi_order_mutation_can_continue` | Fixed, verified |
| Successful write followed by exploration | Retail/airline post-write continuation risk | Termination did not always reflect operation completion | After successful state change, update state and stop if no fresh mutation remains | `test_f5_completed_mutation_gate_blocks_reexecution`, post-write order-cache tests | Fixed, verified |
| Premature final after partial answer | Retail product-count runs stopped after count while user returned a new task | `FINAL` marked task complete even when env returned follow-up with `done=False` | FINAL only terminates when environment is done or no follow-up user reply exists | `test_h5_final_with_followup_user_reply_does_not_terminate_solve` | Fixed, verified |
| Generic grounding over-blocks valid reads | Cross-domain valid reads with semantic strings were blocked | Gate inferred risk from all ID-shaped strings | Only ID-like fields require strict grounding; semantic literals are allowed | `test_iso_date_argument_is_not_treated_as_opaque_id`, non-ID grounding tests | Fixed, verified |
| State stored but unused | Cross-domain repeated loops despite cached facts | State was prompt-only rather than gate-visible | State validity gate must inspect current state before action | `test_i1_completed_auth_phase_blocks_auth_tool_reentry`, `test_i2_state_gate_blocks_search_conflicting_with_bound_date` | Fixed, verified |
| Constraints extracted but not enforced | ACE-style decoys and retail variants passed local syntax | No semantic candidate validator | Constraints are filters, preferences rank only after filtering | G-series and H-series tests | Fixed, verified |
| Partial solution treated as complete | Retail/airline final/write before all operations | Completion checked tool syntax instead of task closure | Completeness validator blocks partial writes/finals | H-series tests, booking completeness tests | Fixed, verified |
| ACE-style decoy passes local checks | Candidate satisfies local slot but violates global task constraints | No adapter-owned global constraint hook | Local-pass/global-fail decoys must be rejected before write/final | `test_acebench_adapter_rejects_local_pass_global_fail_decoy` | Fixed, verified |
| Domain logic leaking into core | Early CARGO-v2 state binding knew airline/retail names directly | Core was doing adapter work | Domain object semantics live only in adapters | `TestCargoV2Adapters` plus module split | Fixed, verified |

## Remaining Limitations

- Classic tau-bench and ACEBench were cloned into `external/`; tau-bench was
  installed locally.  ACEBench non-vLLM data/eval dependencies were installed,
  while upstream `vllm==0.6.1.post1` remains intentionally skipped in the
  generic helper unless `--include-ace-vllm` is used in an isolated
  environment.  Live smoke still needs a model endpoint/API key.
- Airline full booking selection is guarded for slot completeness and state
  consistency, but a complete deterministic cheapest-flight planner is still an
  open extension.  CARGO remains a lightweight gating controller, not a full
  tree-search planner.

## Verification

- `python3 -m unittest tests.test_cargo -v`: local regression suite.
- `python3 -m compileall src tests`: passed.
- `git diff --check`: passed.
- `bash run_project.sh --dry-run`: passed configuration resolution without
  launching a model server or benchmark run.
- `python3 scripts/run_smoke.py --target all`: synthetic checks run offline;
  tau/ACE live smoke is blocked unless `OPENAI_API_KEY` or `OPENAI_BASE_URL`
  is present.
