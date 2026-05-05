# CARGO Known-Issue Ledger

This ledger is the recovery record for the uploaded CARGO runs from
`metrics (24).json` through `metrics (49).json`, their paired trajectories,
`metrics (24).json` through `metrics (46).json`, their paired trajectories,
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
| metrics (34) | airline | 0.0 | 0.0 | 35.4 | 79 | 54 | 2 | 23 | 21 |
| metrics (35) | airline | 0.0 | 0.0 | 35.4 | 79 | 54 | 2 | 23 | 21 |
| metrics (36) | retail | 0.0 | 0.0 | 33.2 | 65 | 43 | 0 | 22 | 23 |
| metrics (37) | airline | 0.0 | 0.0 | 39.4 | 68 | 45 | 1 | 22 | 32 |
| metrics (38) | retail | 0.0 | 0.0 | 28.4 | 18 | 15 | 1 | 2 | 37 |
| metrics (39) | airline | 0.0 | 0.0 | 39.2 | 72 | 42 | 8 | 22 | 28 |
| metrics (40) | retail | 0.0 | 0.0 | 28.2 | 16 | 15 | 0 | 1 | 38 |
| metrics (41) | ACE agent | 1.0 completion | - | 2.0 | 0 | 0 | 0 | 0 | 5 tool calls |
| metrics (42) | airline | 0.0 | 0.0 | 49.8 | 39 | 22 | 7 | 10 | 61 |
| metrics (43) | retail | 0.0 | 0.0 | 28.2 | 17 | 14 | 1 | 2 | 37 |
| metrics (44) | ACE agent | 1.0 completion | - | 2.0 | 0 | 0 | 0 | 0 | 5 tool calls |
| metrics (45) | airline | 0.0 | 0.0 | 47.4 | 46 | 27 | 6 | 13 | 54 |
| metrics (46) | retail | 0.0 | 0.0 | 33.6 | 27 | 22 | 0 | 5 | 42 |
| metrics (47) | airline | 0.0 | 0.0 | 50.2 | 37 | 22 | 5 | 10 | 63 |
| metrics (48) | retail | 0.0 | 0.0 | 32.6 | 23 | 19 | 0 | 4 | 42 |
| metrics (49) | ACE agent | 1.0 completion | - | 2.0 | 0 | 0 | 0 | 0 | 5 tool calls |

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

## CARGO-v4 Architecture Mapping

- Generic core: `src/cargo/core.py` owns layered task state, fact precedence,
  open slots, constraints, preferences, fallback rules, candidate sets,
  obligations, semantic validation hooks, completeness hooks, diagnostics, and
  the deterministic decision engine.
- Adapters: `src/cargo/adapters/` owns benchmark/domain knowledge.  Current
  adapters are `tau_retail`, `tau_airline`, `acebench`, and
  `synthetic_generic`.
- CARGO-v4 keeps action-class-specific validation and adds deterministic
  decision before commitment. READ is retrieval-permissive and may build
  incomplete state; WRITE/IRREVERSIBLE/FINAL are commitment-strict and require
  semantic completeness.
- The decision engine stores candidate sets from READ actions, applies hard
  constraints before preferences, applies fallbacks only after strict
  exhaustion, schedules pipeline stages, suppresses exhausted repeated
  searches, blocks premature ASK_USER questions, and terminates after a true
  blocker or successful write.
- Active task-frame isolation is now explicit: user-bound goal facts are the
  current frame, while conflicting historical tool/cache facts are evidence
  only and are quarantined from semantic task slots by adapters.
- Stage-machine routing now prevents pure catalog/count goals from entering
  auth, prevents booking goals from scanning unrelated reservations, filters
  invalid reservation IDs from scan queues, and emits the terminal `respond`
  after a successful write.
- Task-frame isolation keeps user-bound goal slots separate from nested
  candidate/reservation/profile facts, so tool observations add evidence
  without silently rewriting the current route/date/cabin/product objective.
- No-auth catalog routing sends pure retail product-count/list requests to
  catalog READ actions instead of identity collection. Account/order tasks
  remain auth-strict.
- Successful WRITE/IRREVERSIBLE actions emit a post-write `respond` before
  terminal completion, unless another distinct grounded mutation is pending.
- Obligation guidance converts repeated or generic ASK/FINAL proposals into
  the next grounded READ when user/tool evidence already identifies an open
  information need.
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
| Retail READ over-blocked by semantic state | metrics (36) blocked `get_product_details(product_id=1656367028)` as `action_product_id_conflicts_with_state` | Tool-returned opaque IDs were bound into semantic slots; flat state validity treated retrieval as commitment | Opaque IDs stay typed evidence, not semantic constraints; READ ignores opaque-ID semantic conflicts and only blocks clear ordinary-slot contradictions | `test_tool_observation_ids_do_not_become_semantic_slots`, `test_read_permissive_allows_grounded_product_retrieval_with_incomplete_state` | Fixed, verified |
| Airline booking/change intent not bound | metrics (34)/(35) tasks repeatedly emitted ASK_USER after the user stated a booking/change goal | Intent/route/date/cabin were not represented as task-closure obligations visible to the controller | Bind before propose; create open airline obligations from user text; obligation guide selects search/profile READs before asking again | `test_airline_adapter_binds_booking_intent_and_open_obligation`, `test_i2b_airline_ask_loop_pivots_to_bound_flight_search` | Fixed, verified |
| Repeated direct search without progression | metrics (34)/(35) retried direct/onestop signatures after no progress | Repeat repair only retried/finalized instead of choosing the next information need | If direct search already ran and one-stop is allowed, pivot to grounded one-stop search instead of repeating or finalizing | `test_i2c_repeated_direct_search_pivots_to_onestop_when_allowed` | Fixed, verified |
| Retail wrong variant after successful retrieval | metrics (38) selected clicky/RGB/80% keyboard (`2299424241`) while user required full-size and allowed no-backlight fallback | Final choice was still model/scorer-driven instead of deterministic constraint-priority selection | Hard constraints filter first; RGB is only a preference; no-backlight fallback applies only after clicky/RGB/full-size is unavailable | `test_v4_constraint_priority_selects_full_size_clicky_fallback`, `test_v4_retail_adapter_rejects_wrong_keyboard_variant_from_latest_logs`, `test_v4_retail_two_item_exchange_uses_deterministic_selected_candidates` | Fixed, verified |
| Retail product-scoping leak into keyboard selection | metrics (40) still selected clicky/RGB/80% or treated the keyboard as impossible when the same request also mentioned a Google-compatible thermostat | Product-option constraints were extracted from the whole user request and applied to every product, so thermostat compatibility polluted keyboard selection; when adapter selection returned `None`, the older heuristic fallback revived the wrong keyboard | Adapter semantic constraints must be scoped to option keys present on the current product, and adapter selection must be authoritative for tau-retail | `test_v4_retail_adapter_rejects_wrong_keyboard_variant_from_latest_logs`, `test_v4_retail_adapter_skips_keyboard_when_exact_spec_unavailable_and_user_says_other_only`, `test_v4_retail_two_item_exchange_uses_deterministic_selected_candidates`, `test_v4_retail_exact_keyboard_unavailable_commits_thermostat_only_when_requested` | Fixed, verified |
| Airline empty-search loop | metrics (37) direct and one-stop searches returned empty, then the trajectory continued with repeated search/ASK behavior | Candidate sets did not record exhausted empty results as terminal search evidence | Empty candidate sets are stored with query args and exhausted status; same search is not repeated without new evidence | `test_v4_candidate_set_memory_records_empty_search_exhaustion`, `test_v4_airline_exhausted_searches_finalize_instead_of_looping` | Fixed, verified |
| Premature payment/certificate ask | metrics (37) asked for certificate values before a flight was selected or profile/payment state could determine options | ASK_USER policy lacked pipeline stage awareness | Payment/certificate questions are blocked until a flight/itinerary candidate is selected | `test_v4_airline_blocks_premature_payment_questions` | Fixed, verified |
| Forced ID field accepts plain words | metrics (37) included invalid airline ID attempts such as ordinary words in `reservation_id`/`user_id` fields | Adapter-declared ID fields fell through when the value did not look ID-like | Forced ID fields must be typed evidence, or ID-looking and grounded; ordinary words are rejected | `test_v4_forced_id_fields_reject_plain_words` | Fixed, verified |
| Airline generic ASK before identity | metrics (39) asked generic “how can I assist?” after a user stated a booking/change goal but had not yet provided `user_id` | The obligation guide only used `user_id` after it was known; missing identity did not create a precise current-stage question | Airline account/reservation tasks ask one precise user-id question before profile, reservation, search, or booking work | `test_v4_airline_missing_user_id_asks_precisely_before_search` | Fixed, verified |
| Airline city names used as airport-code search args | metrics (39) searched with user-facing city names such as New York/Seattle, which tau-airline tools expect as three-letter airport codes | The adapter bound semantic route text but did not canonicalize search arguments or allow city/code semantic equivalence | Domain adapters translate ordinary route text into tool-native airport codes while the core still validates against the bound semantic city state; ambiguous regions match DB airport evidence but are not guessed as a search arg | `test_v4_airline_search_uses_airport_codes_but_matches_bound_city_state`, `test_v4_airline_region_word_matches_db_airport_without_canonicalizing_search_to_guess` | Fixed, verified |
| Adapter-declared IDs can bypass grounding when schema is missing | metrics (39) showed invalid `get_reservation_details(reservation_id='though')` style calls in trajectories | Schema enrichment may be absent for a synthesized proposal, so plain-word ID arguments must have an adapter-level state backstop | Adapter ID fields are opaque identifiers even when the schema is weak; plain words are rejected before READ execution | `test_v4_adapter_id_backstop_rejects_plain_word_when_schema_is_missing` | Fixed, verified |
| Structured payment IDs missing from typed state | Regression exposed by stricter forced-ID grounding | Cached order/profile payment fields were visible in evidence text but not as typed ID evidence | Payment, card, and certificate IDs are extracted from durable structured caches | `test_h2_complete_canonical_write_passes_completeness_gate` | Fixed, verified |
| Cached reservation facts overwrite active airline goal | metrics (47) booked/search tasks drifted from New York→Seattle into cached DEN→LAS and other unrelated reservations after profile scan | Tool observations were being promoted into semantic task slots, so the last cached reservation became the active route/date/cabin frame | User-bound task-frame slots are protected; airline adapter quarantines historical route/date/cabin/insurance/payment facts from observations | `test_tool_observation_does_not_overwrite_user_bound_task_frame`, `test_v4_booking_goal_does_not_scan_unrelated_reservations` | Fixed, verified |
| Booking intent scans unrelated reservations | metrics (47) booking tasks scanned every reservation before searching requested flights | The reservation-advance stage treated every airline booking/flight word as requiring reservation retrieval | Booking goals search flight candidates; only modify/cancel/reservation goals scan grounded reservations | `test_v4_booking_goal_does_not_scan_unrelated_reservations`, `test_v4_airline_search_uses_airport_codes_but_matches_bound_city_state` | Fixed, verified |
| Invalid reservation scan IDs survive typed state | metrics (47) included invalid `reservation_id='though'` and `reservation_id=None` attempts | Scan queue trusted typed evidence too broadly after earlier recovery steps | Reservation scan accepts only tau-shaped reservation IDs and skips plain words / null sentinels | `test_v4_reservation_scan_skips_plain_words_and_none`, `test_v4_adapter_id_backstop_rejects_plain_word_when_schema_is_missing` | Fixed, verified |
| Retail no-auth product query enters auth and loops after answer | metrics (48) product-count tasks asked for credentials before catalog reads and continued after the count answer | No explicit stage boundary separated catalog-only goals from account/order goals | Active task-frame stage routes pure catalog/count goals through `list_all_product_types` and `get_product_details`, then treats the computed answer as terminal | `test_v4_task_frame_routes_no_auth_product_query_before_auth`, `test_v4_task_frame_fetches_product_details_after_catalog`, product-count finalizer tests | Fixed, verified |
| Successful retail write lacks terminal respond | metrics (48) exchange writes executed but trajectories ended without the user-facing final response tau-bench expects | Mutation success marked task complete before emitting `respond` | Successful writes now emit one terminal response and stop when no fresh mutation remains | `test_v4_successful_write_emits_terminal_respond`, `test_f5_completed_mutation_gate_blocks_reexecution` | Fixed, verified |

## Remaining Limitations

- Classic tau-bench and ACEBench are present in `external/`; tau-bench
  installed locally after network approval. ACEBench safe data/eval
  dependencies installed, while upstream `vllm==0.6.1.post1` and
  conflict-prone shared pins remain intentionally skipped unless
  `--include-ace-vllm` is used in an isolated environment. Live smoke still
| Airline reservation observation overwrote booking goal | metrics (42)/(45) booking tasks bound New York→Seattle May 20, then profile/reservation reads shifted searches to unrelated stored trips such as DEN→LAS May 27 | Tool observation fields were promoted into semantic task slots without preserving user-slot provenance | User-bound task-frame slots keep provenance; tool object fields are evidence, not automatic goal updates | `test_v4_airline_reservation_obs_does_not_overwrite_booking_anchor`, `test_tool_observation_does_not_overwrite_user_bound_date` | Fixed, verified |
| New booking scanned existing reservations before itinerary search | metrics (42)/(45) new-booking tasks retrieved reservation details after profile instead of searching the requested route/date | Reservation-scan advancement did not distinguish new booking intent from modification/cancel intent | Booking intent suppresses reservation scans and keeps the pipeline on grounded flight search | `test_v4_booking_task_does_not_scan_reservations_before_search` | Fixed, verified |
| Pure product-count task asked for identity | metrics (46) included catalog/count requests that burned turns on auth-like asks despite no account/order operation | The no-auth override caught placeholder lookup tools but not model-proposed `respond`/ASK_USER actions | Pure catalog/count goals route to `list_all_product_types` or grounded `get_product_details` before identity collection | `test_v4_no_auth_product_query_routes_to_catalog_read` | Fixed, verified |
| Successful write did not give simulator a terminal response | metrics (43)/(46) retail traces executed useful writes but did not always produce the post-write `respond` needed for tau-bench STOP/reward | The solve loop marked the task complete and broke before the post-write response path | Terminal completion is deferred until after a deterministic post-write `respond`, unless another distinct grounded mutation remains | `test_solve_emits_auto_respond_after_write`, `test_solve_post_write_responded_flag_set` | Fixed, verified |

## Remaining Limitations

- Classic tau-bench and ACEBench were cloned into `external/`; tau-bench was
  installed locally.  ACEBench data/eval dependencies are installed while
  upstream pins that conflict with the CARGO/LiteLLM runtime
  (`openai==1.64.0`, `python-dotenv==1.0.1`, `vllm==0.6.1.post1`) are skipped
  by default. Use `--include-ace-vllm --include-ace-conflicting-pins` only in
  an isolated virtualenv for exact upstream reproduction. Live smoke still
  needs a model endpoint/API key.
- Airline full booking selection is guarded for slot completeness, state
  consistency, obligation-guided search progression, empty-search exhaustion,
  and premature payment asks. A complete deterministic cheapest-itinerary
  selector after non-empty search results remains the next extension. CARGO
  remains a lightweight risk-gated controller, not a full tree-search planner.

## Verification

- `python3 -m unittest tests.test_cargo`: 251 local regression tests passed.
- `python3 -m compileall src tests`: passed.
- `git diff --check`: passed.
- `bash run_project.sh --dry-run`: passed configuration resolution without
  launching a model server or benchmark run.
- `python3 scripts/benchmark_setup.py --bench all --install --json-out outputs/smoke/benchmark_setup_latest.json`: tau-bench install and ACEBench safe-dependency install completed after network approval.
- `python3 scripts/run_smoke.py --target all --json-out outputs/smoke/smoke_summary_latest.json`: synthetic checks passed offline; tau retail/airline and ACE live smoke were honestly marked blocked because no `OPENAI_API_KEY` or `OPENAI_BASE_URL` was present.
- `python3 scripts/parse_smoke_results.py --smoke-summary outputs/smoke/smoke_summary_latest.json --json-out outputs/smoke/smoke_compact_latest.json`: passed.
- `python3 -m unittest tests.test_cargo -v`: 267 local regression tests passed.
- `python3 -m compileall src tests scripts`: passed.
- `python3 scripts/benchmark_setup.py --bench all --install`: passed with
  tau-bench installed and ACEBench conflicting pins skipped by default.
- `python3 -m pip check`: no broken requirements found.
- `git diff --check`: passed.
- `bash run_project.sh --dry-run`: passed configuration resolution without
  launching a model server or benchmark run.
- `python3 scripts/run_smoke.py --target all --json-out outputs/smoke/smoke_summary_latest.json`:
  synthetic checks passed; tau retail/airline and ACE live smoke are blocked
  unless `OPENAI_API_KEY` or `OPENAI_BASE_URL` is present.
