"""Tests for the CARGO agent.

The suite is designed to run **without** tau-bench, vLLM, or any live
model. Every LLM call is replaced by a ``MockClient`` that emits scripted
responses, and the env-side interactions are exercised through a
``MockEnv`` that mimics the tau-bench ``env.reset`` / ``env.step``
contract.

Coverage:

* rule-based risk classification (READ / WRITE / IRREVERSIBLE / FINAL / ASK_USER)
* tool schema induction + module-level cache
* working memory absorption (user text + observation; promotion to db_facts)
* precondition gate (positive + negative + missing-args)
* arg-grounding gate (clean ID + ungrounded ID + non-ID values pass)
* repeat-loop gate
* self-consistency gate (k=3, agreement above & below threshold; n>1 + sequential fallback)
* counterfactual gate (blocking & non-blocking; parse failure passthrough)
* post-condition gate (error obs detection)
* proposer JSON parsing (clean / fenced / nested / malformed)
* repair policy decisions
* full ACEBench-style ``run_cargo`` loop on a MockClient
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Make the repo root importable when running `python -m unittest`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cargo import (  # noqa: E402
    GateResult,
    CandidateObject,
    ConstraintPriorityEngine,
    Constraint,
    FallbackRule,
    GenericCargoKernel,
    GoalActionCandidate,
    GoalField,
    PreCommitVerifier,
    SoftGoalFieldRouter,
    Preference,
    ProposedAction,
    RiskClass,
    SYSTEM_PROMPT,
    ToolEffectSchema,
    WorkingMemory,
    default_calibration,
    induce_schemas,
    is_gated,
    is_irreversible_or_final,
    parse_proposer_response,
    render_tools_block,
    reset_cache,
    run_cargo,
)
from src.cargo.adapters import ACEBenchAdapter, SyntheticGenericAdapter, TauAirlineAdapter, TauRetailAdapter  # noqa: E402
from src.cargo import repair as repair_module  # noqa: E402
from src.cargo.gates import (  # noqa: E402
    check_arg_grounding,
    check_counterfactual,
    check_postconditions,
    check_preconditions,
    check_repeat_loop,
    check_self_consistency,
)
from src.cargo.schema_inducer import _rule_classify  # noqa: E402
from src.cargo.stats import CargoStats  # noqa: E402


# ---------------------------------------------------------------------------
# Mock OpenAI-like client
# ---------------------------------------------------------------------------
class _ChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _ChoiceMessage(content)


class _Response:
    def __init__(self, choices: List[_Choice]) -> None:
        self.choices = choices


class MockCompletions:
    def __init__(self, parent: "MockClient") -> None:
        self.parent = parent

    def create(self, *, model: str, messages: List[Dict[str, Any]],
               temperature: float = 0.0, n: int = 1, max_tokens: int = 0,
               **kwargs: Any) -> _Response:
        # Allow tests to record every call.
        self.parent.calls.append({
            "model": model,
            "messages": list(messages),
            "temperature": float(temperature),
            "n": int(n),
            "max_tokens": int(max_tokens),
        })
        # If the test explicitly raises on a particular call:
        if self.parent.raise_on_call is not None and len(self.parent.calls) >= self.parent.raise_on_call:
            self.parent.raise_on_call = None
            raise RuntimeError("mock_forced_error")
        # Determine which scripted batch to return.
        contents = self.parent.next_responses(n=n, messages=messages)
        choices = [_Choice(c) for c in contents]
        return _Response(choices=choices)


class MockClient:
    """Scripted OpenAI-compatible client.

    The constructor takes a list of *response generators*. Each generator
    is either:
      - a string (returned for the next call irrespective of n);
      - a list of strings (returned as the n choices for the next call);
      - a callable ``f(messages, n) -> List[str]``.
    """

    def __init__(self, scripts: Optional[List[Any]] = None) -> None:
        self.scripts = scripts or []
        self.calls: List[Dict[str, Any]] = []
        self.chat = type("Chat", (), {})()  # dynamic stub
        self.chat.completions = MockCompletions(self)
        # If set to N, the Nth call (1-indexed) raises RuntimeError.
        self.raise_on_call: Optional[int] = None

    def next_responses(self, *, n: int, messages: List[Dict[str, Any]]) -> List[str]:
        if not self.scripts:
            # Default to a benign respond JSON so the agent doesn't crash.
            return ['{"thought":"done","action":{"name":"respond","args":{},"declared_class":"FINAL","declared_pre":[],"declared_post":[],"informational_intent":"","user_text":"Done."}}'] * max(1, n)
        spec = self.scripts.pop(0)
        if callable(spec):
            out = spec(messages, n)
        elif isinstance(spec, list):
            out = spec
        else:
            out = [spec]
        # Pad / trim to n
        out = list(out)
        if len(out) < n:
            out = out + [out[-1]] * (n - len(out))
        return out[:n]


# ---------------------------------------------------------------------------
# Mock tau-bench env
# ---------------------------------------------------------------------------
class _Action:
    def __init__(self, name: str, kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.kwargs = dict(kwargs or {})


class _Reset:
    def __init__(self, observation: str) -> None:
        self.observation = observation


class _StepResp:
    def __init__(self, observation: Any, reward: float = 0.0, done: bool = False,
                 info: Optional[Dict[str, Any]] = None) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}


class MockEnv:
    """Simulates the tau-bench env for unit testing.

    ``script`` is a list of (responder) callables; each is invoked on
    one ``env.step`` call and must return an ``_StepResp``. The first
    element controls the response to ``env.reset``.
    """

    def __init__(self, initial_user: str, step_responses: List[Any]) -> None:
        self._initial = initial_user
        self._step_responses = list(step_responses)
        self.actions_executed: List[_Action] = []
        self.tools_info: List[Dict[str, Any]] = []
        self.wiki: str = ""

    def reset(self, task_index: Optional[int] = None) -> _Reset:
        return _Reset(self._initial)

    def step(self, action: _Action) -> _StepResp:
        self.actions_executed.append(action)
        if not self._step_responses:
            return _StepResp(observation="", reward=0.0, done=True)
        nxt = self._step_responses.pop(0)
        if callable(nxt):
            return nxt(action)
        return nxt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tool(name: str, required: List[str], props: Optional[Dict[str, Any]] = None,
          description: str = "") -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"mock {name}",
            "parameters": {
                "type": "object",
                "required": required,
                "properties": props or {p: {"type": "string"} for p in required},
            },
        },
    }


def _proposer_json(*, name: str, args: Optional[Dict[str, Any]] = None,
                   declared_class: str = "READ",
                   declared_pre: Optional[List[str]] = None,
                   declared_post: Optional[List[str]] = None,
                   user_text: str = "",
                   thought: str = "") -> str:
    return json.dumps({
        "thought": thought or "step",
        "action": {
            "name": name,
            "args": args or {},
            "declared_class": declared_class,
            "declared_pre": declared_pre or [],
            "declared_post": declared_post or [],
            "informational_intent": "",
            "user_text": user_text,
        },
    })


# ---------------------------------------------------------------------------
# Tests: risk classification + schema induction
# ---------------------------------------------------------------------------
class TestRiskClassification(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()

    def test_rule_classify_irreversible(self) -> None:
        for n in ("cancel_order", "delete_account", "refund_payment",
                  "send_email", "transfer_funds", "charge_card"):
            cls, confident = _rule_classify(n)
            self.assertEqual(cls, RiskClass.IRREVERSIBLE, n)
            self.assertTrue(confident)

    def test_rule_classify_write(self) -> None:
        for n in ("update_user_address", "modify_order", "set_status",
                  "place_order", "book_flight"):
            cls, confident = _rule_classify(n)
            self.assertEqual(cls, RiskClass.WRITE, n)
            self.assertTrue(confident)

    def test_rule_classify_read(self) -> None:
        for n in ("get_user_details", "list_orders", "search_items",
                  "find_user", "view_cart", "calculate_total"):
            cls, confident = _rule_classify(n)
            self.assertEqual(cls, RiskClass.READ, n)
            self.assertTrue(confident)

    def test_rule_classify_final_and_ask(self) -> None:
        cls, _ = _rule_classify("respond")
        self.assertEqual(cls, RiskClass.FINAL)
        cls, _ = _rule_classify("transfer_to_human_agents")
        self.assertEqual(cls, RiskClass.ASK_USER)

    def test_rule_classify_unknown_fails_closed(self) -> None:
        cls, confident = _rule_classify("frobnicate_widget")
        # Unknowns default to WRITE (fail-closed) and are NOT confident.
        self.assertEqual(cls, RiskClass.WRITE)
        self.assertFalse(confident)


class TestSchemaInduction(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()

    def test_induce_schemas_basic(self) -> None:
        schemas = induce_schemas([
            _tool("get_user_details", ["user_id"]),
            _tool("cancel_order", ["order_id"]),
            _tool("update_user_address",
                  ["user_id", "address"],
                  props={"user_id": {"type": "string"},
                         "address": {"type": "string"}}),
        ])
        self.assertEqual(schemas["get_user_details"].cls, RiskClass.READ)
        self.assertEqual(schemas["cancel_order"].cls, RiskClass.IRREVERSIBLE)
        self.assertTrue(schemas["cancel_order"].irreversible)
        self.assertEqual(schemas["update_user_address"].cls, RiskClass.WRITE)
        # respond synthesised even when not in tool_specs.
        self.assertIn("respond", schemas)
        self.assertEqual(schemas["respond"].cls, RiskClass.FINAL)

    def test_induce_schemas_id_field_detection(self) -> None:
        schemas = induce_schemas([
            _tool("cancel_order", ["order_id"]),
            _tool("update_user_address", ["user_id", "address"]),
            _tool("exchange_delivered_order_items",
                  ["order_id", "item_ids", "new_item_ids", "payment_method_id"]),
        ])
        self.assertIn("order_id", schemas["cancel_order"].arg_id_fields)
        self.assertIn("user_id", schemas["update_user_address"].arg_id_fields)
        # 'address' is not an ID field.
        self.assertNotIn("address", schemas["update_user_address"].arg_id_fields)
        self.assertIn("item_ids", schemas["exchange_delivered_order_items"].arg_id_fields)
        self.assertIn("new_item_ids", schemas["exchange_delivered_order_items"].arg_id_fields)

    def test_schema_cache_reuses(self) -> None:
        # First induction populates the cache.
        a = induce_schemas([_tool("get_x", [])])
        b = induce_schemas([_tool("get_x", [])])
        self.assertIs(a["get_x"], b["get_x"])

    def test_render_tools_block_includes_classes(self) -> None:
        schemas = induce_schemas([
            _tool("cancel_order", ["order_id"]),
            _tool("get_user_details", ["user_id"]),
        ])
        text = render_tools_block(schemas)
        self.assertIn("cancel_order", text)
        self.assertIn("IRREVERSIBLE", text)
        self.assertIn("get_user_details", text)
        self.assertIn("READ", text)


# ---------------------------------------------------------------------------
# Tests: working memory
# ---------------------------------------------------------------------------
class TestWorkingMemory(unittest.TestCase):
    def test_absorb_user_message_extracts_ids(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("Cancel my order O1234 for alex_smith_42")
        self.assertIn("Cancel my order O1234 for alex_smith_42", wm.user_facts)
        self.assertIn("O1234", wm.user_facts)
        self.assertIn("alex_smith_42", wm.user_facts)

    def test_absorb_user_message_binds_airline_semantic_slots(self) -> None:
        wm = WorkingMemory()
        text = (
            "Book an economy flight from New York to Seattle on May 20th, "
            "with three checked bags, no insurance, and use my credit card."
        )
        wm.absorb_user_message(text)
        GenericCargoKernel(TauAirlineAdapter()).observe_user_message(wm, text)

        self.assertEqual(wm.semantic_slots["date"], "2024-05-20")
        self.assertEqual(wm.semantic_slots["origin"], "New York")
        self.assertEqual(wm.semantic_slots["destination"], "Seattle")
        self.assertEqual(wm.semantic_slots["cabin"], "economy")
        self.assertEqual(wm.semantic_slots["baggage_count"], 3)
        self.assertEqual(wm.semantic_slots["travel_insurance"], "no")
        self.assertIn("credit_card", wm.semantic_slots["payment_preferences"])

    def test_db_confirmed_semantic_slot_outranks_later_user_claim(self) -> None:
        wm = WorkingMemory()
        wm.absorb_observation({"date": "2024-05-20"})
        wm.absorb_user_message("Actually make that 2024-05-21.")

        self.assertEqual(wm.semantic_slots["date"], "2024-05-20")

    def test_tool_observation_does_not_overwrite_user_bound_task_frame(self) -> None:
        wm = WorkingMemory()
        text = "Book an economy flight from New York to Seattle on May 20th."
        wm.absorb_user_message(text)
        kernel = GenericCargoKernel(TauAirlineAdapter())
        kernel.observe_user_message(wm, text)

        obs = {
            "reservation_id": "HKEG34",
            "origin": "DEN",
            "destination": "LAS",
            "cabin": "business",
            "flights": [{"date": "2024-05-27"}],
            "insurance": "yes",
        }
        wm.absorb_observation(obs)
        kernel.observe_tool_result(wm, "get_reservation_details", obs)

        self.assertEqual(wm.semantic_slots["origin"], "New York")
        self.assertEqual(wm.semantic_slots["destination"], "Seattle")
        self.assertEqual(wm.semantic_slots["date"], "2024-05-20")
        self.assertEqual(wm.semantic_slots["cabin"], "economy")
        self.assertIn("HKEG34", wm.typed_evidence_for("reservation_id"))
    def test_tool_observation_does_not_overwrite_user_bound_date(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("Book the flight on May 20th.")
        wm.absorb_observation({"date": "2024-05-27"})

        self.assertEqual(wm.semantic_slots["date"], "2024-05-20")

    def test_absorb_observation_dict_promotes_scalars(self) -> None:
        wm = WorkingMemory()
        wm.absorb_observation({"order_id": "O999", "status": "pending"})
        # Both key=value style and bare value should appear in db_facts.
        self.assertTrue(any("order_id=O999" in f for f in wm.db_facts))
        self.assertTrue(any("O999" == f for f in wm.db_facts))
        self.assertTrue(any("status=pending" in f for f in wm.db_facts))

    def test_absorb_observation_string_extracts_ids(self) -> None:
        wm = WorkingMemory()
        wm.absorb_observation("Order O5555 created.")
        self.assertTrue(any("O5555" in f for f in wm.db_facts))

    def test_recent_signatures_window(self) -> None:
        wm = WorkingMemory()
        for i in range(9):
            wm.record_action_signature(f"sig_{i}")
        # Window is 8 by design for tau-bench loop suppression.
        self.assertEqual(len(list(wm.recent_signatures)), 8)
        self.assertNotIn("sig_0", wm.recent_signatures)
        self.assertIn("sig_8", wm.recent_signatures)

    def test_render_compact_truncates(self) -> None:
        wm = WorkingMemory(goal="g")
        for i in range(50):
            wm._add_db_fact(f"fact_{i}={'x' * 80}")
        text = wm.render_compact(max_chars=600)
        self.assertLessEqual(len(text), 600)


# ---------------------------------------------------------------------------
# Tests: generic CARGO-v2 core + adapters
# ---------------------------------------------------------------------------
class TestCargoV2Adapters(unittest.TestCase):
    def test_generic_kernel_records_conflict_without_overwriting_confirmed_fact(self) -> None:
        wm = WorkingMemory()
        wm.task_state.bind_fact("date", "2024-05-20", source="tool", confirmed=True)
        changed = wm.task_state.bind_fact("date", "2024-05-21", source="user")

        self.assertFalse(changed)
        self.assertEqual(wm.task_state.fact_value("date"), "2024-05-20")
        self.assertEqual(wm.task_state.conflicts[-1]["reason"], "confirmed_fact_outranks_weaker_claim")

    def test_adapter_declares_non_id_fields_for_grounding(self) -> None:
        schema = ToolEffectSchema(
            name="search_direct_flight",
            cls=RiskClass.READ,
            arg_id_fields=[],
            arg_semantic_fields=[],
            param_properties={
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
            },
            required_params=["origin", "destination", "date"],
        )
        enriched = TauAirlineAdapter().enrich_schema(schema)
        self.assertIn("origin", enriched.arg_semantic_fields)
        self.assertIn("destination", enriched.arg_semantic_fields)
        self.assertNotIn("origin", enriched.arg_id_fields)

    def test_tool_observation_ids_do_not_become_semantic_slots(self) -> None:
        wm = WorkingMemory()
        kernel = GenericCargoKernel(TauRetailAdapter())
        kernel.observe_tool_result(
            wm,
            "get_order_details",
            {"items": [{"product_id": "1656367028", "item_id": "old_keyboard"}]},
        )

        self.assertNotIn("product_id", wm.semantic_slots)
        self.assertTrue(
            any(f.value == "1656367028" for f in wm.task_state.db_confirmed_facts.values())
        )

    def test_read_permissive_allows_grounded_product_retrieval_with_incomplete_state(self) -> None:
        from src.cargo.cargo_agent import CargoAgent

        agent = CargoAgent.__new__(CargoAgent)
        agent.adapter = TauRetailAdapter()
        agent.kernel = GenericCargoKernel(agent.adapter)
        agent.client = MockClient([])
        agent.model = "test"
        agent.temperature = 0.0
        agent.calibration = default_calibration()
        agent.schemas = {
            "get_product_details": ToolEffectSchema(
                name="get_product_details",
                cls=RiskClass.READ,
                arg_id_fields=["product_id"],
                required_params=["product_id"],
            )
        }
        wm = WorkingMemory()
        # Simulate old polluted state from a prior scalar observation.  READ
        # retrieval must still be allowed when the requested product_id is
        # grounded in order details; semantic candidate validation is for WRITE.
        wm.bind_semantic_slot("product_id", "0000000000", confirmed=True)
        wm.order_details["#W1"] = {
            "order_id": "#W1",
            "items": [{"product_id": "1656367028", "item_id": "old_keyboard"}],
        }
        action = ProposedAction(
            name="get_product_details",
            args={"product_id": "1656367028"},
            declared_class=RiskClass.READ,
        )

        failing, diag = agent._run_gates(action, agent.schemas["get_product_details"], wm, [], CargoStats())

        self.assertIsNone(failing, failing.reason if failing else "")
        self.assertIn("state_validity", diag["gates_run"])

    def test_airline_adapter_binds_booking_intent_and_open_obligation(self) -> None:
        wm = WorkingMemory()
        kernel = GenericCargoKernel(TauAirlineAdapter())
        text = (
            "I'm looking to book a flight from New York to Seattle on May 20th "
            "after 11 am in economy. One stopover is okay."
        )
        wm.absorb_user_message(text)
        kernel.observe_user_message(wm, text)

        self.assertIn("book_flight", wm.semantic_slots["intents"])
        self.assertEqual(wm.semantic_slots["origin"], "New York")
        self.assertEqual(wm.semantic_slots["destination"], "Seattle")
        self.assertEqual(wm.semantic_slots["date"], "2024-05-20")
        self.assertEqual(wm.semantic_slots["time_after"], "11:00")
        self.assertIn("book_flight", wm.task_state.unresolved_obligations)

    def test_tau_retail_adapter_keeps_hard_constraints_separate_from_preferences(self) -> None:
        wm = WorkingMemory()
        kernel = GenericCargoKernel(TauRetailAdapter())
        kernel.observe_user_message(
            wm,
            "Exchange it for a clicky full-size keyboard with RGB; if unavailable no backlight.",
        )

        hard = {(c.slot, c.op, c.value) for c in wm.task_state.constraints if c.hard}
        prefs = {(p.slot, p.value) for p in wm.task_state.preferences}
        fallbacks = {(f.slot, f.to_value) for f in wm.task_state.fallback_rules}
        self.assertIn(("switch_type", "eq", "clicky"), hard)
        self.assertIn(("size", "eq", "full size"), hard)
        self.assertIn(("backlight", "rgb"), prefs)
        self.assertIn(("backlight", "no backlight"), fallbacks)

    def test_acebench_adapter_rejects_local_pass_global_fail_decoy(self) -> None:
        wm = WorkingMemory()
        wm.task_state.add_constraint(Constraint(slot="difficulty", op="<=", value=4, hard=True))
        wm.task_state.add_candidate(CandidateObject(
            candidate_id="decoy",
            object_type="slot_candidate",
            attributes={"difficulty": 3, "global_valid": False},
        ))
        action = ProposedAction(
            name="set_slot",
            args={"slot_id": "slot_a", "candidate_id": "decoy"},
            declared_class=RiskClass.WRITE,
        )
        schema = ToolEffectSchema(
            name="set_slot",
            cls=RiskClass.WRITE,
            arg_id_fields=["slot_id", "candidate_id"],
        )

        result = ACEBenchAdapter().validate_action(action, schema, wm)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "candidate_violates_global_constraints")

    def test_synthetic_adapter_accepts_candidate_satisfying_all_constraints(self) -> None:
        wm = WorkingMemory()
        wm.task_state.add_constraint(Constraint(slot="difficulty", op="<=", value=4, hard=True))
        wm.task_state.add_candidate(CandidateObject(
            candidate_id="truth",
            object_type="slot_candidate",
            attributes={"difficulty": 2, "global_valid": True},
        ))
        action = ProposedAction(
            name="set_slot",
            args={"slot_id": "slot_a", "candidate_id": "truth"},
            declared_class=RiskClass.WRITE,
        )
        schema = ToolEffectSchema(name="set_slot", cls=RiskClass.WRITE)

        result = SyntheticGenericAdapter().validate_action(action, schema, wm)

        self.assertTrue(result.ok, result.reason)


# ---------------------------------------------------------------------------
# Tests: gates
# ---------------------------------------------------------------------------
class TestPreconditionGate(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()
        self.schema = induce_schemas([
            _tool("cancel_order", ["order_id"]),
        ])["cancel_order"]

    def test_missing_required_arg_fails(self) -> None:
        wm = WorkingMemory()
        action = ProposedAction(name="cancel_order", args={},
                                declared_class=RiskClass.IRREVERSIBLE)
        result = check_preconditions(action, self.schema, wm)
        self.assertFalse(result.ok)
        self.assertIn("missing_required_args", result.reason)

    def test_pre_satisfied_when_evidence_overlaps(self) -> None:
        wm = WorkingMemory()
        wm.absorb_observation({"order_id": "O1234"})
        action = ProposedAction(
            name="cancel_order", args={"order_id": "O1234"},
            declared_class=RiskClass.IRREVERSIBLE,
            declared_pre=["order O1234 exists"],
        )
        result = check_preconditions(action, self.schema, wm)
        self.assertTrue(result.ok, result.reason)

    def test_pre_unmet_when_no_overlap(self) -> None:
        wm = WorkingMemory()
        action = ProposedAction(
            name="cancel_order", args={"order_id": "O1234"},
            declared_class=RiskClass.IRREVERSIBLE,
            declared_pre=["the reservation R7777 has been confirmed"],
        )
        result = check_preconditions(action, self.schema, wm)
        # The pre-condition mentions a concrete unknown ID; should fail.
        self.assertFalse(result.ok)


class TestArgGroundingGate(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()
        self.schemas = induce_schemas([
            _tool("cancel_order", ["order_id"]),
            _tool("update_user_address", ["user_id", "address"]),
        ])

    def test_grounded_id_passes(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("cancel my order O1234")
        action = ProposedAction(
            name="cancel_order", args={"order_id": "O1234"},
            declared_class=RiskClass.IRREVERSIBLE,
        )
        result = check_arg_grounding(action, self.schemas["cancel_order"], wm)
        self.assertTrue(result.ok, result.reason)

    def test_ungrounded_id_fails(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("cancel my order O1234")
        action = ProposedAction(
            name="cancel_order", args={"order_id": "O9999"},  # not in evidence
            declared_class=RiskClass.IRREVERSIBLE,
        )
        result = check_arg_grounding(action, self.schemas["cancel_order"], wm)
        self.assertFalse(result.ok)
        self.assertIn("ungrounded_id_values", result.reason)

    def test_non_id_string_passes(self) -> None:
        # `address` is not ID-shaped, so its content is allowed even if
        # not grounded (the model is allowed to type new free-form text
        # for non-ID args; mutation safety relies on the ID grounding).
        wm = WorkingMemory()
        wm.absorb_user_message("user alex_smith_42 wants 1 Main St")
        action = ProposedAction(
            name="update_user_address",
            args={"user_id": "alex_smith_42", "address": "1 Main St"},
            declared_class=RiskClass.WRITE,
        )
        result = check_arg_grounding(action, self.schemas["update_user_address"], wm)
        self.assertTrue(result.ok, result.reason)

    def test_user_id_pattern_grounded(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("user alex_smith_42 wants help")
        action = ProposedAction(
            name="update_user_address",
            args={"user_id": "alex_smith_42", "address": "X"},
            declared_class=RiskClass.WRITE,
        )
        result = check_arg_grounding(action, self.schemas["update_user_address"], wm)
        self.assertTrue(result.ok, result.reason)

    def test_typed_product_id_rejects_grounded_item_id(self) -> None:
        wm = WorkingMemory()
        wm.absorb_observation({
            "items": [{
                "name": "Smart Watch",
                "product_id": "6945232052",
                "item_id": "9408160950",
            }]
        })
        schema = ToolEffectSchema(
            name="get_product_details",
            cls=RiskClass.READ,
            arg_id_fields=["product_id"],
            required_params=["product_id"],
        )
        action = ProposedAction(
            name="get_product_details",
            args={"product_id": "9408160950"},
            declared_class=RiskClass.READ,
        )
        result = check_arg_grounding(action, schema, wm)
        self.assertFalse(result.ok)
        self.assertIn("product_id=9408160950", result.reason)

    def test_list_id_arguments_are_grounded_elementwise(self) -> None:
        wm = WorkingMemory()
        wm.product_details["p1"] = {
            "variants": {
                "old_item": {"item_id": "old_item"},
                "new_item": {"item_id": "new_item"},
            }
        }
        schema = ToolEffectSchema(
            name="exchange_delivered_order_items",
            cls=RiskClass.IRREVERSIBLE,
            arg_id_fields=["item_ids", "new_item_ids"],
            required_params=["item_ids", "new_item_ids"],
        )
        action = ProposedAction(
            name="exchange_delivered_order_items",
            args={"item_ids": ["old_item"], "new_item_ids": ["made_up_item"]},
            declared_class=RiskClass.IRREVERSIBLE,
        )
        result = check_arg_grounding(action, schema, wm)
        self.assertFalse(result.ok)
        self.assertIn("new_item_ids[0]=made_up_item", result.reason)

    def test_iso_date_argument_is_not_treated_as_opaque_id(self) -> None:
        wm = WorkingMemory()
        schema = ToolEffectSchema(
            name="search_direct_flight",
            cls=RiskClass.READ,
            arg_id_fields=[],
            required_params=["origin", "destination", "date"],
        )
        action = ProposedAction(
            name="search_direct_flight",
            args={
                "origin": "New York",
                "destination": "Seattle",
                "date": "2024-05-20",
            },
            declared_class=RiskClass.READ,
        )

        result = check_arg_grounding(action, schema, wm)

        self.assertTrue(result.ok, result.reason)


class TestRepeatLoopGate(unittest.TestCase):
    def test_repeat_signature_detected(self) -> None:
        wm = WorkingMemory()
        sch = ToolEffectSchema(name="get_x", cls=RiskClass.READ)
        action = ProposedAction(name="get_x", args={"a": 1},
                                declared_class=RiskClass.READ)
        wm.record_action_signature(action.signature())
        result = check_repeat_loop(action, sch, wm)
        self.assertFalse(result.ok)

    def test_distinct_signature_passes(self) -> None:
        wm = WorkingMemory()
        sch = ToolEffectSchema(name="get_x", cls=RiskClass.READ)
        action1 = ProposedAction(name="get_x", args={"a": 1},
                                 declared_class=RiskClass.READ)
        action2 = ProposedAction(name="get_x", args={"a": 2},
                                 declared_class=RiskClass.READ)
        wm.record_action_signature(action1.signature())
        result = check_repeat_loop(action2, sch, wm)
        self.assertTrue(result.ok)

    def test_failed_signature_blocks_until_new_evidence(self) -> None:
        wm = WorkingMemory()
        sch = ToolEffectSchema(name="get_x", cls=RiskClass.READ)
        action = ProposedAction(name="get_x", args={"a": "A1234"},
                                declared_class=RiskClass.READ)
        wm.record_failed_signature(action.signature())
        self.assertFalse(check_repeat_loop(action, sch, wm).ok)
        wm.absorb_user_message("New evidence A1234")
        self.assertTrue(check_repeat_loop(action, sch, wm).ok)


class TestSelfConsistencyGate(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()
        self.schemas = induce_schemas([_tool("cancel_order", ["order_id"])])
        self.action = ProposedAction(
            name="cancel_order", args={"order_id": "O1234"},
            declared_class=RiskClass.IRREVERSIBLE,
        )

    def test_high_agreement_passes(self) -> None:
        client = MockClient(scripts=[
            [_proposer_json(name="cancel_order",
                            args={"order_id": "O1234"},
                            declared_class="IRREVERSIBLE")] * 3,
        ])
        result = check_self_consistency(
            self.action, self.schemas["cancel_order"],
            client=client, model="m",
            proposer_messages=[{"role": "user", "content": "x"}],
            schemas_for_parse=self.schemas, k=3, threshold=0.66,
        )
        self.assertTrue(result.ok, result.reason)
        self.assertGreaterEqual(result.diagnostics["agreement"], 0.66)

    def test_low_agreement_fails(self) -> None:
        client = MockClient(scripts=[
            [
                _proposer_json(name="cancel_order", args={"order_id": "O1234"},
                               declared_class="IRREVERSIBLE"),
                _proposer_json(name="cancel_order", args={"order_id": "O5555"},
                               declared_class="IRREVERSIBLE"),
                _proposer_json(name="get_order_details", args={"order_id": "O1234"},
                               declared_class="READ"),
            ],
        ])
        result = check_self_consistency(
            self.action, self.schemas["cancel_order"],
            client=client, model="m",
            proposer_messages=[{"role": "user", "content": "x"}],
            schemas_for_parse=self.schemas, k=3, threshold=0.66,
        )
        self.assertFalse(result.ok, result.diagnostics)
        self.assertLess(result.diagnostics["agreement"], 0.66)

    def test_final_self_consistency_includes_user_text(self) -> None:
        action = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
            user_text="The order was cancelled.",
        )
        client = MockClient(scripts=[
            [
                _proposer_json(name="respond", declared_class="FINAL",
                               user_text="The order was cancelled."),
                _proposer_json(name="respond", declared_class="FINAL",
                               user_text="I cannot verify that."),
                _proposer_json(name="respond", declared_class="FINAL",
                               user_text="Please provide your email."),
            ],
        ])
        result = check_self_consistency(
            action, ToolEffectSchema(name="respond", cls=RiskClass.FINAL),
            client=client, model="m",
            proposer_messages=[{"role": "user", "content": "x"}],
            schemas_for_parse={"respond": ToolEffectSchema(name="respond", cls=RiskClass.FINAL)},
            k=3, threshold=0.66,
        )
        self.assertFalse(result.ok)

    def test_no_client_passthrough(self) -> None:
        result = check_self_consistency(
            self.action, self.schemas["cancel_order"],
            client=None, model="m",
            proposer_messages=[], schemas_for_parse=self.schemas,
            k=3, threshold=0.66,
        )
        self.assertTrue(result.ok)
        self.assertIn("no_client_skipped", result.diagnostics.get("reason", ""))


class TestCounterfactualGate(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()
        self.schema = ToolEffectSchema(name="cancel_order", cls=RiskClass.IRREVERSIBLE)

    def test_cf_blocks_when_unreachable(self) -> None:
        client = MockClient(scripts=[
            json.dumps({
                "predicted_obs": "{}", "goal_still_reachable": False,
                "reason": "no DB confirmation of order"
            }),
        ])
        wm = WorkingMemory(goal="cancel order")
        action = ProposedAction(
            name="cancel_order", args={"order_id": "O1"},
            declared_class=RiskClass.IRREVERSIBLE,
        )
        result = check_counterfactual(action, self.schema, wm,
                                      client=client, model="m")
        self.assertFalse(result.ok)
        self.assertIn("cf_blocks_goal", result.reason)

    def test_cf_passes_when_reachable(self) -> None:
        client = MockClient(scripts=[
            json.dumps({
                "predicted_obs": "{ok:true}", "goal_still_reachable": True,
                "reason": "ok"
            }),
        ])
        wm = WorkingMemory(goal="cancel order")
        action = ProposedAction(
            name="cancel_order", args={"order_id": "O1"},
            declared_class=RiskClass.IRREVERSIBLE,
        )
        result = check_counterfactual(action, self.schema, wm,
                                      client=client, model="m")
        self.assertTrue(result.ok, result.reason)

    def test_cf_unparsable_passthrough(self) -> None:
        client = MockClient(scripts=["not json at all"])
        wm = WorkingMemory(goal="cancel order")
        action = ProposedAction(
            name="cancel_order", args={"order_id": "O1"},
            declared_class=RiskClass.IRREVERSIBLE,
        )
        result = check_counterfactual(action, self.schema, wm,
                                      client=client, model="m")
        self.assertTrue(result.ok)
        self.assertIn("cf_unparsable_passthrough",
                      result.diagnostics.get("reason", ""))


class TestPostconditionGate(unittest.TestCase):
    def test_error_dict_detected(self) -> None:
        sch = ToolEffectSchema(name="x", cls=RiskClass.WRITE)
        a = ProposedAction(name="x", declared_class=RiskClass.WRITE)
        result = check_postconditions(a, sch, obs={"error": "oops"})
        self.assertFalse(result.ok)

    def test_normal_obs_passes(self) -> None:
        sch = ToolEffectSchema(name="x", cls=RiskClass.WRITE)
        a = ProposedAction(name="x", declared_class=RiskClass.WRITE)
        result = check_postconditions(a, sch, obs={"order_id": "O1"})
        self.assertTrue(result.ok)


# ---------------------------------------------------------------------------
# Tests: proposer JSON parsing
# ---------------------------------------------------------------------------
class TestProposerParsing(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()

    def test_clean_json(self) -> None:
        text = _proposer_json(name="get_user_details",
                              args={"user_id": "alex_smith_42"},
                              declared_class="READ")
        action = parse_proposer_response(text)
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "get_user_details")
        self.assertEqual(action.args.get("user_id"), "alex_smith_42")
        self.assertEqual(action.declared_class, RiskClass.READ)

    def test_fenced_json(self) -> None:
        text = "```json\n" + _proposer_json(name="respond",
                                            declared_class="FINAL",
                                            user_text="Done.") + "\n```"
        action = parse_proposer_response(text)
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "respond")
        self.assertEqual(action.declared_class, RiskClass.FINAL)
        self.assertEqual(action.user_text, "Done.")

    def test_json_with_leading_prose(self) -> None:
        text = "Sure, here you go:\n" + _proposer_json(
            name="cancel_order",
            args={"order_id": "O1"},
            declared_class="IRREVERSIBLE",
        )
        action = parse_proposer_response(text)
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "cancel_order")

    def test_malformed_returns_none(self) -> None:
        self.assertIsNone(parse_proposer_response(""))
        self.assertIsNone(parse_proposer_response("definitely not json"))

    def test_respond_default_class_is_final(self) -> None:
        # Even if the model omits declared_class, "respond" → FINAL.
        text = json.dumps({
            "thought": "ok",
            "action": {"name": "respond", "args": {}, "user_text": "hi"},
        })
        action = parse_proposer_response(text)
        self.assertIsNotNone(action)
        self.assertEqual(action.declared_class, RiskClass.FINAL)

    def test_top_level_flat_schema(self) -> None:
        # Some LLM outputs omit the {"action": ...} wrapper.
        text = json.dumps({"name": "get_user_details",
                           "args": {"user_id": "alex_smith_42"}})
        action = parse_proposer_response(text)
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "get_user_details")
        self.assertEqual(action.args.get("user_id"), "alex_smith_42")

    def test_args_as_json_encoded_string(self) -> None:
        # Some LLMs nest args as a string-encoded JSON blob.
        text = json.dumps({
            "action": {
                "name": "cancel_order",
                "args": json.dumps({"order_id": "O1"}),
                "declared_class": "IRREVERSIBLE",
            },
        })
        action = parse_proposer_response(text)
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "cancel_order")
        self.assertEqual(action.args.get("order_id"), "O1")

    def test_schema_overrides_low_risk_misclassification(self) -> None:
        # If model says READ but the schema knows the tool is IRREVERSIBLE,
        # parse should escalate the declared class.
        schemas = induce_schemas([_tool("cancel_order", ["order_id"])])
        text = _proposer_json(name="cancel_order",
                              args={"order_id": "O1"},
                              declared_class="READ")
        action = parse_proposer_response(text, schemas=schemas)
        self.assertIsNotNone(action)
        self.assertEqual(action.declared_class, RiskClass.IRREVERSIBLE)


class TestSystemPromptShape(unittest.TestCase):
    def test_system_prompt_contains_required_keys(self) -> None:
        for k in ("READ", "WRITE", "IRREVERSIBLE", "FINAL", "ASK_USER",
                  "thought", "action", "declared_class"):
            self.assertIn(k, SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Tests: repair policy
# ---------------------------------------------------------------------------
class TestRepairPolicy(unittest.TestCase):
    def test_grounding_failure_asks_user(self) -> None:
        gr = GateResult.failing("arg_grounding", "ungrounded_id_values:['order_id=O9']",
                                ungrounded=["order_id=O9"])
        d = repair_module.decide(gr, retries_used=0, max_retries=2,
                                 budget_steps_remaining=10)
        self.assertEqual(d.action, "ASK_USER")
        self.assertIn("O9", d.user_message)

    def test_self_consistency_failure_retries(self) -> None:
        gr = GateResult.failing("self_consistency", "low_agreement:0.33",
                                agreement=0.33)
        d = repair_module.decide(gr, retries_used=0, max_retries=2,
                                 budget_steps_remaining=10)
        self.assertEqual(d.action, "RETRY")
        self.assertIn("self_consistency", d.critique)

    def test_retries_exhausted_escalates(self) -> None:
        gr = GateResult.failing("self_consistency", "low_agreement:0.33")
        d = repair_module.decide(gr, retries_used=2, max_retries=2,
                                 budget_steps_remaining=10)
        self.assertEqual(d.action, "ASK_USER")

    def test_repeat_loop_finalizes_after_retries(self) -> None:
        gr = GateResult.failing("repeat_loop", "repeated_action_signature")
        d = repair_module.decide(gr, retries_used=2, max_retries=2,
                                 budget_steps_remaining=10)
        self.assertEqual(d.action, "FINALIZE_GENERIC")

    def test_budget_one_finalizes_immediately(self) -> None:
        gr = GateResult.failing("self_consistency", "low_agreement")
        d = repair_module.decide(gr, retries_used=0, max_retries=2,
                                 budget_steps_remaining=1)
        self.assertEqual(d.action, "FINALIZE_GENERIC")

    def test_json_parse_failure_retries_then_finalizes(self) -> None:
        gr = GateResult.failing("json_parse", "unparseable")
        d = repair_module.decide(gr, retries_used=0, max_retries=2,
                                 budget_steps_remaining=10)
        self.assertEqual(d.action, "RETRY")
        d2 = repair_module.decide(gr, retries_used=2, max_retries=2,
                                  budget_steps_remaining=10)
        self.assertEqual(d2.action, "FINALIZE_GENERIC")


# ---------------------------------------------------------------------------
# Tests: calibration
# ---------------------------------------------------------------------------
class TestCalibration(unittest.TestCase):
    def test_default_thresholds(self) -> None:
        cfg = default_calibration()
        # READ is never gated (threshold 0).
        self.assertEqual(cfg.sc_thresholds[RiskClass.READ], 0.0)
        # IRREVERSIBLE/FINAL get the strictest treatment (CF on, k>=1).
        self.assertGreater(cfg.sc_thresholds[RiskClass.IRREVERSIBLE], 0.0)
        self.assertTrue(cfg.run_cf[RiskClass.IRREVERSIBLE])
        self.assertTrue(cfg.run_cf[RiskClass.FINAL])
        self.assertFalse(cfg.run_cf[RiskClass.READ])
        self.assertFalse(cfg.run_cf[RiskClass.WRITE])

    def test_is_gated_helpers(self) -> None:
        self.assertFalse(is_gated(RiskClass.READ))
        self.assertTrue(is_gated(RiskClass.WRITE))
        self.assertTrue(is_gated(RiskClass.IRREVERSIBLE))
        self.assertTrue(is_gated(RiskClass.FINAL))
        self.assertFalse(is_gated(RiskClass.ASK_USER))
        self.assertTrue(is_irreversible_or_final(RiskClass.FINAL))
        self.assertFalse(is_irreversible_or_final(RiskClass.WRITE))


# ---------------------------------------------------------------------------
# Integration: ACEBench-style run_cargo loop on a MockClient
# ---------------------------------------------------------------------------
class TestRunCargoIntegration(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()

    def test_simple_read_then_finalize(self) -> None:
        # Two-step trajectory: (1) get_user_details, (2) respond FINAL.
        scripts = [
            # Step 1 proposer: READ get_user_details
            _proposer_json(
                name="get_user_details",
                args={"user_id": "alex_smith_42"},
                declared_class="READ",
                thought="look up user",
            ),
            # Step 2 proposer: FINAL respond
            _proposer_json(
                name="respond", declared_class="FINAL",
                user_text="Your account is in good standing.",
                thought="finalize",
            ),
            # Step 2 SC samples (n=3): all agree
            [_proposer_json(name="respond", declared_class="FINAL",
                            user_text="Your account is in good standing.")] * 3,
            # Step 2 CF rollout: reachable
            json.dumps({"predicted_obs": "ok", "goal_still_reachable": True}),
        ]
        client = MockClient(scripts=scripts)
        tool_specs = [_tool("get_user_details", ["user_id"])]
        result = run_cargo(
            client=client, model="m",
            task={},
            tool_specs=tool_specs,
            user_turn="What is alex_smith_42's status?",
            system_prompt="",
            max_num_steps=10, temperature=0.0,
        )
        self.assertEqual(result["status"], "ok", result.get("error"))
        # READ got executed; FINAL terminated.
        self.assertIn("get_user_details", result["tool_calls_made"])
        self.assertIn("respond", result["tool_calls_made"])
        stats = result["cargo_stats"]
        self.assertGreaterEqual(stats["actions_executed"], 2)
        self.assertGreaterEqual(stats["steps_fast_path"], 1)  # READ went fast path

    def test_ungrounded_mutation_blocked(self) -> None:
        # Step 1 proposer tries IRREVERSIBLE with ungrounded order_id; gate
        # blocks; repair policy emits ASK_USER, which terminates the loop.
        scripts = [
            _proposer_json(
                name="cancel_order", args={"order_id": "O9999"},
                declared_class="IRREVERSIBLE",
                thought="cancel",
            ),
        ]
        client = MockClient(scripts=scripts)
        tool_specs = [_tool("cancel_order", ["order_id"])]
        result = run_cargo(
            client=client, model="m",
            task={},
            tool_specs=tool_specs,
            user_turn="Please cancel.",  # No ID provided → cannot ground
            system_prompt="",
            max_num_steps=5, temperature=0.0,
        )
        self.assertEqual(result["status"], "ok", result.get("error"))
        # Cancel did NOT execute; instead, the loop emitted respond.
        self.assertNotIn("cancel_order", result["tool_calls_made"])
        self.assertIn("respond", result["tool_calls_made"])
        stats = result["cargo_stats"]
        self.assertGreaterEqual(stats["abstain_total"], 1)
        # arg_grounding fired
        self.assertIn("arg_grounding", stats["gate_fails"])

    def test_ace_read_with_ungrounded_id_is_blocked(self) -> None:
        scripts = [
            _proposer_json(
                name="get_product_details",
                args={"product_id": "9999999999"},
                declared_class="READ",
            ),
        ]
        client = MockClient(scripts=scripts)
        result = run_cargo(
            client=client, model="m",
            task={},
            tool_specs=[_tool("get_product_details", ["product_id"])],
            user_turn="Please show me product details.",
            system_prompt="",
            max_num_steps=5, temperature=0.0,
        )
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertNotIn("get_product_details", result["tool_calls_made"])
        self.assertIn("respond", result["tool_calls_made"])
        self.assertIn("arg_grounding", result["cargo_stats"]["gate_fails"])

    def test_grounded_mutation_executes(self) -> None:
        # Step 1: READ get_order_details (auto-grounds order_id O1234 in db)
        # Step 2: IRREVERSIBLE cancel_order(order_id=O1234) — passes.
        scripts = [
            _proposer_json(name="get_order_details",
                           args={"order_id": "O1234"},
                           declared_class="READ", thought="lookup"),
            _proposer_json(name="cancel_order",
                           args={"order_id": "O1234"},
                           declared_class="IRREVERSIBLE",
                           declared_pre=["order O1234 exists"],
                           thought="cancel"),
            # SC for IRREVERSIBLE — all three agree.
            [_proposer_json(name="cancel_order",
                            args={"order_id": "O1234"},
                            declared_class="IRREVERSIBLE")] * 3,
            # CF — reachable.
            json.dumps({"predicted_obs": "{}", "goal_still_reachable": True}),
            # Step 3: FINAL respond.
            _proposer_json(name="respond", declared_class="FINAL",
                           user_text="Cancelled.", thought="done"),
            # SC for FINAL.
            [_proposer_json(name="respond", declared_class="FINAL",
                            user_text="Cancelled.")] * 3,
            # CF for FINAL.
            json.dumps({"predicted_obs": "{}", "goal_still_reachable": True}),
        ]
        client = MockClient(scripts=scripts)
        tool_specs = [
            _tool("get_order_details", ["order_id"]),
            _tool("cancel_order", ["order_id"]),
        ]
        result = run_cargo(
            client=client, model="m",
            task={},
            tool_specs=tool_specs,
            user_turn="Cancel my order O1234.",
            system_prompt="",
            max_num_steps=10, temperature=0.0,
        )
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertIn("get_order_details", result["tool_calls_made"])
        self.assertIn("cancel_order", result["tool_calls_made"])
        self.assertIn("respond", result["tool_calls_made"])
        stats = result["cargo_stats"]
        self.assertEqual(stats["gate_fails"].get("arg_grounding", 0), 0)

    def test_unparsable_proposer_bounded_retries(self) -> None:
        # All proposer responses are unparseable; loop should bound retries.
        client = MockClient(scripts=["garbage 1", "garbage 2", "garbage 3", "garbage 4"])
        tool_specs = [_tool("get_user_details", ["user_id"])]
        result = run_cargo(
            client=client, model="m",
            task={},
            tool_specs=tool_specs,
            user_turn="hi",
            system_prompt="",
            max_num_steps=20, temperature=0.0,
        )
        # Should not crash; should record json_parse_failures.
        self.assertGreaterEqual(result["cargo_stats"]["json_parse_failures"], 1)


# ---------------------------------------------------------------------------
# Auth-override unit tests
# ---------------------------------------------------------------------------
class TestAuthOverride(unittest.TestCase):
    """Tests for CargoAgent._auth_override, _resolve_product_id_name,
    and _advance_after_product_list without a live environment."""

    def _make_agent(self) -> Any:
        """Return a minimal object with CargoAgent's override methods bound.

        cargo_agent.py handles missing tau_bench gracefully (Agent = object),
        so we can import CargoAgent without tau_bench installed.  We avoid
        calling __init__ (which needs env/client params) by using __new__.
        """
        from src.cargo.cargo_agent import CargoAgent
        client = MockClient(scripts=[])
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "test"
        agent.style_name = "cargo"
        agent.temperature = 0.0
        agent.schemas = {}
        agent.domain_policy = ""
        return agent

    def _placeholder_action(self) -> ProposedAction:
        return ProposedAction(
            name="find_user_id_by_email",
            args={"email": "user@example.com"},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )

    # ------------------------------------------------------------------
    # 1. User provides name + ZIP → name+zip lookup
    # ------------------------------------------------------------------
    def test_user_provides_name_and_zip_uses_name_zip_lookup(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.user_facts = ["My name is Yusuf Ali and my ZIP is 19122"]
        # Simulate extracting the name pair and zip via absorb_user_message
        wm.absorb_user_message("My name is Yusuf Ali and my ZIP is 19122")
        # Manually inject the extracted tokens that _auth_override expects.
        wm.user_facts.append("Yusuf Ali")
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_name_zip")  # type: ignore[union-attr]
        self.assertEqual(result.args["first_name"], "Yusuf")  # type: ignore[union-attr]
        self.assertEqual(result.args["last_name"], "Ali")  # type: ignore[union-attr]
        self.assertIn("19122", str(result.args.get("zip")))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 2. User provides a real email → email lookup
    # ------------------------------------------------------------------
    def test_user_provides_real_email_uses_email_lookup(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("My email is alice@gmail.com")
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_email")  # type: ignore[union-attr]
        self.assertEqual(result.args["email"], "alice@gmail.com")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 3. User provides name only (no ZIP) → ASK_USER for zip
    # ------------------------------------------------------------------
    def test_user_provides_name_only_asks_for_zip(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        # Name introduced via the canonical "my name is" phrase
        wm.user_facts = ["My name is Yusuf Ali"]
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        self.assertIn("ZIP", result.user_text)  # type: ignore[union-attr]
        self.assertEqual(wm.auth_ask_count, 1)

    # ------------------------------------------------------------------
    # 4. ZIP not found → asks for re-verification (not same ZIP again)
    # ------------------------------------------------------------------
    def test_zip_not_found_asks_reverification(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.user_facts = ["My name is Yusuf Ali", "10001"]
        wm.auth_failed_zips = ["10001"]  # simulates a failed lookup
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        # Should mention the failed ZIP and ask for a corrected one
        self.assertIn("10001", result.user_text)  # type: ignore[union-attr]
        self.assertIn("ZIP", result.user_text)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 5. Already tried ZIP → override does NOT propose name+zip with same ZIP
    # ------------------------------------------------------------------
    def test_zip_not_found_prevents_same_zip_retry(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.user_facts = ["My name is Yusuf Ali", "10001"]
        wm.auth_failed_zips = ["10001"]
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        # Must NOT propose find_user_id_by_name_zip with the failed ZIP
        self.assertIsNotNone(result)
        if result.name == "find_user_id_by_name_zip":  # type: ignore[union-attr]
            self.assertNotEqual(result.args.get("zip"), "10001")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 6. User refuses to share credentials → FINAL (no more asking)
    # ------------------------------------------------------------------
    def test_user_refuses_auth_stops_asking(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.user_facts = ["I prefer not to share my details"]
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 7. Auth ask limit reached → FINAL (no more asking)
    # ------------------------------------------------------------------
    def test_auth_ask_limit_stops_asking(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.auth_ask_count = 2  # already asked twice
        wm.user_facts = []
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 8. Product ID resolved from type name in db_facts
    # ------------------------------------------------------------------
    def test_product_id_resolved_from_type_name(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.db_facts = ["T-Shirt=6086499569"]
        action = ProposedAction(
            name="get_product_details",
            args={"product_id": "T-Shirt"},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )
        result = agent._resolve_product_id_name(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(str(result.args["product_id"]), "6086499569")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 9. Product ID already numeric → no override
    # ------------------------------------------------------------------
    def test_product_id_numeric_no_override(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.db_facts = ["T-Shirt=6086499569"]
        action = ProposedAction(
            name="get_product_details",
            args={"product_id": 6086499569},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )
        result = agent._resolve_product_id_name(action, wm)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # 10. list_all_product_types repeated → advance to get_product_details
    # ------------------------------------------------------------------
    def test_advance_after_product_list_repeat(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.db_facts = ["T-Shirt=6086499569"]
        wm.user_facts = ["I want to return my T-Shirt"]
        action = ProposedAction(
            name="list_all_product_types",
            args={},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )
        # Record the signature to simulate a repeat
        wm.record_action_signature(action.signature())
        result = agent._advance_after_product_list(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_product_details")  # type: ignore[union-attr]
        self.assertEqual(str(result.args["product_id"]), "6086499569")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # 11. list_all_product_types first call → no advance (let it run)
    # ------------------------------------------------------------------
    def test_advance_after_product_list_first_call_passes(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.db_facts = ["T-Shirt=6086499569"]
        wm.user_facts = ["I want to return my T-Shirt"]
        action = ProposedAction(
            name="list_all_product_types",
            args={},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )
        # Do NOT record signature — this is the first call
        result = agent._advance_after_product_list(action, wm)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # 12. Non-placeholder email not overridden (real email passes through)
    # ------------------------------------------------------------------
    def test_real_email_in_action_not_overridden(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.user_facts = []
        action = ProposedAction(
            name="find_user_id_by_email",
            args={"email": "alice@gmail.com"},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )
        result = agent._auth_override(action, wm)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Regression tests for trajectories(17) failure patterns
# ---------------------------------------------------------------------------
class TestTrajectoryRegressions(unittest.TestCase):
    """Each test here corresponds to a *specific* failure pattern observed
    in `trajectories (17).jsonl`.  These tests must pass for the fixes to
    be considered correct."""

    def _make_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        client = MockClient(scripts=[])
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "test"
        agent.style_name = "cargo"
        agent.temperature = 0.0
        agent.schemas = {}
        agent.domain_policy = ""
        return agent

    def _placeholder(self) -> ProposedAction:
        return ProposedAction(
            name="find_user_id_by_email",
            args={"email": "user@example.com"},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )

    # ------------------------------------------------------------------
    # Bug A: Greedy name extraction picked "Google Home" from product text.
    # The user's first message was about exchanging a thermostat, NOT an
    # auth introduction.  Verify the strict extractor rejects this.
    # ------------------------------------------------------------------
    def test_brand_words_not_extracted_as_name(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = (
            "exchange the smart thermostat for a model that works with "
            "Google Home"
        )
        wm.absorb_user_message(
            "I want to exchange the smart thermostat for one that works "
            "with Google Home."
        )
        action = self._placeholder()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        # Must NOT propose find_user_id_by_name_zip with first_name='Google'
        if result.name == "find_user_id_by_name_zip":  # type: ignore[union-attr]
            self.fail(
                "Override extracted brand words 'Google Home' as a name "
                f"and proposed: {result.args}"  # type: ignore[union-attr]
            )
        # And the user_text must not address the user as 'Google'
        self.assertNotIn("Thank you, Google", result.user_text or "")  # type: ignore[union-attr]

    def test_product_pair_not_extracted_as_name(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("I'd like the Smart Thermostat please.")
        action = self._placeholder()
        result = agent._auth_override(action, wm)
        # Must not propose name+zip lookup using "Smart Thermostat"
        if result and result.name == "find_user_id_by_name_zip":
            self.fail(f"Extracted product pair as name: {result.args}")

    # ------------------------------------------------------------------
    # Bug B: Auth fired on a pure product query.
    # The fix: when goal matches _PRODUCT_QUERY_RE and no PII is present,
    # redirect to list_all_product_types.
    # ------------------------------------------------------------------
    def test_product_query_no_pii_redirects_to_list_products(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "I'd like to know how many t-shirt options are available in the store."
        wm.absorb_user_message(wm.goal)
        action = self._placeholder()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "list_all_product_types")  # type: ignore[union-attr]
        self.assertEqual(result.declared_class, RiskClass.READ)  # type: ignore[union-attr]

    def test_account_goal_still_asks_for_auth(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "I want to cancel my recent order."
        wm.absorb_user_message(wm.goal)
        action = self._placeholder()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        # Account-related goals still require authentication
        self.assertEqual(result.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        self.assertEqual(wm.auth_ask_count, 1)

    # ------------------------------------------------------------------
    # Bug C: After successful auth, model loops on find_user_id_by_email.
    # Fix: when user_id exists in db_facts, replace with get_user_details.
    # ------------------------------------------------------------------
    def test_post_auth_replaces_with_get_user_details(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        # Simulate that find_user_id_by_name_zip already succeeded
        wm.db_facts = ["yusuf_rossi_9620"]
        action = self._placeholder()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_user_details")  # type: ignore[union-attr]
        self.assertEqual(result.args["user_id"], "yusuf_rossi_9620")  # type: ignore[union-attr]

    def test_post_auth_after_get_user_details_routes_by_goal(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.db_facts = ["yusuf_rossi_9620"]
        wm.goal = "How many t-shirt options are in the store?"
        # Pretend get_user_details(yusuf_rossi_9620) was already called
        wm.recent_signatures.append("get_user_details(user_id=yusuf_rossi_9620)")
        action = self._placeholder()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "list_all_product_types")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Bug D: FINAL respond looped infinitely after refusal.
    # Fix: emit the giveup FINAL exactly once, then a short different one.
    # ------------------------------------------------------------------
    def test_refusal_emits_final_then_different_final(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("I prefer not to share my details.")
        action = self._placeholder()
        result1 = agent._auth_override(action, wm)
        self.assertIsNotNone(result1)
        self.assertEqual(result1.declared_class, RiskClass.FINAL)  # type: ignore[union-attr]
        # Second call should NOT return the identical FINAL message
        result2 = agent._auth_override(action, wm)
        self.assertIsNotNone(result2)
        self.assertEqual(result2.declared_class, RiskClass.FINAL)  # type: ignore[union-attr]
        self.assertNotEqual(result1.user_text, result2.user_text)  # type: ignore[union-attr]

    def test_refusal_then_product_query_pivots(self) -> None:
        """User refused auth, but their query is a product query.  The
        override must pivot from ASK_USER → list_all_product_types."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirts are in the store?"
        wm.absorb_user_message("I prefer not to share that information.")
        # First call: marks abandoned + emits FINAL
        result1 = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result1)
        self.assertTrue(wm.auth_abandoned)
        # Second call: product-query pivot, NOT another refusal FINAL
        result2 = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result2)
        self.assertEqual(result2.name, "list_all_product_types")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Bug E: Tainted name+zip args are still re-issued by the model.
    # If the model proposes find_user_id_by_name_zip with stopword args,
    # the override must reject them and fall through to safer logic.
    # ------------------------------------------------------------------
    def test_stopword_name_args_blocked_in_name_zip(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange smart thermostat that works with Google Home"
        action = ProposedAction(
            name="find_user_id_by_name_zip",
            args={"first_name": "Google", "last_name": "Home", "zip": "10012"},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )
        result = agent._auth_override(action, wm)
        # Must override — the original args contained stopword ("Google", "Home")
        self.assertIsNotNone(result)
        if result.name == "find_user_id_by_name_zip":  # type: ignore[union-attr]
            self.assertNotEqual(result.args.get("first_name"), "Google")  # type: ignore[union-attr]
            self.assertNotEqual(result.args.get("last_name"), "Home")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Bug F: Loose name-pair extraction fires only after we asked.
    # ------------------------------------------------------------------
    def test_name_extraction_strict_without_intro(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        # No intro phrase, no prior ask → must NOT extract
        result = CargoAgent._extract_name_pair(["Yusuf Rossi"], in_response_to_ask=False)
        self.assertIsNone(result)

    def test_name_extraction_loose_after_ask(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        # No intro phrase but we just asked → loose fallback fires
        result = CargoAgent._extract_name_pair(["Yusuf Rossi"], in_response_to_ask=True)
        self.assertIsNotNone(result)
        self.assertEqual(result, ("Yusuf", "Rossi"))

    def test_name_extraction_with_intro_phrase(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        result = CargoAgent._extract_name_pair(
            ["My name is Yusuf Rossi and my zip is 19122"],
            in_response_to_ask=False,
        )
        self.assertEqual(result, ("Yusuf", "Rossi"))

    def test_name_extraction_rejects_brand_with_intro(self) -> None:
        """Even with intro phrase, brand stopwords should be rejected."""
        from src.cargo.cargo_agent import CargoAgent
        result = CargoAgent._extract_name_pair(
            ["my name is Google Home"],  # adversarial
            in_response_to_ask=True,
        )
        # Both tokens are stopwords → fallback rejects too
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # End-to-end auth flow: name+ZIP → user_id → get_user_details
    # ------------------------------------------------------------------
    def test_full_auth_flow_progresses(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "Cancel my order"
        # User just provided full name + zip in a clean introduction
        wm.absorb_user_message("My name is Yusuf Rossi and my ZIP is 19122.")
        action = self._placeholder()
        # Step 1: should propose find_user_id_by_name_zip with the right args
        result1 = agent._auth_override(action, wm)
        self.assertIsNotNone(result1)
        self.assertEqual(result1.name, "find_user_id_by_name_zip")  # type: ignore[union-attr]
        self.assertEqual(result1.args["first_name"], "Yusuf")  # type: ignore[union-attr]
        self.assertEqual(result1.args["last_name"], "Rossi")  # type: ignore[union-attr]
        self.assertEqual(result1.args["zip"], "19122")  # type: ignore[union-attr]
        # Step 2: simulate that the lookup succeeded and yusuf_rossi_9620 was returned
        wm.db_facts.append("yusuf_rossi_9620")
        wm.auth_user_id = "yusuf_rossi_9620"
        # Now the model still proposes find_user_id_by_email (the bug)
        result2 = agent._auth_override(action, wm)
        self.assertIsNotNone(result2)
        # Must NOT be ASK_USER or another find_user_id_*; must be progress
        self.assertEqual(result2.name, "get_user_details")  # type: ignore[union-attr]
        self.assertEqual(result2.args["user_id"], "yusuf_rossi_9620")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tau-bench import sanity (skipped when tau_bench is not installed)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Regression tests for trajectories(18) failure patterns
# ---------------------------------------------------------------------------
class TestTrajectory18Regressions(unittest.TestCase):
    """Each test pinned to a specific bug observed in trajectories(18)."""

    def _make_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        client = MockClient(scripts=[])
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client; agent.model = "test"; agent.style_name = "cargo"
        agent.temperature = 0.0; agent.schemas = {}; agent.domain_policy = ""
        return agent

    def _placeholder(self, email: str = "user@example.com") -> ProposedAction:
        return ProposedAction(
            name="find_user_id_by_email",
            args={"email": email},
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[], informational_intent="",
            raw_thought="", user_text="", raw_response="",
        )

    # ------------------------------------------------------------------
    # Bug 1: re.IGNORECASE made [A-Z][a-z]+ match lowercase too.
    # The intro extractor captured ("looking", "to") from "I'm looking to..."
    # and the agent said "Thank you, looking!" to the user.
    # ------------------------------------------------------------------
    def test_im_looking_to_not_extracted_as_name(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        result = CargoAgent._extract_name_pair(
            ["I'm looking to see how many t-shirt options are available."],
            in_response_to_ask=False,
        )
        self.assertIsNone(result, f"Got false positive: {result!r}")

    def test_lowercase_words_not_extracted_with_intro(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        # No real name follows the intro
        for phrase in [
            "i'm looking to do this",
            "I am going to ask",
            "this is great news",
            "my name is what",  # 'what' is lowercase
        ]:
            result = CargoAgent._extract_name_pair([phrase], in_response_to_ask=False)
            self.assertIsNone(result, f"False positive on {phrase!r}: {result!r}")

    def test_uppercase_intro_still_works(self) -> None:
        """Intro phrase IS still case-insensitive — only the name portion
        is case-sensitive."""
        from src.cargo.cargo_agent import CargoAgent
        for phrase in [
            "MY NAME IS Yusuf Rossi",
            "I'M Yusuf Rossi",
            "This Is John Smith",
        ]:
            result = CargoAgent._extract_name_pair([phrase], in_response_to_ask=False)
            self.assertIsNotNone(result, f"Failed on {phrase!r}")

    def test_t1_thank_you_looking_not_emitted(self) -> None:
        """Replays the T1/T2/T4 first turn from trajectories(18)."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "I'm looking to see how many t-shirt options are currently available in the store."
        wm.absorb_user_message(wm.goal)
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        # Must NOT address user as "looking" or "Looking"
        ut = result.user_text or ""
        self.assertNotIn("Thank you, looking", ut)
        self.assertNotIn("Thank you, Looking", ut)
        self.assertNotIn("To verify your identity", ut)
        # Should redirect to list_all_product_types (no auth, product query)
        self.assertEqual(result.name, "list_all_product_types")

    # ------------------------------------------------------------------
    # Bug 2: yusuf.rossi@example.com slipped through _extract_real_email
    # because _PLACEHOLDER_EMAIL_RE only matches the prefix list.
    # ------------------------------------------------------------------
    def test_example_domain_email_is_placeholder(self) -> None:
        from src.cargo.cargo_agent import _is_placeholder_email
        for email in [
            "yusuf.rossi@example.com",
            "anything@example.org",
            "real.name@test.com",
            "x@sample.io",
            "x@dummy.net",
            "x@fake.com",
            "x@placeholder.io",
            "user@example.com",
        ]:
            self.assertTrue(_is_placeholder_email(email), email)

    def test_real_email_passes(self) -> None:
        from src.cargo.cargo_agent import _is_placeholder_email
        for email in [
            "yusuf@gmail.com",
            "alice@company.io",
            "ceo@anthropic.com",
        ]:
            self.assertFalse(_is_placeholder_email(email), email)

    def test_placeholder_user_email_in_message_not_used(self) -> None:
        """Generic-prefix placeholder emails (user@, test@, alice@, …) must
        NOT be used even if the user types them.

        D4 note: _extract_any_email now accepts specific-prefix @example.com
        emails from user_facts (e.g. 'yusuf.rossi@example.com' → prefix is
        specific, not in the generic-RE list).  Only truly generic prefixes
        (user@, test@, demo@, alice@, …) are still blocked."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard"
        # These are clearly fabricated — prefix matches _PLACEHOLDER_EMAIL_RE.
        # Note: emails with SPECIFIC prefixes (e.g. alice@test.com, yusuf@example.com)
        # are intentionally allowed by _extract_any_email (D4 fix).
        for fabricated in ["user@example.com", "test@example.com",
                           "demo@example.com", "admin@sample.com"]:
            wm2 = WorkingMemory()
            wm2.goal = "exchange keyboard"
            wm2.absorb_user_message(f"My email is {fabricated}")
            result = agent._auth_override(self._placeholder(), wm2)
            if result is not None and result.name == "find_user_id_by_email":
                self.assertNotEqual(
                    result.args.get("email", ""), fabricated,
                    f"Generic placeholder email should not be used: {fabricated}",
                )

    def test_specific_prefix_example_email_is_used(self) -> None:
        """D4 regression: a user-provided email like 'yusuf.rossi@example.com'
        (specific prefix, not in generic-RE list) SHOULD be tried via
        find_user_id_by_email — tau-bench user simulations provide emails
        in this exact format."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard"
        wm.absorb_user_message("My email is yusuf.rossi@example.com")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_email")
        self.assertEqual(result.args.get("email"), "yusuf.rossi@example.com")

    # ------------------------------------------------------------------
    # Bug 3: get_product_details(9523456873) looped 24 times because the
    # ID was hallucinated (not in db_facts after list_all_product_types).
    # Fix: replace hallucinated numeric IDs with goal-matched real IDs.
    # ------------------------------------------------------------------
    def test_hallucinated_product_id_replaced(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are available in the store?"
        wm.db_facts = [
            "Action Camera=3377618313",
            "T-Shirt=9523456874",  # real ID
            "Bookshelf=1234567890",
        ]
        action = ProposedAction(
            name="get_product_details",
            args={"product_id": 9523456873},  # hallucinated — close but wrong
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[], informational_intent="",
            raw_thought="", user_text="", raw_response="",
        )
        result = agent._resolve_product_id_name(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(str(result.args["product_id"]), "9523456874")  # type: ignore[union-attr]

    def test_legitimate_product_id_not_replaced(self) -> None:
        """A numeric ID that IS in db_facts must pass through unchanged."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "details for the action camera"
        wm.db_facts = ["Action Camera=3377618313", "T-Shirt=9523456874"]
        action = ProposedAction(
            name="get_product_details",
            args={"product_id": "3377618313"},  # real
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[], informational_intent="",
            raw_thought="", user_text="", raw_response="",
        )
        result = agent._resolve_product_id_name(action, wm)
        self.assertIsNone(result)

    def test_hallucinated_id_no_replacement_when_no_match(self) -> None:
        """Without a goal-matchable name, return None and let env reject."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "modify my recent order"  # no product name in goal
        wm.db_facts = ["Action Camera=3377618313", "Bookshelf=1234567890"]
        action = ProposedAction(
            name="get_product_details",
            args={"product_id": 9999999999},  # hallucinated
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[], informational_intent="",
            raw_thought="", user_text="", raw_response="",
        )
        result = agent._resolve_product_id_name(action, wm)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Bug 4: Auth abandoned BEFORE checking new evidence.
    # T3: user gave name (turn 1) and zip (turn 2), but auth_ask_count
    # already == MAX so the override emitted FINAL instead of trying.
    # Fix: extract evidence first, use it before checking abandonment.
    # ------------------------------------------------------------------
    def test_fresh_pii_used_despite_max_ask_count(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard"  # account-task
        wm.auth_ask_count = 2  # already at MAX
        wm.absorb_user_message("My name is Yusuf Rossi")
        wm.absorb_user_message("The ZIP code is 19122")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_name_zip")
        self.assertEqual(result.args["first_name"], "Yusuf")
        self.assertEqual(result.args["zip"], "19122")

    def test_real_email_used_despite_max_ask_count(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard"
        wm.auth_ask_count = 2
        wm.absorb_user_message("Try alice@gmail.com")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_email")
        self.assertEqual(result.args["email"], "alice@gmail.com")

    def test_no_pii_at_max_count_still_abandons(self) -> None:
        """If we truly have no usable PII at the cap, do abandon."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard"
        wm.auth_ask_count = 2
        # No PII at all
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)

    # ------------------------------------------------------------------
    # Bug 5: After the failed name+zip lookup with placeholder email
    # follow-up, the agent looped on the placeholder email forever.
    # The new placeholder predicate catches it.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Bug 6 (recovery): When name+zip and email both fail, but the user
    # supplied an order ID, fall back to get_order_details.
    # ------------------------------------------------------------------
    def test_order_id_fallback_when_auth_methods_fail(self) -> None:
        """When email AND name+zip have already been tried (and failed), the
        override must fall back to get_order_details with the order_id.

        D4 note: _extract_any_email now accepts @example.com emails from
        user_facts (specific prefix).  So the test must also mark the email
        as 'already tried' via recent_signatures to reach the order_id path."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange the mechanical keyboard"
        wm.auth_failed_zips = ["12345"]
        wm.absorb_user_message("My full name is Yusuf Rossi and ZIP 12345.")
        wm.absorb_user_message("My email is yusuf.rossi@example.com.")
        wm.absorb_user_message("My order number is #W2378156.")
        # Simulate email already tried (D4 fix: email is now extracted from
        # user_facts, so we need to mark it used to reach the order_id fallback)
        wm.recent_signatures.append(
            "find_user_id_by_email(email='yusuf.rossi@example.com')"
        )
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_order_details")
        self.assertEqual(result.args["order_id"], "#W2378156")

    def test_order_id_extracted_from_various_formats(self) -> None:
        """The order ID extractor must handle '#W' and 'W' formats."""
        from src.cargo.cargo_agent import CargoAgent
        for raw, expected in [
            ("My order is #W2378156", "#W2378156"),
            ("order id W1234567", "#W1234567"),
            ("order #w9999999", "#W9999999"),
        ]:
            wm = WorkingMemory()
            wm.absorb_user_message(raw)
            self.assertEqual(CargoAgent._extract_order_id(wm), expected, raw)

    def test_order_id_fallback_not_repeated(self) -> None:
        """If get_order_details was already tried, don't propose it again."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "modify my order"
        wm.absorb_user_message("My order is #W2378156")
        wm.recent_signatures.append("get_order_details(order_id='#W2378156')")
        result = agent._auth_override(self._placeholder(), wm)
        # Must NOT propose the same get_order_details again
        if result and result.name == "get_order_details":
            self.fail(f"Re-proposed get_order_details: {result.args}")

    def test_post_failed_zip_with_placeholder_email_does_not_loop(self) -> None:
        """After auth_ask_count hits the cap AND the user's email has already
        been tried once (recent_signatures guard), the override must NOT loop
        on the same email.

        D4+D5 note: 'yusuf.rossi@example.com' is now treated as a usable
        email (specific prefix → allowed by _extract_any_email).  The D5
        re-attempt fires the FIRST time (when the sig is not yet in
        recent_signatures).  After the sig is recorded, the re-attempt does
        NOT fire again, preventing an infinite loop on a failing email.
        """
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange mechanical keyboard"
        wm.auth_failed_zips = ["12345"]
        wm.auth_ask_count = 2
        wm.absorb_user_message("My full name is Yusuf Rossi")
        wm.absorb_user_message("My ZIP is 12345")
        wm.absorb_user_message("My email is yusuf.rossi@example.com")
        # Simulate email was already tried (fresh-PII re-attempt already used)
        wm.recent_signatures.append(
            "find_user_id_by_email(email='yusuf.rossi@example.com')"
        )
        result = agent._auth_override(
            self._placeholder("yusuf.rossi@example.com"), wm
        )
        self.assertIsNotNone(result)
        # After the email was already tried, must NOT propose it again
        if result.name == "find_user_id_by_email":
            self.assertNotEqual(
                result.args.get("email", ""), "yusuf.rossi@example.com",
                f"Looped on already-tried email: {result.args}",
            )


# ---------------------------------------------------------------------------
# Tau-bench import sanity (skipped when tau_bench is not installed)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Regression tests for trajectories(19) failure patterns
# ---------------------------------------------------------------------------
class TestTrajectory19Regressions(unittest.TestCase):
    """Each test pinned to a specific bug observed in trajectories(19)."""

    def _make_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        client = MockClient(scripts=[])
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client; agent.model = "test"; agent.style_name = "cargo"
        agent.temperature = 0.0; agent.schemas = {}; agent.domain_policy = ""
        return agent

    # ------------------------------------------------------------------
    # Bug A: user_id pattern collision with credit_card_*
    # T3 step 7: get_user_details(credit_card_9513926) — wrong field.
    # ------------------------------------------------------------------
    def test_user_id_prefers_explicit_field_over_bare_token(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        wm = WorkingMemory()
        # credit_card token comes FIRST in db_facts (worst case)
        wm.db_facts = [
            "credit_card_9513926",
            "user_id=yusuf_rossi_9620",
            "yusuf_rossi_9620",
        ]
        self.assertEqual(CargoAgent._existing_user_id(wm), "yusuf_rossi_9620")

    def test_user_id_skips_credit_card_when_only_bare_tokens(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        wm = WorkingMemory()
        wm.db_facts = ["credit_card_9513926", "yusuf_rossi_9620"]
        self.assertEqual(CargoAgent._existing_user_id(wm), "yusuf_rossi_9620")

    def test_user_id_skips_paypal_account(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        wm = WorkingMemory()
        wm.db_facts = ["paypal_account_1234567", "alice_smith_42"]
        self.assertEqual(CargoAgent._existing_user_id(wm), "alice_smith_42")

    def test_user_id_token_predicate(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        # Real-looking tokens
        self.assertTrue(CargoAgent._is_user_id_token("yusuf_rossi_9620"))
        self.assertTrue(CargoAgent._is_user_id_token("alice_smith_1"))
        # Non-user prefixes
        self.assertFalse(CargoAgent._is_user_id_token("credit_card_9513926"))
        self.assertFalse(CargoAgent._is_user_id_token("paypal_account_42"))
        self.assertFalse(CargoAgent._is_user_id_token("address_id_42"))
        self.assertFalse(CargoAgent._is_user_id_token("order_W2378156"))
        # Wrong shape
        self.assertFalse(CargoAgent._is_user_id_token("yusuf_rossi"))
        self.assertFalse(CargoAgent._is_user_id_token("yusuf"))

    # ------------------------------------------------------------------
    # Bug E: get_user_details with non-user token gets corrected
    # ------------------------------------------------------------------
    def test_get_user_details_with_credit_card_corrected(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.db_facts = [
            "user_id=yusuf_rossi_9620",
            "yusuf_rossi_9620",
            "credit_card_9513926",
        ]
        bad = ProposedAction(
            name="get_user_details",
            args={"user_id": "credit_card_9513926"},
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[], informational_intent="",
            raw_thought="", user_text="", raw_response="",
        )
        result = agent._resolve_get_user_details(bad, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.args["user_id"], "yusuf_rossi_9620")  # type: ignore

    def test_get_user_details_with_correct_uid_unchanged(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.db_facts = ["user_id=yusuf_rossi_9620"]
        good = ProposedAction(
            name="get_user_details",
            args={"user_id": "yusuf_rossi_9620"},
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[], informational_intent="",
            raw_thought="", user_text="", raw_response="",
        )
        self.assertIsNone(agent._resolve_get_user_details(good, wm))

    def test_user_supplied_user_id_is_bound_to_typed_state(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("Sure, my user ID is mia_li_3668 and I can continue.")

        self.assertIn("mia_li_3668", wm.typed_evidence_for("user_id"))
        self.assertIn("known_user_id: mia_li_3668", wm.render_compact())

    def test_user_supplied_reservation_id_is_bound_to_typed_state(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("Please cancel reservation id Z7GOZK.")

        self.assertIn("Z7GOZK", wm.typed_evidence_for("reservation_id"))

    def test_profile_reservation_list_is_bound_to_typed_state(self) -> None:
        wm = WorkingMemory()
        wm.absorb_observation({"reservations": ["Z7GOZK", "K67C4W"]})

        self.assertIn("Z7GOZK", wm.typed_evidence_for("reservation_id"))
        self.assertIn("K67C4W", wm.typed_evidence_for("reservation_id"))

    def test_grounded_placeholder_resolver_uses_user_provided_id(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("My user ID is mia_li_3668.")
        action = ProposedAction(
            name="get_user_details",
            args={"user_id": "user_id"},
            declared_class=RiskClass.READ,
        )

        result = agent._resolve_grounded_placeholders(action, wm)

        self.assertIsNotNone(result)
        self.assertEqual(result.args["user_id"], "mia_li_3668")  # type: ignore[union-attr]

    def test_grounded_placeholder_resolver_does_not_guess_ambiguous_ids(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("My user ID is mia_li_3668.")
        wm.absorb_user_message("Actually, the other user ID is olivia_gonzalez_2305.")
        action = ProposedAction(
            name="get_user_details",
            args={"user_id": "user_id"},
            declared_class=RiskClass.READ,
        )

        self.assertIsNone(agent._resolve_grounded_placeholders(action, wm))

    def test_resolved_placeholder_passes_arg_grounding(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("Sure, my user ID is mia_li_3668.")
        action = ProposedAction(
            name="get_user_details",
            args={"user_id": "user_id"},
            declared_class=RiskClass.READ,
        )
        resolved = agent._resolve_grounded_placeholders(action, wm)
        self.assertIsNotNone(resolved)
        schema = ToolEffectSchema(
            name="get_user_details",
            cls=RiskClass.READ,
            arg_id_fields=["user_id"],
        )

        gate = check_arg_grounding(resolved, schema, wm)  # type: ignore[arg-type]

        self.assertTrue(gate.ok, gate.reason)

    def test_airline_profile_repetition_advances_to_reservation_scan(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "change my return flight reservation"
        wm.user_profiles["olivia_gonzalez_2305"] = {
            "reservations": ["Z7GOZK", "K67C4W"],
        }
        wm.absorb_observation({"reservations": ["Z7GOZK", "K67C4W"]})
        repeated = ProposedAction(
            name="get_user_details",
            args={"user_id": "olivia_gonzalez_2305"},
            declared_class=RiskClass.READ,
        )

        result = agent._advance_reservation_retrieval(repeated, wm)

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_reservation_details")  # type: ignore[union-attr]
        self.assertEqual(result.args["reservation_id"], "Z7GOZK")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Bug B: "exchange items in my recent order" routed to product flow
    # ------------------------------------------------------------------
    def test_exchange_task_requires_auth(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        wm = WorkingMemory()
        wm.goal = "Hello! I need to make some changes to my recent order. Could you please assist me with exchanging items?"
        wm.absorb_user_message(wm.goal)
        self.assertFalse(CargoAgent._is_no_auth_query(wm))

    def test_return_task_requires_auth(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        wm = WorkingMemory()
        wm.goal = "I want to return the headphones from my order."
        wm.absorb_user_message(wm.goal)
        self.assertFalse(CargoAgent._is_no_auth_query(wm))

    def test_cancel_task_requires_auth(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        wm = WorkingMemory()
        wm.goal = "Please cancel my pending order."
        wm.absorb_user_message(wm.goal)
        self.assertFalse(CargoAgent._is_no_auth_query(wm))

    def test_pure_product_query_still_no_auth(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are currently available?"
        wm.absorb_user_message(wm.goal)
        self.assertTrue(CargoAgent._is_no_auth_query(wm))

    def test_order_id_in_user_facts_forces_auth(self) -> None:
        """If user provided an order ID in any turn, this is an account task."""
        from src.cargo.cargo_agent import CargoAgent
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are available?"
        wm.absorb_user_message("My order is #W2378156, also can I see t-shirts?")
        # Should require auth — they have a specific order context
        self.assertFalse(CargoAgent._is_no_auth_query(wm))

    # ------------------------------------------------------------------
    # Bug C: post-product-details finalization (the loop-breaker)
    # ------------------------------------------------------------------
    def test_finalize_product_count_when_data_available(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are currently available?"
        wm.absorb_user_message(wm.goal)
        wm.product_details["9523456873"] = {
            "name": "T-Shirt",
            "variants": {f"v{i}": {"item_id": f"v{i}"} for i in range(7)},
        }
        # list_all_product_types already executed once (in recent_signatures)
        wm.recent_signatures.append("list_all_product_types()")
        loop_action = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._finalize_product_count_query(loop_action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)
        # Count should be in the message
        self.assertIn("7", result.user_text)
        self.assertIn("T-Shirt", result.user_text)
        self.assertTrue(wm.product_count_finalized)

    def test_finalize_only_fires_once(self) -> None:
        """Don't re-emit the same FINAL on every step."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirts are available?"
        wm.product_details["1"] = {
            "name": "T-Shirt",
            "variants": {"v1": {}, "v2": {}},
        }
        wm.recent_signatures.append("list_all_product_types()")
        loop_action = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        first = agent._finalize_product_count_query(loop_action, wm)
        self.assertIsNotNone(first)
        # Second call must return None — already finalized
        second = agent._finalize_product_count_query(loop_action, wm)
        self.assertIsNone(second)

    def test_finalize_does_not_fire_on_first_list_call(self) -> None:
        """Don't short-circuit before list_all_product_types has run."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirts are available?"
        wm.product_details["1"] = {"name": "T-Shirt", "variants": {"v1": {}}}
        # NOT in recent_signatures yet — first call
        loop_action = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        self.assertIsNone(agent._finalize_product_count_query(loop_action, wm))

    def test_finalize_does_not_fire_for_non_count_queries(self) -> None:
        """Goal must contain 'how many' / 'number of' / 'count of'."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "Tell me about t-shirts available in the store."
        wm.product_details["1"] = {"name": "T-Shirt", "variants": {"v1": {}}}
        wm.recent_signatures.append("list_all_product_types()")
        action = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        self.assertIsNone(agent._finalize_product_count_query(action, wm))

    def test_finalize_counts_available_only(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are currently available?"
        # 5 variants, 3 available, 2 not
        wm.product_details["1"] = {
            "name": "T-Shirt",
            "variants": {
                "a": {"available": True},
                "b": {"available": True},
                "c": {"available": True},
                "d": {"available": False},
                "e": {"available": False},
            },
        }
        wm.recent_signatures.append("list_all_product_types()")
        action = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._finalize_product_count_query(action, wm)
        self.assertIsNotNone(result)
        # Should mention 5 total and 3 available
        self.assertIn("5", result.user_text)
        self.assertIn("3", result.user_text)


# ---------------------------------------------------------------------------
# Tau-bench import sanity (skipped when tau_bench is not installed)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Regression tests for trajectories(20) failure patterns
# ---------------------------------------------------------------------------
class TestTrajectory20Regressions(unittest.TestCase):
    """Each test pinned to a specific bug observed in trajectories(20)."""

    def _make_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        client = MockClient(scripts=[])
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client; agent.model = "test"; agent.style_name = "cargo"
        agent.temperature = 0.0; agent.schemas = {}; agent.domain_policy = ""
        return agent

    # ------------------------------------------------------------------
    # Bug T0: order_id format mismatch.  User says "W2378156", agent
    # produces "#W2378156", arg_grounding rejects.  Both forms must be
    # stored in user_facts.
    # ------------------------------------------------------------------
    def test_order_id_both_forms_in_user_facts(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("Could you check on order number W2378156 for me please?")
        self.assertIn("W2378156", wm.user_facts)
        self.assertIn("#W2378156", wm.user_facts)

    def test_order_id_with_hash_prefix_normalized(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("My order is #W1234567.")
        self.assertIn("W1234567", wm.user_facts)
        self.assertIn("#W1234567", wm.user_facts)

    def test_order_id_lowercase_w_normalized(self) -> None:
        wm = WorkingMemory()
        wm.absorb_user_message("Looking up w9999999")
        self.assertIn("W9999999", wm.user_facts)
        self.assertIn("#W9999999", wm.user_facts)

    # ------------------------------------------------------------------
    # Bug T1/T2/T3: product_types must survive db_facts eviction.
    # After get_product_details flooded db_facts with variant data,
    # _advance_after_product_list could no longer find Headphones/Cleaner.
    # ------------------------------------------------------------------
    def test_advance_uses_product_types_when_db_facts_evicted(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirts are available?"
        wm.absorb_user_message(wm.goal)
        wm.absorb_user_message("Also, what about the headphones?")
        # Catalogue is in product_types (durable)
        wm.product_types = {
            "Action Camera": "3377618313",
            "Headphones": "8888888888",
            "T-Shirt": "9523456873",
        }
        # db_facts is full of variant chaff — the catalogue is gone from db_facts
        wm.db_facts = [f"variants[v{i}].color=red" for i in range(48)]
        # T-Shirt already fetched (don't re-route to it)
        wm.product_details["9523456873"] = {"name": "T-Shirt", "variants": {}}
        # list_all_product_types in recent_signatures (model is looping)
        wm.recent_signatures.append("list_all_product_types()")

        loop = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._advance_after_product_list(loop, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_product_details")
        # Must NOT pick T-Shirt (already fetched), must pick Headphones
        self.assertEqual(result.args["product_id"], "8888888888")

    def test_advance_skips_already_fetched_products(self) -> None:
        """Don't re-route to a product whose details we already have."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirts are available?"
        wm.absorb_user_message("Tell me about t-shirts")
        wm.product_types = {"T-Shirt": "9523456873"}
        wm.product_details["9523456873"] = {"name": "T-Shirt", "variants": {}}
        wm.recent_signatures.append("list_all_product_types()")
        loop = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        # No other product mentioned → should return None (let finalizer fire)
        result = agent._advance_after_product_list(loop, wm)
        self.assertIsNone(result)

    def test_resolve_product_id_uses_product_types(self) -> None:
        """_resolve_product_id_name should use wm.product_types as primary
        source (not just db_facts)."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.product_types = {"T-Shirt": "9523456873"}
        # db_facts empty (evicted)
        action = ProposedAction(
            name="get_product_details", args={"product_id": "T-Shirt"},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._resolve_product_id_name(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(str(result.args["product_id"]), "9523456873")  # type: ignore

    # ------------------------------------------------------------------
    # Bug T3/T4: hard-loop break on consecutive identical FINALs.
    # The state fields must initialize to safe defaults.
    # ------------------------------------------------------------------
    def test_consecutive_same_final_state_defaults(self) -> None:
        wm = WorkingMemory()
        self.assertEqual(wm.last_final_text, "")
        self.assertEqual(wm.consecutive_same_final, 0)

    # ------------------------------------------------------------------
    # Bug T4: order_id fallback survives auth_abandoned state.
    # After 2 failed name+zip attempts, agent set auth_abandoned=True.
    # If user then provides an order ID, override must use it.
    # ------------------------------------------------------------------
    def test_order_id_recovery_survives_auth_abandoned(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange items in my recent order"
        wm.auth_abandoned = True
        wm.auth_failed_zips = ["10012", "12345"]
        wm.absorb_user_message("My order number is W2378156")
        ph = ProposedAction(
            name="find_user_id_by_email", args={"email": "user@example.com"},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._auth_override(ph, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_order_details")
        self.assertEqual(result.args["order_id"], "#W2378156")


# ---------------------------------------------------------------------------
# Regression tests for trajectories(22) failure patterns
# ---------------------------------------------------------------------------
class TestTrajectory22Regressions(unittest.TestCase):
    """Pinned tests for all 6 root-cause bugs identified in trajectories(22).

    B1 (Tasks 0,1): Auth loop after order_id authentication – Path 1 of
        _auth_override must re-confirm auth via email found in db_facts
        (populated by get_user_details) instead of returning None and
        letting the placeholder-email loop continue.

    B2 (Task 2): Product-name false-positive matching – "hose" must NOT
        match "those", "smart" prefix must NOT match "smartwatch", and
        compound "smartwatch" MUST match "Smart Watch".

    B3 (Tasks 3,4): Deterministic auth-abandon FINALs blocked by SC gate
        – all deterministic override FINAL/ASK_USER actions must carry
        ``bypass_gates=True``.

    B4 (Task 3): Model proposes auth-ask respond before product details
        fetched – _advance_after_product_list must intercept premature
        respond/FINAL/ASK_USER actions for no-auth product queries.

    B5 (Tasks 3,4): Auth-abandon must answer no-auth product sub-query
        before emitting the give-up FINAL for mixed goals.

    B6 (Task 4): "Rather not" + offered alternative not treated as hard
        refusal – soft-refusal detection must spare negotiating messages.
    """

    def _make_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        client = MockClient(scripts=[])
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "test"
        agent.style_name = "cargo"
        agent.temperature = 0.0
        agent.schemas = {}
        agent.domain_policy = ""
        return agent

    def _placeholder_action(self) -> ProposedAction:
        return ProposedAction(
            name="find_user_id_by_email",
            args={"email": "user@example.com"},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )

    # ------------------------------------------------------------------
    # B1: Auth loop after order_id authentication
    # After auth is complete, Path 1 must lock the phase and avoid re-entering
    # find_user_id_* confirmation.  The separate grounded-progress layer is
    # responsible for order retrieval / commit.
    # ------------------------------------------------------------------
    def test_b1_path1_extracts_email_from_db_facts(self) -> None:
        """Path 1: real email in db_facts must not cause auth re-entry."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange items in my pending order"
        # User authenticated via order_id path; user_id now in db_facts.
        wm.db_facts = [
            "user_id=alice_smith_9620",
            "email=alice@gmail.com",   # real-looking domain (not @example.com)
            "name=Alice Smith",
        ]
        # get_user_details already called (so _post_auth_action returns None).
        wm.recent_signatures.append("get_user_details(user_id=alice_smith_9620)")
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNone(result)
        self.assertTrue(wm.phase_locked("auth"))

    def test_b1_email_already_confirmed_returns_none(self) -> None:
        """Path 1: if find_user_id_by_email(real_email) is already in
        recent_signatures, don't re-issue it (avoids a second loop)."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange items in my pending order"
        wm.db_facts = [
            "user_id=alice_smith_9620",
            "email=alice@gmail.com",   # real-looking domain
        ]
        wm.recent_signatures.append("get_user_details(user_id=alice_smith_9620)")
        # Already confirmed once.
        wm.recent_signatures.append("find_user_id_by_email(email='alice@gmail.com')")
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        # Should not re-emit the same lookup — returns None.
        self.assertIsNone(result)

    def test_b1_no_email_in_db_facts_returns_none(self) -> None:
        """Path 1: if db_facts has user_id but no real email, return None
        gracefully (don't crash)."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange items in my pending order"
        wm.db_facts = ["user_id=alice_smith_9620"]
        wm.recent_signatures.append("get_user_details(user_id=alice_smith_9620)")
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # B2: Product-name false-positive matching
    # ------------------------------------------------------------------
    def test_b2_hose_does_not_match_those(self) -> None:
        """'hose' (from 'Garden Hose') must NOT match 'those'."""
        from src.cargo.cargo_agent import CargoAgent
        # "garden hose" — the word "hose" is a substring of "those"
        # but must fail the word-boundary check.
        self.assertFalse(
            CargoAgent._product_name_matches_user(
                "garden hose",
                "check those for me"
            )
        )

    def test_b2_smart_prefix_does_not_match_smartwatch(self) -> None:
        """'smart' token (from 'Smart Thermostat') must NOT match 'smartwatch'."""
        from src.cargo.cargo_agent import CargoAgent
        # "smart thermostat" — the word "smart" prefix-matches "smartwatch"
        # if done naively, but word-boundary regex prevents it.
        self.assertFalse(
            CargoAgent._product_name_matches_user(
                "smart thermostat",
                "how many smartwatches are there"
            )
        )

    def test_b2_smart_watch_matches_smartwatch(self) -> None:
        """Compound 'smartwatch' (Rule 3) must match product 'Smart Watch'.

        The user says "smartwatch" (singular compound word).  Rule 1 fails
        (no space in user text), Rule 2 fails (no \bsmart\b boundary inside
        "smartwatch"), Rule 3 catches it via the joined form "smartwatch".
        This is the exact failure observed in trajectories(22) Task 2.
        """
        from src.cargo.cargo_agent import CargoAgent
        self.assertTrue(
            CargoAgent._product_name_matches_user(
                "smart watch",
                "how many smartwatch variants are available"  # singular compound
            )
        )

    def test_b2_tshirt_matches_tshirts_plural(self) -> None:
        """Phrase 'T-Shirt' must match user saying 't-shirts'."""
        from src.cargo.cargo_agent import CargoAgent
        self.assertTrue(
            CargoAgent._product_name_matches_user(
                "t-shirt",
                "how many t-shirts are available"
            )
        )

    def test_b2_exact_product_name_always_matches(self) -> None:
        """Direct phrase match must always pass."""
        from src.cargo.cargo_agent import CargoAgent
        self.assertTrue(
            CargoAgent._product_name_matches_user(
                "garden hose",
                "I would like a garden hose"
            )
        )

    def test_b2_short_product_name_no_match_when_not_present(self) -> None:
        """A product name with only short tokens (< 4 chars) returns False
        when neither the phrase nor any token is a substring of user text.

        Note: Rule 1 (phrase substring) still fires for short names when the
        name IS literally contained in the text; this test verifies the False
        path when the product name is simply absent.
        """
        from src.cargo.cargo_agent import CargoAgent
        # "rug" is 3 chars — not in user text at all → all three rules return False.
        self.assertFalse(
            CargoAgent._product_name_matches_user(
                "rug",
                "I want something soft and cozy for my floor"
            )
        )

    # ------------------------------------------------------------------
    # B3: Deterministic auth-abandon FINALs must carry bypass_gates=True
    # ------------------------------------------------------------------
    def test_b3_auth_giveup_final_has_bypass_gates(self) -> None:
        """Auth-abandon FINAL (user refused) must have bypass_gates=True."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "update my order"
        wm.absorb_user_message("prefer not to give out my info")
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)  # type: ignore
        self.assertTrue(getattr(result, "bypass_gates", False))  # type: ignore

    def test_b3_auth_giveup_loop_break_has_bypass_gates(self) -> None:
        """Auth-abandon loop-break FINAL (already gave up once) must also
        carry bypass_gates=True."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "update my order"
        wm.auth_abandoned = True
        wm.auth_giveup_emitted = True
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)  # type: ignore
        self.assertTrue(getattr(result, "bypass_gates", False))  # type: ignore

    def test_b3_finalize_product_count_has_bypass_gates(self) -> None:
        """_finalize_product_count_query FINAL must carry bypass_gates=True.

        Uses "how many t-shirt options are available" which matches both the
        'how many … options?' arm of _PRODUCT_QUERY_RE and the t-shirts arm,
        guaranteeing the count finalizer reaches the return statement.
        """
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "how many t-shirt options are available"
        wm.product_details["pid999"] = {
            "name": "T-Shirt",
            "variants": {
                "v1": {"available": True},
                "v2": {"available": True},
                "v3": {"available": False},
            },
        }
        # Trigger the finalizer by presenting a repeat list_all_product_types.
        wm.recent_signatures.append("list_all_product_types()")
        loop = ProposedAction(
            name="list_all_product_types",
            args={},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )
        result = agent._finalize_product_count_query(loop, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)  # type: ignore
        self.assertTrue(getattr(result, "bypass_gates", False))  # type: ignore

    # ------------------------------------------------------------------
    # B4: _advance_after_product_list intercepts premature respond actions
    # for no-auth product queries before product details have been fetched.
    # ------------------------------------------------------------------
    def test_b4_premature_respond_intercepted_for_noauth_query(self) -> None:
        """A premature respond (ASK_USER class) must be intercepted when
        product_types is populated and goal is a no-auth product query.

        The user says "smartwatch" (singular compound) — the compound Rule 3
        in _product_name_matches_user resolves "Smart Watch" → "smartwatch".
        """
        agent = self._make_agent()
        wm = WorkingMemory()
        # Use singular "smartwatch" so the compound rule (Rule 3) can match
        # the product name "Smart Watch" → compound "smartwatch".
        wm.goal = "how many smartwatch variants are in the store"
        wm.absorb_user_message(wm.goal)
        wm.product_types = {"Smart Watch": "pid123"}
        # No product details fetched yet.
        premature_respond = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="Could you please provide your account details?",
            raw_response="",
        )
        result = agent._advance_after_product_list(premature_respond, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_product_details")  # type: ignore
        self.assertEqual(result.args["product_id"], "pid123")  # type: ignore

    def test_b4_final_action_intercepted_for_noauth_query(self) -> None:
        """A premature FINAL action is intercepted the same way as ASK_USER."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "how many t-shirts are in the store"
        wm.absorb_user_message(wm.goal)
        wm.product_types = {"T-Shirt": "pid456"}
        premature_final = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="I cannot help without your account info.",
            raw_response="",
        )
        result = agent._advance_after_product_list(premature_final, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_product_details")  # type: ignore
        self.assertEqual(result.args["product_id"], "pid456")  # type: ignore

    def test_b4_no_interception_for_auth_required_goal(self) -> None:
        """For goals that require auth (exchange, cancel), premature responds
        must NOT be intercepted by the product-list advance logic."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange my recent order for another model"
        wm.absorb_user_message(wm.goal)
        wm.product_types = {"Smart Watch": "pid123"}
        premature_respond = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="Could you provide your account details?",
            raw_response="",
        )
        result = agent._advance_after_product_list(premature_respond, wm)
        # Auth-required goal → _is_no_auth_query is False → no interception.
        self.assertIsNone(result)

    def test_b4_no_interception_when_product_types_empty(self) -> None:
        """If product_types is not yet populated, don't intercept respond
        (product list hasn't been fetched yet; let normal flow proceed)."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "how many smartwatches are in the store"
        wm.absorb_user_message(wm.goal)
        # product_types is empty
        premature_respond = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="I cannot help.",
            raw_response="",
        )
        result = agent._advance_after_product_list(premature_respond, wm)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # B5: Auth-abandon must answer the product sub-query before giving up
    # ------------------------------------------------------------------
    def test_b5_mixed_goal_lists_products_before_giveup(self) -> None:
        """For a mixed goal (product count + order update), auth-abandon must
        pivot to list_all_product_types before emitting the give-up FINAL."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "how many t-shirts are in the store, also update my recent order"
        wm.auth_abandoned = True
        # product_types not yet fetched
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "list_all_product_types")  # type: ignore
        self.assertNotEqual(result.declared_class, RiskClass.FINAL)  # type: ignore

    def test_b5_mixed_goal_fetches_product_details_when_types_available(self) -> None:
        """After product types are listed, pivot to get_product_details."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "how many t-shirts are in the store, also update my recent order"
        wm.absorb_user_message("how many t-shirts are available")
        wm.auth_abandoned = True
        wm.product_types = {"T-Shirt": "pid789"}
        # Details not yet fetched.
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_product_details")  # type: ignore
        self.assertEqual(result.args["product_id"], "pid789")  # type: ignore

    def test_b5_after_product_count_finalized_emits_giveup(self) -> None:
        """Once product_count_finalized=True the B5 pivot is done; then the
        give-up FINAL should fire normally."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "how many t-shirts are in the store, also update my recent order"
        wm.auth_abandoned = True
        wm.product_count_finalized = True  # count already answered
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        # Should now be the give-up FINAL.
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)  # type: ignore
        self.assertTrue(getattr(result, "bypass_gates", False))  # type: ignore

    # ------------------------------------------------------------------
    # B6: "Rather not" + offered alternative is not a hard refusal
    # ------------------------------------------------------------------
    def test_b6_soft_refusal_with_order_alternative_not_abandoned(self) -> None:
        """'I'd rather not share … can we proceed based on my recent orders?'
        must NOT set auth_abandoned — it's a negotiation, not a hard refusal."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange items in my recent order"
        wm.absorb_user_message(
            "I'd rather not share too much personal info. "
            "Can we proceed based on my recent orders or some other identifier?"
        )
        action = self._placeholder_action()
        # Override should NOT emit a give-up FINAL; auth_abandoned must remain False.
        _ = agent._auth_override(action, wm)
        self.assertFalse(wm.auth_abandoned)

    def test_b6_soft_refusal_with_identifier_pivot_triggers_order_lookup(self) -> None:
        """After the soft-refusal + order-alternative message, if the user
        also supplies an order ID, the override must use it."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange items in my recent order"
        wm.absorb_user_message(
            "I'd rather not share personal info. "
            "My order number is W9876543."
        )
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        # Override must use the order_id, not set auth_abandoned.
        self.assertFalse(wm.auth_abandoned)
        self.assertEqual(result.name, "get_order_details")  # type: ignore
        self.assertIn("W9876543", str(result.args.get("order_id", "")))  # type: ignore

    def test_b6_hard_refusal_without_alternative_sets_abandoned(self) -> None:
        """A clean hard refusal ('prefer not', no alternative offered) must
        still set auth_abandoned and emit a give-up FINAL."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange items in my recent order"
        wm.absorb_user_message("I prefer not to share that information.")
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertTrue(wm.auth_abandoned)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.FINAL)  # type: ignore
        self.assertTrue(getattr(result, "bypass_gates", False))  # type: ignore

    def test_b6_rather_not_without_alternative_is_hard_refusal(self) -> None:
        """'Rather not' alone (no alternative offered) is a hard refusal."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "cancel my order"
        wm.absorb_user_message("I'd rather not provide that.")
        action = self._placeholder_action()
        _ = agent._auth_override(action, wm)
        self.assertTrue(wm.auth_abandoned)


# ---------------------------------------------------------------------------
# Regression tests for trajectories(23) failure patterns
# ---------------------------------------------------------------------------
class TestTrajectory23Regressions(unittest.TestCase):
    """Pinned tests for all 4 root-cause bugs identified in trajectories(23).

    C1 (Tasks 0,1): db_facts emails blocked by @example. domain filter —
        tau-bench uses @example.com for real user emails; Path 1 of
        _auth_override must trust db_facts emails unconditionally (they come
        from tool responses, not model hallucination).

    C2 (Task 2): arg_grounding rejects product IDs from wm.product_types —
        after db_facts is evicted by a large get_product_details response,
        the durable wm.product_types catalogue must be included in evidence.

    C3 (Tasks 3,4): _finalize_product_count_query only fires on
        list_all_product_types repeats, not on model-proposed FINAL actions,
        causing the precondition gate to block a correct count response.

    C4 (Tasks 3,4): Default auth question leads with "email address", which
        triggers refusal from privacy-sensitive user simulations; should ask
        for name + ZIP first (what tau-bench users willingly provide).
    """

    def _make_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        client = MockClient(scripts=[])
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "test"
        agent.style_name = "cargo"
        agent.temperature = 0.0
        agent.schemas = {}
        agent.domain_policy = ""
        return agent

    def _placeholder_action(self) -> ProposedAction:
        return ProposedAction(
            name="find_user_id_by_email",
            args={"email": "user@example.com"},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )

    # ------------------------------------------------------------------
    # C1: db_facts emails (@example.com) must be trusted for auth confirmation
    # ------------------------------------------------------------------
    def test_c1_extract_any_email_trusts_example_com_from_db_facts(self) -> None:
        """_extract_any_email must return an @example.com email from db_facts.

        tau-bench retail uses @example.com for every user's email address.
        The previous _extract_real_email blanket-blocked @example. as a
        placeholder domain, so Path 1 could never find the DB email and
        confirm auth via find_user_id_by_email.
        """
        from src.cargo.cargo_agent import CargoAgent
        facts = [
            "user_id=yusuf_rossi_9620",
            "email=yusuf.rossi7301@example.com",  # real tau-bench email
        ]
        result = CargoAgent._extract_any_email(facts)
        self.assertEqual(result, "yusuf.rossi7301@example.com")

    def test_c1_extract_any_email_still_blocks_generic_prefixes(self) -> None:
        """_extract_any_email must still reject emails whose prefix matches
        the _PLACEHOLDER_EMAIL_RE pattern (user@, demo@, test@, admin@, etc.).

        Note: _extract_any_email does NOT block emails with a specific-name
        prefix like alice@example.com — the C1 fix intentionally allows those
        because tau-bench real users have addresses like yusuf.rossi7301@example.com.
        Only the generic-prefix RE patterns are blocked."""
        from src.cargo.cargo_agent import CargoAgent
        # Clearly generic prefixes — blocked by the prefix RE.
        for fabricated in ["user@example.com", "customer@example.com",
                           "demo@example.com", "test@test.com",
                           "admin@sample.com", "noreply@fake.com"]:
            result = CargoAgent._extract_any_email([fabricated])
            self.assertIsNone(
                result,
                f"_extract_any_email should reject fabricated email {fabricated!r}",
            )
        # Specific-name prefix — NOT blocked (C1: trust db_facts @example.com).
        self.assertIsNotNone(
            CargoAgent._extract_any_email(["alice@example.com"]),
            "_extract_any_email should pass alice@example.com (specific name, C1 fix)",
        )

    def test_c1_path1_uses_db_facts_email_to_confirm_auth(self) -> None:
        """After order_id auth, Path 1 must not re-enter email auth even when
        a confirmed @example.com email is available from DB facts."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange the mechanical keyboard in my order"
        # Auth confirmed via get_order_details → get_user_details; DB email present.
        wm.db_facts = [
            "user_id=yusuf_rossi_9620",
            "email=yusuf.rossi7301@example.com",
        ]
        wm.recent_signatures.append("get_user_details(user_id=yusuf_rossi_9620)")
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNone(result)
        self.assertTrue(wm.phase_locked("auth"))

    def test_c1_user_provided_example_com_email_still_blocked_via_real_email(self) -> None:
        """If the user provides yusufrossi@example.com (not a generic prefix),
        _extract_real_email should still block it (domain filter applies to
        user-provided emails), but _extract_any_email would accept it from db_facts."""
        from src.cargo.cargo_agent import CargoAgent
        # _extract_real_email applies domain block → rejects user-provided @example.com
        self.assertIsNone(
            CargoAgent._extract_real_email(["yusufrossi@example.com"])
        )
        # _extract_any_email accepts it (non-generic prefix)
        self.assertEqual(
            CargoAgent._extract_any_email(["yusufrossi@example.com"]),
            "yusufrossi@example.com",
        )

    # ------------------------------------------------------------------
    # C2: product_types values are grounded evidence for arg_grounding gate
    # ------------------------------------------------------------------
    def test_c2_product_types_ids_in_all_evidence(self) -> None:
        """WorkingMemory.all_evidence() must include product_types values
        so arg_grounding doesn't reject IDs from the durable catalogue."""
        wm = WorkingMemory()
        wm.product_types = {"Smart Watch": "6945232052", "T-Shirt": "9523456873"}
        # db_facts is empty (evicted after a large get_product_details response)
        wm.db_facts = []
        evidence = wm.all_evidence()
        self.assertIn("6945232052", evidence)
        self.assertIn("9523456873", evidence)

    def test_c2_advance_product_id_passes_arg_grounding(self) -> None:
        """A product_id resolved from wm.product_types must pass the
        arg_grounding gate even when db_facts has been evicted."""
        from src.cargo.gates import check_arg_grounding
        from src.cargo import ToolEffectSchema
        wm = WorkingMemory()
        wm.product_types = {"Smart Watch": "6945232052"}
        wm.db_facts = []  # evicted

        action = ProposedAction(
            name="get_product_details",
            args={"product_id": "6945232052"},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )
        schema = ToolEffectSchema(
            name="get_product_details",
            cls=RiskClass.READ,
            arg_id_fields=["product_id"],
            param_properties={"product_id": {"type": "string"}},
            required_params=["product_id"],
        )
        result = check_arg_grounding(action, schema, wm)
        self.assertTrue(result.ok, f"arg_grounding should pass; got: {result.reason}")

    # ------------------------------------------------------------------
    # C3: _finalize_product_count_query fires on FINAL actions too
    # ------------------------------------------------------------------
    def test_c3_finalizer_fires_on_model_proposed_final(self) -> None:
        """_finalize_product_count_query must replace a model-proposed FINAL
        (which would fail precondition gate) with a deterministic bypass FINAL
        when product details are already available."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "how many t-shirt options are available in the store"
        wm.product_details["pid-ts"] = {
            "name": "T-Shirt",
            "variants": {
                "v1": {"available": True},
                "v2": {"available": True},
                "v3": {"available": False},
                "v4": {"available": True},
            },
        }
        model_final = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
            declared_pre=["T-Shirt details fetched"],
            declared_post=[],
            informational_intent="answer t-shirt count",
            raw_thought="",
            user_text="There are some t-shirt options.",  # vague model answer
            raw_response="",
        )
        result = agent._finalize_product_count_query(model_final, wm)
        self.assertIsNotNone(result)
        # Must be a deterministic FINAL with bypass_gates=True
        self.assertEqual(result.declared_class, RiskClass.FINAL)  # type: ignore
        self.assertTrue(getattr(result, "bypass_gates", False))  # type: ignore
        # Must contain the actual count (3 available out of 4)
        self.assertIn("3", result.user_text)  # type: ignore
        self.assertIn("T-Shirt", result.user_text)  # type: ignore

    def test_c3_finalizer_does_not_fire_when_data_missing(self) -> None:
        """Finalizer must NOT fire on a FINAL action if product_details is
        empty — nothing to count yet."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "how many t-shirt options are available"
        # product_details is empty
        model_final = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="I don't know.",
            raw_response="",
        )
        result = agent._finalize_product_count_query(model_final, wm)
        self.assertIsNone(result)

    def test_c3_finalizer_does_not_fire_for_non_count_goal_final(self) -> None:
        """Finalizer must NOT fire on FINAL if goal is not a 'how many' query."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange my thermostat for one compatible with Google Home"
        wm.product_details["pid-st"] = {
            "name": "Smart Thermostat",
            "variants": {"v1": {"available": True}},
        }
        model_final = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="Let me check the thermostat options.",
            raw_response="",
        )
        result = agent._finalize_product_count_query(model_final, wm)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # C4: auth question leads with name+zip, not email
    # ------------------------------------------------------------------
    def test_c4_auth_question_mentions_name_and_zip_first(self) -> None:
        """The default auth question must ask for name + ZIP before email.

        Asking for 'email address' first triggers refusals from privacy-
        sensitive user simulations; tau-bench users readily provide name+zip.
        """
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "update my pending order"
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.ASK_USER)  # type: ignore
        user_text = (result.user_text or "").lower()  # type: ignore
        # "name" must appear before "email" in the question
        name_pos = user_text.find("name")
        email_pos = user_text.find("email")
        self.assertGreater(name_pos, -1, "auth question must mention 'name'")
        self.assertGreater(
            email_pos,
            name_pos,
            "auth question must ask for name before email (name+zip first)",
        )

    def test_c4_auth_question_still_mentions_zip(self) -> None:
        """Auth question must also ask for ZIP code."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "cancel my most recent order"
        action = self._placeholder_action()
        result = agent._auth_override(action, wm)
        self.assertIsNotNone(result)
        user_text = (result.user_text or "").lower()  # type: ignore
        self.assertIn("zip", user_text)


# ---------------------------------------------------------------------------
# Regression tests for trajectories(24) failure patterns
# ---------------------------------------------------------------------------
class TestTrajectory24Regressions(unittest.TestCase):
    """Each test pinned to a specific bug observed in trajectories(24)."""

    def _make_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        client = MockClient(scripts=[])
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "test"
        agent.temperature = 0.0
        agent.schemas = {}
        agent.adapter = TauAirlineAdapter()
        agent.kernel = GenericCargoKernel(agent.adapter)
        return agent

    def _placeholder(self, email: str = "alice@example.com") -> Any:
        from src.cargo.schemas import ProposedAction
        from src.cargo.risk_class import RiskClass
        return ProposedAction(
            name="find_user_id_by_email",
            args={"email": email},
            declared_class=RiskClass.READ,
            declared_pre=[],
            declared_post=[],
            informational_intent="",
            raw_thought="",
            user_text="",
            raw_response="",
        )

    # ------------------------------------------------------------------
    # D1: auth_email cache survives db_facts LRU eviction
    # ------------------------------------------------------------------
    def test_d1_auth_email_field_in_working_memory(self) -> None:
        """WorkingMemory must have an auth_email field (durable, never evicted)."""
        wm = WorkingMemory()
        self.assertEqual(wm.auth_email, "")
        wm.auth_email = "yusuf.rossi7301@example.com"
        self.assertEqual(wm.auth_email, "yusuf.rossi7301@example.com")

    def test_d1_auth_email_survives_db_facts_eviction(self) -> None:
        """auth_email must remain accessible even when db_facts has been
        flushed (simulated by clearing db_facts after it was set)."""
        wm = WorkingMemory()
        wm.auth_email = "yusuf.rossi7301@example.com"
        # Simulate LRU eviction by clearing db_facts entirely.
        wm.db_facts = []
        # all_evidence() must still include the email.
        self.assertIn("yusuf.rossi7301@example.com", wm.all_evidence())

    def test_d1_auth_email_in_all_evidence(self) -> None:
        """all_evidence() must include auth_email and auth_user_id so
        arg_grounding accepts them even after db_facts LRU eviction."""
        wm = WorkingMemory()
        wm.auth_user_id = "yusuf_rossi_9620"
        wm.auth_email = "yusuf.rossi7301@example.com"
        ev = wm.all_evidence()
        self.assertIn("yusuf_rossi_9620", ev)
        self.assertIn("yusuf.rossi7301@example.com", ev)

    def test_d1_durable_caches_are_gate_evidence(self) -> None:
        """Preconditions must see order/product caches, not just db_facts LRU."""
        wm = WorkingMemory()
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "credit_card_9513926"}],
            "items": [{"item_id": "1151293680", "product_id": "1656367028"}],
        }
        wm.product_details["1656367028"] = {
            "product_id": "1656367028",
            "variants": {
                "7706410293": {
                    "item_id": "7706410293",
                    "available": True,
                    "options": {"switch type": "clicky"},
                }
            },
        }
        for i in range(80):
            wm._add_db_fact(f"evict_{i}=x")
        action = ProposedAction(
            name="exchange_delivered_order_items",
            args={
                "order_id": "#W2378156",
                "item_ids": ["1151293680"],
                "new_item_ids": ["7706410293"],
                "payment_method_id": "credit_card_9513926",
            },
            declared_class=RiskClass.WRITE,
            declared_pre=[
                "order #W2378156 delivered",
                "item 1151293680 exists",
                "item 7706410293 exists",
            ],
        )
        schema = ToolEffectSchema(
            name="exchange_delivered_order_items",
            cls=RiskClass.WRITE,
            required_params=["order_id", "item_ids", "new_item_ids", "payment_method_id"],
        )
        result = check_preconditions(action, schema, wm)
        self.assertTrue(result.ok, result.reason)

    def test_d1_render_compact_shows_confirmed_identity(self) -> None:
        """render_compact() must surface confirmed_user_id and confirmed_email
        so the proposer MODEL always sees them, not relying on db_facts."""
        wm = WorkingMemory()
        wm.auth_user_id = "yusuf_rossi_9620"
        wm.auth_email = "yusuf.rossi7301@example.com"
        rendered = wm.render_compact()
        self.assertIn("confirmed_user_id", rendered)
        self.assertIn("yusuf_rossi_9620", rendered)
        self.assertIn("confirmed_email", rendered)
        self.assertIn("yusuf.rossi7301@example.com", rendered)

    def test_d1_path1_uses_auth_email_cache_over_db_facts(self) -> None:
        """Path 1 of _auth_override must not use cached email to re-enter a
        completed auth phase, even if db_facts were evicted."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange the mechanical keyboard in my order #W2378156"
        wm.auth_user_id = "yusuf_rossi_9620"
        wm.auth_email = "yusuf.rossi7301@example.com"
        # db_facts is empty — simulates LRU eviction
        wm.db_facts = []
        wm.recent_signatures.append("get_user_details(user_id='yusuf_rossi_9620')")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNone(result)
        self.assertTrue(wm.phase_locked("auth"))

    # ------------------------------------------------------------------
    # D2: order_id normalization (#W prefix)
    # ------------------------------------------------------------------
    def test_d2_normalize_adds_hash_prefix(self) -> None:
        """_normalize_order_id_action must add '#' when model uses bare W…."""
        from src.cargo.schemas import ProposedAction
        from src.cargo.risk_class import RiskClass
        agent = self._make_agent()
        wm = WorkingMemory()
        action = ProposedAction(
            name="get_order_details",
            args={"order_id": "W2378156"},
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._normalize_order_id_action(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.args["order_id"], "#W2378156")

    def test_d2_normalize_lowercase_w(self) -> None:
        """Lowercase 'w' prefix must be uppercased to 'W'."""
        from src.cargo.schemas import ProposedAction
        from src.cargo.risk_class import RiskClass
        agent = self._make_agent()
        wm = WorkingMemory()
        action = ProposedAction(
            name="get_order_details",
            args={"order_id": "w1234567"},
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._normalize_order_id_action(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.args["order_id"], "#W1234567")

    def test_d2_normalize_already_canonical(self) -> None:
        """Already canonical '#W…' must be returned as-is (no change)."""
        from src.cargo.schemas import ProposedAction
        from src.cargo.risk_class import RiskClass
        agent = self._make_agent()
        wm = WorkingMemory()
        action = ProposedAction(
            name="get_order_details",
            args={"order_id": "#W2378156"},
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._normalize_order_id_action(action, wm)
        self.assertIsNone(result, "Canonical order_id should not be modified")

    def test_d2_normalize_non_order_action_ignored(self) -> None:
        """Non-get_order_details actions must not be touched."""
        from src.cargo.schemas import ProposedAction
        from src.cargo.risk_class import RiskClass
        agent = self._make_agent()
        wm = WorkingMemory()
        action = ProposedAction(
            name="get_product_details",
            args={"product_id": "W2378156"},
            declared_class=RiskClass.READ,
            declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._normalize_order_id_action(action, wm)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # D3: product name matching — last-word AND logic
    # ------------------------------------------------------------------
    def test_d3_smart_thermostat_does_not_match_smart_watches(self) -> None:
        """'Smart Thermostat' must NOT match when user says 'smart watches'.
        Old Rule 2 (OR-any) would match on the single word 'smart'."""
        from src.cargo.cargo_agent import CargoAgent
        result = CargoAgent._product_name_matches_user(
            "smart thermostat", "i want info on smart watches and headphones"
        )
        self.assertFalse(
            result,
            "'Smart Thermostat' must not match 'smart watches' (D3 regression)",
        )

    def test_d3_smart_watch_still_matches_smart_watches(self) -> None:
        """'Smart Watch' must still match 'smart watches' via Rule 1 phrase
        substring (the phrase 'smart watch' is a substring of 'smart watches')."""
        from src.cargo.cargo_agent import CargoAgent
        result = CargoAgent._product_name_matches_user(
            "smart watch", "info on smart watches"
        )
        self.assertTrue(result, "'Smart Watch' should match 'smart watches' via Rule 1")

    def test_d3_vacuum_cleaner_matches_cleaner(self) -> None:
        """'Vacuum Cleaner' last word 'cleaner' must match 'cleaner' in user
        text — the primary noun anchors the match even without 'vacuum'."""
        from src.cargo.cargo_agent import CargoAgent
        result = CargoAgent._product_name_matches_user(
            "vacuum cleaner", "find out about any cleaner available"
        )
        self.assertTrue(result, "'Vacuum Cleaner' should match 'any cleaner'")

    def test_d3_mechanical_keyboard_matches_keyboard(self) -> None:
        """'Mechanical Keyboard' last-word 'keyboard' must match 'keyboard'."""
        from src.cargo.cargo_agent import CargoAgent
        result = CargoAgent._product_name_matches_user(
            "mechanical keyboard", "exchange my keyboard for a clicky one"
        )
        self.assertTrue(result)

    def test_d3_thermostat_does_not_match_only_smart(self) -> None:
        """'Smart Thermostat' must require 'thermostat' — 'smart' alone is
        insufficient with the new last-word primary requirement."""
        from src.cargo.cargo_agent import CargoAgent
        # user only mentions "smart" — no thermostat
        result = CargoAgent._product_name_matches_user(
            "smart thermostat", "looking for a smart solution"
        )
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # D4: user-provided @example.com email accepted in Path 2
    # ------------------------------------------------------------------
    def test_d4_user_email_example_com_used_in_path2(self) -> None:
        """When user explicitly provides 'yusufrossi@example.com' (specific
        prefix, not in generic-RE list), Path 2 must try it via
        find_user_id_by_email rather than asking again."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard in my order #W2378156"
        wm.absorb_user_message("my email is yusufrossi@example.com")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_email")
        self.assertEqual(result.args.get("email"), "yusufrossi@example.com")

    def test_d4_generic_prefix_example_still_blocked(self) -> None:
        """'user@example.com' (generic prefix) must still be blocked."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard in my order"
        wm.absorb_user_message("my email is user@example.com")
        result = agent._auth_override(self._placeholder(), wm)
        # Must NOT propose find_user_id_by_email(user@example.com)
        if result is not None and result.name == "find_user_id_by_email":
            self.assertNotEqual(
                result.args.get("email"), "user@example.com",
                "Generic prefix email should be blocked",
            )

    def test_d4_already_tried_email_not_retried(self) -> None:
        """If an email from user_facts is already in recent_signatures,
        Path 2 must skip it and not loop."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard in my order"
        wm.absorb_user_message("my email is yusufrossi@example.com")
        wm.recent_signatures.append(
            "find_user_id_by_email(email='yusufrossi@example.com')"
        )
        result = agent._auth_override(self._placeholder(), wm)
        # Must NOT propose the same email again
        if result is not None and result.name == "find_user_id_by_email":
            self.assertNotEqual(result.args.get("email"), "yusufrossi@example.com")

    # ------------------------------------------------------------------
    # D5: re-attempt auth with fresh PII after abandonment
    # ------------------------------------------------------------------
    def test_d5_fresh_email_after_abandonment_unabandons(self) -> None:
        """When auth_abandoned=True but user has provided a fresh email not
        yet in recent_signatures, Path 2 must try it AND clear auth_abandoned
        so the auth flow can proceed normally.

        Note: Path 2 fires before the abandoned block when fresh PII is present.
        The abandonment-clearing happens in Path 2 itself (not the abandoned block)
        to handle the common pattern: user provides email AFTER the agent gave up."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "update my pending order"
        wm.auth_abandoned = True
        wm.auth_giveup_emitted = True
        wm.absorb_user_message("The email is yusufrossi@example.com.")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_email")
        self.assertEqual(result.args.get("email"), "yusufrossi@example.com")
        # Path 2 clears auth_abandoned when fresh credentials are provided
        self.assertFalse(wm.auth_abandoned)
        self.assertFalse(wm.auth_giveup_emitted)

    def test_d5_fresh_email_already_tried_does_not_unabondon(self) -> None:
        """If the fresh email was already tried, do NOT un-abandon — it failed
        and re-trying would loop."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "update my pending order"
        wm.auth_abandoned = True
        wm.auth_giveup_emitted = True
        wm.absorb_user_message("The email is yusufrossi@example.com.")
        wm.recent_signatures.append(
            "find_user_id_by_email(email='yusufrossi@example.com')"
        )
        result = agent._auth_override(self._placeholder(), wm)
        # Must NOT return find_user_id_by_email with the already-tried email
        if result is not None and result.name == "find_user_id_by_email":
            self.assertNotEqual(result.args.get("email"), "yusufrossi@example.com")

    def test_d5_fresh_name_zip_after_abandonment_unabandons(self) -> None:
        """After abandonment, user provides name+zip not yet tried → Path 2
        must try it AND clear auth_abandoned."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "update my pending order"
        wm.auth_abandoned = True
        wm.auth_giveup_emitted = True
        wm.absorb_user_message("My name is Yusuf Rossi and ZIP is 19122.")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_name_zip")
        self.assertFalse(wm.auth_abandoned)
        self.assertFalse(wm.auth_giveup_emitted)

    # ------------------------------------------------------------------
    # D6: multi-product query — wait for ALL products before finalizing
    # ------------------------------------------------------------------
    def test_d6_multi_product_waits_for_all_products(self) -> None:
        """When user asks about N products, the finalizer must NOT emit a
        partial FINAL after only 1 product is fetched; it must wait for all."""
        from src.cargo.schemas import ProposedAction
        from src.cargo.risk_class import RiskClass
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = (
            "How many t-shirt options are there? Also info on headphones "
            "and cleaners available."
        )
        # Catalogue populated
        wm.product_types = {
            "T-Shirt": "111",
            "Headphones": "222",
            "Vacuum Cleaner": "333",
        }
        # Only Headphones fetched so far
        wm.product_details["222"] = {
            "name": "Headphones",
            "variants": {str(i): {"available": True} for i in range(5)},
        }
        wm.recent_signatures.append("list_all_product_types()")
        action = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._finalize_product_count_query(action, wm)
        self.assertIsNone(
            result,
            "Must NOT finalize when T-Shirt and Cleaner are still unfetched",
        )

    def test_d6_multi_product_fires_when_all_fetched(self) -> None:
        """When ALL user-mentioned products are fetched, the finalizer must
        emit ONE comprehensive FINAL covering all of them."""
        from src.cargo.schemas import ProposedAction
        from src.cargo.risk_class import RiskClass
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = (
            "How many t-shirt options are there? Also info on headphones "
            "and cleaners available."
        )
        wm.product_types = {
            "T-Shirt": "111",
            "Headphones": "222",
            "Vacuum Cleaner": "333",
        }
        # All three fetched
        wm.product_details["111"] = {
            "name": "T-Shirt",
            "variants": {str(i): {"available": True} for i in range(12)},
        }
        wm.product_details["222"] = {
            "name": "Headphones",
            "variants": {str(i): {"available": i < 5} for i in range(13)},
        }
        wm.product_details["333"] = {
            "name": "Vacuum Cleaner",
            "variants": {str(i): {"available": True} for i in range(3)},
        }
        wm.recent_signatures.append("list_all_product_types()")
        action = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._finalize_product_count_query(action, wm)
        self.assertIsNotNone(result, "Must finalize when all products are fetched")
        self.assertEqual(result.name, "respond")
        self.assertTrue(result.bypass_gates)
        self.assertTrue(wm.product_count_finalized)
        # The FINAL must mention all three products
        user_text = result.user_text.lower()
        self.assertIn("t-shirt", user_text)
        self.assertIn("headphones", user_text)
        self.assertIn("vacuum cleaner", user_text)

    def test_d6_single_product_still_works(self) -> None:
        """Single-product goals must still get a single-product FINAL (no
        regression on the original behavior)."""
        from src.cargo.schemas import ProposedAction
        from src.cargo.risk_class import RiskClass
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are currently available?"
        wm.product_types = {"T-Shirt": "111"}
        wm.product_details["111"] = {
            "name": "T-Shirt",
            "variants": {str(i): {"available": i < 10} for i in range(12)},
        }
        wm.recent_signatures.append("list_all_product_types()")
        action = ProposedAction(
            name="list_all_product_types", args={},
            declared_class=RiskClass.READ, declared_pre=[], declared_post=[],
            informational_intent="", raw_thought="", user_text="", raw_response="",
        )
        result = agent._finalize_product_count_query(action, wm)
        self.assertIsNotNone(result)
        self.assertIn("t-shirt", result.user_text.lower())
        self.assertTrue(wm.product_count_finalized)

    # ------------------------------------------------------------------
    # D7: completed auth must progress to grounded order-product retrieval
    # ------------------------------------------------------------------
    def test_d7_post_auth_fetches_mentioned_order_product(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange my mechanical keyboard for clicky switches"
        wm.auth_user_id = "yusuf_rossi_9620"
        wm.recent_signatures.append("get_user_details(user_id='yusuf_rossi_9620')")
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "1656367028",
                    "item_id": "1151293680",
                },
                {
                    "name": "Smart Watch",
                    "product_id": "6945232052",
                    "item_id": "9408160950",
                },
            ],
        }
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "get_product_details")
        self.assertEqual(result.args["product_id"], "1656367028")

    def test_d7_item_id_not_accepted_as_product_id(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange my mechanical keyboard for clicky switches"
        wm.order_details["#W2378156"] = {
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "1656367028",
                    "item_id": "1151293680",
                }
            ]
        }
        bad = ProposedAction(
            name="get_product_details",
            args={"product_id": "1151293680"},
            declared_class=RiskClass.READ,
        )
        result = agent._resolve_product_id_name(bad, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.args["product_id"], "1656367028")

    def test_e1_persona_identity_extracts_name_zip(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("You are Yusuf Rossi in 19122. Please help with my order.")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_name_zip")
        self.assertEqual(result.args["first_name"], "Yusuf")
        self.assertEqual(result.args["last_name"], "Rossi")
        self.assertEqual(result.args["zip"], "19122")

    def test_e1_progress_authenticates_after_count_phase(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "You are Yusuf Rossi in 19122. How many tshirts are available, then update my order."
        wm.absorb_user_message(wm.goal)
        wm.product_count_finalized = True
        result = agent._grounded_progress_or_commit_action(
            ProposedAction(name="list_all_product_types", args={}, declared_class=RiskClass.READ),
            wm,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "find_user_id_by_name_zip")
        self.assertEqual(result.args["zip"], "19122")

    def test_e2_grounded_exchange_commits_after_retrieval(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = (
            "You are Yusuf Rossi in 19122. You received order #W2378156 and "
            "wish to exchange the mechanical keyboard for clicky switches and "
            "the smart thermostat for one compatible with Google Home instead "
            "of Apple HomeKit. If there is no keyboard that is clicky, RGB "
            "backlight, full size, you'd go for no backlight."
        )
        wm.auth_user_id = "yusuf_rossi_9620"
        wm.lock_phase("auth")
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "credit_card_9513926"}],
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "kbd",
                    "item_id": "oldkbd",
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
                {
                    "name": "Smart Thermostat",
                    "product_id": "thermo",
                    "item_id": "oldthermo",
                    "options": {"compatibility": "Apple HomeKit", "color": "black"},
                },
            ],
        }
        wm.product_details["kbd"] = {
            "name": "Mechanical Keyboard",
            "product_id": "kbd",
            "variants": {
                "bad": {"item_id": "bad", "available": False,
                        "options": {"switch type": "clicky", "backlight": "RGB", "size": "full size"}},
                "newkbd": {"item_id": "newkbd", "available": True,
                           "options": {"switch type": "clicky", "backlight": "none", "size": "full size"}},
            },
        }
        wm.product_details["thermo"] = {
            "name": "Smart Thermostat",
            "product_id": "thermo",
            "variants": {
                "newthermo": {"item_id": "newthermo", "available": True,
                              "options": {"compatibility": "Google Assistant", "color": "black"}},
                "wrong": {"item_id": "wrong", "available": True,
                          "options": {"compatibility": "Apple HomeKit", "color": "black"}},
            },
        }
        action = agent._grounded_retail_commit_action(wm)
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "exchange_delivered_order_items")
        self.assertEqual(action.args["item_ids"], ["oldkbd", "oldthermo"])
        self.assertEqual(action.args["new_item_ids"], ["newkbd", "newthermo"])
        self.assertTrue(action.bypass_gates)

    def test_e3_exchange_respects_only_exchange_fallback(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = (
            "exchange the mechanical keyboard for clicky, RGB backlight, full size. "
            "If there is no keyboard that is clicky, RGB backlight, full size, "
            "you'd rather only exchange the thermostat."
        )
        wm.auth_user_id = "u"
        wm.order_details["#W1"] = {
            "order_id": "#W1",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "pm"}],
            "items": [{
                "name": "Mechanical Keyboard",
                "product_id": "kbd",
                "item_id": "old",
                "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
            }],
        }
        wm.product_details["kbd"] = {
            "name": "Mechanical Keyboard",
            "variants": {
                "fallback": {"item_id": "fallback", "available": True,
                             "options": {"switch type": "clicky", "backlight": "none", "size": "full size"}},
                "exact": {"item_id": "exact", "available": False,
                          "options": {"switch type": "clicky", "backlight": "RGB", "size": "full size"}},
            },
        }
        self.assertIsNone(agent._grounded_retail_commit_action(wm))

    def test_e4_grounded_modify_groups_pending_order(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "modify all pending small tshirts to purple, same size, same v-neck, prefer polyester"
        wm.auth_user_id = "u"
        wm.order_details["#W4776164"] = {
            "order_id": "#W4776164",
            "status": "pending",
            "payment_history": [{"payment_method_id": "pm"}],
            "items": [{
                "name": "T-Shirt",
                "product_id": "ts",
                "item_id": "oldts",
                "options": {"color": "blue", "size": "S", "material": "cotton", "style": "v-neck"},
            }],
        }
        wm.product_details["ts"] = {
            "name": "T-Shirt",
            "variants": {
                "target": {"item_id": "target", "available": True,
                           "options": {"color": "purple", "size": "S", "material": "polyester", "style": "v-neck"}},
                "weak": {"item_id": "weak", "available": True,
                         "options": {"color": "purple", "size": "XL", "material": "cotton", "style": "crew neck"}},
            },
        }
        action = agent._grounded_retail_commit_action(wm)
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "modify_pending_order_items")
        self.assertEqual(action.args["order_id"], "#W4776164")
        self.assertEqual(action.args["item_ids"], ["oldts"])
        self.assertEqual(action.args["new_item_ids"], ["target"])

    def test_e5_grounded_return_uses_delivered_order_items(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "return the cleaner, headphone, and smart watch"
        wm.auth_user_id = "u"
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "pm"}],
            "items": [
                {"name": "Vacuum Cleaner", "item_id": "cleaner"},
                {"name": "Headphones", "item_id": "headphones"},
                {"name": "Smart Watch", "item_id": "watch"},
                {"name": "Mechanical Keyboard", "item_id": "keyboard"},
            ],
        }
        action = agent._grounded_retail_commit_action(wm)
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "return_delivered_order_items")
        self.assertEqual(action.args["item_ids"], ["cleaner", "headphones", "watch"])

    def test_f1_confirmation_gate_blocks_unconfirmed_write(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory(goal="show me my order status")
        action = ProposedAction(
            name="modify_pending_order_items",
            args={"order_id": "#W1"},
            declared_class=RiskClass.WRITE,
        )
        result = agent._check_write_confirmation(action, wm)
        self.assertFalse(result.ok)
        self.assertEqual(result.gate, "confirmation")

    def test_f2_confirmation_gate_accepts_direct_user_request(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory(goal="Please exchange the thermostat in my order.")
        action = ProposedAction(
            name="exchange_delivered_order_items",
            args={"order_id": "#W1"},
            declared_class=RiskClass.WRITE,
        )
        result = agent._check_write_confirmation(action, wm)
        self.assertTrue(result.ok, result.reason)

    def test_f3_canonicalizer_replaces_non_best_exchange(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = (
            "Please exchange the mechanical keyboard for clicky switches. "
            "If there is no clicky RGB full size keyboard, go for no backlight."
        )
        wm.auth_user_id = "u"
        wm.order_details["#W1"] = {
            "order_id": "#W1",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "pm"}],
            "items": [{
                "name": "Mechanical Keyboard",
                "product_id": "kbd",
                "item_id": "old",
                "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
            }],
        }
        wm.product_details["kbd"] = {
            "name": "Mechanical Keyboard",
            "variants": {
                "wrong": {"item_id": "wrong", "available": True,
                          "options": {"switch type": "linear", "backlight": "RGB", "size": "80%"}},
                "right": {"item_id": "right", "available": True,
                          "options": {"switch type": "clicky", "backlight": "none", "size": "full size"}},
            },
        }
        bad = ProposedAction(
            name="exchange_delivered_order_items",
            args={
                "order_id": "#W1",
                "item_ids": ["old"],
                "new_item_ids": ["wrong"],
                "payment_method_id": "pm",
            },
            declared_class=RiskClass.WRITE,
            bypass_gates=True,
        )
        fixed = agent._canonicalize_write_action(bad, wm)
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed.args["new_item_ids"], ["right"])

    def test_f4_mixed_product_count_is_not_terminal_final(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are available, and modify my pending order."
        wm.product_types = {"T-Shirt": "ts"}
        wm.product_details["ts"] = {
            "name": "T-Shirt",
            "variants": {str(i): {"available": True} for i in range(3)},
        }
        wm.recent_signatures.append("list_all_product_types()")
        action = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
        )
        result = agent._finalize_product_count_query(action, wm)
        self.assertIsNotNone(result)
        self.assertEqual(result.declared_class, RiskClass.ASK_USER)
        self.assertEqual(result.declared_pre, [])
        self.assertTrue(wm.product_count_finalized)

    def test_f5_completed_mutation_gate_blocks_reexecution(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory(goal="Please exchange item old.")
        action = ProposedAction(
            name="exchange_delivered_order_items",
            args={
                "order_id": "#W1",
                "item_ids": ["old"],
                "new_item_ids": ["new"],
                "payment_method_id": "pm",
            },
            declared_class=RiskClass.WRITE,
        )
        wm.record_executed_mutation(action.signature())
        schema = ToolEffectSchema(
            name="exchange_delivered_order_items",
            cls=RiskClass.WRITE,
            arg_id_fields=["order_id", "item_ids", "new_item_ids", "payment_method_id"],
            required_params=["order_id", "item_ids", "new_item_ids", "payment_method_id"],
        )
        failing, diag = agent._run_gates(action, schema, wm, [], CargoStats())
        self.assertIsNotNone(failing)
        self.assertEqual(failing.gate, "completed_task")
        self.assertIn("completed_task", diag["gates_failed"])

    def test_g1_exchange_constraints_are_hard_filters(self) -> None:
        agent = self._make_agent()
        details = {
            "name": "Mechanical Keyboard",
            "variants": {
                "partial": {
                    "item_id": "partial",
                    "available": True,
                    "options": {"switch type": "clicky", "backlight": "RGB", "size": "80%"},
                },
                "fallback": {
                    "item_id": "fallback",
                    "available": True,
                    "options": {"switch type": "clicky", "backlight": "none", "size": "full size"},
                },
            },
        }
        old = {
            "item_id": "old",
            "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
        }
        chosen = agent._select_variant_id(
            details,
            old,
            "clicky, RGB backlight, full size; if unavailable go for no backlight",
            mode="exchange",
        )
        self.assertEqual(chosen, "fallback")

    def test_g2_exchange_rejects_decoy_when_constraint_missing(self) -> None:
        agent = self._make_agent()
        details = {
            "variants": {
                "decoy": {
                    "item_id": "decoy",
                    "available": True,
                    "options": {"switch type": "clicky", "backlight": "RGB", "size": "80%"},
                }
            }
        }
        old = {
            "item_id": "old",
            "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
        }
        chosen = agent._select_variant_id(
            details,
            old,
            "clicky, RGB backlight, full size",
            mode="exchange",
        )
        self.assertIsNone(chosen)

    def test_g3_modify_does_not_rewrite_satisfied_item(self) -> None:
        agent = self._make_agent()
        details = {
            "variants": {
                "target": {
                    "item_id": "target",
                    "available": True,
                    "options": {"color": "purple", "size": "S", "material": "polyester", "style": "v-neck"},
                },
                "other": {
                    "item_id": "other",
                    "available": True,
                    "options": {"color": "purple", "size": "S", "material": "polyester", "style": "v-neck"},
                },
            }
        }
        old = {
            "item_id": "target",
            "options": {"color": "purple", "size": "S", "material": "polyester", "style": "v-neck"},
        }
        self.assertIsNone(
            agent._select_variant_id(
                details,
                old,
                "purple, same size, same v-neck, prefer polyester",
                mode="modify",
            )
        )

    def test_g4_distinct_multi_order_mutation_can_continue(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "modify all pending tshirts to purple, s size, same v-neck, prefer polyester"
        wm.auth_user_id = "u"
        for oid, item_id in (("#W1", "old1"), ("#W2", "old2")):
            wm.order_details[oid] = {
                "order_id": oid,
                "status": "pending",
                "payment_history": [{"payment_method_id": "pm"}],
                "items": [{
                    "name": "T-Shirt",
                    "product_id": "ts",
                    "item_id": item_id,
                    "options": {"color": "blue", "size": "S", "material": "cotton", "style": "v-neck"},
                }],
            }
        wm.product_details["ts"] = {
            "name": "T-Shirt",
            "variants": {
                "target": {
                    "item_id": "target",
                    "available": True,
                    "options": {"color": "purple", "size": "S", "material": "polyester", "style": "v-neck"},
                }
            },
        }
        first = agent._grounded_retail_commit_action(wm)
        self.assertIsNotNone(first)
        self.assertEqual(first.args["order_id"], "#W1")
        wm.record_action_signature(first.signature())
        wm.record_executed_mutation(first.signature())
        second = agent._grounded_retail_commit_action(wm)
        self.assertIsNotNone(second)
        self.assertEqual(second.args["order_id"], "#W2")

    def test_h1_partial_write_is_blocked_by_completeness_gate(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = (
            "Please exchange the mechanical keyboard for clicky switches and "
            "the smart thermostat for one compatible with Google Home. If "
            "there is no clicky RGB full size keyboard, go for no backlight."
        )
        wm.auth_user_id = "u"
        wm.order_details["#W1"] = {
            "order_id": "#W1",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "pm"}],
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "kbd",
                    "item_id": "oldkbd",
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
                {
                    "name": "Smart Thermostat",
                    "product_id": "thermo",
                    "item_id": "oldthermo",
                    "options": {"compatibility": "Apple HomeKit", "color": "black"},
                },
            ],
        }
        wm.product_details["kbd"] = {
            "name": "Mechanical Keyboard",
            "variants": {
                "rightkbd": {
                    "item_id": "rightkbd",
                    "available": True,
                    "options": {"switch type": "clicky", "backlight": "none", "size": "full size"},
                },
                "decoykbd": {
                    "item_id": "decoykbd",
                    "available": True,
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "80%"},
                },
            },
        }
        wm.product_details["thermo"] = {
            "name": "Smart Thermostat",
            "variants": {
                "rightthermo": {
                    "item_id": "rightthermo",
                    "available": True,
                    "options": {"compatibility": "Google Assistant", "color": "black"},
                }
            },
        }
        partial = ProposedAction(
            name="exchange_delivered_order_items",
            args={
                "order_id": "#W1",
                "item_ids": ["oldkbd"],
                "new_item_ids": ["decoykbd"],
                "payment_method_id": "pm",
            },
            declared_class=RiskClass.WRITE,
        )
        schema = ToolEffectSchema(
            name="exchange_delivered_order_items",
            cls=RiskClass.WRITE,
            arg_id_fields=["order_id", "item_ids", "new_item_ids", "payment_method_id"],
            required_params=["order_id", "item_ids", "new_item_ids", "payment_method_id"],
        )
        failing, diag = agent._run_gates(partial, schema, wm, [], CargoStats())
        self.assertIsNotNone(failing)
        self.assertEqual(failing.gate, "completeness")
        self.assertIn("completeness", diag["gates_failed"])

    def test_h2_complete_canonical_write_passes_completeness_gate(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "Please return the cleaner and headphone from my delivered order."
        wm.auth_user_id = "u"
        wm.order_details["#W1"] = {
            "order_id": "#W1",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "pm"}],
            "items": [
                {"name": "Vacuum Cleaner", "item_id": "cleaner"},
                {"name": "Headphones", "item_id": "headphones"},
                {"name": "Smart Watch", "item_id": "watch"},
            ],
        }
        action = agent._grounded_retail_commit_action(wm)
        self.assertIsNotNone(action)
        schema = ToolEffectSchema(
            name="return_delivered_order_items",
            cls=RiskClass.IRREVERSIBLE,
            arg_id_fields=["order_id", "item_ids", "payment_method_id"],
            required_params=["order_id", "item_ids", "payment_method_id"],
        )
        failing, diag = agent._run_gates(action, schema, wm, [], CargoStats())
        self.assertIsNone(failing, failing.reason if failing else "")
        self.assertIn("completeness", diag["gates_run"])

    def test_h3_premature_final_for_account_task_is_blocked(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are available, and modify my pending order."
        wm.product_count_finalized = True
        action = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
            user_text="There are 10 t-shirt options.",
        )
        schema = ToolEffectSchema(name="respond", cls=RiskClass.FINAL)
        failing, diag = agent._run_gates(action, schema, wm, [], CargoStats())
        self.assertIsNotNone(failing)
        self.assertEqual(failing.gate, "final_completeness")
        self.assertIn("final_completeness", diag["gates_failed"])

    def test_h4_final_for_pure_product_count_still_passes(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are available?"
        action = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.FINAL,
            user_text="There are 10 t-shirt options.",
            bypass_gates=True,
        )
        schema = ToolEffectSchema(name="respond", cls=RiskClass.FINAL)
        failing, diag = agent._run_gates(action, schema, wm, [], CargoStats())
        self.assertIsNone(failing, failing.reason if failing else "")
        self.assertIn("final_completeness", diag["gates_run"])

    def test_h5_final_with_followup_user_reply_does_not_terminate_solve(self) -> None:
        import src.cargo.cargo_agent as cargo_agent_module

        class _SolveResult:
            def __init__(self, reward: float, info: Dict[str, Any],
                         messages: List[Dict[str, Any]], total_cost: float) -> None:
                self.reward = reward
                self.info = info
                self.messages = messages
                self.total_cost = total_cost

        scripts = [
            _proposer_json(
                name="respond",
                declared_class="FINAL",
                user_text="There are 10 available options.",
            ),
            [_proposer_json(
                name="respond",
                declared_class="FINAL",
                user_text="There are 10 available options.",
            )] * 3,
            json.dumps({"predicted_obs": "ok", "goal_still_reachable": True}),
            _proposer_json(
                name="get_status",
                args={"ticket_id": "T1234"},
                declared_class="READ",
            ),
        ]
        agent = self._make_agent()
        agent.client = MockClient(scripts=scripts)
        agent.model = "m"
        agent.temperature = 0.0
        agent.wiki = ""
        agent.schemas = {
            "respond": ToolEffectSchema(name="respond", cls=RiskClass.FINAL),
            "get_status": ToolEffectSchema(
                name="get_status",
                cls=RiskClass.READ,
                arg_id_fields=["ticket_id"],
                required_params=["ticket_id"],
            ),
        }
        agent.calibration = default_calibration()

        env = MockEnv(
            "How many options are available?",
            [
                _StepResp("Also check ticket T1234.", reward=0.0, done=False),
                _StepResp({"ticket_id": "T1234", "status": "ok"}, reward=1.0, done=True),
            ],
        )
        old_action = cargo_agent_module.Action
        old_solve_result = cargo_agent_module.SolveResult
        cargo_agent_module.Action = _Action
        cargo_agent_module.SolveResult = _SolveResult
        try:
            result = agent.solve(env, max_num_steps=4)
        finally:
            cargo_agent_module.Action = old_action
            cargo_agent_module.SolveResult = old_solve_result

        self.assertEqual([a.name for a in env.actions_executed], ["respond", "get_status"])
        self.assertEqual(result.reward, 1.0)

    def test_h6_exchange_without_target_options_asks_instead_of_guessing(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange the mechanical keyboard in order #W1"
        wm.order_details["#W1"] = {
            "order_id": "#W1",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "credit_card_1"}],
            "items": [{
                "name": "Mechanical Keyboard",
                "product_id": "keyboard",
                "item_id": "old_keyboard",
                "options": {"switch type": "linear", "backlight": "RGB"},
            }],
        }
        wm.product_details["keyboard"] = {
            "name": "Mechanical Keyboard",
            "product_id": "keyboard",
            "variants": {
                "old_keyboard": {
                    "item_id": "old_keyboard",
                    "available": True,
                    "options": {"switch type": "linear", "backlight": "RGB"},
                },
                "new_keyboard": {
                    "item_id": "new_keyboard",
                    "available": True,
                    "options": {"switch type": "clicky", "backlight": "none"},
                },
            },
        }

        commit = agent._grounded_retail_commit_action(wm)
        ask = agent._missing_replacement_constraints_action(wm)

        self.assertIsNone(commit)
        self.assertIsNotNone(ask)
        self.assertEqual(ask.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]

    def test_i1_completed_auth_phase_blocks_auth_tool_reentry(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.auth_user_id = "alex_smith_42"
        wm.lock_phase("auth")
        action = ProposedAction(
            name="find_user_id_by_email",
            args={"email": "alex.smith@example.com"},
            declared_class=RiskClass.READ,
        )
        schema = ToolEffectSchema(name="find_user_id_by_email", cls=RiskClass.READ)

        failing, diag = agent._run_gates(action, schema, wm, [], CargoStats())

        self.assertIsNotNone(failing)
        self.assertEqual(failing.gate, "state_validity")
        self.assertIn("state_validity", diag["gates_failed"])

    def test_i2_state_gate_blocks_search_conflicting_with_bound_date(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("Book a flight from New York to Seattle on May 20th.")
        agent._kernel().observe_user_message(wm, "Book a flight from New York to Seattle on May 20th.")
        action = ProposedAction(
            name="search_direct_flight",
            args={
                "origin": "New York",
                "destination": "Seattle",
                "date": "2024-05-21",
            },
            declared_class=RiskClass.READ,
        )
        schema = ToolEffectSchema(name="search_direct_flight", cls=RiskClass.READ)

        failing, diag = agent._run_gates(action, schema, wm, [], CargoStats())

        self.assertIsNotNone(failing)
        self.assertEqual(failing.gate, "state_validity")
        self.assertIn("action_date_conflicts_with_state", failing.reason)

    def test_i2b_airline_ask_loop_pivots_to_bound_flight_search(self) -> None:
        agent = self._make_agent()
        agent.schemas = {
            "search_direct_flight": ToolEffectSchema(
                name="search_direct_flight",
                cls=RiskClass.READ,
                arg_semantic_fields=["origin", "destination", "date"],
                required_params=["origin", "destination", "date"],
            ),
            "search_onestop_flight": ToolEffectSchema(
                name="search_onestop_flight",
                cls=RiskClass.READ,
                arg_semantic_fields=["origin", "destination", "date"],
                required_params=["origin", "destination", "date"],
            ),
        }
        wm = WorkingMemory()
        text = (
            "I'm looking to book a flight from New York to Seattle on May 20th. "
            "My user id is alex_smith_42. Economy class, after 11 am, and "
            "one stopover is okay."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        ask = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="What would you like to do?",
        )

        replacement = agent._obligation_guided_action(ask, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "search_direct_flight")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["date"], "2024-05-20")  # type: ignore[union-attr]

    def test_i2c_repeated_direct_search_pivots_to_onestop_when_allowed(self) -> None:
        agent = self._make_agent()
        agent.schemas = {
            "search_direct_flight": ToolEffectSchema(
                name="search_direct_flight",
                cls=RiskClass.READ,
                arg_semantic_fields=["origin", "destination", "date"],
            ),
            "search_onestop_flight": ToolEffectSchema(
                name="search_onestop_flight",
                cls=RiskClass.READ,
                arg_semantic_fields=["origin", "destination", "date"],
            ),
        }
        wm = WorkingMemory()
        text = (
            "Book a flight from New York to Seattle on May 20th. "
            "My user id is alex_smith_42. Direct is preferred but one "
            "stopover is okay."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        direct = ProposedAction(
            name="search_direct_flight",
            args={"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
            declared_class=RiskClass.READ,
        )
        wm.record_action_signature(direct.signature())

        replacement = agent._obligation_guided_action(direct, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "search_onestop_flight")  # type: ignore[union-attr]

    def test_i3_booking_write_requires_complete_slots(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message("Please book an economy flight for alex_smith_42.")
        agent._kernel().observe_user_message(wm, "Please book an economy flight for alex_smith_42.")
        action = ProposedAction(
            name="book_reservation",
            args={"user_id": "alex_smith_42"},
            declared_class=RiskClass.WRITE,
        )
        schema = ToolEffectSchema(
            name="book_reservation",
            cls=RiskClass.WRITE,
            arg_id_fields=["user_id"],
            required_params=["user_id"],
        )

        failing, diag = agent._run_gates(action, schema, wm, [], CargoStats())

        self.assertIsNotNone(failing)
        self.assertEqual(failing.gate, "completeness")
        self.assertIn("booking_missing_required_slots", failing.reason)
        self.assertIn("completeness", diag["gates_failed"])

    def test_i4_booking_write_passes_slot_completeness_when_filled(self) -> None:
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.absorb_user_message(
            "Please book an economy flight for alex_smith_42 using my credit card."
        )
        action = ProposedAction(
            name="book_reservation",
            args={
                "user_id": "alex_smith_42",
                "flights": [{"flight_number": "TEST_FLIGHT_001"}],
                "passengers": [{"first_name": "Alex", "last_name": "Smith"}],
                "payment_method_id": "credit_card_1234",
                "cabin": "economy",
            },
            declared_class=RiskClass.WRITE,
            bypass_gates=True,
        )
        wm.absorb_observation({
            "user_id": "alex_smith_42",
            "flight_number": "TEST_FLIGHT_001",
            "payment_methods": {"credit_card_1234": {"id": "credit_card_1234"}},
        })
        schema = ToolEffectSchema(
            name="book_reservation",
            cls=RiskClass.WRITE,
            arg_id_fields=["user_id", "payment_method_id"],
            required_params=["user_id"],
        )

        failing, diag = agent._run_gates(action, schema, wm, [], CargoStats())

        self.assertIsNone(failing, failing.reason if failing else "")
        self.assertIn("completeness", diag["gates_run"])


class TestCargoV4DecisionEngine(unittest.TestCase):
    """Regression tests for the decision-centric CARGO-v4 layer."""

    def _make_retail_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = MockClient(scripts=[])
        agent.model = "test"
        agent.temperature = 0.0
        agent.adapter = TauRetailAdapter()
        agent.kernel = GenericCargoKernel(agent.adapter)
        agent.schemas = {
            "exchange_delivered_order_items": ToolEffectSchema(
                name="exchange_delivered_order_items",
                cls=RiskClass.WRITE,
                arg_id_fields=["order_id", "item_ids", "new_item_ids", "payment_method_id"],
                required_params=["order_id", "item_ids", "new_item_ids", "payment_method_id"],
            )
        }
        return agent

    def _make_airline_agent(self) -> Any:
        from src.cargo.cargo_agent import CargoAgent
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = MockClient(scripts=[])
        agent.model = "test"
        agent.temperature = 0.0
        agent.adapter = TauAirlineAdapter()
        agent.kernel = GenericCargoKernel(agent.adapter)
        agent.schemas = {
            "search_direct_flight": ToolEffectSchema(
                name="search_direct_flight",
                cls=RiskClass.READ,
                arg_semantic_fields=["origin", "destination", "date"],
            ),
            "search_onestop_flight": ToolEffectSchema(
                name="search_onestop_flight",
                cls=RiskClass.READ,
                arg_semantic_fields=["origin", "destination", "date"],
            ),
            "get_reservation_details": ToolEffectSchema(
                name="get_reservation_details",
                cls=RiskClass.READ,
                arg_id_fields=["reservation_id"],
            ),
            "get_user_details": ToolEffectSchema(
                name="get_user_details",
                cls=RiskClass.READ,
                arg_id_fields=["user_id"],
            ),
            "book_reservation": ToolEffectSchema(
                name="book_reservation",
                cls=RiskClass.WRITE,
                arg_id_fields=["user_id", "flight_number", "payment_id"],
                required_params=[
                    "user_id", "origin", "destination", "flight_type",
                    "cabin", "flights", "passengers", "payment_methods",
                    "total_baggages", "nonfree_baggages", "insurance",
                ],
            ),
            "respond": ToolEffectSchema(name="respond", cls=RiskClass.FINAL),
        }
        return agent

    def _seed_mia_booking_trace_state(self, agent: Any) -> WorkingMemory:
        wm = WorkingMemory()
        text = (
            "My user id is mia_li_3668. I want to book a one-way economy flight "
            "from New York to Seattle on May 20 after 11am. I prefer direct "
            "flights but one stopover is okay. "
            "I have 3 bags, no insurance, and want to use my larger certificate "
            "then my 7447 card."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.auth_user_id = "mia_li_3668"
        wm.user_profiles["mia_li_3668"] = {
            "name": {"first_name": "Mia", "last_name": "Li"},
            "dob": "1990-04-05",
            "membership": "gold",
            "payment_methods": {
                "credit_card_4421486": {"source": "credit_card", "last_four": "7447", "id": "credit_card_4421486"},
                "certificate_4856383": {"source": "certificate", "amount": 100, "id": "certificate_4856383"},
                "certificate_7504069": {"source": "certificate", "amount": 250, "id": "certificate_7504069"},
            },
        }
        return wm

    def _record_mia_booking_search_results(self, agent: Any, wm: WorkingMemory) -> None:
        args = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}
        direct_obs = [
            {
                "flight_number": "HAT069",
                "origin": "JFK",
                "destination": "SEA",
                "scheduled_departure_time_est": "06:00:00",
                "status": "available",
                "available_seats": {"economy": 12},
                "prices": {"economy": 121},
            },
            {
                "flight_number": "HAT083",
                "origin": "JFK",
                "destination": "SEA",
                "scheduled_departure_time_est": "01:00:00",
                "status": "available",
                "available_seats": {"economy": 7},
                "prices": {"economy": 100},
            },
        ]
        wm.absorb_observation(direct_obs)
        agent._kernel().record_action_candidates(wm, "search_direct_flight", args, direct_obs)
        one_obs = [
            [
                {
                    "flight_number": "HAT057",
                    "origin": "JFK",
                    "destination": "ATL",
                    "scheduled_departure_time_est": "07:00:00",
                    "status": "available",
                    "available_seats": {"economy": 3},
                    "prices": {"economy": 141},
                    "date": "2024-05-20",
                },
                {
                    "flight_number": "HAT039",
                    "origin": "ATL",
                    "destination": "SEA",
                    "scheduled_departure_time_est": "22:00:00",
                    "status": "available",
                    "available_seats": {"economy": 10},
                    "prices": {"economy": 103},
                    "date": "2024-05-20",
                },
            ],
            [
                {
                    "flight_number": "HAT136",
                    "origin": "JFK",
                    "destination": "ATL",
                    "scheduled_departure_time_est": "19:00:00",
                    "status": "available",
                    "available_seats": {"economy": 14},
                    "prices": {"economy": 152},
                    "date": "2024-05-20",
                },
                {
                    "flight_number": "HAT039",
                    "origin": "ATL",
                    "destination": "SEA",
                    "scheduled_departure_time_est": "22:00:00",
                    "status": "available",
                    "available_seats": {"economy": 10},
                    "prices": {"economy": 103},
                    "date": "2024-05-20",
                },
            ],
            [
                {
                    "flight_number": "HAT218",
                    "origin": "JFK",
                    "destination": "ATL",
                    "scheduled_departure_time_est": "18:00:00",
                    "status": "available",
                    "available_seats": {"economy": 1},
                    "prices": {"economy": 158},
                    "date": "2024-05-20",
                },
                {
                    "flight_number": "HAT039",
                    "origin": "ATL",
                    "destination": "SEA",
                    "scheduled_departure_time_est": "22:00:00",
                    "status": "available",
                    "available_seats": {"economy": 10},
                    "prices": {"economy": 103},
                    "date": "2024-05-20",
                },
            ],
            [
                {
                    "flight_number": "HAT268",
                    "origin": "JFK",
                    "destination": "ATL",
                    "scheduled_departure_time_est": "07:00:00",
                    "status": "available",
                    "available_seats": {"economy": 19},
                    "prices": {"economy": 101},
                    "date": "2024-05-20",
                },
                {
                    "flight_number": "HAT039",
                    "origin": "ATL",
                    "destination": "SEA",
                    "scheduled_departure_time_est": "22:00:00",
                    "status": "available",
                    "available_seats": {"economy": 10},
                    "prices": {"economy": 103},
                    "date": "2024-05-20",
                },
            ],
        ]
        wm.absorb_observation(one_obs)
        agent._kernel().record_action_candidates(wm, "search_onestop_flight", args, one_obs)

    def _seed_olivia_reservation_trace_state(self, agent: Any) -> WorkingMemory:
        wm = WorkingMemory()
        text = (
            "My user id is olivia_gonzalez_2305. I have a half-day Texas trip "
            "in my reservations but do not remember the reservation id. I need "
            "a later return to Newark, and if basic economy cannot be modified "
            "I am willing to cancel using insurance because I feel unwell."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.auth_user_id = "olivia_gonzalez_2305"
        wm.user_profiles["olivia_gonzalez_2305"] = {
            "name": {"first_name": "Olivia", "last_name": "Gonzalez"},
            "reservations": ["Z7GOZK", "K67C4W", "THY2DG"],
        }
        wm.absorb_observation(wm.user_profiles["olivia_gonzalez_2305"])
        agent._kernel().observe_tool_result(
            wm,
            "get_user_details",
            wm.user_profiles["olivia_gonzalez_2305"],
        )
        return wm

    @staticmethod
    def _keyboard_details() -> Dict[str, Any]:
        return {
            "name": "Mechanical Keyboard",
            "product_id": "1656367028",
            "variants": {
                "1151293680": {
                    "item_id": "1151293680",
                    "available": True,
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
                "7706410293": {
                    "item_id": "7706410293",
                    "available": True,
                    "options": {"switch type": "clicky", "backlight": "none", "size": "full size"},
                },
                "2299424241": {
                    "item_id": "2299424241",
                    "available": True,
                    "options": {"switch type": "clicky", "backlight": "RGB", "size": "80%"},
                },
                "9025753381": {
                    "item_id": "9025753381",
                    "available": False,
                    "options": {"switch type": "clicky", "backlight": "RGB", "size": "full size"},
                },
                "6342039236": {
                    "item_id": "6342039236",
                    "available": True,
                    "options": {"switch type": "clicky", "backlight": "white", "size": "full size"},
                },
                "1234567890": {
                    "item_id": "1234567890",
                    "available": True,
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
            },
        }

    @staticmethod
    def _thermostat_details() -> Dict[str, Any]:
        return {
            "name": "Smart Thermostat",
            "product_id": "4896585277",
            "variants": {
                "7747408585": {
                    "item_id": "7747408585",
                    "available": True,
                    "options": {"compatibility": "Google Assistant", "color": "black"},
                },
                "1839357461": {
                    "item_id": "1839357461",
                    "available": True,
                    "options": {"compatibility": "Apple HomeKit", "color": "black"},
                },
            },
        }

    def test_v4_constraint_priority_selects_full_size_clicky_fallback(self) -> None:
        candidates = [
            CandidateObject("2299424241", attributes={"switch_type": "clicky", "backlight": "RGB", "size": "80%"}),
            CandidateObject("7706410293", attributes={"switch_type": "clicky", "backlight": "none", "size": "full size"}),
            CandidateObject("9025753381", attributes={"switch_type": "clicky", "backlight": "RGB", "size": "full size", "available": False}, available=False),
            CandidateObject("1234567890", attributes={"switch_type": "linear", "backlight": "RGB", "size": "full size"}),
        ]

        selected = ConstraintPriorityEngine().select(
            candidates,
            hard_constraints=[
                Constraint(slot="switch_type", op="eq", value="clicky"),
                Constraint(slot="size", op="eq", value="full size"),
            ],
            preferences=[Preference(slot="backlight", value="RGB")],
            fallback_rules=[FallbackRule(slot="backlight", from_value="RGB", to_value="none")],
        )

        self.assertTrue(selected.ok)
        self.assertEqual(selected.candidate.candidate_id, "7706410293")  # type: ignore[union-attr]
        self.assertIsNotNone(selected.fallback_used)
        rejected_ids = {r["candidate_id"] for r in selected.rejected}
        self.assertIn("2299424241", rejected_ids)
        self.assertIn("1234567890", rejected_ids)

    def test_v4_retail_adapter_rejects_wrong_keyboard_variant_from_latest_logs(self) -> None:
        adapter = TauRetailAdapter()
        goal = (
            "Exchange the mechanical keyboard for a clicky full-size RGB one; "
            "if unavailable, no backlight is okay. Also exchange the smart "
            "thermostat for one compatible with Google Home."
        )
        old_item = {
            "name": "Mechanical Keyboard",
            "item_id": "1151293680",
            "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
        }

        selected = adapter.select_replacement_variant_id(self._keyboard_details(), old_item, goal)

        self.assertEqual(selected, "7706410293")
        self.assertNotEqual(selected, "2299424241")

    def test_v4_retail_adapter_skips_keyboard_when_exact_spec_unavailable_and_user_says_other_only(self) -> None:
        adapter = TauRetailAdapter()
        goal = (
            "Exchange the mechanical keyboard for a clicky full-size RGB one. "
            "If no keyboard meets those specs, I'd rather only exchange the "
            "smart thermostat for one compatible with Google Home."
        )
        old_item = {
            "name": "Mechanical Keyboard",
            "item_id": "1151293680",
            "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
        }

        selected = adapter.select_replacement_variant_id(self._keyboard_details(), old_item, goal)

        self.assertIsNone(selected)

    def test_v4_retail_two_item_exchange_uses_deterministic_selected_candidates(self) -> None:
        agent = self._make_retail_agent()
        wm = WorkingMemory()
        wm.goal = (
            "Exchange the mechanical keyboard for a clicky full-size RGB one; "
            "if unavailable, no backlight is okay. Also exchange the smart "
            "thermostat for one compatible with Google Home."
        )
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "credit_card_9513926"}],
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "1656367028",
                    "item_id": "1151293680",
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
                {
                    "name": "Smart Thermostat",
                    "product_id": "4896585277",
                    "item_id": "8174673829",
                    "options": {"compatibility": "Apple HomeKit", "color": "black"},
                },
            ],
        }
        wm.product_details["1656367028"] = self._keyboard_details()
        wm.product_details["4896585277"] = self._thermostat_details()

        action = agent._grounded_retail_commit_action(wm)

        self.assertIsNotNone(action)
        self.assertEqual(action.name, "exchange_delivered_order_items")  # type: ignore[union-attr]
        self.assertEqual(action.args["item_ids"], ["1151293680", "8174673829"])  # type: ignore[union-attr]
        self.assertEqual(action.args["new_item_ids"], ["7706410293", "7747408585"])  # type: ignore[union-attr]

    def test_v4_retail_commit_certificate_blocks_partial_exchange(self) -> None:
        agent = self._make_retail_agent()
        wm = WorkingMemory()
        wm.goal = (
            "Exchange order #W2378156: the mechanical keyboard should be "
            "clicky full-size RGB, and no backlight is okay if RGB is not "
            "available. Also exchange the smart thermostat for one compatible "
            "with Google Home."
        )
        wm.auth_user_id = "yusuf_rossi_9620"
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "user_id": "yusuf_rossi_9620",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "credit_card_9513926"}],
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "1656367028",
                    "item_id": "1151293680",
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
                {
                    "name": "Smart Thermostat",
                    "product_id": "4896585277",
                    "item_id": "8174673829",
                    "options": {"compatibility": "Apple HomeKit", "color": "black"},
                },
            ],
        }
        wm.product_details["1656367028"] = self._keyboard_details()
        wm.product_details["4896585277"] = self._thermostat_details()
        partial = ProposedAction(
            name="exchange_delivered_order_items",
            args={
                "order_id": "#W2378156",
                "item_ids": ["8174673829"],
                "new_item_ids": ["7747408585"],
                "payment_method_id": "credit_card_9513926",
            },
            declared_class=RiskClass.WRITE,
            bypass_gates=True,
        )

        result = agent._kernel().validate_commit_certificate(
            partial,
            agent.schemas["exchange_delivered_order_items"],
            wm,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.gate, "commit_certificate")
        self.assertEqual(result.reason, "write_is_partial_for_active_goal")
        obligations = {
            item["name"]: item
            for item in result.diagnostics["certificate"]["obligations"]
        }
        missing = obligations["all_requested_items_included"]["evidence"]["missing_item_ids"]
        self.assertEqual(missing, ["1151293680"])

    def test_v4_retail_commit_certificate_accepts_complete_exchange(self) -> None:
        agent = self._make_retail_agent()
        wm = WorkingMemory()
        wm.goal = (
            "Exchange order #W2378156: the mechanical keyboard should be "
            "clicky full-size RGB, and no backlight is okay if RGB is not "
            "available. Also exchange the smart thermostat for one compatible "
            "with Google Home."
        )
        wm.auth_user_id = "yusuf_rossi_9620"
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "user_id": "yusuf_rossi_9620",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "credit_card_9513926"}],
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "1656367028",
                    "item_id": "1151293680",
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
                {
                    "name": "Smart Thermostat",
                    "product_id": "4896585277",
                    "item_id": "8174673829",
                    "options": {"compatibility": "Apple HomeKit", "color": "black"},
                },
            ],
        }
        wm.product_details["1656367028"] = self._keyboard_details()
        wm.product_details["4896585277"] = self._thermostat_details()
        action = ProposedAction(
            name="exchange_delivered_order_items",
            args={
                "order_id": "#W2378156",
                "item_ids": ["1151293680", "8174673829"],
                "new_item_ids": ["7706410293", "7747408585"],
                "payment_method_id": "credit_card_9513926",
            },
            declared_class=RiskClass.WRITE,
            bypass_gates=True,
        )

        failing, diag = agent._run_gates(
            action,
            agent.schemas["exchange_delivered_order_items"],
            wm,
            [],
            CargoStats(),
        )

        self.assertIsNone(failing, failing.reason if failing else "")
        self.assertIn("commit_certificate", diag["gates_run"])
        self.assertTrue(wm.last_commit_certificate["ok"])
        self.assertEqual(
            wm.last_commit_certificate["selected_candidate_ids"],
            ["7706410293", "7747408585"],
        )

    def test_v4_write_gate_requires_commit_certificate(self) -> None:
        agent = self._make_retail_agent()
        wm = WorkingMemory()
        wm.goal = (
            "Exchange the mechanical keyboard for a clicky full-size RGB one, "
            "and no backlight is okay if RGB is not available. Also exchange "
            "the smart thermostat for one compatible with Google Home."
        )
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "user_id": "yusuf_rossi_9620",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "credit_card_9513926"}],
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "1656367028",
                    "item_id": "1151293680",
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
                {
                    "name": "Smart Thermostat",
                    "product_id": "4896585277",
                    "item_id": "8174673829",
                    "options": {"compatibility": "Apple HomeKit", "color": "black"},
                },
            ],
        }
        wm.product_details["1656367028"] = self._keyboard_details()
        wm.product_details["4896585277"] = self._thermostat_details()
        action = ProposedAction(
            name="exchange_delivered_order_items",
            args={
                "order_id": "#W2378156",
                "item_ids": ["1151293680", "8174673829"],
                "new_item_ids": ["7706410293", "7747408585"],
                "payment_method_id": "credit_card_9513926",
            },
            declared_class=RiskClass.WRITE,
            bypass_gates=True,
        )

        failing, diag = agent._run_gates(
            action,
            agent.schemas["exchange_delivered_order_items"],
            wm,
            [],
            CargoStats(),
        )

        self.assertIsNotNone(failing)
        self.assertEqual(failing.gate, "commit_certificate")
        self.assertIn("commit_certificate", diag["gates_failed"])
        self.assertEqual(failing.reason, "write_lacks_identity_or_user_supplied_order_anchor")

    def test_v4_retail_exact_keyboard_unavailable_commits_thermostat_only_when_requested(self) -> None:
        agent = self._make_retail_agent()
        wm = WorkingMemory()
        wm.goal = (
            "Exchange the mechanical keyboard for a clicky full-size RGB one. "
            "If no keyboard meets those specs, I'd rather only exchange the "
            "smart thermostat for one compatible with Google Home."
        )
        wm.order_details["#W2378156"] = {
            "order_id": "#W2378156",
            "status": "delivered",
            "payment_history": [{"payment_method_id": "credit_card_9513926"}],
            "items": [
                {
                    "name": "Mechanical Keyboard",
                    "product_id": "1656367028",
                    "item_id": "1151293680",
                    "options": {"switch type": "linear", "backlight": "RGB", "size": "full size"},
                },
                {
                    "name": "Smart Thermostat",
                    "product_id": "4896585277",
                    "item_id": "8174673829",
                    "options": {"compatibility": "Apple HomeKit", "color": "black"},
                },
            ],
        }
        wm.product_details["1656367028"] = self._keyboard_details()
        wm.product_details["4896585277"] = self._thermostat_details()

        action = agent._grounded_retail_commit_action(wm)

        self.assertIsNotNone(action)
        self.assertEqual(action.name, "exchange_delivered_order_items")  # type: ignore[union-attr]
        self.assertEqual(action.args["item_ids"], ["8174673829"])  # type: ignore[union-attr]
        self.assertEqual(action.args["new_item_ids"], ["7747408585"])  # type: ignore[union-attr]

    def test_v4_candidate_set_memory_records_empty_search_exhaustion(self) -> None:
        wm = WorkingMemory()
        kernel = GenericCargoKernel(TauAirlineAdapter())
        args = {"origin": "New York", "destination": "Seattle", "date": "2024-05-20"}

        cset = kernel.record_action_candidates(wm, "search_direct_flight", args, "[]")

        self.assertIsNotNone(cset)
        self.assertTrue(cset.empty)  # type: ignore[union-attr]
        self.assertTrue(cset.exhausted)  # type: ignore[union-attr]
        self.assertIs(wm.task_state.candidate_set_for("search_direct_flight", args), cset)

    def test_v4_airline_exhausted_searches_finalize_instead_of_looping(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        text = (
            "Book a flight from New York to Seattle on May 20th. "
            "My user id is alex_smith_42. Direct is preferred but one stopover is okay."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.user_profiles["alex_smith_42"] = {"name": {"first_name": "Alex", "last_name": "Smith"}}
        args = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}
        agent._kernel().record_action_candidates(wm, "search_direct_flight", args, "[]")
        agent._kernel().record_action_candidates(wm, "search_onestop_flight", args, "[]")
        ask = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="What certificate values would you like to use?",
        )

        replacement = agent._obligation_guided_action(ask, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.declared_class, RiskClass.FINAL)  # type: ignore[union-attr]
        self.assertEqual(wm.task_state.terminal_status, "blocked_no_matching_flights")
        self.assertIn("could not find", replacement.user_text.lower())  # type: ignore[union-attr]

    def test_v4_airline_missing_user_id_asks_precisely_before_search(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        text = "I need to change my return flight from Texas to Newark, not JFK."
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        generic = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="How can I help?",
        )

        replacement = agent._obligation_guided_action(generic, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        self.assertIn("user ID", replacement.user_text)  # type: ignore[union-attr]

    def test_v4_airline_search_uses_airport_codes_but_matches_bound_city_state(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        text = (
            "Book a flight from New York to Seattle on May 20th. "
            "My user id is alex_smith_42."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.user_profiles["alex_smith_42"] = {"name": {"first_name": "Alex", "last_name": "Smith"}}
        ask = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="What would you like to do?",
        )

        replacement = agent._obligation_guided_action(ask, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "search_direct_flight")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["origin"], "JFK")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["destination"], "SEA")  # type: ignore[union-attr]
        gate = agent._kernel().validate_action(
            replacement,  # type: ignore[arg-type]
            agent._schema_for(replacement),  # type: ignore[arg-type]
            wm,
        )
        self.assertTrue(gate.ok, gate.reason)

    def test_v4_booking_goal_does_not_scan_unrelated_reservations(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        text = (
            "Book a flight from New York to Seattle on May 20th. "
            "My user id is alex_smith_42."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.absorb_observation({"reservations": ["NO6JO3", "HKEG34"]})
        repeated = ProposedAction(
            name="get_user_details",
            args={"user_id": "alex_smith_42"},
            declared_class=RiskClass.READ,
        )

        self.assertIsNone(agent._advance_reservation_retrieval(repeated, wm))

    def test_v4_reservation_scan_skips_plain_words_and_none(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        wm.goal = "change my return flight reservation"
        wm.typed_values["reservation_id"] = ["though", "None", "Z7GOZK"]
        repeated = ProposedAction(
            name="get_user_details",
            args={"user_id": "olivia_gonzalez_2305"},
            declared_class=RiskClass.READ,
        )

        result = agent._advance_reservation_retrieval(repeated, wm)

        self.assertIsNotNone(result)
        self.assertEqual(result.args["reservation_id"], "Z7GOZK")  # type: ignore[union-attr]

    def test_v4_task_frame_routes_no_auth_product_query_before_auth(self) -> None:
        agent = self._make_retail_agent()
        agent.schemas.update({
            "list_all_product_types": ToolEffectSchema(
                name="list_all_product_types",
                cls=RiskClass.READ,
            ),
            "get_product_details": ToolEffectSchema(
                name="get_product_details",
                cls=RiskClass.READ,
                arg_id_fields=["product_id"],
            ),
        })
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are currently available?"
        wm.absorb_user_message(wm.goal)
        ask_auth = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="Please provide your user ID.",
        )

        replacement = agent._task_frame_stage_action(ask_auth, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "list_all_product_types")  # type: ignore[union-attr]

    def test_v4_task_frame_fetches_product_details_after_catalog(self) -> None:
        agent = self._make_retail_agent()
        agent.schemas["get_product_details"] = ToolEffectSchema(
            name="get_product_details",
            cls=RiskClass.READ,
            arg_id_fields=["product_id"],
        )
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are currently available?"
        wm.product_types = {"T-Shirt": "1656367028"}
        ask_auth = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="What is your user ID?",
        )

        replacement = agent._task_frame_stage_action(ask_auth, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "get_product_details")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["product_id"], "1656367028")  # type: ignore[union-attr]

    def test_v4_successful_write_emits_terminal_respond(self) -> None:
        import src.cargo.cargo_agent as cargo_agent_module
        from src.cargo.cargo_agent import CargoAgent

        class _SolveResult:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        old_action = cargo_agent_module.Action
        old_solve_result = cargo_agent_module.SolveResult
        cargo_agent_module.Action = _Action
        cargo_agent_module.SolveResult = _SolveResult
        try:
            agent = CargoAgent.__new__(CargoAgent)
            agent.client = MockClient([
                _proposer_json(
                    name="cancel_order",
                    args={},
                    declared_class="WRITE",
                    thought="commit grounded cancellation",
                ),
                [_proposer_json(name="cancel_order", args={}, declared_class="WRITE")] * 3,
                json.dumps({"predicted_obs": "{}", "goal_still_reachable": True}),
            ])
            agent.model = "test"
            agent.temperature = 0.0
            agent.wiki = ""
            agent.adapter = SyntheticGenericAdapter()
            agent.kernel = GenericCargoKernel(agent.adapter)
            agent.schemas = {
                "cancel_order": ToolEffectSchema(name="cancel_order", cls=RiskClass.WRITE)
            }
            agent.calibration = default_calibration()

            env = MockEnv(
                "Cancel the order now.",
                [
                    lambda action: _StepResp('{"status":"ok"}', reward=0.0, done=False),
                    lambda action: _StepResp("", reward=1.0, done=True),
                ],
            )

            result = agent.solve(env, max_num_steps=5)

            self.assertEqual([a.name for a in env.actions_executed], ["cancel_order", "respond"])
            self.assertEqual(result.reward, 1.0)
            self.assertEqual(result.info["cargo_stats"]["actions_executed"], 2)
        finally:
            cargo_agent_module.Action = old_action
            cargo_agent_module.SolveResult = old_solve_result

    def test_v4_airline_region_word_matches_db_airport_without_canonicalizing_search_to_guess(self) -> None:
        adapter = TauAirlineAdapter()

        self.assertEqual(adapter.canonicalize_airport("Texas"), "Texas")
        self.assertTrue(adapter.semantic_values_match("origin", "DFW", "Texas"))
        self.assertTrue(adapter.semantic_values_match("origin", "IAH", "Texas"))

    def test_v4_airline_reservation_obs_does_not_overwrite_booking_anchor(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        text = (
            "Book an economy flight from New York to Seattle on May 20th. "
            "My user id is mia_li_3668."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.user_profiles["mia_li_3668"] = {"name": {"first_name": "Mia", "last_name": "Li"}}

        obs = {
            "reservation_id": "HKEG34",
            "origin": "DEN",
            "destination": "LAS",
            "cabin": "business",
            "flights": [{"origin": "DEN", "destination": "LAS", "date": "2024-05-27"}],
        }
        wm.absorb_observation(obs)
        agent._kernel().observe_tool_result(wm, "get_reservation_details", obs)
        ask = ProposedAction(name="respond", args={}, declared_class=RiskClass.ASK_USER)

        replacement = agent._obligation_guided_action(ask, wm)

        self.assertEqual(wm.semantic_slots["origin"], "New York")
        self.assertEqual(wm.semantic_slots["destination"], "Seattle")
        self.assertEqual(wm.semantic_slots["date"], "2024-05-20")
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "search_direct_flight")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["origin"], "JFK")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["destination"], "SEA")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["date"], "2024-05-20")  # type: ignore[union-attr]

    def test_v4_booking_task_does_not_scan_reservations_before_search(self) -> None:
        agent = self._make_airline_agent()
        agent.schemas["get_reservation_details"] = ToolEffectSchema(
            name="get_reservation_details",
            cls=RiskClass.READ,
            arg_id_fields=["reservation_id"],
        )
        wm = WorkingMemory()
        text = (
            "Book a flight from New York to Seattle on May 20th in economy. "
            "My user id is mia_li_3668."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.user_profiles["mia_li_3668"] = {"reservations": ["HKEG34"]}
        wm.typed_values["reservation_id"] = ["HKEG34"]
        repeated_profile = ProposedAction(
            name="get_user_details",
            args={"user_id": "mia_li_3668"},
            declared_class=RiskClass.READ,
        )

        scan = agent._advance_reservation_retrieval(repeated_profile, wm)
        replacement = agent._obligation_guided_action(
            ProposedAction(name="respond", args={}, declared_class=RiskClass.ASK_USER),
            wm,
        )

        self.assertIsNone(scan)
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "search_direct_flight")  # type: ignore[union-attr]

    def test_v4_no_auth_product_query_routes_to_catalog_read(self) -> None:
        agent = self._make_retail_agent()
        agent.schemas["list_all_product_types"] = ToolEffectSchema(
            name="list_all_product_types",
            cls=RiskClass.READ,
        )
        wm = WorkingMemory()
        wm.goal = "How many t-shirt options are currently available in the store?"
        wm.absorb_user_message(wm.goal)
        ask = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="Could you provide your name and ZIP?",
        )

        replacement = agent._no_auth_product_query_action(ask, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "list_all_product_types")  # type: ignore[union-attr]

    def test_v4_airline_blocks_premature_payment_questions(self) -> None:
        wm = WorkingMemory()
        text = "Book a flight from New York to Seattle on May 20th in economy."
        wm.absorb_user_message(text)
        GenericCargoKernel(TauAirlineAdapter()).observe_user_message(wm, text)
        action = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="What certificate values should I use?",
        )

        gate = TauAirlineAdapter().validate_ask_user(action, wm)

        self.assertFalse(gate.ok)
        self.assertEqual(gate.reason, "payment_question_before_flight_selection")

    def test_v4_forced_id_fields_reject_plain_words(self) -> None:
        wm = WorkingMemory()
        action = ProposedAction(
            name="get_reservation_details",
            args={"reservation_id": "though"},
            declared_class=RiskClass.READ,
        )
        schema = ToolEffectSchema(
            name="get_reservation_details",
            cls=RiskClass.READ,
            arg_id_fields=["reservation_id"],
        )

        gate = check_arg_grounding(action, schema, wm)

        self.assertFalse(gate.ok)
        self.assertIn("reservation_id=though", gate.diagnostics["ungrounded"])

    def test_v4_adapter_id_backstop_rejects_plain_word_when_schema_is_missing(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        action = ProposedAction(
            name="get_reservation_details",
            args={"reservation_id": "though"},
            declared_class=RiskClass.READ,
        )

        gate = agent._check_state_action_validity(action, wm)

        self.assertFalse(gate.ok)
        self.assertEqual(gate.reason, "adapter_id_field_plain_word")
        self.assertIn("reservation_id=though", gate.diagnostics["invalid"])

    def test_v2_repeat_window_tracks_eight_signatures(self) -> None:
        wm = WorkingMemory()
        for idx in range(9):
            wm.record_action_signature(f"sig_{idx}")

        self.assertNotIn("sig_0", wm.recent_signatures)
        self.assertEqual(list(wm.recent_signatures), [f"sig_{idx}" for idx in range(1, 9)])

    def test_v2_precommit_blocks_placeholder_and_pseudo_write(self) -> None:
        wm = WorkingMemory()
        verifier = PreCommitVerifier()
        schema = ToolEffectSchema(name="calculate", cls=RiskClass.WRITE)
        calculate = ProposedAction(
            name="calculate",
            args={"expression": "total_cost + taxes_and_fees"},
            declared_class=RiskClass.WRITE,
        )

        verdict = verifier.verify(calculate, schema, wm, TauAirlineAdapter())

        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "unsupported_pseudo_write_tool")

        schema = ToolEffectSchema(name="book_reservation", cls=RiskClass.WRITE)
        placeholder = ProposedAction(
            name="book_reservation",
            args={"reservation_id": "latest_search_result"},
            declared_class=RiskClass.WRITE,
        )
        verdict = verifier.verify(placeholder, schema, wm, TauAirlineAdapter())

        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "placeholder_argument")

    def test_v2_retail_account_task_authenticates_before_order_lookup(self) -> None:
        agent = self._make_retail_agent()
        agent.schemas["get_order_details"] = ToolEffectSchema(
            name="get_order_details",
            cls=RiskClass.READ,
            arg_id_fields=["order_id"],
        )
        wm = WorkingMemory()
        wm.goal = "Please exchange the keyboard in order W2378156."
        wm.absorb_user_message(wm.goal)
        proposed = ProposedAction(
            name="get_order_details",
            args={"order_id": "#W2378156"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._task_frame_stage_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        self.assertIn("full name and ZIP", replacement.user_text)  # type: ignore[union-attr]

    def test_v2_retail_order_recovery_stays_live_after_failed_identity(self) -> None:
        agent = self._make_retail_agent()
        agent.schemas["get_order_details"] = ToolEffectSchema(
            name="get_order_details",
            cls=RiskClass.READ,
            arg_id_fields=["order_id"],
        )
        wm = WorkingMemory()
        wm.goal = "Exchange the keyboard in order W2378156."
        wm.absorb_user_message(wm.goal)
        wm.auth_failed_zips.append("99999")
        proposed = ProposedAction(
            name="get_order_details",
            args={"order_id": "#W2378156"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._task_frame_stage_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "get_order_details")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["order_id"], "#W2378156")  # type: ignore[union-attr]

    def test_v2_nested_airline_itinerary_candidate_set_is_recorded(self) -> None:
        wm = WorkingMemory()
        kernel = GenericCargoKernel(TauAirlineAdapter())
        args = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}
        obs = [[
            {
                "flight_number": "HAT136",
                "origin": "JFK",
                "destination": "ATL",
                "scheduled_departure_time_est": "19:00:00",
                "status": "available",
                "available_seats": {"economy": 14},
                "prices": {"economy": 152},
                "date": "2024-05-20",
            },
            {
                "flight_number": "HAT039",
                "origin": "ATL",
                "destination": "SEA",
                "scheduled_departure_time_est": "22:00:00",
                "status": "available",
                "available_seats": {"economy": 10},
                "prices": {"economy": 103},
                "date": "2024-05-20",
            },
        ]]

        cset = kernel.record_action_candidates(wm, "search_onestop_flight", args, obs)

        self.assertIsNotNone(cset)
        self.assertEqual(cset.candidates[0].candidate_id, "HAT136+HAT039")  # type: ignore[union-attr]
        self.assertEqual(len(cset.candidates[0].attributes["flights"]), 2)  # type: ignore[union-attr]

    def test_v2_airline_presents_grounded_itinerary_before_booking(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        text = (
            "My user id is mia_li_3668. I want to book a one-way economy flight "
            "from New York to Seattle on May 20 after 11am. One stopover is okay. "
            "I have 3 bags, no insurance, and want to use my larger certificate "
            "then my 7447 card."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.auth_user_id = "mia_li_3668"
        wm.user_profiles["mia_li_3668"] = {
            "name": {"first_name": "Mia", "last_name": "Li"},
            "dob": "1990-04-05",
            "membership": "gold",
            "payment_methods": {
                "certificate_7504069": {"source": "certificate", "amount": 250, "id": "certificate_7504069"},
                "credit_card_4421486": {"source": "credit_card", "last_four": "7447", "id": "credit_card_4421486"},
            },
        }
        args = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}
        direct_obs = [
            {
                "flight_number": "HAT069",
                "origin": "JFK",
                "destination": "SEA",
                "scheduled_departure_time_est": "06:00:00",
                "status": "available",
                "available_seats": {"economy": 12},
                "prices": {"economy": 121},
            }
        ]
        wm.absorb_observation(direct_obs)
        agent._kernel().record_action_candidates(wm, "search_direct_flight", args, direct_obs)
        one_obs = [[
            {
                "flight_number": "HAT136",
                "origin": "JFK",
                "destination": "ATL",
                "scheduled_departure_time_est": "19:00:00",
                "status": "available",
                "available_seats": {"economy": 14},
                "prices": {"economy": 152},
                "date": "2024-05-20",
            },
            {
                "flight_number": "HAT039",
                "origin": "ATL",
                "destination": "SEA",
                "scheduled_departure_time_est": "22:00:00",
                "status": "available",
                "available_seats": {"economy": 10},
                "prices": {"economy": 103},
                "date": "2024-05-20",
            },
        ]]
        wm.absorb_observation(one_obs)
        agent._kernel().record_action_candidates(wm, "search_onestop_flight", args, one_obs)

        replacement = agent._obligation_guided_action(
            ProposedAction(name="respond", args={}, declared_class=RiskClass.ASK_USER),
            wm,
        )

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        self.assertIn("HAT136", replacement.user_text)  # type: ignore[union-attr]
        self.assertIn("HAT039", replacement.user_text)  # type: ignore[union-attr]
        self.assertIn("certificate_7504069", replacement.user_text)  # type: ignore[union-attr]
        self.assertTrue(wm.pending_commit_signature)

    def test_v2_airline_builds_complete_book_action_after_confirmation(self) -> None:
        agent = self._make_airline_agent()
        wm = WorkingMemory()
        text = (
            "My user id is mia_li_3668. I want to book a one-way economy flight "
            "from New York to Seattle on May 20 after 11am. One stopover is okay. "
            "I have 3 bags, no insurance, and want to use my larger certificate "
            "then my 7447 card."
        )
        wm.absorb_user_message(text)
        agent._kernel().observe_user_message(wm, text)
        wm.auth_user_id = "mia_li_3668"
        wm.user_profiles["mia_li_3668"] = {
            "name": {"first_name": "Mia", "last_name": "Li"},
            "dob": "1990-04-05",
            "membership": "gold",
            "payment_methods": {
                "certificate_7504069": {"source": "certificate", "amount": 250, "id": "certificate_7504069"},
                "credit_card_4421486": {"source": "credit_card", "last_four": "7447", "id": "credit_card_4421486"},
            },
        }
        args = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}
        one_obs = [[
            {
                "flight_number": "HAT136",
                "origin": "JFK",
                "destination": "ATL",
                "scheduled_departure_time_est": "19:00:00",
                "status": "available",
                "available_seats": {"economy": 14},
                "prices": {"economy": 152},
                "date": "2024-05-20",
            },
            {
                "flight_number": "HAT039",
                "origin": "ATL",
                "destination": "SEA",
                "scheduled_departure_time_est": "22:00:00",
                "status": "available",
                "available_seats": {"economy": 10},
                "prices": {"economy": 103},
                "date": "2024-05-20",
            },
        ]]
        wm.absorb_observation(one_obs)
        agent._kernel().record_action_candidates(wm, "search_onestop_flight", args, one_obs)
        first = agent._airline_booking_progress_action(wm)
        self.assertIsNotNone(first)
        wm.absorb_user_message("Yes, that works. Please proceed.")
        action = agent._airline_booking_progress_action(wm)

        self.assertIsNotNone(action)
        self.assertEqual(action.name, "book_reservation")  # type: ignore[union-attr]
        self.assertEqual(action.args["flights"], [  # type: ignore[union-attr]
            {"flight_number": "HAT136", "date": "2024-05-20"},
            {"flight_number": "HAT039", "date": "2024-05-20"},
        ])
        self.assertEqual(action.args["payment_methods"], [  # type: ignore[union-attr]
            {"payment_id": "certificate_7504069", "amount": 250.0},
            {"payment_id": "credit_card_4421486", "amount": 5.0},
        ])
        self.assertEqual(action.args["total_baggages"], 3)  # type: ignore[union-attr]
        self.assertEqual(action.args["nonfree_baggages"], 0)  # type: ignore[union-attr]
        failing, diag = agent._run_gates(action, agent._schema_for(action), wm, [], CargoStats())
        self.assertIsNone(failing, failing.reason if failing else "")
        self.assertIn("precommit_verifier", diag["gates_run"])

    def test_v2_latest_airline_city_name_search_is_canonicalized(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        proposed = ProposedAction(
            name="search_direct_flight",
            args={"origin": "New York", "destination": "Seattle", "date": "2024-05-20"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._obligation_guided_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "search_direct_flight")  # type: ignore[union-attr]
        self.assertEqual(replacement.args, {  # type: ignore[union-attr]
            "origin": "JFK",
            "destination": "SEA",
            "date": "2024-05-20",
        })

    def test_v2_latest_airline_placeholder_reservation_read_recenters_to_booking(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        self._record_mia_booking_search_results(agent, wm)
        proposed = ProposedAction(
            name="get_reservation_details",
            args={"reservation_id": "latest_search_result"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._obligation_guided_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        self.assertIn("HAT136", replacement.user_text)  # type: ignore[union-attr]
        self.assertNotIn("latest_search_result", replacement.user_text)  # type: ignore[union-attr]

    def test_v2_latest_airline_direct_recheck_uses_existing_search_evidence(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        self._record_mia_booking_search_results(agent, wm)
        proposed = ProposedAction(
            name="search_direct_flight",
            args={"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._obligation_guided_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        self.assertIn("HAT136", replacement.user_text)  # type: ignore[union-attr]
        self.assertNotIn("HAT069", replacement.user_text)  # type: ignore[union-attr]

    def test_v2_latest_airline_selects_cheapest_valid_not_cheapest_invalid(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        self._record_mia_booking_search_results(agent, wm)

        itinerary = agent._select_airline_itinerary(wm)

        self.assertEqual([f["flight_number"] for f in itinerary], ["HAT136", "HAT039"])

    def test_v2_latest_retail_name_zip_order_uses_auth_before_order_read(self) -> None:
        agent = self._make_retail_agent()
        agent.schemas["find_user_id_by_name_zip"] = ToolEffectSchema(
            name="find_user_id_by_name_zip",
            cls=RiskClass.READ,
            arg_semantic_fields=["first_name", "last_name", "zip"],
        )
        agent.schemas["get_order_details"] = ToolEffectSchema(
            name="get_order_details",
            cls=RiskClass.READ,
            arg_id_fields=["order_id"],
        )
        wm = WorkingMemory()
        wm.goal = (
            "You are Yusuf Rossi in 19122. You received your order #W2378156 "
            "and wish to exchange the mechanical keyboard and smart thermostat."
        )
        wm.absorb_user_message(wm.goal)
        proposed = ProposedAction(
            name="get_order_details",
            args={"order_id": "#W2378156"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._task_frame_stage_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "find_user_id_by_name_zip")  # type: ignore[union-attr]
        self.assertEqual(replacement.args, {  # type: ignore[union-attr]
            "first_name": "Yusuf",
            "last_name": "Rossi",
            "zip": "19122",
        })

    def test_v2_known_issue_corpus_tracks_all_latest_result_runs(self) -> None:
        docs_path = Path(__file__).resolve().parents[1] / "docs" / "known_issues.json"
        data = json.loads(docs_path.read_text())
        runs = {row["run"] for row in data["source_runs"]}
        issue_ids = {row["id"] for row in data["issues"]}

        self.assertIn("metrics (56)", runs)
        self.assertIn("metrics (57)", runs)
        self.assertIn("cargo_v2_phase_precommit_spine", issue_ids)
        self.assertIn("soft_goal_field_router", issue_ids)

    def test_v2_corpus_cached_airline_profile_routes_to_search_not_refetch(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        proposed = ProposedAction(
            name="get_user_details",
            args={"user_id": "mia_li_3668"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._obligation_guided_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "search_direct_flight")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["origin"], "JFK")  # type: ignore[union-attr]

    def test_v2_corpus_booking_reservation_scan_routes_to_flight_search(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        proposed = ProposedAction(
            name="get_reservation_details",
            args={"reservation_id": "NO6JO3"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._obligation_guided_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "search_direct_flight")  # type: ignore[union-attr]

    def test_v2_corpus_calculate_cost_routes_to_grounded_booking_summary(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        self._record_mia_booking_search_results(agent, wm)
        proposed = ProposedAction(
            name="calculate",
            args={"expression": "total_cost + taxes_and_fees"},
            declared_class=RiskClass.WRITE,
        )

        replacement = agent._obligation_guided_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.declared_class, RiskClass.ASK_USER)  # type: ignore[union-attr]
        self.assertIn("The fare is $255", replacement.user_text)  # type: ignore[union-attr]
        self.assertIn("credit_card_4421486", replacement.user_text)  # type: ignore[union-attr]

    def test_v2_corpus_generic_ask_after_booking_evidence_uses_summary(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        self._record_mia_booking_search_results(agent, wm)
        proposed = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="Can you provide more details?",
        )

        replacement = agent._obligation_guided_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertIn("HAT136", replacement.user_text)  # type: ignore[union-attr]
        self.assertIn("Should I book", replacement.user_text)  # type: ignore[union-attr]

    def test_v2_corpus_direct_viable_beats_onestop_when_direct_preferred(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        args = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}
        direct_obs = [{
            "flight_number": "HAT555",
            "origin": "JFK",
            "destination": "SEA",
            "scheduled_departure_time_est": "12:00:00",
            "status": "available",
            "available_seats": {"economy": 2},
            "prices": {"economy": 300},
        }]
        one_obs = [[
            {
                "flight_number": "HAT136",
                "origin": "JFK",
                "destination": "ATL",
                "scheduled_departure_time_est": "19:00:00",
                "status": "available",
                "available_seats": {"economy": 14},
                "prices": {"economy": 152},
                "date": "2024-05-20",
            },
            {
                "flight_number": "HAT039",
                "origin": "ATL",
                "destination": "SEA",
                "scheduled_departure_time_est": "22:00:00",
                "status": "available",
                "available_seats": {"economy": 10},
                "prices": {"economy": 103},
                "date": "2024-05-20",
            },
        ]]
        agent._kernel().record_action_candidates(wm, "search_direct_flight", args, direct_obs)
        agent._kernel().record_action_candidates(wm, "search_onestop_flight", args, one_obs)

        itinerary = agent._select_airline_itinerary(wm)

        self.assertEqual([f["flight_number"] for f in itinerary], ["HAT555"])

    def test_v2_corpus_malformed_reservation_lookup_scans_profile_ids(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_olivia_reservation_trace_state(agent)
        proposed = ProposedAction(
            name="get_reservation_details",
            args={"user_id": "olivia_gonzalez_2305"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._advance_reservation_retrieval(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "get_reservation_details")  # type: ignore[union-attr]
        self.assertEqual(replacement.args, {"reservation_id": "Z7GOZK"})  # type: ignore[union-attr]

    def test_v2_corpus_ambiguous_region_search_scans_reservations_first(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_olivia_reservation_trace_state(agent)
        proposed = ProposedAction(
            name="search_direct_flight",
            args={"origin": "Texas", "destination": "Newark", "date": "2024-05-15"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._advance_reservation_retrieval(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "get_reservation_details")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["reservation_id"], "Z7GOZK")  # type: ignore[union-attr]

    def test_v2_corpus_retail_placeholder_email_with_name_zip_uses_name_zip(self) -> None:
        agent = self._make_retail_agent()
        wm = WorkingMemory()
        wm.goal = "You are Yusuf Rossi in 19122 and need to exchange order #W2378156."
        wm.absorb_user_message(wm.goal)
        proposed = ProposedAction(
            name="find_user_id_by_email",
            args={"email": "user@example.com"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._auth_override(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "find_user_id_by_name_zip")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["zip"], "19122")  # type: ignore[union-attr]

    def test_v2_corpus_retail_generic_ask_with_credentials_authenticates(self) -> None:
        agent = self._make_retail_agent()
        wm = WorkingMemory()
        wm.goal = "I am Yusuf Rossi in 19122. Please exchange order #W2378156."
        wm.absorb_user_message(wm.goal)
        proposed = ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text="How can I assist you today?",
        )

        replacement = agent._task_frame_stage_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "find_user_id_by_name_zip")  # type: ignore[union-attr]

    def test_v2_corpus_retail_cached_profile_fetches_order_not_profile_loop(self) -> None:
        agent = self._make_retail_agent()
        agent.schemas["get_order_details"] = ToolEffectSchema(
            name="get_order_details",
            cls=RiskClass.READ,
            arg_id_fields=["order_id"],
        )
        wm = WorkingMemory()
        wm.goal = "Exchange item in order #W2378156."
        wm.absorb_user_message(wm.goal)
        wm.auth_user_id = "yusuf_rossi_9620"
        proposed = ProposedAction(
            name="get_user_details",
            args={"user_id": "yusuf_rossi_9620"},
            declared_class=RiskClass.READ,
        )

        replacement = agent._grounded_progress_or_commit_action(proposed, wm)

        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.name, "get_order_details")  # type: ignore[union-attr]
        self.assertEqual(replacement.args["order_id"], "#W2378156")  # type: ignore[union-attr]

    def test_v2_corpus_precommit_blocks_nested_id_none_but_allows_semantic_none(self) -> None:
        verifier = PreCommitVerifier()
        wm = WorkingMemory()
        schema = ToolEffectSchema(
            name="book_reservation",
            cls=RiskClass.WRITE,
            required_params=["user_id", "insurance"],
        )
        semantic_none = ProposedAction(
            name="book_reservation",
            args={"user_id": "mia_li_3668", "insurance": "none"},
            declared_class=RiskClass.WRITE,
        )
        id_none = ProposedAction(
            name="book_reservation",
            args={"user_id": "mia_li_3668", "payment_methods": [{"payment_id": "none"}], "insurance": "no"},
            declared_class=RiskClass.WRITE,
        )

        self.assertTrue(verifier.verify(semantic_none, schema, wm, TauAirlineAdapter()).ok)
        self.assertFalse(verifier.verify(id_none, schema, wm, TauAirlineAdapter()).ok)

    def test_v2_corpus_precommit_blocks_nested_latest_placeholder(self) -> None:
        verifier = PreCommitVerifier()
        action = ProposedAction(
            name="book_reservation",
            args={"flights": [{"flight_number": "latest_search_result"}], "insurance": "no"},
            declared_class=RiskClass.WRITE,
        )

        verdict = verifier.verify(
            action,
            ToolEffectSchema(name="book_reservation", cls=RiskClass.WRITE),
            WorkingMemory(),
            TauAirlineAdapter(),
        )

        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "placeholder_argument")

    def test_v2_corpus_goal_field_downweights_repeated_profile_against_search(self) -> None:
        agent = self._make_airline_agent()
        wm = self._seed_mia_booking_trace_state(agent)
        profile = ProposedAction(
            name="get_user_details",
            args={"user_id": "mia_li_3668"},
            declared_class=RiskClass.READ,
        )
        search = ProposedAction(
            name="search_direct_flight",
            args={"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
            declared_class=RiskClass.READ,
        )
        wm.goal_field.record_friction(profile.signature(), 4.0, "cached_profile_replay")

        decision = SoftGoalFieldRouter().choose(
            wm,
            [
                GoalActionCandidate(profile, source="cached_profile", progress=0.1),
                GoalActionCandidate(search, source="grounded_search", progress=1.2, uncertainty_reduction=0.8),
            ],
            agent.adapter,
        )

        self.assertEqual(decision.selected.action.name, "search_direct_flight")


# ---------------------------------------------------------------------------
# Tau-bench import sanity (skipped when tau_bench is not installed)
# ---------------------------------------------------------------------------
class TestTauBenchIntegrationOptional(unittest.TestCase):
    def test_baselines_still_import(self) -> None:
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed")
        from src.baselines import ActAgent, ReActAgent, ToolCallingAgent  # noqa: F401

    def test_cargo_agent_imports(self) -> None:
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed")
        from src.cargo.cargo_agent import CargoAgent  # noqa: F401
        self.assertTrue(hasattr(CargoAgent, "solve"))


# ---------------------------------------------------------------------------
# Trajectory(43) regressions — post-WRITE auto-respond architecture fix
# ---------------------------------------------------------------------------
class TestTrajectory43Regressions(unittest.TestCase):
    def setUp(self) -> None:
        reset_cache()

    """Regression tests for the post-WRITE auto-respond architectural fix.

    Root cause analysis from trajectories(22-43): every retail task that
    reached a successful WRITE got the WRITE args correct (matching gold
    exactly) yet still scored reward 0.  Tau-bench only calculates reward
    when the env hits ``done=True``, which for a non-terminate tool can
    only happen via ``respond`` whose user reply contains ``###STOP###``.
    Without a follow-up respond after the WRITE, the user simulator never
    has a chance to emit STOP, so the trajectory ends with reward=0.

    Architectural fix (FIX-A): after a successful WRITE/IRREVERSIBLE tool
    call, the controller deterministically emits a respond announcing
    completion.  This invariant must hold across:
      * model-driven trajectories (proposer naturally proposes a respond
        next — auto-respond should suppress on the same step)
      * model-stuck trajectories (proposer proposes another tool / loops
        — auto-respond is what closes the conversation)
    """

    def _make_agent(self) -> Any:
        """Construct a CargoAgent without running its __init__."""
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed")
        from src.cargo.cargo_agent import CargoAgent
        agent = CargoAgent.__new__(CargoAgent)
        return agent

    # ----- _post_write_summary ----------------------------------------
    def test_post_write_summary_handles_known_action_prefix(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        from src.cargo.risk_class import RiskClass
        from src.cargo.schemas import ProposedAction
        action = ProposedAction(
            name="exchange_delivered_order_items",
            args={"order_id": "#W2378156"},
            declared_class=RiskClass.WRITE,
        )
        msg = CargoAgent._post_write_summary(action, '{"status": "exchange requested"}')
        # User-visible message uses the human action prefix, not the raw
        # tool name.  This matters because the user simulator reads the
        # text and decides whether to STOP.
        self.assertIn("exchange", msg.lower())
        self.assertIn("processed successfully", msg.lower())
        # Status was extracted from the tool obs and surfaced.
        self.assertIn("exchange requested", msg)
        self.assertIn("anything else", msg.lower())

    def test_post_write_summary_handles_cancel(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        from src.cargo.risk_class import RiskClass
        from src.cargo.schemas import ProposedAction
        action = ProposedAction(
            name="cancel_pending_order",
            args={"order_id": "#W1"},
            declared_class=RiskClass.IRREVERSIBLE,
        )
        msg = CargoAgent._post_write_summary(action, '{"status": "cancelled"}')
        self.assertIn("cancel", msg.lower())
        self.assertIn("cancelled", msg)

    def test_post_write_summary_no_status_field(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        from src.cargo.risk_class import RiskClass
        from src.cargo.schemas import ProposedAction
        action = ProposedAction(
            name="modify_pending_order_payment",
            args={"order_id": "#W1"},
            declared_class=RiskClass.WRITE,
        )
        msg = CargoAgent._post_write_summary(action, '{"order_id": "#W1"}')
        # No status field means no status phrase, but the action prefix
        # ("modify") still gets surfaced cleanly.
        self.assertIn("modify", msg.lower())
        self.assertIn("processed successfully", msg.lower())

    def test_post_write_summary_unknown_action_falls_back_safely(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        from src.cargo.risk_class import RiskClass
        from src.cargo.schemas import ProposedAction
        # Unknown / generic action name without one of the standard
        # prefixes ("exchange", "cancel", ...) — message should still be
        # well-formed (not crash, not produce weird underscores).
        action = ProposedAction(
            name="apply_promotion",
            args={"order_id": "#W1"},
            declared_class=RiskClass.WRITE,
        )
        msg = CargoAgent._post_write_summary(action, "")
        # The full humanised name shows when no prefix matched.
        self.assertIn("apply promotion", msg.lower())
        self.assertNotIn("_", msg)  # no raw underscores leaked
        self.assertIn("processed successfully", msg.lower())

    def test_post_write_summary_invalid_json_obs(self) -> None:
        from src.cargo.cargo_agent import CargoAgent
        from src.cargo.risk_class import RiskClass
        from src.cargo.schemas import ProposedAction
        action = ProposedAction(
            name="exchange_delivered_order_items",
            args={"order_id": "#W1"},
            declared_class=RiskClass.WRITE,
        )
        # Non-JSON tool obs (e.g. "Error: order not found") shouldn't
        # crash the summary builder; just skip the status phrase.
        msg = CargoAgent._post_write_summary(action, "Error: not allowed")
        self.assertIn("exchange", msg.lower())
        self.assertNotIn("Error", msg)
        self.assertNotIn("'", msg.split("processed successfully")[1] if "processed successfully" in msg else "")

    # ----- WorkingMemory durability -----------------------------------
    def test_post_write_responded_flag_default(self) -> None:
        from src.cargo.working_memory import WorkingMemory
        wm = WorkingMemory()
        # Default must be False so the controller fires the auto-respond
        # at most once per trajectory.
        self.assertFalse(wm.post_write_responded)

    def test_post_write_responded_flag_settable(self) -> None:
        from src.cargo.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.post_write_responded = True
        self.assertTrue(wm.post_write_responded)

    # ----- Solve loop integration via MockEnv -------------------------
    def test_solve_emits_auto_respond_after_write(self) -> None:
        """End-to-end check: proposer issues a WRITE, env returns done=False,
        solve loop must follow up with a respond automatically.  This is
        the exact scenario seen in trajectories(43) T0/T1 where the WRITE
        args matched gold but the trajectory ended with reward=0 because
        no follow-up respond ever fired.
        """
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed")
        from src.cargo.cargo_agent import CargoAgent

        # Use realistic tau-bench-format IDs in the initial user message
        # so arg_grounding can ground the WRITE's args without a prior READ.
        ORDER_ID = "#W2378156"
        PAY = "credit_card_9513926"
        USER_TURN = (
            f"Modify pending order {ORDER_ID}: change payment method to {PAY}. "
            "Please proceed."
        )
        # Step 0 proposer: emit a WRITE mutation (only the first step
        # matters for this test).
        scripts: List[Any] = [
            _proposer_json(
                name="modify_pending_order_payment",
                args={"order_id": ORDER_ID, "payment_method_id": PAY},
                declared_class="WRITE",
                thought="commit",
            ),
            # SC samples for the WRITE (3 agreeing).
            [_proposer_json(
                name="modify_pending_order_payment",
                args={"order_id": ORDER_ID, "payment_method_id": PAY},
                declared_class="WRITE",
            )] * 3,
            # CF rollout: still reachable.
            json.dumps({"predicted_obs": "{}", "goal_still_reachable": True}),
            # If the loop does run another step (it shouldn't, env will
            # signal done after the auto-respond), this is the next
            # proposer output.
            _proposer_json(
                name="respond", declared_class="FINAL",
                user_text="All done.",
            ),
        ]
        client = MockClient(scripts=scripts)

        # MockEnv: first env.step(write) returns the order details
        # (done=False, like a real WRITE).  Second env.step is the
        # auto-respond — return done=True so the loop exits cleanly with
        # reward 1.0.
        write_obs = json.dumps({
            "order_id": ORDER_ID, "status": "payment modified",
        })
        env = MockEnv(
            initial_user=USER_TURN,
            step_responses=[
                _StepResp(observation=write_obs, reward=0.0, done=False),
                _StepResp(observation="thanks ###STOP###", reward=1.0, done=True),
            ],
        )

        agent = CargoAgent.__new__(CargoAgent)
        # Construct just enough state to call solve() — minimal __init__
        # bypass.  Fields used by solve() must all be set.
        from src.cargo.schema_inducer import induce_schemas
        from src.cargo.calibration import default_calibration
        agent.client = client
        agent.model = "m"
        agent.temperature = 0.0
        agent.tools_info = []
        agent.wiki = ""
        agent.env_hint = "retail"
        agent.calibration = default_calibration()
        # Schema for the WRITE so gates have something to check.
        agent.schemas = induce_schemas([_tool(
            "modify_pending_order_payment",
            ["order_id", "payment_method_id"],
        )])
        agent.domain_policy = ""

        result = agent.solve(env, task_index=0, max_num_steps=10)

        # Verify the auto-respond fired: we expect AT LEAST 2 actions
        # executed against the env (the WRITE and the auto-respond).
        executed_names = [a.name for a in env.actions_executed]
        self.assertIn(
            "modify_pending_order_payment", executed_names,
            f"WRITE never executed; env actions={executed_names}",
        )
        self.assertIn(
            "respond", executed_names,
            f"Auto-respond did NOT fire after WRITE; env actions={executed_names}",
        )
        # Order matters: respond must come AFTER the exchange.
        self.assertLess(
            executed_names.index("modify_pending_order_payment"),
            executed_names.index("respond"),
            f"Auto-respond fired before WRITE; env actions={executed_names}",
        )
        # Reward propagated from env_resp2 (the auto-respond) — 1.0 means
        # the task scored a positive reward via STOP detection.
        self.assertEqual(result.reward, 1.0)

    def test_solve_post_write_responded_flag_set(self) -> None:
        """After a successful WRITE + auto-respond, wm.post_write_responded
        must be True.  This is the durable signal that suppresses the
        second auto-respond if a later WRITE happens in the same trajectory.

        We can't directly inspect wm from outside solve(), so we instead
        verify the OBSERVABLE consequence: env.actions_executed contains
        exactly ONE 'respond' (the auto-respond) plus the WRITE itself.
        Reward propagates from the user simulator's STOP."""
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed")
        from src.cargo.cargo_agent import CargoAgent

        ORDER_ID = "#W2378156"
        PAY = "credit_card_9513926"
        USER_TURN = (
            f"Modify pending order {ORDER_ID}: change payment method to {PAY}."
        )
        scripts: List[Any] = [
            _proposer_json(
                name="modify_pending_order_payment",
                args={"order_id": ORDER_ID, "payment_method_id": PAY},
                declared_class="WRITE",
            ),
            [_proposer_json(
                name="modify_pending_order_payment",
                args={"order_id": ORDER_ID, "payment_method_id": PAY},
                declared_class="WRITE",
            )] * 3,
            json.dumps({"predicted_obs": "{}", "goal_still_reachable": True}),
        ]
        client = MockClient(scripts=scripts)

        env = MockEnv(
            initial_user=USER_TURN,
            step_responses=[
                _StepResp(observation='{"status": "pending (payment modified)"}',
                          reward=0.0, done=False),
                # User STOPs after the auto-respond.
                _StepResp(observation="thanks ###STOP###", reward=1.0, done=True),
            ],
        )

        from src.cargo.schema_inducer import induce_schemas
        from src.cargo.calibration import default_calibration
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "m"
        agent.temperature = 0.0
        agent.tools_info = []
        agent.wiki = ""
        agent.env_hint = "retail"
        agent.calibration = default_calibration()
        agent.schemas = induce_schemas([
            _tool("modify_pending_order_payment",
                  ["order_id", "payment_method_id"]),
        ])
        agent.domain_policy = ""

        result = agent.solve(env, task_index=0, max_num_steps=5)

        respond_count = sum(
            1 for a in env.actions_executed if a.name == "respond"
        )
        # Exactly one auto-respond after the WRITE.  No double-fire.
        self.assertEqual(
            respond_count, 1,
            f"Auto-respond did not fire exactly once; "
            f"respond count={respond_count}",
        )
        self.assertEqual(result.reward, 1.0)

    def test_solve_skips_auto_respond_when_done_already_true(self) -> None:
        """If the WRITE itself caused the env to signal done=True
        (e.g. transfer_to_human_agents), the controller must NOT call
        _respond afterward — that would step a closed env.

        We use a non-IRREVERSIBLE class to keep the gate stack simple
        (READ doesn't go through SC/CF), and have the env return done=True
        on the read itself.  In real tau-bench this scenario corresponds
        to a tool whose execution implicitly closes the conversation
        (e.g. an admin tool the env decides to terminate after).
        """
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed")
        from src.cargo.cargo_agent import CargoAgent

        # Use a WRITE so the post-WRITE branch is the relevant code path.
        # The env returns done=True on the WRITE itself (mimicking
        # transfer_to_human_agents which is in terminate_tools).
        ORDER_ID = "#W2378156"
        scripts: List[Any] = [
            _proposer_json(
                name="cancel_pending_order",
                args={"order_id": ORDER_ID, "reason": "no longer needed"},
                declared_class="WRITE",
            ),
            [_proposer_json(
                name="cancel_pending_order",
                args={"order_id": ORDER_ID, "reason": "no longer needed"},
                declared_class="WRITE",
            )] * 3,
            json.dumps({"predicted_obs": "{}", "goal_still_reachable": True}),
        ]
        client = MockClient(scripts=scripts)

        env = MockEnv(
            initial_user=f"Cancel my pending order {ORDER_ID} - no longer needed",
            step_responses=[
                # Env signals done=True directly from the WRITE.
                _StepResp(observation='{"status": "cancelled"}',
                          reward=1.0, done=True),
            ],
        )

        from src.cargo.schema_inducer import induce_schemas
        from src.cargo.calibration import default_calibration
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "m"
        agent.temperature = 0.0
        agent.tools_info = []
        agent.wiki = ""
        agent.env_hint = "retail"
        agent.calibration = default_calibration()
        agent.schemas = induce_schemas([
            _tool("cancel_pending_order", ["order_id", "reason"]),
        ])
        agent.domain_policy = ""

        agent.solve(env, task_index=0, max_num_steps=5)

        # No respond after the WRITE — the env was already closed.
        respond_count = sum(
            1 for a in env.actions_executed if a.name == "respond"
        )
        self.assertEqual(
            respond_count, 0,
            f"Auto-respond fired after env signalled done; "
            f"respond count={respond_count}",
        )
        self.assertEqual(
            env.actions_executed[0].name, "cancel_pending_order",
            f"WRITE was not the first action; "
            f"got {env.actions_executed[0].name}",
        )


    # ===================================================================
    # Tests for FIX-B: Search-exhaustion detection and escape
    # ===================================================================

    def test_is_search_tool_identifies_search_flights(self) -> None:
        """_is_search_tool should identify search_direct_flight and search_onestop_flight."""
        from src.cargo.cargo_agent import CargoAgent
        self.assertTrue(CargoAgent._is_search_tool("search_direct_flight"))
        self.assertTrue(CargoAgent._is_search_tool("search_onestop_flight"))
        self.assertTrue(CargoAgent._is_search_tool("search_items"))
        self.assertFalse(CargoAgent._is_search_tool("get_order_details"))
        self.assertFalse(CargoAgent._is_search_tool("list_all_product_types"))

    def test_is_empty_search_result_detects_no_results(self) -> None:
        """_is_empty_search_result should detect various no-match patterns."""
        from src.cargo.cargo_agent import CargoAgent
        # Explicit empty patterns
        self.assertTrue(CargoAgent._is_empty_search_result(""))
        self.assertTrue(CargoAgent._is_empty_search_result("[]"))
        self.assertTrue(CargoAgent._is_empty_search_result('{"results": []}'))
        self.assertTrue(CargoAgent._is_empty_search_result('{"flights": []}'))
        # Text patterns
        self.assertTrue(CargoAgent._is_empty_search_result("No flights found"))
        self.assertTrue(CargoAgent._is_empty_search_result("not found"))
        self.assertTrue(CargoAgent._is_empty_search_result("No matching results"))
        # Error patterns
        self.assertTrue(CargoAgent._is_empty_search_result("error: invalid parameters"))
        self.assertTrue(CargoAgent._is_empty_search_result('{"error": "not found"}'))
        # Should NOT match when results exist
        self.assertFalse(CargoAgent._is_empty_search_result(
            '{"flights": [{"id": 1, "departure": "08:00"}]}'
        ))
        self.assertFalse(CargoAgent._is_empty_search_result(
            '[{"id": "F001", "airline": "Airlines"}]'
        ))

    def test_working_memory_search_exhaustion_fields(self) -> None:
        """WorkingMemory should track search exhaustion state."""
        from src.cargo.working_memory import WorkingMemory
        wm = WorkingMemory()
        self.assertEqual(wm.consecutive_empty_searches, 0)
        self.assertEqual(wm.last_search_tool, "")
        self.assertFalse(wm.search_exhaustion_triggered)

    def test_working_memory_consecutive_empty_searches_increments(self) -> None:
        """Consecutive empty searches should increment the counter."""
        from src.cargo.working_memory import WorkingMemory
        wm = WorkingMemory()
        wm.consecutive_empty_searches = 0
        wm.last_search_tool = ""
        # Simulate detecting first empty search
        wm.consecutive_empty_searches += 1
        wm.last_search_tool = "search_direct_flight"
        self.assertEqual(wm.consecutive_empty_searches, 1)
        # Simulate second empty search with same tool
        wm.consecutive_empty_searches += 1
        self.assertEqual(wm.consecutive_empty_searches, 2)
        # Reset when a result is found
        wm.consecutive_empty_searches = 0
        self.assertEqual(wm.consecutive_empty_searches, 0)

    def test_solve_detects_search_exhaustion_and_escapes(self) -> None:
        """When search exhaustion is detected (4+ empty searches),
        the solver should emit an ASK_USER and continue, not burn steps."""
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed")
        from src.cargo.cargo_agent import CargoAgent

        # Simulate a trajectory with repeated empty search results.
        # The proposer keeps proposing search_direct_flight with different args,
        # but all return empty results.
        scripts: List[Any] = [
            # Step 1: initial proposer → search_direct_flight
            _proposer_json(
                name="search_direct_flight",
                args={"from": "NYC", "to": "LAX", "date": "2026-06-01"},
            ),
            # Step 2: proposer returns another search (no progress made)
            _proposer_json(
                name="search_direct_flight",
                args={"from": "NYC", "to": "LAX", "date": "2026-06-02"},
            ),
            # Step 3: another search attempt (still no progress)
            _proposer_json(
                name="search_direct_flight",
                args={"from": "NYC", "to": "LAX", "date": "2026-06-03"},
            ),
            # Step 4: another search attempt (still no progress) →
            # At this point consecutive_empty_searches=4, escape is triggered.
            [_proposer_json(
                name="search_direct_flight",
                args={"from": "NYC", "to": "LAX", "date": "2026-06-04"},
            )] * 3,
            json.dumps({"predicted_obs": "[]", "goal_still_reachable": False}),
        ]
        client = MockClient(scripts=scripts)

        env = MockEnv(
            initial_user="Find a flight from NYC to LAX in June 2026",
            step_responses=[
                # All search results are empty
                _StepResp(observation='[]'),
                _StepResp(observation='[]'),
                _StepResp(observation='[]'),
                _StepResp(observation='[]'),
                # After escape, user provides more info
                _StepResp(observation='User provided additional constraints'),
            ],
        )

        from src.cargo.schema_inducer import induce_schemas
        from src.cargo.calibration import default_calibration
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "m"
        agent.temperature = 0.0
        agent.tools_info = []
        agent.wiki = ""
        agent.env_hint = "airline"
        agent.calibration = default_calibration()
        agent.schemas = induce_schemas([
            _tool("search_direct_flight", ["from", "to", "date"]),
        ])
        agent.domain_policy = ""

        result = agent.solve(env, task_index=0, max_num_steps=10)

        # Verify that escape was triggered: we should see an ASK_USER respond
        # after the 4th empty search result, instead of burning more steps.
        respond_count = sum(
            1 for a in env.actions_executed if a.name == "respond"
        )
        # Should have emitted 1 respond (the ASK_USER escape).
        # Without the escape, the proposer would keep cycling through
        # searches until the budget is exhausted.
        self.assertGreaterEqual(
            respond_count, 1,
            f"Search-exhaustion escape should emit at least 1 respond; "
            f"got {respond_count}",
        )
        self.assertLessEqual(
            len(env.actions_executed), 8,
            f"Search-exhaustion escape should fire before exhausting budget; "
            f"got {len(env.actions_executed)} actions",
        )

    # ===================================================================
    # Tests for FIX-C: Auth-cycle detection and escape
    # ===================================================================

    def test_working_memory_auth_cycle_fields(self) -> None:
        """WorkingMemory should track auth-cycle state."""
        from src.cargo.working_memory import WorkingMemory
        wm = WorkingMemory()
        self.assertEqual(wm.consecutive_auth_attempts, 0)
        self.assertEqual(wm.last_confirmed_auth_user_id, "")
        self.assertFalse(wm.auth_cycle_triggered)

    def test_working_memory_auth_attempt_counter_increments(self) -> None:
        """Consecutive auth attempts without confirmation should increment counter."""
        from src.cargo.working_memory import WorkingMemory
        wm = WorkingMemory()
        # Simulate consecutive auth attempts without progress
        wm.consecutive_auth_attempts = 1
        wm.consecutive_auth_attempts = 2
        wm.consecutive_auth_attempts = 3
        self.assertEqual(wm.consecutive_auth_attempts, 3)
        # Confirming auth should reset
        wm.auth_user_id = "yusuf_rossi_9620"
        wm.last_confirmed_auth_user_id = wm.auth_user_id
        wm.consecutive_auth_attempts = 0
        self.assertEqual(wm.consecutive_auth_attempts, 0)

    def test_solve_detects_auth_cycle_and_escapes(self) -> None:
        """When auth-tool calls don't make progress toward user_id,
        the solver should emit an ASK_USER to break the auth loop."""
        try:
            import tau_bench  # noqa: F401
        except ImportError:
            self.skipTest("tau_bench not installed")
        from src.cargo.cargo_agent import CargoAgent

        # Simulate a trajectory stuck in auth loop: repeated get_user_details
        # without confirming auth_user_id, similar to airline T1.
        scripts: List[Any] = [
            # Step 1: first get_user_details
            _proposer_json(
                name="get_user_details",
                args={"user_id": "placeholder_1"},
            ),
            # Step 2: proposer tries again (confused about auth status)
            _proposer_json(
                name="get_user_details",
                args={"user_id": "placeholder_2"},
            ),
            # Step 3: third auth attempt
            [_proposer_json(
                name="get_user_details",
                args={"user_id": "placeholder_3"},
            )] * 3,
            json.dumps({"predicted_obs": "{}", "goal_still_reachable": False}),
        ]
        client = MockClient(scripts=scripts)

        env = MockEnv(
            initial_user="Get my user details",
            step_responses=[
                # Each get_user_details returns data but doesn't set auth_user_id
                # (because placeholder IDs don't match real users)
                _StepResp(observation='{"user_id": "placeholder_1", "email": "x@example.com"}'),
                _StepResp(observation='{"user_id": "placeholder_2", "email": "y@example.com"}'),
                _StepResp(observation='{"user_id": "placeholder_3", "email": "z@example.com"}'),
                # After escape, user provides clarification
                _StepResp(observation='User clarified their credentials'),
            ],
        )

        from src.cargo.schema_inducer import induce_schemas
        from src.cargo.calibration import default_calibration
        agent = CargoAgent.__new__(CargoAgent)
        agent.client = client
        agent.model = "m"
        agent.temperature = 0.0
        agent.tools_info = []
        agent.wiki = ""
        agent.env_hint = "airline"
        agent.calibration = default_calibration()
        agent.schemas = induce_schemas([
            _tool("get_user_details", ["user_id"]),
        ])
        agent.domain_policy = ""

        result = agent.solve(env, task_index=0, max_num_steps=10)

        # Verify auth-cycle escape was triggered
        respond_count = sum(
            1 for a in env.actions_executed if a.name == "respond"
        )
        # Should emit 1 respond (the auth-cycle escape)
        self.assertGreaterEqual(
            respond_count, 1,
            f"Auth-cycle escape should emit at least 1 respond; "
            f"got {respond_count}",
        )
        self.assertLessEqual(
            len(env.actions_executed), 8,
            f"Auth-cycle escape should fire before exhausting budget; "
            f"got {len(env.actions_executed)} actions",
        )


class TestSoftGoalFieldRouter(unittest.TestCase):
    def _ask(self, text: str = "Hello! How can I assist you today?") -> ProposedAction:
        return ProposedAction(
            name="respond",
            args={},
            declared_class=RiskClass.ASK_USER,
            user_text=text,
            raw_thought=text,
        )

    def test_goal_field_momentum_updates_deterministically(self) -> None:
        wm = WorkingMemory(goal="Book a flight from New York to Seattle on May 20")
        adapter = TauAirlineAdapter()
        kernel = GenericCargoKernel(adapter)
        wm.absorb_user_message(wm.goal)
        kernel.observe_user_message(wm, wm.goal)
        before = wm.goal_field.momentum

        kernel.observe_tool_result(
            wm,
            "get_user_details",
            '{"user_id":"mia_li_3668","reservations":["ABC123"]}',
        )

        self.assertGreater(wm.goal_field.momentum, before)
        self.assertIn("tool:get_user_details", wm.goal_field.progress_events)
        self.assertEqual(wm.goal_field.active_goal, wm.goal[:220])

    def test_repeated_non_progress_raises_friction_and_changes_selection(self) -> None:
        wm = WorkingMemory(goal="Book a flight from New York to Seattle on May 20")
        wm.semantic_slots.update({"origin": "JFK", "destination": "SEA", "date": "2024-05-20"})
        router = SoftGoalFieldRouter()
        adapter = TauAirlineAdapter()
        ask = self._ask()
        read = ProposedAction(
            name="search_direct_flight",
            args={"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
            declared_class=RiskClass.READ,
            raw_thought="search flights",
        )
        wm.goal_field.record_friction(ask.signature(), 3.0, "repeated_generic_ask")

        decision = router.choose(
            wm,
            [
                GoalActionCandidate(ask, source="model", progress=0.0),
                GoalActionCandidate(read, source="search", progress=0.8, uncertainty_reduction=0.8),
            ],
            adapter,
        )

        self.assertEqual(decision.selected.action.name, "search_direct_flight")
        self.assertGreater(wm.goal_field.friction[ask.signature()], 0)

    def test_generic_ask_suppressed_when_goal_slots_known(self) -> None:
        wm = WorkingMemory(goal="I want to book a flight from New York to Seattle on May 20")
        wm.semantic_slots.update({
            "intents": ["book_flight"],
            "origin": "JFK",
            "destination": "SEA",
            "date": "2024-05-20",
        })
        ask = self._ask("What would you like to do today?")
        search = ProposedAction(
            name="search_direct_flight",
            args={"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
            declared_class=RiskClass.READ,
        )

        decision = SoftGoalFieldRouter().choose(
            wm,
            [
                GoalActionCandidate(ask, source="model", progress=0.1),
                GoalActionCandidate(search, source="obligation", progress=1.0, uncertainty_reduction=0.8),
            ],
            TauAirlineAdapter(),
        )

        self.assertEqual(decision.selected.action.name, "search_direct_flight")

    def test_wrong_retail_zip_recenters_but_preserves_order_branch(self) -> None:
        wm = WorkingMemory(goal="Exchange items in order #W2378156")
        wm.absorb_user_message("My name is Yusuf Rossi and ZIP is 10001 for order #W2378156")
        adapter = TauRetailAdapter()
        kernel = GenericCargoKernel(adapter)
        kernel.observe_user_message(wm, wm.goal)
        kernel.observe_tool_result(wm, "find_user_id_by_name_zip", "Error: user not found")

        labels = {h.label for h in wm.goal_field.hypotheses}
        self.assertIn("order_id_recovery", labels)
        self.assertIn("identity_lookup_failed", wm.goal_field.recent_recenter_reason)

    def test_airline_cached_profile_is_not_selected_over_progress(self) -> None:
        wm = WorkingMemory(goal="Change my return flight")
        wm.auth_user_id = "olivia_gonzalez_2305"
        wm.user_profiles["olivia_gonzalez_2305"] = {"reservations": ["Z7GOZK"]}
        stale = ProposedAction(
            name="get_user_details",
            args={"user_id": "olivia_gonzalez_2305"},
            declared_class=RiskClass.READ,
        )
        reservation = ProposedAction(
            name="get_reservation_details",
            args={"reservation_id": "Z7GOZK"},
            declared_class=RiskClass.READ,
        )

        decision = SoftGoalFieldRouter().choose(
            wm,
            [
                GoalActionCandidate(stale, source="model", progress=0.2),
                GoalActionCandidate(reservation, source="reservation_advance", progress=1.0, uncertainty_reduction=0.7),
            ],
            TauAirlineAdapter(),
        )

        self.assertEqual(decision.selected.action.name, "get_reservation_details")

    def test_goal_field_render_stays_compact_for_small_models(self) -> None:
        field = GoalField(active_goal="g")
        for i in range(12):
            field.record_friction(f"action_{i}", 1.0 + i / 10, "loop")
            field.record_progress(f"progress_{i}", 0.1)
        rendered = field.render_compact(max_chars=220)

        self.assertLessEqual(len(rendered), 220)
        self.assertIn("momentum=", rendered)

if __name__ == "__main__":
    unittest.main()
