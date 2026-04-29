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
        ])
        self.assertIn("order_id", schemas["cancel_order"].arg_id_fields)
        self.assertIn("user_id", schemas["update_user_address"].arg_id_fields)
        # 'address' is not an ID field.
        self.assertNotIn("address", schemas["update_user_address"].arg_id_fields)

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
        for i in range(8):
            wm.record_action_signature(f"sig_{i}")
        # Window is 5 by design.
        self.assertEqual(len(list(wm.recent_signatures)), 5)
        self.assertNotIn("sig_0", wm.recent_signatures)
        self.assertIn("sig_7", wm.recent_signatures)

    def test_render_compact_truncates(self) -> None:
        wm = WorkingMemory(goal="g")
        for i in range(50):
            wm._add_db_fact(f"fact_{i}={'x' * 80}")
        text = wm.render_compact(max_chars=600)
        self.assertLessEqual(len(text), 600)


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


if __name__ == "__main__":
    unittest.main()
