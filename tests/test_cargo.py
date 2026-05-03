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
        """User typed yusuf.rossi@example.com — must NOT be used as a real email."""
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange keyboard"
        wm.absorb_user_message("My email is yusuf.rossi@example.com")
        result = agent._auth_override(self._placeholder(), wm)
        self.assertIsNotNone(result)
        # Must NOT propose find_user_id_by_email with the placeholder
        if result.name == "find_user_id_by_email":
            self.assertFalse(
                "@example." in result.args.get("email", ""),
                f"Placeholder email reused: {result.args}",
            )

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
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange the mechanical keyboard"
        wm.auth_failed_zips = ["12345"]
        wm.absorb_user_message("My full name is Yusuf Rossi and ZIP 12345.")
        wm.absorb_user_message("My email is yusuf.rossi@example.com.")
        wm.absorb_user_message("My order number is #W2378156.")
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
        agent = self._make_agent()
        wm = WorkingMemory()
        wm.goal = "exchange mechanical keyboard"
        wm.auth_failed_zips = ["12345"]
        wm.auth_ask_count = 2
        wm.absorb_user_message("My full name is Yusuf Rossi")
        wm.absorb_user_message("My ZIP is 12345")
        wm.absorb_user_message("My email is yusuf.rossi@example.com")
        # Model proposes the placeholder again — override must catch it
        result = agent._auth_override(
            self._placeholder("yusuf.rossi@example.com"), wm
        )
        self.assertIsNotNone(result)
        # Must NOT issue the placeholder email lookup
        if result.name == "find_user_id_by_email":
            self.assertFalse(
                "@example." in result.args.get("email", ""),
                f"Looped on placeholder: {result.args}",
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
