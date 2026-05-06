from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rex.agent import RexAgent
from src.rex.ace_loop import run_rex
from src.rex.experience import (
    ExperienceCard,
    ExperienceRetriever,
    audit_card,
    build_experience_bank,
    contains_unredacted_sensitive,
    redact_text,
    render_experience_brief,
)
from src.runners import tau_runner


class _Function:
    def __init__(self, name: str, arguments: Dict[str, Any]) -> None:
        self.name = name
        self.arguments = json.dumps(arguments)


class _ToolCall:
    def __init__(self, name: str, arguments: Dict[str, Any], call_id: str = "call_1") -> None:
        self.id = call_id
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content: str = "", tool_calls: Optional[List[_ToolCall]] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, message: _Message) -> None:
        self.message = message


class _Response:
    def __init__(self, message: _Message) -> None:
        self.choices = [_Choice(message)]


class _Completions:
    def __init__(self, parent: "_Client") -> None:
        self.parent = parent

    def create(self, **kwargs: Any) -> _Response:
        self.parent.calls.append(kwargs)
        if not self.parent.scripts:
            return _Response(_Message("done"))
        nxt = self.parent.scripts.pop(0)
        if callable(nxt):
            nxt = nxt(kwargs)
        if isinstance(nxt, _Message):
            return _Response(nxt)
        return _Response(_Message(str(nxt)))


class _Client:
    def __init__(self, scripts: List[Any]) -> None:
        self.scripts = list(scripts)
        self.calls: List[Dict[str, Any]] = []
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(self)


class _Reset:
    def __init__(self, observation: str) -> None:
        self.observation = observation


class _Step:
    def __init__(self, observation: str, reward: float = 0.0, done: bool = False, info: Optional[Dict[str, Any]] = None) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}


class _Env:
    def __init__(self, initial: str, responses: List[_Step]) -> None:
        self.initial = initial
        self.responses = list(responses)
        self.actions: List[Any] = []
        self.tools_info = [
            {"type": "function", "function": {"name": "get_order_details", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "exchange_delivered_order_items", "parameters": {"type": "object"}}},
        ]
        self.wiki = "Policy: confirm before exchanges."

    def reset(self, task_index: Optional[int] = None) -> _Reset:
        return _Reset(self.initial)

    def step(self, action: Any) -> _Step:
        self.actions.append(action)
        if not self.responses:
            return _Step("", done=True)
        return self.responses.pop(0)


class TestRexExperienceBank(unittest.TestCase):
    def test_redaction_removes_pii_and_ids(self) -> None:
        raw = "alice@example.com #W1234567 yusuf_rossi_9620 credit_card_1234567 ABC123 1151293680"
        redacted = redact_text(raw)
        self.assertNotIn("alice@example.com", redacted)
        self.assertNotIn("#W1234567", redacted)
        self.assertNotIn("yusuf_rossi_9620", redacted)
        self.assertNotIn("credit_card_1234567", redacted)
        self.assertNotIn("1151293680", redacted)
        self.assertFalse(contains_unredacted_sensitive(redacted))

    def test_audit_rejects_unredacted_test_overlap(self) -> None:
        card = ExperienceCard(
            card_id="bad",
            domain="retail",
            source="unit",
            intent="return",
            instruction_template="Return #W1234567",
            needed_evidence=[],
            tool_sequence=["return_delivered_order_items"],
            confirmation="confirm",
            common_trap="none",
        )
        ok, reason = audit_card(card)
        self.assertFalse(ok)
        self.assertIn("unredacted", reason)

    def test_generated_bank_contains_retail_and_policy_cards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest = build_experience_bank(output_dir=Path(td))
            self.assertGreater(manifest["cards_by_domain"].get("retail", 0), 0)
            self.assertGreater(manifest["cards_by_domain"].get("airline", 0), 0)
            text = (Path(td) / "retail.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("@example.com", text)
            self.assertNotRegex(text, r"#W\d+")

    def test_retriever_returns_diverse_similar_examples(self) -> None:
        cards = [
            ExperienceCard("r1", "retail", "unit", "return_items", "return a delivered order", ["order"], ["return_delivered_order_items"], "confirm", "payment"),
            ExperienceCard("r2", "retail", "unit", "exchange_items", "exchange keyboard with clicky switches", ["variants"], ["exchange_delivered_order_items"], "confirm", "constraints"),
            ExperienceCard("a1", "airline", "unit", "book_flight", "book direct flight with bags", ["profile"], ["book_reservation"], "confirm", "route"),
        ]
        out = ExperienceRetriever(cards).search("exchange my keyboard for clicky switches", k=2)
        self.assertEqual(out[0].intent, "exchange_items")
        self.assertEqual(len(out), 2)

    def test_prompt_guard_says_never_copy_ids(self) -> None:
        brief = render_experience_brief([
            ExperienceCard("r1", "retail", "unit", "return_items", "return shape", ["order"], ["return_delivered_order_items"], "confirm", "trap")
        ])
        self.assertIn("Never copy IDs", brief)


class TestRexAgent(unittest.TestCase):
    def test_non_mutating_tool_executes_without_reflection(self) -> None:
        client = _Client([
            "intent: order lookup; first_tool: get_order_details",
            _Message("", [_ToolCall("get_order_details", {"order_id": "#W1"})]),
        ])
        env = _Env("Please check order #W1", [_Step('{"ok": true}', reward=1.0, done=True)])
        agent = RexAgent(env.tools_info, env.wiki, "model", env_hint="retail", client=client, bank_dir=Path(tempfile.gettempdir()) / "missing_rex_bank")
        res = agent.solve(env, task_index=0, max_num_steps=2)
        self.assertEqual(res.reward, 1.0)
        self.assertEqual(env.actions[0].name, "get_order_details")
        self.assertEqual(res.info["rex_stats"]["reflection_calls"], 0)

    def test_mutating_tool_reflection_allow_executes_original_args(self) -> None:
        args = {"order_id": "#W1", "item_ids": ["1"], "new_item_ids": ["2"], "payment_method_id": "credit_card_1"}
        client = _Client([
            "intent: exchange",
            _Message("", [_ToolCall("exchange_delivered_order_items", args)]),
            '{"allow": true, "reason": "confirmed"}',
        ])
        env = _Env("Yes, exchange exactly this item.", [_Step('{"status": "exchange requested"}', reward=1.0, done=True)])
        agent = RexAgent(env.tools_info, env.wiki, "model", env_hint="retail", client=client, bank_dir=Path(tempfile.gettempdir()) / "missing_rex_bank")
        res = agent.solve(env, task_index=0, max_num_steps=2)
        self.assertEqual(env.actions[0].name, "exchange_delivered_order_items")
        self.assertEqual(env.actions[0].kwargs, args)
        self.assertEqual(res.info["rex_stats"]["reflection_calls"], 1)
        self.assertEqual(res.info["rex_stats"]["reflection_blocks"], 0)

    def test_mutating_tool_reflection_block_asks_user_not_write(self) -> None:
        client = _Client([
            "intent: exchange",
            _Message("", [_ToolCall("exchange_delivered_order_items", {"order_id": "#W1"})]),
            '{"allow": false, "reason": "missing confirmation", "ask_user": "Please confirm the exact exchange."}',
        ])
        env = _Env("Exchange this maybe.", [_Step("###STOP###", done=True)])
        agent = RexAgent(env.tools_info, env.wiki, "model", env_hint="retail", client=client, bank_dir=Path(tempfile.gettempdir()) / "missing_rex_bank")
        res = agent.solve(env, task_index=0, max_num_steps=2)
        self.assertEqual(env.actions[0].name, "respond")
        self.assertIn("confirm", env.actions[0].kwargs["content"].lower())
        self.assertEqual(res.info["rex_stats"]["reflection_blocks"], 1)

    def test_startup_analysis_call_has_no_tools(self) -> None:
        client = _Client(["analysis", _Message("done")])
        env = _Env("What can you do?", [_Step("###STOP###", done=True)])
        agent = RexAgent(env.tools_info, env.wiki, "model", env_hint="retail", client=client, bank_dir=Path(tempfile.gettempdir()) / "missing_rex_bank")
        agent.solve(env, task_index=0, max_num_steps=1)
        self.assertNotIn("tools", client.calls[0])
        self.assertIn("tools", client.calls[1])

    def test_retrieved_examples_do_not_force_example_ids_into_tool_args(self) -> None:
        client = _Client([
            "intent: order lookup",
            _Message("", [_ToolCall("get_order_details", {"order_id": "#W9999999"})]),
        ])
        env = _Env("Check order #W9999999", [_Step("{}", done=True)])
        agent = RexAgent(env.tools_info, env.wiki, "model", env_hint="retail", client=client, bank_dir=Path(tempfile.gettempdir()) / "missing_rex_bank")
        agent.solve(env, task_index=0, max_num_steps=2)
        system_prompt = client.calls[1]["messages"][0]["content"]
        self.assertNotRegex(system_prompt, r"#W\d+")
        self.assertEqual(env.actions[0].kwargs["order_id"], "#W9999999")


class TestRexWiring(unittest.TestCase):
    def test_tau_runner_resolves_rex_and_not_cargo(self) -> None:
        self.assertIn("rex", tau_runner.AGENT_CHOICES)
        self.assertNotIn("cargo", tau_runner.AGENT_CHOICES)
        cls = tau_runner._resolve_agent_cls("rex")
        self.assertIs(cls, RexAgent)

    def test_ace_rex_loop_records_tool_calls(self) -> None:
        client = _Client([
            "intent: use tool",
            _Message("", [_ToolCall("lookup", {"x": "1"})]),
            _Message("done"),
        ])
        res = run_rex(
            client=client,
            model="model",
            task={"system": "Use tools."},
            tool_specs=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            user_turn="look up x",
            max_num_steps=2,
            temperature=0.0,
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["tool_calls_made"], ["lookup"])
        self.assertIn("rex_stats", res)

    def test_run_project_dry_run_mentions_rex_not_cargo(self) -> None:
        text = Path("run_project.sh").read_text(encoding="utf-8")
        self.assertIn("baseline,act,react,rex", text)
        self.assertNotIn("baseline,act,react,cargo", text)

    def test_exactly_two_shell_scripts(self) -> None:
        scripts = sorted(str(p) for p in Path(".").glob("*.sh"))
        self.assertEqual(scripts, ["run_project.sh", "setup_env.sh"])


if __name__ == "__main__":
    unittest.main()
