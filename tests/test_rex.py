from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rex.agent import RexAgent
from src.rex.ace_loop import run_rex
from src.rex.experience import (
    DEFAULT_RUNTIME_DIR,
    ExperienceCard,
    ExperienceRetriever,
    audit_card,
    build_experience_bank,
    contains_unredacted_sensitive,
    distill_trajectory,
    load_experience_cards,
    promote_trajectories,
    redact_text,
    render_experience_brief,
)
from src.runners import ace_runner, tau_runner


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


class TestRuntimeMemoryDistillation(unittest.TestCase):
    """Tests for the trajectory-distillation + runtime-memory architecture."""

    def _ok_record(
        self,
        *,
        reward: float = 1.0,
        tools: Optional[List[str]] = None,
        user_text: str = "Please exchange my keyboard.",
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if tools is None:
            tools = ["get_order_details", "exchange_delivered_order_items"]
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": user_text},
        ]
        for i, t in enumerate(tools):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": t, "arguments": "{}"},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "name": t,
                "content": "{}",
            })
        return {
            "task_id": 0,
            "trial": 0,
            "reward": reward,
            "info": info or {},
            "messages": messages,
            "status": "ok",
            "error": "",
        }

    def test_distill_success_returns_card_with_outcome_tag(self) -> None:
        rec = self._ok_record(reward=1.0)
        card = distill_trajectory(rec, domain="retail", idx=0)
        self.assertIsNotNone(card)
        self.assertEqual(card.domain, "retail")
        self.assertTrue(card.intent.endswith("__success"))
        self.assertIn("exchange_delivered_order_items", card.tool_sequence)
        self.assertEqual(card.source, "runtime_successful")

    def test_distill_partial_returns_card_tagged_partial(self) -> None:
        rec = self._ok_record(reward=0.5)
        card = distill_trajectory(rec, domain="retail")
        self.assertIsNotNone(card)
        self.assertTrue(card.intent.endswith("__partial"))
        self.assertEqual(card.source, "runtime_partial")

    def test_distill_failure_returns_avoid_card_with_error_trap(self) -> None:
        rec = self._ok_record(reward=0.0, info={"error": "exchange_args_invalid"})
        card = distill_trajectory(rec, domain="retail")
        self.assertIsNotNone(card)
        self.assertTrue(card.intent.endswith("__avoid"))
        # The error message becomes the trap so future runs don't repeat it
        self.assertIn("exchange_args_invalid", card.common_trap)

    def test_distill_skips_record_with_no_tool_calls(self) -> None:
        rec = self._ok_record(tools=[])
        self.assertIsNone(distill_trajectory(rec, domain="retail"))

    def test_distill_skips_record_with_controller_error(self) -> None:
        rec = self._ok_record(reward=1.0)
        rec["status"] = "error"
        rec["info"] = {"error": "boom"}
        self.assertIsNone(distill_trajectory(rec, domain="retail"))

    def test_distill_redacts_sensitive_strings_in_user_text(self) -> None:
        rec = self._ok_record(
            user_text="Exchange order #W1234567 paid via credit_card_1234567",
        )
        card = distill_trajectory(rec, domain="retail")
        self.assertIsNotNone(card)
        self.assertNotIn("#W1234567", card.instruction_template)
        self.assertNotIn("credit_card_1234567", card.instruction_template)

    def test_promote_trajectories_writes_to_runtime_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rdir = Path(td)
            recs = [
                self._ok_record(reward=1.0),
                self._ok_record(reward=0.5, tools=["get_order_details"]),
            ]
            manifest = promote_trajectories(recs, domain="retail", runtime_dir=rdir)
            self.assertEqual(manifest["promoted"], 2)
            self.assertTrue((rdir / "retail.jsonl").exists())
            written = (rdir / "retail.jsonl").read_text(encoding="utf-8")
            self.assertIn("runtime_successful", written)
            self.assertIn("runtime_partial", written)

    def test_promote_dedupes_repeat_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rdir = Path(td)
            recs = [self._ok_record(reward=1.0)]
            m1 = promote_trajectories(recs, domain="retail", runtime_dir=rdir)
            m2 = promote_trajectories(recs, domain="retail", runtime_dir=rdir)
            self.assertEqual(m1["promoted"], 1)
            self.assertEqual(m2["promoted"], 0)
            self.assertEqual(m2["total_runtime_cards_after"], 1)

    def test_promote_blocks_test_split_records_in_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rdir = Path(td)
            rec = self._ok_record(reward=1.0, info={"task_split": "test"})
            manifest = promote_trajectories(
                [rec], domain="retail", runtime_dir=rdir, allow_test_split=False,
            )
            self.assertEqual(manifest["promoted"], 0)
            self.assertTrue(any(r.get("reason") == "test_split_blocked" for r in manifest["rejected"]))

    def test_load_combines_seed_and_runtime_cards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bank_dir = Path(td) / "seed"
            runtime_dir = Path(td) / "runtime"
            build_experience_bank(output_dir=bank_dir)
            seed_only = load_experience_cards("retail", bank_dir, runtime_dir=runtime_dir)
            recs = [self._ok_record(reward=1.0, tools=["get_user_details", "get_order_details"])]
            promote_trajectories(recs, domain="retail", runtime_dir=runtime_dir)
            with_runtime = load_experience_cards("retail", bank_dir, runtime_dir=runtime_dir)
            self.assertGreater(len(with_runtime), len(seed_only))
            self.assertTrue(any(c.source.startswith("runtime_") for c in with_runtime))

    def test_load_dedupes_overlap_between_seed_and_runtime(self) -> None:
        # Two cards with identical (intent, tool_sequence, common_trap) should
        # collapse to one after _dedupe_cards.
        from src.rex.experience import _dedupe_cards
        c1 = ExperienceCard("a", "retail", "seed", "i", "tmpl", ["e"], ["t1"], "conf", "trap")
        c2 = ExperienceCard("b", "retail", "runtime", "i", "tmpl", ["e"], ["t1"], "conf", "trap")
        out = _dedupe_cards([c1, c2])
        self.assertEqual(len(out), 1)


class TestStatefulRetrieval(unittest.TestCase):
    """The agent must re-retrieve based on evolving conversation state."""

    def test_query_includes_latest_tool_observation(self) -> None:
        agent = RexAgent(
            tools_info=[],
            wiki="",
            model="m",
            env_hint="retail",
            client=_Client([]),
            bank_dir=Path(tempfile.gettempdir()) / "missing_rex_bank_q",
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Cancel my order."},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "name": "get_order_details", "content": "{\"status\": \"pending\"}"},
            {"role": "user", "content": "yes please cancel"},
        ]
        query = agent._retrieval_query("Cancel my order.", messages)
        self.assertIn("Cancel my order.", query)
        self.assertIn("get_order_details", query)
        self.assertIn("pending", query)

    def test_solve_refreshes_brief_after_threshold(self) -> None:
        # Three model turns: assistant tool_call, assistant tool_call,
        # final assistant respond. With refresh_every=1 the brief must be
        # rebuilt at least once mid-trajectory.
        os_env_prev = os.environ.get("REX_RETRIEVAL_REFRESH_EVERY")
        os.environ["REX_RETRIEVAL_REFRESH_EVERY"] = "1"
        try:
            client = _Client([
                "intent: order lookup",
                _Message("", [_ToolCall("get_order_details", {"order_id": "#W1"})]),
                _Message("", [_ToolCall("get_order_details", {"order_id": "#W2"})]),
                _Message("done"),
            ])
            env = _Env("Look up two orders.", [
                _Step('{"status": "delivered"}'),
                _Step('{"status": "pending"}'),
                _Step("###STOP###", reward=1.0, done=True),
            ])
            agent = RexAgent(
                env.tools_info, env.wiki, "model",
                env_hint="retail", client=client,
                bank_dir=Path(tempfile.gettempdir()) / "missing_rex_bank_r",
            )
            res = agent.solve(env, task_index=0, max_num_steps=5)
            self.assertGreaterEqual(res.info["rex_stats"]["retrieval_refreshes"], 1)
        finally:
            if os_env_prev is None:
                os.environ.pop("REX_RETRIEVAL_REFRESH_EVERY", None)
            else:
                os.environ["REX_RETRIEVAL_REFRESH_EVERY"] = os_env_prev

    def test_solve_with_refresh_disabled_does_not_refresh(self) -> None:
        os_env_prev = os.environ.get("REX_RETRIEVAL_REFRESH_EVERY")
        os.environ["REX_RETRIEVAL_REFRESH_EVERY"] = "0"
        try:
            client = _Client([
                "intent: lookup",
                _Message("", [_ToolCall("get_order_details", {"order_id": "#W1"})]),
                _Message("done"),
            ])
            env = _Env("Look up.", [
                _Step('{"status": "delivered"}'),
                _Step("###STOP###", reward=1.0, done=True),
            ])
            agent = RexAgent(
                env.tools_info, env.wiki, "model",
                env_hint="retail", client=client,
                bank_dir=Path(tempfile.gettempdir()) / "missing_rex_bank_r2",
            )
            res = agent.solve(env, task_index=0, max_num_steps=5)
            self.assertEqual(res.info["rex_stats"]["retrieval_refreshes"], 0)
        finally:
            if os_env_prev is None:
                os.environ.pop("REX_RETRIEVAL_REFRESH_EVERY", None)
            else:
                os.environ["REX_RETRIEVAL_REFRESH_EVERY"] = os_env_prev


class TestRunnerPromotionWiring(unittest.TestCase):
    """The runners must promote trajectories at the end of a run."""

    def test_tau_runner_should_promote_default_dev_split(self) -> None:
        ns = argparse.Namespace(promote_runtime_memory="auto", task_split="train")
        self.assertTrue(tau_runner._should_promote(ns))

    def test_tau_runner_should_not_promote_default_test_split(self) -> None:
        ns = argparse.Namespace(promote_runtime_memory="auto", task_split="test")
        self.assertFalse(tau_runner._should_promote(ns))

    def test_tau_runner_always_promotes_when_forced(self) -> None:
        ns = argparse.Namespace(promote_runtime_memory="always", task_split="test")
        self.assertTrue(tau_runner._should_promote(ns))

    def test_tau_runner_never_promotes_when_disabled(self) -> None:
        ns = argparse.Namespace(promote_runtime_memory="never", task_split="train")
        self.assertFalse(tau_runner._should_promote(ns))

    def test_tau_runner_promote_writes_to_runtime_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rdir = Path(td)
            ns = argparse.Namespace(
                promote_runtime_memory="auto",
                task_split="train",
                env="retail",
                runtime_dir=str(rdir),
            )
            recs = [
                {
                    "task_id": 0,
                    "trial": 0,
                    "reward": 1.0,
                    "status": "ok",
                    "info": {},
                    "messages": [
                        {"role": "user", "content": "Cancel my order."},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": "1",
                                "type": "function",
                                "function": {"name": "cancel_pending_order", "arguments": "{}"},
                            }],
                        },
                        {"role": "tool", "name": "cancel_pending_order", "content": "{}"},
                    ],
                }
            ]
            manifest = tau_runner._promote_records_to_runtime(ns, recs)
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["promoted"], 1)
            self.assertTrue((rdir / "retail.jsonl").exists())

    def test_ace_runner_promotion_uses_ace_domain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rdir = Path(td)
            ns = argparse.Namespace(
                promote_runtime_memory="auto",
                runtime_dir=str(rdir),
                agent="rex",
            )
            recs = [
                {
                    "index": 0,
                    "controller": "rex",
                    "status": "ok",
                    "actual_tools": ["lookup_user", "fetch_data"],
                    "tool_coverage": 1.0,
                    "messages": [
                        {"role": "user", "content": "Look up the user."},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": "1",
                                "type": "function",
                                "function": {"name": "lookup_user", "arguments": "{}"},
                            }],
                        },
                        {"role": "tool", "name": "lookup_user", "content": "{}"},
                    ],
                }
            ]
            manifest = ace_runner._promote_ace_records(ns, recs)
            self.assertEqual(manifest["status"], "ok")
            self.assertGreaterEqual(manifest["promoted"], 1)
            self.assertTrue((rdir / "ace.jsonl").exists())


class TestRuntimeMemoryEnvIntegration(unittest.TestCase):
    """Environment-variable wiring across runner and agent."""

    def test_rex_agent_uses_runtime_memory_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bank_dir = Path(td) / "seed"
            runtime_dir = Path(td) / "runtime"
            build_experience_bank(output_dir=bank_dir)
            # Promote a card so runtime memory is non-empty
            rec = {
                "task_id": 0,
                "trial": 0,
                "reward": 1.0,
                "status": "ok",
                "info": {},
                "messages": [
                    {"role": "user", "content": "Refund my returned order."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "1",
                            "type": "function",
                            "function": {"name": "return_delivered_order_items", "arguments": "{}"},
                        }],
                    },
                    {"role": "tool", "name": "return_delivered_order_items", "content": "{}"},
                ],
            }
            promote_trajectories([rec], domain="retail", runtime_dir=runtime_dir)
            client = _Client(["analysis", _Message("done")])
            env = _Env("Anything to do?", [_Step("###STOP###", done=True)])
            agent = RexAgent(
                env.tools_info, env.wiki, "model",
                env_hint="retail", client=client,
                bank_dir=bank_dir, runtime_dir=runtime_dir,
            )
            self.assertGreater(len(agent.cards), 0)
            runtime_card_ids = [c.card_id for c in agent.cards if c.source.startswith("runtime_")]
            self.assertGreaterEqual(len(runtime_card_ids), 1)


if __name__ == "__main__":
    unittest.main()
