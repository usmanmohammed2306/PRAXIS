"""Tests for the continual procedural-memory architecture.

These cover the new modules added on top of the legacy ``experience.py``:

  * config.py
  * memory_types.py
  * sanitize.py
  * trajectory_parser.py
  * embeddings.py
  * retrieval.py
  * distill.py
  * playbook.py
  * memory_store.py
  * memory_quality.py
  * working_state.py
  * retrieval_logging.py
  * pipeline.py

The legacy tests in ``test_rex.py`` continue to validate
backward-compatible surface area.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestRexConfig(unittest.TestCase):
    def test_from_env_uses_defaults_when_unset(self) -> None:
        from src.rex.config import RexConfig, reset_default_config

        reset_default_config()
        cfg = RexConfig.from_env()
        self.assertEqual(cfg.top_k, 3)
        self.assertEqual(cfg.retrieval_refresh_every, 2)
        self.assertEqual(cfg.embedding_backend, "tfidf")
        self.assertTrue(cfg.promotion_block_test_split)

    def test_from_env_picks_up_overrides(self) -> None:
        from src.rex.config import RexConfig

        prev_top = os.environ.get("REX_TOP_K")
        prev_backend = os.environ.get("REX_EMBEDDING_BACKEND")
        os.environ["REX_TOP_K"] = "7"
        os.environ["REX_EMBEDDING_BACKEND"] = "tfidf"
        try:
            cfg = RexConfig.from_env()
            self.assertEqual(cfg.top_k, 7)
            self.assertEqual(cfg.embedding_backend, "tfidf")
        finally:
            if prev_top is None:
                os.environ.pop("REX_TOP_K", None)
            else:
                os.environ["REX_TOP_K"] = prev_top
            if prev_backend is None:
                os.environ.pop("REX_EMBEDDING_BACKEND", None)
            else:
                os.environ["REX_EMBEDDING_BACKEND"] = prev_backend

    def test_with_overrides_returns_new_instance(self) -> None:
        from src.rex.config import RexConfig

        cfg = RexConfig.from_env()
        new_cfg = cfg.with_overrides(top_k=99)
        self.assertEqual(new_cfg.top_k, 99)
        self.assertEqual(cfg.top_k, 3)


# ---------------------------------------------------------------------------
# Memory types
# ---------------------------------------------------------------------------
class TestMemoryTypes(unittest.TestCase):
    def test_process_memory_card_round_trip_via_dict(self) -> None:
        from src.rex.memory_types import ProcessMemoryCard

        card = ProcessMemoryCard(
            card_id="x", timestamp_ms=1, benchmark="tau-bench", environment="retail",
            controller_source="rex", task_category="exchange__success", source="runtime_successful",
            procedural_summary="exchange flow", instruction_template="Exchange item",
            successful_patterns=["did the right thing"], failure_patterns=["picked wrong variant"],
            recovery_heuristics=["retry with constraints"], recommended_next_tools=["get_order_details"],
            forbidden_behaviors=["no IDs"], required_evidence=["product variants"],
            tool_ordering_hints=["get_order_details", "exchange_delivered_order_items"],
            confirmation="confirm exact write", common_trap="constraints first",
            retrieval_text="", confidence=0.9,
        )
        data = card.to_dict()
        self.assertEqual(data["card_id"], "x")
        self.assertIn("retrieval_text", data)
        round_trip = ProcessMemoryCard.from_dict(data)
        self.assertEqual(round_trip.card_id, "x")
        self.assertEqual(round_trip.outcome, "seed")

    def test_lift_legacy_card(self) -> None:
        from src.rex.experience import ExperienceCard
        from src.rex.memory_types import from_experience_card, to_experience_card_kwargs

        legacy = ExperienceCard(
            card_id="abc", domain="retail", source="seed", intent="return_items",
            instruction_template="return one item", needed_evidence=["order"],
            tool_sequence=["return_delivered_order_items"], confirmation="confirm",
            common_trap="trap",
        )
        pmc = from_experience_card(legacy)
        self.assertEqual(pmc.card_id, "abc")
        self.assertEqual(pmc.environment, "retail")
        self.assertIn("return_delivered_order_items", pmc.tool_ordering_hints)
        kwargs = to_experience_card_kwargs(pmc)
        self.assertEqual(kwargs["domain"], "retail")
        self.assertEqual(kwargs["intent"], "return_items")

    def test_signature_dedup_key_changes_with_environment(self) -> None:
        from src.rex.memory_types import ProcessMemoryCard

        a = ProcessMemoryCard.from_dict({
            "card_id": "1", "environment": "retail", "task_category": "x",
            "tool_ordering_hints": ["a", "b"], "common_trap": "trap",
        })
        b = ProcessMemoryCard.from_dict({
            "card_id": "2", "environment": "airline", "task_category": "x",
            "tool_ordering_hints": ["a", "b"], "common_trap": "trap",
        })
        self.assertNotEqual(a.signature(), b.signature())


# ---------------------------------------------------------------------------
# Sanitize
# ---------------------------------------------------------------------------
class TestSanitize(unittest.TestCase):
    def test_sanitize_text_redacts_email_and_ids(self) -> None:
        from src.rex.sanitize import contains_unredacted_sensitive, sanitize_text

        s = sanitize_text("alice@example.com #W1234567 yusuf_rossi_9620 1151293680")
        self.assertNotIn("alice@example.com", s)
        self.assertNotIn("#W1234567", s)
        self.assertNotIn("yusuf_rossi_9620", s)
        self.assertNotIn("1151293680", s)
        self.assertFalse(contains_unredacted_sensitive(s))

    def test_sanitize_dict_recurses(self) -> None:
        from src.rex.sanitize import sanitize_dict

        out = sanitize_dict({"k": "alice@example.com", "nested": {"v": "#W1234567"}})
        self.assertNotIn("alice", json.dumps(out))
        self.assertNotIn("#W1234567", json.dumps(out))

    def test_audit_text_returns_reason(self) -> None:
        from src.rex.sanitize import audit_text

        ok, reason = audit_text("contains alice@example.com")
        self.assertFalse(ok)
        self.assertIn("unredacted", reason)

    def test_compress_repeated_phrases(self) -> None:
        from src.rex.sanitize import compress_repeated_phrases

        out = compress_repeated_phrases([
            "do not copy ids",
            "Do Not Copy IDS",  # case-normalized duplicate of the first
            "use real evidence",
            "use real evidence",
            "use real evidence",
            "verify before mutating",  # unique
        ])
        # Default max_repeats=1 keeps the first occurrence of each
        # normalized phrase.
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "do not copy ids")


# ---------------------------------------------------------------------------
# Trajectory parser
# ---------------------------------------------------------------------------
def _trajectory_record(
    *,
    reward: float = 1.0,
    tools: List[str] = None,
    user_text: str = "Please cancel my order #W1234567.",
    error: str = "",
    status: str = "ok",
    info: Dict[str, Any] = None,
    benchmark: str = "tau-bench",
    env: str = "retail",
) -> Dict[str, Any]:
    if tools is None:
        tools = ["get_order_details", "cancel_pending_order"]
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": user_text},
    ]
    for i, t in enumerate(tools):
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": t, "arguments": "{}"},
            }],
        })
        messages.append({"role": "tool", "tool_call_id": f"call_{i}", "name": t, "content": "{}"})
    return {
        "task_id": 0, "trial": 0, "reward": reward, "messages": messages,
        "status": status, "error": error,
        "info": dict(info or {}, benchmark=benchmark, env=env, controller="rex"),
    }


class TestTrajectoryParser(unittest.TestCase):
    def test_parse_record_extracts_tool_calls(self) -> None:
        from src.rex.trajectory_parser import parse_record

        rec = _trajectory_record(tools=["get_order_details", "cancel_pending_order"])
        s = parse_record(rec)
        self.assertIsNotNone(s)
        self.assertEqual(s.tool_calls, ["get_order_details", "cancel_pending_order"])
        self.assertEqual(s.environment, "retail")
        self.assertEqual(s.benchmark, "tau-bench")
        self.assertEqual(s.controller, "rex")
        self.assertNotIn("#W1234567", s.initial_user)

    def test_parse_record_handles_error_records(self) -> None:
        from src.rex.trajectory_parser import parse_record

        rec = _trajectory_record(reward=0.0, error="boom", status="error")
        s = parse_record(rec)
        self.assertEqual(s.status, "error")
        self.assertIn("boom", s.error)

    def test_iter_jsonl_skips_malformed_lines(self) -> None:
        from src.rex.trajectory_parser import iter_jsonl_records

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "traj.jsonl"
            p.write_text(
                json.dumps({"task_id": 1, "messages": []}) + "\n"
                + "garbage line\n"
                + json.dumps({"task_id": 2, "messages": []}) + "\n",
                encoding="utf-8",
            )
            records = list(iter_jsonl_records(p))
            self.assertEqual(len(records), 2)

    def test_parse_jsonl_path_streams(self) -> None:
        from src.rex.trajectory_parser import parse_jsonl_path

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "traj.jsonl"
            with p.open("w", encoding="utf-8") as f:
                f.write(json.dumps(_trajectory_record()) + "\n")
                f.write(json.dumps(_trajectory_record(reward=0.0, error="boom")) + "\n")
            summaries = list(parse_jsonl_path(p))
            self.assertEqual(len(summaries), 2)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class TestEmbeddings(unittest.TestCase):
    def test_tfidf_embedder_is_deterministic(self) -> None:
        from src.rex.embeddings import HashedTfidfEmbedder

        e = HashedTfidfEmbedder(dim=64).fit(["alpha beta gamma", "alpha delta epsilon"])
        v1 = e.encode("alpha beta")
        v2 = e.encode("alpha beta")
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), 64)

    def test_cosine_similarity_handles_empty(self) -> None:
        from src.rex.embeddings import cosine_similarity

        self.assertEqual(cosine_similarity([], []), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_embedding_cache_persists(self) -> None:
        from src.rex.embeddings import (
            CachedEmbedder,
            EmbeddingCache,
            HashedTfidfEmbedder,
        )

        with tempfile.TemporaryDirectory() as td:
            cache = EmbeddingCache(cache_dir=Path(td), backend_name="tfidf", model_name="hashed_64")
            emb = CachedEmbedder(backend=HashedTfidfEmbedder(dim=64), cache=cache).fit([
                "alpha beta", "gamma delta",
            ])
            v1 = emb.encode("alpha beta")
            emb.flush()
            # New cache reads the persisted vector.
            cache2 = EmbeddingCache(cache_dir=Path(td), backend_name="tfidf", model_name="hashed_64")
            self.assertIsNotNone(cache2.get("alpha beta"))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _process_card(env: str, intent: str, sequence: List[str], trap: str = "trap"):
    from src.rex.memory_types import ProcessMemoryCard

    return ProcessMemoryCard.from_dict({
        "card_id": f"{env}-{intent}",
        "environment": env,
        "task_category": intent,
        "tool_ordering_hints": list(sequence),
        "recommended_next_tools": list(sequence),
        "common_trap": trap,
        "instruction_template": " ".join(sequence) + " " + intent,
        "procedural_summary": f"{intent}: {' -> '.join(sequence)}",
        "required_evidence": ["evidence"],
        "confirmation": "ask before write",
    })


class TestHybridRetriever(unittest.TestCase):
    def test_search_returns_top_k_and_diverse(self) -> None:
        from src.rex.retrieval import HybridRetriever, build_query

        cards = [
            _process_card("retail", "exchange__success", ["get_order_details", "exchange_delivered_order_items"]),
            _process_card("retail", "return__success", ["get_order_details", "return_delivered_order_items"]),
            _process_card("airline", "book__success", ["search_direct_flight", "book_reservation"]),
        ]
        retriever = HybridRetriever(cards)
        q = build_query(initial_user="exchange my keyboard", messages=[], environment="retail")
        result = retriever.search(q)
        self.assertGreater(len(result.cards), 0)
        # Environment-matched card should rank above the airline one.
        envs = [c.environment for c in result.cards]
        self.assertEqual(envs[0], "retail")

    def test_search_handles_empty_corpus(self) -> None:
        from src.rex.retrieval import HybridRetriever, build_query

        retriever = HybridRetriever([])
        result = retriever.search(build_query(initial_user="anything", messages=[]))
        self.assertEqual(len(result.cards), 0)

    def test_query_includes_failed_tool_signal(self) -> None:
        from src.rex.retrieval import build_query

        messages = [
            {"role": "user", "content": "Find my order"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "get_order_details", "arguments": "{}"},
            }]},
            {"role": "tool", "name": "get_order_details", "content": "Error: not found"},
        ]
        q = build_query(initial_user="Find my order", messages=messages, environment="retail")
        self.assertEqual(q.last_tool, "get_order_details")
        self.assertIn("get_order_details", q.failed_tools)


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------
class TestDistillation(unittest.TestCase):
    def test_distill_record_to_card_succeeds_for_reward_1(self) -> None:
        from src.rex.distill import distill_record_to_card

        rec = _trajectory_record(reward=1.0, tools=["get_order_details", "cancel_pending_order"])
        result = distill_record_to_card(rec)
        self.assertIsNotNone(result.card)
        card = result.card
        self.assertEqual(card.outcome, "successful")
        self.assertIn("cancel_pending_order", card.tool_ordering_hints)
        self.assertGreater(card.confidence, 0.5)

    def test_distill_record_to_card_avoid_for_failure(self) -> None:
        from src.rex.distill import distill_record_to_card

        rec = _trajectory_record(
            reward=0.0,
            tools=["cancel_pending_order"],  # mutating without verifier
        )
        rec["info"]["error"] = "missing_evidence"
        result = distill_record_to_card(rec)
        self.assertIsNotNone(result.card)
        self.assertEqual(result.card.outcome, "avoid")
        self.assertTrue(any("verify" in fb.lower() for fb in result.card.forbidden_behaviors))

    def test_distill_skips_records_with_no_tools(self) -> None:
        from src.rex.distill import distill_record_to_card

        rec = _trajectory_record(reward=1.0, tools=[])
        result = distill_record_to_card(rec)
        self.assertIsNone(result.card)

    def test_distill_audit_blocks_unredacted_text(self) -> None:
        # If somehow raw PII slipped through to the card, the audit fails.
        from src.rex.distill import distill_record_to_card

        # Force an error string that contains an unredacted email — the
        # parser sanitizes user_text but the audit step is the security
        # boundary on top.
        rec = _trajectory_record(reward=1.0, tools=["get_order_details"])
        rec["info"]["error"] = "user contacted alice@example.com"
        result = distill_record_to_card(rec)
        # Distill sanitizes error too, so this should *succeed* (proving
        # the redaction layer holds).
        self.assertIsNotNone(result.card)
        joined = " ".join(result.card.failure_patterns + [result.card.common_trap])
        self.assertNotIn("alice@example.com", joined)

    def test_extract_anti_patterns_loop_detection(self) -> None:
        from src.rex.distill import extract_anti_patterns
        from src.rex.memory_types import TrajectorySummary

        s = TrajectorySummary(
            task_id="0", trial=0, benchmark="tau-bench", environment="retail",
            controller="rex", reward=0.0, status="ok", error="",
            initial_user="", last_user="",
            tool_calls=["get_order", "get_order", "get_order", "get_order"],
            tool_arguments_redacted=[], tool_observations_redacted=[],
            assistant_messages=4, user_messages=1, tool_messages=4,
        )
        anti = extract_anti_patterns(s)
        self.assertTrue(any("loop" in a.lower() for a in anti))


# ---------------------------------------------------------------------------
# Playbook
# ---------------------------------------------------------------------------
class TestPlaybook(unittest.TestCase):
    def test_synthesize_playbook_includes_sections(self) -> None:
        from src.rex.playbook import synthesize_playbook

        cards = [
            _process_card("retail", "exchange__success", ["get_order_details", "exchange_delivered_order_items"]),
        ]
        cards[0].successful_patterns = ["exchange flow worked"]
        cards[0].failure_patterns = ["wrong variant"]
        cards[0].recovery_heuristics = ["retry with constraints"]

        pb = synthesize_playbook(cards)
        self.assertIn("Sequencing guidance", pb.body)
        self.assertIn("get_order_details", pb.body)
        self.assertGreater(pb.char_count, 0)

    def test_playbook_respects_size_budget(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.playbook import synthesize_playbook

        cards = [
            _process_card("retail", f"intent_{i}", ["a", "b"], trap="trap " * 50)
            for i in range(20)
        ]
        cfg = RexConfig.from_env().with_overrides(playbook_max_chars=400, playbook_max_lines=8)
        pb = synthesize_playbook(cards, config=cfg)
        self.assertLessEqual(pb.char_count, 405)

    def test_playbook_empty_cards_produces_no_guidance_message(self) -> None:
        from src.rex.playbook import synthesize_playbook

        pb = synthesize_playbook([])
        self.assertIn("No retrieved", pb.body)


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------
class TestMemoryStore(unittest.TestCase):
    def test_bulk_add_dedupes_and_persists(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_store import MemoryStore

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed", runtime_dir=Path(td) / "runtime"
            )
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=cfg.runtime_dir, config=cfg)
            cards = [
                _process_card("retail", "x", ["a", "b"]),
                _process_card("retail", "x", ["a", "b"]),  # dup
                _process_card("retail", "y", ["c", "d"]),
            ]
            written = store.bulk_add_cards(cards, domain="retail")
            self.assertEqual(len(written), 2)
            re_loaded = store.load_runtime("retail")
            self.assertEqual(len(re_loaded), 2)

    def test_load_all_combines_seed_and_runtime(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_store import MemoryStore
        from src.rex.memory_types import ProcessMemoryCard

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed", runtime_dir=Path(td) / "runtime"
            )
            cfg.bank_dir.mkdir(parents=True, exist_ok=True)
            seed_path = cfg.bank_dir / "retail.jsonl"
            seed_card = _process_card("retail", "seed_intent", ["a"])
            seed_path.write_text(json.dumps(seed_card.to_dict()) + "\n", encoding="utf-8")
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=cfg.runtime_dir, config=cfg)
            runtime_card = _process_card("retail", "runtime_intent", ["b"])
            store.bulk_add_cards([runtime_card], domain="retail")
            both = store.load_all("retail")
            self.assertEqual(len(both), 2)

    def test_pruning_caps_runtime_size(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_store import MemoryStore

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed",
                runtime_dir=Path(td) / "runtime",
                max_runtime_cards_per_domain=3,
            )
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=cfg.runtime_dir, config=cfg)
            cards = [_process_card("retail", f"intent_{i}", ["a"]) for i in range(8)]
            store.bulk_add_cards(cards, domain="retail")
            after = store.load_runtime("retail")
            self.assertLessEqual(len(after), 3)

    def test_index_file_written(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_store import MemoryStore

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed", runtime_dir=Path(td) / "runtime"
            )
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=cfg.runtime_dir, config=cfg)
            store.bulk_add_cards([_process_card("retail", "x", ["a"])], domain="retail")
            store.save_memory_index()
            self.assertTrue((Path(td) / "runtime" / "index.json").exists())


# ---------------------------------------------------------------------------
# Memory quality
# ---------------------------------------------------------------------------
class TestMemoryQuality(unittest.TestCase):
    def test_detect_duplicates_on_overlapping_content(self) -> None:
        from src.rex.memory_quality import detect_duplicates

        a = _process_card("retail", "x", ["a", "b"])
        a.successful_patterns = ["did the right thing", "verified before write"]
        b = _process_card("retail", "y", ["c", "d"])
        b.successful_patterns = ["did the right thing", "verified before write"]
        out = detect_duplicates([a, b])
        self.assertEqual(len(out), 1)

    def test_apply_age_decay_drops_old_unused_cards(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_quality import apply_age_decay
        from src.rex.memory_types import ProcessMemoryCard

        old = ProcessMemoryCard.from_dict({
            "card_id": "old",
            "environment": "retail",
            "task_category": "x",
            "tool_ordering_hints": ["a"],
            "common_trap": "trap",
            "confidence": 1.0,
            "usage_count": 0,
            "timestamp_ms": 0,
        })
        cfg = RexConfig.from_env().with_overrides(
            quality_decay_half_life_days=1.0, quality_min_usage_for_decay=10,
        )
        modified = apply_age_decay([old], now_ms=int(time.time() * 1000), config=cfg)
        self.assertEqual(modified, 1)
        self.assertLess(old.confidence, 0.9)

    def test_consolidate_returns_report(self) -> None:
        from src.rex.memory_quality import consolidate

        cards = [
            _process_card("retail", "x", ["a", "b"]),
            _process_card("retail", "y", ["c", "d"]),
        ]
        report = consolidate(cards)
        self.assertEqual(report.total_cards, 2)


# ---------------------------------------------------------------------------
# Working state
# ---------------------------------------------------------------------------
class TestWorkingState(unittest.TestCase):
    def test_failed_tool_recorded_when_observation_signals_error(self) -> None:
        from src.rex.working_state import working_state_for_messages

        messages = [
            {"role": "user", "content": "find my order"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "get_order_details", "arguments": "{}"},
            }]},
            {"role": "tool", "name": "get_order_details", "content": "Error: not found"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "2", "type": "function",
                "function": {"name": "get_user_details", "arguments": "{}"},
            }]},
            {"role": "tool", "name": "get_user_details", "content": "{\"id\": \"<USER_ID>\"}"},
        ]
        s = working_state_for_messages(messages, initial_user="find my order")
        self.assertIn("get_order_details", s.failed_tools)
        self.assertIn("get_user_details", s.successful_tools)
        self.assertTrue(s.verification_done)

    def test_retry_count_increments_on_repeated_args(self) -> None:
        from src.rex.working_state import working_state_for_messages

        messages = [
            {"role": "user", "content": "look up"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "get_order_details", "arguments": '{"id":"1"}'},
            }]},
            {"role": "tool", "name": "get_order_details", "content": "Error"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "2", "type": "function",
                "function": {"name": "get_order_details", "arguments": '{"id":"1"}'},
            }]},
            {"role": "tool", "name": "get_order_details", "content": "Error"},
        ]
        s = working_state_for_messages(messages)
        self.assertGreaterEqual(s.retry_count, 1)


# ---------------------------------------------------------------------------
# Retrieval logging
# ---------------------------------------------------------------------------
class TestRetrievalLogging(unittest.TestCase):
    def test_logger_writes_events_and_aggregator_reads_them(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_types import RetrievalQuery, RetrievalResult
        from src.rex.playbook import synthesize_playbook
        from src.rex.retrieval_logging import RetrievalLogger, aggregate_logs

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(retrieval_log_dir=Path(td))
            logger = RetrievalLogger(config=cfg)
            cards = [_process_card("retail", "x", ["a", "b"])]
            playbook = synthesize_playbook(cards)
            logger.log(
                benchmark="tau-bench", environment="retail", controller="rex",
                task_id="0", trial=0, step_index=0,
                query=RetrievalQuery(text="test", environment="retail"),
                result=RetrievalResult(cards=cards, scores=[1.0], diagnostic={"num_corpus": 1}),
                playbook=playbook,
            )
            agg = aggregate_logs(Path(td))
            self.assertGreaterEqual(agg["totals"]["events"], 1)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------
class TestPipeline(unittest.TestCase):
    def test_promote_records_writes_v2_card(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_store import MemoryStore
        from src.rex.pipeline import promote_records

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed", runtime_dir=Path(td) / "runtime"
            )
            recs = [_trajectory_record(reward=1.0)]
            manifest = promote_records(
                recs, domain="retail", config=cfg, runtime_dir=cfg.runtime_dir,
            )
            self.assertGreaterEqual(manifest["promoted"], 1)
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=cfg.runtime_dir, config=cfg)
            self.assertGreaterEqual(len(store.load_runtime("retail")), 1)

    def test_pipeline_blocks_test_split_by_default(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.pipeline import promote_records

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed", runtime_dir=Path(td) / "runtime"
            )
            rec = _trajectory_record(reward=1.0, info={"task_split": "test"})
            manifest = promote_records(
                [rec], domain="retail", config=cfg, runtime_dir=cfg.runtime_dir,
            )
            self.assertEqual(manifest["promoted"], 0)
            self.assertTrue(manifest["blocked"])

    def test_load_corpus_combines_seed_and_runtime(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_store import MemoryStore
        from src.rex.pipeline import load_corpus_for_domain, promote_records

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed", runtime_dir=Path(td) / "runtime"
            )
            # Seed with a card directly in the v2 store.
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=cfg.runtime_dir, config=cfg)
            cfg.bank_dir.mkdir(parents=True, exist_ok=True)
            seed_path = cfg.bank_dir / "retail.jsonl"
            seed_path.write_text(
                json.dumps(_process_card("retail", "seed_x", ["a"]).to_dict()) + "\n",
                encoding="utf-8",
            )
            promote_records(
                [_trajectory_record(reward=1.0)],
                domain="retail", config=cfg, runtime_dir=cfg.runtime_dir,
            )
            corpus = load_corpus_for_domain(
                "retail", config=cfg, runtime_dir=cfg.runtime_dir, seed_dir=cfg.bank_dir,
            )
            self.assertGreaterEqual(len(corpus), 2)

    def test_refresh_playbook_evolves_with_state(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.pipeline import refresh_playbook
        from src.rex.retrieval import HybridRetriever

        cards = [
            _process_card("retail", "exchange__success", ["get_order_details", "exchange_delivered_order_items"]),
            _process_card("retail", "return__success", ["get_order_details", "return_delivered_order_items"]),
        ]
        retriever = HybridRetriever(cards, config=RexConfig.from_env())
        messages = [
            {"role": "user", "content": "exchange my keyboard"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "get_order_details", "arguments": "{}"},
            }]},
            {"role": "tool", "name": "get_order_details", "content": "Error: not found"},
        ]
        playbook, result, state = refresh_playbook(
            retriever=retriever,
            initial_user="exchange my keyboard",
            messages=messages,
            environment="retail",
            step_index=2,
        )
        self.assertGreater(len(playbook.body), 0)
        self.assertGreater(len(result.cards), 0)
        self.assertIn("get_order_details", state.failed_tools)


# ---------------------------------------------------------------------------
# End-to-end safety: nothing leaks through the prompt
# ---------------------------------------------------------------------------
class TestBenchmarkSafety(unittest.TestCase):
    def test_no_pii_in_distilled_card_after_promotion(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_store import MemoryStore
        from src.rex.pipeline import promote_records

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed", runtime_dir=Path(td) / "runtime"
            )
            rec = _trajectory_record(
                reward=1.0,
                user_text="Cancel order #W1234567 paid via credit_card_1234567 for alice@example.com",
            )
            promote_records(
                [rec], domain="retail", config=cfg, runtime_dir=cfg.runtime_dir,
            )
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=cfg.runtime_dir, config=cfg)
            for c in store.load_runtime("retail"):
                blob = json.dumps(c.to_dict(), ensure_ascii=False, default=str)
                self.assertNotIn("#W1234567", blob)
                self.assertNotIn("credit_card_1234567", blob)
                self.assertNotIn("alice@example.com", blob)

    def test_test_split_block_prevents_runtime_growth(self) -> None:
        from src.rex.config import RexConfig
        from src.rex.memory_store import MemoryStore
        from src.rex.pipeline import promote_records

        with tempfile.TemporaryDirectory() as td:
            cfg = RexConfig.from_env().with_overrides(
                bank_dir=Path(td) / "seed", runtime_dir=Path(td) / "runtime"
            )
            recs = [_trajectory_record(reward=1.0, info={"task_split": "test"}) for _ in range(3)]
            promote_records(
                recs, domain="retail", config=cfg, runtime_dir=cfg.runtime_dir,
            )
            store = MemoryStore(seed_dir=cfg.bank_dir, runtime_dir=cfg.runtime_dir, config=cfg)
            self.assertEqual(len(store.load_runtime("retail")), 0)


if __name__ == "__main__":
    unittest.main()
