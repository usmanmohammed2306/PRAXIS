"""Tau-bench REx-RPE agent.

Continual procedural-memory tool-calling agent. The agent:

  1. **Retrieves** relevant procedural cards from the merged seed+runtime
     memory bank using the hybrid lexical+embedding :class:`HybridRetriever`.
  2. **Synthesizes** a compact :class:`TacticalPlaybook` and injects it
     into the system prompt at every refresh point.
  3. **Executes** native OpenAI-style tool calls.
  4. **Reflects** before mutating tools (write-only SABER-style guard).
  5. **Refreshes** retrieval at every ``REX_RETRIEVAL_REFRESH_EVERY``
     effective steps so the playbook evolves with the trajectory state.
  6. **Logs** retrieval diagnostics to ``$REX_RETRIEVAL_LOG_DIR`` for
     post-run summary aggregation.

All cards are leakage-audited; nothing in the prompt contains raw IDs.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # tau-bench is optional during unit tests.
    from tau_bench.agents.base import Agent
    from tau_bench.types import Action, SolveResult  # type: ignore
except Exception:  # pragma: no cover - exercised in minimal test envs
    class Agent:  # type: ignore[no-redef]
        pass

    class Action:  # type: ignore[no-redef]
        def __init__(self, name: str, kwargs: Optional[Dict[str, Any]] = None) -> None:
            self.name = name
            self.kwargs = dict(kwargs or {})

    class SolveResult:  # type: ignore[no-redef]
        def __init__(self, reward: float, info: Dict[str, Any], messages: List[Dict[str, Any]], total_cost: float = 0.0) -> None:
            self.reward = reward
            self.info = info
            self.messages = messages
            self.total_cost = total_cost

from ..common.openai_client import get_client
from .config import RexConfig
from .decision_retrieval import ExperienceConditionedRetriever, extract_operational_context
from .experience import (
    ExperienceRetriever,
    load_experience_cards,
    redact_text,
    render_experience_brief,
)
from .memory_types import from_experience_card
from .pipeline import refresh_playbook
from .playbook import synthesize_playbook
from .retrieval import HybridRetriever, build_query
from .retrieval_context import build_experience_query
from .retrieval_logging import RetrievalLogger
from .state_graph import OperationalStateGraph, TransitionOutcome
from .working_state import working_state_for_messages


RESPOND_TOOL_NAME = "respond"
RESPOND_MAX_CHARS = 900

MUTATING_TOOLS_RETAIL = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "modify_user_address",
    "transfer_to_human_agents",
}

MUTATING_TOOLS_AIRLINE = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
    "send_certificate",
    "transfer_to_human_agents",
}


def _is_context_overflow(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        ("context" in s and "length" in s)
        or "maximum context length" in s
        or "contextwindowexceeded" in s
        or exc.__class__.__name__ == "ContextWindowExceededError"
    )


def _float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _obs_text(env_resp: Any) -> str:
    if env_resp is None:
        return ""
    obs = getattr(env_resp, "observation", env_resp)
    if isinstance(obs, (dict, list)):
        try:
            return json.dumps(obs, ensure_ascii=False, default=str)
        except Exception:
            return str(obs)
    return str(obs)


def _extract_initial_user_message(env_reset: Any) -> str:
    if env_reset is None:
        return ""
    for attr in ("observation", "content", "message", "user_message"):
        v = getattr(env_reset, attr, None)
        if v:
            return str(v)
    if isinstance(env_reset, str):
        return env_reset
    return str(env_reset)


def _assistant_message_dict(msg: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
    tcs = getattr(msg, "tool_calls", None) or []
    if tcs:
        out["tool_calls"] = [
            {
                "id": getattr(tc, "id", f"rex_tool_{i}"),
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for i, tc in enumerate(tcs)
        ]
    return out


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}


def _tool_names(tools: Sequence[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


class RexAgent(Agent):  # type: ignore[misc]
    """Native tool-calling agent with continual procedural-memory retrieval.

    The agent keeps the legacy :class:`ExperienceRetriever` corpus accessible
    via ``self.cards`` / ``self.retriever`` for backward-compatible surface
    area, but delegates retrieval at runtime to a :class:`HybridRetriever`
    that combines BM25 with deterministic embeddings and uses the
    ``WorkingState`` to evolve queries with the trajectory.
    """

    style_name = "rex"

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        env_hint: str = "",
        client: Any = None,
        bank_dir: Optional[Path] = None,
        runtime_dir: Optional[Path] = None,
        config: Optional[RexConfig] = None,
        retrieval_logger: Optional[RetrievalLogger] = None,
    ) -> None:
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.model = model
        self.provider = provider
        self.temperature = float(temperature)
        self.env_hint = env_hint or self._infer_domain()
        self.client = client or get_client()
        default_bank = Path(os.environ.get("REX_EXPERIENCE_DIR", str(Path.cwd() / "outputs" / "experience_bank")))
        self.bank_dir = bank_dir or default_bank
        self.runtime_dir = runtime_dir
        # Resolve config last so explicit args win.
        cfg_overrides: Dict[str, Any] = {"bank_dir": Path(self.bank_dir)}
        if self.runtime_dir is not None:
            cfg_overrides["runtime_dir"] = Path(self.runtime_dir)
        self.config = (config or RexConfig.from_env()).with_overrides(**cfg_overrides)

        # ------------------------------------------------------------------
        # Legacy corpus — v1 ExperienceCard format from seed bank only.
        # Kept for backwards-compatible diagnostics and the legacy fallback
        # retriever. Runtime cards are loaded below via load_corpus_for_domain.
        # ------------------------------------------------------------------
        self.cards = load_experience_cards(
            self._experience_domain(), self.bank_dir, runtime_dir=self.runtime_dir,
        )
        self.retriever = ExperienceRetriever(self.cards)

        # ------------------------------------------------------------------
        # Hybrid pipeline corpus — loaded via load_corpus_for_domain so that
        # BOTH seed AND runtime ProcessMemoryCards (v2 schema written by
        # pipeline_promote_records) are included.  This is the correct path:
        # _read_cards_jsonl used by load_experience_cards silently drops v2
        # cards because it tries ExperienceCard(**data) on every line.
        # load_corpus_for_domain uses MemoryStore._read_cards which handles
        # both schemas via _is_legacy_experience_card().
        # ------------------------------------------------------------------
        import sys as _sys
        try:
            from .pipeline import load_corpus_for_domain as _load_corpus
            process_cards = _load_corpus(
                self._experience_domain(),
                config=self.config,
            )
            print(
                f"[rex] corpus loaded: domain={self._experience_domain()} "
                f"cards={len(process_cards)} "
                f"(seed={sum(1 for c in process_cards if str(c.card_id).startswith('seed-'))} "
                f"runtime={sum(1 for c in process_cards if not str(c.card_id).startswith('seed-'))})",
                file=_sys.stderr,
            )
        except Exception as _e:
            # Fallback: lift v1 cards that load_experience_cards did find.
            process_cards = [from_experience_card(c) for c in self.cards]
            print(
                f"[rex] corpus fallback (load_corpus_for_domain failed: {_e}): "
                f"{len(process_cards)} cards",
                file=_sys.stderr,
            )
        self.process_cards = process_cards
        try:
            self.hybrid_retriever: Optional[HybridRetriever] = (
                HybridRetriever(process_cards, config=self.config)
                if process_cards else None
            )
        except Exception:
            # If embedding setup fails for any reason, fall back to legacy retrieval.
            self.hybrid_retriever = None

        # ------------------------------------------------------------------
        # Experience-conditioned retriever (state-triggered mid-trajectory)
        # Retrieves prior experiences matching the current execution state
        # rather than broad text similarity. Used for refresh after failures,
        # retries, escalations, and verification steps.
        # ------------------------------------------------------------------
        try:
            self.operational_retriever: Optional[ExperienceConditionedRetriever] = (
                ExperienceConditionedRetriever(process_cards)
            )
            self.enable_operational_retrieval = True
        except Exception:
            self.operational_retriever = None
            self.enable_operational_retrieval = False

        self.retrieval_logger = retrieval_logger or RetrievalLogger(config=self.config)
        self.enable_startup_analysis = bool(self.config.startup_analysis)
        self.enable_reflection = bool(self.config.mutation_reflection)
        # Stateful re-retrieval cadence; 0 disables (only initial brief is used).
        self.retrieval_refresh_every = int(self.config.retrieval_refresh_every or 0)

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------
    def _infer_domain(self) -> str:
        names = set(_tool_names(self.tools_info))
        if "get_order_details" in names or "exchange_delivered_order_items" in names:
            return "retail"
        if "get_reservation_details" in names or "book_reservation" in names:
            return "airline"
        return "generic"

    def _experience_domain(self) -> str:
        if self.env_hint in {"retail", "airline"}:
            return self.env_hint
        return "bfcl" if self.env_hint == "bfcl" else "retail"

    def _mutating_tools(self) -> set[str]:
        if self.env_hint == "retail":
            return MUTATING_TOOLS_RETAIL
        if self.env_hint == "airline":
            return MUTATING_TOOLS_AIRLINE
        return set()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _system_prompt(self, brief: str, startup_analysis: str = "") -> str:
        parts = [
            "You are REx-RPE, a confident customer-service tool agent.",
            "Use the provided native function tools. Make exactly one tool call at a time and wait for its result.",
            "The retrieved examples below are procedural memory only: copy the process, never copy IDs, emails, payment methods, dates, prices, or arguments.",
            "When the user's request is resolved, reply with a concise final answer grounded in current tool observations.",
            "Keep reasoning brief. Do not emit custom JSON unless a tool schema requires JSON arguments.",
            "\n--- Retrieved Experience Brief ---\n" + brief,
        ]
        if startup_analysis:
            parts.append("\n--- Startup Analysis ---\n" + startup_analysis)
        if self.wiki:
            parts.append("\n--- Domain Policy ---\n" + self.wiki)
        return "\n".join(parts)

    def _initial_brief(self, initial_user: str) -> tuple[str, List[str]]:
        """Build the first experience brief at trajectory start."""
        if self.hybrid_retriever and self.hybrid_retriever.num_cards:
            query = build_query(
                initial_user=initial_user,
                messages=[],
                environment=self._experience_domain(),
                benchmark=self._benchmark_label(),
                controller="rex",
                step_index=0,
                top_k=self.config.top_k,
            )
            result = self.hybrid_retriever.search(query)
            playbook = synthesize_playbook(result.cards, config=self.config)
            self.retrieval_logger.log(
                benchmark=self._benchmark_label(),
                environment=self._experience_domain(),
                controller="rex",
                task_id="initial",
                trial=0,
                step_index=0,
                query=query,
                result=result,
                playbook=playbook,
            )
            return playbook.body, result.card_ids()
        # Legacy fallback
        selected = self.retriever.search(initial_user, k=3)
        return render_experience_brief(selected), [c.card_id for c in selected]

    def _benchmark_label(self) -> str:
        if self.env_hint in {"retail", "airline"}:
            return "tau-bench"
        if self.env_hint == "bfcl":
            return "BFCL"
        return "unknown"

    def _startup_analysis(self, initial_user: str, brief: str) -> str:
        if not self.enable_startup_analysis:
            return ""
        max_words = max(40, int(self.config.startup_analysis_max_words))
        prompt = (
            f"Return a concise plan in <={max_words} words. Do not call tools.\n"
            "Fields: intent, first_tool, evidence_needed, write_risk.\n"
            "Use retrieved examples as process only and never copy example IDs.\n\n"
            f"Experience brief:\n{brief}\n\nUser request:\n{redact_text(initial_user)}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
            )
            content = (resp.choices[0].message.content or "").strip()
            return content[:900]
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Mutation reflection
    # ------------------------------------------------------------------
    def _reflection_prompt(self, tool_name: str, args: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        recent = messages[-10:]
        compact = []
        for m in recent:
            role = m.get("role")
            content = m.get("content", "")
            if role == "tool":
                content = f"{m.get('name')}: {str(content)[:900]}"
            compact.append(f"{role}: {str(content)[:900]}")
        return [
            {
                "role": "system",
                "content": (
                    "You are the REx mutation reflection check. Return JSON only:\n"
                    "{\"allow\": true|false, \"reason\": \"short\", \"ask_user\": \"question if blocked\"}\n"
                    "Allow the mutation if ALL of:\n"
                    "  1. Tool observations (get_/find_/search_ results) confirm the IDs and state needed.\n"
                    "  2. The user's original request or a follow-up message explicitly asks for this action.\n"
                    "  3. No policy rule prohibits it (e.g. order already cancelled, item not returnable).\n"
                    "Block ONLY for a clear policy violation or genuinely missing evidence — NOT because "
                    "the user has not said 'yes' a second time. The user's original request is sufficient "
                    "authorisation for the actions it explicitly describes. Examples are not evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Proposed mutating tool: {tool_name}\n"
                    f"Arguments: {json.dumps(args, ensure_ascii=False, sort_keys=True)}\n"
                    f"Recent current-task evidence:\n" + "\n".join(compact)
                ),
            },
        ]

    def _reflection_allows(self, content: str) -> tuple[bool, str, str]:
        data = _safe_json_loads(content)
        if "allow" in data:
            return bool(data.get("allow")), str(data.get("reason") or ""), str(data.get("ask_user") or "")
        text = (content or "").lower()
        block = any(x in text for x in ("cannot verify", "not allowed", "forbidden", "do not execute", "policy violation", "block"))
        return (not block), content[:160], ""

    def _check_mutation(self, tool_name: str, args: Dict[str, Any], messages: List[Dict[str, Any]], stats: Dict[str, Any]) -> tuple[bool, str]:
        if not self.enable_reflection or tool_name not in self._mutating_tools():
            return True, ""
        stats["reflection_calls"] += 1
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._reflection_prompt(tool_name, args, messages),
                temperature=0.0,
                max_tokens=180,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            stats["reflection_errors"] += 1
            return True, f"reflection_error_allowed:{exc}"
        allow, reason, ask = self._reflection_allows(content)
        if allow:
            return True, reason
        stats["reflection_blocks"] += 1
        question = ask.strip() or (
            "Before I make that change, please confirm the exact action and details you want me to apply."
        )
        return False, question[:RESPOND_MAX_CHARS]

    # ------------------------------------------------------------------
    # Stateful retrieval helpers (legacy + new)
    # ------------------------------------------------------------------
    def _retrieval_query(
        self,
        initial_user: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Build a stateful retrieval *string* (legacy compat for tests).

        The hybrid pipeline uses :func:`build_query` to construct a
        :class:`RetrievalQuery` directly, but tests rely on this string
        surface. The two stay aligned: the string here is what would be
        passed as the ``text`` field of the structured query.
        """
        parts: List[str] = [initial_user or ""]
        last_user = ""
        last_tool_name = ""
        last_tool_obs = ""
        for m in messages:
            role = m.get("role")
            if role == "user" and m.get("content"):
                last_user = str(m.get("content"))
            elif role == "tool":
                last_tool_name = str(m.get("name") or "")
                last_tool_obs = str(m.get("content") or "")
        if last_user and last_user != initial_user:
            parts.append(last_user)
        if last_tool_name:
            parts.append(last_tool_name)
        if last_tool_obs:
            parts.append(last_tool_obs[:240])
        return redact_text(" ".join(p for p in parts if p))

    def _refresh_brief_with_state_graph(
        self,
        state_graph: OperationalStateGraph,
    ) -> tuple[str, List[Any]]:
        """Retrieve prior experiences matching the current execution state.

        Called mid-trajectory when the agent's state changes significantly
        (failure, retry, escalation, etc.).  Retrieves experiences from
        Humans 1/2/3 that match the current situation.

        Returns ``(brief, card_objs)``; falls back to empty on any error.
        """
        if not self.operational_retriever or not state_graph:
            return "", []

        try:
            # Retrieve cards conditioned on operational phase
            cards = self.operational_retriever.retrieve_for_phase(state_graph, top_k=3)
            brief_parts = []
            for card in cards:
                if card.recommended_next_tools:
                    brief_parts.append(
                        f"Next steps for {card.task_category}: {', '.join(card.recommended_next_tools)}"
                    )
                if card.common_trap:
                    brief_parts.append(f"Caution: {card.common_trap}")
                if card.recovery_heuristics:
                    brief_parts.append(f"Recovery: {'; '.join(card.recovery_heuristics[:2])}")

            if brief_parts:
                brief = "\n".join(brief_parts)
            else:
                brief = "Continue with current approach based on operational state."

            return brief, cards
        except Exception:
            return "", []

    def _refresh_brief(
        self,
        initial_user: str,
        messages: List[Dict[str, Any]],
        *,
        step_index: int = 0,
        task_id: Any = "",
        trial: int = 0,
    ) -> tuple[str, List[Any]]:
        """Re-run retrieval with current state and return ``(brief, card_objs)``.

        Uses the hybrid retriever when available; falls back to the legacy
        BM25-style retriever otherwise (kept for environments where embedding
        backends fail).
        """
        if self.hybrid_retriever and self.hybrid_retriever.num_cards:
            playbook, result, _state = refresh_playbook(
                retriever=self.hybrid_retriever,
                initial_user=initial_user,
                messages=messages,
                environment=self._experience_domain(),
                benchmark=self._benchmark_label(),
                controller="rex",
                step_index=step_index,
                config=self.config,
                logger=self.retrieval_logger,
                task_id=task_id,
                trial=trial,
            )
            return playbook.body, result.cards
        # Legacy fallback
        query = self._retrieval_query(initial_user, messages)
        selected = self.retriever.search(query, k=3)
        return render_experience_brief(selected), selected

    def _update_operational_state(
        self,
        state_graph: OperationalStateGraph,
        messages: List[Dict[str, Any]],
        last_tool_called: Optional[str],
        last_tool_args: Optional[Dict[str, Any]],
        tool_result: Optional[str],
        tool_result_name: Optional[str],
    ) -> None:
        """Update operational state graph with latest tool execution."""
        if not state_graph or not last_tool_called:
            return

        # Record tool execution outcome
        outcome_str = tool_result or ""
        is_error = any(
            k in (outcome_str or "").lower()
            for k in ("error", "not found", "invalid", "failed", "denied", "unavailable")
        )

        # Determine outcome type
        if is_error:
            state_graph.record_tool_execution(
                last_tool_called,
                TransitionOutcome.FAILURE,
                error_type="tool_error",
            )
            state_graph.record_failure(
                last_tool_called,
                error_type="tool_error",
                error_detail=outcome_str[:200] if outcome_str else None,
            )
        else:
            state_graph.record_tool_execution(
                last_tool_called,
                TransitionOutcome.SUCCESS,
                evidence_produced=[outcome_str[:200]] if outcome_str else [],
            )

        # Record transition if we know the next tool (from messages)
        # For now, we just track the execution

    def _replace_system_prompt(
        self,
        messages: List[Dict[str, Any]],
        new_brief: str,
        startup_analysis: str,
    ) -> None:
        if not messages:
            return
        if messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["content"] = self._system_prompt(new_brief, startup_analysis)

    def _trim_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= 16:
            return messages
        tail = list(messages[-14:])
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        return [messages[0]] + tail

    # ------------------------------------------------------------------
    # Solve loop
    # ------------------------------------------------------------------
    def solve(
        self,
        env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        env_reset = env.reset(task_index=task_index)
        initial_user = _extract_initial_user_message(env_reset)
        brief, initial_card_ids = self._initial_brief(initial_user)
        startup = self._startup_analysis(initial_user, brief)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt(brief, startup)}]
        if initial_user:
            messages.append({"role": "user", "content": initial_user})

        # Initialize operational state graph for decision-conditioned retrieval
        state_graph = OperationalStateGraph() if self.enable_operational_retrieval else None

        cards_used_ids: List[str] = list(initial_card_ids)
        stats: Dict[str, Any] = {
            "controller": "rex",
            "cards_loaded": len(self.cards),
            "cards_used": cards_used_ids,
            "startup_analysis": bool(startup),
            "reflection_calls": 0,
            "reflection_blocks": 0,
            "reflection_errors": 0,
            "tool_calls_executed": 0,
            "mutating_tool_calls": 0,
            "retrieval_refreshes": 0,
            "retrieval_backend": "hybrid" if self.hybrid_retriever else "legacy",
            "operational_retrieval_enabled": self.enable_operational_retrieval,
            "playbook_chars": len(brief),
        }
        reward: float = 0.0
        info: Dict[str, Any] = {}
        total_cost = 0.0
        step_error = ""
        done = False

        effective_steps = 0
        last_tool_called: Optional[str] = None
        last_tool_args: Optional[Dict[str, Any]] = None
        for _ in range(max_num_steps):
            if (
                self.retrieval_refresh_every
                and effective_steps > 0
                and effective_steps % self.retrieval_refresh_every == 0
            ):
                # Try operational state graph retrieval first if enabled
                if state_graph and self.enable_operational_retrieval:
                    new_brief, new_selected = self._refresh_brief_with_state_graph(state_graph)
                    if new_brief:
                        self._replace_system_prompt(messages, new_brief, startup)
                        stats["retrieval_refreshes"] += 1
                        stats["playbook_chars"] = len(new_brief)
                        for c in new_selected:
                            cid = getattr(c, "card_id", "")
                            if cid and cid not in cards_used_ids:
                                cards_used_ids.append(cid)
                        stats["cards_used"] = cards_used_ids
                    else:
                        # Fall back to hybrid retrieval if operational retrieval fails
                        new_brief, new_selected = self._refresh_brief(
                            initial_user,
                            messages,
                            step_index=effective_steps,
                            task_id=task_index if task_index is not None else "",
                            trial=0,
                        )
                        if new_brief:
                            self._replace_system_prompt(messages, new_brief, startup)
                            stats["retrieval_refreshes"] += 1
                            stats["playbook_chars"] = len(new_brief)
                            for c in new_selected:
                                cid = getattr(c, "card_id", "")
                                if cid and cid not in cards_used_ids:
                                    cards_used_ids.append(cid)
                            stats["cards_used"] = cards_used_ids
                else:
                    # Use traditional hybrid retrieval
                    new_brief, new_selected = self._refresh_brief(
                        initial_user,
                        messages,
                        step_index=effective_steps,
                        task_id=task_index if task_index is not None else "",
                        trial=0,
                    )
                    self._replace_system_prompt(messages, new_brief, startup)
                    stats["retrieval_refreshes"] += 1
                    stats["playbook_chars"] = len(new_brief)
                    for c in new_selected:
                        cid = getattr(c, "card_id", "")
                        if cid and cid not in cards_used_ids:
                            cards_used_ids.append(cid)
                    stats["cards_used"] = cards_used_ids

            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools_info or None,
                    tool_choice="auto" if self.tools_info else None,
                    temperature=self.temperature,
                )
            except Exception as exc:
                if _is_context_overflow(exc) and len(messages) > 4:
                    messages = self._trim_messages(messages)
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=self.tools_info or None,
                            tool_choice="auto" if self.tools_info else None,
                            temperature=self.temperature,
                        )
                    except Exception as exc2:
                        step_error = f"chat_completion_failed: {exc2}"
                        break
                else:
                    step_error = f"chat_completion_failed: {exc}"
                    break

            msg = resp.choices[0].message
            messages.append(_assistant_message_dict(msg))
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        kwargs = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        kwargs = {}
                    if name in self._mutating_tools():
                        stats["mutating_tool_calls"] += 1
                    allowed, block_text = self._check_mutation(name, kwargs, messages, stats)
                    if not allowed:
                        env_resp = env.step(Action(name=RESPOND_TOOL_NAME, kwargs={"content": block_text}))
                        user_reply = _obs_text(env_resp)
                        if user_reply:
                            messages.append({"role": "user", "content": user_reply})
                        reward = _float(getattr(env_resp, "reward", reward), reward)
                        info = getattr(env_resp, "info", info) or info
                        done = bool(getattr(env_resp, "done", False))
                        if done:
                            break
                        continue
                    try:
                        env_resp = env.step(Action(name=name, kwargs=kwargs))
                    except Exception as env_exc:
                        if _is_context_overflow(env_exc):
                            step_error = f"env_step_context_overflow: {env_exc}"
                            done = True
                            break
                        raise
                    stats["tool_calls_executed"] += 1
                    tool_result = _obs_text(env_resp)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": getattr(tc, "id", f"rex_{stats['tool_calls_executed']}"),
                        "name": name,
                        "content": tool_result,
                    })

                    # Update operational state graph with tool execution result
                    if state_graph:
                        self._update_operational_state(
                            state_graph,
                            messages,
                            name,
                            kwargs,
                            tool_result,
                            name,
                        )
                        # Track last tool for next iteration
                        last_tool_called = name
                        last_tool_args = kwargs

                    reward = _float(getattr(env_resp, "reward", reward), reward)
                    info = getattr(env_resp, "info", info) or info
                    done = bool(getattr(env_resp, "done", False))
                    if done:
                        break
                if done:
                    break
                messages = self._trim_messages(messages)
                effective_steps += 1
                continue

            content = (getattr(msg, "content", "") or "").strip()
            if len(content) > RESPOND_MAX_CHARS:
                content = content[:RESPOND_MAX_CHARS] + " ..."
            try:
                env_resp = env.step(Action(name=RESPOND_TOOL_NAME, kwargs={"content": content}))
            except Exception as env_exc:
                if _is_context_overflow(env_exc):
                    step_error = f"env_respond_context_overflow: {env_exc}"
                    break
                raise
            user_reply = _obs_text(env_resp)
            if user_reply:
                messages.append({"role": "user", "content": user_reply})
            reward = _float(getattr(env_resp, "reward", reward), reward)
            info = getattr(env_resp, "info", info) or info
            done = bool(getattr(env_resp, "done", False))
            messages = self._trim_messages(messages)
            effective_steps += 1
            if done:
                break

        # Final working-state summary added to diagnostics for post-run analysis.
        try:
            final_state = working_state_for_messages(messages, initial_user=initial_user)
            stats["working_state"] = final_state.to_dict()
        except Exception:
            pass

        # Add operational state graph to diagnostics if available
        if state_graph:
            try:
                stats["operational_state_graph"] = state_graph.to_dict()
            except Exception:
                pass

        info = dict(info) if info else {}
        if step_error:
            info["error"] = step_error
        info.setdefault("controller", "rex")
        info["rex_stats"] = stats
        return SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)


__all__ = ["RexAgent"]
