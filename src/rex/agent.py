"""Tau-bench REx-RPE agent.

REx intentionally uses the native OpenAI-compatible tool-calling surface. The
experience layer is prompt-only procedural memory; it never supplies concrete
task answers.
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
from .experience import ExperienceRetriever, load_experience_cards, redact_text, render_experience_brief


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
    """Native tool-calling agent with leakage-safe procedural retrieval."""

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
        # Runtime memory dir: auto-resolves to ${REX_RUNTIME_DIR} env or default.
        self.runtime_dir = runtime_dir
        self.cards = load_experience_cards(
            self._experience_domain(), self.bank_dir, runtime_dir=self.runtime_dir,
        )
        self.retriever = ExperienceRetriever(self.cards)
        self.enable_startup_analysis = os.environ.get("REX_STARTUP_ANALYSIS", "1") != "0"
        self.enable_reflection = os.environ.get("REX_MUTATION_REFLECTION", "1") != "0"
        # Stateful re-retrieval: refresh the playbook every N effective steps.
        # 0 disables (only the initial brief is used).
        self.retrieval_refresh_every = int(os.environ.get("REX_RETRIEVAL_REFRESH_EVERY", "2") or 0)

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
        return "ace" if self.env_hint == "ace" else "retail"

    def _mutating_tools(self) -> set[str]:
        if self.env_hint == "retail":
            return MUTATING_TOOLS_RETAIL
        if self.env_hint == "airline":
            return MUTATING_TOOLS_AIRLINE
        return set()

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

    def _startup_analysis(self, initial_user: str, brief: str) -> str:
        if not self.enable_startup_analysis:
            return ""
        prompt = (
            "Return a concise plan in <=120 words. Do not call tools.\n"
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
                    "Allow only if current conversation/tool evidence supports the exact write, policy permits it, "
                    "and the user has confirmed the exact action/arguments when confirmation is needed. "
                    "Examples are not evidence."
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
        block = any(x in text for x in ("missing", "cannot verify", "not allowed", "forbidden", "ask the user", "do not execute"))
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

    def _retrieval_query(
        self,
        initial_user: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Build a stateful retrieval query.

        The query evolves with the trajectory: it includes the initial user
        request, the latest user reply, the most recent tool name, and a
        truncated snippet of its observation. This is the architectural
        change that turns retrieval from a one-shot thing into a stateful
        process: cards retrieved at step 5 reflect what we've already
        learned, not what the user said at step 0.
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

    def _refresh_brief(
        self,
        initial_user: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[str, List[Any]]:
        """Re-run retrieval with the current trajectory state and return the brief."""
        query = self._retrieval_query(initial_user, messages)
        selected = self.retriever.search(query, k=3)
        return render_experience_brief(selected), selected

    def _replace_system_prompt(
        self,
        messages: List[Dict[str, Any]],
        new_brief: str,
        startup_analysis: str,
    ) -> None:
        """Mutate ``messages`` in place to replace the system prompt with a fresh brief."""
        if not messages:
            return
        if messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["content"] = self._system_prompt(new_brief, startup_analysis)

    def _trim_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= 16:
            return messages
        # Keep system prompt plus the recent task evidence. This is the context
        # cleaner that avoids old CARGO-style stale fixed points.
        tail = list(messages[-14:])
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        return [messages[0]] + tail

    def solve(
        self,
        env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        env_reset = env.reset(task_index=task_index)
        initial_user = _extract_initial_user_message(env_reset)
        selected = self.retriever.search(initial_user, k=3)
        brief = render_experience_brief(selected)
        startup = self._startup_analysis(initial_user, brief)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt(brief, startup)}]
        if initial_user:
            messages.append({"role": "user", "content": initial_user})

        cards_used_ids: List[str] = [c.card_id for c in selected]
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
        }
        reward: float = 0.0
        info: Dict[str, Any] = {}
        total_cost = 0.0
        step_error = ""
        done = False

        effective_steps = 0
        for _ in range(max_num_steps):
            # Stateful re-retrieval: every N effective steps, regenerate the
            # experience brief from the current trajectory state. This is the
            # core architectural change that turns retrieval from "look once"
            # into "look as the conversation evolves."
            if (
                self.retrieval_refresh_every
                and effective_steps > 0
                and effective_steps % self.retrieval_refresh_every == 0
            ):
                new_brief, new_selected = self._refresh_brief(initial_user, messages)
                self._replace_system_prompt(messages, new_brief, startup)
                stats["retrieval_refreshes"] += 1
                # Track distinct cards seen across refreshes for diagnostics.
                for c in new_selected:
                    if c.card_id not in cards_used_ids:
                        cards_used_ids.append(c.card_id)
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
                    messages.append({
                        "role": "tool",
                        "tool_call_id": getattr(tc, "id", f"rex_{stats['tool_calls_executed']}"),
                        "name": name,
                        "content": _obs_text(env_resp),
                    })
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

        info = dict(info) if info else {}
        if step_error:
            info["error"] = step_error
        info.setdefault("controller", "rex")
        info["rex_stats"] = stats
        return SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)


__all__ = ["RexAgent"]
