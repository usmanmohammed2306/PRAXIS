"""ACEBench loop for REx-RPE.

ACEBench Agent is scored offline against emitted tool calls. The loop therefore
uses retrieval and startup analysis, but does not block tool calls with mutation
reflection because no live state is mutated in this driver.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .experience import ExperienceRetriever, load_experience_cards, render_experience_brief


def _system_prompt(task: Dict[str, Any], brief: str, startup: str = "") -> str:
    sys_msg = task.get("system") or task.get("system_prompt") or ""
    parts = [
        "You are REx-RPE, a confident multi-step tool agent.",
        "Use native tool calls. Make one tool call at a time and wait for the simulated result.",
        "Retrieved examples are procedural memory only; never copy arguments from examples.",
        "\n--- Retrieved Experience Brief ---\n" + brief,
    ]
    if startup:
        parts.append("\n--- Startup Analysis ---\n" + startup)
    if isinstance(sys_msg, str) and sys_msg.strip():
        parts.append("\n--- Task system prompt ---\n" + sys_msg.strip())
    return "\n".join(parts)


def _startup_analysis(client, model: str, user_turn: str, brief: str) -> str:
    if os.environ.get("REX_STARTUP_ANALYSIS", "1") == "0":
        return ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": (
                    "Return <=100 words: intent, first tool, required parameters, and likely API sequence. "
                    "Do not call tools.\n\n"
                    f"Experience brief:\n{brief}\n\nUser request:\n{user_turn}"
                ),
            }],
            temperature=0.0,
            max_tokens=160,
        )
        return (resp.choices[0].message.content or "").strip()[:800]
    except Exception:
        return ""


def _assistant_message_dict(msg: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
    tcs = getattr(msg, "tool_calls", None) or []
    if tcs:
        out["tool_calls"] = [
            {
                "id": getattr(tc, "id", f"rex_ace_{i}"),
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for i, tc in enumerate(tcs)
        ]
    return out


def _tool_brief_cards(tool_specs: Sequence[Dict[str, Any]], user_turn: str) -> List[Any]:
    cards = load_experience_cards("ace", Path(os.environ.get("REX_EXPERIENCE_DIR", str(Path.cwd() / "outputs" / "experience_bank"))))
    return ExperienceRetriever(cards).search(user_turn + " " + " ".join(_tool_names(tool_specs)), k=3)


def _tool_names(tool_specs: Sequence[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for t in tool_specs or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def run_rex(
    *,
    client,
    model: str,
    task: Dict[str, Any],
    tool_specs: List[Dict[str, Any]],
    user_turn: str,
    max_num_steps: int,
    temperature: float,
) -> Dict[str, Any]:
    selected = _tool_brief_cards(tool_specs, user_turn)
    brief = render_experience_brief(selected)
    startup = _startup_analysis(client, model, user_turn, brief)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": _system_prompt(task, brief, startup)}]
    if user_turn:
        messages.append({"role": "user", "content": user_turn})

    tool_calls_made: List[str] = []
    status = "ok"
    error = ""
    try:
        for _ in range(max_num_steps):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_specs or None,
                tool_choice="auto" if tool_specs else None,
                temperature=temperature,
            )
            msg = resp.choices[0].message
            messages.append(_assistant_message_dict(msg))
            tcs = getattr(msg, "tool_calls", None) or []
            if not tcs:
                break
            stub = json.dumps({
                "status": "simulated",
                "note": "ACEBench tool results are evaluated offline.",
            })
            for tc in tcs:
                tool_calls_made.append(tc.function.name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tc, "id", f"rex_ace_{len(tool_calls_made)}"),
                    "name": tc.function.name,
                    "content": stub,
                })
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "tool_calls_made": tool_calls_made,
        "rex_stats": {
            "cards_loaded": len(load_experience_cards("ace")),
            "cards_used": [c.card_id for c in selected],
            "startup_analysis": bool(startup),
        },
    }


__all__ = ["run_rex"]
