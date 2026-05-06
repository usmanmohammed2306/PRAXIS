"""ACEBench loop for REx-RPE.

Same architecture as :mod:`src.rex.agent`: hybrid retrieval, working-state
queries, tactical playbook synthesis, and per-step retrieval logging.
ACEBench is offline-scored — there is no live env to mutate — so mutation
reflection is a no-op here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import RexConfig
from .experience import (
    ExperienceRetriever,
    load_experience_cards,
    render_experience_brief,
)
from .memory_types import from_experience_card
from .pipeline import refresh_playbook
from .playbook import synthesize_playbook
from .retrieval import HybridRetriever, build_query
from .retrieval_logging import RetrievalLogger


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


def _startup_analysis(client, model: str, user_turn: str, brief: str, *, enabled: bool, max_words: int) -> str:
    if not enabled:
        return ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": (
                    f"Return <={max_words} words: intent, first tool, required parameters, "
                    "and likely API sequence. Do not call tools.\n\n"
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


def _ace_runtime_dir(config: Optional[RexConfig] = None) -> Path:
    if config:
        return Path(config.runtime_dir)
    env_dir = os.environ.get("REX_RUNTIME_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.cwd() / "outputs" / "experience_runtime"


def _ace_bank_dir(config: Optional[RexConfig] = None) -> Path:
    if config:
        return Path(config.bank_dir)
    env_dir = os.environ.get("REX_EXPERIENCE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.cwd() / "outputs" / "experience_bank"


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
    config: Optional[RexConfig] = None,
    retrieval_logger: Optional[RetrievalLogger] = None,
) -> Dict[str, Any]:
    cfg = config or RexConfig.from_env()
    bank_dir = _ace_bank_dir(cfg)
    runtime_dir = _ace_runtime_dir(cfg)
    legacy_cards = load_experience_cards("ace", bank_dir, runtime_dir=runtime_dir)
    legacy_retriever = ExperienceRetriever(legacy_cards)
    process_cards = [from_experience_card(c) for c in legacy_cards]

    hybrid: Optional[HybridRetriever]
    try:
        hybrid = HybridRetriever(process_cards, config=cfg) if process_cards else None
    except Exception:
        hybrid = None

    logger = retrieval_logger or RetrievalLogger(config=cfg)

    if hybrid and hybrid.num_cards:
        # Use hybrid pipeline for the initial brief.
        query = build_query(
            initial_user=user_turn,
            messages=[],
            environment="ace",
            benchmark="ACEBench",
            controller="rex",
            step_index=0,
            top_k=cfg.top_k,
        )
        result = hybrid.search(query)
        playbook = synthesize_playbook(result.cards, config=cfg)
        brief = playbook.body
        cards_used_ids: List[str] = result.card_ids()
        logger.log(
            benchmark="ACEBench",
            environment="ace",
            controller="rex",
            task_id=str(task.get("id") or task.get("task_id") or "0"),
            trial=0,
            step_index=0,
            query=query,
            result=result,
            playbook=playbook,
        )
    else:
        selected = legacy_retriever.search(
            user_turn + " " + " ".join(_tool_names(tool_specs)), k=cfg.top_k,
        )
        brief = render_experience_brief(selected)
        cards_used_ids = [c.card_id for c in selected]

    startup = _startup_analysis(
        client, model, user_turn, brief,
        enabled=bool(cfg.startup_analysis),
        max_words=int(cfg.startup_analysis_max_words),
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(task, brief, startup)},
    ]
    if user_turn:
        messages.append({"role": "user", "content": user_turn})

    refresh_every = int(cfg.retrieval_refresh_every or 0)
    refreshes = 0

    tool_calls_made: List[str] = []
    status = "ok"
    error = ""
    try:
        for step_idx in range(max_num_steps):
            # Stateful re-retrieval mid-trajectory.
            if (
                refresh_every
                and step_idx > 0
                and step_idx % refresh_every == 0
                and hybrid
                and hybrid.num_cards
            ):
                playbook, result, _state = refresh_playbook(
                    retriever=hybrid,
                    initial_user=user_turn,
                    messages=messages,
                    environment="ace",
                    benchmark="ACEBench",
                    controller="rex",
                    step_index=step_idx,
                    config=cfg,
                    logger=logger,
                    task_id=str(task.get("id") or task.get("task_id") or "0"),
                    trial=0,
                )
                refreshes += 1
                for c in result.cards:
                    if c.card_id not in cards_used_ids:
                        cards_used_ids.append(c.card_id)
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = _system_prompt(task, playbook.body, startup)

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
            "cards_loaded": len(legacy_cards),
            "cards_used": cards_used_ids,
            "startup_analysis": bool(startup),
            "retrieval_refreshes": refreshes,
            "retrieval_backend": "hybrid" if hybrid else "legacy",
            "playbook_chars": len(brief),
        },
    }


__all__ = ["run_rex"]
