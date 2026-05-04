"""CARGO loop for ACEBench Agent (offline-stub scoring).

ACEBench's Agent split is graded offline against the saved trajectory:
the runner stubs every tool result and what's measured is the *sequence
of tool calls* the agent chooses to issue. This loop mirrors the
tau-bench CARGO agent but uses the offline stub for tool observations,
so the trajectory looks identical to the baseline runs (same OpenAI-chat
shape, same set of tool names emitted) — only the controller differs.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from . import repair
from .calibration import default_calibration
from .stats import CargoStats
from .gates import (
    check_arg_grounding,
    check_counterfactual,
    check_postconditions,
    check_preconditions,
    check_repeat_loop,
    check_self_consistency,
)
from .proposer import (
    SYSTEM_PROMPT,
    parse_proposer_response,
    render_proposer_user_message,
    render_tools_block,
    trim_history,
)
from .risk_class import RiskClass, is_gated, is_irreversible_or_final
from .schema_inducer import induce_schemas
from .schemas import GateResult, ProposedAction, ToolEffectSchema
from .working_memory import WorkingMemory


_RESPOND = "respond"
_STUB_OBS = json.dumps({
    "status": "simulated",
    "note": "ACEBench tool results are evaluated offline.",
})


def _system_prompt(domain_policy: str = "") -> str:
    parts: List[str] = [SYSTEM_PROMPT]
    if domain_policy and domain_policy.strip():
        parts.append("--- Domain policy ---\n" + domain_policy.strip())
    return "\n\n".join(parts)


def _schema_for(name: str, declared: RiskClass,
                schemas: Dict[str, ToolEffectSchema]) -> ToolEffectSchema:
    if name in schemas:
        return schemas[name]
    return ToolEffectSchema(
        name=name,
        cls=declared,
        irreversible=is_irreversible_or_final(declared),
        preconditions=[],
        postconditions=[],
        arg_id_fields=[],
        param_properties={},
        required_params=[],
        description="(unknown — synthesised)",
    )


def _run_gates(
    *,
    action: ProposedAction,
    schema: ToolEffectSchema,
    wm: WorkingMemory,
    proposer_messages: List[Dict[str, Any]],
    schemas: Dict[str, ToolEffectSchema],
    client,
    model: str,
    calibration,
    stats: CargoStats,
):
    diag: Dict[str, Any] = {"gates_run": [], "gates_failed": []}
    rl = check_repeat_loop(action, schema, wm)
    diag["gates_run"].append("repeat_loop")
    stats.record_gate(rl)
    if not rl.ok:
        diag["gates_failed"].append("repeat_loop")
        return rl, diag
    if not is_gated(action.declared_class):
        ag = check_arg_grounding(action, schema, wm)
        diag["gates_run"].append("arg_grounding")
        stats.record_gate(ag)
        if not ag.ok:
            diag["gates_failed"].append("arg_grounding")
            return ag, diag
        return None, diag
    pc = check_preconditions(action, schema, wm)
    diag["gates_run"].append("preconditions")
    stats.record_gate(pc)
    if not pc.ok:
        diag["gates_failed"].append("preconditions")
        return pc, diag
    ag = check_arg_grounding(action, schema, wm)
    diag["gates_run"].append("arg_grounding")
    stats.record_gate(ag)
    if not ag.ok:
        diag["gates_failed"].append("arg_grounding")
        return ag, diag
    if not getattr(action, "bypass_gates", False):
        k = calibration.sc_k.get(action.declared_class, 0)
        threshold = calibration.sc_thresholds.get(action.declared_class, 0.0)
        if k >= 1 and threshold > 0:
            sc = check_self_consistency(
                action, schema,
                client=client, model=model,
                proposer_messages=proposer_messages,
                schemas_for_parse=schemas,
                k=k, threshold=threshold,
            )
            diag["gates_run"].append("self_consistency")
            diag["sc_agreement"] = sc.diagnostics.get("agreement")
            stats.record_gate(sc)
            if not sc.ok:
                diag["gates_failed"].append("self_consistency")
                return sc, diag
        if (calibration.run_cf.get(action.declared_class, False)
                and is_irreversible_or_final(action.declared_class)):
            cf = check_counterfactual(
                action, schema, wm,
                client=client, model=model, temperature=0.0,
            )
            diag["gates_run"].append("counterfactual")
            diag["cf_predicted_blocking"] = bool(cf.diagnostics.get("cf_reason")) and not cf.ok
            stats.record_gate(cf)
            if not cf.ok:
                diag["gates_failed"].append("counterfactual")
                return cf, diag
    return None, diag


def run_cargo(
    *,
    client,
    model: str,
    task: Dict[str, Any],
    tool_specs: List[Dict[str, Any]],
    user_turn: str,
    system_prompt: str,
    max_num_steps: int,
    temperature: float,
) -> Dict[str, Any]:
    """ACEBench CARGO loop. Returns the per-task record dict expected by
    the ACEBench runner: ``status``, ``error``, ``messages``,
    ``tool_calls_made``, ``cargo_stats``."""
    domain_policy = system_prompt or ""
    sys_blob = _system_prompt(domain_policy=domain_policy)

    schemas = induce_schemas(
        tool_specs, client=client, model=model, temperature=0.0,
    )
    calibration = default_calibration()

    wm = WorkingMemory(goal=user_turn, budget_steps=max_num_steps)
    if user_turn:
        wm.absorb_user_message(user_turn)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": sys_blob}]
    if user_turn:
        messages.append({"role": "user", "content": user_turn})

    tool_calls_made: List[str] = []
    stats = CargoStats()
    status = "ok"
    error = ""
    critique = ""
    retries_used = 0
    consecutive_parse_failures = 0
    MAX_PARSE_FAIL = 3

    try:
        for step in range(max_num_steps):
            wm.budget_steps = max_num_steps - step
            stats.steps_total += 1

            tools_block = render_tools_block(schemas)
            history_tail = trim_history(messages, n_turns=8)
            user_msg = render_proposer_user_message(
                wm=wm,
                tools_block=tools_block,
                history_tail=history_tail,
                critique=critique,
                domain_policy="",
            )
            proposer_messages = [
                {"role": "system", "content": sys_blob},
                {"role": "user", "content": user_msg},
            ]
            try:
                resp = client.chat.completions.create(
                    model=model, messages=proposer_messages,
                    temperature=temperature, max_tokens=600,
                )
                raw_text = (resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error = f"chat_completion_failed: {exc.__class__.__name__}: {exc}"
                break

            messages.append({"role": "assistant", "content": raw_text})
            action = parse_proposer_response(raw_text, schemas=schemas)

            if action is None:
                consecutive_parse_failures += 1
                stats.json_parse_failures += 1
                if consecutive_parse_failures >= MAX_PARSE_FAIL:
                    error = "json_parse_failures_exceeded"
                    break
                gr = GateResult.failing("json_parse", "unparseable_proposer_output")
                stats.record_gate(gr)
                stats.abstain_total += 1
                rd = repair.decide(
                    gr, retries_used=retries_used, max_retries=2,
                    budget_steps_remaining=wm.budget_steps,
                )
                if rd.action == "RETRY":
                    critique = rd.critique
                    retries_used += 1
                    stats.repair_retry += 1
                    continue
                # ASK_USER / FINALIZE → emit a respond stub and break.
                tool_calls_made.append(_RESPOND)
                stats.repair_finalize += 1
                break

            consecutive_parse_failures = 0
            schema = _schema_for(action.name, action.declared_class, schemas)
            failing, diag = _run_gates(
                action=action, schema=schema, wm=wm,
                proposer_messages=proposer_messages, schemas=schemas,
                client=client, model=model, calibration=calibration,
                stats=stats,
            )
            step_record = {
                "step": step,
                "action_name": action.name,
                "declared_class": action.declared_class.value,
                "args": dict(action.args),
                "thought": action.raw_thought[:200],
                "fast_path": not is_gated(action.declared_class),
                "gates_run": diag.get("gates_run", []),
                "gates_failed": diag.get("gates_failed", []),
                "abstain_reason": failing.reason if failing else "",
                "sc_agreement": diag.get("sc_agreement"),
                "cf_predicted_blocking": diag.get("cf_predicted_blocking"),
            }

            if failing is not None:
                wm.record_failed_signature(action.signature())
                stats.steps_gated += 1
                stats.abstain_total += 1
                rd = repair.decide(
                    failing, proposed=action,
                    retries_used=retries_used, max_retries=2,
                    budget_steps_remaining=wm.budget_steps,
                )
                step_record["repair_action"] = rd.action
                stats.record_step(step_record)
                if rd.action == "RETRY":
                    critique = rd.critique
                    retries_used += 1
                    stats.repair_retry += 1
                    continue
                tool_calls_made.append(_RESPOND)
                stats.repair_ask_user += int(rd.action == "ASK_USER")
                stats.repair_finalize += int(rd.action == "FINALIZE_GENERIC")
                break

            critique = ""
            retries_used = 0
            wm.record_action_signature(action.signature())

            translated_call_id = f"cargo_{step}_{action.name}"
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": translated_call_id,
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": json.dumps(action.args, default=str),
                    },
                }],
            })
            tool_calls_made.append(action.name)
            stats.actions_executed += 1
            cls = action.declared_class.value
            stats.executed_by_class[cls] = stats.executed_by_class.get(cls, 0) + 1
            stats.steps_gated += int(is_gated(action.declared_class))
            stats.steps_fast_path += int(not is_gated(action.declared_class))

            # FINAL / ASK_USER / respond ⇒ end of trajectory in offline mode.
            terminal = (
                action.declared_class in (RiskClass.FINAL, RiskClass.ASK_USER)
                or action.name.lower() in (_RESPOND, "send_user", "finish", "final", "answer")
            )
            stub = _STUB_OBS if not terminal else json.dumps(
                {"status": "simulated", "note": "final."}
            )
            messages.append({
                "role": "tool",
                "tool_call_id": translated_call_id,
                "name": action.name,
                "content": stub,
            })
            step_record["executed"] = True
            stats.record_step(step_record)
            if not terminal:
                # Absorb stub as a fact so subsequent steps don't repeat.
                wm.absorb_observation(json.loads(stub))
            else:
                break
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "status": status,
        "error": error,
        "messages": messages,
        "tool_calls_made": tool_calls_made,
        "cargo_stats": stats.snapshot(),
    }


__all__ = ["run_cargo"]
