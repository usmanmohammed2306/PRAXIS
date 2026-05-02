"""CARGO agent for tau-bench.

Drop-in replacement for the baseline agents (vanilla / Act / ReAct). The
agent uses a custom JSON-emitting proposer (no native tool-calling) so it
can declare risk class, pre/post-conditions, and free-form user text in
one structured response. Each declared mutation passes through the
calibrated gate before reaching ``env.step``.

This file exists at the integration boundary with tau-bench. The kernel
of CARGO (gates / schemas / repair) lives in pure-Python modules with no
tau-bench dependency, so it can be unit-tested without a live env.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:  # tau-bench is optional at import-time so unit tests can run without it.
    from tau_bench.agents.base import Agent
    from tau_bench.types import Action, SolveResult
    _HAS_TAU = True
except Exception:  # noqa: BLE001
    Agent = object  # type: ignore[assignment]
    Action = None  # type: ignore[assignment]
    SolveResult = None  # type: ignore[assignment]
    _HAS_TAU = False

from . import repair
from .calibration import default_calibration
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
from .stats import CargoStats
from .working_memory import WorkingMemory


def _get_openai_client():
    """Lazy import of the project's OpenAI client (depends on ``openai`` SDK).

    Imported inside CargoAgent.__init__ so this module is importable in
    environments without the OpenAI SDK (e.g. unit tests that pass a
    MockClient instead of going through ``get_client``).
    """
    from ..common.openai_client import get_client
    return get_client()


RESPOND_TOOL_NAME = "respond"
RESPOND_MAX_CHARS = 800


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _initial_user_message(env_reset: Any) -> str:
    if env_reset is None:
        return ""
    for attr in ("observation", "content", "message", "user_message"):
        v = getattr(env_reset, attr, None)
        if v:
            return str(v)
    if isinstance(env_reset, str):
        return env_reset
    return str(env_reset)


# ---------------------------------------------------------------------------
# CargoAgent
# ---------------------------------------------------------------------------
class CargoAgent(Agent):  # type: ignore[misc]
    """tau-bench Agent that runs the CARGO loop."""

    style_name = "cargo"

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str = "openai",
        temperature: float = 0.0,
        env_hint: str = "",
    ) -> None:
        if not _HAS_TAU:
            # Graceful "not implemented" if someone instantiates this without
            # tau_bench installed; the test suite avoids this path.
            raise RuntimeError(
                "tau_bench is not installed; CargoAgent cannot be instantiated."
            )
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.model = model
        self.provider = provider
        self.temperature = float(temperature)
        self.env_hint = env_hint
        self.client = _get_openai_client()
        # Schema induction is cached at module level — repeated agent
        # instances within the same process never re-induce.
        self.schemas: Dict[str, ToolEffectSchema] = induce_schemas(
            self.tools_info,
            client=self.client,
            model=self.model,
            temperature=0.0,
        )
        self.calibration = default_calibration()

    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        parts: List[str] = [SYSTEM_PROMPT]
        if self.wiki:
            parts.append("--- Domain policy ---\n" + self.wiki.strip())
        return "\n\n".join(parts)

    def _build_proposer_messages(
        self,
        wm: WorkingMemory,
        history: List[Dict[str, Any]],
        critique: str,
    ) -> List[Dict[str, Any]]:
        tools_block = render_tools_block(self.schemas)
        # n_turns=8 messages ≈ 2-3 complete steps; compression keeps it ≤300 tokens.
        history_tail = trim_history(history, n_turns=8, max_chars=800)
        user_msg = render_proposer_user_message(
            wm=wm,
            tools_block=tools_block,
            history_tail=history_tail,
            critique=critique,
            domain_policy="",  # already in the system prompt
        )
        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_msg},
        ]

    def _call_proposer(
        self,
        proposer_messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 600,
    ) -> Tuple[Optional[ProposedAction], str]:
        T = self.temperature if temperature is None else float(temperature)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=proposer_messages,
                temperature=T,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if _is_context_overflow(exc):
                # Trim and retry once.
                trimmed = [proposer_messages[0]] + proposer_messages[-1:]
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=trimmed,
                        temperature=T,
                        max_tokens=max_tokens,
                    )
                except Exception:
                    return None, ""
            else:
                return None, ""
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            text = ""
        parsed = parse_proposer_response(text, schemas=self.schemas)
        return parsed, text

    # ------------------------------------------------------------------
    # Gate orchestration
    # ------------------------------------------------------------------
    def _run_gates(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm: WorkingMemory,
        proposer_messages: List[Dict[str, Any]],
        stats: CargoStats,
    ) -> Tuple[Optional[GateResult], Dict[str, Any]]:
        """Run the gate stack on a proposed action.

        Returns (failing_gate, diagnostics). If failing_gate is None,
        the action passed every gate.
        """
        diag: Dict[str, Any] = {"gates_run": [], "gates_failed": []}

        # Repeat-loop is checked for every class — cheap and high-yield.
        rl = check_repeat_loop(action, schema, wm)
        diag["gates_run"].append("repeat_loop")
        stats.record_gate(rl)
        if not rl.ok:
            diag["gates_failed"].append("repeat_loop")
            return rl, diag

        if not is_gated(action.declared_class):
            # Even on the fast path, run arg_grounding so hallucinated
            # placeholder values (e.g. "user@example.com") are caught before
            # the tool executes and wastes the first step.
            ag = check_arg_grounding(action, schema, wm)
            diag["gates_run"].append("arg_grounding")
            stats.record_gate(ag)
            if not ag.ok:
                diag["gates_failed"].append("arg_grounding")
                return ag, diag
            return None, diag

        # 4a. Pre-conditions
        pc = check_preconditions(action, schema, wm)
        diag["gates_run"].append("preconditions")
        stats.record_gate(pc)
        if not pc.ok:
            diag["gates_failed"].append("preconditions")
            return pc, diag

        # 4b. Argument grounding
        ag = check_arg_grounding(action, schema, wm)
        diag["gates_run"].append("arg_grounding")
        stats.record_gate(ag)
        if not ag.ok:
            diag["gates_failed"].append("arg_grounding")
            return ag, diag

        # 4c. Self-consistency (k samples per class).
        k = self.calibration.sc_k.get(action.declared_class, 0)
        threshold = self.calibration.sc_thresholds.get(action.declared_class, 0.0)
        if k >= 1 and threshold > 0:
            sc = check_self_consistency(
                action, schema,
                client=self.client,
                model=self.model,
                proposer_messages=proposer_messages,
                schemas_for_parse=self.schemas,
                k=k,
                threshold=threshold,
            )
            diag["gates_run"].append("self_consistency")
            stats.record_gate(sc)
            diag["sc_agreement"] = sc.diagnostics.get("agreement")
            if not sc.ok:
                diag["gates_failed"].append("self_consistency")
                return sc, diag

        # 4d. Counterfactual rollout (IRREV / FINAL only).
        if self.calibration.run_cf.get(action.declared_class, False) and is_irreversible_or_final(action.declared_class):
            cf = check_counterfactual(
                action, schema, wm,
                client=self.client, model=self.model,
                temperature=0.0,
            )
            diag["gates_run"].append("counterfactual")
            stats.record_gate(cf)
            diag["cf_predicted_blocking"] = bool(cf.diagnostics.get("cf_reason")) and not cf.ok
            if not cf.ok:
                diag["gates_failed"].append("counterfactual")
                return cf, diag

        return None, diag

    # ------------------------------------------------------------------
    # Main loop (mirrors the baseline contract for tau-bench)
    # ------------------------------------------------------------------
    def solve(
        self,
        env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> "SolveResult":
        env_reset = env.reset(task_index=task_index)
        initial_user = _initial_user_message(env_reset)

        wm = WorkingMemory(goal=initial_user, budget_steps=max_num_steps)
        wm.absorb_user_message(initial_user)

        # ``messages`` is the *trajectory* surface tau-bench / scorers see.
        # We render it in the same OpenAI-chat shape the baselines use so
        # downstream tooling (judges, replay, label classifiers) is happy.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
        ]
        if initial_user:
            messages.append({"role": "user", "content": initial_user})

        stats = CargoStats()
        reward: float = 0.0
        info: Dict[str, Any] = {}
        total_cost: float = 0.0
        done = False
        step_error: str = ""

        critique = ""
        retries_used = 0
        consecutive_parse_failures = 0
        MAX_CONSEC_PARSE_FAIL = 3

        for step in range(max_num_steps):
            wm.budget_steps = max_num_steps - step
            stats.steps_total += 1

            proposer_messages = self._build_proposer_messages(wm, messages, critique)
            action, raw_text = self._call_proposer(proposer_messages)
            messages.append({"role": "assistant", "content": raw_text or ""})

            if action is None:
                consecutive_parse_failures += 1
                stats.json_parse_failures += 1
                if consecutive_parse_failures >= MAX_CONSEC_PARSE_FAIL:
                    step_error = "json_parse_failures_exceeded"
                    # Force a polite finalize.
                    env_resp = self._respond(
                        env,
                        "I'm having trouble formatting my response. "
                        "Could you confirm the details and try again?",
                    )
                    if env_resp is None:
                        break
                    user_reply = _obs_text(env_resp)
                    if user_reply:
                        messages.append({"role": "user", "content": user_reply})
                        wm.absorb_user_message(user_reply)
                    reward = _float(getattr(env_resp, "reward", reward), reward)
                    info = getattr(env_resp, "info", info) or info
                    done = bool(getattr(env_resp, "done", False))
                    if done:
                        break
                    consecutive_parse_failures = 0
                    continue
                # Soft retry with a JSON-format critique.
                gr = GateResult.failing("json_parse", "unparseable_proposer_output")
                stats.record_gate(gr)
                stats.abstain_total += 1
                rd = repair.decide(
                    gr,
                    retries_used=retries_used,
                    max_retries=2,
                    budget_steps_remaining=wm.budget_steps,
                )
                if rd.action == "RETRY":
                    critique = rd.critique
                    retries_used += 1
                    stats.repair_retry += 1
                    continue
                # Fallthrough = ASK_USER / FINALIZE_GENERIC: send a respond.
                env_resp = self._respond(env, rd.user_message)
                if env_resp is None:
                    break
                user_reply = _obs_text(env_resp)
                if user_reply:
                    messages.append({"role": "user", "content": user_reply})
                    wm.absorb_user_message(user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                if rd.action == "ASK_USER":
                    stats.repair_ask_user += 1
                else:
                    stats.repair_finalize += 1
                if done:
                    break
                critique = ""
                retries_used = 0
                continue

            consecutive_parse_failures = 0
            schema = self._schema_for(action)
            failing, diag = self._run_gates(action, schema, wm, proposer_messages, stats)

            step_record: Dict[str, Any] = {
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
                stats.steps_gated += 1
                stats.abstain_total += 1
                rd = repair.decide(
                    failing,
                    proposed=action,
                    retries_used=retries_used,
                    max_retries=2,
                    budget_steps_remaining=wm.budget_steps,
                )
                step_record["repair_action"] = rd.action
                stats.record_step(step_record)

                if rd.action == "RETRY":
                    critique = rd.critique
                    retries_used += 1
                    stats.repair_retry += 1
                    continue
                # ASK_USER / FINALIZE_GENERIC → send a respond.
                env_resp = self._respond(env, rd.user_message or
                                         "Could you provide a bit more detail so I can proceed?")
                if env_resp is None:
                    break
                user_reply = _obs_text(env_resp)
                if user_reply:
                    messages.append({"role": "user", "content": user_reply})
                    wm.absorb_user_message(user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                if rd.action == "ASK_USER":
                    stats.repair_ask_user += 1
                else:
                    stats.repair_finalize += 1
                if done:
                    break
                critique = ""
                retries_used = 0
                continue

            # --- Action passed every gate. Execute. --------------------
            critique = ""
            retries_used = 0
            if is_gated(action.declared_class):
                stats.steps_gated += 1
            else:
                stats.steps_fast_path += 1

            wm.record_action_signature(action.signature())

            # FINAL / ASK_USER → route through respond with the model's user_text.
            if (action.declared_class in (RiskClass.FINAL, RiskClass.ASK_USER)
                    or action.name.lower() in (RESPOND_TOOL_NAME, "send_user", "finish", "final", "answer")):
                content = (action.user_text or action.raw_thought or "").strip()
                if not content:
                    content = "Is there anything else I can help you with?"
                env_resp = self._respond(env, content)
                if env_resp is None:
                    break
                # Log the translated tool call for trajectory consumers.
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": f"cargo_{step}",
                        "type": "function",
                        "function": {
                            "name": RESPOND_TOOL_NAME,
                            "arguments": json.dumps({"content": content[:RESPOND_MAX_CHARS]}),
                        },
                    }],
                })
                user_reply = _obs_text(env_resp)
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"cargo_{step}",
                    "name": RESPOND_TOOL_NAME,
                    "content": user_reply,
                })
                if user_reply:
                    wm.absorb_user_message(user_reply)
                reward = _float(getattr(env_resp, "reward", reward), reward)
                info = getattr(env_resp, "info", info) or info
                done = bool(getattr(env_resp, "done", False))
                stats.actions_executed += 1
                cls = action.declared_class.value
                stats.executed_by_class[cls] = stats.executed_by_class.get(cls, 0) + 1
                step_record["executed"] = True
                stats.record_step(step_record)
                if done:
                    break
                continue

            # Real tool dispatch.
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
            try:
                env_resp = env.step(Action(name=action.name, kwargs=dict(action.args)))
            except Exception as env_exc:
                if _is_context_overflow(env_exc):
                    step_error = f"env_step_context_overflow: {env_exc}"
                    done = True
                    break
                # Don't crash the trajectory: log the failure as an obs and
                # let the agent decide how to recover.
                err_obs = json.dumps({"status": "error", "error": str(env_exc)})
                messages.append({
                    "role": "tool",
                    "tool_call_id": translated_call_id,
                    "name": action.name,
                    "content": err_obs,
                })
                wm.last_error = str(env_exc)[:80]  # tip 8: minimal failure record
                wm.absorb_observation({"status": "error", "error": str(env_exc)[:80]})
                step_record["env_error"] = str(env_exc)[:80]
                stats.record_step(step_record)
                continue

            tool_obs = _obs_text(env_resp)
            messages.append({
                "role": "tool",
                "tool_call_id": translated_call_id,
                "name": action.name,
                "content": tool_obs,
            })
            # Surface tool-level errors in last_error so the proposer's STATE
            # block reflects why the previous step failed, not just the
            # post-condition label.
            if tool_obs and (
                tool_obs.lstrip().lower().startswith("error")
                or '"error"' in tool_obs.lower()
            ):
                wm.last_error = tool_obs[:80]
            wm.absorb_observation(tool_obs if not isinstance(env_resp, type(None))
                                  else "")
            # Post-condition check (advisory).
            try:
                post_obs = json.loads(tool_obs) if tool_obs.strip().startswith(("{", "[")) else tool_obs
            except Exception:
                post_obs = tool_obs
            pc = check_postconditions(action, schema, obs=post_obs)
            stats.record_gate(pc)
            if not pc.ok:
                wm.last_error = pc.reason[:80]  # tip 8: minimal failure record
                step_record["post_error"] = pc.reason

            reward = _float(getattr(env_resp, "reward", reward), reward)
            info = getattr(env_resp, "info", info) or info
            done = bool(getattr(env_resp, "done", False))
            stats.actions_executed += 1
            cls = action.declared_class.value
            stats.executed_by_class[cls] = stats.executed_by_class.get(cls, 0) + 1
            step_record["executed"] = True
            stats.record_step(step_record)
            if done:
                break

        info = dict(info) if info else {}
        if step_error:
            info["error"] = step_error
        info.setdefault("controller", self.style_name)
        info["cargo_stats"] = stats.snapshot()
        return SolveResult(  # type: ignore[call-arg]
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _schema_for(self, action: ProposedAction) -> ToolEffectSchema:
        """Return the schema for ``action.name``, synthesising one for unknown
        tools so the gate logic always has something to check."""
        if action.name in self.schemas:
            return self.schemas[action.name]
        # Synthesise a minimal schema using the declared class.
        from .schemas import ToolEffectSchema
        return ToolEffectSchema(
            name=action.name,
            cls=action.declared_class,
            irreversible=is_irreversible_or_final(action.declared_class),
            preconditions=[],
            postconditions=[],
            arg_id_fields=[],
            param_properties={},
            required_params=[],
            description="(unknown tool — synthesised schema)",
        )

    def _respond(self, env, content: str):
        """Issue a ``respond`` action to the env. Returns the env response,
        or ``None`` if the env raised a non-recoverable error."""
        msg = content.strip()[:RESPOND_MAX_CHARS] or "Is there anything else I can help with?"
        try:
            return env.step(Action(name=RESPOND_TOOL_NAME, kwargs={"content": msg}))
        except Exception as env_exc:
            if _is_context_overflow(env_exc):
                return None
            raise


__all__ = ["CargoAgent"]
