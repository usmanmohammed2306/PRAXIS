"""tau-bench retail adapter."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..core import (
    BaseCargoAdapter,
    CandidateObject,
    CommitCertificate,
    Constraint,
    ConstraintPriorityEngine,
    FallbackRule,
    GoalActionCandidate,
    GoalField,
    GoalHypothesis,
    Preference,
    TaskState,
    normalize_key,
)
from ..risk_class import RiskClass
from ..schemas import ProposedAction, ToolEffectSchema


class TauRetailAdapter(BaseCargoAdapter):
    name = "tau_retail"
    benchmark_name = "tau-bench"
    domain_name = "retail"
    id_fields = {
        "user_id", "order_id", "item_id", "item_ids", "new_item_ids",
        "product_id", "product_ids", "payment_method_id",
    }
    semantic_fields = {
        "product_type", "option", "color", "size", "switch_type",
        "backlight", "compatibility",
    }

    def bind_user_message(self, text: str, state: TaskState) -> List[Tuple[str, Any, bool]]:
        updates = super().bind_user_message(text, state)
        low = str(text or "").lower()

        if "clicky" in low:
            state.add_constraint(Constraint(slot="switch_type", op="eq", value="clicky", source="user", hard=True))
            updates.append(("switch_type", "clicky", False))
        if "linear" in low:
            state.add_constraint(Constraint(slot="switch_type", op="eq", value="linear", source="user", hard=True))
            updates.append(("switch_type", "linear", False))
        if re.search(r"\bfull[- ]?size\b", low):
            state.add_constraint(Constraint(slot="size", op="eq", value="full size", source="user", hard=True))
            updates.append(("size", "full size", False))
        if re.search(r"\b80%|eighty percent\b", low):
            state.add_constraint(Constraint(slot="size", op="eq", value="80%", source="user", hard=True))
            updates.append(("size", "80%", False))
        if "rgb" in low:
            state.add_preference(Preference(slot="backlight", value="rgb", rank=0))
            updates.append(("backlight_preference", "rgb", False))
        if re.search(r"\b(no backlight|without backlight|no lights?)\b", low):
            if re.search(r"\b(if|unless|unavailable|otherwise)\b", low):
                state.add_fallback(FallbackRule(slot="backlight", from_value="rgb", to_value="no backlight"))
            else:
                state.add_constraint(Constraint(slot="backlight", op="eq", value="no backlight", source="user", hard=True))
            updates.append(("backlight", "no backlight", False))
        if "google home" in low or "google assistant" in low:
            state.add_constraint(
                Constraint(slot="compatibility", op="contains", value="google", source="user", hard=True)
            )
            updates.append(("compatibility", "google", False))
        return updates

    def select_replacement_variant_id(
        self,
        details: Dict[str, Any],
        old_item: Dict[str, Any],
        goal: str,
    ) -> Optional[str]:
        """Select a replacement with hard constraints before preferences.

        This adapter method owns retail semantics.  The CARGO core supplies the
        generic priority engine; the adapter translates user wording and product
        options into constraints/preferences/fallbacks.
        """
        variants = details.get("variants") or {}
        if not isinstance(variants, dict):
            return None
        hard, prefs, fallbacks = self._variant_policy(details, old_item, goal)
        if not hard and not prefs and not fallbacks:
            return None
        old_id = str(old_item.get("item_id") or "")
        candidates: List[CandidateObject] = []
        for variant_id, variant in variants.items():
            if not isinstance(variant, dict):
                continue
            vid = str(variant.get("item_id") or variant_id)
            if vid == old_id:
                continue
            attrs = _flatten_options(variant.get("options") or {})
            attrs["available"] = bool(variant.get("available", True))
            attrs["price"] = variant.get("price")
            candidates.append(CandidateObject(
                candidate_id=vid,
                object_type=str(details.get("name") or ""),
                attributes=attrs,
                available=bool(variant.get("available", True)),
            ))
        exact_or_skip = _requires_exact_or_skip(goal)
        selection = ConstraintPriorityEngine().select(
            candidates,
            hard_constraints=hard,
            preferences=prefs,
            fallback_rules=[] if exact_or_skip else fallbacks,
        )
        if (
            selection.ok
            and selection.candidate is not None
            and prefs
            and exact_or_skip
            and not _matches_all_preferences(selection.candidate.attributes, prefs)
        ):
            return None
        return selection.candidate.candidate_id if selection.ok and selection.candidate else None

    def _variant_policy(
        self,
        details: Dict[str, Any],
        old_item: Dict[str, Any],
        goal: str,
    ) -> Tuple[List[Constraint], List[Preference], List[FallbackRule]]:
        low = str(goal or "").lower()
        old_options = old_item.get("options") or {}
        option_keys = _available_option_keys(details, old_options)

        def has_option(slot: str) -> bool:
            return normalize_key(slot) in option_keys

        hard: List[Constraint] = []
        prefs: List[Preference] = []
        fallbacks: List[FallbackRule] = []

        if has_option("switch_type") and "clicky" in low:
            hard.append(Constraint(slot="switch_type", op="eq", value="clicky", hard=True))
        elif has_option("switch_type") and "tactile" in low:
            hard.append(Constraint(slot="switch_type", op="eq", value="tactile", hard=True))
        elif has_option("switch_type") and "linear" in low:
            hard.append(Constraint(slot="switch_type", op="eq", value="linear", hard=True))

        explicit_size = False
        if has_option("size") and re.search(r"\bfull[- ]?size\b", low):
            hard.append(Constraint(slot="size", op="eq", value="full size", hard=True))
            explicit_size = True
        elif has_option("size") and re.search(r"\b80\s*%|eighty percent\b", low):
            hard.append(Constraint(slot="size", op="eq", value="80%", hard=True))
            explicit_size = True
        elif has_option("size") and re.search(r"\b60\s*%|sixty percent\b", low):
            hard.append(Constraint(slot="size", op="eq", value="60%", hard=True))
            explicit_size = True

        old_size = _option_lookup(old_options, "size")
        if (
            has_option("size")
            and old_size
            and not explicit_size
            and re.search(r"\b(similar|same|exchange|swap|replace)\b", low)
        ):
            hard.append(Constraint(slot="size", op="eq", value=old_size, hard=True))

        if has_option("compatibility") and ("google home" in low or "google assistant" in low):
            hard.append(Constraint(slot="compatibility", op="contains", value="google", hard=True))

        if has_option("backlight") and "rgb" in low:
            prefs.append(Preference(slot="backlight", value="RGB", rank=0))
        if has_option("backlight") and re.search(r"\b(no backlight|without backlight|no lights?)\b", low) and re.search(
            r"\b(if|unless|unavailable|otherwise|if not)\b", low
        ):
            fallbacks.append(FallbackRule(slot="backlight", from_value="RGB", to_value="none"))

        return _dedupe_constraints(hard), prefs, fallbacks

    def build_commit_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm,
    ) -> CommitCertificate:
        if action.name == "exchange_delivered_order_items":
            return self._exchange_certificate(action, schema, wm)
        if action.name == "return_delivered_order_items":
            return self._return_certificate(action, schema, wm)
        return super().build_commit_certificate(action, schema, wm)

    def goal_stage(self, wm) -> str:
        text = _goal_text(wm)
        if getattr(wm, "task_completed", False):
            return "complete"
        if getattr(wm, "phase_locked", lambda _p: False)("mutation"):
            return "post_write"
        if re.search(r"\b(how\s+many|number\s+of|count)\b", text) and not getattr(wm, "product_count_finalized", False):
            if not getattr(wm, "product_types", {}):
                return "catalog"
            return "catalog_details"
        if re.search(r"\b(exchange|return|modify|change|update|cancel)\b", text):
            if not getattr(wm, "auth_user_id", "") and not getattr(wm, "order_details", {}):
                return "identity_or_order_anchor"
            if not getattr(wm, "order_details", {}):
                return "order_retrieval"
            if re.search(r"\b(exchange|modify|change|update)\b", text):
                return "product_variant_retrieval"
            return "commit_ready"
        return super().goal_stage(wm)

    def update_goal_field(
        self,
        field: GoalField,
        wm,
        *,
        event: str,
        action_name: str = "",
        obs: Any = None,
    ) -> None:
        super().update_goal_field(field, wm, event=event, action_name=action_name, obs=obs)
        if event == "user":
            text = _goal_text(wm)
            if re.search(r"\b(exchange|return|modify|change|update)\b", text):
                field.add_hypothesis(GoalHypothesis(
                    hypothesis_id="retail_account_task",
                    label="retail_account_task",
                    confidence=0.75,
                    anchors={"order_ids": list(getattr(wm, "typed_evidence_for", lambda _k: [])("order_id"))[-3:]},
                    last_evidence_turn=field.turn,
                ))
            if re.search(r"\b(how\s+many|number\s+of|count)\b", text):
                field.add_hypothesis(GoalHypothesis(
                    hypothesis_id="retail_catalog_query",
                    label="retail_catalog_query",
                    confidence=0.7,
                    anchors={"product_query": True},
                    last_evidence_turn=field.turn,
                ))
        if event == "tool" and action_name == "find_user_id_by_name_zip":
            if obs is not None and "not found" in str(obs).lower():
                field.recenter("identity_lookup_failed_preserve_order_branch")
                field.add_hypothesis(GoalHypothesis(
                    hypothesis_id="order_id_recovery",
                    label="order_id_recovery",
                    confidence=0.8,
                    anchors={"order_ids": list(getattr(wm, "typed_evidence_for", lambda _k: [])("order_id"))[-3:]},
                    last_evidence_turn=field.turn,
                ))

    def score_goal_action(self, candidate: GoalActionCandidate, wm) -> float:
        action = candidate.action
        score = 0.0
        if action.name == "get_order_details" and getattr(wm, "auth_failed_zips", []):
            score += 1.1
        if action.name == "get_product_details":
            score += 0.45
        if action.name in {
            "exchange_delivered_order_items",
            "return_delivered_order_items",
            "modify_pending_order_items",
        }:
            score += 1.0
        if action.name.startswith("find_user_id_by_name_zip"):
            zip_arg = str(action.args.get("zip") or "")
            if zip_arg and zip_arg in getattr(wm, "auth_failed_zips", []):
                score -= 3.0
        if action.declared_class == RiskClass.ASK_USER and getattr(wm, "order_details", {}):
            score -= 1.0
        return score

    def _exchange_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm,
    ) -> CommitCertificate:
        cert = super().build_commit_certificate(action, schema, wm)
        cert.certificate_type = f"{self.name}.exchange_commit"
        args = action.args or {}
        order_id = str(args.get("order_id") or "").strip()
        order = wm.order_details.get(order_id) or wm.order_details.get(_canonical_order_id(order_id))
        cert.require(
            "order_loaded",
            isinstance(order, dict),
            "order_details_not_loaded",
            order_id=order_id,
        )
        if not isinstance(order, dict):
            return cert.finalize()
        cert.require(
            "order_is_delivered",
            _order_status(order) == "delivered",
            "order_not_delivered",
            status=_order_status(order),
        )
        cert.require(
            "identity_or_order_anchor",
            _identity_or_order_anchor_ok(order, order_id, wm),
            "write_lacks_identity_or_user_supplied_order_anchor",
            order_user_id=order.get("user_id"),
            auth_user_id=getattr(wm, "auth_user_id", ""),
        )
        old_ids = _as_string_list(args.get("item_ids"))
        new_ids = _as_string_list(args.get("new_item_ids"))
        cert.require(
            "one_to_one_item_mapping",
            bool(old_ids) and len(old_ids) == len(new_ids),
            "exchange_item_id_mapping_incomplete",
            item_ids=old_ids,
            new_item_ids=new_ids,
        )
        included = dict(zip(old_ids, new_ids))
        goal = _goal_text(wm)
        expected: Dict[str, str] = {}
        missing_details: List[str] = []
        no_replacement: List[str] = []
        skipped_by_user: List[str] = []
        for item in order.get("items") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            pid = str(item.get("product_id") or "")
            old_id = str(item.get("item_id") or "").strip()
            if not old_id or not pid or not _product_name_matches_user(name.lower(), goal):
                continue
            details = wm.product_details.get(pid)
            if not isinstance(details, dict):
                missing_details.append(pid)
                continue
            selected = self.select_replacement_variant_id(details, item, goal)
            if selected:
                expected[old_id] = str(selected)
            elif _requires_exact_or_skip(goal):
                skipped_by_user.append(old_id)
            else:
                no_replacement.append(old_id)
        cert.selected_candidate_ids.extend(expected.values())
        cert.require(
            "product_details_loaded_for_requested_items",
            not missing_details,
            "requested_product_details_missing",
            product_ids=missing_details,
        )
        cert.require(
            "requested_items_have_valid_replacements_or_allowed_skip",
            not no_replacement,
            "requested_item_has_no_proven_replacement",
            item_ids=no_replacement,
            skipped_by_user=skipped_by_user,
        )
        missing = [old_id for old_id in expected if old_id not in included]
        extra = [old_id for old_id in included if old_id not in expected]
        wrong = [
            {"item_id": old_id, "expected_new_item_id": expected[old_id], "proposed_new_item_id": included[old_id]}
            for old_id in expected
            if old_id in included and included[old_id] != expected[old_id]
        ]
        cert.require(
            "all_requested_items_included",
            not missing,
            "write_is_partial_for_active_goal",
            missing_item_ids=missing,
            expected_item_ids=sorted(expected.keys()),
            proposed_item_ids=old_ids,
        )
        cert.require(
            "no_unrequested_items_included",
            not extra,
            "write_includes_unrequested_items",
            extra_item_ids=extra,
        )
        cert.require(
            "selected_candidates_match_constraints",
            not wrong,
            "selected_candidate_not_proven_best",
            mismatches=wrong,
        )
        payment_ok = bool(str(args.get("payment_method_id") or "").strip())
        cert.require(
            "payment_method_grounded",
            payment_ok,
            "payment_method_missing",
            payment_method_id=args.get("payment_method_id"),
        )
        return cert.finalize()

    def _return_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm,
    ) -> CommitCertificate:
        cert = super().build_commit_certificate(action, schema, wm)
        cert.certificate_type = f"{self.name}.return_commit"
        args = action.args or {}
        order_id = str(args.get("order_id") or "").strip()
        order = wm.order_details.get(order_id) or wm.order_details.get(_canonical_order_id(order_id))
        cert.require("order_loaded", isinstance(order, dict), "order_details_not_loaded", order_id=order_id)
        if not isinstance(order, dict):
            return cert.finalize()
        cert.require("order_is_delivered", _order_status(order) == "delivered", "order_not_delivered")
        cert.require(
            "identity_or_order_anchor",
            _identity_or_order_anchor_ok(order, order_id, wm),
            "write_lacks_identity_or_user_supplied_order_anchor",
        )
        goal = _goal_text(wm)
        expected = [
            str(item.get("item_id") or "").strip()
            for item in order.get("items") or []
            if isinstance(item, dict)
            and str(item.get("item_id") or "").strip()
            and _product_name_matches_user(str(item.get("name") or "").lower(), goal)
        ]
        proposed = _as_string_list(args.get("item_ids"))
        cert.selected_candidate_ids.extend(proposed)
        cert.require(
            "all_requested_items_included",
            set(expected).issubset(set(proposed)),
            "write_is_partial_for_active_goal",
            expected_item_ids=expected,
            proposed_item_ids=proposed,
        )
        cert.require(
            "no_unrequested_items_included",
            set(proposed).issubset(set(expected)),
            "write_includes_unrequested_items",
            extra_item_ids=[item_id for item_id in proposed if item_id not in expected],
        )
        cert.require(
            "payment_method_grounded",
            bool(str(args.get("payment_method_id") or "").strip()),
            "payment_method_missing",
            payment_method_id=args.get("payment_method_id"),
        )
        return cert.finalize()


def _flatten_options(options: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in options.items():
        out[normalize_key(str(key))] = value
    return out


def _option_lookup(options: Dict[str, Any], name: str) -> Optional[Any]:
    n = normalize_key(name)
    for key, value in options.items():
        if normalize_key(str(key)) == n:
            return value
    return None


def _available_option_keys(details: Dict[str, Any], old_options: Dict[str, Any]) -> set[str]:
    keys = {normalize_key(str(k)) for k in (old_options or {}).keys()}
    variants = details.get("variants") or {}
    if isinstance(variants, dict):
        for variant in variants.values():
            if not isinstance(variant, dict):
                continue
            options = variant.get("options") or {}
            if isinstance(options, dict):
                keys.update(normalize_key(str(k)) for k in options.keys())
    return keys


def _matches_all_preferences(attrs: Dict[str, Any], prefs: List[Preference]) -> bool:
    for pref in prefs:
        actual = attrs.get(normalize_key(pref.slot))
        if normalize_key(str(actual)) != normalize_key(str(pref.value)):
            return False
    return True


def _requires_exact_or_skip(goal: str) -> bool:
    low = str(goal or "").lower()
    return bool(
        re.search(r"\brather\s+only\s+exchange\b", low)
        or re.search(r"\bjust\s+exchange\b", low)
        or re.search(r"\bonly\s+exchange\s+the\s+other\b", low)
    )


def _dedupe_constraints(items: List[Constraint]) -> List[Constraint]:
    out: List[Constraint] = []
    seen = set()
    for item in items:
        sig = (item.slot, item.op, str(item.value).lower())
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def _canonical_order_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("#"):
        return text
    return "#" + text


def _goal_text(wm) -> str:
    return (str(getattr(wm, "goal", "") or "") + " " + " ".join(getattr(wm, "user_facts", []) or [])).lower()


def _order_status(order: Dict[str, Any]) -> str:
    return str(order.get("status") or "").strip().lower()


def _identity_or_order_anchor_ok(order: Dict[str, Any], order_id: str, wm) -> bool:
    auth_user_id = str(getattr(wm, "auth_user_id", "") or "").strip()
    order_user_id = str(order.get("user_id") or "").strip()
    if auth_user_id and (not order_user_id or auth_user_id == order_user_id):
        return True
    goal = _goal_text(wm)
    oid = str(order_id or order.get("order_id") or "").strip()
    if not oid:
        return False
    bare = oid.lstrip("#").lower()
    return oid.lower() in goal or bare in goal


def _product_name_matches_user(name_norm: str, all_user: str) -> bool:
    if not name_norm or not all_user:
        return False
    name_spaced = re.sub(r"[-_]+", " ", name_norm)
    user_spaced = re.sub(r"[-_]+", " ", all_user)
    if name_spaced in user_spaced or name_norm in all_user:
        return True
    words = name_norm.split()
    sig_words = [w for w in words if len(w) >= 4]
    if len(words) > 1 and sig_words:
        last_word = words[-1]
        if len(last_word) >= 4 and re.search(rf"\b{re.escape(last_word)}s?\b", all_user):
            return True
        return all(re.search(rf"\b{re.escape(w)}s?\b", all_user) for w in sig_words)
    if sig_words:
        tok = sig_words[0]
        if re.search(rf"\b{re.escape(tok)}s?\b", all_user):
            return True
        if tok.endswith("s") and len(tok) > 4:
            return bool(re.search(rf"\b{re.escape(tok[:-1])}s?\b", all_user))
    compound = re.sub(r"[^a-z0-9]+", "", name_norm)
    user_compound = re.sub(r"[^a-z0-9]+", "", all_user)
    return bool(len(compound) >= 6 and compound in user_compound)


__all__ = ["TauRetailAdapter"]
