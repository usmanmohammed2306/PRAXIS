"""tau-bench retail adapter."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..core import (
    BaseCargoAdapter,
    CandidateObject,
    Constraint,
    ConstraintPriorityEngine,
    FallbackRule,
    Preference,
    TaskState,
    normalize_key,
)


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
        selection = ConstraintPriorityEngine().select(
            candidates,
            hard_constraints=hard,
            preferences=prefs,
            fallback_rules=fallbacks,
        )
        if (
            selection.ok
            and selection.candidate is not None
            and prefs
            and selection.fallback_used is None
            and _requires_exact_or_skip(goal)
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


__all__ = ["TauRetailAdapter"]
