"""tau-bench airline adapter."""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from ..core import BaseCargoAdapter, Preference, TaskState
from ..schemas import GateResult, ProposedAction, ToolEffectSchema


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


class TauAirlineAdapter(BaseCargoAdapter):
    name = "tau_airline"
    benchmark_name = "tau-bench"
    domain_name = "airline"
    id_fields = {
        "user_id", "reservation_id", "reservation_number", "flight_number",
        "payment_method_id", "certificate_id", "card_id",
    }
    semantic_fields = {
        "date", "origin", "destination", "cabin", "trip_type",
        "baggage_count", "travel_insurance", "payment_preferences",
        "time_preference",
    }

    def bind_user_message(self, text: str, state: TaskState) -> List[Tuple[str, Any, bool]]:
        updates = super().bind_user_message(text, state)
        s = str(text or "")
        low = s.lower()

        route = re.search(
            r"\bfrom\s+([A-Za-z][A-Za-z .'-]{1,40}?)\s+to\s+"
            r"([A-Za-z][A-Za-z .'-]{1,40}?)(?:[.,;]| on | in | at |$)",
            s,
            re.I,
        )
        if route:
            updates += self._fact(state, "origin", _clean(route.group(1)))
            updates += self._fact(state, "destination", _clean(route.group(2)))
        depart = re.search(
            r"\b(?:departing|leaving|flying out)\s+from\s+"
            r"([A-Z]{3}|[A-Za-z][A-Za-z .'-]{1,40})(?:[.,;]| on | in | at |$)",
            s,
            re.I,
        )
        if depart:
            updates += self._fact(state, "origin", _clean(depart.group(1)))

        if "basic economy" in low:
            updates += self._fact(state, "cabin", "basic economy")
        elif "business class" in low or re.search(r"\bbusiness\b", low):
            updates += self._fact(state, "cabin", "business")
        elif re.search(r"\beconomy\b", low):
            updates += self._fact(state, "cabin", "economy")

        if re.search(r"\bround[- ]trip\b", low) or re.search(r"\breturn (?:flight|trip)\b", low):
            updates += self._fact(state, "trip_type", "round trip")
        if re.search(r"\bone[- ]way\b", low):
            updates += self._fact(state, "trip_type", "one way")

        bag_match = re.search(
            r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine)\s+"
            r"(?:checked\s+)?bags?\b",
            low,
        )
        if bag_match:
            raw = bag_match.group(1)
            updates += self._fact(state, "baggage_count", int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw))

        if re.search(r"\b(no|without|do not want|don't want)\s+(?:travel\s+)?insurance\b", low):
            updates += self._fact(state, "travel_insurance", "no")
        elif re.search(r"\b(?:buy|get|add|want|need|using my)\s+(?:travel\s+)?insurance\b", low):
            updates += self._fact(state, "travel_insurance", "yes")

        prefs: List[str] = []
        if "certificate" in low:
            prefs.append("certificate")
        if "gift card" in low or "gift_card" in low:
            prefs.append("gift_card")
        if "credit card" in low or " card" in low:
            prefs.append("credit_card")
        for pref in prefs:
            state.add_preference(Preference(slot="payment", value=pref, rank=len(state.preferences)))
            updates += self._fact(state, "payment_preference", pref)

        if re.search(r"\bcheapest\b", low):
            updates += self._fact(state, "time_preference", "cheapest")
        if re.search(r"\bdirect\b", low):
            updates += self._fact(state, "time_preference", "direct_preferred")
        if re.search(r"\bone[- ]?stop|stopover\b", low):
            updates += self._fact(state, "time_preference", "onestop_allowed")
        return updates

    def validate_write_completeness(self, action: ProposedAction, wm) -> Optional[GateResult]:
        if action.name != "book_reservation":
            return None
        args = action.args or {}
        missing: List[str] = []
        if not (args.get("user_id") or wm.auth_user_id or wm.typed_evidence_for("user_id")):
            missing.append("user_id")
        if not (args.get("flights") or args.get("flight_numbers") or args.get("flight_number")):
            missing.append("flights")
        if not args.get("passengers"):
            missing.append("passengers")
        if not (
            args.get("payment_methods")
            or args.get("payment_method_id")
            or wm.semantic_slots.get("payment_preferences")
            or wm.semantic_slots.get("payment_preference")
            or wm.typed_evidence_for("payment_method_id")
        ):
            missing.append("payment")
        if not (args.get("cabin") or args.get("cabin_class") or wm.semantic_slots.get("cabin")):
            missing.append("cabin")
        if missing:
            return GateResult.failing(
                "completeness",
                "booking_missing_required_slots",
                missing=missing,
            )
        return GateResult.passing("completeness", adapter=self.name)

    def _fact(self, state: TaskState, key: str, value: Any) -> List[Tuple[str, Any, bool]]:
        if value in (None, ""):
            return []
        state.bind_fact(key, value, source="user")
        return [(key, value, False)]


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .,;:!?")


__all__ = ["TauAirlineAdapter"]
