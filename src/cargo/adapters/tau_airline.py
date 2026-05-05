"""tau-bench airline adapter."""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from ..core import BaseCargoAdapter, Obligation, Preference, TaskState
from ..schemas import GateResult, ProposedAction, ToolEffectSchema


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

_AIRPORT_ALIASES = {
    "atlanta": "ATL",
    "boston": "BOS",
    "charlotte": "CLT",
    "chicago": "ORD",
    "dallas": "DFW",
    "denver": "DEN",
    "detroit": "DTW",
    "houston": "IAH",
    "las vegas": "LAS",
    "los angeles": "LAX",
    "minneapolis": "MSP",
    "newark": "EWR",
    "new york": "JFK",
    "orlando": "MCO",
    "philadelphia": "PHL",
    "phoenix": "PHX",
    "san francisco": "SFO",
    "seattle": "SEA",
}
_REGION_ALIASES = {"texas": {"DFW", "IAH"}}


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
        "payment_preference", "time_preference", "time_after", "intent",
    }

    def bind_user_message(self, text: str, state: TaskState) -> List[Tuple[str, Any, bool]]:
        updates = super().bind_user_message(text, state)
        s = str(text or "")
        low = s.lower()

        if re.search(r"\b(book|reserve|purchase)\b", low) and re.search(r"\b(flight|ticket|trip)\b", low):
            updates += self._fact(state, "intent", "book_flight")
            _add_requested_operation(state, "book_flight")
        if re.search(r"\b(change|modify|update|reschedule|switch)\b", low) and re.search(r"\b(flight|reservation|trip|ticket)\b", low):
            updates += self._fact(state, "intent", "modify_flight")
            _add_requested_operation(state, "modify_flight")

        uid = re.search(r"\b([a-z]+_[a-z]+_\d{1,8})\b", low)
        if uid:
            updates += self._fact(state, "user_id", uid.group(1))

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
        after = re.search(r"\bafter\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", low)
        if after:
            hour = int(after.group(1))
            minute = int(after.group(2) or 0)
            ampm = after.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            updates += self._fact(state, "time_after", f"{hour:02d}:{minute:02d}")
        return updates

    def update_obligations(self, state: TaskState, wm) -> None:
        intents = _state_values(state, "intent")
        text = " ".join(str(v).lower() for v in intents)
        user_text = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        if "book_flight" in text or (
            re.search(r"\b(book|reserve|purchase)\b", user_text)
            and re.search(r"\b(flight|ticket|trip)\b", user_text)
        ):
            required = {"origin", "destination", "date", "cabin"}
            if re.search(r"\bchecked\s+bags?|bags?\b", user_text):
                required.add("baggage_count")
            if "insurance" in user_text:
                required.add("travel_insurance")
            state.upsert_obligation(Obligation(
                obligation_id="book_flight",
                operation_type="book_flight",
                required_slots=required,
                candidate_retrieval_needs={"direct_flights"},
                selected_candidate_needs={"itinerary"},
                confirmation_required=True,
                execution_required=True,
                status="open",
            ))
        elif "modify_flight" in text or (
            re.search(r"\b(change|modify|update|reschedule|switch)\b", user_text)
            and re.search(r"\b(flight|reservation|trip|ticket)\b", user_text)
        ):
            state.upsert_obligation(Obligation(
                obligation_id="modify_flight",
                operation_type="modify_flight",
                required_slots={"date"},
                candidate_retrieval_needs={"reservation_profile", "flight_options"},
                selected_candidate_needs={"reservation", "replacement_itinerary"},
                confirmation_required=True,
                execution_required=True,
                status="open",
            ))
        state.refresh_open_slots()

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

    def validate_ask_user(self, action: ProposedAction, wm) -> GateResult:
        text = str(action.user_text or action.raw_thought or "").lower()
        if _is_payment_question(text) and not _flight_candidate_selected(wm):
            return GateResult.failing(
                "state_validity",
                "payment_question_before_flight_selection",
                adapter=self.name,
                validation_level="ask_user",
            )
        return super().validate_ask_user(action, wm)

    def validate_final_completeness(self, action: ProposedAction, wm) -> Optional[GateResult]:
        if wm.task_state.terminal_status == "blocked_no_matching_flights":
            return GateResult.passing(
                "final_completeness",
                adapter=self.name,
                terminal_status=wm.task_state.terminal_status,
            )
        return super().validate_final_completeness(action, wm)

    def canonicalize_airport(self, value: Any, *, field: str = "") -> str:
        return _airport_code(value) or _clean(str(value or ""))

    def semantic_values_match(self, key: str, proposed: Any, expected: Any) -> bool:
        if key not in {"origin", "destination"}:
            return False
        proposed_code = _airport_code(proposed)
        expected_values = expected if isinstance(expected, list) else [expected]
        for value in expected_values:
            expected_code = _airport_code(value)
            if proposed_code and expected_code and proposed_code == expected_code:
                return True
            expected_region = _region_airports(value)
            if proposed_code and expected_region and proposed_code in expected_region:
                return True
            if _clean(str(proposed or "")).lower() == _clean(str(value or "")).lower():
                return True
        return False

    def _fact(self, state: TaskState, key: str, value: Any) -> List[Tuple[str, Any, bool]]:
        if value in (None, ""):
            return []
        state.bind_fact(key, value, source="user")
        if key == "intent" and value not in state.intent:
            state.intent.append(str(value))
        return [(key, value, False)]


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .,;:!?")


def _airport_code(value: Any) -> str:
    raw = _clean(str(value or ""))
    if re.fullmatch(r"[A-Za-z]{3}", raw):
        return raw.upper()
    norm = re.sub(r"[^a-z0-9 ]+", " ", raw.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return ""
    if norm in _AIRPORT_ALIASES:
        return _AIRPORT_ALIASES[norm]
    # Phrases such as "only EWR, not JFK" or "Seattle airport" should still
    # reduce to the actionable IATA token when one is present.
    m = re.search(r"\b([A-Za-z]{3})\b", raw)
    if m and m.group(1).upper() in set(_AIRPORT_ALIASES.values()):
        return m.group(1).upper()
    for phrase, code in sorted(_AIRPORT_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", norm):
            return code
    return ""


def _region_airports(value: Any) -> set[str]:
    raw = _clean(str(value or ""))
    norm = re.sub(r"[^a-z0-9 ]+", " ", raw.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return set(_REGION_ALIASES.get(norm, set()))


def _add_requested_operation(state: TaskState, operation: str) -> None:
    if operation not in state.requested_operations:
        state.requested_operations.append(operation)


def _state_values(state: TaskState, key: str) -> List[Any]:
    fact = state.get_fact(key)
    vals: List[Any] = []
    if key == "intent":
        vals.extend(state.intent)
    if fact is not None:
        vals.append(fact.value)
    vals.extend(p.value for p in state.preferences if p.slot == key)
    return vals


def _is_payment_question(text: str) -> bool:
    return bool(
        re.search(
            r"\b(payment|pay|card|credit\s+card|gift\s+card|certificate|"
            r"voucher|billing)\b",
            text,
        )
    )


def _flight_candidate_selected(wm) -> bool:
    state = getattr(wm, "task_state", None)
    if state is None:
        return False
    selected = getattr(state, "selected_objects", {}) or {}
    if any(k in selected for k in ("itinerary", "flight", "flights")):
        return True
    # Some future adapters may bind selected flight numbers as semantic slots.
    slots = getattr(wm, "semantic_slots", {}) or {}
    return bool(slots.get("selected_flight") or slots.get("selected_itinerary"))


__all__ = ["TauAirlineAdapter"]
