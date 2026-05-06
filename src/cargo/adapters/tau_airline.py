"""tau-bench airline adapter."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..core import (
    BaseCargoAdapter,
    CommitCertificate,
    GoalActionCandidate,
    GoalField,
    GoalHypothesis,
    Obligation,
    Preference,
    TaskState,
    normalize_key,
)
from ..risk_class import RiskClass
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
    task_frame_fields = {
        "origin", "destination", "date", "cabin", "cabin_class",
        "trip_type", "flight_type", "baggage_count", "total_baggages",
        "nonfree_baggages", "travel_insurance", "insurance",
        "payment_preference", "payment_preferences", "time_preference",
        "time_after",
    }

    def absorb_observation(self, obs: Any, state: TaskState, action_name: str = "") -> List[Tuple[str, Any, bool]]:
        """Store airline tool evidence without overwriting the active goal.

        Reservation/profile/search observations carry fields named
        ``origin``, ``destination``, ``date``, and ``cabin`` for many objects.
        Those are candidate/current-reservation facts, not automatically the
        user's requested route/date/cabin.  The generic state still records
        them as DB-confirmed evidence, but the adapter does not return them as
        task-frame slot updates for ``WorkingMemory.semantic_slots``.
        """
        updates = super().absorb_observation(obs, state, action_name)
        task_frame_fields = {
            "date", "origin", "destination", "cabin", "trip_type",
            "baggage_count", "travel_insurance", "insurance",
            "time_preference", "time_after", "flight_type",
        }
        filtered: List[Tuple[str, Any, bool]] = []
        for key, value, confirmed in updates:
            leaf = normalize_key(str(key).split(".")[-1])
            if leaf in task_frame_fields:
                continue
            filtered.append((key, value, confirmed))
        return filtered

    def bind_user_message(self, text: str, state: TaskState) -> List[Tuple[str, Any, bool]]:
        updates = super().bind_user_message(text, state)
        s = str(text or "")
        low = s.lower()

        route_like = bool(re.search(r"\bfrom\s+[A-Za-z][A-Za-z .'-]{1,40}?\s+to\s+[A-Za-z]", s, re.I))
        booking_verb = bool(
            re.search(r"\b(book|reserve|purchase|fly|flying)\b", low)
            or re.search(r"\b(?:travel|go)\s+(?:from|to)\b", low)
        )
        if booking_verb and (re.search(r"\b(flight|ticket|trip)\b", low) or route_like):
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

    def absorb_observation(
        self,
        obs: Any,
        state: TaskState,
        action_name: str = "",
    ) -> List[Tuple[str, Any, bool]]:
        """Treat airline tool data as evidence, not as the active goal frame.

        Reservation/profile reads often contain unrelated routes, cabins,
        dates, and insurance choices from historical bookings.  Those facts
        are valuable for candidate selection, but letting them overwrite the
        user-bound task frame is exactly how a booking request for New York →
        Seattle drifts into a cached Denver → Las Vegas reservation.  Opaque
        IDs and payment facts still bind normally.
        """
        updates: List[Tuple[str, Any, bool]] = []
        for key, value in _iter_airline_scalars(_coerce_airline_obs(obs)):
            leaf = normalize_key(str(key).split(".")[-1])
            if leaf in self.task_frame_fields:
                state.conflicts.append({
                    "key": leaf,
                    "value": value,
                    "source_action": action_name,
                    "reason": "cached_airline_fact_quarantined_from_task_frame",
                })
                continue
            state.bind_fact(key, value, source="tool", confirmed=True)
            updates.append((key, value, True))
        return updates

    def update_obligations(self, state: TaskState, wm) -> None:
        intents = _state_values(state, "intent")
        text = " ".join(str(v).lower() for v in intents)
        user_text = (wm.goal + " " + " ".join(wm.user_facts)).lower()
        route_like = bool(re.search(r"\bfrom\s+[A-Za-z][A-Za-z .'-]{1,40}?\s+to\s+[A-Za-z]", wm.goal + " " + " ".join(wm.user_facts), re.I))
        booking_verb = bool(
            re.search(r"\b(book|reserve|purchase|fly|flying)\b", user_text)
            or re.search(r"\b(?:travel|go)\s+(?:from|to)\b", user_text)
        )
        if "book_flight" in text or (
            booking_verb and (re.search(r"\b(flight|ticket|trip)\b", user_text) or route_like)
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

    def build_commit_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm,
    ) -> CommitCertificate:
        if action.name == "book_reservation":
            return self._book_certificate(action, schema, wm)
        if action.name.startswith("update_reservation_") or action.name == "cancel_reservation":
            return self._reservation_write_certificate(action, schema, wm)
        return super().build_commit_certificate(action, schema, wm)

    def goal_stage(self, wm) -> str:
        if getattr(wm, "task_completed", False):
            return "complete"
        if getattr(wm.task_state, "terminal_status", ""):
            return "terminal"
        slots = getattr(wm, "semantic_slots", {}) or {}
        user_id = getattr(wm, "auth_user_id", "") or (getattr(wm, "typed_evidence_for", lambda _k: [])("user_id") or [""])[-1]
        intents = slots.get("intents") or slots.get("intent") or []
        if not isinstance(intents, list):
            intents = [intents]
        intent_text = " ".join(str(v).lower() for v in intents)
        if not user_id and re.search(r"\b(book|reserve|purchase|change|modify|update|cancel)\b", _goal_text(wm)):
            return "identity"
        if user_id and user_id not in getattr(wm, "user_profiles", {}):
            return "profile"
        if "book_flight" in intent_text:
            if not (slots.get("origin") and slots.get("destination") and slots.get("date")):
                return "bind_route"
            direct = wm.task_state.candidate_set_for(
                "search_direct_flight",
                {
                    "origin": self.canonicalize_airport(slots.get("origin"), field="origin"),
                    "destination": self.canonicalize_airport(slots.get("destination"), field="destination"),
                    "date": str(slots.get("date")),
                },
            )
            one = wm.task_state.candidate_set_for(
                "search_onestop_flight",
                {
                    "origin": self.canonicalize_airport(slots.get("origin"), field="origin"),
                    "destination": self.canonicalize_airport(slots.get("destination"), field="destination"),
                    "date": str(slots.get("date")),
                },
            )
            if not direct:
                return "search_direct"
            if direct.exhausted and not one:
                return "search_onestop"
            return "candidate_selection"
        if "modify_flight" in intent_text or getattr(wm, "reservation_details", {}):
            return "reservation_scan" if not getattr(wm, "reservation_details", {}) else "reservation_selection"
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
        slots = getattr(wm, "semantic_slots", {}) or {}
        intents = slots.get("intents") or slots.get("intent") or []
        if not isinstance(intents, list):
            intents = [intents]
        for intent in intents[-3:]:
            if intent:
                field.add_hypothesis(GoalHypothesis(
                    hypothesis_id=normalize_key(str(intent)),
                    label=str(intent),
                    confidence=0.75,
                    anchors={
                        "origin": slots.get("origin"),
                        "destination": slots.get("destination"),
                        "date": slots.get("date"),
                    },
                    last_evidence_turn=field.turn,
                ))
        if event == "tool" and action_name in {"search_direct_flight", "search_onestop_flight"}:
            if obs is not None and str(obs).strip() not in {"", "[]"} and "[]" not in str(obs).strip()[:4]:
                field.record_progress("flight_candidates_available", 0.8)
        if event in {"gate_failure", "execute"} and action_name == "get_user_details":
            field.record_friction("cached_profile_loop", 0.6, "profile_phase_complete")

    def score_goal_action(self, candidate: GoalActionCandidate, wm) -> float:
        action = candidate.action
        score = 0.0
        slots = getattr(wm, "semantic_slots", {}) or {}
        if action.name == "get_user_details":
            uid = str(action.args.get("user_id") or "").strip()
            if uid and uid in getattr(wm, "user_profiles", {}):
                score -= 3.0
        if action.name.startswith("search_") and slots.get("origin") and slots.get("destination") and slots.get("date"):
            score += 0.9
        if action.name == "get_reservation_details":
            score += 0.55
        if action.name in {"book_reservation", "cancel_reservation"} or action.name.startswith("update_reservation_"):
            score += 1.0
        if action.declared_class == RiskClass.ASK_USER and (slots.get("origin") or getattr(wm, "reservation_details", {})):
            score -= 1.4
        if action.declared_class == RiskClass.FINAL and getattr(wm.task_state, "terminal_status", ""):
            score += 1.4
        return score

    def _book_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm,
    ) -> CommitCertificate:
        cert = super().build_commit_certificate(action, schema, wm)
        cert.certificate_type = f"{self.name}.book_commit"
        args = action.args or {}
        user_id = str(args.get("user_id") or getattr(wm, "auth_user_id", "") or "").strip()
        grounded_user_ids = set(getattr(wm, "typed_evidence_for", lambda _k: [])("user_id"))
        grounded_user_ids.update(getattr(wm, "user_profiles", {}).keys())
        if getattr(wm, "auth_user_id", ""):
            grounded_user_ids.add(str(getattr(wm, "auth_user_id")))
        cert.require(
            "identity_grounded",
            bool(user_id and user_id in grounded_user_ids),
            "booking_user_id_not_grounded",
            user_id=user_id,
        )
        for slot in ("origin", "destination", "date", "cabin"):
            expected = (getattr(wm, "semantic_slots", {}) or {}).get(slot)
            proposed = args.get(slot)
            if slot == "cabin":
                proposed = proposed or args.get("cabin_class")
            if slot == "date" and proposed in (None, "", []):
                flight_dates = [
                    str(f.get("date") or "").strip()
                    for f in _flight_entries(args)
                    if str(f.get("date") or "").strip()
                ]
                proposed = flight_dates[0] if len(set(flight_dates)) == 1 else flight_dates
            if expected in (None, "", []):
                continue
            cert.require(
                f"active_goal_{slot}_consistent",
                _airline_slot_matches(self, slot, proposed, expected),
                f"booking_{slot}_conflicts_with_active_goal",
                expected=expected,
                proposed=proposed,
            )
        flights = _flight_entries(args)
        flight_numbers = [str(f.get("flight_number") or "").strip() for f in flights if str(f.get("flight_number") or "").strip()]
        cert.selected_candidate_ids.extend(flight_numbers)
        seen_flights = _seen_flight_numbers(wm)
        cert.require(
            "selected_flights_grounded",
            bool(flight_numbers) and all(num in seen_flights for num in flight_numbers),
            "selected_flight_not_in_evidence",
            flight_numbers=flight_numbers,
            seen_flights=sorted(seen_flights),
        )
        payment_ids = _payment_ids_from_book_args(args)
        seen_payments = _seen_payment_ids(wm)
        cert.require(
            "payment_methods_grounded",
            bool(payment_ids) and all(pid in seen_payments for pid in payment_ids),
            "booking_payment_method_not_grounded",
            payment_ids=payment_ids,
            seen_payment_ids=sorted(seen_payments),
        )
        cert.require(
            "passengers_complete",
            bool(args.get("passengers")),
            "booking_passengers_missing",
            passengers=args.get("passengers"),
        )
        return cert.finalize()

    def _reservation_write_certificate(
        self,
        action: ProposedAction,
        schema: ToolEffectSchema,
        wm,
    ) -> CommitCertificate:
        cert = super().build_commit_certificate(action, schema, wm)
        cert.certificate_type = f"{self.name}.reservation_commit"
        args = action.args or {}
        reservation_id = str(args.get("reservation_id") or args.get("reservation_number") or "").strip()
        reservation = (getattr(wm, "reservation_details", {}) or {}).get(reservation_id)
        grounded_ids = set(getattr(wm, "typed_evidence_for", lambda _k: [])("reservation_id"))
        cert.require(
            "reservation_grounded",
            bool(reservation_id and (reservation_id in grounded_ids or isinstance(reservation, dict))),
            "reservation_id_not_grounded",
            reservation_id=reservation_id,
        )
        if isinstance(reservation, dict):
            auth_user_id = str(getattr(wm, "auth_user_id", "") or "").strip()
            reservation_user_id = str(reservation.get("user_id") or "").strip()
            cert.require(
                "reservation_identity_consistent",
                not auth_user_id or not reservation_user_id or auth_user_id == reservation_user_id,
                "reservation_user_conflicts_with_confirmed_identity",
                auth_user_id=auth_user_id,
                reservation_user_id=reservation_user_id,
            )
        if action.name == "update_reservation_flights":
            flights = _flight_entries(args)
            cert.selected_candidate_ids.extend(
                str(f.get("flight_number") or "").strip()
                for f in flights
                if str(f.get("flight_number") or "").strip()
            )
            cert.require(
                "replacement_itinerary_complete",
                bool(flights),
                "replacement_itinerary_missing",
                flights=flights,
            )
        if action.name in {"update_reservation_flights", "update_reservation_baggages"}:
            payment_id = str(args.get("payment_id") or args.get("payment_method_id") or "").strip()
            seen_payments = _seen_payment_ids(wm)
            cert.require(
                "payment_method_grounded",
                bool(payment_id and payment_id in seen_payments),
                "reservation_payment_method_not_grounded",
                payment_id=payment_id,
                seen_payment_ids=sorted(seen_payments),
            )
        return cert.finalize()

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


def _coerce_airline_obs(obs: Any) -> Any:
    if isinstance(obs, (dict, list)):
        return obs
    if isinstance(obs, str):
        text = obs.strip()
        if not text:
            return None
        if text[0] in "[{":
            try:
                return json.loads(text)
            except Exception:
                return None
    return None


def _iter_airline_scalars(value: Any, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_airline_scalars(item, path)
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _iter_airline_scalars(item, f"{prefix}[{idx}]")
        return
    if value not in (None, "") and prefix:
        yield prefix, value


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


def _airline_slot_matches(adapter: TauAirlineAdapter, slot: str, proposed: Any, expected: Any) -> bool:
    if proposed in (None, "", []):
        return False
    if slot in {"origin", "destination"}:
        return adapter.semantic_values_match(slot, proposed, expected)
    p = _clean(str(proposed or "")).lower().replace("_", " ")
    vals = expected if isinstance(expected, list) else [expected]
    for value in vals:
        e = _clean(str(value or "")).lower().replace("_", " ")
        if p == e or (p and e and (p in e or e in p)):
            return True
    return False


def _flight_entries(args: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = args.get("flights") or args.get("flight_numbers") or args.get("flight_number")
    if isinstance(raw, list):
        entries: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                entries.append(dict(item))
            elif item not in (None, ""):
                entries.append({"flight_number": str(item)})
        return entries
    if raw in (None, ""):
        return []
    return [{"flight_number": str(raw)}]


def _seen_flight_numbers(wm) -> set[str]:
    seen: set[str] = set()
    for value in getattr(wm, "typed_evidence_for", lambda _k: [])("flight_number"):
        if str(value).strip():
            seen.add(str(value).strip())
    state = getattr(wm, "task_state", None)
    for candidate in getattr(state, "candidate_objects", {}).values() if state else []:
        cid = str(getattr(candidate, "candidate_id", "") or "").strip()
        if cid:
            seen.add(cid)
        attrs = getattr(candidate, "attributes", {}) or {}
        for key in ("flight_number", "id"):
            val = str(attrs.get(key) or "").strip()
            if val:
                seen.add(val)
        for flight in attrs.get("flights") or attrs.get("value") or []:
            if isinstance(flight, Mapping):
                val = str(flight.get("flight_number") or flight.get("id") or "").strip()
                if val:
                    seen.add(val)
    text = getattr(wm, "all_evidence", lambda: "")()
    seen.update(re.findall(r"\bHAT\d{3}\b", text))
    for match in re.finditer(r"flight_number[\"'=:\s]+([A-Za-z0-9_/-]+)", text):
        seen.add(match.group(1).strip("\"'"))
    return seen


def _payment_ids_from_book_args(args: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    raw = args.get("payment_methods") or args.get("payment_method_id") or args.get("payment_id")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                val = item.get("payment_id") or item.get("payment_method_id") or item.get("id")
                if val not in (None, ""):
                    out.append(str(val).strip())
            elif item not in (None, ""):
                out.append(str(item).strip())
    elif raw not in (None, ""):
        out.append(str(raw).strip())
    return [value for value in out if value]


def _seen_payment_ids(wm) -> set[str]:
    seen: set[str] = set()
    for key in ("payment_method_id", "payment_id", "card_id", "certificate_id"):
        for value in getattr(wm, "typed_evidence_for", lambda _k: [])(key):
            if str(value).strip():
                seen.add(str(value).strip())
    text = getattr(wm, "all_evidence", lambda: "")()
    seen.update(re.findall(r"\b(?:credit_card|gift_card|certificate)_[0-9A-Za-z]+\b", text))
    return seen


def _goal_text(wm) -> str:
    return (str(getattr(wm, "goal", "") or "") + " " + " ".join(getattr(wm, "user_facts", []) or [])).lower()


__all__ = ["TauAirlineAdapter"]
