"""Generated CARGO-N corpus regressions.

The fixture contains 400 scenario records mined from the latest 800 tau
trajectories shared in Downloads:

* trajectories (56)/(60): retail
* trajectories (57)/(59): airline

These tests do not replay hidden benchmark answers.  Each generated method
checks a controller invariant represented by the source trajectory cluster.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cargo import (  # noqa: E402
    CandidateObject,
    CandidateSet,
    GenericCargoKernel,
    GoalActionCandidate,
    GoalField,
    PredictiveGradientScheduler,
    PreCommitVerifier,
    ProposedAction,
    RiskClass,
    SoftGoalFieldRouter,
    ToolEffectSchema,
    WorkingMemory,
)
from src.cargo.adapters import TauAirlineAdapter, TauRetailAdapter  # noqa: E402
from src.cargo.cargo_agent import CargoAgent  # noqa: E402
from src.cargo.gates import check_arg_grounding  # noqa: E402


FIXTURE = ROOT / "tests/fixtures/cargo_n_corpus_cases.json"


def _load_cases() -> List[Dict[str, Any]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if len(cases) != 400:
        raise AssertionError(f"expected 400 corpus cases, got {len(cases)}")
    return cases


def _action(
    name: str,
    args: Dict[str, Any] | None = None,
    cls: RiskClass = RiskClass.READ,
    text: str = "",
) -> ProposedAction:
    return ProposedAction(
        name=name,
        args=dict(args or {}),
        declared_class=cls,
        raw_thought=text or name,
        user_text=text,
    )


class CargoNCorpusRegressionTests(unittest.TestCase):
    maxDiff = None

    def _make_airline_agent(self) -> CargoAgent:
        agent = CargoAgent.__new__(CargoAgent)
        agent.adapter = TauAirlineAdapter()
        agent.kernel = GenericCargoKernel(agent.adapter)
        agent.schemas = {
            "respond": ToolEffectSchema(name="respond", cls=RiskClass.ASK_USER),
            "get_user_details": ToolEffectSchema(
                name="get_user_details",
                cls=RiskClass.READ,
                arg_id_fields=["user_id"],
                required_params=["user_id"],
                param_properties={"user_id": {"type": "string"}},
            ),
            "get_reservation_details": ToolEffectSchema(
                name="get_reservation_details",
                cls=RiskClass.READ,
                arg_id_fields=["reservation_id"],
                required_params=["reservation_id"],
                param_properties={"reservation_id": {"type": "string"}},
            ),
            "search_direct_flight": ToolEffectSchema(
                name="search_direct_flight",
                cls=RiskClass.READ,
                arg_semantic_fields=["origin", "destination", "date"],
                required_params=["origin", "destination", "date"],
                param_properties={
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
            ),
            "search_onestop_flight": ToolEffectSchema(
                name="search_onestop_flight",
                cls=RiskClass.READ,
                arg_semantic_fields=["origin", "destination", "date"],
                required_params=["origin", "destination", "date"],
                param_properties={
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
            ),
            "book_reservation": ToolEffectSchema(
                name="book_reservation",
                cls=RiskClass.WRITE,
                arg_id_fields=["user_id", "payment_methods"],
            ),
            "update_reservation_flights": ToolEffectSchema(
                name="update_reservation_flights",
                cls=RiskClass.WRITE,
                arg_id_fields=["reservation_id", "payment_id"],
            ),
        }
        return agent

    def _make_retail_agent(self) -> CargoAgent:
        agent = CargoAgent.__new__(CargoAgent)
        agent.adapter = TauRetailAdapter()
        agent.kernel = GenericCargoKernel(agent.adapter)
        agent.schemas = {
            "respond": ToolEffectSchema(name="respond", cls=RiskClass.ASK_USER),
            "find_user_id_by_email": ToolEffectSchema(
                name="find_user_id_by_email",
                cls=RiskClass.READ,
                required_params=["email"],
                param_properties={"email": {"type": "string"}},
            ),
            "find_user_id_by_name_zip": ToolEffectSchema(
                name="find_user_id_by_name_zip",
                cls=RiskClass.READ,
                required_params=["first_name", "last_name", "zip"],
                param_properties={
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "zip": {"type": "string"},
                },
            ),
            "get_user_details": ToolEffectSchema(
                name="get_user_details",
                cls=RiskClass.READ,
                arg_id_fields=["user_id"],
            ),
            "get_order_details": ToolEffectSchema(
                name="get_order_details",
                cls=RiskClass.READ,
                arg_id_fields=["order_id"],
            ),
            "list_all_product_types": ToolEffectSchema(name="list_all_product_types", cls=RiskClass.READ),
            "get_product_details": ToolEffectSchema(
                name="get_product_details",
                cls=RiskClass.READ,
                arg_id_fields=["product_id"],
            ),
            "exchange_delivered_order_items": ToolEffectSchema(
                name="exchange_delivered_order_items",
                cls=RiskClass.WRITE,
                arg_id_fields=["order_id", "item_ids", "new_item_ids", "payment_method_id"],
            ),
            "return_delivered_order_items": ToolEffectSchema(
                name="return_delivered_order_items",
                cls=RiskClass.IRREVERSIBLE,
                arg_id_fields=["order_id", "item_ids", "payment_method_id"],
            ),
        }
        return agent

    def _run_case(self, case: Dict[str, Any]) -> None:
        category = str(case["category"])
        if category.startswith("airline_"):
            self._run_airline_case(case)
            return
        if category.startswith("retail_"):
            self._run_retail_case(case)
            return
        self._run_core_case(case)

    # ------------------------------------------------------------------
    # Airline invariants
    # ------------------------------------------------------------------
    def _airline_memory(self, goal: str = "") -> WorkingMemory:
        wm = WorkingMemory(goal=goal or "Book an economy flight from New York to Seattle on May 20th.")
        text = wm.goal + " My user id is mia_li_3668."
        wm.absorb_user_message(text)
        GenericCargoKernel(TauAirlineAdapter()).observe_user_message(wm, text)
        wm.auth_user_id = "mia_li_3668"
        return wm

    def _run_airline_case(self, case: Dict[str, Any]) -> None:
        category = case["category"]
        if category in {
            "airline_known_user_ask_suppression",
            "airline_generic_ask_after_evidence",
        }:
            wm = self._airline_memory(case.get("goal_excerpt") or "")
            ask = _action("respond", cls=RiskClass.ASK_USER, text="What would you like to do today?")
            search = _action(
                "search_direct_flight",
                {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
            )
            decision = SoftGoalFieldRouter().choose(
                wm,
                [
                    GoalActionCandidate(ask, source="model", progress=0.0),
                    GoalActionCandidate(search, source="obligation", progress=1.0, uncertainty_reduction=0.8),
                ],
                TauAirlineAdapter(),
            )
            self.assertEqual(decision.selected.action.name, "search_direct_flight", case["case_id"])
            self.assertIn("gradient", wm.goal_field.last_decision)
            return

        if category == "airline_cached_profile_no_refetch":
            wm = self._airline_memory(case.get("goal_excerpt") or "Change my return flight.")
            wm.user_profiles["mia_li_3668"] = {"reservations": ["Z7GOZK"]}
            stale = _action("get_user_details", {"user_id": "mia_li_3668"})
            scan = _action("get_reservation_details", {"reservation_id": "Z7GOZK"})
            decision = SoftGoalFieldRouter().choose(
                wm,
                [
                    GoalActionCandidate(stale, source="model", progress=0.1),
                    GoalActionCandidate(scan, source="reservation_scan", progress=1.0, uncertainty_reduction=0.6),
                ],
                TauAirlineAdapter(),
            )
            self.assertEqual(decision.selected.action.name, "get_reservation_details", case["case_id"])
            return

        if category == "airline_reservation_drift_guard":
            wm = self._airline_memory("Book an economy flight from New York to Seattle on May 20th.")
            before = dict(wm.semantic_slots)
            obs = {
                "reservation_id": "HKEG34",
                "origin": "DEN",
                "destination": "LAS",
                "date": "2024-05-27",
                "cabin": "business",
            }
            wm.absorb_observation(obs)
            GenericCargoKernel(TauAirlineAdapter()).observe_tool_result(wm, "get_reservation_details", obs)
            self.assertEqual(wm.semantic_slots.get("origin"), before.get("origin"), case["case_id"])
            self.assertEqual(wm.semantic_slots.get("destination"), before.get("destination"), case["case_id"])
            self.assertTrue(wm.task_state.conflicts)
            return

        if category == "airline_malformed_reservation_lookup":
            agent = self._make_airline_agent()
            wm = WorkingMemory(goal="Change my reservation.")
            action = _action("get_reservation_details", {"reservation_id": "though"})
            gate = agent._check_state_action_validity(action, wm)
            self.assertFalse(gate.ok, case["case_id"])
            self.assertEqual(gate.reason, "adapter_id_field_plain_word")
            return

        if category == "airline_city_canonicalization":
            agent = self._make_airline_agent()
            wm = self._airline_memory("Book a direct flight from New York to Seattle on May 20th.")
            wm.user_profiles["mia_li_3668"] = {"name": {"first_name": "Mia", "last_name": "Li"}}
            proposal = _action(
                "search_direct_flight",
                {"origin": "New York", "destination": "Seattle", "date": "2024-05-20"},
            )
            replacement = agent._airline_obligation_action(proposal, wm)
            self.assertIsNotNone(replacement, case["case_id"])
            self.assertEqual(replacement.args["origin"], "JFK")
            self.assertEqual(replacement.args["destination"], "SEA")
            return

        if category == "airline_search_exhaustion":
            wm = self._airline_memory("Book a direct flight from New York to Seattle on May 20th.")
            args = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}
            wm.task_state.record_candidate_set("search_direct_flight", args, [])
            search = _action("search_direct_flight", args)
            alt = _action("search_onestop_flight", args)
            decision = SoftGoalFieldRouter().choose(
                wm,
                [
                    GoalActionCandidate(search, source="repeated_search", progress=0.2),
                    GoalActionCandidate(alt, source="relaxed_search", progress=0.9, uncertainty_reduction=0.8),
                ],
                TauAirlineAdapter(),
            )
            self.assertEqual(decision.selected.action.name, "search_onestop_flight", case["case_id"])
            return

        if category == "airline_booking_progress":
            agent = self._make_airline_agent()
            wm = self._airline_memory("Book an economy flight from New York to Seattle on May 20th with no insurance.")
            wm.user_profiles["mia_li_3668"] = {
                "name": {"first_name": "Mia", "last_name": "Li"},
                "dob": "1990-04-05",
                "membership": "regular",
                "payment_methods": {"credit_card_4421486": {"source": "credit_card", "id": "credit_card_4421486"}},
            }
            flight = {
                "flight_number": "HAT136",
                "origin": "JFK",
                "destination": "SEA",
                "date": "2024-05-20",
                "scheduled_departure_time_est": "12:30",
                "available_seats": {"economy": 3},
                "prices": {"economy": 100},
            }
            wm.task_state.record_candidate_set(
                "search_direct_flight",
                {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
                [CandidateObject("HAT136", "flight", flight)],
            )
            action = agent._airline_booking_progress_action(wm)
            self.assertIsNotNone(action, case["case_id"])
            self.assertEqual(action.declared_class, RiskClass.ASK_USER)
            self.assertIn("Should I book", action.user_text)
            return

        if category == "airline_update_multi_reservation":
            wm = self._airline_memory("Downgrade my upcoming business flights to economy.")
            wm.reservation_details["JG7FMM"] = {"reservation_id": "JG7FMM", "user_id": "mia_li_3668"}
            wm.typed_values["reservation_id"] = ["JG7FMM"]
            wm.typed_values["payment_method_id"] = ["credit_card_4421486"]
            wm.user_profiles["mia_li_3668"] = {
                "payment_methods": {"credit_card_4421486": {"source": "credit_card", "id": "credit_card_4421486"}}
            }
            action = _action(
                "update_reservation_flights",
                {
                    "reservation_id": "JG7FMM",
                    "cabin": "economy",
                    "flights": [{"flight_number": "HAT028", "date": "2024-05-21"}],
                    "payment_id": "credit_card_4421486",
                },
                RiskClass.WRITE,
            )
            cert = TauAirlineAdapter().build_commit_certificate(
                action,
                ToolEffectSchema(name="update_reservation_flights", cls=RiskClass.WRITE),
                wm,
            )
            self.assertTrue(cert.ok, cert.to_dict())
            return

        raise AssertionError(f"unhandled airline category {category}")

    # ------------------------------------------------------------------
    # Retail invariants
    # ------------------------------------------------------------------
    def _run_retail_case(self, case: Dict[str, Any]) -> None:
        category = case["category"]
        if category == "retail_name_zip_auth_before_order":
            agent = self._make_retail_agent()
            wm = WorkingMemory(goal="You are Yusuf Rossi in 19122. Exchange item in order #W2378156.")
            wm.absorb_user_message(wm.goal)
            proposal = _action("get_order_details", {"order_id": "#W2378156"})
            replacement = agent._retail_auth_phase_action(proposal, wm)
            self.assertIsNotNone(replacement, case["case_id"])
            self.assertEqual(replacement.name, "find_user_id_by_name_zip")
            return

        if category == "retail_wrong_zip_quarantine":
            wm = WorkingMemory(goal="Exchange items in order #W2378156")
            wm.absorb_user_message("My name is Yusuf Rossi and ZIP is 10001 for order #W2378156")
            kernel = GenericCargoKernel(TauRetailAdapter())
            kernel.observe_user_message(wm, wm.goal)
            kernel.observe_tool_result(wm, "find_user_id_by_name_zip", "Error: user not found")
            labels = {h.label for h in wm.goal_field.hypotheses}
            self.assertIn("order_id_recovery", labels, case["case_id"])
            self.assertIn("identity_lookup_failed", wm.goal_field.recent_recenter_reason)
            return

        if category == "retail_email_auth_progress":
            agent = self._make_retail_agent()
            wm = WorkingMemory(goal="My email is yusuf.rossi7301@example.com. Return my order.")
            wm.absorb_user_message(wm.goal)
            proposal = _action("respond", cls=RiskClass.ASK_USER, text="What is your user id?")
            replacement = agent._retail_auth_phase_action(proposal, wm)
            self.assertIsNotNone(replacement, case["case_id"])
            self.assertEqual(replacement.name, "find_user_id_by_email")
            return

        if category == "retail_order_cache_progress":
            wm = WorkingMemory(goal="Return item from order #W2378156.")
            wm.auth_user_id = "yusuf_rossi_9620"
            get_order = _action("get_order_details", {"order_id": "#W2378156"})
            ask = _action("respond", cls=RiskClass.ASK_USER, text="How can I assist?")
            decision = SoftGoalFieldRouter().choose(
                wm,
                [
                    GoalActionCandidate(ask, source="model", progress=0.0),
                    GoalActionCandidate(get_order, source="order_recovery", progress=1.0, uncertainty_reduction=0.5),
                ],
                TauRetailAdapter(),
            )
            self.assertEqual(decision.selected.action.name, "get_order_details", case["case_id"])
            return

        if category == "retail_mixed_catalog_account_goals":
            agent = self._make_retail_agent()
            wm = WorkingMemory(goal="How many t-shirt options are available, and exchange my order #W2378156.")
            wm.absorb_user_message(wm.goal)
            self.assertFalse(agent._is_no_auth_query(wm), case["case_id"])
            final = _action("respond", cls=RiskClass.FINAL, text="There are 10 options.")
            gate = agent._check_final_completeness(final, wm)
            self.assertFalse(gate.ok)
            return

        if category == "retail_hard_constraints_before_preferences":
            adapter = TauRetailAdapter()
            details = {
                "name": "Mechanical Keyboard",
                "variants": {
                    "old": {"item_id": "old", "options": {"switch_type": "linear", "size": "80%", "backlight": "RGB"}, "available": True},
                    "bad_pref": {"item_id": "bad_pref", "options": {"switch_type": "clicky", "size": "80%", "backlight": "RGB"}, "available": True},
                    "good": {"item_id": "good", "options": {"switch_type": "clicky", "size": "full size", "backlight": "none"}, "available": True},
                },
            }
            selected = adapter.select_replacement_variant_id(
                details,
                {"item_id": "old", "options": {"switch_type": "linear", "size": "80%", "backlight": "RGB"}},
                "exchange for a full-size clicky keyboard; RGB preferred, no backlight is ok if needed",
            )
            self.assertEqual(selected, "good", case["case_id"])
            return

        if category == "retail_post_success_termination":
            wm = WorkingMemory(goal="Return the order.")
            action = _action("return_delivered_order_items", {"order_id": "#W2378156"}, RiskClass.IRREVERSIBLE)
            wm.record_executed_mutation(action.signature())
            self.assertTrue(wm.mutation_already_executed(action.signature()), case["case_id"])
            return

        if category == "retail_placeholder_id_blocking":
            wm = WorkingMemory()
            action = _action(
                "exchange_delivered_order_items",
                {"order_id": "#W2378156", "item_ids": ["latest_item_id"], "new_item_ids": ["1234567890"]},
                RiskClass.WRITE,
            )
            schema = ToolEffectSchema(name="exchange_delivered_order_items", cls=RiskClass.WRITE)
            verdict = PreCommitVerifier().verify(action, schema, wm, TauRetailAdapter())
            self.assertFalse(verdict.ok, case["case_id"])
            self.assertEqual(verdict.reason, "placeholder_argument")
            return

        raise AssertionError(f"unhandled retail category {category}")

    # ------------------------------------------------------------------
    # Core invariants
    # ------------------------------------------------------------------
    def _run_core_case(self, case: Dict[str, Any]) -> None:
        category = case["category"]
        if category == "core_precommit_path_awareness":
            verifier = PreCommitVerifier()
            wm = WorkingMemory()
            semantic_none = _action("update_preferences", {"special_request": "none"}, RiskClass.WRITE)
            id_none = _action("update_preferences", {"reservation_id": "none"}, RiskClass.WRITE)
            schema = ToolEffectSchema(name="update_preferences", cls=RiskClass.WRITE)
            self.assertTrue(verifier.verify(semantic_none, schema, wm, TauAirlineAdapter()).ok, case["case_id"])
            self.assertFalse(verifier.verify(id_none, schema, wm, TauAirlineAdapter()).ok, case["case_id"])
            return

        if category == "core_belief_compactness":
            wm = WorkingMemory(goal="Book a flight from New York to Seattle.")
            for idx in range(30):
                wm.goal_field.record_progress(f"progress_{idx}", 0.1)
                wm.goal_field.record_friction(f"sig_{idx}", 0.2, "loop")
            snapshot = PredictiveGradientScheduler().snapshot(wm, TauAirlineAdapter())
            rendered = snapshot.render_compact(max_chars=420)
            self.assertLessEqual(len(rendered), 420, case["case_id"])
            self.assertIn("gradient=", rendered)
            return

        if category == "core_friction_gradient":
            wm = WorkingMemory(goal="Book a flight.")
            action = _action("respond", cls=RiskClass.ASK_USER, text="What would you like?")
            sig = action.signature()
            wm.goal_field.record_critique(sig, "repeat_loop", "same generic ask")
            wm.goal_field.record_critique(sig, "repeat_loop", "same generic ask")
            self.assertIn(sig, wm.goal_field.friction_blacklist, case["case_id"])
            gradient = PredictiveGradientScheduler().pick(wm, TauAirlineAdapter())
            self.assertIn(gradient.kind, {"ESCALATE", "ASK_USER", "RESPOND"})
            return

        if category == "core_smoke_fixture_parse":
            self.assertEqual(case["source_file"], f"trajectories ({case['source_run']}).jsonl")
            self.assertIn(case["domain"], {"core", "airline", "retail"})
            self.assertIsInstance(case["observed_tools"], list)
            return

        raise AssertionError(f"unhandled core category {category}")


def _make_test(case: Dict[str, Any]):
    def test(self: CargoNCorpusRegressionTests) -> None:
        self._run_case(case)

    test.__name__ = f"test_{case['case_id']}"
    test.__doc__ = (
        f"{case['case_id']} {case['category']} from "
        f"{case['source_file']} row {case['row_index']}"
    )
    return test


for _case in _load_cases():
    setattr(CargoNCorpusRegressionTests, f"test_{_case['case_id']}", _make_test(_case))


if __name__ == "__main__":
    unittest.main()
