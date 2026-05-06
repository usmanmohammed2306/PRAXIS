"""Leakage-safe experience cards for REx-RPE.

The bank is procedural memory, not answer retrieval. Cards teach tool order,
evidence requirements, confirmation points, and common traps. IDs and PII are
redacted before a card can be used in a prompt, and test-split identifiers are
blocked by an audit pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_TAU = ROOT / "external" / "tau-bench"
DEFAULT_BANK_DIR = ROOT / "outputs" / "experience_bank"


SENSITIVE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
    (re.compile(r"#W\d+\b"), "<ORDER_ID>"),
    (re.compile(r"\b(?:credit_card|gift_card|paypal|certificate)_\d+\b"), "<PAYMENT_ID>"),
    (re.compile(r"\b[a-z]+_[a-z]+_\d{3,5}\b"), "<USER_ID>"),
    (re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{6}\b"), "<RESERVATION_ID>"),
    (re.compile(r"\b\d{9,10}\b"), "<ITEM_OR_PRODUCT_ID>"),
]

UNREDACTED_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"#W\d+\b"),
    re.compile(r"\b(?:credit_card|gift_card|paypal|certificate)_\d+\b"),
    re.compile(r"\b[a-z]+_[a-z]+_\d{3,5}\b"),
    re.compile(r"\b\d{9,10}\b"),
]


@dataclass(frozen=True)
class ExperienceCard:
    card_id: str
    domain: str
    source: str
    intent: str
    instruction_template: str
    needed_evidence: List[str]
    tool_sequence: List[str]
    confirmation: str
    common_trap: str

    def prompt_text(self) -> str:
        sequence = " -> ".join(self.tool_sequence) if self.tool_sequence else "respond"
        evidence = "; ".join(self.needed_evidence)
        return (
            f"[{self.card_id}] intent={self.intent}\n"
            f"Process: {sequence}\n"
            f"Evidence to ground: {evidence}\n"
            f"Confirm before write: {self.confirmation}\n"
            f"Trap to avoid: {self.common_trap}\n"
            f"Similar request shape: {self.instruction_template}"
        )

    def searchable_text(self) -> str:
        return " ".join([
            self.domain,
            self.intent,
            self.instruction_template,
            " ".join(self.needed_evidence),
            " ".join(self.tool_sequence),
            self.common_trap,
        ])


def redact_text(text: Any) -> str:
    out = str(text or "")
    for pattern, repl in SENSITIVE_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 1]


def _truncate(text: str, limit: int = 360) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def contains_unredacted_sensitive(text: str) -> bool:
    return any(p.search(text) for p in UNREDACTED_PATTERNS)


def _task_module(path: Path, module_name: str) -> Any:
    if str(EXTERNAL_TAU) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_TAU))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import task module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _task_fields(task: Any) -> Tuple[str, List[str], List[str]]:
    instruction = getattr(task, "instruction", None)
    if instruction is None and isinstance(task, dict):
        instruction = task.get("instruction", "")
    actions = getattr(task, "actions", None)
    if actions is None and isinstance(task, dict):
        actions = task.get("actions", [])
    action_names: List[str] = []
    raw_args: List[str] = []
    for a in actions or []:
        name = getattr(a, "name", None)
        kwargs = getattr(a, "kwargs", None)
        if isinstance(a, dict):
            name = a.get("name")
            kwargs = a.get("kwargs") or a.get("arguments")
        if name:
            action_names.append(str(name))
        if kwargs:
            raw_args.append(json.dumps(kwargs, sort_keys=True, default=str))
    return str(instruction or ""), action_names, raw_args


def _intent_from_actions(domain: str, actions: Sequence[str], instruction: str) -> str:
    if not actions:
        return "answer_or_clarify"
    uniq = []
    for a in actions:
        if a not in uniq:
            uniq.append(a)
    if len(uniq) == 1:
        return uniq[0]
    if domain == "retail":
        return "multi_order_or_mixed_retail_work"
    if domain == "airline":
        return "multi_reservation_airline_work"
    return "multi_step_tool_work"


def _evidence_for(domain: str, actions: Sequence[str]) -> List[str]:
    evidence = ["authenticate the user using current conversation facts"]
    if domain == "retail":
        if any("order" in a or "return" in a or "exchange" in a for a in actions):
            evidence.append("fetch order details and use item/product/payment IDs from tool observations")
        if any("items" in a or "exchange" in a for a in actions):
            evidence.append("fetch product variants and apply hard constraints before preferences")
    elif domain == "airline":
        evidence.append("fetch user profile once, then choose reservations by active route/date/cabin frame")
        if any("flight" in a or "book" in a for a in actions):
            evidence.append("search flights only for the currently requested route and date")
        if any("baggage" in a for a in actions):
            evidence.append("calculate free versus paid bags from membership/cabin policy")
    else:
        evidence.append("use only tool-schema arguments supported by the current task")
    return evidence


def _process_for(domain: str, actions: Sequence[str]) -> List[str]:
    seq = ["ground_identity"]
    if domain == "retail":
        if any("order" in a or "return" in a or "exchange" in a or "modify" in a or "cancel" in a for a in actions):
            seq.append("read_relevant_order")
        if any("items" in a or "exchange" in a for a in actions):
            seq.append("read_product_variants")
    elif domain == "airline":
        seq.extend(["read_user_profile", "read_active_reservation"])
        if any("flight" in a or "book" in a for a in actions):
            seq.append("search_or_reuse_flights")
    for a in actions:
        if a not in seq:
            seq.append(a)
    if any(_looks_mutating(a) for a in actions):
        seq.insert(max(1, len(seq) - len(actions)), "confirm_exact_write")
    return seq


def _looks_mutating(tool_name: str) -> bool:
    return bool(re.match(r"^(cancel|modify|return|exchange|book|update|send|transfer|delete|create|add|remove)", tool_name))


def _confirmation_for(actions: Sequence[str]) -> str:
    if any(_looks_mutating(a) for a in actions):
        return "ask the user to confirm the exact target, selected option, payment/refund method, and irreversible effect"
    return "no write confirmation needed; answer from retrieved evidence"


def _trap_for(domain: str, actions: Sequence[str]) -> str:
    if domain == "retail":
        if any("exchange" in a or "items" in a for a in actions):
            return "do not copy example item IDs; select current task variants from current product details"
        if any("return" in a for a in actions):
            return "do not invent payment IDs; use a current account payment method or ask"
        return "do not loop on placeholder emails; use current user-provided identity facts"
    if domain == "airline":
        if any("flight" in a or "book" in a for a in actions):
            return "preserve active route/date frame; do not drift to old reservation segments"
        if any("cancel" in a for a in actions):
            return "basic economy cannot be modified; cancellation depends on policy and user confirmation"
        return "do not repeatedly ask for user_id after profile has been fetched"
    return "follow the current tool schema; examples teach sequence, not arguments"


def _card_from_task(domain: str, source: str, idx: int, task: Any) -> Optional[ExperienceCard]:
    instruction, actions, raw_args = _task_fields(task)
    if not instruction and not actions:
        return None
    redacted_instruction = _truncate(redact_text(instruction))
    probe = " ".join([redacted_instruction, " ".join(redact_text(x) for x in raw_args)])
    if contains_unredacted_sensitive(probe):
        return None
    return ExperienceCard(
        card_id=f"{domain}-{source}-{idx:03d}",
        domain=domain,
        source=source,
        intent=_intent_from_actions(domain, actions, instruction),
        instruction_template=redacted_instruction,
        needed_evidence=_evidence_for(domain, actions),
        tool_sequence=_process_for(domain, actions),
        confirmation=_confirmation_for(actions),
        common_trap=_trap_for(domain, actions),
    )


def _load_retail_tasks(split: str) -> List[Any]:
    path = EXTERNAL_TAU / "tau_bench" / "envs" / "retail" / f"tasks_{split}.py"
    if not path.exists():
        return []
    mod = _task_module(path, f"_rex_retail_{split}")
    return list(getattr(mod, f"TASKS_{split.upper()}", []))


def _load_airline_test_ids() -> Set[str]:
    path = EXTERNAL_TAU / "tau_bench" / "envs" / "airline" / "tasks_test.py"
    if not path.exists():
        return set()
    mod = _task_module(path, "_rex_airline_test")
    ids: Set[str] = set()
    for task in getattr(mod, "TASKS", []):
        instruction, _, raw_args = _task_fields(task)
        for text in [instruction] + raw_args:
            ids.update(_raw_sensitive_tokens(text))
    return ids


def _raw_sensitive_tokens(text: str) -> Set[str]:
    out: Set[str] = set()
    for pattern, _ in SENSITIVE_PATTERNS:
        out.update(m.group(0) for m in pattern.finditer(str(text or "")))
    return out


def _policy_airline_cards() -> List[ExperienceCard]:
    shapes = [
        ("airline_policy_cancel_basic_economy", "basic_economy_cancel_or_transfer",
         ["read_user_profile", "read_active_reservation", "explain_basic_economy_limit", "confirm_exact_write", "cancel_reservation"],
         "basic economy modification is blocked; if cancellation is policy-allowed, confirm reservation and reason before cancel"),
        ("airline_policy_book_flight", "book_new_reservation",
         ["read_user_profile", "search_direct_flight", "search_onestop_flight_if_allowed", "present_selected_itinerary", "confirm_exact_write", "book_reservation"],
         "keep route/date/cabin/bags/payment frame from the current request"),
        ("airline_policy_update_bags", "update_baggages",
         ["read_user_profile", "read_active_reservation", "calculate_free_bags", "confirm_exact_write", "update_reservation_baggages"],
         "membership and cabin determine nonfree bags; do not charge free bags"),
        ("airline_policy_change_flight", "change_reservation_flights",
         ["read_user_profile", "read_active_reservation", "search_current_route_options", "present_selected_itinerary", "confirm_exact_write", "update_reservation_flights"],
         "do not reuse stale route segments after the user corrects origin/destination"),
        ("airline_policy_certificate", "delayed_flight_certificate",
         ["read_user_profile", "read_matching_reservation", "confirm_compensation_policy", "confirm_exact_write", "send_certificate"],
         "compensation is policy-bound; do not promise refund when only certificate is allowed"),
    ]
    cards: List[ExperienceCard] = []
    for cid, intent, seq, trap in shapes:
        cards.append(ExperienceCard(
            card_id=cid,
            domain="airline",
            source="policy_generated",
            intent=intent,
            instruction_template="Policy-derived support card generated from airline tool names and wiki, not from test answers.",
            needed_evidence=_evidence_for("airline", seq),
            tool_sequence=seq,
            confirmation="confirm exact reservation, itinerary/payment/baggage fields, and policy effect before write",
            common_trap=trap,
        ))
    return cards


def _ace_schema_cards() -> List[ExperienceCard]:
    shapes = [
        ("ace_schema_login_then_action", "login_or_auth_before_private_data",
         ["inspect_required_auth_tool", "call_login_or_auth_tool", "call_requested_data_tool"],
         "if a tool description says login/auth is required, do that first"),
        ("ace_schema_multi_step", "multi_step_api_sequence",
         ["identify_target_object", "read_or_create_prerequisite", "call_final_requested_tool"],
         "do not stop after the first helper call when the user asked for a final mutation"),
        ("ace_schema_similar_api", "similar_api_disambiguation",
         ["compare_tool_descriptions", "select_exact_api_by parameters", "call_only_relevant_tool"],
         "similar tool names are decoys; use parameters and descriptions"),
    ]
    return [
        ExperienceCard(
            card_id=cid,
            domain="ace",
            source="schema_generated",
            intent=intent,
            instruction_template="ACE schema-derived support card; no possible_answer content used.",
            needed_evidence=["current user instruction", "current tool descriptions", "required parameters"],
            tool_sequence=seq,
            confirmation="only ask if required parameters are missing",
            common_trap=trap,
        )
        for cid, intent, seq, trap in shapes
    ]


def audit_card(card: ExperienceCard, forbidden_tokens: Optional[Set[str]] = None) -> Tuple[bool, str]:
    text = json.dumps(asdict(card), ensure_ascii=False)
    if contains_unredacted_sensitive(text):
        return False, "unredacted_sensitive_token"
    for tok in forbidden_tokens or set():
        if tok and tok in text:
            return False, f"forbidden_test_token:{tok}"
    return True, ""


def build_experience_bank(
    *,
    output_dir: Path = DEFAULT_BANK_DIR,
    include_retail_dev: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden_airline = _load_airline_test_ids()
    cards: List[ExperienceCard] = []
    rejected: List[Dict[str, str]] = []

    for split in ("train", "dev" if include_retail_dev else ""):
        if not split:
            continue
        for idx, task in enumerate(_load_retail_tasks(split)):
            card = _card_from_task("retail", split, idx, task)
            if not card:
                rejected.append({"domain": "retail", "source": split, "reason": "redaction_failed"})
                continue
            ok, reason = audit_card(card)
            if ok:
                cards.append(card)
            else:
                rejected.append({"domain": "retail", "source": split, "reason": reason})

    for card in _policy_airline_cards():
        ok, reason = audit_card(card, forbidden_airline)
        if ok:
            cards.append(card)
        else:
            rejected.append({"domain": "airline", "source": card.source, "reason": reason})

    for card in _ace_schema_cards():
        ok, reason = audit_card(card)
        if ok:
            cards.append(card)
        else:
            rejected.append({"domain": "ace", "source": card.source, "reason": reason})

    by_domain: Dict[str, List[ExperienceCard]] = {}
    for card in cards:
        by_domain.setdefault(card.domain, []).append(card)
    for domain, domain_cards in by_domain.items():
        path = output_dir / f"{domain}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for card in domain_cards:
                f.write(json.dumps(asdict(card), ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "status": "ok",
        "output_dir": str(output_dir),
        "num_cards": len(cards),
        "cards_by_domain": {k: len(v) for k, v in sorted(by_domain.items())},
        "rejected": rejected[:50],
        "policy": {
            "no_previous_eval_trajectories": True,
            "no_ace_possible_answer": True,
            "retail_sources": ["tasks_train.py", "tasks_dev.py" if include_retail_dev else ""],
            "airline_sources": ["policy_generated"],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_experience_cards(domain: str, bank_dir: Path = DEFAULT_BANK_DIR) -> List[ExperienceCard]:
    path = bank_dir / f"{domain}.jsonl"
    if not path.exists():
        try:
            build_experience_bank(output_dir=bank_dir)
        except Exception:
            return _policy_airline_cards() if domain == "airline" else (_ace_schema_cards() if domain == "ace" else [])
    cards: List[ExperienceCard] = []
    if not path.exists():
        return cards
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cards.append(ExperienceCard(**data))
            except Exception:
                continue
    return cards


class ExperienceRetriever:
    """Tiny BM25-style retriever with diversity by intent."""

    def __init__(self, cards: Sequence[ExperienceCard]) -> None:
        self.cards = list(cards)
        self._docs = [_tokenize(c.searchable_text()) for c in self.cards]
        df: Dict[str, int] = {}
        for toks in self._docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = max(1, len(self.cards))
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._avgdl = sum(len(d) for d in self._docs) / max(1, len(self._docs))

    def search(self, query: str, *, k: int = 3) -> List[ExperienceCard]:
        q = _tokenize(redact_text(query))
        if not q or not self.cards:
            return self.cards[:k]
        qset = set(q)
        scored: List[Tuple[float, int, ExperienceCard]] = []
        for i, (card, doc) in enumerate(zip(self.cards, self._docs)):
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            dl = len(doc) or 1
            for t in qset:
                if t not in tf:
                    continue
                freq = tf[t]
                idf = self._idf.get(t, 0.0)
                score += idf * (freq * 2.2) / (freq + 1.2 * (1 - 0.75 + 0.75 * dl / max(1.0, self._avgdl)))
            scored.append((score, -i, card))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out: List[ExperienceCard] = []
        seen_intents: Set[str] = set()
        for score, _, card in scored:
            if score <= 0 and out:
                continue
            if card.intent in seen_intents and len(out) < min(k, 2):
                continue
            out.append(card)
            seen_intents.add(card.intent)
            if len(out) >= k:
                return out
        for _, _, card in scored:
            if card not in out:
                out.append(card)
            if len(out) >= k:
                break
        return out[:k]


def render_experience_brief(cards: Sequence[ExperienceCard]) -> str:
    if not cards:
        return "No retrieved experience cards are available. Use only the current policy, tools, and user facts."
    lines = [
        "Use these as procedural memory only. Never copy IDs, emails, payment methods, dates, or arguments from examples.",
    ]
    for idx, card in enumerate(cards, start=1):
        lines.append(f"\nExample {idx}:")
        lines.append(card.prompt_text())
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser("rex.experience")
    parser.add_argument("--output-dir", default=str(DEFAULT_BANK_DIR))
    parser.add_argument("--no-retail-dev", action="store_true")
    ns = parser.parse_args()
    manifest = build_experience_bank(output_dir=Path(ns.output_dir), include_retail_dev=not ns.no_retail_dev)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
