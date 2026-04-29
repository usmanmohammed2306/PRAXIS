"""Risk-class typing for CARGO actions.

Each tool / action is bucketed into one of five risk classes. The class
selects the gating policy:

  * ``READ``         — pure information retrieval; fast path, no gates.
  * ``WRITE``        — modifies state but the change is recoverable
                       in-domain (e.g. updating a field that can be
                       updated again).
  * ``IRREVERSIBLE`` — has a real-world side-effect that cannot be
                       undone in-domain (cancel, refund, charge, send_email).
  * ``FINAL``        — the agent's terminal answer to the user (the
                       benchmark scores against the final state).
  * ``ASK_USER``     — explicit "ask the user" pseudo-action; cheap
                       passthrough.
"""
from __future__ import annotations

from enum import Enum


class RiskClass(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    IRREVERSIBLE = "IRREVERSIBLE"
    FINAL = "FINAL"
    ASK_USER = "ASK_USER"

    @classmethod
    def parse(cls, raw: object) -> "RiskClass":
        """Tolerant parser used by the proposer and schema inducer."""
        if isinstance(raw, RiskClass):
            return raw
        if not isinstance(raw, str):
            return cls.READ
        s = raw.strip().upper().replace("-", "_")
        # Normalize a few common variants we see from LLMs.
        aliases = {
            "MUTATION": "WRITE",
            "MUTATE": "WRITE",
            "WRITE_REVERSIBLE": "WRITE",
            "WRITE_IRREVERSIBLE": "IRREVERSIBLE",
            "IRREV": "IRREVERSIBLE",
            "IRREVERSIBLE_WRITE": "IRREVERSIBLE",
            "DESTRUCTIVE": "IRREVERSIBLE",
            "TERMINAL": "FINAL",
            "FINISH": "FINAL",
            "RESPOND": "FINAL",
            "ASK": "ASK_USER",
            "ASK-USER": "ASK_USER",
            "ASKUSER": "ASK_USER",
            "USER": "ASK_USER",
            "GET": "READ",
            "INFO": "READ",
            "QUERY": "READ",
            "SEARCH": "READ",
        }
        s = aliases.get(s, s)
        try:
            return cls(s)
        except ValueError:
            return cls.READ


# Convenience predicates -----------------------------------------------------
def is_gated(c: RiskClass) -> bool:
    """True iff the action class triggers the calibrated gate."""
    return c in (RiskClass.WRITE, RiskClass.IRREVERSIBLE, RiskClass.FINAL)


def is_irreversible_or_final(c: RiskClass) -> bool:
    return c in (RiskClass.IRREVERSIBLE, RiskClass.FINAL)


__all__ = ["RiskClass", "is_gated", "is_irreversible_or_final"]
