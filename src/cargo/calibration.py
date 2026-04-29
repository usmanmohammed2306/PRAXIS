"""Per-class calibration thresholds for the CARGO gate.

In v1 we ship sensible defaults derived from the blueprint:
  * READ           : no gating, threshold ignored.
  * WRITE          : SC threshold = 2/3 (k_agree >= 2 of 3 samples).
  * IRREVERSIBLE   : SC threshold = 2/3, plus counterfactual rollout.
  * FINAL          : SC threshold = 2/3, plus counterfactual rollout.
  * ASK_USER       : passthrough (sender of the question is by definition
                     un-grounded — that's the whole point).

A future extension fits these τ_c on a held-out calibration split so that
P(correct ∈ kept-set | class=c) ≥ 1 − α_c. The fitter scaffolding lives
in :func:`fit_thresholds` and accepts a list of dev rollouts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .risk_class import RiskClass


@dataclass
class CalibrationConfig:
    sc_thresholds: Dict[RiskClass, float]
    # Whether to run the counterfactual gate for each class.
    run_cf: Dict[RiskClass, bool]
    # Per-class number of SC samples (k). Reducing for WRITE saves cost.
    sc_k: Dict[RiskClass, int]


def default_calibration() -> CalibrationConfig:
    """The shipped defaults — used unless the user provides a fit file."""
    return CalibrationConfig(
        sc_thresholds={
            RiskClass.READ: 0.0,
            RiskClass.WRITE: float(os.environ.get("CARGO_SC_TAU_WRITE", "0.66")),
            RiskClass.IRREVERSIBLE: float(os.environ.get("CARGO_SC_TAU_IRREV", "0.66")),
            RiskClass.FINAL: float(os.environ.get("CARGO_SC_TAU_FINAL", "0.66")),
            RiskClass.ASK_USER: 0.0,
        },
        run_cf={
            RiskClass.READ: False,
            RiskClass.WRITE: False,
            RiskClass.IRREVERSIBLE: os.environ.get("CARGO_CF_IRREV", "1") == "1",
            RiskClass.FINAL: os.environ.get("CARGO_CF_FINAL", "1") == "1",
            RiskClass.ASK_USER: False,
        },
        sc_k={
            RiskClass.READ: 0,
            RiskClass.WRITE: int(os.environ.get("CARGO_SC_K_WRITE", "3")),
            RiskClass.IRREVERSIBLE: int(os.environ.get("CARGO_SC_K_IRREV", "3")),
            RiskClass.FINAL: int(os.environ.get("CARGO_SC_K_FINAL", "3")),
            RiskClass.ASK_USER: 0,
        },
    )


def fit_thresholds(rollouts: List[Dict], *, target_abstain_rates: Optional[Dict[RiskClass, float]] = None) -> CalibrationConfig:
    """Fit per-class SC thresholds from logged baseline rollouts.

    ``rollouts`` is expected to be a list of dict records each with
    ``per_step`` entries containing ``declared_class`` and ``sc_agreement``
    (logged by CargoAgent in its diagnostic JSONL). The fitter chooses τ_c
    so that the fraction of steps in class ``c`` that fall below τ_c is
    approximately ``target_abstain_rates[c]`` (defaults: 5%/15%/25%/25%).

    For v1 the shipped defaults work without calibration data; this hook
    is provided so a researcher can re-fit and pass a JSON config to the
    agent without code changes.
    """
    cfg = default_calibration()
    if not rollouts:
        return cfg
    targets = target_abstain_rates or {
        RiskClass.READ: 0.0,
        RiskClass.WRITE: 0.15,
        RiskClass.IRREVERSIBLE: 0.25,
        RiskClass.FINAL: 0.25,
        RiskClass.ASK_USER: 0.0,
    }
    # Collect per-class agreements.
    by_class: Dict[RiskClass, List[float]] = {c: [] for c in RiskClass}
    for r in rollouts:
        for step in r.get("per_step", []) or []:
            try:
                c = RiskClass.parse(step.get("declared_class"))
                a = step.get("sc_agreement")
                if isinstance(a, (int, float)):
                    by_class[c].append(float(a))
            except Exception:
                continue
    for c, agreements in by_class.items():
        if not agreements or c in (RiskClass.READ, RiskClass.ASK_USER):
            continue
        agreements.sort()
        target_rate = targets.get(c, 0.2)
        idx = max(0, min(len(agreements) - 1, int(target_rate * len(agreements))))
        # Use a small floor to avoid degenerate τ=0 thresholds.
        cfg.sc_thresholds[c] = max(0.5, agreements[idx])
    return cfg


__all__ = ["CalibrationConfig", "default_calibration", "fit_thresholds"]
