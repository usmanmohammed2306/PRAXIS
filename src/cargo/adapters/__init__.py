"""Pluggable benchmark/domain adapters for CARGO-v2."""
from __future__ import annotations

from typing import Any, Dict, List

from ..core import BaseCargoAdapter, CargoDomainAdapter
from .acebench import ACEBenchAdapter
from .synthetic_generic import SyntheticGenericAdapter
from .tau_airline import TauAirlineAdapter
from .tau_retail import TauRetailAdapter


def select_adapter(
    env_hint: str = "",
    tools_info: List[Dict[str, Any]] | None = None,
    wiki: str = "",
) -> CargoDomainAdapter:
    """Select an adapter using explicit hints first, then tool/policy shape."""
    hint = (env_hint or "").lower()
    if "airline" in hint:
        return TauAirlineAdapter()
    if "retail" in hint:
        return TauRetailAdapter()
    if "ace" in hint:
        return ACEBenchAdapter()

    names = " ".join(_tool_name(t).lower() for t in (tools_info or []))
    blob = f"{names} {wiki}".lower()
    if any(x in blob for x in ("reservation", "flight", "search_direct_flight")):
        return TauAirlineAdapter()
    if any(x in blob for x in ("order", "product", "exchange_delivered_order_items")):
        return TauRetailAdapter()
    if any(x in blob for x in ("set_slot", "query_attribute", "done", "candidate")):
        return ACEBenchAdapter()
    return SyntheticGenericAdapter()


def _tool_name(spec: Dict[str, Any]) -> str:
    fn = spec.get("function", spec) if isinstance(spec, dict) else {}
    return str(fn.get("name", "")) if isinstance(fn, dict) else ""


__all__ = [
    "ACEBenchAdapter",
    "BaseCargoAdapter",
    "CargoDomainAdapter",
    "SyntheticGenericAdapter",
    "TauAirlineAdapter",
    "TauRetailAdapter",
    "select_adapter",
]
