"""tau-bench retail adapter."""
from __future__ import annotations

import re
from typing import Any, List, Tuple

from ..core import BaseCargoAdapter, Constraint, FallbackRule, Preference, TaskState


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


__all__ = ["TauRetailAdapter"]
