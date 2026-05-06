"""Synthesize a compact tactical playbook from retrieved process-memory cards.

The playbook is what gets injected into the system prompt at runtime. It
must be:

  * **Concise** — bounded by ``config.playbook_max_chars``.
  * **Actionable** — sequencing, verification, recovery, anti-patterns.
  * **Leak-free** — synthesized from already-redacted card fields.
  * **Stable** — same retrieved cards → identical playbook output.

Synthesis groups card content into named buckets and renders each bucket
to a small bullet list, then trims to the budget.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .config import RexConfig, default_config
from .memory_types import ProcessMemoryCard, TacticalPlaybook
from .sanitize import compress_repeated_phrases


_SECTION_TITLES: List[str] = [
    "Likely successful next actions",
    "Verification steps",
    "Sequencing guidance",
    "Recovery heuristics",
    "Common failure traps",
    "Forbidden behaviors",
]


def _bucketize(cards: Sequence[ProcessMemoryCard]) -> Dict[str, List[str]]:
    """Group card contents into named buckets used by the playbook."""
    buckets: Dict[str, List[str]] = {t: [] for t in _SECTION_TITLES}

    for card in cards:
        if card.recommended_next_tools:
            buckets["Likely successful next actions"].append(
                f"`{card.task_category}` →  " + " -> ".join(card.recommended_next_tools)
            )
        if card.required_evidence:
            for ev in card.required_evidence:
                buckets["Verification steps"].append(ev)
        if card.tool_ordering_hints:
            buckets["Sequencing guidance"].append(
                "Order: " + " -> ".join(card.tool_ordering_hints)
            )
        if card.confirmation:
            buckets["Sequencing guidance"].append(
                f"Before write: {card.confirmation}"
            )
        if card.recovery_heuristics:
            for r in card.recovery_heuristics:
                buckets["Recovery heuristics"].append(r)
        if card.failure_patterns:
            for f in card.failure_patterns:
                buckets["Common failure traps"].append(f)
        if card.common_trap:
            buckets["Common failure traps"].append(card.common_trap)
        if card.forbidden_behaviors:
            for f in card.forbidden_behaviors:
                buckets["Forbidden behaviors"].append(f)
    return buckets


def _trim_section(items: Sequence[str], char_budget: int) -> List[str]:
    """Compress repeats then clip until the section fits the per-section budget."""
    cleaned = compress_repeated_phrases(items)
    out: List[str] = []
    used = 0
    for line in cleaned:
        snippet = line.strip()
        if not snippet:
            continue
        if len(snippet) > char_budget:
            snippet = snippet[: max(0, char_budget - 3)].rstrip() + "..."
        if used + len(snippet) + 2 > char_budget:
            break
        out.append(snippet)
        used += len(snippet) + 2
    return out


def synthesize_playbook(
    cards: Sequence[ProcessMemoryCard],
    *,
    config: Optional[RexConfig] = None,
) -> TacticalPlaybook:
    """Render ``cards`` to a :class:`TacticalPlaybook` ready for prompting.

    Heuristics:

      * Sections appear in fixed order (``_SECTION_TITLES``) so the prompt
        is stable across runs.
      * Each section is trimmed to approximately
        ``config.playbook_section_chars`` before assembly (soft per-section limit).
      * The final playbook is hard-capped to ``config.playbook_max_chars`` and
        ``config.playbook_max_lines``; sections are added sequentially until
        budget is exhausted, then assembly stops.
    """
    cfg = config or default_config()
    if not cards:
        body = (
            "No retrieved procedural guidance is available for this task. "
            "Use only current policy, tools, and user-provided facts."
        )
        return TacticalPlaybook(
            body=body,
            sections={},
            card_ids=[],
            char_count=len(body),
        )

    buckets = _bucketize(cards)
    sections: Dict[str, List[str]] = {}
    for title in _SECTION_TITLES:
        kept = _trim_section(buckets.get(title, []), cfg.playbook_section_chars)
        if kept:
            sections[title] = kept

    header = "Relevant procedural guidance from prior successful interactions:"
    lines: List[str] = [header]
    line_budget = max(4, int(cfg.playbook_max_lines))
    char_budget = max(200, int(cfg.playbook_max_chars))
    used_chars = len(header)
    for title in _SECTION_TITLES:
        items = sections.get(title)
        if not items:
            continue
        sec_lines: List[str] = [f"\n## {title}"]
        for it in items:
            sec_lines.append(f"- {it}")
        new_chars = sum(len(s) + 1 for s in sec_lines)
        if used_chars + new_chars > char_budget:
            # Truncate the section to whatever fits.
            allowed = char_budget - used_chars - len(sec_lines[0]) - 1
            if allowed < 40:
                break
            kept_items: List[str] = []
            for it in items:
                bullet = f"- {it}"
                if allowed - len(bullet) < 0:
                    break
                kept_items.append(bullet)
                allowed -= len(bullet) + 1
            if kept_items:
                lines.append(sec_lines[0])
                lines.extend(kept_items)
                used_chars += len(sec_lines[0]) + sum(len(s) + 1 for s in kept_items)
            break
        lines.extend(sec_lines)
        used_chars += new_chars
        if len(lines) >= line_budget:
            break

    body = "\n".join(lines).strip()
    if len(body) > char_budget:
        body = body[: char_budget - 3].rstrip() + "..."

    return TacticalPlaybook(
        body=body,
        sections=sections,
        card_ids=[c.card_id for c in cards],
        char_count=len(body),
    )


def render_playbook_for_prompt(
    cards: Sequence[ProcessMemoryCard],
    *,
    config: Optional[RexConfig] = None,
) -> str:
    """Convenience: synthesize and return just the body string."""
    return synthesize_playbook(cards, config=config).body


__all__ = [
    "synthesize_playbook",
    "render_playbook_for_prompt",
]
