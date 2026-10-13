"""Backward-compatible UI diff with additive typed, span-level analysis.

The original numeric_changes/entity_changes/hedge_changes/similarity/
mutation_types/segments keys remain available. Shared analysis is deterministic
and explicitly inferred, never evidence of observed transmission.
"""

import difflib
from datetime import datetime
from typing import TypedDict

from app.services.evolution.text_changes import (
    HEDGE_WORDS,
    MutationAnalysis,
    analyze_text_change,
    tokenize,
    tokens_with_spans,
)

__all__ = [
    "HEDGE_WORDS",
    "DiffSegment",
    "MutationDiffResult",
    "diff_versions",
    "tokenize",
]


class DiffSegment(TypedDict):
    type: str
    old: str
    new: str
    kinds: list[str]


class MutationDiffResult(TypedDict):
    numeric_changes: list[dict[str, str]]
    entity_changes: list[dict[str, str]]
    hedge_changes: list[dict[str, str]]
    polarity_changes: list[dict[str, str]]
    attribution_changes: list[dict[str, str]]
    scope_changes: list[dict[str, str]]
    similarity: float
    mutation_types: list[str]
    segments: list[DiffSegment]
    analysis: MutationAnalysis
    algorithm_version: str
    meaningful_change: bool
    observed_propagation: bool


def _legacy_changes(analysis: MutationAnalysis, kind: str) -> list[dict[str, str]]:
    return [
        {"removed": value, "added": ""}
        for change in analysis["changes"]
        if change["kind"] == kind
        for value in change["removed"]
    ] + [
        {"removed": "", "added": value}
        for change in analysis["changes"]
        if change["kind"] == kind
        for value in change["added"]
    ]


def diff_versions(
    old_text: str,
    new_text: str,
    *,
    older_timestamp: datetime | None = None,
    newer_timestamp: datetime | None = None,
    context_note: str | None = None,
    provenance_url: str | None = None,
) -> MutationDiffResult:
    analysis = analyze_text_change(
        old_text,
        new_text,
        older_timestamp=older_timestamp,
        newer_timestamp=newer_timestamp,
        context_note=context_note,
        provenance_url=provenance_url,
    )
    old, new = tokens_with_spans(old_text or ""), tokens_with_spans(new_text or "")
    matcher = difflib.SequenceMatcher(
        a=[t.normalized for t in old],
        b=[t.normalized for t in new],
        autojunk=False,
    )
    segments: list[DiffSegment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        kinds: set[str] = set()
        if tag != "equal":
            for span in analysis["changed_spans"]:
                older_span, newer_span = span["older_span"], span["newer_span"]
                overlaps_old = (
                    i1 < i2
                    and old[i1].start < older_span["end"]
                    and old[i2 - 1].end > older_span["start"]
                )
                overlaps_new = (
                    j1 < j2
                    and new[j1].start < newer_span["end"]
                    and new[j2 - 1].end > newer_span["start"]
                )
                if overlaps_old or overlaps_new:
                    kinds.update(span["kinds"])
            if not kinds:
                kinds.add("wording")
        segments.append(
            {
                "type": "same" if tag == "equal" else "changed",
                "old": " ".join(t.text for t in old[i1:i2]),
                "new": " ".join(t.text for t in new[j1:j2]),
                "kinds": sorted(kinds),
            }
        )

    hedge_changes = [
        {"word": value, "direction": "lost"}
        for change in analysis["changes"]
        if change["kind"] == "hedge"
        for value in change["removed"]
    ] + [
        {"word": value, "direction": "gained"}
        for change in analysis["changes"]
        if change["kind"] == "hedge"
        for value in change["added"]
    ]
    return {
        "numeric_changes": _legacy_changes(analysis, "numeric"),
        "entity_changes": _legacy_changes(analysis, "entity"),
        "hedge_changes": hedge_changes,
        "polarity_changes": _legacy_changes(analysis, "polarity"),
        "attribution_changes": _legacy_changes(analysis, "attribution"),
        "scope_changes": _legacy_changes(analysis, "scope"),
        "similarity": analysis["similarity"],
        "mutation_types": analysis["mutation_types"],
        "segments": segments,
        "analysis": analysis,
        "algorithm_version": analysis["algorithm_version"],
        "meaningful_change": analysis["meaningful_change"],
        "observed_propagation": False,
    }
