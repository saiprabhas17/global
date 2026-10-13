"""Deterministic English text-change heuristics, not a transmission detector.

Spans are half-open Python character offsets into the original input strings.
Only lexical changes count as meaningful; whitespace, case and standalone
punctuation changes cannot by themselves establish a mutation parent.
"""

import difflib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypedDict

from app.services.evolution.quantities import Quantity, extract_quantities, quantity_key

ALGORITHM_VERSION = "gmee-typed-text-v2"

_NUMBER_PATTERN = r"[+−-]?(?:[$€£]\s*)?\d+(?:[.,]\d+)*(?:\s?%|[kmbKMB]\b)?"
_NUMBER_RE = re.compile(rf"^{_NUMBER_PATTERN}$")
_TOKEN_RE = re.compile(
    rf"{_NUMBER_PATTERN}|[^\W\d_]+(?:['’][^\W\d_]+)*|\w+|[^\w\s]",
    re.UNICODE,
)

HEDGE_WORDS = frozenset(
    {
        "allegedly",
        "reportedly",
        "unverified",
        "claimed",
        "claims",
        "rumored",
        "rumoured",
        "rumour",
        "rumor",
        "purportedly",
        "seemingly",
        "apparently",
        "may",
        "might",
        "could",
        "possibly",
        "perhaps",
        "sources",
        "according",
        "believed",
        "suspected",
        "speculation",
        "unconfirmed",
        "likely",
        "unlikely",
        "probably",
        "potentially",
        "estimated",
    }
)
POLARITY_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "without",
        "deny",
        "denies",
        "denied",
        "false",
        "untrue",
        "cannot",
        "can't",
        "won't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "doesn't",
        "didn't",
        "don't",
        "hasn't",
        "haven't",
        "hadn't",
        "couldn't",
        "wouldn't",
        "shouldn't",
        "mustn't",
    }
)
ATTRIBUTION_WORDS = frozenset(
    {
        "according",
        "sources",
        "source",
        "said",
        "says",
        "say",
        "stated",
        "states",
        "reported",
        "reports",
        "report",
        "claimed",
        "claims",
        "announced",
        "announces",
        "cited",
        "cites",
        "told",
        "spokesperson",
    }
)
SCOPE_WORDS = frozenset(
    {
        "all",
        "every",
        "everyone",
        "everything",
        "any",
        "some",
        "many",
        "most",
        "few",
        "several",
        "none",
        "only",
        "each",
        "both",
        "entire",
        "always",
        "sometimes",
        "often",
        "everywhere",
        "nationwide",
        "worldwide",
        "globally",
        "locally",
        "approximately",
        "about",
        "nearly",
        "at least",
        "at most",
        "more than",
        "less than",
        "up to",
        "over",
        "under",
    }
)
_SCOPE_PHRASES = ("at least", "at most", "more than", "less than", "up to")
_STOPWORDS = (
    frozenset(
        {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "it",
            "its",
            "as",
            "by",
            "with",
            "from",
            "that",
            "this",
            "has",
            "have",
            "had",
            "but",
            "will",
            "after",
            "before",
            "into",
            "than",
            "then",
            "new",
            "first",
            "officials",
            "president",
        }
    )
    | HEDGE_WORDS
    | POLARITY_WORDS
    | ATTRIBUTION_WORDS
    | SCOPE_WORDS
)

LIMITATIONS = (
    "English lexical heuristics only; no semantic entailment, fact checking, or model inference.",
    "Capitalized words are entity proxies, not named-entity recognition; sentence starts, aliases and uncased names can be misclassified.",
    "Quantity normalization covers explicit common English units and decimal formats only; ambiguous locales, inferred units, dates, number words and currency exchange are not resolved.",
    "Polarity, hedge, attribution and scope lexicons do not resolve negation scope, quotation, sarcasm or coreference.",
    "Case, whitespace and standalone punctuation are ignored for meaningful-change classification; this can miss punctuation-only meaning changes.",
    "Lexical differences and timestamps do not prove copying, causal lineage or observed propagation.",
    "Naive timestamps are interpreted as UTC; missing or equal timestamps cannot establish temporal precedence.",
    "Correction and update cues report wording in supplied text or user context; they do not independently verify a correction, citation, claim accuracy or event identity.",
)

ChangeKind = Literal[
    "numeric", "entity", "polarity", "hedge", "attribution", "scope", "wording"
]
_KIND_TYPES: dict[ChangeKind, str] = {
    "numeric": "NUMERIC_DRIFT",
    "entity": "ENTITY_SUBSTITUTION",
    "polarity": "POLARITY_SHIFT",
    "hedge": "HEDGING_SHIFT",
    "attribution": "ATTRIBUTION_SHIFT",
    "scope": "SCOPE_SHIFT",
    "wording": "WORDING_DRIFT",
}
_TYPED_KINDS: tuple[ChangeKind, ...] = (
    "numeric",
    "entity",
    "polarity",
    "hedge",
    "attribution",
    "scope",
)


@dataclass(frozen=True)
class TextToken:
    text: str
    start: int
    end: int
    canonical: str | None = None

    @property
    def normalized(self) -> str:
        if self.canonical is not None:
            return self.canonical
        return re.sub(
            r"\s+", "", self.text.casefold().replace("’", "'").replace("−", "-")
        )

    @property
    def lexical(self) -> bool:
        return any(c.isalnum() for c in self.text)


class TextSpan(TypedDict):
    start: int
    end: int
    text: str


class ChangedSpan(TypedDict):
    operation: str
    older_span: TextSpan
    newer_span: TextSpan
    kinds: list[ChangeKind]


class ChangeDetail(ChangedSpan):
    kind: ChangeKind
    mutation_type: str
    removed: list[str]
    added: list[str]
    direction: str


class ContextSignal(TypedDict):
    kind: str
    text: str
    start: int
    end: int
    basis: str


class QuantitiesByVersion(TypedDict):
    older: list[Quantity]
    newer: list[Quantity]


class MutationAnalysis(TypedDict):
    algorithm_version: str
    mutation_types: list[str]
    changes: list[ChangeDetail]
    changed_spans: list[ChangedSpan]
    similarity: float
    meaningful_change: bool
    observed_propagation: bool
    inference: str
    text_basis: str
    span_offset_unit: str
    older_timestamp: str | None
    newer_timestamp: str | None
    temporal_order: str
    lag_seconds: float | None
    limitations: list[str]
    relation_category: str
    relation_explanation: str
    context_signals: list[ContextSignal]
    quantities: QuantitiesByVersion
    quantity_equivalent: bool
    context_note: str | None
    provenance_url: str | None
    provenance_basis: str


def tokens_with_spans(text: str) -> list[TextToken]:
    return [TextToken(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def tokenize(text: str) -> list[str]:
    return [t.text for t in tokens_with_spans(text or "")]


def _comparison_tokens(text: str, quantities: list[Quantity]) -> list[TextToken]:
    """Collapse supported quantities while retaining original character spans."""
    tokens: list[TextToken] = []
    cursor = 0
    for quantity in quantities:
        tokens.extend(
            TextToken(t.text, t.start + cursor, t.end + cursor)
            for t in tokens_with_spans(text[cursor : quantity["start"]])
            if t.lexical
        )
        tokens.append(
            TextToken(
                quantity["text"], quantity["start"], quantity["end"],
                quantity_key(quantity),
            )
        )
        cursor = quantity["end"]
    tokens.extend(
        TextToken(t.text, t.start + cursor, t.end + cursor)
        for t in tokens_with_spans(text[cursor:])
        if t.lexical
    )
    return tokens


_CONTEXT_CUES = {
    "correction": re.compile(r"\b(?:correction|corrected|retracted|retraction|erratum)\b", re.IGNORECASE),
    "update": re.compile(r"\b(?:updated|revised|revision|latest (?:figures|estimate|count)|new estimate)\b", re.IGNORECASE),
}


def _context_signals(newer_text: str, context_note: str | None) -> list[ContextSignal]:
    signals: list[ContextSignal] = []
    for basis, text in (("newer_text", newer_text), ("user_context", context_note or "")):
        for kind, pattern in _CONTEXT_CUES.items():
            for match in pattern.finditer(text):
                prefix = text[max(0, match.start() - 40) : match.start()]
                # Do not promote a negated correction/update to a positive cue.
                if re.search(r"\b(?:not|never|no|without)\b[^.!?;]{0,24}$", prefix, re.IGNORECASE):
                    continue
                signals.append({
                    "kind": kind, "text": match.group(), "start": match.start(),
                    "end": match.end(), "basis": basis,
                })
    return sorted(signals, key=lambda item: (item["basis"], item["start"], item["kind"]))


def utc_timestamp(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _span(text: str, tokens: list[TextToken], start: int, end: int) -> TextSpan:
    if start == end:
        offset = tokens[start].start if start < len(tokens) else len(text)
        return {"start": offset, "end": offset, "text": ""}
    lo, hi = tokens[start].start, tokens[end - 1].end
    return {"start": lo, "end": hi, "text": text[lo:hi]}


def _entity(token: TextToken) -> bool:
    return (
        token.text.isalpha()
        and token.text[:1].isupper()
        and (len(token.text) > 2 or (len(token.text) > 1 and token.text.isupper()))
        and token.normalized not in _STOPWORDS
    )


def _features(tokens: list[TextToken], kind: ChangeKind) -> list[str]:
    if kind == "numeric":
        return [
            re.sub(r"\s+", "", t.text.casefold().replace("−", "-"))
            for t in tokens
            if t.canonical is not None or _NUMBER_RE.fullmatch(t.text)
        ]
    if kind == "entity":
        return [t.text for t in tokens if _entity(t)]
    lexicons = {
        "polarity": POLARITY_WORDS,
        "hedge": HEDGE_WORDS,
        "attribution": ATTRIBUTION_WORDS,
        "scope": SCOPE_WORDS,
    }
    if kind == "wording":
        return [t.normalized for t in tokens]
    words = [t.normalized for t in tokens]
    found = [w for w in words if w in lexicons[kind]]
    if kind == "scope":
        found.extend(
            phrase
            for phrase in _SCOPE_PHRASES
            if any(words[i : i + 2] == phrase.split() for i in range(len(words) - 1))
        )
    return found


def _touches_attribution(tokens: list[TextToken], start: int, end: int) -> bool:
    # Include the source next to a reporting cue, so "Alice said" -> "Bob said"
    # changes attribution even when the reporting verb itself is unchanged.
    context = tokens[max(0, start - 4) : min(len(tokens), end + 4)]
    return any(t.normalized in ATTRIBUTION_WORDS for t in context)


def analyze_text_change(
    older_text: str,
    newer_text: str,
    *,
    older_timestamp: datetime | None = None,
    newer_timestamp: datetime | None = None,
    context_note: str | None = None,
    provenance_url: str | None = None,
) -> MutationAnalysis:
    older_quantities, newer_quantities = extract_quantities(older_text), extract_quantities(newer_text)
    old = _comparison_tokens(older_text, older_quantities)
    new = _comparison_tokens(newer_text, newer_quantities)
    matcher = difflib.SequenceMatcher(
        a=[t.normalized for t in old],
        b=[t.normalized for t in new],
        autojunk=False,
    )
    changed_spans: list[ChangedSpan] = []
    changes: list[ChangeDetail] = []
    found_kinds: set[ChangeKind] = set()
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue
        older_span, newer_span = (
            _span(older_text, old, i1, i2),
            _span(newer_text, new, j1, j2),
        )
        old_run, new_run = old[i1:i2], new[j1:j2]
        kinds: list[ChangeKind] = []
        for kind in _TYPED_KINDS:
            removed, added = _features(old_run, kind), _features(new_run, kind)
            attribution_source = (
                kind == "attribution"
                and (
                    _touches_attribution(old, i1, i2)
                    or _touches_attribution(new, j1, j2)
                )
                and any(_entity(t) for t in old_run + new_run)
            )
            # Scope phrases may straddle an opcode boundary ("at least" ->
            # "at most"); retain the complete phrase's context for detection.
            if kind == "scope":
                old_scope = _features(old[max(0, i1 - 1) : min(len(old), i2 + 1)], kind)
                new_scope = _features(new[max(0, j1 - 1) : min(len(new), j2 + 1)], kind)
                if old_scope != new_scope:
                    removed, added = old_scope, new_scope
            if attribution_source and removed == added:
                removed = [t.normalized for t in old_run if _entity(t)]
                added = [t.normalized for t in new_run if _entity(t)]
            if removed == added:
                continue
            kinds.append(kind)
            found_kinds.add(kind)
            changes.append(
                {
                    "operation": operation,
                    "older_span": older_span,
                    "newer_span": newer_span,
                    "kinds": [kind],
                    "kind": kind,
                    "mutation_type": _KIND_TYPES[kind],
                    "removed": removed,
                    "added": added,
                    "direction": "shifted"
                    if removed and added
                    else "lost"
                    if removed
                    else "gained",
                }
            )
        if not kinds:
            kinds = ["wording"]
            changes.append(
                {
                    "operation": operation,
                    "older_span": older_span,
                    "newer_span": newer_span,
                    "kinds": kinds,
                    "kind": "wording",
                    "mutation_type": "WORDING_DRIFT",
                    "removed": _features(old_run, "wording"),
                    "added": _features(new_run, "wording"),
                    "direction": "shifted"
                    if old_run and new_run
                    else "lost"
                    if old_run
                    else "gained",
                }
            )
        changed_spans.append(
            {
                "operation": operation,
                "older_span": older_span,
                "newer_span": newer_span,
                "kinds": kinds,
            }
        )

    similarity = matcher.ratio()
    meaningful = bool(changed_spans)
    types = [_KIND_TYPES[k] for k in _TYPED_KINDS if k in found_kinds]
    if not meaningful:
        types = ["NEAR_DUPLICATE"]
    elif similarity < 0.65:
        types.append("FRAMING_SHIFT")
    elif similarity < 0.9 or not types:
        types.append("WORDING_DRIFT")

    older_at, newer_at = utc_timestamp(older_timestamp), utc_timestamp(newer_timestamp)
    lag = (
        (newer_at - older_at).total_seconds()
        if older_at is not None and newer_at is not None
        else None
    )
    temporal_order = (
        "unknown"
        if lag is None
        else "strictly_older"
        if lag > 0
        else "equal"
        if lag == 0
        else "reversed"
    )
    quantity_equivalent = bool(
        older_quantities and not meaningful
        and [q["text"].strip().casefold() for q in older_quantities]
        != [q["text"].strip().casefold() for q in newer_quantities]
    )
    signals = _context_signals(newer_text, context_note)
    if not meaningful:
        relation_category = "EQUIVALENT_QUANTITY" if quantity_equivalent else "NO_LEXICAL_CHANGE"
        relation_explanation = (
            "The ordered quantities and remaining wording match after explicit unit conversion."
            if quantity_equivalent else "No meaningful lexical change was found."
        )
    elif temporal_order in {"equal", "reversed"}:
        relation_category = "CONTEXT_REQUIRED"
        relation_explanation = "The timestamps do not establish the supplied older-to-newer order."
    elif any(s["kind"] == "correction" for s in signals):
        relation_category = "REPORTED_CORRECTION"
        relation_explanation = "A correction cue appears in the supplied text or context; the correction and claim accuracy remain unverified."
    elif any(s["kind"] == "update" for s in signals):
        relation_category = "REPORTED_UPDATE"
        relation_explanation = "An update cue appears in the supplied text or context; a changed figure may be a reporting update and is not evidence of misinformation."
    elif any(k in found_kinds for k in ("numeric", "entity", "polarity", "attribution")):
        relation_category = "CONTEXT_REQUIRED"
        relation_explanation = "A substantive textual difference needs event context and independent evidence before it can be judged a correction, contradiction or distortion."
    else:
        relation_category = "TEXTUAL_CHANGE"
        relation_explanation = "The wording changed; lexical comparison alone does not establish changed meaning, accuracy or copying."
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "mutation_types": types,
        "changes": changes,
        "changed_spans": changed_spans,
        "similarity": round(similarity, 4),
        "meaningful_change": meaningful,
        "observed_propagation": False,
        "inference": "inferred_text_change",
        "text_basis": "original_input_character_offsets",
        "span_offset_unit": "unicode_code_points",
        "older_timestamp": older_at.isoformat() if older_at is not None else None,
        "newer_timestamp": newer_at.isoformat() if newer_at is not None else None,
        "temporal_order": temporal_order,
        "lag_seconds": lag,
        "limitations": list(LIMITATIONS),
        "relation_category": relation_category,
        "relation_explanation": relation_explanation,
        "context_signals": signals,
        "quantities": {"older": older_quantities, "newer": newer_quantities},
        "quantity_equivalent": quantity_equivalent,
        "context_note": context_note,
        "provenance_url": provenance_url,
        "provenance_basis": "user_supplied_reference_unverified" if provenance_url else "text_comparison_only",
    }
