"""Versioned human instructions and source-anchored annotation validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUBRIC_VERSION = "claim-pairs-v2"
MUTATION_TYPES = (
    "NUMERIC_DRIFT", "ENTITY_SUBSTITUTION", "POLARITY_SHIFT", "HEDGING_SHIFT",
    "ATTRIBUTION_SHIFT", "SCOPE_SHIFT", "FRAMING_SHIFT", "WORDING_DRIFT", "NEAR_DUPLICATE",
)
ContextRelation = Literal[
    "PARAPHRASE", "UPDATE_OR_CORRECTION", "INCOMPATIBLE_ASSERTION",
    "INSUFFICIENT_CONTEXT", "UNRELATED",
]
RUBRIC: dict[str, Any] = {
    "version": RUBRIC_VERSION,
    "labels": {
        "SAME_STORY": "The same factual proposition, allowing wording changes without a material change in meaning.",
        "EVOLVED": "A shared event/proposition with changed detail, qualification, attribution or scope. This does not establish falsehood or copying.",
        "DISTINCT": "Different events or unrelated propositions, even when entities or words overlap.",
    },
    "context_relations": {
        "PARAPHRASE": "Equivalent meaning given the available context.",
        "UPDATE_OR_CORRECTION": "The available text explicitly identifies an update or correction; record the basis in notes.",
        "INCOMPATIBLE_ASSERTION": "Assertions cannot both hold under the same entity, time, quantity and scope; this does not decide which is true.",
        "INSUFFICIENT_CONTEXT": "The available text cannot distinguish an update, changed scope or incompatibility. Prefer this when timing or attribution is missing.",
        "UNRELATED": "There is no shared factual proposition to compare.",
    },
    "mutation_types": list(MUTATION_TYPES),
    "span_offsets": "Unicode code points; start inclusive/end exclusive",
    "guidance": [
        "Judge independently before discussing the pair. Model scores and other votes are hidden.",
        "A false factual assertion is still a claim. Relatedness, text changes and truth are separate targets.",
        "Pair A/B order does not establish chronology or copying. A larger number may be a legitimate update.",
        "Select exact changed text on either side. An insertion/deletion needs only the side containing the text.",
        "An empty span list means explicitly reviewed with no changed span; omit it when spans were not annotated.",
        "Independent review never erases dissent. Gold requires at least two authenticated humans with unanimous current labels; revisions retain history.",
    ],
}


class ChangedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    side: Literal["a", "b"]
    start: int = Field(ge=0, strict=True)
    end: int = Field(ge=1, strict=True)
    text: str = Field(min_length=1, max_length=2000)
    mutation_type: str

    @model_validator(mode="after")
    def check_interval(self) -> ChangedSpan:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        if self.mutation_type not in MUTATION_TYPES:
            raise ValueError("unknown mutation type; consult /eval/rubric")
        return self


def validate_spans(spans: list[ChangedSpan], text_a: str, text_b: str) -> list[dict[str, Any]]:
    """Exact source anchoring rejects stale selections and UTF-16 mistakes."""
    normalized: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for span in spans:
        source = text_a if span.side == "a" else text_b
        if span.end > len(source) or source[span.start:span.end] != span.text:
            raise ValueError("changed span does not match the stored claim text at its Unicode offsets")
        normalized[(span.side, span.start, span.end, span.mutation_type)] = span.model_dump()
    return [normalized[key] for key in sorted(normalized)]
