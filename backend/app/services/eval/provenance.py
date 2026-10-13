"""Authentication evidence and conservative, origin-aware gold consensus."""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from typing import Any

from app.services.eval.annotation import RUBRIC_VERSION

LABELS = ("SAME_STORY", "EVOLVED", "DISTINCT")
ORIGINS = ("human", "automatic", "legacy", "test")
CONSENSUS_RULE = "at least two distinct authenticated user IDs; all current human labels agree"


def origin_of(vote: dict[str, Any]) -> str:
    origin = vote.get("origin", "legacy")
    return str(origin) if origin in ORIGINS else "legacy"


def human_identity(vote: dict[str, Any]) -> str | None:
    """Never infer human provenance from a name, or count an identity twice."""
    if origin_of(vote) != "human" or vote.get("label") not in LABELS:
        return None
    value = vote.get("annotator_user_id")
    if value is None:
        return None
    try:
        user_id = str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None
    if vote.get("annotator") != f"user:{user_id}":
        return None
    return user_id


def pair_consensus(votes: list[dict[str, Any]]) -> dict[str, Any]:
    """Disagreement (including conflicting copies of a vote) is unresolved."""
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        identity = human_identity(vote)
        if identity is not None:
            by_user[identity].append(vote)
    labels = {str(v["label"]) for vs in by_user.values() for v in vs}
    status = "disagreement" if len(labels) > 1 else (
        "agreed" if len(by_user) >= 2 else "needs_human_votes"
    )
    mutation_types = None
    mutation_status = "not_annotated"
    if status == "agreed":
        annotations = [v.get("mutation_types") for vs in by_user.values() for v in vs]
        known = [tuple(sorted(set(v))) for v in annotations if isinstance(v, list)]
        if known:
            if len(set(known)) > 1:
                mutation_status = "disagreement"
            elif len(known) != len(annotations):
                mutation_status = "incomplete"
            else:
                mutation_status = "agreed"
                mutation_types = list(known[0])
    annotation_results: dict[str, Any] = {}
    for field in ("context_relation", "changed_spans"):
        values = [
            v.get("provenance", {}).get(field)
            if isinstance(v.get("provenance"), dict)
            and v["provenance"].get("rubric_version") == RUBRIC_VERSION else None
            for vs in by_user.values() for v in vs
        ]
        known_values = [value for value in values if value is not None]
        # Span equality is exact; overlapping spans do not silently become gold.
        encoded = {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in known_values}
        annotation_status = "not_annotated"
        agreed_value = None
        if status == "agreed" and known_values:
            if len(encoded) > 1:
                annotation_status = "disagreement"
            elif len(known_values) != len(values):
                annotation_status = "incomplete"
            else:
                annotation_status, agreed_value = "agreed", known_values[0]
        annotation_results[f"{field}_status"] = annotation_status
        annotation_results[field] = agreed_value
    return {
        "status": status,
        "label": next(iter(labels)) if status == "agreed" else None,
        "human_vote_count": len(by_user),
        "human_user_ids": sorted(by_user),
        "mutation_status": mutation_status,
        "mutation_types": mutation_types,
        **annotation_results,
        "rule": CONSENSUS_RULE,
    }


def consensus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row.get("pid", row.get("pair_id")))].append(row)
    output = []
    for pid, votes in sorted(by_pair.items()):
        result = pair_consensus(votes)
        if result["status"] != "agreed":
            continue
        eligible = sorted(
            (v for v in votes if human_identity(v) is not None),
            key=lambda v: (str(v["annotator_user_id"]), str(v.get("id", ""))),
        )
        base = dict(eligible[0])
        base.update({
            "pid": pid,
            "label": result["label"],
            "mutation_types": result["mutation_types"],
            "votes": [str(v["label"]) for v in eligible],
            "human_vote_count": result["human_vote_count"],
            "consensus": result,
        })
        output.append(base)
    return output


def exploratory_consensus(rows: list[dict[str, Any]], origin: str) -> list[dict[str, Any]]:
    """Weak-label majority is diagnostic only; ties are not adjudicated gold."""
    if origin == "human" or origin not in ORIGINS:
        raise ValueError("Exploratory consensus requires automatic, legacy, or test origin")
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if origin_of(row) == origin and row.get("label") in LABELS:
            by_pair[str(row.get("pid", row.get("pair_id")))][str(row.get("annotator", row.get("id")))] = row
    output = []
    for pid, identities in sorted(by_pair.items()):
        votes = list(identities.values())
        counts = Counter(str(v["label"]) for v in votes)
        winners = [label for label, n in counts.items() if n == max(counts.values())]
        if len(winners) != 1:
            continue
        chosen = min(votes, key=lambda v: str(v.get("annotator", "")))
        base = dict(chosen)
        base.update({"pid": pid, "label": winners[0], "votes": [str(v["label"]) for v in votes]})
        output.append(base)
    return output


def origin_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output = {}
    for origin in ORIGINS:
        votes = [v for v in rows if origin_of(v) == origin]
        output[origin] = {
            "votes": len(votes),
            "pairs": len({str(v.get("pid", v.get("pair_id"))) for v in votes}),
        }
    return output
