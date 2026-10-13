"""Coverage and agreement counts distinguish authenticated humans from all origins."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from app.services.eval.metrics import cohen_kappa
from app.services.eval.provenance import (
    ORIGINS,
    human_identity,
    origin_counts,
    origin_of,
    pair_consensus,
)


def per_annotator(rows: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for annotator, label, n in rows:
        entry = output.setdefault(annotator, {"labels": {}, "total": 0})
        entry["labels"][label] = entry["labels"].get(label, 0) + n
        entry["total"] += n
    return output


def build_progress(pairs: list[dict[str, Any]], votes: list[dict[str, Any]]) -> dict[str, Any]:
    human = [v for v in votes if human_identity(v) is not None]
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        by_pair[str(vote["pair_id"])].append(vote)
    results = {str(p["pair_id"]): pair_consensus(by_pair[str(p["pair_id"])]) for p in pairs}
    human_pairs = {str(v["pair_id"]) for v in human}
    agreed = {pid for pid, result in results.items() if result["status"] == "agreed"}
    statuses = Counter(result["status"] for result in results.values())
    tallies = Counter((str(v["annotator"]), str(v["label"])) for v in human)

    matrix: dict[str, dict[str, str]] = defaultdict(dict)
    for vote in human:
        matrix[str(vote["pair_id"])][str(vote["annotator"])] = str(vote["label"])
    agreement = []
    for a, b in combinations(sorted({v["annotator"] for v in human}), 2):
        shared = [(m[a], m[b]) for _, m in sorted(matrix.items()) if a in m and b in m]
        if not shared:
            continue
        kappa = cohen_kappa([v[0] for v in shared], [v[1] for v in shared])
        agreement.append({
            "annotators": [a, b], "pairs": len(shared),
            "kappa": round(kappa, 4) if kappa is not None else None,
            "raw_agreement": sum(x == y for x, y in shared) / len(shared),
            "origin": "human", "warning": "Descriptive agreement; shared-pair count does not establish annotation independence or validity.",
        })

    def coverage(field: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for pair in pairs:
            grouped[str(pair[field])].append(str(pair["pair_id"]))
        return [{field: key, "pairs": len(ids), "labeled": sum(pid in human_pairs for pid in ids), "consensus": sum(pid in agreed for pid in ids)} for key, ids in sorted(grouped.items())]

    origins: dict[str, dict[str, Any]] = {origin: dict(counts) for origin, counts in origin_counts(votes).items()}
    for origin in ORIGINS:
        origins[origin]["labels"] = dict(sorted(Counter(str(v["label"]) for v in votes if origin_of(v) == origin).items()))
    return {
        "total_pairs": len(pairs),
        "labeled_votes": len(human), "labeled_pairs_distinct": len(human_pairs),
        "human_votes": len(human), "human_labeled_pairs": len(human_pairs),
        "consensus_pairs": len(agreed),
        "disagreement_pairs": statuses["disagreement"],
        "needs_human_votes_pairs": statuses["needs_human_votes"],
        "total_votes_all_origins": len(votes),
        "labeled_pairs_all_origins": len(by_pair),
        "origins": origins,
        "invalid_human_votes": origins["human"]["votes"] - len(human),
        "by_bucket": coverage("bucket"), "by_split": coverage("split"),
        "per_annotator": per_annotator((a, label, n) for (a, label), n in sorted(tallies.items())),
        "inter_annotator": agreement,
        "complete": bool(pairs) and len(agreed) == len(pairs),
        "warning": "Complete means unanimous current labels from at least two authenticated humans per pair, not publication validity. Automatic, legacy and test votes never complete human coverage.",
    }
