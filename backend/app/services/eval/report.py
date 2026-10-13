"""Origin-aware live diagnostics, explicitly distinct from held-out research."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.services.eval.metrics import (
    auroc,
    average_precision,
    best_f1_threshold,
    bootstrap_ci,
    f1_at_threshold,
    sweep_thresholds,
)
from app.services.eval.provenance import (
    CONSENSUS_RULE,
    consensus,
    exploratory_consensus,
    human_identity,
    origin_counts,
    pair_consensus,
)

__all__ = ["consensus"]

POSITIVE_LABELS = ("SAME_STORY", "EVOLVED")
ENGINE_OPERATING_POINT = 0.60
THRESHOLD_SWEEP = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
WARNING = (
    "Exploratory live-corpus diagnostics, not a publication result: no frozen "
    "event-disjoint train/dev/test protocol. A pair needs at least two independent "
    "authenticated human votes with no disagreement. No vote/sample-count threshold "
    "establishes publication validity. Cosine is a score, not a probability; "
    "same-sample best-F1 is descriptive only. Use export_research_dataset.py and "
    "run_research_experiment.py for held-out evaluation."
)


def score_diagnostics(pairs: list[dict[str, Any]], n_boot: int = 500, seed: int = 1729) -> dict[str, Any]:
    if not pairs:
        return {"n_pairs": 0, "n_positive": 0, "metrics": None}
    labels = [p["label"] in POSITIVE_LABELS for p in pairs]
    scores = [float(p["sim"]) for p in pairs]
    both_classes = 0 < sum(labels) < len(labels)
    return {
        "n_pairs": len(pairs), "n_positive": sum(labels),
        "metrics": {
            "auroc": round(auroc(scores, labels), 4) if both_classes else None,
            "auprc": average_precision(scores, labels) if both_classes else None,
            "auroc_ci95": bootstrap_ci(scores, labels, auroc, n_boot=n_boot, seed=seed) if both_classes else None,
            "ci_unit": "pair (descriptive only; correlated event groups are not accounted for)",
            "best_f1": best_f1_threshold(scores, labels),
            "threshold_selection": "same sample; exploratory, never a held-out operating point",
            "f1_at_operating_point": f1_at_threshold(scores, labels, ENGINE_OPERATING_POINT),
            "score_kind": "score_not_probability", "brier": None, "ece": None,
            "calibration_message": "Raw cosine does not define a calibrated probability; Brier/ECE are intentionally not computed",
        },
    }


def build_report(rows: list[dict[str, Any]], n_boot: int = 500, seed: int = 1729) -> dict[str, Any]:
    pairs = consensus(rows)
    human = [r for r in rows if human_identity(r) is not None]
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pid"]), []).append(row)
    statuses = Counter(pair_consensus(v)["status"] for v in by_pair.values())
    diagnostics = score_diagnostics(pairs, n_boot=n_boot, seed=seed)
    report: dict[str, Any] = {
        "status": "exploratory", "publication_ready": False,
        "n_pairs": len(pairs), "n_positive": diagnostics["n_positive"],
        "n_votes": len(human), "n_votes_all_origins": len(rows),
        "origin_counts": origin_counts(rows), "human_consensus_status": dict(sorted(statuses.items())),
        "consensus_rule": CONSENSUS_RULE,
        "operating_point": ENGINE_OPERATING_POINT,
        "method": "stored engine cosine (historical embedding model/version not verified)",
        "warning": WARNING,
        "exploratory_by_origin": {},
    }
    for origin in ("legacy", "automatic", "test"):
        weak = exploratory_consensus(rows, origin)
        report["exploratory_by_origin"][origin] = {
            "status": "exploratory_weak_labels", "gold": False,
            "warning": f"{origin} votes are not independently authenticated human gold; do not quote these diagnostics as human evaluation",
            **score_diagnostics(weak, n_boot=n_boot, seed=seed),
        }
    if not pairs:
        report["message"] = "no resolved authenticated-human consensus pairs yet; label with two independent signed-in users and resolve disagreements"
        return report
    report["metrics"] = diagnostics["metrics"]
    labels = [p["label"] in POSITIVE_LABELS for p in pairs]
    scores = [float(p["sim"]) for p in pairs]
    report["sweep"] = sweep_thresholds(scores, labels, list(THRESHOLD_SWEEP))
    report["reliability"] = []
    buckets: dict[str, dict[str, int]] = {}
    for pair in pairs:
        entry = buckets.setdefault(str(pair["bucket"]), {"n": 0, "SAME_STORY": 0, "EVOLVED": 0, "DISTINCT": 0})
        entry["n"] += 1
        entry[str(pair["label"])] += 1
    report["by_bucket"] = buckets
    return report


async def collect_labeled_pairs(db: Any) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from app.models.eval import EvalPair, EvalPairLabel

    statement = select(
        EvalPair.id.label("pid"), EvalPair.bucket,
        EvalPair.sim_score.label("sim"), EvalPair.split, EvalPair.event_group,
        EvalPairLabel.id.label("vote_id"), EvalPairLabel.annotator,
        EvalPairLabel.annotator_user_id, EvalPairLabel.origin,
        EvalPairLabel.label, EvalPairLabel.mutation_types,
    ).join(EvalPairLabel, EvalPairLabel.pair_id == EvalPair.id).order_by(EvalPair.id, EvalPairLabel.origin, EvalPairLabel.annotator)
    return [dict(row._mapping) for row in (await db.execute(statement)).all()]
