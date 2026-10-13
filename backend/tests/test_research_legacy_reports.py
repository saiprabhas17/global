"""Compatibility publication scripts must not certify pseudo-gold or sample size."""

from collections import Counter
from typing import Any

from scripts.generate_pub_report import (
    compute_inter_annotator_agreement,
    generate_md_report,
)
from scripts.run_eval import evaluate
from tests.test_research_support import research_records


def legacy_rows() -> list[dict[str, Any]]:
    rows = []
    for record in research_records():
        for signal in ("old-signal-a", "old-signal-b"):
            rows.append({"pid": record["pair_id"], "bucket": record["bucket"], "sim": record["cosine_score"], "text_a": record["text_a"], "text_b": record["text_b"], "label": record["gold_label"], "origin": "legacy", "annotator": signal, "emb_a": None, "emb_b": None})
    return rows


def test_weak_labels_are_exploratory_only_not_humans():
    rows = legacy_rows()
    report = evaluate(rows, n_boot=20)
    assert report["n_pairs"] == report["n_votes"] == 0
    assert report["n_votes_all_origins"] == 36
    assert report["exploratory_by_origin"]["legacy"]["n_pairs"] == 18
    assert report["exploratory_by_origin"]["legacy"]["metrics"]["brier"] is None
    assert report["status"] == "exploratory" and report["warning"]
    assert compute_inter_annotator_agreement(rows) == {}
    legacy_agreement = compute_inter_annotator_agreement(rows, origin="legacy")
    assert legacy_agreement and all(not value["authenticated_human"] for value in legacy_agreement.values())


def test_markdown_keeps_useful_tables_without_publication_assurances():
    rows = legacy_rows()
    report = evaluate(rows, n_boot=20)
    stats = {"total_pairs": 18, "total_labels": 36, "buckets": dict(Counter(row["bucket"] for row in rows[::2])), "annotators": {"legacy:old-signal-a": {"EVOLVED": 3, "SAME_STORY": 6, "DISTINCT": 9}}, "origins": {"legacy": 36}, "human_annotators": []}
    markdown = generate_md_report(report, stats)
    assert "Exploratory" in markdown and "Not Publication Gold" in markdown
    assert "legacy (not human gold)" in markdown and "Vote provenance" in markdown
    assert "Voter Label Distribution" in markdown and "Pairs by Similarity Bucket" in markdown
    assert "WARNING" in markdown
    assert "Sample size meets" not in markdown and "sufficient for publication" not in markdown
