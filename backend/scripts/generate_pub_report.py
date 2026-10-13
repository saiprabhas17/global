# mypy: disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-any-generics=False

"""Compatibility-named exploratory report generator, NOT publication gold.

Preserves bucket/distribution/agreement and retrieval diagnostics. Primary
agreement/consensus uses authenticated humans only; automatic/legacy/test
weak-label diagnostics are explicitly separate. Machine voters are not
independent human annotators and agreement/sample size proves no validity.
Raw cosine/TF-IDF are scores, not calibrated probabilities.

Usage: python scripts/generate_pub_report.py [--output-json PATH] [--output-md PATH]
Defaults write new eval_exploratory_report.json/.md, never overwriting the
user's historical eval_report.json or eval_publication_report.md. For research
use export_research_dataset.py and run_research_experiment.py instead.
"""

import argparse
import ast
import asyncio
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.db.postgres import async_session_maker
from app.services.eval.metrics import cohen_kappa
from app.services.eval.provenance import human_identity, origin_of
from app.services.eval.report import WARNING
from scripts.run_eval import evaluate

OUT_JSON = Path(__file__).resolve().parent.parent / "eval_exploratory_report.json"
OUT_MD = Path(__file__).resolve().parent.parent / "eval_exploratory_report.md"


async def load_all_data() -> tuple[list[dict], dict]:
    """Load all labeled pairs + raw labels for inter-annotator analysis."""
    async with async_session_maker() as db:
        # Full labeled pairs with claim texts and embeddings
        rows = (
            await db.execute(
                text("""
                    SELECT p.id AS pid, p.bucket, p.sim_score AS sim,
                           p.split, p.event_group,
                           ta.claim_text AS text_a, tb.claim_text AS text_b,
                           ta.embedding AS emb_a, tb.embedding AS emb_b,
                           l.annotator, l.annotator_user_id, l.origin, l.label, l.mutation_types
                    FROM eval_pairs p
                    JOIN claims ta ON ta.id = p.claim_a_id
                    JOIN claims tb ON tb.id = p.claim_b_id
                    JOIN eval_pair_labels l ON l.pair_id = p.id
                    ORDER BY p.id, l.origin, l.annotator
                """)
            )
        ).all()
        labeled = [dict(r._mapping) for r in rows]

        # Stats
        total_pairs = (await db.execute(text("SELECT count(*) FROM eval_pairs"))).scalar()
        total_labels = (await db.execute(text("SELECT count(*) FROM eval_pair_labels"))).scalar()
        bucket_counts = (await db.execute(
            text("SELECT bucket, count(*) FROM eval_pairs GROUP BY bucket ORDER BY bucket")
        )).all()

        # Annotator stats
        annotator_stats = (await db.execute(text("""
            SELECT origin, annotator, label, count(*)
            FROM eval_pair_labels
            GROUP BY origin, annotator, label
            ORDER BY origin, annotator, label
        """))).all()

    stats: dict[str, Any] = {
        "total_pairs": total_pairs,
        "total_labels": total_labels,
        "buckets": {r[0]: r[1] for r in bucket_counts},
        "annotators": {},
        "origins": defaultdict(int),
        "human_annotators": sorted({r["annotator"] for r in labeled if human_identity(r) is not None}),
    }
    for origin, ann, label, count in annotator_stats:
        stats["annotators"].setdefault(f"{origin}:{ann}", {})[label] = count
        stats["origins"][origin] += count

    return labeled, stats


def parse_embedding(raw) -> np.ndarray:
    if raw is None:
        return np.zeros((768,), dtype=np.float32)
    if isinstance(raw, str):
        return np.asarray(ast.literal_eval(raw), dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def sbert_scores(rows: list[dict]) -> np.ndarray:
    out = []
    for r in rows:
        a = parse_embedding(r["emb_a"])
        b = parse_embedding(r["emb_b"])
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        out.append(float(a @ b) / denom if denom else 0.0)
    return np.asarray(out, dtype=np.float32)


def tfidf_scores(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    all_texts = texts_a + texts_b
    mat = normalize(vec.fit_transform(all_texts))
    n = len(texts_a)
    return np.asarray(np.sum(mat[:n].multiply(mat[n:]), axis=1)).ravel()


def compute_inter_annotator_agreement(rows: list[dict], origin: str = "human") -> dict:
    """Human agreement by default; weak-signal agreement is separate/diagnostic."""
    rows = [r for r in rows if human_identity(r) is not None] if origin == "human" else [r for r in rows if origin_of(r) == origin]
    # Group by (pair_id, annotator) -> label
    pair_annotator = {}
    for r in rows:
        pair_annotator[(str(r["pid"]), str(r["annotator"]))] = str(r["label"])

    # Find all annotators
    annotators = sorted({str(r["annotator"]) for r in rows})

    # For each pair of annotators, find shared pairs and compute kappa
    agreements = {}
    for a1, a2 in combinations(annotators, 2):
        shared_pairs = []
        labels_a1 = []
        labels_a2 = []
        all_pids = {str(r["pid"]) for r in rows}
        for pid in sorted(all_pids):
            l1 = pair_annotator.get((pid, a1))
            l2 = pair_annotator.get((pid, a2))
            if l1 is not None and l2 is not None:
                shared_pairs.append(pid)
                labels_a1.append(l1)
                labels_a2.append(l2)

        if shared_pairs:
            kappa = cohen_kappa(labels_a1, labels_a2)
            raw_agreement = sum(1 for x, y in zip(labels_a1, labels_a2) if x == y) / len(shared_pairs)
            agreements[f"{a1} vs {a2}"] = {
                "shared_pairs": len(shared_pairs),
                "cohen_kappa": round(kappa, 4) if kappa is not None else None,
                "raw_agreement": round(raw_agreement, 4),
                "origin": origin,
                "authenticated_human": origin == "human",
            }

    return agreements


def generate_md_report(report: dict, stats: dict) -> str:
    """Render diagnostic tables without misrepresenting weak votes as gold."""
    lines = []
    lines.append("# GMEE Exploratory Evaluation Report — Not Publication Gold\n")
    lines.append("> Live-corpus diagnostics. Automatic/legacy/test labels are weak signals, not independent human annotation or publication results.\n")
    lines.append("---\n")

    # Dataset summary
    lines.append("## 1. Evaluation Dataset Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total eval pairs | **{stats['total_pairs']}** |")
    lines.append(f"| Total annotations (labels) | **{stats['total_labels']}** |")
    lines.append(f"| Human consensus pairs (>=2 authenticated users; no disagreement) | **{report['n_pairs']}** |")
    lines.append(f"| Positive pairs (SAME_STORY + EVOLVED) | **{report['n_positive']}** ({report['n_positive']/max(report['n_pairs'],1)*100:.1f}%) |")
    lines.append(f"| Negative pairs (DISTINCT) | **{report['n_pairs'] - report['n_positive']}** ({(report['n_pairs'] - report['n_positive'])/max(report['n_pairs'],1)*100:.1f}%) |")
    lines.append(f"| Authenticated human annotator count | **{len(stats.get('human_annotators', []))}** |")
    lines.append("| Primary annotation method | Authenticated human votes only; weak origins excluded |")
    lines.append("")
    lines.append("### Vote provenance (origins never pooled into gold)\n")
    lines.append("| Origin | Votes |")
    lines.append("|---|---:|")
    for origin in ("human", "automatic", "legacy", "test"):
        lines.append(f"| {origin} | {stats.get('origins', {}).get(origin, 0)} |")
    lines.append("")
    lines.append("### Separate exploratory weak-label diagnostics\n")
    lines.append("| Origin | Diagnostic majority pairs | AUROC | AUPRC |")
    lines.append("|---|---:|---:|---:|")
    for origin, weak in report.get("exploratory_by_origin", {}).items():
        metrics = weak.get("metrics") or {}
        roc = f"{metrics['auroc']:.4f}" if isinstance(metrics.get("auroc"), (int, float)) else "N/A"
        pr = f"{metrics['auprc']:.4f}" if isinstance(metrics.get("auprc"), (int, float)) else "N/A"
        lines.append(f"| {origin} (not human gold) | {weak['n_pairs']} | {roc} | {pr} |")
    lines.append("")

    # Bucket distribution
    lines.append("### Pairs by Similarity Bucket\n")
    lines.append("| Bucket | Cosine Range | Pairs |")
    lines.append("|--------|-------------|-------|")
    bucket_ranges = {"b95": "0.95–1.00", "b85": "0.85–0.95", "b75": "0.75–0.85",
                     "b60": "0.60–0.75", "b40": "0.40–0.60", "b00": "0.00–0.40"}
    for b in ("b95", "b85", "b75", "b60", "b40", "b00"):
        n = stats["buckets"].get(b, 0)
        lines.append(f"| {b} | {bucket_ranges.get(b, '?')} | {n} |")
    lines.append("")

    # Annotator distribution
    if stats.get("annotators"):
        lines.append("### Voter Label Distribution (origin:name; machines are not humans)\n")
        lines.append("| Voter and origin | SAME_STORY | EVOLVED | DISTINCT | Total |")
        lines.append("|-----------|-----------|---------|----------|-------|")
        for ann, labels in sorted(stats["annotators"].items()):
            ss = labels.get("SAME_STORY", 0)
            ev = labels.get("EVOLVED", 0)
            di = labels.get("DISTINCT", 0)
            total = ss + ev + di
            lines.append(f"| {ann} | {ss} | {ev} | {di} | {total} |")
        lines.append("")

    # Inter-annotator agreement
    if "inter_annotator" in report:
        lines.append("## 2. Authenticated Human Agreement (descriptive only)\n")
        lines.append("| Annotator Pair | Shared Pairs | Cohen's κ | Raw Agreement |")
        lines.append("|---------------|-------------|-----------|---------------|")
        for pair_name, agr in report["inter_annotator"].items():
            k = f"{agr['cohen_kappa']:.3f}" if agr['cohen_kappa'] is not None else "N/A"
            lines.append(f"| {pair_name} | {agr['shared_pairs']} | {k} | {agr['raw_agreement']:.1%} |")
        lines.append("")

        # Interpret kappa
        kappas = [v["cohen_kappa"] for v in report["inter_annotator"].values() if v["cohen_kappa"] is not None]
        if kappas:
            avg_kappa = sum(kappas) / len(kappas)
            lines.append(f"> Descriptive mean κ = {avg_kappa:.3f}; agreement does not establish annotator independence, label truth, or publication validity.\n")

    # Main results
    lines.append("## 3. Exploratory Same-Sample Retrieval Diagnostics\n")
    lines.append("TF-IDF and best-F1 use this same diagnostic sample, not train/dev/test. Raw cosine/TF-IDF are scores; Brier/ECE are unavailable.\n")
    lines.append("| Method | AUROC | Descriptive pair CI | Best F1 (same sample) | F1@0.60 | Brier | ECE |")
    lines.append("|--------|-------|--------|---------|---------|-------|-----|")
    for name in ("sbert", "tfidf", "engine_embedding"):
        m = report.get(name, {})
        if not m:
            continue
        ci = m.get("auroc_ci95", {}) or {}
        ci_txt = f"[{ci.get('low', '?'):.2f}, {ci.get('high', '?'):.2f}]" if ci else "N/A"
        bf = m.get("best_f1", {})
        f1e = m.get("f1_at_engine_threshold", {})
        brier_v = f"{m.get('brier', 'N/A'):.4f}" if isinstance(m.get('brier'), (int, float)) else "N/A"
        ece_v = f"{m.get('ece', 'N/A'):.4f}" if isinstance(m.get('ece'), (int, float)) else "N/A"
        roc = f"{m['auroc']:.4f}" if isinstance(m.get('auroc'), (int, float)) else "N/A"
        best = f"{bf['f1']:.4f}" if isinstance(bf.get('f1'), (int, float)) else "N/A"
        fixed = f"{f1e['f1']:.4f}" if isinstance(f1e.get('f1'), (int, float)) else "N/A"
        lines.append(f"| {name} | {roc} | {ci_txt} | {best} | {fixed} | {brier_v} | {ece_v} |")
    lines.append("")

    # Significance
    if "paired_same_sample_diagnostics" in report:
        lines.append("### Exploratory McNemar Discordance (not grouped inference)\n")
        diagnostics = report["paired_same_sample_diagnostics"]
        for test_name, values in diagnostics.items():
            if isinstance(values, dict):
                lines.append(f"- **{test_name}**: b={values['b']:.0f}, c={values['c']:.0f}; uncorrected diagnostic p={values['p_value']:.4f}")
        lines.append(f"> {diagnostics.get('warning', '')}\n")

    # Per-bucket breakdown
    if "by_bucket" in report:
        lines.append("## 4. Per-Bucket Agreement\n")
        lines.append("| Bucket | Pairs | SAME_STORY | EVOLVED | DISTINCT | Positive Rate |")
        lines.append("|--------|-------|-----------|---------|----------|---------------|")
        for b in ("b95", "b85", "b75", "b60", "b40", "b00"):
            d = report["by_bucket"].get(b, {})
            if not d:
                continue
            n = d.get("n", 0)
            ss = d.get("SAME_STORY", 0)
            ev = d.get("EVOLVED", 0)
            di = d.get("DISTINCT", 0)
            pos_rate = (ss + ev) / n * 100 if n else 0
            lines.append(f"| {b} | {n} | {ss} | {ev} | {di} | {pos_rate:.1f}% |")
        lines.append("")

    # Threshold sweep
    if "sweep" in report:
        lines.append("## 5. Threshold Sweep\n")
        lines.append("| Threshold | Precision | Recall | F1 |")
        lines.append("|-----------|-----------|--------|-----|")
        for entry in report["sweep"]:
            t = entry.get("threshold", 0)
            p = entry.get("precision")
            r = entry.get("recall")
            f = entry.get("f1")
            lines.append(f"| {t:.2f} | {(p or 0):.4f} | {(r or 0):.4f} | {(f or 0):.4f} |")
        lines.append("")

    # Reliability diagram data
    if report.get("reliability"):
        lines.append("## 6. Calibration (Reliability Diagram Data)\n")
        lines.append("| Bin | N | Mean Score | Empirical Rate | Gap |")
        lines.append("|-----|---|-----------|----------------|-----|")
        for b in report["reliability"]:
            lines.append(
                f"| [{b['lo']:.1f}, {b['hi']:.1f}] | {b['n']} | "
                f"{b['mean_score']:.4f} | {b['empirical_rate']:.4f} | {b['gap']:.4f} |"
            )
        lines.append("")

    # Warning
    lines.append(f"> **WARNING: {report.get('warning') or WARNING}**\n")

    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory live report; publication requires the frozen human-dataset CLI")
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUT_MD)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_md.exists() or args.output_json.resolve() == args.output_md.resolve():
        parser.error("Use two distinct new output paths; historical reports are never overwritten")
    print("Loading live votes for exploratory diagnostics (not publication gold)...")
    rows, stats = await load_all_data()
    report = evaluate(rows, n_boot=1000, seed=1729)
    report["inter_annotator"] = compute_inter_annotator_agreement(rows)
    report["weak_signal_agreement"] = {origin: compute_inter_annotator_agreement(rows, origin) for origin in ("automatic", "legacy", "test")}
    for path in (args.output_json, args.output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as file:
        json.dump(report, file, sort_keys=True, indent=2, allow_nan=False)
    with args.output_md.open("x", encoding="utf-8") as file:
        file.write(generate_md_report(report, stats))
    print(f"Human consensus pairs={report['n_pairs']}; authenticated human votes={report['n_votes']}; all-origin votes={report['n_votes_all_origins']}")
    print(f"WARNING: {report['warning']}")
    print(f"Exploratory reports: {args.output_json}, {args.output_md}")


if __name__ == "__main__":
    asyncio.run(main())
