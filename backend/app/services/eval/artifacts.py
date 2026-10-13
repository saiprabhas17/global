"""JSON/Markdown/CSV and dependency-free SVG research artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from html import escape
from pathlib import Path
from typing import Any

from app.services.eval.dataset import DatasetError, canonical_json

COLORS = ("#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2", "#475569")


def line_svg(title: str, x_label: str, y_label: str, series: list[tuple[str, list[tuple[float, float]]]], x_range: tuple[float, float] = (0.0, 1.0)) -> str:
    width, height = 900, 600
    left, top, plot_width, plot_height = 80, 65, 590, 435
    lo, hi = x_range

    def xy(x: float, y: float) -> tuple[float, float]:
        return left + (x - lo) / (hi - lo) * plot_width, top + (1 - y) * plot_height

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">', f"<title>{escape(title)}</title>", '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="30" font-family="sans-serif" font-size="20">{escape(title)}</text>']
    for i in range(6):
        fraction = i / 5
        x, y = xy(lo + fraction * (hi - lo), fraction)
        lines.extend([
            f'<path d="M {left} {y:.2f} H {left + plot_width}" stroke="#e5e7eb"/>',
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{fraction:.1f}</text>',
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{lo + fraction * (hi - lo):.2f}</text>',
        ])
    lines.append(f'<path d="M {left} {top} V {top + plot_height} H {left + plot_width}" fill="none" stroke="#111827"/>')
    for i, (name, points) in enumerate(series):
        color = COLORS[i % len(COLORS)]
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(a, b) for a, b in points))
        lines.extend([
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>',
            f'<path d="M 700 {85 + i * 34} H 720" stroke="{color}" stroke-width="3"/>',
            f'<text x="726" y="{90 + i * 34}" font-family="sans-serif" font-size="11">{escape(name)}</text>',
        ])
    lines.extend([
        f'<text x="{left + plot_width / 2}" y="555" text-anchor="middle" font-family="sans-serif" font-size="16">{escape(x_label)}</text>',
        f'<text transform="translate(24 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="16">{escape(y_label)}</text>',
        '</svg>\n',
    ])
    return "\n".join(lines)


def _number(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (float, int)) else "N/A"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# GMEE frozen human-gold research evaluation", "",
        "> This report does not certify publication validity. No results are generated from weak pseudo-gold.", "",
        f"- Dataset version: `{report['dataset_version']}`",
        f"- Dataset SHA-256: `{report['dataset_sha256']}`",
        f"- Protocol: `{report['protocol_version']}`; seed: `{report['config']['seed']}`",
        f"- Task: {report['task']}", "",
        "## Protocol", "",
        "TF-IDF, imputation/scaling, logistic fitting and calibration use **train only**. Each method's decision threshold is selected on **dev only**. Test metrics use these frozen thresholds; test never selects a method. Logistic sigmoid calibration uses connected-group-disjoint out-of-fold train scores. Raw cosine/TF-IDF are scores, not probabilities.", "",
        "## Dataset support", "",
        "| Split | Pairs | Positive | Negative | Events | Connected groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, support in report["validation"]["support"].items():
        lines.append(f"| {split} | {support['pairs']} | {support['positive']} | {support['negative']} | {support['events']} | {support['independent_groups']} |")
    lines.extend(["", "## Untouched test results", "", "| Method | Score kind | Dev threshold | Precision | Recall | F1 | AUPRC (AP) | AUROC | Brier | ECE |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, method in report["methods"].items():
        metrics = method["test"]
        values = " | ".join(_number(metrics[key]) for key in ("precision", "recall", "f1", "auprc", "auroc", "brier", "ece"))
        lines.append(f"| {name} | {method['score_kind']} | {_number(method['threshold'])} | {values} |")
    lines.extend(["", "Brier/ECE and reliability curves are provided only for explicitly train-calibrated probabilities. AUPRC is non-interpolated average precision and depends on prevalence.", "", "## Paired connected-group bootstrap differences", "", "Differences are A minus B using frozen dev thresholds and identical test component resamples. Intervals are descriptive, not multiple-comparison-corrected significance claims.", "", "| A | B | Metric | Observed difference | 95% CI |", "|---|---|---|---:|---|"])
    bootstrap = report["grouped_bootstrap"]
    for comparison in bootstrap["comparisons"]:
        for metric, result in comparison["metrics"].items():
            ci = result["ci95"]
            interval = f"[{ci['low']:.4f}, {ci['high']:.4f}]" if ci else "Unavailable"
            lines.append(f"| {comparison['a']} | {comparison['b']} | {metric} | {_number(result['observed'])} | {interval} |")
    lines.extend(["", f"Groups: {bootstrap['independent_groups']}; valid resamples: {bootstrap['valid_resamples']}/{bootstrap['attempted_resamples']}; omitted single-class resamples: {bootstrap['skipped_single_class_resamples']}.", "", "## Separate mutation evaluation", ""])
    mutation = report["mutation_evaluation"]
    if mutation["status"] == "evaluated":
        lines.extend([f"Analyzer: `{mutation['analyzer']}` (text-derived predictions, not relatedness labels).", f"Human mutation-gold test pairs: {mutation['n_gold_pairs']}; macro-F1: {_number(mutation['macro_f1'])}; exact-set accuracy: {_number(mutation['subset_accuracy'])}.", "", "| Mutation type | Gold support | Predicted | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"])
        for name, metric in mutation["per_type"].items():
            lines.append(f"| {name} | {metric['support']} | {metric['predicted']} | {_number(metric['precision'])} | {_number(metric['recall'])} | {_number(metric['f1'])} |")
        lines.extend(["", mutation["macro_f1_definition"], "", mutation["warning"]])
    else:
        lines.append(mutation["reason"])
    lines.extend(["", "## Graphs", "", "![Test precision–recall](pr.svg)", "", "![Dev-only F1 threshold selection](f1.svg)", "", "![Test reliability of calibrated probabilities](reliability.svg)", "", "## Reproducibility", "", "`report.json` records versions, seeds, configuration, membership, fitted-state and source hashes. `model_states.json` records fitted vocabulary/IDF, scaling and logistic/calibrator parameters. `pair_predictions.json`/`.csv` include every pair/model prediction and IDs. `manifest.json` hashes all output artifacts. Error analysis is in `report.json`.", "", "## Limitations", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(["", "## Versions", "", "```json", json.dumps(report["versions"], sort_keys=True, indent=2), "```", ""])
    return "\n".join(lines)


def _csv_predictions(predictions: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    columns = ["dataset_version", "pair_id", "event_id", "split", "claim_a_id", "claim_b_id", "article_a_id", "article_b_id", "text_a", "text_b", "gold_label", "gold_relatedness", "model", "score", "score_kind", "probability", "dev_threshold", "predicted_relatedness", "correct", "gold_mutation_types", "predicted_mutation_types", "human_user_ids"]
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for prediction in predictions:
        row = {key: prediction.get(key) for key in columns}
        for key in ("gold_mutation_types", "predicted_mutation_types", "human_user_ids"):
            row[key] = canonical_json(row[key]) if row[key] is not None else ""
        # Spreadsheet formulas are not part of a claim's text; CSV keeps them inert.
        for key in ("text_a", "text_b", "event_id", "dataset_version", "pair_id"):
            value = row[key]
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                row[key] = "'" + value
        writer.writerow(row)
    return output.getvalue()


def write_artifacts(report: dict[str, Any], predictions: list[dict[str, Any]], states: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise DatasetError(f"Output directory {output_dir} is not empty; choose a new directory to preserve previous results")
    pr, f1, reliability = [], [], [("ideal calibration", [(0.0, 0.0), (1.0, 1.0)])]
    for name, method in report["methods"].items():
        pr.append((name, [(0.0, 1.0)] + [(p["recall"], p["precision"]) for p in method["test_pr_curve"]]))
        f1.append((name, sorted((p["threshold"], p["f1"]) for p in method["dev_pr_curve"])))
        if method["score_kind"] == "calibrated_probability":
            reliability.append((name, [(p["mean_score"], p["empirical_rate"]) for p in method["test_reliability"]]))
    content = {
        "pair_predictions.json": json.dumps(predictions, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "pair_predictions.csv": _csv_predictions(predictions),
        "model_states.json": json.dumps(states, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "pr.svg": line_svg("Untouched test precision–recall", "Recall", "Precision", pr),
        "f1.svg": line_svg("Dev-only threshold selection (not a test sweep)", "Decision threshold (raw score or calibrated probability)", "Dev F1", f1, (-1.0, 1.0)),
        "reliability.svg": line_svg("Untouched test: calibrated probabilities only", "Mean calibrated probability", "Empirical relatedness rate", reliability),
    }
    report["artifact_sha256"] = {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in sorted(content.items())}
    content["report.json"] = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    content["report.md"] = markdown_report(report)
    manifest = {
        "protocol_version": report["protocol_version"], "dataset_version": report["dataset_version"],
        "dataset_sha256": report["dataset_sha256"], "seed": report["config"]["seed"],
        "artifacts": {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in sorted(content.items())},
    }
    content["manifest.json"] = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in sorted(content.items()):
        with (output_dir / name).open("xb") as file:
            file.write(value.encode("utf-8"))
    return manifest
