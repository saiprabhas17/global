"""Evaluate sentence-level claim detection on appropriate human targets.

LIAR veracity labels cannot label claim presence and are deliberately refused.
ClaimBuster's binary human check-worthiness target is evaluated separately; an
extractor's nonempty output is only a proxy classifier for that task.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))


def get_ground_truth(dataset: str, row: Any) -> int:
    if dataset == "liar":
        raise ValueError("LIAR labels veracity, not claim presence/check-worthiness. False factual assertions are still claims; use human claim_present labels or ClaimBuster.")
    key = "cfs_score" if dataset == "claimbuster" else "claim_present"
    value = str(row.get(key, "")).strip()
    if value not in {"0", "1", "0.0", "1.0"}:
        raise ValueError(f"{key} must be a binary human target (0 or 1), not a model score or missing value")
    return int(float(value))


def load_examples(dataset: str, path: Path, sample_size: int, seed: int) -> tuple[list[dict[str, Any]], str]:
    if dataset not in {"human", "claimbuster", "liar"}:
        raise ValueError("dataset must be human or claimbuster")
    if dataset == "liar":
        get_ground_truth(dataset, {})
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for number, row in enumerate(csv.DictReader(raw.decode("utf-8-sig").splitlines()), 2):
        text = row.get("text", "").strip()
        if not text:
            raise ValueError(f"CSV line {number}: text must not be blank")
        rows.append({"source_line": number, "text": text, "gold": get_ground_truth(dataset, row)})
    if not rows:
        raise ValueError("Dataset contains no examples")
    if sample_size < 0:
        raise ValueError("sample-size must be nonnegative; 0 evaluates all rows")
    if 0 < sample_size < len(rows):
        rows = [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), sample_size))]
    if {row["gold"] for row in rows} != {0, 1}:
        raise ValueError("Selected data must contain both positive and negative human targets")
    return rows, hashlib.sha256(raw).hexdigest()


async def evaluate(dataset: str, data_path: str, sample_size: int, seed: int = 1729) -> dict[str, Any]:
    examples, digest = load_examples(dataset, Path(data_path), sample_size, seed)
    from app.services.nlp.llm_client import HuggingFaceLLMClient

    client = HuggingFaceLLMClient()
    predictions = []
    for row in examples:
        extracted = await client.extract_claims(row["text"])
        predictions.append({**row, "predicted": int(bool(extracted))})
    tp = sum(r["gold"] == 1 and r["predicted"] == 1 for r in predictions)
    fp = sum(r["gold"] == 0 and r["predicted"] == 1 for r in predictions)
    fn = sum(r["gold"] == 1 and r["predicted"] == 0 for r in predictions)
    tn = sum(r["gold"] == 0 and r["predicted"] == 0 for r in predictions)
    precision, recall = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    return {
        "protocol_version": "sentence-claim-detection-v2", "dataset": dataset,
        "task": "check_worthiness_proxy" if dataset == "claimbuster" else "sentence_claim_presence",
        "dataset_sha256": digest, "sample_size": len(predictions), "seed": seed,
        "precision": precision, "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "model_used": client.MODEL_ID, "eval_timestamp": datetime.now(UTC).isoformat(),
        "publication_ready": False, "predictions": predictions,
        "limitation": "Nonempty extractor output is a sentence classifier. This does not evaluate atomic extraction fidelity, span accuracy or truth. ClaimBuster targets check-worthiness, a distinct task from factual-claim presence. Verify annotation provenance and frozen splits before scientific use.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["human", "claimbuster", "liar"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new path to preserve prior results")
        results = asyncio.run(evaluate(args.dataset, args.data_path, args.sample_size, args.seed))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"{results['task']}: n={results['sample_size']}, F1={results['f1']:.4f}; {args.output}")
    except (ValueError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
