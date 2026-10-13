"""FEVER supplied-evidence label evaluation; never use other claims as evidence.

Requires a separate JSONL evidence file, one {id, passages:[{page,line,text}]}
object per claim, and an explicit oracle/retrieved evidence mode. The evaluator
measures NLI aggregation on these frozen passages, not the official FEVER score
or GMEE's weighted corpus heuristic. An empty passage list is explicit abstention.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
CLASSES = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def compute_metrics(truths: list[str], predictions: list[str]) -> dict[str, Any]:
    if not truths or len(truths) != len(predictions) or any(v not in CLASSES for v in [*truths, *predictions]):
        raise ValueError("Nonempty, aligned valid FEVER labels are required")
    matrix = {label: dict.fromkeys(CLASSES, 0) for label in CLASSES}
    for truth, prediction in zip(truths, predictions):
        matrix[truth][prediction] += 1
    metrics = {}
    for label in CLASSES:
        tp = matrix[label][label]
        fp = sum(matrix[t][label] for t in CLASSES if t != label)
        fn = sum(matrix[label][p] for p in CLASSES if p != label)
        metrics[label] = {"support": sum(matrix[label].values()), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {"macro_f1": sum(m["f1"] for m in metrics.values()) / len(CLASSES), "per_class": metrics, "confusion_matrix": matrix}


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if not records or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"{path}: expected nonempty JSONL objects")
    return records, hashlib.sha256(raw).hexdigest()


def prepare_examples(claims: list[dict[str, Any]], evidence: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    indexed = {}
    for row in evidence:
        key = str(row.get("id", ""))
        if not key or key in indexed or not isinstance(row.get("passages"), list):
            raise ValueError("Evidence must contain one unique id and passages list per claim")
        passages = row["passages"]
        if len(passages) > 50:
            raise ValueError("Evidence is limited to 50 frozen passages per claim")
        for passage in passages:
            if not isinstance(passage, dict) or not isinstance(passage.get("text"), str) or not passage["text"].strip() or not isinstance(passage.get("page"), str) or not passage["page"].strip() or isinstance(passage.get("line"), bool) or not isinstance(passage.get("line"), int) or passage["line"] < 0:
                raise ValueError("Each evidence passage requires a Wikipedia page, nonnegative line, and actual text")
        indexed[key] = passages
    output, seen = [], set()
    for row in claims:
        key = str(row.get("id", ""))
        if not key or key in seen or row.get("label") not in CLASSES or not isinstance(row.get("claim"), str) or not row["claim"].strip():
            raise ValueError("Claims require unique ids, actual text and valid FEVER labels")
        seen.add(key)
        if key not in indexed:
            raise ValueError(f"Claim {key} has no frozen evidence record; retrieve Wikipedia passages or provide an explicit empty list. Other evaluation claims are never evidence.")
        output.append({"id": key, "claim": row["claim"], "gold": row["label"], "passages": indexed[key]})
    if not output or sample_size < 0:
        raise ValueError("Nonempty data and a nonnegative sample-size are required")
    if 0 < sample_size < len(output):
        output = [output[i] for i in sorted(random.Random(seed).sample(range(len(output)), sample_size))]
    return output


async def evaluate(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from app.services.verdict.engine import classify_stance

    predictions = []
    for row in examples:
        evidence = []
        for passage in row["passages"]:
            result = await classify_stance(row["claim"], passage["text"])
            evidence.append({**passage, "stance": result.stance})
        stances = {p["stance"] for p in evidence}
        supported, refuted = "entailment" in stances, "contradiction" in stances
        prediction = "SUPPORTS" if supported and not refuted else "REFUTES" if refuted and not supported else "NOT ENOUGH INFO"
        predictions.append({**row, "passages": evidence, "prediction": prediction, "mixed_evidence": supported and refuted})
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--evidence-mode", choices=["oracle", "retrieved"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output exists; choose a new path to preserve historical results")
        claims, claim_hash = read_jsonl(args.data)
        evidence, evidence_hash = read_jsonl(args.evidence)
        predictions = asyncio.run(evaluate(prepare_examples(claims, evidence, args.sample_size, args.seed)))
        result = {
            **compute_metrics([p["gold"] for p in predictions], [p["prediction"] for p in predictions]),
            "protocol_version": "fever-supplied-evidence-v2", "task": "supplied_evidence_label_accuracy_not_official_fever_score",
            "evidence_mode": args.evidence_mode, "dataset_sha256": claim_hash, "evidence_sha256": evidence_hash,
            "sample_size": len(predictions), "seed": args.seed, "publication_ready": False,
            "gold_support": dict(Counter(p["gold"] for p in predictions)), "predictions": predictions,
            "eval_timestamp": datetime.now(UTC).isoformat(),
            "limitation": "Oracle evidence is an upper-bound diagnostic, not retrieval performance. Retrieved evidence must be frozen without reading test labels. Mixed stance abstains. Official FEVER requires gold evidence-set matching, which this script does not compute; use the official scorer for that claim.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"Supplied-evidence ({args.evidence_mode}) n={len(predictions)}, macro-F1={result['macro_f1']:.4f}; {args.output}")
    except (ValueError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
