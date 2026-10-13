"""Compatibility-named cosine baseline against annotated claim-pair labels.

This is a score-threshold diagnostic, not the deployed typed/temporal parent
selector or evidence of copying. Empty or invalid gold is refused.
"""

import argparse
import datetime
import json
import math
import os
from typing import Any

import pandas as pd


def evaluate_mutation(
    pairs_path: str,
    threshold: float,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> dict[str, Any]:
    """Load pairs, calculate cosine similarity, and compute classification metrics."""
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Cosine threshold must be finite and between -1 and 1")

    df = pd.read_csv(pairs_path)

    # Clean and filter non-empty rows with valid claims
    if "claim_a" in df.columns and "claim_b" in df.columns:
        df = df.dropna(subset=["claim_a", "claim_b"]).copy()
    else:
        df = pd.DataFrame()

    n_pairs = len(df)

    label_distribution = {
        "EVOLVED_FROM": 0,
        "SIMILAR_TO": 0,
        "UNRELATED": 0,
        "DUPLICATE": 0,
    }

    if n_pairs == 0 or "label" not in df.columns:
        raise ValueError("Nonempty annotated claim_a, claim_b and label fields are required; this experiment is unperformed")

    # Normalize labels
    df["label_norm"] = df["label"].astype(str).str.strip().str.upper()
    if not df["label_norm"].isin(label_distribution).all():
        raise ValueError("Unknown or missing pair labels; use independently reviewed annotations")
    for lbl in label_distribution:
        label_distribution[lbl] = int((df["label_norm"] == lbl).sum())

    # Load SentenceTransformer model
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model '{model_name}' for mutation evaluation...")
    model = SentenceTransformer(model_name)

    claims_a = df["claim_a"].astype(str).tolist()
    claims_b = df["claim_b"].astype(str).tolist()

    print(f"Encoding {n_pairs} claim pairs...")
    embeddings_a = model.encode(claims_a, convert_to_tensor=True, normalize_embeddings=True)
    embeddings_b = model.encode(claims_b, convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarity for each corresponding pair
    import torch
    cos_scores = torch.sum(embeddings_a * embeddings_b, dim=-1).cpu().tolist()

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for score, (_, row) in zip(cos_scores, df.iterrows()):
        truth = 1 if row["label_norm"] == "EVOLVED_FROM" else 0
        pred = 1 if score >= threshold else 0

        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 1 and pred == 0:
            fn += 1
        elif truth == 0 and pred == 0:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "task": "cosine_threshold_baseline_against_pair_annotation",
        "model": model_name,
        "publication_ready": False,
        "limitation": "This does not evaluate typed spans, temporal parent selection, observed propagation or truth. Independently validate annotation provenance and use frozen event-disjoint splits.",
        "threshold": threshold,
        "n_pairs": n_pairs,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "label_distribution": label_distribution,
        "eval_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mutation detection on annotated claim pairs.")
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to CSV file containing annotated claim pairs",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write output JSON results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for EVOLVED_FROM classification (default: 0.85)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name (default: deployed all-mpnet-base-v2)",
    )

    args = parser.parse_args()
    if os.path.exists(args.output):
        parser.error("Output already exists; choose a new path to preserve historical results")

    results = evaluate_mutation(
        pairs_path=args.pairs,
        threshold=args.threshold,
        model_name=args.model,
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "x", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("MUTATION DETECTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Pairs Evaluated:     {results['n_pairs']}")
    print(f"Similarity Threshold:{results['threshold']}")
    print(f"Precision:           {results['precision']:.4f}")
    print(f"Recall:              {results['recall']:.4f}")
    print(f"F1 Score:            {results['f1']:.4f}")
    print(f"Confusion Matrix:    {results['confusion_matrix']}")
    print(f"Label Distribution:  {results['label_distribution']}")
    print(f"Saved to:            {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
