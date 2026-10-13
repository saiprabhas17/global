"""Leakage-checked, train-fit/dev-select/test-evaluate research protocol.

No database, downloads, or model re-embedding. Stored cosine is a score;
sigmoid calibration is fit on train only. Logistic ablations use train-only
out-of-fold, group-disjoint sigmoid calibration. Test never selects a model,
feature vocabulary, imputation state, calibrator, or decision threshold.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from app.services.eval.annotation import MUTATION_TYPES
from app.services.eval.dataset import (
    POSITIVE_LABELS,
    SPLITS,
    DatasetError,
    canonical_json,
    records_bytes,
    validate_dataset,
)
from app.services.eval.features import lexical_jaccard
from app.services.eval.metrics import (
    auroc,
    average_precision,
    best_f1_threshold,
    brier_score,
    expected_calibration_error,
    precision_recall_curve,
    reliability_bins,
)

PROTOCOL_VERSION = "gmee-experiment-v1"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 1729
    bootstrap: int = 1000
    logistic_c: float = 1.0
    calibration_c: float = 1_000_000.0
    calibration_max_folds: int = 3

    def validate(self) -> None:
        if not 20 <= self.bootstrap <= 10000:
            raise DatasetError("bootstrap must be 20–10000 (default 1000); fewer resamples do not support the reported percentile intervals")
        if not 0 <= self.seed < 2**32:
            raise DatasetError("seed must be an integer in [0, 2**32)")
        if not math.isfinite(self.logistic_c) or self.logistic_c <= 0 or not math.isfinite(self.calibration_c) or self.calibration_c <= 0 or self.calibration_max_folds < 2:
            raise DatasetError("Regularization must be positive/finite and grouped calibration needs at least two folds")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output = {}
    for key, value in pairs:
        if key in output:
            raise DatasetError(f"Duplicate JSON field {key!r}; do not hide conflicting provenance in repeated keys")
        output[key] = value
    return output


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        data = path.read_bytes()
        lines = data.decode("utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Cannot read UTF-8 JSONL dataset {path}: {exc}") from exc
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_unique_object)
        except (ValueError, TypeError) as exc:
            raise DatasetError(f"Dataset line {number}: {exc}") from exc
        if not isinstance(record, dict):
            raise DatasetError(f"Dataset line {number}: each line must be one versioned record object")
        records.append(record)
    return records, hashlib.sha256(data).hexdigest()


def _hash_state(state: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(state).encode()).hexdigest()


def environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scikit-learn", "scipy", "sqlalchemy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def source_hashes() -> dict[str, str]:
    backend = Path(__file__).resolve().parents[3]
    paths = [f"app/services/eval/{name}.py" for name in ("research", "dataset", "features", "provenance", "metrics", "artifacts")]
    paths += ["app/services/verdict/mutation_diff.py", "scripts/run_research_experiment.py", "scripts/export_research_dataset.py"]
    return {name: hashlib.sha256((backend / name).read_bytes()).hexdigest() for name in paths if (backend / name).is_file()}


def binary_metrics(scores: list[float], labels: list[bool], threshold: float, calibrated: bool) -> dict[str, Any]:
    predicted = [score >= threshold for score in scores]
    tp = sum(p and y for p, y in zip(predicted, labels))
    fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels))
    tn = sum(not p and not y for p, y in zip(predicted, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    both = 0 < sum(labels) < len(labels)
    return {
        "n": len(labels), "positive": sum(labels), "negative": len(labels) - sum(labels),
        "precision": precision, "recall": recall, "f1": f1,
        "auprc": average_precision(scores, labels) if both else None,
        "auprc_definition": "non-interpolated average precision",
        "auroc": auroc(scores, labels) if both else None,
        "brier": brier_score(scores, labels) if calibrated else None,
        "ece": expected_calibration_error(scores, labels) if calibrated else None,
        "calibration_status": "train-only sigmoid calibrated probability" if calibrated else "not applicable: raw score is not a probability",
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _sigmoid_state(model: Any) -> dict[str, Any]:
    return {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "classes": model.classes_.tolist()}


def _logistic_pipeline(config: ExperimentConfig) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=config.logistic_c, solver="lbfgs", max_iter=1000, random_state=config.seed),
    )


def calibration_folds(y: np.ndarray, groups: np.ndarray, config: ExperimentConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    n_folds = min(config.calibration_max_folds, min(Counter(y.tolist()).values()), len(set(groups)))
    if n_folds < 2:
        raise DatasetError("Train cannot support group-disjoint calibration; collect both classes across at least two independent training event groups")
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    folds = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    for fit, held in folds:
        if len(set(y[fit])) < 2 or set(groups[fit]) & set(groups[held]):
            raise DatasetError("Training event groups cannot support train-only calibration: an out-of-fold training partition lacks a class. Collect positive and negative human gold across more independent training events; do not borrow dev/test labels")
    return folds


def fit_group_calibrated_logistic(
    x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]], config: ExperimentConfig,
) -> tuple[list[float], dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    out_of_fold = np.full(len(y_train), np.nan)
    for fit, held in folds:
        model = _logistic_pipeline(config)
        model.fit(x_train[fit], y_train[fit])
        out_of_fold[held] = model.decision_function(x_train[held])
    if not np.isfinite(out_of_fold).all():
        raise DatasetError("Grouped calibration did not produce a finite held-out training score for every row")
    calibrator = LogisticRegression(C=config.calibration_c, solver="lbfgs", max_iter=1000, random_state=config.seed)
    calibrator.fit(out_of_fold.reshape(-1, 1), y_train)
    model = _logistic_pipeline(config)
    model.fit(x_train, y_train)
    decisions = model.decision_function(x_all)
    scores = calibrator.predict_proba(decisions.reshape(-1, 1))[:, 1].tolist()
    imputer, scaler, classifier = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    state = {
        "imputer": {"strategy": "constant", "fill_value": 0.0, "statistics": imputer.statistics_.tolist(), "missing_indicator_columns": imputer.indicator_.features_.tolist()},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "classifier": _sigmoid_state(classifier), "calibrator": _sigmoid_state(calibrator),
        "calibration": "sigmoid on group-disjoint out-of-fold train decision scores; final base model refit on all train",
        "out_of_fold_decisions": out_of_fold.tolist(),
    }
    return scores, state


def fit_methods(records: list[dict[str, Any]], validation: dict[str, Any], config: ExperimentConfig) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_indices = [i for i, r in enumerate(records) if r["split"] == "train"]
    train = [records[i] for i in train_indices]
    y_train = np.asarray([int(r["gold_label"] in POSITIVE_LABELS) for r in train])
    groups = np.asarray([validation["group_by_pair"][r["pair_id"]] for r in train])
    folds = calibration_folds(y_train, groups, config)
    train_texts = sorted({r[key] for r in train for key in ("text_a", "text_b")})
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, norm="l2")
    try:
        vectorizer.fit(train_texts)
    except ValueError as exc:
        raise DatasetError(f"Train-only TF-IDF cannot fit the claim texts: {exc}; collect substantive training claims") from exc
    a, b = (vectorizer.transform([r[key] for r in records]) for key in ("text_a", "text_b"))
    tfidf = np.asarray(a.multiply(b).sum(axis=1)).ravel().tolist()
    cosine = [float(r["cosine_score"]) for r in records]
    cosine_train = np.asarray([cosine[i] for i in train_indices]).reshape(-1, 1)
    calibrator = LogisticRegression(C=config.calibration_c, solver="lbfgs", max_iter=1000, random_state=config.seed)
    calibrator.fit(cosine_train, y_train)
    calibrated_cosine = calibrator.predict_proba(np.asarray(cosine).reshape(-1, 1))[:, 1].tolist()
    states = {
        "tfidf": {"vocabulary": vectorizer.vocabulary_, "idf": vectorizer.idf_.tolist(), "unique_train_documents": len(train_texts), "analyzer": "char_wb", "ngram_range": [3, 5], "sublinear_tf": True, "norm": "l2"},
        "cosine": {"source": "stored cosine_score; no fitting; historical embedding model provenance supplied by dataset"},
        "cosine_calibrated": {"calibrator": _sigmoid_state(calibrator), "calibration": "train-only sigmoid mapping of fixed stored cosine"},
    }
    methods = {
        "tfidf": {"scores": tfidf, "score_kind": "score_not_probability", "features": ["train-fitted char TF-IDF cosine"]},
        "cosine": {"scores": cosine, "score_kind": "score_not_probability", "features": ["cosine_score"]},
        "cosine_calibrated": {"scores": calibrated_cosine, "score_kind": "calibrated_probability", "features": ["cosine_score"]},
    }
    names = ["cosine_score", "lexical_jaccard"]
    warnings = []
    entity_train = [r.get("features", {}).get("entity_jaccard") for r in train]
    if any(v is not None for v in entity_train):
        names.append("entity_jaccard")
    else:
        warnings.append("Entity ablations omitted: no entity_jaccard evidence in train. Dev/test feature availability never enables a train-absent feature.")
    feature_rows = []
    for record in records:
        features = record.get("features", {})
        lexical = features.get("lexical_jaccard")
        if lexical is None:
            lexical = lexical_jaccard(record["text_a"], record["text_b"])
        values = [float(record["cosine_score"]), float(lexical)]
        if "entity_jaccard" in names:
            value = features.get("entity_jaccard")
            values.append(float(value) if value is not None else np.nan)
        feature_rows.append(values)
    x = np.asarray(feature_rows, dtype=float)
    ablations = {
        "logistic_all": names,
        "logistic_without_cosine": [n for n in names if n != "cosine_score"],
        "logistic_without_lexical": [n for n in names if n != "lexical_jaccard"],
    }
    if "entity_jaccard" in names:
        ablations["logistic_without_entity"] = [n for n in names if n != "entity_jaccard"]
    fold_evidence = [{"fit_pair_ids": [train[i]["pair_id"] for i in fit], "calibration_pair_ids": [train[i]["pair_id"] for i in held]} for fit, held in folds]
    for name, feature_names in ablations.items():
        columns = [names.index(n) for n in feature_names]
        values, state = fit_group_calibrated_logistic(x[train_indices][:, columns], y_train, x[:, columns], folds, config)
        state.update({"features": feature_names, "folds": fold_evidence})
        states[name] = state
        methods[name] = {"scores": values, "score_kind": "calibrated_probability", "features": feature_names}
    for name, method in methods.items():
        if any(not math.isfinite(score) for score in method["scores"]):
            raise DatasetError(f"{name} produced a non-finite score; no result written")
        method["fitted_state_sha256"] = _hash_state(states[name])
    return methods, states, warnings


def _f1(scores: list[float], labels: list[bool], threshold: float) -> float:
    tp = sum(s >= threshold and y for s, y in zip(scores, labels))
    fp = sum(s >= threshold and not y for s, y in zip(scores, labels))
    fn = sum(s < threshold and y for s, y in zip(scores, labels))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def grouped_bootstrap_differences(
    records: list[dict[str, Any]], scores: dict[str, list[float]], thresholds: dict[str, float],
    group_by_pair: dict[str, str], n_boot: int, seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[group_by_pair[record["pair_id"]]].append(i)
    labels = [r["gold_label"] in POSITIVE_LABELS for r in records]
    names = sorted(scores)
    comparison_names = list(combinations(names, 2))
    observed = {name: {"f1": _f1(values, labels, thresholds[name]), "auroc": auroc(values, labels), "auprc": average_precision(values, labels)} for name, values in scores.items()}
    differences: dict[tuple[str, str], dict[str, list[float]]] = {pair: {metric: [] for metric in ("f1", "auroc", "auprc")} for pair in comparison_names}
    rng = random.Random(seed)
    group_ids = sorted(groups)
    valid = 0
    for _ in range(n_boot):
        indices = [i for _ in group_ids for i in groups[rng.choice(group_ids)]]
        sampled_labels = [labels[i] for i in indices]
        if not any(sampled_labels) or all(sampled_labels):
            continue
        valid += 1
        metrics = {}
        for name in names:
            values = [scores[name][i] for i in indices]
            metrics[name] = {"f1": _f1(values, sampled_labels, thresholds[name]), "auroc": auroc(values, sampled_labels), "auprc": average_precision(values, sampled_labels)}
        for a, b in comparison_names:
            for metric in differences[(a, b)]:
                a_value, b_value = metrics[a][metric], metrics[b][metric]
                if a_value is not None and b_value is not None:
                    differences[(a, b)][metric].append(a_value - b_value)
    comparisons = []
    for (a, b), sampled_metrics in differences.items():
        entry: dict[str, Any] = {"a": a, "b": b, "difference": "a-minus-b", "metrics": {}}
        for metric, sampled_values in sampled_metrics.items():
            interval = None
            if len(sampled_values) >= 10:
                low, high = np.quantile(sampled_values, [0.025, 0.975], method="linear")
                interval = {"low": float(low), "high": float(high), "mean": float(np.mean(sampled_values))}
            a_observed, b_observed = observed[a][metric], observed[b][metric]
            entry["metrics"][metric] = {"observed": a_observed - b_observed if a_observed is not None and b_observed is not None else None, "ci95": interval}
        comparisons.append(entry)
    return {
        "unit": "connected event/claim/article/identical-text component", "paired": True,
        "partition": "test", "seed": seed, "attempted_resamples": n_boot,
        "valid_resamples": valid, "skipped_single_class_resamples": n_boot - valid,
        "independent_groups": len(groups), "interval": "percentile 95%, linear quantiles",
        "thresholds_frozen_from": "dev", "comparisons": comparisons,
        "warning": "Cluster bootstrap is descriptive, not an independence or publication-validity guarantee; single-class resamples are omitted and no multiple-comparison claim is made" if valid >= 10 else "Insufficient valid grouped resamples; intervals are null, not fabricated",
    }


def mutation_evaluation(records: list[dict[str, Any]], predicted: dict[str, list[str]]) -> dict[str, Any]:
    test = [r for r in records if r["split"] == "test" and r.get("gold_mutation_types") is not None]
    if not test:
        return {"status": "unavailable", "reason": "No independently agreed human mutation-type gold in untouched test; binary relatedness is not mutation detection"}
    names = sorted(set(MUTATION_TYPES) | {t for r in test for t in r["gold_mutation_types"]} | {t for r in test for t in predicted[r["pair_id"]]})
    per_type = {}
    for name in names:
        gold = [name in r["gold_mutation_types"] for r in test]
        guesses = [name in predicted[r["pair_id"]] for r in test]
        tp = sum(g and p for g, p in zip(gold, guesses))
        fp = sum(not g and p for g, p in zip(gold, guesses))
        fn = sum(g and not p for g, p in zip(gold, guesses))
        per_type[name] = {"support": sum(gold), "predicted": sum(guesses), "tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    evaluated = [v["f1"] for v in per_type.values() if v["f1"] is not None]
    errors = []
    for record in test:
        gold_set, guess = set(record["gold_mutation_types"]), set(predicted[record["pair_id"]])
        if gold_set != guess:
            errors.append({"pair_id": record["pair_id"], "event_id": record["event_id"], "gold": sorted(gold_set), "predicted": sorted(guess), "false_positive_types": sorted(guess - gold_set), "false_negative_types": sorted(gold_set - guess)})
    return {
        "status": "evaluated", "task": "separate multilabel mutation detection", "partition": "test",
        "analyzer": "app.services.verdict.mutation_diff.diff_versions", "orientation": "text_a -> text_b (pair order, not inferred chronology)",
        "n_gold_pairs": len(test), "macro_f1": sum(evaluated) / len(evaluated) if evaluated else None,
        "macro_f1_labels": [name for name, v in per_type.items() if v["f1"] is not None],
        "macro_f1_definition": "mean per-type F1 across types with gold or prediction support; zero-support/zero-prediction types excluded",
        "subset_accuracy": (len(test) - len(errors)) / len(test), "per_type": per_type, "errors": errors,
        "unsupported_gold_types": sorted({t for r in test for t in r["gold_mutation_types"]} - set(MUTATION_TYPES)),
        "warning": "Mutation predictions come from the existing text analyzer, never the binary similarity classifier or human gold. Partial human mutation coverage may be selection-biased.",
    }


def _errors(records: list[dict[str, Any]], scores: list[float], threshold: float) -> dict[str, Any]:
    false_positive, false_negative = [], []
    by_label: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "errors": 0})
    for record, score in zip(records, scores):
        gold = record["gold_label"] in POSITIVE_LABELS
        predicted = score >= threshold
        entry = by_label[record["gold_label"]]
        entry["n"] += 1
        entry["errors"] += int(gold != predicted)
        if predicted and not gold:
            false_positive.append(record["pair_id"])
        if gold and not predicted:
            false_negative.append(record["pair_id"])
    return {"false_positive_pair_ids": false_positive, "false_negative_pair_ids": false_negative, "by_gold_label": dict(sorted(by_label.items()))}


def run_experiment(
    records: list[dict[str, Any]], config: ExperimentConfig | None = None,
    dataset_sha256: str | None = None,
    mutation_analyzer: Callable[[str, str], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = config or ExperimentConfig()
    config.validate()
    validation = validate_dataset(records, require_support=True)
    records = sorted(records, key=lambda r: r["pair_id"])
    methods, states, warnings = fit_methods(records, validation, config)
    split_indices = {split: [i for i, r in enumerate(records) if r["split"] == split] for split in SPLITS}
    predicted_mutations = {}
    if any(r.get("gold_mutation_types") is not None for r in records):
        if mutation_analyzer is None:
            from app.services.verdict.mutation_diff import diff_versions

            mutation_analyzer = diff_versions
        for record in records:
            result = mutation_analyzer(record["text_a"], record["text_b"])
            types = result.get("mutation_types")
            if not isinstance(types, list) or any(not isinstance(t, str) for t in types):
                raise DatasetError(f"Mutation analyzer returned invalid types for {record['pair_id']}; no result written")
            predicted_mutations[record["pair_id"]] = sorted(set(types))
    normalized_sha = hashlib.sha256(records_bytes(records)).hexdigest()
    report = {
        "protocol_version": PROTOCOL_VERSION, "status": "held_out_human_evaluation", "publication_ready": False,
        "task": "binary relatedness: SAME_STORY/EVOLVED positive, DISTINCT negative; not misinformation truth verification or mutation detection",
        "dataset_version": validation["dataset_version"], "dataset_sha256": dataset_sha256 or normalized_sha,
        "normalized_records_sha256": normalized_sha,
        "config": {**asdict(config), "tfidf": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 1, "sublinear_tf": True, "train_unique_texts_only": True}, "threshold_selection": "dev best F1; highest threshold breaks ties", "calibration": "sigmoid; train only; logistic calibration uses group-disjoint out-of-fold train scores", "missing_features": "lexical absent -> deterministic text Jaccard; entity absent -> train-fit zero imputation plus missing indicator; train-absent entity feature omitted"},
        "versions": environment_versions(), "source_sha256": source_hashes(),
        "validation": validation, "methods": {}, "test_error_analysis": {},
        "mutation_evaluation": mutation_evaluation(records, predicted_mutations),
        "warnings": [
            "No minimum sample count, human agreement level, or bootstrap interval alone guarantees publication validity. The sample floor only prevents undefined/unsupported computations.",
            "Stratified sampling and active-learning train selection can bias prevalence-sensitive precision/AUPRC/calibration; review the target population and frozen sampling protocol.",
            "Test is evaluated only after all fitting and dev threshold selection; no method is selected by test performance. Repeated reuse of test across external runs is not prevented by this CLI.",
            "Stored cosine is not a probability and its original embedding model/version must be documented in dataset provenance; no model/version is inferred from historical vectors.",
            "Text, annotator UUIDs and notes may require licensing/consent before distributing dataset or prediction artifacts.",
            *warnings,
        ],
    }
    thresholds, test_scores = {}, {}
    predictions = []
    for name, method in sorted(methods.items()):
        dev_scores = [method["scores"][i] for i in split_indices["dev"]]
        dev_labels = [records[i]["gold_label"] in POSITIVE_LABELS for i in split_indices["dev"]]
        selected = best_f1_threshold(dev_scores, dev_labels)
        threshold = selected["threshold"]
        if threshold is None:
            raise DatasetError(f"Cannot select {name} threshold on dev; both gold classes and finite scores are required")
        thresholds[name] = float(threshold)
        entry = {
            "features": method["features"], "score_kind": method["score_kind"],
            "fitted_on": "train", "fitted_state_sha256": method["fitted_state_sha256"],
            "threshold": float(threshold), "threshold_selected_on": "dev",
            "dev_selection": selected,
        }
        calibrated = method["score_kind"] == "calibrated_probability"
        for split in ("dev", "test"):
            indices = split_indices[split]
            values = [method["scores"][i] for i in indices]
            labels = [records[i]["gold_label"] in POSITIVE_LABELS for i in indices]
            entry[split] = binary_metrics(values, labels, float(threshold), calibrated)
            entry[f"{split}_pr_curve"] = precision_recall_curve(values, labels)
            if split == "test":
                test_scores[name] = values
                entry["test_reliability"] = reliability_bins(values, labels) if calibrated else []
                report["test_error_analysis"][name] = _errors([records[i] for i in indices], values, float(threshold))
        report["methods"][name] = entry
        for i, record in enumerate(records):
            score = method["scores"][i]
            gold = record["gold_label"] in POSITIVE_LABELS
            prediction = {
                key: record[key] for key in ("dataset_version", "pair_id", "event_id", "split", "claim_a_id", "claim_b_id", "article_a_id", "article_b_id", "text_a", "text_b", "gold_label")
            }
            prediction.update({
                "model": name, "score": score, "score_kind": method["score_kind"],
                "probability": score if calibrated else None, "dev_threshold": float(threshold),
                "gold_relatedness": gold, "predicted_relatedness": score >= threshold,
                "correct": (score >= threshold) == gold,
                "gold_mutation_types": record.get("gold_mutation_types"),
                "predicted_mutation_types": predicted_mutations.get(record["pair_id"]),
                "human_user_ids": sorted({v["annotator_user_id"] for v in record["human_labels"]}),
            })
            predictions.append(prediction)
    report["grouped_bootstrap"] = grouped_bootstrap_differences(
        [records[i] for i in split_indices["test"]], test_scores, thresholds,
        validation["group_by_pair"], config.bootstrap, config.seed,
    )
    report["training_pair_ids"] = [records[i]["pair_id"] for i in split_indices["train"]]
    report["dev_pair_ids"] = [records[i]["pair_id"] for i in split_indices["dev"]]
    report["test_pair_ids"] = [records[i]["pair_id"] for i in split_indices["test"]]
    return report, sorted(predictions, key=lambda p: (p["pair_id"], p["model"])), states
