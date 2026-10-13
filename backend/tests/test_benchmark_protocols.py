"""Legacy benchmark runners refuse invalid scientific tasks before model loading."""
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def runner(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "evaluation" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_veracity_is_never_an_extraction_target() -> None:
    module = runner("eval_claim_extraction")
    for truth_label in ("true", "false", "half-true"):
        with pytest.raises(ValueError, match="LIAR labels veracity"):
            module.get_ground_truth("liar", {"label": truth_label})
    assert module.get_ground_truth("human", {"claim_present": "1"}) == 1
    with pytest.raises(ValueError, match="binary human target"):
        module.get_ground_truth("human", {"claim_present": "0.8"})


def test_fever_requires_actual_frozen_passages_and_preserves_abstention() -> None:
    module = runner("eval_verdict_engine")
    claims = [{"id": 1, "claim": "The bridge opened.", "label": "SUPPORTS"}]
    with pytest.raises(ValueError, match="no frozen evidence record"):
        module.prepare_examples(claims, [], 0, 42)
    with pytest.raises(ValueError, match="actual text"):
        module.prepare_examples(claims, [{"id": 1, "passages": [{"page": "Bridge", "line": 1}]}], 0, 42)
    assert module.prepare_examples(claims, [{"id": 1, "passages": []}], 0, 42)[0]["passages"] == []


def test_empty_mutation_data_is_unperformed_not_zero_accuracy(tmp_path: Path) -> None:
    data = tmp_path / "pairs.csv"
    data.write_text("claim_a,claim_b,label\n")
    with pytest.raises(ValueError, match="experiment is unperformed"):
        runner("eval_mutation_detection").evaluate_mutation(str(data), 0.85)
