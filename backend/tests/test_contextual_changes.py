"""Unit-normalized lexical changes and conservative contextual decisions."""
from datetime import UTC, datetime, timedelta

import pytest

from app.services.evolution.quantities import extract_quantities
from app.services.evolution.text_changes import analyze_text_change


@pytest.mark.parametrize(("older", "newer"), [
    ("The road measures 1 km.", "The road measures 1000 m."),
    ("The parcel weighs 1000 grams.", "The parcel weighs 1 kilogram."),
    ("The council spent $1 million.", "The council spent $1000000."),
])
def test_equivalent_units_do_not_fabricate_numeric_mutation(older: str, newer: str) -> None:
    result = analyze_text_change(older, newer)
    assert result["quantity_equivalent"] is True
    assert result["relation_category"] == "EQUIVALENT_QUANTITY"
    assert result["meaningful_change"] is False
    for quantities, source in ((result["quantities"]["older"], older), (result["quantities"]["newer"], newer)):
        for quantity in quantities:
            assert source[quantity["start"]:quantity["end"]] == quantity["text"]


def test_unit_and_entity_context_cannot_be_discarded() -> None:
    assert "NUMERIC_DRIFT" in analyze_text_change("The road measures 1 km.", "The road measures 1 m.")["mutation_types"]
    assert analyze_text_change("Alice carried 1 kg and Bob 2 kg.", "Alice carried 2 kg and Bob 1 kg.")["meaningful_change"] is True
    assert extract_quantities("Observed on 2024-01-01 at 12:30.") == []


def test_changed_number_needs_context_not_a_truth_verdict() -> None:
    result = analyze_text_change("The agency reported 10 cases.", "The agency reported 12 cases.")
    assert result["relation_category"] == "CONTEXT_REQUIRED"
    assert result["observed_propagation"] is False


def test_explicit_update_and_unverified_reference_are_disclosed() -> None:
    now = datetime.now(UTC)
    result = analyze_text_change(
        "The agency reported 10 cases.", "The updated count is 12 cases.",
        older_timestamp=now, newer_timestamp=now + timedelta(hours=1),
        provenance_url="https://example.org/revision",
    )
    assert result["relation_category"] == "REPORTED_UPDATE"
    assert result["provenance_basis"] == "user_supplied_reference_unverified"
    assert result["context_signals"][0]["basis"] == "newer_text"


@pytest.mark.parametrize("note", ["This was not corrected.", "No correction was issued."])
def test_negated_correction_cue_is_not_promoted(note: str) -> None:
    result = analyze_text_change("The count is 10.", "The count is 12.", context_note=note)
    assert result["relation_category"] == "CONTEXT_REQUIRED"


def test_reversed_time_never_confirms_update() -> None:
    now = datetime.now(UTC)
    result = analyze_text_change("The count is 10.", "The updated count is 12.",
        older_timestamp=now, newer_timestamp=now - timedelta(hours=1))
    assert result["relation_category"] == "CONTEXT_REQUIRED"
