"""Deterministic typed mutation tests: no NER/embedding/model downloads."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.services.evolution.text_changes import ALGORITHM_VERSION, analyze_text_change


@pytest.mark.parametrize(
    ("older", "newer", "expected"),
    [
        ("The cost was $1,200.50.", "The cost was $1,800.75.", "NUMERIC_DRIFT"),
        (
            "Alice signed the agreement.",
            "Bob signed the agreement.",
            "ENTITY_SUBSTITUTION",
        ),
        (
            "The agency did not approve it.",
            "The agency did approve it.",
            "POLARITY_SHIFT",
        ),
        (
            "The agency reportedly approved it.",
            "The agency approved it.",
            "HEDGING_SHIFT",
        ),
        (
            "Alice said the bridge is safe.",
            "Bob said the bridge is safe.",
            "ATTRIBUTION_SHIFT",
        ),
        (
            "All residents were evacuated.",
            "Some residents were evacuated.",
            "SCOPE_SHIFT",
        ),
        ("At least 12 people arrived.", "At most 12 people arrived.", "SCOPE_SHIFT"),
        ("Rates rose by 3.5%.", "Rates rose by 4%.", "NUMERIC_DRIFT"),
        ("It can't reopen.", "It can reopen.", "POLARITY_SHIFT"),
        (
            "Temperatures reached −12 degrees.",
            "Temperatures reached 12 degrees.",
            "NUMERIC_DRIFT",
        ),
        ("🚀 Launch cost €1,200.", "🚀 Launch cost €1,300.", "NUMERIC_DRIFT"),
    ],
)
def test_typed_changes_and_exact_source_spans(older, newer, expected):
    analysis = analyze_text_change(older, newer)
    assert expected in analysis["mutation_types"]
    assert analysis["meaningful_change"] is True
    assert analysis["algorithm_version"] == ALGORITHM_VERSION
    assert analysis["span_offset_unit"] == "unicode_code_points"
    assert analysis["observed_propagation"] is False
    assert any(c["mutation_type"] == expected for c in analysis["changes"])
    for change in analysis["changed_spans"] + analysis["changes"]:
        for span, original in ((change["older_span"], older), (change["newer_span"], newer)):
            assert 0 <= span["start"] <= span["end"] <= len(original)
            assert original[span["start"] : span["end"]] == span["text"]
    assert analysis == analyze_text_change(older, newer)


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        ("The price is 10%.", "  the PRICE is 10% !  "),
        ("It can't reopen.", "It can’t reopen!"),
        ("Identical text.", "Identical text."),
        ("It reached −12 degrees.", "It reached -12 degrees."),
        ("", ""),
    ],
)
def test_formatting_only_is_not_meaningful(older, newer):
    analysis = analyze_text_change(older, newer)
    assert analysis["mutation_types"] == ["NEAR_DUPLICATE"]
    assert analysis["meaningful_change"] is False
    assert analysis["changes"] == analysis["changed_spans"] == []
    assert analysis["similarity"] == 1.0


def test_repeated_numbers_are_not_compared_as_global_sets():
    analysis = analyze_text_change(
        "North received 10 doses and South received 20 doses.",
        "North received 20 doses and South received 10 doses.",
    )
    assert "NUMERIC_DRIFT" in analysis["mutation_types"]
    numeric = [c for c in analysis["changes"] if c["kind"] == "numeric"]
    assert numeric
    assert {v for c in numeric for v in c["removed"]} == {"10", "20"}
    assert {v for c in numeric for v in c["added"]} == {"10", "20"}


@pytest.mark.parametrize(
    ("older", "newer", "direction"),
    [
        ("The project may succeed.", "The project will succeed.", "lost"),
        ("The project will succeed.", "The project may succeed.", "gained"),
    ],
)
def test_hedge_direction_is_older_to_newer(older, newer, direction):
    analysis = analyze_text_change(older, newer)
    hedge = next(c for c in analysis["changes"] if c["kind"] == "hedge")
    assert hedge["direction"] == direction
    assert "may" in hedge["removed" if direction == "lost" else "added"]


def test_wording_change_is_meaningful_even_at_high_similarity():
    older = (
        "The agency published the detailed official report about the new bridge today."
    )
    newer = (
        "The agency published the detailed official report about the old bridge today."
    )
    analysis = analyze_text_change(older, newer)
    assert analysis["similarity"] > 0.9
    assert analysis["meaningful_change"] is True
    assert analysis["mutation_types"] == ["WORDING_DRIFT"]


def test_deleting_entire_text_has_valid_zero_width_new_span():
    analysis = analyze_text_change("Some residents may leave.", "")
    assert analysis["changed_spans"][0]["newer_span"] == {
        "start": 0,
        "end": 0,
        "text": "",
    }
    assert analysis["meaningful_change"] is True


def test_inserting_entire_text_has_valid_zero_width_old_span():
    analysis = analyze_text_change("", "Some residents may leave.")
    assert analysis["changed_spans"][0]["older_span"] == {
        "start": 0,
        "end": 0,
        "text": "",
    }


@pytest.mark.parametrize(
    ("lag", "expected"), [(0, "equal"), (1, "strictly_older"), (-1, "reversed")]
)
def test_temporal_order_is_explicit_and_not_observed(lag, expected):
    older_at = datetime(2026, 9, 20, tzinfo=UTC)
    analysis = analyze_text_change(
        "10 people",
        "12 people",
        older_timestamp=older_at,
        newer_timestamp=older_at + timedelta(seconds=lag),
    )
    assert analysis["temporal_order"] == expected
    assert analysis["lag_seconds"] == lag
    assert analysis["observed_propagation"] is False


def test_missing_timestamps_do_not_use_wall_clock_time():
    analysis = analyze_text_change("10 people", "12 people")
    assert analysis["temporal_order"] == "unknown"
    assert analysis["lag_seconds"] is None
    assert analysis["older_timestamp"] is analysis["newer_timestamp"] is None


def test_naive_and_offset_timestamps_are_normalized_to_utc():
    analysis = analyze_text_change(
        "10 people",
        "12 people",
        older_timestamp=datetime(2026, 9, 20, 12, tzinfo=UTC).replace(tzinfo=None),
        newer_timestamp=datetime(2026, 9, 20, 14, tzinfo=timezone(timedelta(hours=1))),
    )
    assert analysis["lag_seconds"] == 3600
    assert analysis["older_timestamp"] is not None
    assert analysis["newer_timestamp"] is not None
    assert analysis["older_timestamp"].endswith("+00:00")
    assert analysis["newer_timestamp"].endswith("+00:00")


def test_limitations_are_part_of_every_result():
    limitations = " ".join(
        analyze_text_change("10 people", "12 people")["limitations"]
    ).lower()
    for limitation in ("heuristics", "entity", "unit", "scope", "observed", "utc"):
        assert limitation in limitations
