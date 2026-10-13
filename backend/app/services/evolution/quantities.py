"""Conservative, span-preserving normalization of common English quantities.

Values use Decimal and explicit conversion factors: no exchange rates, inferred
units, number words, dates, or locale guessing. Ordered context remains part of
comparison, so swapping two quantities does not count as an equivalent claim.
"""

import re
from decimal import Decimal, InvalidOperation, localcontext
from typing import TypedDict


class Quantity(TypedDict):
    text: str
    start: int
    end: int
    value: str
    unit: str
    dimension: str


# Alias -> (dimension, canonical unit, conversion multiplier).
_UNITS: dict[str, tuple[str, str, str]] = {}
for aliases, dimension, unit, factor in (
    ("metre metres meter meters m", "length", "m", "1"),
    ("kilometre kilometres kilometer kilometers km", "length", "m", "1000"),
    ("centimetre centimetres centimeter centimeters cm", "length", "m", ".01"),
    ("millimetre millimetres millimeter millimeters mm", "length", "m", ".001"),
    ("mile miles mi", "length", "m", "1609.344"),
    ("foot feet ft", "length", "m", ".3048"),
    ("inch inches", "length", "m", ".0254"),
    ("kilogram kilograms kg", "mass", "kg", "1"),
    ("gram grams g", "mass", "kg", ".001"),
    ("milligram milligrams mg", "mass", "kg", ".000001"),
    ("tonne tonnes", "mass", "kg", "1000"),
    ("pound pounds lb lbs", "mass", "kg", ".45359237"),
    ("second seconds sec secs", "duration", "s", "1"),
    ("minute minutes min mins", "duration", "s", "60"),
    ("hour hours hr hrs", "duration", "s", "3600"),
    ("day days", "duration", "s", "86400"),
    ("percent percentage", "ratio", "%", "1"),
    ("dollar dollars usd", "currency:USD", "USD", "1"),
    ("euro euros eur", "currency:EUR", "EUR", "1"),
    ("gbp", "currency:GBP", "GBP", "1"),
):
    for alias in aliases.split():
        _UNITS[alias] = (dimension, unit, factor)

_MULTIPLIERS = {
    "thousand": Decimal(1000),
    "million": Decimal(1000000),
    "billion": Decimal(1000000000),
    "k": Decimal(1000),
    # 'm' is a metre unless a currency symbol disambiguates it.
    "b": Decimal(1000000000),
}
_UNIT_PATTERN = "|".join(sorted(_UNITS, key=len, reverse=True))
_QUANTITY_RE = re.compile(
    r"(?<![\w.])(?P<sign>[+−-]?)(?P<currency>[$€£]?)\s*"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+))"
    r"(?P<multiplier>\s*(?:thousand|million|billion|[kKbB]))?"
    rf"(?P<unit>\s*(?:%|(?:{_UNIT_PATTERN})\b))?"
    r"(?![\w])",
    re.IGNORECASE,
)
_CURRENCY = {"$": "USD", "€": "EUR", "£": "GBP"}
_DATE_TIME_RE = re.compile(
    r"\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}:\d{2}(?::\d{2})?)\b"
)


def extract_quantities(text: str) -> list[Quantity]:
    """Return only explicitly written, supported numerical quantities."""
    quantities: list[Quantity] = []
    date_times = [match.span() for match in _DATE_TIME_RE.finditer(text)]
    for match in _QUANTITY_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in date_times):
            continue
        try:
            value = Decimal(match["number"].replace(",", ""))
        except InvalidOperation:
            continue
        if match["sign"] in {"-", "−"}:
            value = value.copy_negate()
        multiplier = (match["multiplier"] or "").strip().casefold()
        # Preserve long inputs exactly rather than rounding to Decimal's
        # process-wide default precision during multiplication/normalization.
        with localcontext() as context:
            context.prec = len(match["number"]) + 32
            value *= _MULTIPLIERS.get(multiplier, Decimal(1))
        written_unit = (match["unit"] or "").strip().casefold()
        currency = match["currency"]
        if currency and written_unit == "m":
            with localcontext() as context:
                context.prec = len(match["number"]) + 32
                value *= Decimal(1000000)
            written_unit = ""
        dimension, unit = "count", "count"
        if written_unit == "%":
            dimension, unit = "ratio", "%"
        elif written_unit:
            dimension, unit, factor = _UNITS[written_unit]
            with localcontext() as context:
                context.prec = len(match["number"]) + 32
                value *= Decimal(factor)
        if currency:
            # $5 kg is contradictory syntax; keep the number separate from kg
            # instead of silently treating unlike dimensions as equivalent.
            if written_unit and dimension != f"currency:{_CURRENCY[currency]}":
                continue
            unit = _CURRENCY[currency]
            dimension = f"currency:{unit}"
        with localcontext() as context:
            context.prec = len(match["number"]) + 32
            canonical = format(value.normalize(), "f")
        if value == 0:
            canonical = "0"
        quantities.append(
            {
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "value": canonical,
                "unit": unit,
                "dimension": dimension,
            }
        )
    return quantities


def quantity_key(quantity: Quantity) -> str:
    return f"quantity:{quantity['dimension']}:{quantity['value']}"
