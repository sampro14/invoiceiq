"""Do two readings of a field agree? Used for cross-checking and for grounding in the text layer."""

import re
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any

from agent.normalize import fold, fold_id, parse_amount, parse_date
from agent.state import HEADER_PATHS, LINE_LEAVES, InvoiceFields

_FREE_TEXT = ("vendor.name", "vendor.address", "description")
_IDS = ("vendor.tax_id", "invoice.number", "invoice.po_number")


def similar(a: str, b: str, threshold: float = 0.8) -> bool:
    x, y = fold(a), fold(b)
    if x == y:
        return True
    if not x or not y:
        return False
    shorter, longer = sorted((x, y), key=len)
    # One reader adding "Pvt Ltd" or a bill-to city is not a disagreement about the value.
    if len(shorter) >= 4 and f" {shorter} " in f" {longer} " and len(shorter) >= 0.5 * len(longer):
        return True
    return SequenceMatcher(None, x, y).ratio() >= threshold


_ZERO_IS_ABSENT = ("amounts.discount", "amounts.shipping", "amounts.tax_amount", "amounts.tax_rate")


def values_agree(path: str, a: Any, b: Any) -> bool:
    if path in _ZERO_IS_ABSENT:
        a, b = a or None, b or None
    if a is None or b is None:
        return a is b
    if isinstance(a, Decimal) and isinstance(b, Decimal):
        return abs(a - b) <= Decimal("0.005")
    if path in _IDS:
        return fold_id(a) == fold_id(b)
    if path.endswith(_FREE_TEXT):
        return similar(str(a), str(b))
    return fold(a) == fold(b)


def find_disagreements(
    first: InvoiceFields, second: InvoiceFields, only: list[str] | None = None
) -> list[str]:
    """Paths (of `first`) where the two readings differ. A different number of lines puts every
    line in doubt, because row-by-row comparison no longer means anything."""
    out = [
        p
        for p in HEADER_PATHS
        if (only is None or p in only) and not values_agree(p, first.get(p), second.get(p))
    ]
    wants_lines = only is None or any(p.startswith("line_items") for p in only)
    if wants_lines:
        if len(first.line_items) != len(second.line_items):
            out.append("line_items")
        else:
            for i in range(len(first.line_items)):
                for leaf in LINE_LEAVES:
                    path = f"line_items.{i}.{leaf}"
                    if (only is None or path in only or "line_items" in only) and not values_agree(
                        path, first.get(path), second.get(path)
                    ):
                        out.append(path)
    return out


_NUMBER = re.compile(r"\d[\d.,]*\d|\d")
_DATEISH = re.compile(
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}(?:st|nd|rd|th)?[-/. ]+(?:\d{1,2}|[A-Za-z]{3,9})\.?,?[-/. ]+\d{2,4}"
    r"|[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}"
)


class Grounding:
    """Is a value actually printed in the document's text layer?"""

    def __init__(self, text: str):
        self._folded = f" {fold(text)} "
        self._ids = fold_id(text)
        self._numbers = {n for m in _NUMBER.findall(text) if (n := parse_amount(m)) is not None}
        self._dates = {d for m in _DATEISH.findall(text) if (d := parse_date(m)) is not None}

    def supports(self, path: str, value: Any) -> bool | None:
        """None means grounding does not apply to this field."""
        if value is None or path in ("invoice.currency", "amounts.tax_rate", "amounts.tax_amount"):
            return None  # currency may be a symbol; tax may be the sum of CGST and SGST
        if isinstance(value, Decimal):
            return any(abs(abs(value) - n) <= Decimal("0.005") for n in self._numbers)
        if path in ("invoice.issue_date", "invoice.due_date"):
            day = parse_date(value)
            return None if day is None else day in self._dates
        if path in _IDS:
            return fold_id(value) in self._ids
        words = fold(value).split()
        if not words:
            return None
        hits = sum(f" {w} " in self._folded for w in words)
        return hits >= 0.7 * len(words)
