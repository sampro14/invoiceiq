from datetime import date
from decimal import Decimal

import pytest

from agent.normalize import (
    fold,
    fold_id,
    normalize_currency,
    parse_amount,
    parse_date,
    parse_decimal,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234.50", "1234.50"), ("₹ 1,23,456.00", "123456.00"), ("1.234,56", "1234.56"),
        ("1.591.600", "1591600"), ("12,000", "12000"), ("60.000", "60000"), ("0.125", "0.125"),
        ("12,5", "12.5"), ("(45.00)", "-45.00"), ("-45.00", "-45.00"), ("45.00-", "-45.00"),
        ("USD 99", "99"), ("5", "5"), ("AED 95,507.79", "95507.79"), ("1 234.50", "1234.50"),
        (3, "3"), (2.5, "2.5"),
    ],
)  # fmt: skip
def test_parse_amount(text, expected):
    assert parse_amount(text) == Decimal(expected)


@pytest.mark.parametrize("text", ["", None, "abc", "--", True, float("nan"), float("inf")])
def test_parse_amount_rejects(text):
    assert parse_amount(text) is None


def test_plain_decimals_from_models_keep_three_places():
    assert parse_decimal("1.125") == Decimal("1.125")
    assert parse_decimal("1.591.600") == Decimal("1591600")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-01", date(2026, 3, 1)), ("01/03/2026", date(2026, 3, 1)), ("15/01/2019", date(2019, 1, 15)),
        ("08-Jan-1999", date(1999, 1, 8)), ("Mar 5, 2026", date(2026, 3, 5)), ("2026/3/9", date(2026, 3, 9)),
        ("25 Dec 2018 8:13 PM", date(2018, 12, 25)),
    ],
)  # fmt: skip
def test_parse_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "text", ["", None, "soon", "March 2026", "2026-13-45", "net 30", "30 days"]
)
def test_parse_date_rejects_partial_or_invalid(text):
    assert parse_date(text) is None


def test_currency_and_folding():
    assert normalize_currency("₹") == "INR"
    assert normalize_currency("rm") == "MYR"
    assert normalize_currency("aed") == "AED"
    assert normalize_currency("XQZ") == "XQZ"
    assert fold("  ACME,  Pvt. Ltd ") == "acme pvt ltd"
    assert fold_id("INV/05-24 / 467") == "inv0524467"
