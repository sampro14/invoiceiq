"""Turning printed text into comparable values. Shared by the agent and the eval harness, so a
field is scored exactly the way it is validated."""

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

from dateutil import parser as dateparser

# "$" is ambiguous; it maps to USD only as a last resort, after the model's own answer.
CURRENCY_SYMBOLS = {
    "₹": "INR", "RS": "INR", "RS.": "INR", "€": "EUR", "£": "GBP", "¥": "JPY", "$": "USD",
    "US$": "USD", "RM": "MYR", "RP": "IDR", "RP.": "IDR", "DHS": "AED", "DH": "AED",
    "S$": "SGD", "A$": "AUD", "C$": "CAD", "HK$": "HKD", "د.إ": "AED",
}  # fmt: skip

ISO_4217 = frozenset(
    [
        "AED",
        "AFN",
        "ALL",
        "AMD",
        "AOA",
        "ARS",
        "AUD",
        "AWG",
        "AZN",
        "BAM",
        "BBD",
        "BDT",
        "BGN",
        "BHD",
        "BIF",
        "BMD",
        "BND",
        "BOB",
        "BRL",
        "BSD",
        "BTN",
        "BWP",
        "BYN",
        "BZD",
        "CAD",
        "CDF",
        "CHF",
        "CLP",
        "CNY",
        "COP",
        "CRC",
        "CUP",
        "CVE",
        "CZK",
        "DJF",
        "DKK",
        "DOP",
        "DZD",
        "EGP",
        "ERN",
        "ETB",
        "EUR",
        "FJD",
        "FKP",
        "GBP",
        "GEL",
        "GHS",
        "GIP",
        "GMD",
        "GNF",
        "GTQ",
        "GYD",
        "HKD",
        "HNL",
        "HTG",
        "HUF",
        "IDR",
        "ILS",
        "INR",
        "IQD",
        "IRR",
        "ISK",
        "JMD",
        "JOD",
        "JPY",
        "KES",
        "KGS",
        "KHR",
        "KMF",
        "KPW",
        "KRW",
        "KWD",
        "KYD",
        "KZT",
        "LAK",
        "LBP",
        "LKR",
        "LRD",
        "LSL",
        "LYD",
        "MAD",
        "MDL",
        "MGA",
        "MKD",
        "MMK",
        "MNT",
        "MOP",
        "MRU",
        "MUR",
        "MVR",
        "MWK",
        "MXN",
        "MYR",
        "MZN",
        "NAD",
        "NGN",
        "NIO",
        "NOK",
        "NPR",
        "NZD",
        "OMR",
        "PAB",
        "PEN",
        "PGK",
        "PHP",
        "PKR",
        "PLN",
        "PYG",
        "QAR",
        "RON",
        "RSD",
        "RUB",
        "RWF",
        "SAR",
        "SBD",
        "SCR",
        "SDG",
        "SEK",
        "SGD",
        "SHP",
        "SLE",
        "SOS",
        "SRD",
        "SSP",
        "STN",
        "SVC",
        "SYP",
        "SZL",
        "THB",
        "TJS",
        "TMT",
        "TND",
        "TOP",
        "TRY",
        "TTD",
        "TWD",
        "TZS",
        "UAH",
        "UGX",
        "USD",
        "UYU",
        "UZS",
        "VED",
        "VES",
        "VND",
        "VUV",
        "WST",
        "XAF",
        "XCD",
        "XCG",
        "XOF",
        "XPF",
        "YER",
        "ZAR",
        "ZMW",
        "ZWG",
    ]
)

_NUM = re.compile(r"[-+(]?\s*\d[\d.,'   ]*\d|[-+(]?\s*\d")


def parse_amount(value: object) -> Decimal | None:
    """Parse "₹ 1,23,456.50", "1.234,56", "(12.00)" or "1 234.5" into a Decimal.

    A lone separator followed by exactly three digits ("1.591", "12,000") is read as a thousands
    separator, because amounts never carry three decimals in practice.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        d = Decimal(str(value))
        return d if d.is_finite() else None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    match = _NUM.search(text)
    if not match:
        return None
    raw = match.group(0)
    prefix = text[: match.start()].rstrip()
    negative = raw.lstrip().startswith(("-", "(")) or prefix.endswith("-") or text.endswith("-")
    digits = re.sub(r"[^\d.,]", "", raw)
    if "," in digits and "." in digits:
        decimal_sep = "," if digits.rfind(",") > digits.rfind(".") else "."
        other = "." if decimal_sep == "," else ","
        digits = digits.replace(other, "").replace(decimal_sep, ".")
    elif "," in digits or "." in digits:
        sep = "," if "," in digits else "."
        head, _, tail = digits.rpartition(sep)
        if digits.count(sep) > 1 or (len(tail) == 3 and head not in ("", "0")):
            digits = digits.replace(sep, "")
        else:
            digits = f"{head}.{tail}"
    try:
        result = Decimal(digits)
    except InvalidOperation:
        return None
    return -result if negative else result


def parse_decimal(value: object) -> Decimal | None:
    """Parse a value a model was asked to return as a plain decimal ("1234.5", "0.125").

    Unlike parse_amount, "1.125" keeps its decimals: unit prices and quantities can have three.
    """
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        return Decimal(value.strip())
    return parse_amount(value)


def parse_date(value: object) -> date | None:
    """Parse to a date. Ambiguous numeric dates are day-first, the convention in this demo's
    markets (India, UAE, Malaysia)."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if len(text) < 6 or not re.search(r"\d", text):
        return None
    iso = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:\D|$)", text)
    try:
        if iso:
            return date(*(int(g) for g in iso.groups()))
        # dateutil fills missing parts from a default; two different defaults expose that.
        from datetime import datetime

        a = dateparser.parse(text, dayfirst=True, default=datetime(2001, 1, 1))
        b = dateparser.parse(text, dayfirst=True, default=datetime(2002, 2, 2))
        return a.date() if a.date() == b.date() else None
    except (ValueError, OverflowError, TypeError):
        return None


def normalize_currency(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text in ISO_4217:
        return text
    return CURRENCY_SYMBOLS.get(text, text)


def fold(value: object) -> str:
    """Case, whitespace and punctuation folded, for text comparison."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).replace("_", " ").strip()


def fold_id(value: object) -> str:
    """Identifiers compare on their letters and digits only: "INV/05-24" equals "inv-05 24"."""
    return re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", str(value or "")).casefold())


def money(value: Decimal) -> str:
    return f"{value:.2f}"
