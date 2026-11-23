"""Business rules. Pure functions over the extracted fields: no model is ever asked to do
arithmetic or judge a format."""

import re
from decimal import Decimal

from agent.injection import find_instructions
from agent.normalize import ISO_4217, parse_date
from agent.state import CheckResult, InvoiceFields, all_paths, required_paths

TOLERANCE = Decimal("0.01")

GSTIN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
UAE_TRN = re.compile(r"^\d{15}$")
GENERIC_TAX_ID = re.compile(r"^[A-Z0-9]{5,20}$")


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= TOLERANCE


def check_required(f: InvoiceFields) -> CheckResult:
    missing = [p for p in required_paths(f) if f.get(p) in (None, "")]
    if f.document_type == "invoice" and not f.line_items:
        missing.append("line_items")
    if missing:
        return CheckResult(
            rule="required_fields", status="fail", detail="Missing: " + ", ".join(missing), fields=missing
        )  # fmt: skip
    return CheckResult(
        rule="required_fields", status="pass", detail="Every required field is present"
    )


def check_line_math(f: InvoiceFields) -> CheckResult:
    rule = "line_math"
    bad: list[int] = []
    details: list[str] = []
    checked = 0
    for i, line in enumerate(f.line_items):
        if line.quantity is None or line.unit_price is None or line.line_total is None:
            continue
        checked += 1
        expected = line.quantity * line.unit_price
        if not _close(expected, line.line_total):
            bad.append(i)
            details.append(
                f"line {i + 1}: {line.quantity} × {line.unit_price} = {expected:.2f}, printed {line.line_total}"
            )
    if bad:
        paths = [
            f"line_items.{i}.{k}" for i in bad for k in ("quantity", "unit_price", "line_total")
        ]
        return CheckResult(rule=rule, status="fail", detail="; ".join(details), fields=paths)
    if not checked:
        return CheckResult(
            rule=rule, status="warn", detail="No line has quantity, price and total to check"
        )
    return CheckResult(
        rule=rule, status="pass", detail=f"Quantity × unit price matches on {checked} line(s)"
    )


def check_lines_sum(f: InvoiceFields) -> CheckResult:
    rule = "lines_to_subtotal"
    totals = [line.line_total for line in f.line_items]
    if not totals or any(t is None for t in totals):
        return CheckResult(rule=rule, status="warn", detail="Not enough line totals to add up")
    total = sum((t for t in totals if t is not None), Decimal(0))
    # A receipt with no printed subtotal adds its lines straight to the total.
    target, name = f.amounts.subtotal, "subtotal"
    if target is None and f.document_type == "receipt":
        target, name = f.amounts.total, "total"
    if target is None:
        return CheckResult(rule=rule, status="warn", detail="No subtotal to compare with")
    if _close(total, target):
        return CheckResult(rule=rule, status="pass", detail=f"Lines add up to the {name}: {total}")
    return CheckResult(
        rule=rule, status="fail",
        detail=f"Lines add up to {total}, {name} says {target} (off by {abs(total - target)})",
        fields=["line_items", f"amounts.{name}"],
    )  # fmt: skip


def check_total(f: InvoiceFields) -> CheckResult:
    rule = "subtotal_tax_total"
    a = f.amounts
    if a.total is None or a.subtotal is None:
        return CheckResult(rule=rule, status="warn", detail="Subtotal or total not present")
    zero = Decimal(0)
    tax, discount, shipping = a.tax_amount or zero, abs(a.discount or zero), a.shipping or zero
    expected = a.subtotal - discount + shipping + tax
    parts = f"{a.subtotal}"
    if discount:
        parts += f" − {discount} discount"
    if shipping:
        parts += f" + {shipping} shipping"
    parts += f" + {tax} tax"
    if _close(expected, a.total):
        return CheckResult(rule=rule, status="pass", detail=f"{parts} = {a.total}")
    if f.document_type == "receipt" and _close(a.subtotal - discount + shipping, a.total):
        return CheckResult(
            rule=rule, status="pass", detail=f"Tax-inclusive receipt: {a.subtotal} = {a.total}"
        )
    return CheckResult(
        rule=rule, status="fail",
        detail=f"{parts} = {expected}, total says {a.total} (off by {abs(expected - a.total)})",
        fields=["amounts.subtotal", "amounts.discount", "amounts.shipping", "amounts.tax_amount", "amounts.total"],
    )  # fmt: skip


def check_dates(f: InvoiceFields) -> CheckResult:
    rule = "dates"
    issue, due = f.invoice.issue_date, f.invoice.due_date
    bad = [
        path
        for path, value in (("invoice.issue_date", issue), ("invoice.due_date", due))
        if value is not None and parse_date(value) is None
    ]
    if bad:
        return CheckResult(
            rule=rule, status="fail", fields=bad,
            detail="Not a date: " + ", ".join(f"{p} = {f.get(p)!r}" for p in bad),
        )  # fmt: skip
    if issue is None and due is None:
        return CheckResult(rule=rule, status="warn", detail="No dates to check")
    if issue and due:
        issued, dued = parse_date(issue), parse_date(due)
        if issued and dued and dued < issued:
            return CheckResult(
                rule=rule, status="fail", detail=f"Due date {due} is before issue date {issue}",
                fields=["invoice.issue_date", "invoice.due_date"],
            )  # fmt: skip
    return CheckResult(rule=rule, status="pass", detail="Dates parse and are in order")


def check_tax_id(f: InvoiceFields) -> CheckResult:
    rule = "tax_id"
    raw = f.vendor.tax_id
    if not raw:
        return CheckResult(rule=rule, status="warn", detail="No vendor tax ID on the document")
    tax_id = re.sub(r"[\s\-./:]", "", raw).upper()
    tax_id = re.sub(r"^(GSTIN|GST|VAT|TRN|TIN)(NO)?(?=\w{5})", "", tax_id)
    address = (f.vendor.address or "").casefold()
    if f.invoice.currency == "INR" or "india" in address:
        ok, kind = bool(GSTIN.match(tax_id)), "India GSTIN"
    elif f.invoice.currency == "AED" or re.search(
        r"\buae\b|united arab emirates|dubai|abu dhabi", address
    ):
        ok, kind = bool(UAE_TRN.match(tax_id)), "UAE TRN (15 digits)"
    elif GSTIN.match(tax_id):
        ok, kind = True, "India GSTIN"
    else:
        ok, kind = bool(GENERIC_TAX_ID.match(tax_id)), "generic VAT/tax ID"
    if ok:
        return CheckResult(rule=rule, status="pass", detail=f"{raw} matches {kind} format")
    return CheckResult(
        rule=rule,
        status="fail",
        detail=f"{raw} does not match {kind} format",
        fields=["vendor.tax_id"],
    )


def check_currency(f: InvoiceFields) -> CheckResult:
    rule = "currency"
    code = f.invoice.currency
    if code is None:
        return CheckResult(rule=rule, status="warn", detail="No currency found")
    if code in ISO_4217:
        return CheckResult(rule=rule, status="pass", detail=f"{code} is a valid ISO 4217 code")
    return CheckResult(
        rule=rule,
        status="fail",
        detail=f"{code!r} is not an ISO 4217 code",
        fields=["invoice.currency"],
    )


def check_instruction_text(f: InvoiceFields, suspicious: list[str]) -> CheckResult:
    """Never repairable: re-asking cannot make a hostile document trustworthy."""
    found = list(suspicious)
    for path in all_paths(f):
        value = f.get(path)
        if isinstance(value, str):
            found += find_instructions(value)
    if found:
        return CheckResult(
            rule="instruction_text", status="fail",
            detail="Instruction-like text treated as data: " + " | ".join(dict.fromkeys(found)),
        )  # fmt: skip
    return CheckResult(
        rule="instruction_text", status="pass", detail="No text addressed to the model"
    )


def run_checks(f: InvoiceFields, suspicious: list[str] | None = None) -> list[CheckResult]:
    return [
        check_required(f),
        check_line_math(f),
        check_lines_sum(f),
        check_total(f),
        check_dates(f),
        check_tax_id(f),
        check_currency(f),
        check_instruction_text(f, suspicious or []),
    ]
