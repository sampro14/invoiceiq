"""Deterministic readers used when no API key is configured, in tests and in keyless CI.

They are not models and the app never presents them as such: every result carries
`mode: "offline"`. The two use different methods on purpose, so cross-checking them still means
something:

* OfflineRules reads labels ("Invoice No:", "Total").
* OfflineAnchors ignores labels where it can and works from positions and arithmetic.

Neither can read an image.
"""

import json
import re
import time
from decimal import Decimal
from typing import Any

from agent.injection import find_instructions
from agent.llm.base import LLMError, LLMRequest, LLMResponse
from agent.normalize import CURRENCY_SYMBOLS, parse_amount, parse_date

_MONEY = r"-?\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})|-?\d+\.\d{1,2}"
_DATE = r"(?:\d{1,2}[-/ .](?:\d{1,2}|[A-Za-z]{3,9})[-/ .]\d{4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9} \d{1,2}, \d{4})"
_CODES = re.compile(r"\b(INR|AED|USD|EUR|GBP|SGD|MYR|AUD|CAD|JPY|SAR)\b|(₹|€|£|\$)")
_GSTIN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z]{3}\b")


def _plain(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


class _Offline:
    provider = "offline"
    model = ""
    vision = False

    def _read(self, text: str) -> dict[str, Any]:
        raise NotImplementedError

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.images or not request.document_text:
            raise LLMError(
                "unsupported",
                "Offline mode reads digital PDFs only. Set GEMINI_API_KEY to read scans and photos.",
                self.provider,
            )
        started = time.perf_counter()
        keys = request.schema["required"]
        data = self._read(request.document_text)
        answer = {k: data.get(k) for k in keys}
        for k in ("line_items", "uncertain_fields", "suspicious_text"):
            if k in answer and answer[k] is None:
                answer[k] = []
        return LLMResponse(
            text=json.dumps(answer),
            provider=self.provider,
            model=self.model,
            tokens_in=0,
            tokens_out=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _currency(text: str) -> str | None:
    m = _CODES.search(text)
    if not m:
        return None
    return m.group(1) or CURRENCY_SYMBOLS.get(m.group(2))


_ROW = re.compile(
    rf"^\s*(?:\d{{1,2}}\s+)?(?P<desc>\S.*?)\s+(?P<qty>\d+(?:\.\d+)?)\s+(?P<price>{_MONEY})\s+(?P<total>{_MONEY})\s*$"
)


class OfflineRules(_Offline):
    model = "offline-rules"

    def _label(self, text: str, label: str, value: str) -> str | None:
        m = re.search(rf"(?im)(?:^|\s{{2,}}){label}\s*[:#]?\s*({value})", text)
        return m.group(1).strip() if m else None

    def _amount(self, text: str, label: str) -> str | None:
        raw = self._label(text, label, rf"(?:[A-Z]{{3}}|[₹€£$]|Rs\.?)?\s*-?\(?(?:{_MONEY})\)?")
        return _plain(parse_amount(raw)) if raw else None

    def _read(self, text: str) -> dict[str, Any]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        items = []
        for line in lines:
            m = _ROW.match(line)
            if m and not re.match(r"(?i)\s*(sub\s*total|total|tax|cgst|sgst|igst|vat)", line):
                items.append({
                    "description": " ".join(m["desc"].split()),
                    "quantity": m["qty"],
                    "unit_price": _plain(parse_amount(m["price"])),
                    "line_total": _plain(parse_amount(m["total"])),
                })  # fmt: skip
        taxes = [
            parse_amount(a)
            for a in re.findall(
                rf"(?im)^.*?\b(?:CGST|SGST|IGST|VAT|Tax)\b[^\n]*?({_MONEY})\s*$", text
            )
        ]
        rates = [
            Decimal(r)
            for r in re.findall(
                r"(?i)\b(?:CGST|SGST|IGST|VAT|Tax)\b[^\n%]{0,8}?(\d+(?:\.\d+)?)\s*%", text
            )
        ]
        tax = sum((t for t in taxes if t is not None), Decimal(0)) if taxes else None
        first = [re.split(r"\s{2,}", ln.strip())[0] for ln in lines[:4]]
        vendor = next((s for s in first if not re.search(r"(?i)invoice|receipt", s)), None)
        address = (
            first[first.index(vendor) + 1]
            if vendor and first.index(vendor) + 1 < len(first)
            else None
        )
        due = self._label(text, r"due date", _DATE)
        issue = self._label(text, r"(?:invoice date|issue date|date of issue|date)", _DATE)
        parsed_issue, parsed_due = parse_date(issue), parse_date(due)
        return {
            "document_type": "receipt" if re.search(r"(?i)\breceipt\b", text) else "invoice",
            "vendor_name": vendor,
            "vendor_address": address,
            "vendor_tax_id": self._label(text, r"(?:GSTIN|TRN|VAT (?:no|id|number)|tax id)", r"[0-9A-Z]{8,18}"),
            "invoice_number": self._label(text, r"invoice\s*(?:no\.?|number|id|#)", r"[A-Z0-9][\w/-]+"),
            "issue_date": parsed_issue.isoformat() if parsed_issue else issue,
            "due_date": parsed_due.isoformat() if parsed_due else due,
            "currency": _currency(text),
            "po_number": self._label(text, r"p\.?o\.?\s*(?:no\.?|number|#)", r"[A-Z0-9][\w/-]+"),
            "subtotal": self._amount(text, r"(?:sub\s*-?total|taxable value)"),
            "discount": self._amount(text, r"discount"),
            "shipping": self._amount(text, r"(?:shipping|freight)"),
            "tax_amount": _plain(tax),
            "tax_rate": _plain(sum(rates, Decimal(0))) if rates else None,
            "total": self._amount(text, r"(?:grand total|total due|amount due|total)"),
            "line_items": items,
            "suspicious_text": find_instructions(text),
        }  # fmt: skip


class OfflineAnchors(_Offline):
    model = "offline-anchors"

    def _read(self, text: str) -> dict[str, Any]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        items, row_numbers = [], set()
        for n, line in enumerate(lines):
            tokens = re.findall(r"-?[\d,]+(?:\.\d+)?", line)
            values = [parse_amount(t) for t in tokens]
            if len(values) < 3 or None in values[-3:]:
                continue
            q, p, total = values[-3:]
            assert q is not None and p is not None and total is not None
            # A row is a line item if it has a name and three trailing numbers shaped like q, p, total.
            name = re.split(r"\s{2,}|\s+\d", re.sub(r"^\s*\d{1,2}\s+", "", line).strip())[0]
            if q < 10000 and re.search(r"[A-Za-z]{3}", name) and re.search(r"\.\d{2}\s*$", line):
                if re.match(r"(?i)(sub|total|tax|[cis]gst|vat)", name):
                    continue
                desc = re.sub(r"^\s*\d{1,2}\s+", "", line).strip()
                desc = re.split(rf"\s+{re.escape(tokens[-3])}\s+{re.escape(tokens[-2])}", desc)[0]
                items.append({
                    "description": " ".join(desc.split()), "quantity": _plain(q),
                    "unit_price": _plain(p), "line_total": _plain(total),
                })  # fmt: skip
                row_numbers.add(n)
        # Everything after the last row is the totals block.
        start = max(row_numbers) + 1 if row_numbers else 0
        tail = [
            (line, parse_amount(m[-1]))
            for line in lines[start:]
            if (m := re.findall(_MONEY, line)) and not re.search(r"%\s*$", line)
        ]
        amounts = [a for _, a in tail if a is not None]
        subtotal = amounts[0] if amounts else None
        total = max(amounts, key=abs) if amounts else None
        if total is not None and amounts.count(total) == 1 and amounts[-1] > 0:
            total = amounts[-1]
        middle = amounts[1:-1] if len(amounts) > 2 else []
        discount = sum((abs(a) for a in middle if a < 0), Decimal(0)) or None
        tax = sum((a for a in middle if a > 0), Decimal(0)) if middle else None
        dates = sorted(
            {
                d.isoformat()
                for raw in re.findall(_DATE, "\n".join(lines[: start or None]))
                if (d := parse_date(raw))
            }
        )
        head = re.split(r"\s{2,}", lines[0].strip())[0] if lines else None
        gstin = _GSTIN.search(text)
        trn = re.search(r"\b\d{15}\b", text)
        number = next(
            (
                t
                for t in re.findall(r"\b[A-Z]{1,5}[-/][A-Z0-9/-]*\d[A-Z0-9/-]*\b", text)
                if not t.startswith("PO")
            ),
            None,
        )
        po = re.search(r"\bPO[-/ ]?\d[\w-]*", text)
        rates = [Decimal(r) for r in re.findall(r"(\d+(?:\.\d+)?)\s*%", "\n".join(lines[start:]))]
        second = re.split(r"\s{2,}", lines[1].strip())[0] if len(lines) > 1 else None
        return {
            "document_type": "invoice" if re.search(r"(?i)invoice", text) else "receipt",
            "vendor_name": head,
            "vendor_address": second,
            "vendor_tax_id": gstin.group(0) if gstin else (trn.group(0) if trn else None),
            "invoice_number": number,
            "issue_date": dates[0] if dates else None,
            "due_date": dates[-1] if len(dates) > 1 else None,
            "currency": _currency(text),
            "po_number": po.group(0) if po else None,
            "subtotal": _plain(subtotal),
            "discount": _plain(discount),
            "tax_amount": _plain(tax),
            "tax_rate": _plain(sum(rates, Decimal(0))) if rates else None,
            "total": _plain(total),
            "line_items": items,
            "suspicious_text": find_instructions(text),
        }
