"""Scoring. Every function is pure, so every number in the report can be traced to one."""

import math
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import SequenceMatcher
from statistics import mean, median
from typing import Any

from scipy.optimize import linear_sum_assignment

from agent.normalize import fold, fold_id, normalize_currency, parse_amount, parse_date
from agent.state import LINE_LEAVES

AMOUNTS = {
    "amounts.subtotal", "amounts.discount", "amounts.shipping", "amounts.tax_amount", "amounts.total",
    "quantity", "unit_price", "line_total",
}  # fmt: skip
DATES = {"invoice.issue_date", "invoice.due_date"}
IDS = {"vendor.tax_id", "invoice.number", "invoice.po_number"}
FREE_TEXT = {"vendor.address", "description"}


def canonical(path: str, value: Any) -> Any:
    """Exact match after normalisation: dates to ISO, amounts to 2 decimals, text folded."""
    if value is None or value == "":
        return None
    if path in AMOUNTS:
        d = value if isinstance(value, Decimal) else parse_amount(value)
        return None if d is None else f"{abs(d) if path == 'amounts.discount' else d:.2f}"
    if path == "amounts.tax_rate":
        d = value if isinstance(value, Decimal) else parse_amount(value)
        return None if d is None else f"{d:.2f}"
    if path in DATES:
        parsed = parse_date(value)
        return parsed.isoformat() if parsed else fold(value)
    if path == "invoice.currency":
        return normalize_currency(value)
    if path in IDS:
        return fold_id(value) or None
    return fold(value) or None


def matches(path: str, predicted: Any, truth: Any) -> bool:
    p, t = canonical(path, predicted), canonical(path, truth)
    if p == t:
        return True
    if p is None or t is None or path not in FREE_TEXT:
        return False
    # Free text has no single right spelling. This is the deterministic stand-in for the LLM judge
    # (eval/judge.py), whose agreement with hand labels is reported separately.
    return SequenceMatcher(None, p.replace(" ", ""), t.replace(" ", "")).ratio() >= 0.9


@dataclass
class Counts:
    correct: int = 0
    predicted: int = 0
    truth: int = 0

    def add(self, other: "Counts") -> None:
        self.correct += other.correct
        self.predicted += other.predicted
        self.truth += other.truth

    @property
    def f1(self) -> float | None:
        if not self.predicted and not self.truth:
            return None
        p = self.correct / self.predicted if self.predicted else 0.0
        r = self.correct / self.truth if self.truth else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class DocScore:
    header: Counts = field(default_factory=Counts)
    lines: Counts = field(default_factory=Counts)
    exact: bool = False
    wrong: list[str] = field(default_factory=list)
    # (confidence, was it right), for calibration
    calibration: list[tuple[float, bool]] = field(default_factory=list)


def score_document(
    predicted: dict[str, Any] | None,
    confidence: dict[str, float],
    labels: dict[str, Any],
    scored: list[str],
    required: list[str],
) -> DocScore:
    """`scored` lists the fields this dataset has ground truth for; nothing else is counted, for or
    against. `predicted` is InvoiceFields.model_dump(mode="json")."""
    out = DocScore()
    for path in scored:
        if path == "line_items":
            continue
        group, leaf = path.split(".")
        value = ((predicted or {}).get(group) or {}).get(leaf)
        truth = labels.get(path)
        has_p, has_t = canonical(path, value) is not None, canonical(path, truth) is not None
        if path in ("amounts.discount", "amounts.shipping", "amounts.tax_amount"):
            # Zero and absent mean the same thing.
            has_p = has_p and canonical(path, value) != "0.00"
            has_t = has_t and canonical(path, truth) != "0.00"
        ok = matches(path, value, truth) if has_p and has_t else has_p == has_t
        out.header.add(Counts(int(ok and has_t), int(has_p), int(has_t)))
        if not ok:
            out.wrong.append(path)
        if has_p and path in confidence:
            out.calibration.append((confidence[path], ok))
    if "line_items" in scored:
        _score_lines(predicted, confidence, labels.get("line_items") or [], labels, out)
    needed = [p for p in required if p in scored]
    out.exact = predicted is not None and not any(
        w in needed or (w.startswith("line_items") and "line_items" in needed) for w in out.wrong
    )
    return out


def _score_lines(
    predicted: dict[str, Any] | None,
    confidence: dict[str, float],
    truth: list[dict[str, Any]],
    labels: dict[str, Any],
    out: DocScore,
) -> None:
    leaves = labels.get("line_leaves") or list(LINE_LEAVES)
    guess = (predicted or {}).get("line_items") or []
    has = lambda row, leaf: canonical(leaf, row.get(leaf)) is not None  # noqa: E731
    pairs: list[tuple[int, int]] = []
    if guess and truth:
        gain = [[_similarity(g, t, leaves) for t in truth] for g in guess]
        rows, cols = linear_sum_assignment(gain, maximize=True)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if gain[r][c] > 0]
    by_guess = dict(pairs)
    for i, g in enumerate(guess):
        t = truth[by_guess[i]] if i in by_guess else {}
        for leaf in leaves:
            if not has(g, leaf):
                continue
            # A field the annotators left blank on a row we matched is not counted as invented.
            if t and not has(t, leaf):
                continue
            ok = bool(t) and matches(leaf, g.get(leaf), t.get(leaf))
            out.lines.add(Counts(int(ok), 1, 0))
            path = f"line_items.{i}.{leaf}"
            if not ok:
                out.wrong.append(path)
            if path in confidence:
                out.calibration.append((confidence[path], ok))
    matched = {t for _, t in pairs}
    for j, t in enumerate(truth):
        for leaf in leaves:
            if not has(t, leaf):
                continue
            out.lines.truth += 1
            if j not in matched:
                out.wrong.append(f"line_items.missing{j}.{leaf}")
    for i, j in pairs:
        for leaf in leaves:
            if has(truth[j], leaf) and not has(guess[i], leaf):
                out.wrong.append(f"line_items.{i}.{leaf}")


def _similarity(g: dict[str, Any], t: dict[str, Any], leaves: list[str]) -> float:
    score = 0.0
    for leaf in leaves:
        a, b = canonical(leaf, g.get(leaf)), canonical(leaf, t.get(leaf))
        if a is None or b is None:
            continue
        score += (
            1.0 if a == b else SequenceMatcher(None, a, b).ratio() if leaf == "description" else 0.0
        )
    return score


def expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    total, error = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if inside:
            gap = abs(mean(c for c, _ in inside) - mean(float(ok) for _, ok in inside))
            error += len(inside) / total * gap
    return error


def percentile(values: list[float], p: float) -> float | None:
    """Nearest rank, which stays honest on small sets."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def rate(flags: list[bool]) -> float | None:
    return mean(map(float, flags)) if flags else None


def p50(values: list[float]) -> float | None:
    return median(values) if values else None
