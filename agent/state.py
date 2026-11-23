"""The one typed object passed between graph nodes, plus the schema it extracts.

Two deliberate differences from a naive design: the document's bytes and text are *not* part of
the state. State is checkpointed to SQLite so a paused run can resume, and the demo promises that
uploads never touch disk. They live in `agent.documents.DocumentStore` (memory only) instead.
"""

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

from agent.normalize import normalize_currency, parse_date, parse_decimal

FileKind = Literal["pdf_text", "pdf_scan", "image"]
Decision = Literal["approved", "review", "failed"]


def _clean_str(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text[:500] if text and text.lower() not in {"null", "none", "n/a"} else None


def _date_str(value: object) -> str | None:
    """ISO when parseable, otherwise the printed text so the date rule can report it."""
    text = _clean_str(value)
    parsed = parse_date(text)
    return parsed.isoformat() if parsed else text


def _rate(value: object) -> Decimal | None:
    return parse_decimal(str(value).replace("%", "")) if value is not None else None


def _plain(value: Decimal) -> str:
    return format(value, "f")


Text = Annotated[str | None, BeforeValidator(_clean_str)]
DateText = Annotated[str | None, BeforeValidator(_date_str)]
Currency = Annotated[str | None, BeforeValidator(normalize_currency)]
Amount = Annotated[
    Decimal | None,
    BeforeValidator(parse_decimal),
    PlainSerializer(
        lambda v: None if v is None else _plain(v), return_type=str | None, when_used="json"
    ),
]
Rate = Annotated[
    Decimal | None,
    BeforeValidator(_rate),
    PlainSerializer(
        lambda v: None if v is None else _plain(v), return_type=str | None, when_used="json"
    ),
]


class Vendor(BaseModel):
    name: Text = None
    address: Text = None
    tax_id: Text = None


class InvoiceInfo(BaseModel):
    number: Text = None
    issue_date: DateText = None
    due_date: DateText = None
    currency: Currency = None
    po_number: Text = None


class Amounts(BaseModel):
    subtotal: Amount = None
    # Discount and shipping extend the v1 field list: without them every discounted but perfectly
    # valid invoice fails "subtotal + tax = total" and is sent to review.
    discount: Amount = None
    shipping: Amount = None
    tax_amount: Amount = None
    tax_rate: Rate = None
    total: Amount = None


class LineItem(BaseModel):
    description: Text = None
    quantity: Amount = None
    unit_price: Amount = None
    line_total: Amount = None


class InvoiceFields(BaseModel):
    document_type: Literal["invoice", "receipt"] = "invoice"
    vendor: Vendor = Field(default_factory=Vendor)
    invoice: InvoiceInfo = Field(default_factory=InvoiceInfo)
    amounts: Amounts = Field(default_factory=Amounts)
    line_items: list[LineItem] = Field(default_factory=list)

    def get(self, path: str) -> Any:
        """Read "vendor.name" or "line_items.0.quantity"."""
        node: Any = self
        for part in path.split("."):
            if isinstance(node, list):
                index = int(part)
                if index >= len(node):
                    return None
                node = node[index]
            else:
                node = getattr(node, part)
        return node

    def with_values(self, values: dict[str, Any]) -> "InvoiceFields":
        """Copy with some paths replaced, re-validated through the schema."""
        data = self.model_dump()
        for path, value in values.items():
            parts = path.split(".")
            node: Any = data
            for part in parts[:-1]:
                node = node[int(part)] if isinstance(node, list) else node[part]
            leaf = parts[-1]
            if isinstance(node, list):
                node[int(leaf)] = value
            elif leaf not in node:
                raise KeyError(path)
            else:
                node[leaf] = value
        return InvoiceFields.model_validate(data)


HEADER_PATHS: tuple[str, ...] = (
    "vendor.name", "vendor.address", "vendor.tax_id",
    "invoice.number", "invoice.issue_date", "invoice.due_date", "invoice.currency",
    "invoice.po_number",
    "amounts.subtotal", "amounts.discount", "amounts.shipping", "amounts.tax_amount",
    "amounts.tax_rate", "amounts.total",
)  # fmt: skip
LINE_LEAVES: tuple[str, ...] = ("description", "quantity", "unit_price", "line_total")

REQUIRED_HEADER: dict[str, tuple[str, ...]] = {
    "invoice": (
        "vendor.name", "invoice.number", "invoice.issue_date", "invoice.currency",
        "amounts.subtotal", "amounts.total",
    ),
    # A till receipt usually prints no invoice number, currency code or subtotal. Holding it to the
    # invoice list would send every receipt to review.
    "receipt": ("vendor.name", "invoice.issue_date", "amounts.total"),
}  # fmt: skip


def required_paths(fields: InvoiceFields) -> list[str]:
    paths = list(REQUIRED_HEADER[fields.document_type])
    leaves = LINE_LEAVES if fields.document_type == "invoice" else ("description", "line_total")
    for i in range(len(fields.line_items)):
        paths += [f"line_items.{i}.{leaf}" for leaf in leaves]
    return paths


def all_paths(fields: InvoiceFields) -> list[str]:
    paths = list(HEADER_PATHS)
    for i in range(len(fields.line_items)):
        paths += [f"line_items.{i}.{leaf}" for leaf in LINE_LEAVES]
    return paths


class CheckResult(BaseModel):
    rule: str
    status: Literal["pass", "warn", "fail"]
    detail: str
    # Fields implicated in a failure; empty means there is nothing specific to re-ask.
    fields: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status != "fail"


class LLMCall(BaseModel):
    node: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int


class NodeMetric(BaseModel):
    node: str
    attempt: int
    latency_ms: int


class RunMetrics(BaseModel):
    nodes: list[NodeMetric] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)

    @property
    def tokens_in(self) -> int:
        return sum(c.tokens_in for c in self.llm_calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.tokens_out for c in self.llm_calls)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def latency_ms(self) -> int:
        return sum(n.latency_ms for n in self.nodes)


class RepairRecord(BaseModel):
    attempt: int
    rules: list[str]
    fields: list[str]
    changed: list[str]


class Correction(BaseModel):
    path: str
    before: str | None
    after: str | None


class InvoiceState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=False)

    run_id: str
    file_kind: FileKind | None = None
    page_count: int = 0
    prompt_version: str = ""

    fields: InvoiceFields | None = None
    uncertain: dict[str, float] = Field(default_factory=dict)  # the reader's own doubts
    suspicious_text: list[str] = Field(default_factory=list)
    schema_error: str | None = None
    reader: str | None = None  # "provider:model" that produced `fields`

    second_fields: InvoiceFields | None = None
    second_status: Literal["pending", "done", "skipped", "unavailable"] = "pending"
    second_paths: list[str] | None = None  # None: every field was cross-checked
    second_detail: str = ""

    confidence: dict[str, float] = Field(default_factory=dict)
    disagreements: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)

    repair_attempts: int = 0
    repair_stalled: bool = False
    repairs: list[RepairRecord] = Field(default_factory=list)

    decision: Decision | None = None
    review_reasons: list[str] = Field(default_factory=list)
    reviewed_by_human: bool = False
    corrections: list[Correction] = Field(default_factory=list)
    overrides: list[str] = Field(default_factory=list)
    error: str | None = None

    path: list[str] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
