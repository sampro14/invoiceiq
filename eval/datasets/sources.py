"""Loading documents. Public sets are downloaded at a pinned revision into eval/datasets/cache
(git-ignored); images are never redistributed. Labels are derived here, not stored in manifests."""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

from agent.llm.router import system_ssl_context
from agent.normalize import parse_amount, parse_date
from eval.datasets import adversarial
from eval.datasets.synthetic import Spec, make_spec

CACHE = Path(__file__).parent / "cache"


@dataclass(frozen=True)
class Remote:
    repo: str
    revision: str
    file: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.file}"


REMOTES = {
    "cord": Remote("naver-clova-ix/cord-v2", "7f0115a4b758a71d6473b8d085751692da2fef98",
                   "data/test-00000-of-00001-9c204eb3f4e11791.parquet"),
    "sroie": Remote("jsdnrs/ICDAR2019-SROIE", "bffe40c26759f3376ec2b3ae9031dbba54cd587c",
                    "data/test-00000-of-00001.parquet"),
    "fatura-train": Remote("mathieu1256/FATURA2-invoices", "bcbb2fbb3c4701b87f5659ecbfbc55ad695aac21",
                           "data/train-00000-of-00001.parquet"),
    "fatura-test": Remote("mathieu1256/FATURA2-invoices", "bcbb2fbb3c4701b87f5659ecbfbc55ad695aac21",
                          "data/test-00000-of-00001.parquet"),
}  # fmt: skip


@dataclass
class EvalDoc:
    id: str
    dataset: str
    data: bytes
    labels: dict[str, Any]
    scored: list[str]  # fields this document has ground truth for
    required: list[str]  # the subset that decides "document fully correct"
    lock: str
    expect: dict[str, Any] = field(default_factory=dict)  # adversarial documents only
    clean: bool = False  # known to pass every rule, so sending it to review is a false flag


def fetch(name: str) -> Path:
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    print(f"downloading {name} from {REMOTES[name].repo} …", flush=True)
    with (
        httpx.Client(follow_redirects=True, timeout=600, verify=system_ssl_context()) as http,
        http.stream("GET", REMOTES[name].url) as r,
    ):
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    part.rename(path)
    return path


@lru_cache
def _parquet(name: str) -> pq.ParquetFile:
    return pq.ParquetFile(fetch(name))


def read_rows(name: str, rows: list[int]) -> dict[int, dict[str, Any]]:
    """Only the row groups that are needed, so the 290 MB file never sits in memory at once."""
    f = _parquet(name)
    wanted, out, start = sorted(rows), {}, 0
    for g in range(f.num_row_groups):
        n = f.metadata.row_group(g).num_rows
        inside = [r for r in wanted if start <= r < start + n]
        if inside:
            table = f.read_row_group(g)
            for r in inside:
                out[r] = {c: table.column(c)[r - start].as_py() for c in table.column_names}
        start += n
    return out


def index_column(name: str, column: str) -> list[Any]:
    f = _parquet(name)
    if column == "image.path":
        values: list[Any] = []
        for g in range(f.num_row_groups):
            # Struct leaves can be read without decoding the image bytes.
            table = f.read_row_group(g, columns=["image.path"])
            values += table.column("image").combine_chunks().field("path").to_pylist()
        return values
    return list(pq.read_table(fetch(name), columns=[column]).column(column).to_pylist())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Label extraction ----------------------------------------------------------------------------


def sroie_labels(row: dict[str, Any]) -> dict[str, Any]:
    e = row["entities"]
    return {
        "vendor.name": e["company"],
        "vendor.address": e["address"],
        "invoice.issue_date": e["date"],
        "amounts.total": e["total"],
    }


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _cord_amount(text: Any) -> str | None:
    d = parse_amount(_first(text))
    return None if d is None else str(abs(d))


def cord_labels(row: dict[str, Any]) -> dict[str, Any]:
    g = json.loads(row["ground_truth"])["gt_parse"]
    sub, total = g.get("sub_total", {}), g.get("total", {})
    labels: dict[str, Any] = {
        "amounts.subtotal": _cord_amount(sub.get("subtotal_price")),
        "amounts.tax_amount": _cord_amount(sub.get("tax_price")),
        "amounts.total": _cord_amount(total.get("total_price")),
    }
    menu = g.get("menu", [])
    menu = menu if isinstance(menu, list) else [menu]
    # Modifiers ("ADD CHICKEN BOX") are nested in CORD and flat in our schema; whether they are lines
    # of their own is a convention, so those receipts are not scored on lines.
    if menu and not any(isinstance(m, dict) and "sub" in m for m in menu):
        items = []
        for m in menu:
            quantity = re.search(r"\d+(?:[.,]\d+)?", str(_first(m.get("cnt")) or ""))
            items.append({
                "description": _first(m.get("nm")),
                "quantity": quantity.group(0).replace(",", ".") if quantity else None,
                "unit_price": _cord_amount(m.get("unitprice")),
                "line_total": _cord_amount(m.get("price")),
            })  # fmt: skip
        labels["line_items"] = items
    return {k: v for k, v in labels.items() if v is not None}


# FATURA2 tag ids, established by inspection (the dataset card does not list them).
_FATURA = {1: "total", 3: "date", 4: "due", 6: "seller", 12: "number", 13: "other"}


def fatura_labels(row: dict[str, Any]) -> dict[str, Any]:
    text: dict[str, list[str]] = defaultdict(list)
    for token, tag in zip(row["tokens"], row["ner_tags"], strict=True):
        if tag in _FATURA:
            text[_FATURA[tag]].append(token)
    t = {k: " ".join(v) for k, v in text.items()}
    labels: dict[str, Any] = {}
    if "seller" in t:
        labels["vendor.name"] = t["seller"]
    if m := re.search(r"(?:ID|#|number|no\.?)\s*(\S+)$", t.get("number", ""), re.IGNORECASE):
        labels["invoice.number"] = m.group(1)
    for key, path in (("date", "invoice.issue_date"), ("due", "invoice.due_date")):
        if (m := re.search(r"\d{1,2}-[A-Za-z]{3}-\d{4}", t.get(key, ""))) and parse_date(
            m.group(0)
        ):
            labels[path] = m.group(0)
    if m := re.search(r"([\d.]+)\s*(EUR|USD|\$|€|£)?\s*$", t.get("total", "")):
        labels["amounts.total"] = m.group(1)
        if m.group(2) in ("EUR", "USD"):
            labels["invoice.currency"] = m.group(2)
    other = t.get("other", "")
    for pattern, path in (
        (r"SUB_TOTAL\s*:?\s*([\d.]+)", "amounts.subtotal"),
        (r"TAX\s*:?\s*VAT\s*\([\d.]+%\)\s*:?\s*([\d.]+)", "amounts.tax_amount"),
        (r"DISCOUNT\s*\([\d.]+%\)\s*:?\s*\(-\)\s*([\d.]+)", "amounts.discount"),
        (r"PO Number\s*:?\s*(\d+)", "invoice.po_number"),
    ):
        if m := re.search(pattern, other):
            labels[path] = m.group(1)
    return labels


REQUIRED = {
    "sroie": ["vendor.name", "invoice.issue_date", "amounts.total"],
    "cord": ["amounts.total", "line_items"],
    "fatura": ["vendor.name", "invoice.number", "invoice.issue_date", "invoice.currency", "amounts.subtotal", "amounts.total"],
    "gst": ["vendor.name", "invoice.number", "invoice.issue_date", "invoice.currency", "amounts.subtotal",
            "amounts.total", "line_items"],
}  # fmt: skip
REQUIRED["adversarial"] = REQUIRED["gst"]


# --- Documents -----------------------------------------------------------------------------------


def gst_spec(i: int) -> Spec:
    """Seeds 1 to 50; every third one is a scan, so the set exercises both reading paths."""
    spec = make_spec(i)
    if i % 3 == 0:
        spec.kind = "image"
        spec.degrade = {"dpi": 150, "angle": (i % 5 - 2) * 0.6, "noise": 25}
    return spec


def _synthetic_doc(
    doc_id: str, dataset: str, spec: Spec, expect: dict[str, Any], clean: bool
) -> EvalDoc:
    labels = spec.labels()
    scored = [k for k in labels if k != "document_type"]
    return EvalDoc(doc_id, dataset, spec.render(), labels, scored, REQUIRED[dataset], spec.spec_hash(), expect, clean)  # fmt: skip


def load(entries: Iterable[dict[str, Any]]) -> list[EvalDoc]:
    """Load manifest entries, refusing any document that no longer matches its lock."""
    entries = list(entries)
    by_file: dict[str, list[int]] = defaultdict(list)
    for e in entries:
        if "file" in e:
            by_file[e["file"]].append(e["row"])
    rows = {name: read_rows(name, wanted) for name, wanted in by_file.items()}
    hostile = {doc_id: (spec, expect) for doc_id, spec, expect in adversarial.build()}
    docs = []
    for e in entries:
        doc = _load_one(e, rows, hostile)
        if e.get("lock") and doc.lock != e["lock"]:
            raise ValueError(f"{doc.id} does not match its manifest lock; the eval set has changed")
        docs.append(doc)
    return docs


def _load_one(e: dict[str, Any], rows: dict[str, Any], hostile: dict[str, Any]) -> EvalDoc:
    dataset = e["dataset"]
    if dataset == "gst":
        return _synthetic_doc(e["id"], "gst", gst_spec(e["seed"]), {}, clean=True)
    if dataset == "adversarial":
        spec, expect = hostile[e["id"]]
        return _synthetic_doc(e["id"], "adversarial", spec, expect, clean=False)
    row = rows[e["file"]][e["row"]]
    labels = {"sroie": sroie_labels, "cord": cord_labels, "fatura": fatura_labels}[dataset](row)
    data = row["image"]["bytes"]
    lock = sha(data + json.dumps(labels, sort_keys=True).encode())
    required = [p for p in REQUIRED[dataset] if p in labels]
    return EvalDoc(e["id"], dataset, data, labels, list(labels), required, lock)
