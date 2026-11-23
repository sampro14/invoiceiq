"""Choose the splits and lock them. Run once: python -m eval.datasets.build_manifests

Re-running must reproduce the committed files byte for byte (there is a test for the synthetic part).
The 50 + 300 split follows the design: 100 SROIE, 100 CORD, 100 FATURA (2 per layout), 50 GST, with
the 30 adversarial documents kept as a set of their own. The smoke set comes from dev and adversarial
only, so CI never looks at a held-out document.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.datasets import adversarial
from eval.datasets.sources import index_column, load

OUT = Path(__file__).parent / "manifests"
DEV_SIZE = {"sroie": 14, "cord": 14, "fatura": 14, "gst": 8}
SMOKE_SIZE = 5
SMOKE_ADVERSARIAL = ["adv-degraded-00", "adv-error-00-total", "adv-error-09-tax_id", "adv-injection-00", "adv-injection-02"]  # fmt: skip


def order(key: str) -> str:
    """A fixed pseudo-random order that does not depend on file order."""
    return hashlib.sha256(f"invoiceiq-split-1:{key}".encode()).hexdigest()


def candidates() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    keys = index_column("sroie", "key")
    sroie: list[dict[str, Any]] = [
        {"id": f"sroie-{k}", "dataset": "sroie", "file": "sroie", "row": i}
        for i, k in enumerate(keys)
    ]
    out["sroie"] = sorted(sroie, key=lambda e: order(e["id"]))[:100]
    count = len(index_column("cord", "ground_truth"))
    out["cord"] = [
        {"id": f"cord-{i:03d}", "dataset": "cord", "file": "cord", "row": i} for i in range(count)
    ]
    per_layout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for file in ("fatura-train", "fatura-test"):
        for i, path in enumerate(index_column(file, "image.path")):
            m = re.match(r"(Template\d+)_(Instance\d+)", path)
            assert m
            entry = {"id": f"fatura-{m[1]}-{m[2]}", "dataset": "fatura", "file": file, "row": i}
            per_layout[m[1]].append(entry)
    out["fatura"] = [
        e
        for layout in sorted(per_layout)
        for e in sorted(per_layout[layout], key=lambda e: order(e["id"]))[:2]
    ]
    out["gst"] = [{"id": f"gst-{i:03d}", "dataset": "gst", "seed": i} for i in range(1, 51)]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    smoke: list[dict[str, Any]] = []
    for dataset, entries in candidates().items():
        shuffled = sorted(entries, key=lambda e: order("dev:" + e["id"]))
        n = DEV_SIZE[dataset]
        dev += shuffled[:n]
        smoke += shuffled[:SMOKE_SIZE]
        heldout += shuffled[n:]
    hostile = [{"id": doc_id, "dataset": "adversarial"} for doc_id, _, _ in adversarial.build()]
    smoke += [e for e in hostile if e["id"] in SMOKE_ADVERSARIAL]
    for name, entries in (
        ("dev", dev),
        ("heldout", heldout),
        ("smoke", smoke),
        ("adversarial", hostile),
    ):
        docs = load([{k: v for k, v in e.items()} for e in entries])
        locked = [{**e, "lock": d.lock} for e, d in zip(entries, docs, strict=True)]
        text = json.dumps(sorted(locked, key=lambda e: e["id"]), indent=1) + "\n"
        (OUT / f"{name}.json").write_text(text)
        print(name, len(entries))
    digest = hashlib.sha256((OUT / "heldout.json").read_bytes()).hexdigest()
    (OUT / "heldout.lock").write_text(digest + "\n")


if __name__ == "__main__":
    main()
