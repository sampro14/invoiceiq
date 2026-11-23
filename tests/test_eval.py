import json
from pathlib import Path

import pytest

from agent.llm.schema import Wire, wire_to_fields
from agent.state import all_paths
from eval import metrics, report
from eval.datasets import adversarial
from eval.datasets.sources import EvalDoc, cord_labels, fatura_labels, gst_spec, load
from eval.datasets.synthetic import gstin, make_spec, with_error
from eval.gate import compare
from eval.gate import main as gate_main
from eval.judge import agreement
from eval.manifests import DIR, entries_for, read
from tests.conftest import good_wire

ROOT = Path(__file__).parent.parent


def predicted(**over):
    return wire_to_fields(Wire.model_validate(good_wire(**over))).model_dump(mode="json")


def truth(**over):
    w = good_wire(**over)
    return {
        "vendor.name": w["vendor_name"], "invoice.issue_date": "1 Mar 2026", "amounts.total": "354",
        "line_items": w["line_items"],
    }  # fmt: skip


SCORED = ["vendor.name", "invoice.issue_date", "amounts.total", "line_items"]


def score(p, labels=None, scored=SCORED):
    confidence = {}
    if p:
        confidence = dict.fromkeys(all_paths(wire_to_fields(Wire.model_validate(good_wire()))), 0.9)
    return metrics.score_document(
        p, confidence, labels or truth(), scored, ["vendor.name", "amounts.total", "line_items"]
    )


def test_perfect_prediction():
    s = score(predicted())
    assert s.header.f1 == 1 and s.lines.f1 == 1 and s.exact and s.wrong == []


def test_normalisation_is_applied_before_comparing():
    assert score(predicted(vendor_name="  DECCAN fasteners, LLP")).exact
    assert metrics.matches("amounts.total", "354.0", "354")
    assert not metrics.matches("amounts.total", "354.01", "354")
    assert metrics.matches("invoice.number", "INV/05-24/467", "inv 05-24/467")
    assert not metrics.matches("vendor.name", "Deccan Fasteners LLC", "Deccan Fasteners LLP")


def test_one_wrong_field():
    s = score(predicted(total="345.00"))
    assert s.header.f1 == pytest.approx(2 / 3) and not s.exact and s.wrong == ["amounts.total"]


def test_missing_prediction_costs_recall_and_invented_one_costs_precision():
    s = score(predicted(vendor_name=None))
    assert (s.header.correct, s.header.predicted, s.header.truth) == (2, 2, 3)
    s = metrics.score_document(predicted(), {}, {"amounts.total": "354", "invoice.po_number": None}, ["amounts.total", "invoice.po_number"], [])  # fmt: skip
    assert s.header.f1 == 1
    s = metrics.score_document(predicted(po_number="PO-1"), {}, {"amounts.total": "354", "invoice.po_number": None}, ["amounts.total", "invoice.po_number"], [])  # fmt: skip
    assert (s.header.correct, s.header.predicted, s.header.truth) == (1, 2, 1)


def test_unlabelled_fields_are_not_scored():
    s = metrics.score_document(
        predicted(), {}, {"amounts.total": "354"}, ["amounts.total"], ["amounts.total"]
    )
    assert s.header.predicted == 1


def test_failed_document_counts_as_wrong():
    s = score(None)
    assert s.header.f1 == 0 and s.lines.f1 == 0 and not s.exact


def test_lines_are_matched_not_compared_by_position():
    items = good_wire()["line_items"]
    s = score(predicted(line_items=items[::-1]))
    assert s.lines.f1 == 1 and s.exact
    s = score(predicted(line_items=items[:1]))
    assert (s.lines.correct, s.lines.predicted, s.lines.truth) == (4, 4, 8) and not s.exact
    extra = {"description": "Ghost", "quantity": "9", "unit_price": "9.00", "line_total": "81.00"}
    s = score(predicted(line_items=[*items, extra]))
    assert (s.lines.correct, s.lines.predicted, s.lines.truth) == (8, 12, 8) and not s.exact


def test_blank_annotations_on_a_matched_line_are_not_penalised():
    labels = truth()
    labels["line_items"] = [{**row, "unit_price": None} for row in labels["line_items"]]
    s = score(predicted(), labels)
    assert s.lines.f1 == 1 and s.exact


def test_calibration_error():
    assert metrics.expected_calibration_error([]) is None
    assert metrics.expected_calibration_error([(1.0, True)] * 4) == 0
    assert metrics.expected_calibration_error([(0.9, True), (0.9, False)]) == pytest.approx(0.4)


def test_percentile():
    assert metrics.percentile([], 95) is None
    assert metrics.percentile(list(range(1, 101)), 95) == 95
    assert metrics.percentile([5.0], 95) == 5.0


# --- Data -----------------------------------------------------------------------------------------


def test_splits_have_the_designed_sizes_and_do_not_overlap():
    dev, heldout, smoke, hostile = (read(n) for n in ("dev", "heldout", "smoke", "adversarial"))
    assert [len(x) for x in (dev, heldout, smoke, hostile)] == [50, 300, 25, 30]
    ids = lambda x: {e["id"] for e in x}  # noqa: E731
    assert not ids(dev) & ids(heldout) and not ids(hostile) & ids(heldout)
    assert ids(smoke) <= ids(dev) | ids(hostile), "CI must never look at held-out documents"
    locks = [e["lock"] for e in dev + heldout + hostile]
    assert len(set(locks)) == len(locks), "the same document appears twice"


def test_heldout_is_hash_locked(tmp_path, monkeypatch):
    for name in ("heldout.json", "heldout.lock"):
        (tmp_path / name).write_text((DIR / name).read_text())
    (tmp_path / "heldout.json").write_text(
        (DIR / "heldout.json").read_text().replace("gst-", "gst_", 1)
    )
    monkeypatch.setattr("eval.manifests.DIR", tmp_path)
    with pytest.raises(ValueError, match=r"heldout\.lock"):
        read("heldout")


def test_generated_documents_still_match_their_locks():
    entries = [e for e in entries_for("full") if e["dataset"] in ("gst", "adversarial")]
    assert len(load(entries)) == len(entries)
    tampered = [{**entries[0], "lock": "0" * 64}]
    with pytest.raises(ValueError, match="does not match"):
        load(tampered)


def test_no_heldout_value_leaks_into_the_prompts():
    prompts = "\n".join(p.read_text() for p in (ROOT / "agent" / "prompts").glob("*.md"))
    for e in read("heldout"):
        if e["dataset"] == "gst":
            spec = gst_spec(e["seed"])
            for value in (spec.vendor_name, spec.number, spec.tax_id):
                assert value not in prompts


def test_clean_synthetic_invoices_pass_every_rule():
    from agent.state import InvoiceFields
    from agent.validators import run_checks

    for seed in range(1, 51):
        labels = gst_spec(seed).labels()
        flat = {k: v for k, v in labels.items() if "." in k}
        fields = InvoiceFields(line_items=labels["line_items"]).with_values(flat)
        assert [c.rule for c in run_checks(fields) if c.status == "fail"] == [], seed


@pytest.mark.parametrize("error", sorted(set(adversarial.ERRORS)))
def test_each_planted_error_breaks_the_rule_it_claims(error):
    from agent.state import InvoiceFields
    from agent.validators import run_checks

    spec, rule = with_error(make_spec(9100), error)
    labels = spec.labels()
    fields = InvoiceFields(line_items=labels["line_items"]).with_values(
        {k: v for k, v in labels.items() if "." in k}
    )
    assert [c.rule for c in run_checks(fields) if c.status == "fail"] == [rule]


def test_generated_gstin_matches_a_published_example():
    """27AAPFU0939F1ZV is the checksum example used in GSTN documentation."""
    import random

    class Fixed(random.Random):
        picks = iter(["AAPFU", "0939"])

        def choices(self, population, k=1, **_):
            return list(next(self.picks))

        def choice(self, seq):
            return "1" if seq == "123" else "F"

    assert gstin(Fixed(), "27") == "27AAPFU0939F1ZV"


def test_cord_and_fatura_label_parsing():
    row = {"ground_truth": json.dumps({"gt_parse": {
        "menu": [{"nm": "TEA", "cnt": "2X", "price": "24.000"}, {"nm": "BUN", "price": "9,000"}],
        "sub_total": {"subtotal_price": "33.000", "discount_price": "-3.000"}, "total": {"total_price": "33.000"}}})}  # fmt: skip
    labels = cord_labels(row)
    assert labels["amounts.total"] == "33000" and labels["line_items"][0]["quantity"] == "2"
    assert labels["line_items"][1]["quantity"] is None
    nested = json.loads(row["ground_truth"])
    nested["gt_parse"]["menu"][0]["sub"] = {"nm": "SUGAR"}
    assert "line_items" not in cord_labels({"ground_truth": json.dumps(nested)})
    tokens = [
        "INVOICE",
        "#",
        "INV/30-14/832",
        "Due",
        "Date",
        "06-Feb-2007",
        "TOTAL",
        "441.14",
        "EUR",
        "SUB_TOTAL",
        "430.00",
    ]
    tags = [12, 12, 12, 4, 4, 4, 1, 1, 1, 13, 13]
    assert fatura_labels({"tokens": tokens, "ner_tags": tags}) == {
        "invoice.number": "INV/30-14/832", "invoice.due_date": "06-Feb-2007", "amounts.total": "441.14",
        "invoice.currency": "EUR", "amounts.subtotal": "430.00",
    }  # fmt: skip


# --- Runner, gate, report ---------------------------------------------------------------------------


def make_report(f1=0.95, exact=0.85, p95=8000.0, **over):
    overall = {k: 0.5 for k, _, _ in report.ROWS}
    overall |= {
        "field_f1": f1,
        "doc_exact": exact,
        "latency_p95_ms": p95,
        "injection_changed": 0,
        "n": 20,
        "failed": 0,
    }
    system = {"overall": overall, "by_dataset": {"gst": {**overall}}}
    return {"git_sha": "a" * 40, "created_at": "2026-11-23T00:00:00+00:00", "mode": "live", "set": "smoke", "n_docs": 25,
            "fingerprint": "x", "manifest_sha256": "m", "models": {"reader": "<b>m</b>"}, "targets": {"field_f1": "≥ 0.90"},
            "systems": {"agent": system, "baseline": system}, **over}  # fmt: skip


@pytest.mark.parametrize(
    ("now", "ok"),
    [
        ({}, True), ({"f1": 0.931}, True), ({"f1": 0.929}, False), ({"exact": 0.82}, True), ({"exact": 0.81}, False),
        ({"p95": 9900.0}, True), ({"p95": 10100.0}, False), ({"f1": None}, False),
    ],
)  # fmt: skip
def test_gate_thresholds(now, ok):
    assert (compare(make_report(**now), make_report()) == []) is ok


def test_gate_ignores_timer_noise_but_not_new_injections():
    assert compare(make_report(p95=20.0), make_report(p95=10.0)) == []
    worse = make_report()
    worse["systems"] = json.loads(json.dumps(worse["systems"]))
    worse["systems"]["agent"]["overall"]["injection_changed"] = 1
    assert compare(worse, make_report())


def test_gate_refuses_to_compare_offline_with_live(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        compare(make_report(mode="offline"), make_report())
    current = tmp_path / "c.json"
    current.write_text(json.dumps(make_report()))
    assert gate_main(["--current", str(current), "--main", str(tmp_path / "none.json")]) == 0


async def test_runner_resumes_and_scores(monkeypatch, tmp_path):
    from eval import run_eval
    from tests.conftest import make_settings

    calls = []

    async def fake(doc, deps):
        calls.append(doc.id)
        return {"fields": predicted(), "confidence": {}, "decision": "approved", "failed_rules": [],
                "latency_ms": 5.0, "cost_usd": 0.0, "tokens_in": 1, "tokens_out": 1}  # fmt: skip

    monkeypatch.setitem(run_eval.RUNNERS, "agent", fake)
    docs = [
        EvalDoc(f"d{i}", "gst", b"", truth(), SCORED, ["vendor.name"], "l", clean=True)
        for i in range(3)
    ]
    deps = type("D", (), {"settings": make_settings()})()
    raw = tmp_path / "r" / "agent.jsonl"
    await run_eval.predict(docs[:2], "agent", deps, raw, 0)
    records = await run_eval.predict(docs, "agent", deps, raw, 0)
    assert calls == ["d0", "d1", "d2"], "finished documents are not run again"
    overall = run_eval.aggregate(docs, records, "agent")["overall"]
    assert (
        overall["field_f1"] == 1 and overall["false_flag_rate"] == 0 and overall["doc_exact"] == 1
    )


def test_site_builds_with_and_without_results(tmp_path):
    results, out = tmp_path / "r", tmp_path / "o"
    results.mkdir()
    report.build(results, out, 'https://x.example/"><script>')
    page = (out / "index.html").read_text()
    assert "<script>" not in page.split("</head>")[1].replace('<script src="wake.js"></script>', "")
    assert "has not been run" in (out / "eval.html").read_text()
    assert not (out / "results" / "latest.json").exists()
    for i in range(3):
        r = make_report(git_sha=f"{i}" * 40, created_at=f"2026-11-2{i}T00:00:00+00:00")
        (results / f"{i}.json").write_text(json.dumps(r))
    (results / "h.json").write_text(json.dumps(make_report(set="full", git_sha="f" * 40)))
    report.build(results, out, "https://x.example")
    html = (out / "eval.html").read_text()
    assert "polyline" in html and "<b>m</b>" not in html
    assert json.loads((out / "results" / "latest.json").read_text())["set"] == "full"


def test_offline_results_never_become_the_headline():
    assert report.headline([make_report(set="full", mode="offline")]) is None
    assert report.headline([make_report(set="smoke")]) is None


def test_judge_agreement_needs_human_labels():
    rows = [{"judge": "1", "human": "", "string_metric": "1"}]
    assert agreement(rows)["judge"]["agreement"] is None
    rows = [
        {"judge": a, "human": b, "string_metric": b}
        for a, b in (("1", "1"), ("0", "0"), ("1", "0"), ("0", "0"))
    ]
    assert agreement(rows)["judge"]["agreement"] == 0.75
    assert agreement(rows)["string_metric"]["kappa"] == 1
