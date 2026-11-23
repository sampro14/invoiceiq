"""Build the public site: a static landing page that wakes the app, and the eval report.

python -m eval.report --results eval/results --out _site --app-url https://…
"""

import argparse
import json
import shutil
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
SITE = ROOT / "site"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _f3(v: float) -> str:
    return f"{v:.3f}"


def _ms(v: float) -> str:
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


ROWS: list[tuple[str, str, Any]] = [
    ("field_f1", "Field-level F1", _f3),
    ("doc_exact", "Documents fully correct", _pct),
    ("line_item_f1", "Line-item F1", _f3),
    ("approved_exact", "Approved documents that were fully correct", _pct),
    ("review_rate", "Sent to a person", _pct),
    ("catch_rate", "Planted errors caught", _pct),
    ("false_flag_rate", "Clean documents wrongly sent to review", _pct),
    ("ece", "Calibration error (ECE)", _f3),
    ("injection_changed", "Injections that changed the output", lambda v: f"{v:.0f}"),
    ("injection_flagged", "Injections flagged", _pct),
    ("latency_p50_ms", "Latency, median", _ms),
    ("latency_p95_ms", "Latency, p95", _ms),
    ("cost_per_doc_usd", "Cost per document, list price", lambda v: f"${v:.5f}"),
]
NAMES = {"sroie": "SROIE receipts", "cord": "CORD receipts", "fatura": "FATURA invoices", "gst": "Synthetic GST invoices",
         "adversarial": "Adversarial"}  # fmt: skip


def _cell(values: dict[str, Any], key: str, fmt: Any) -> str:
    v = values.get(key)
    return "n/a" if v is None else str(fmt(v))


def load_reports(folder: Path) -> list[dict[str, Any]]:
    reports = []
    for path in folder.glob("*.json"):
        if path.name.startswith("latest"):
            continue
        reports.append(json.loads(path.read_text()))
    return sorted(reports, key=lambda r: r["created_at"])


def headline(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    live = [r for r in reports if r["mode"] == "live" and r["set"] in ("heldout", "full")]
    return live[-1] if live else None


def comparison_table(report: dict[str, Any]) -> str:
    agent = report["systems"].get("agent", {}).get("overall", {})
    base = report["systems"].get("baseline", {}).get("overall", {})
    body = "".join(
        f"<tr><td>{escape(name)}</td><td>{_cell(base, key, f)}</td><td><b>{_cell(agent, key, f)}</b></td>"
        f"<td>{escape(report['targets'].get(key, ''))}</td></tr>"
        for key, name, f in ROWS
    )
    return (
        '<div class="scroll"><table><thead><tr><th scope="col">Metric</th><th scope="col">Single prompt</th>'
        f'<th scope="col">Agent</th><th scope="col">Target</th></tr></thead><tbody>{body}</tbody></table></div>'
    )


def dataset_table(report: dict[str, Any]) -> str:
    agent = report["systems"].get("agent", {}).get("by_dataset", {})
    base = report["systems"].get("baseline", {}).get("by_dataset", {})
    body = "".join(
        f"<tr><td>{escape(NAMES.get(name, name))}</td><td>{a['n']}</td><td>{_cell(base.get(name, {}), 'field_f1', _f3)}</td>"
        f"<td><b>{_cell(a, 'field_f1', _f3)}</b></td><td>{_cell(a, 'doc_exact', _pct)}</td><td>{_cell(a, 'line_item_f1', _f3)}</td>"
        f"<td>{a['failed']}</td></tr>"
        for name, a in agent.items()
    )
    return (
        '<div class="scroll"><table><thead><tr><th scope="col">Dataset</th><th scope="col">Docs</th>'
        '<th scope="col">Single prompt F1</th><th scope="col">Agent F1</th><th scope="col">Fully correct</th>'
        f'<th scope="col">Line F1</th><th scope="col">Failed</th></tr></thead><tbody>{body}</tbody></table></div>'
    )


def bars(report: dict[str, Any]) -> str:
    """Agent against baseline on the three accuracy measures. One axis, 0 to 1."""
    agent = report["systems"].get("agent", {}).get("overall", {})
    base = report["systems"].get("baseline", {}).get("overall", {})
    keys = [
        ("field_f1", "Field F1"),
        ("doc_exact", "Fully correct"),
        ("line_item_f1", "Line-item F1"),
    ]
    left, width, band, thick = 112, 420, 62, 18
    parts = []
    for i, (key, name) in enumerate(keys):
        y = i * band + 8
        parts.append(
            f'<text x="{left - 12}" y="{y + thick + 5}" class="lab" text-anchor="end">{name}</text>'
        )
        for j, (values, cls, who) in enumerate(
            ((agent, "s1", "Agent"), (base, "s2", "Single prompt"))
        ):
            v = values.get(key)
            if v is None:
                continue
            w, top = max(v * width, 1), y + j * (thick + 2)
            r = min(4, w)
            parts.append(
                f'<path class="{cls}" tabindex="0" data-tip="{who}, {name}: {v:.3f}" '
                f'd="M{left} {top}h{w - r:.1f}a{r} {r} 0 0 1 {r} {r}v{thick - 2 * r}a{r} {r} 0 0 1 -{r} {r}h-{w - r:.1f}z"/>'
                f'<text x="{left + w + 8:.1f}" y="{top + 13}" class="val">{v:.3f}</text>'
            )
    height = len(keys) * band + 4
    return (
        '<p class="legend"><span class="k1"></span>Agent <span class="k2"></span>Single prompt</p>'
        f'<svg class="chart" viewBox="0 0 600 {height}" role="img" aria-label="Agent and single prompt accuracy; values are in the table">'
        f'<line x1="{left}" x2="{left}" y1="0" y2="{height}" class="axis"/>{"".join(parts)}</svg>'
    )


def history(reports: list[dict[str, Any]], mode: str) -> str:
    runs = [
        r for r in reports if r["set"] == "smoke" and r["mode"] == mode and "agent" in r["systems"]
    ][-40:]
    if len(runs) < 2:
        return '<p class="note">The history chart appears after two smoke runs on main.</p>'
    left, right, top, bottom, w, h = 44, 120, 12, 28, 720, 240
    x = lambda i: left + i * (w - left - right) / (len(runs) - 1)  # noqa: E731
    y = lambda v: top + (1 - v) * (h - top - bottom)  # noqa: E731
    parts = []
    for tick in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line x1="{left}" x2="{w - right}" y1="{y(tick)}" y2="{y(tick)}" class="grid"/>'
            f'<text x="{left - 8}" y="{y(tick) + 4}" class="lab" text-anchor="end">{tick:.2f}</text>'
        )
    series = [("field_f1", "Field F1", "l1", "s1"), ("doc_exact", "Fully correct", "l2", "s2")]
    ends = []
    for key, name, line, dot in series:
        points = [
            (x(i), y(v), v, r)
            for i, r in enumerate(runs)
            if (v := r["systems"]["agent"]["overall"][key]) is not None
        ]
        if not points:
            continue
        parts.append(
            f'<polyline class="{line}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, _, _ in points)}"/>'
        )
        for a, b, v, r in points:
            tip = f"{name}: {v:.3f}, {r['created_at'][:10]}, commit {r['git_sha'][:7]}"
            parts.append(
                f'<circle class="{dot} ring" cx="{a:.1f}" cy="{b:.1f}" r="4" tabindex="0" data-tip="{escape(tip)}"/>'
            )
        ends.append((points[-1][1], name))
    # Direct end labels only when they do not collide; the legend always names both.
    if len(ends) < 2 or abs(ends[0][0] - ends[1][0]) > 16:
        parts += [
            f'<text x="{w - right + 12}" y="{b + 4:.1f}" class="val">{n}</text>' for b, n in ends
        ]
    return (
        '<p class="legend"><span class="k1"></span>Field F1 <span class="k2"></span>Documents fully correct</p>'
        f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Smoke-set accuracy on main over time; values are in the table">{"".join(parts)}</svg>'
        + history_table(runs)
    )


def history_table(runs: list[dict[str, Any]]) -> str:
    body = "".join(
        f"<tr><td>{r['created_at'][:10]}</td><td>{escape(r['git_sha'][:7])}</td>"
        f"<td>{_cell(r['systems']['agent']['overall'], 'field_f1', _f3)}</td>"
        f"<td>{_cell(r['systems']['agent']['overall'], 'doc_exact', _pct)}</td>"
        f"<td>{_cell(r['systems']['agent']['overall'], 'latency_p95_ms', _ms)}</td></tr>"
        for r in reversed(runs)
    )
    return (
        '<details><summary>Every run</summary><div class="scroll"><table><thead><tr><th scope="col">Date</th>'
        '<th scope="col">Commit</th><th scope="col">Field F1</th><th scope="col">Fully correct</th>'
        f'<th scope="col">p95</th></tr></thead><tbody>{body}</tbody></table></div></details>'
    )


def render_eval(reports: list[dict[str, Any]]) -> str:
    best = headline(reports)
    if best:
        models = ", ".join(
            f"{escape(k.replace('_', ' '))} {escape(v)}" for k, v in best["models"].items()
        )
        main = (
            f'<p class="note">{best["n_docs"]} documents, {escape(best["set"])} set, {best["created_at"][:10]}, commit '
            f"{escape(best['git_sha'][:7])}. Models: {models}.</p>"
            f"{bars(best)}{comparison_table(best)}<h2>By dataset</h2>{dataset_table(best)}"
        )
    else:
        main = (
            '<p class="empty">The held-out evaluation has not been run with live models yet. '
            "Numbers appear here only after it has.</p>"
        )
    modes = [
        m
        for m in ("live", "offline")
        if any(r["set"] == "smoke" and r["mode"] == m for r in reports)
    ]
    if not modes:
        ci = '<p class="note">No smoke run has been recorded yet.</p>'
    else:
        mode = modes[0]
        warning = (
            ""
            if mode == "live"
            else '<p class="empty">These runs used rule-based stand-ins, because CI has no model keys. '
            "They show that the pipeline works, not how accurate the models are.</p>"
        )
        ci = warning + history(reports, mode)
    return (SITE / "eval.html").read_text().replace("<!--MAIN-->", main).replace("<!--CI-->", ci)


def build(results: Path, out: Path, app_url: str) -> None:
    reports = load_reports(results)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("site.css", "site.js", "wake.js"):
        shutil.copy(SITE / name, out / name)
    (out / "index.html").write_text((SITE / "index.html").read_text().replace("__APP_URL__", escape(app_url.rstrip("/"), quote=True)))  # fmt: skip
    (out / "eval.html").write_text(render_eval(reports))
    stored = out / "results"
    stored.mkdir(exist_ok=True)
    for r in reports:
        (stored / f"{r['git_sha']}-{r['set']}-{r['mode']}.json").write_text(
            json.dumps(r, indent=1) + "\n"
        )
    for mode in ("live", "offline"):
        smoke = [r for r in reports if r["set"] == "smoke" and r["mode"] == mode]
        if smoke:
            (stored / f"latest-smoke-{mode}.json").write_text(
                json.dumps(smoke[-1], indent=1) + "\n"
            )
    if best := headline(reports):
        (stored / "latest.json").write_text(json.dumps(best, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, default=ROOT / "eval" / "results")
    p.add_argument("--out", type=Path, default=ROOT / "_site")
    p.add_argument("--app-url", default="http://localhost:8001")
    args = p.parse_args()
    build(args.results, args.out, args.app_url)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
