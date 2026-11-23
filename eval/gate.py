"""Fail a pull request that makes the agent worse than the last run on main.

python -m eval.gate --current smoke.json --main previous.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

F1_DROP = 0.02
EXACT_DROP = 0.03
P95_RISE = 0.25
# Below this a 25% rise is timer noise, not a regression (offline runs take a few milliseconds).
P95_MIN_RISE_MS = 250


def compare(current: dict[str, Any], main: dict[str, Any]) -> list[str]:
    for key in ("mode", "set"):
        if current[key] != main[key]:
            raise ValueError(
                f"cannot compare: {key} is {current[key]} here and {main[key]} on main"
            )
    now, before = current["systems"]["agent"]["overall"], main["systems"]["agent"]["overall"]
    problems = []

    def dropped(key: str, limit: float, name: str) -> None:
        if before[key] is None:
            return
        if now[key] is None or before[key] - now[key] > limit + 1e-9:
            problems.append(
                f"{name} fell from {before[key]:.3f} to {now[key] if now[key] is None else round(now[key], 3)}"
            )

    dropped("field_f1", F1_DROP, "Field F1")
    dropped("doc_exact", EXACT_DROP, "Document exact match")
    a, b = before["latency_p95_ms"], now["latency_p95_ms"]
    if a is not None and b is not None and b > a * (1 + P95_RISE) and b - a > P95_MIN_RISE_MS:
        problems.append(f"p95 latency rose from {a:.0f} ms to {b:.0f} ms")
    if (now.get("injection_changed") or 0) > (before.get("injection_changed") or 0):
        problems.append(f"{now['injection_changed']} injection(s) now change the output")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--main", type=Path, required=True)
    args = p.parse_args(argv)
    current = json.loads(args.current.read_text())
    if not args.main.exists():
        print("No earlier run on main to compare with; this run becomes the baseline.")
        return 0
    previous = json.loads(args.main.read_text())
    if previous.get("manifest_sha256") != current.get("manifest_sha256"):
        print(
            "The smoke set itself changed, so scores are not comparable; this run becomes the baseline."
        )
        return 0
    problems = compare(current, previous)
    if problems:
        print("Eval gate FAILED:\n" + "\n".join(f"  - {x}" for x in problems))
        return 1
    print("Eval gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
