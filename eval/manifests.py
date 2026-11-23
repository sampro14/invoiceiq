import hashlib
import json
from pathlib import Path
from typing import Any

DIR = Path(__file__).parent / "datasets" / "manifests"
SETS = {
    "smoke": ["smoke"],
    "dev": ["dev"],
    "heldout": ["heldout"],
    "adversarial": ["adversarial"],
    "full": ["heldout", "adversarial"],
}


def read(name: str) -> list[dict[str, Any]]:
    path = DIR / f"{name}.json"
    if name == "heldout":
        expected = (DIR / "heldout.lock").read_text().strip()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("heldout.json does not match heldout.lock")
    entries: list[dict[str, Any]] = json.loads(path.read_text())
    return entries


def entries_for(set_name: str) -> list[dict[str, Any]]:
    return [e for part in SETS[set_name] for e in read(part)]
