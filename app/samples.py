import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DIR = Path(__file__).parent / "samples"


@dataclass(frozen=True)
class SampleFile:
    id: str
    title: str
    blurb: str
    path: Path

    @property
    def kind(self) -> str:
        return "pdf" if self.path.suffix == ".pdf" else "image"

    @property
    def preview(self) -> Path:
        return DIR / f"{self.id}.preview.png"

    @property
    def cache(self) -> Path:
        return DIR / "cache" / f"{self.id}.json"

    def cached_run(self) -> dict[str, Any] | None:
        """A recorded live run, replayed when the quota is gone. Written by scripts.cache_samples."""
        if not self.cache.exists():
            return None
        data: dict[str, Any] = json.loads(self.cache.read_text())
        return data


@lru_cache
def all_samples() -> dict[str, SampleFile]:
    entries = json.loads((DIR / "index.json").read_text())
    return {e["id"]: SampleFile(e["id"], e["title"], e["blurb"], DIR / e["file"]) for e in entries}
