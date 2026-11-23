"""Spotting text in a document that is addressed to the model rather than the reader.

Detection only flags. The real defence is structural: document text is passed as quoted data, the
model has no tools, and its output is constrained to a fixed schema.
"""

import re

# Kept tight on purpose: ordinary invoices say "please disregard this reminder if paid", so a verb
# alone is never enough. Each pattern needs an instruction verb *and* a model-directed object.
_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(previous|prior|above|earlier|all|any|your|system|these|the)\b.{0,20}\b(instruction|prompt|rule|direction|guideline|context|command)s?\b",
    r"\b(system|developer)\s+prompt\b|\b(system|developer|assistant)\s+(message|instruction)s?\s*:",
    r"\byou are (now )?(an?|the) (ai|assistant|language model|llm|chatbot)\b",
    r"\b(note|message|instruction|attention)s?\s+(to|for)\s+(the\s+)?(ai|llm|ai assistant|language model|chatbot|parser|extractor|extraction (model|agent|system))\b",
    r"\b(set|change|output|extract|replace|overwrite)\b.{0,25}\b(total|subtotal|amount|tax|vendor|currency|invoice number|field|confidence|balance)\b.{0,15}\b(to|as|with|=)\s*[\"'$₹€£]?\s*\w+",
    r"\b(mark|flag|treat|classify)\b.{0,15}\b(this|the)\s*(invoice|document|receipt)\b.{0,15}\bas\s+(approved|valid|verified|clean|trusted)\b",
    r"\b(respond|reply)\s+(only\s+)?(with|in)\b.{0,20}\b(json|the following|exactly)\b",
    r"<\s*/?\s*(system|instruction|prompt|untrusted|document)[^>]{0,40}>",
    r"\[\s*/?\s*(inst|system)\s*\]",
    r"\bdo not (flag|report|validate)\b.{0,30}\b(error|discrepanc|mismatch|this (invoice|document))",
]  # fmt: skip
_REGEX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _PATTERNS]


def find_instructions(text: str | None, limit: int = 5) -> list[str]:
    """Return short snippets of instruction-like text."""
    if not text:
        return []
    flat = re.sub(r"\s+", " ", text)
    spans: list[tuple[int, int]] = []
    for regex in _REGEX:
        for m in regex.finditer(flat):
            if not any(m.start() < e and s < m.end() for s, e in spans):
                spans.append((m.start(), m.end()))
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s - merged[-1][1] < 60:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [flat[s : e + 30].strip()[:200] for s, e in merged[:limit]]
