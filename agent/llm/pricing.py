"""List prices in USD per 1M tokens (input, output), logged even on the free tier.

Checked 2026-11-22 against ai.google.dev/gemini-api/docs/pricing and console.groq.com/docs/models.
Override with PRICE_OVERRIDES="model=in/out,..." rather than editing for a one-off.
"""

PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),  # introductory; 1.50/7.50 from 2027-01-01
    "gemini-4-hydrogen": (1.50, 7.50),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
    "offline-rules": (0.0, 0.0),
    "offline-anchors": (0.0, 0.0),
}


def parse_overrides(spec: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        model, _, prices = item.rpartition("=")
        price_in, _, price_out = prices.partition("/")
        out[model] = (float(price_in), float(price_out))
    return out


def cost_usd(
    model: str, tokens_in: int, tokens_out: int, overrides: dict[str, tuple[float, float]]
) -> float:
    """Unknown models raise: a silent zero would make the cost claim a lie."""
    price = overrides.get(model) or PRICES.get(model)
    if price is None:
        raise KeyError(f"No list price for {model!r}; set PRICE_OVERRIDES")
    return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
