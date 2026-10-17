"""Claude API prices (USD per 1M tokens) and cost calculation from response usage."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute TTL write
    # Haiku 5.5 bills prompts longer than this at a second rate card.
    long_prompt_threshold: int | None = None
    long_input: float = 0.0
    long_output: float = 0.0


# First-party Claude API list prices. Override in config if Anthropic changes them.
PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(input=10.0, output=50.0, cache_read=0.25, cache_write=12.5),
    "claude-opus-5-5": ModelPrice(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0),
    "claude-opus-5": ModelPrice(input=5.0, output=25.0, cache_read=0.50, cache_write=6.25),
    "claude-opus-4-8": ModelPrice(input=5.0, output=25.0, cache_read=0.50, cache_write=6.25),
    "claude-sonnet-5-5": ModelPrice(input=2.0, output=10.0, cache_read=0.20, cache_write=2.5),
    "claude-sonnet-5": ModelPrice(input=2.0, output=10.0, cache_read=0.20, cache_write=2.5),
    "claude-haiku-5-5": ModelPrice(
        input=0.10,
        output=0.50,
        cache_read=0.01,
        cache_write=0.125,
        long_prompt_threshold=100_000,
        long_input=0.50,
        long_output=2.50,
    ),
}

# Unknown model ids are priced like the most expensive known model, so budget checks err safe.
_FALLBACK = max(PRICES.values(), key=lambda p: p.output)


def price_for(model: str) -> ModelPrice:
    for known, price in PRICES.items():
        if model == known or model.startswith(known + "-") or model.endswith("." + known):
            return price
    log.warning("no price for model %r, using the most expensive known rate", model)
    return _FALLBACK


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    p = price_for(model)
    prompt = input_tokens + cache_write_tokens + cache_read_tokens
    in_rate, out_rate = p.input, p.output
    cache_read, cache_write = p.cache_read, p.cache_write
    if p.long_prompt_threshold is not None and prompt > p.long_prompt_threshold:
        scale = p.long_input / p.input
        in_rate, out_rate = p.long_input, p.long_output
        cache_read, cache_write = cache_read * scale, cache_write * scale
    total = (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_write_tokens * cache_write
        + cache_read_tokens * cache_read
    )
    return total / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough upper-bound token estimate for budget pre-checks (Cyrillic tokenizes densely)."""
    return len(text) // 2 + 1
