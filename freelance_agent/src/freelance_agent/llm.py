"""Claude API wrapper: structured outputs, spend accounting and budget pre-checks on every call."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from .budget import BudgetGuard
from .config import Effort
from .db import Store
from .pricing import cost_usd, estimate_tokens, price_for

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Models that accept the server-side refusal fallback (`fallbacks: "default"`) on the Claude API.
_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The model could not produce a usable answer for this particular input."""


class LLMUnavailable(LLMError):
    """Key, permissions, network or overload problem: nothing is wrong with the job itself."""


class LLM:
    def __init__(self, client: anthropic.AsyncAnthropic, store: Store, guard: BudgetGuard,
                 server_side_fallback: bool = True) -> None:
        self.client = client
        self.store = store
        self.guard = guard
        self.server_side_fallback = server_side_fallback

    async def structured(
        self,
        *,
        purpose: str,
        model: str,
        effort: Effort,
        system: str,
        user: str,
        schema: type[T],
        max_tokens: int,
        job_id: int | None = None,
    ) -> T:
        price = price_for(model)
        estimate = (
            estimate_tokens(system + user) * price.input + max_tokens * price.output
        ) / 1_000_000
        self.guard.check(estimate, purpose)

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            # Frozen per-purpose system prompt first so repeated calls hit the prompt cache.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort},
        )
        if self.server_side_fallback and model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        try:
            response = await self.client.beta.messages.parse(**kwargs)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.RateLimitError, anthropic.InternalServerError) as e:
            raise LLMUnavailable(f"Claude API {e.status_code}: {e.message}") from e
        except anthropic.APIStatusError as e:
            # Billing, overload and workspace misconfiguration are not the job's fault.
            if e.status_code >= 500 or e.status_code in (402, 529) or "workspace" in str(e.message):
                raise LLMUnavailable(f"Claude API {e.status_code}: {e.message}") from e
            raise LLMError(f"{purpose}: Claude API {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"Claude API unreachable: {e}") from e

        usage = response.usage
        served_by = response.model or model
        cost = cost_usd(
            served_by,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
        )
        self.store.add_llm_call(
            purpose=purpose,
            model=served_by,
            cost_usd=cost,
            job_id=job_id,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
        )
        log.info("%s via %s: $%.4f", purpose, served_by, cost)

        if response.stop_reason == "refusal":
            raise LLMError(f"{purpose}: Claude declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{purpose}: output hit max_tokens={max_tokens}")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"{purpose}: no structured output in response")
        return parsed
