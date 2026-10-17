import asyncio

import pytest

from freelance_agent.agents.brain import Brain, _shorten
from freelance_agent.budget import BudgetExceeded, BudgetGuard
from freelance_agent.config import Budget
from freelance_agent.economics import decide
from freelance_agent.llm import LLM, LLMError
from freelance_agent.models import Job

from .conftest import FakeClaude, make_settings, make_store

JOB = Job(source="flru", external_id="1", url="https://www.fl.ru/projects/1/", title="Парсер",
          description="Нужен парсер каталога в CSV, 500 товаров.", budget_max=12000, currency="RUB")


def setup(tmp_path, fake: FakeClaude, budget: Budget | None = None):
    settings = make_settings(tmp_path)
    store = make_store()
    guard = BudgetGuard(store, budget or settings.budget)
    llm = LLM(fake.client(), store, guard, settings.models.server_side_fallback)
    return settings, store, Brain(llm, settings)


def test_triage_uses_haiku_low_effort_without_fallback(tmp_path, fake_claude):
    settings, store, brain = setup(tmp_path, fake_claude)
    job_id, _ = store.upsert_job(JOB)
    a = asyncio.run(brain.assess(JOB, job_id))
    assert a.fit_score == 85 and a.complexity == "small"
    req = fake_claude.requests[0]
    body = req["body"]
    assert body["model"] == "claude-haiku-5-5"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "fallbacks" not in body
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "<job_post>" in body["messages"][0]["content"]
    # 2000 in / 500 out on Haiku 5.5
    assert store.llm_spend() == pytest.approx(2000 * 0.1e-6 + 500 * 0.5e-6)


def test_proposal_uses_opus_with_server_side_fallback(tmp_path, fake_claude):
    settings, store, brain = setup(tmp_path, fake_claude)
    job_id, _ = store.upsert_job(JOB)
    a = asyncio.run(brain.assess(JOB, job_id))
    d = decide(JOB, a, settings.profile, settings.economics, 0.05)
    draft = asyncio.run(brain.write_proposal(JOB, job_id, a, d))
    assert draft.period_days == 2 and "CSV" in draft.text
    req = fake_claude.requests[1]
    assert req["body"]["model"] == "claude-opus-5-5"
    assert req["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["headers"]["anthropic-beta"]
    assert "Price to quote: 10000 RUB" in req["body"]["messages"][0]["content"]


def test_refusal_and_truncation_raise_but_are_still_billed(tmp_path, fake_claude):
    _, store, brain = setup(tmp_path, fake_claude)
    fake_claude.stop_reason = "refusal"
    with pytest.raises(LLMError, match="declined"):
        asyncio.run(brain.assess(JOB, 1))
    fake_claude.stop_reason = "max_tokens"
    with pytest.raises(LLMError, match="max_tokens"):
        asyncio.run(brain.assess(JOB, 1))
    assert store.llm_spend() > 0


def test_budget_is_checked_before_any_request(tmp_path, fake_claude):
    _, _, brain = setup(tmp_path, fake_claude, Budget(daily_llm_limit_usd=0.0001, monthly_llm_limit_usd=1))
    with pytest.raises(BudgetExceeded):
        asyncio.run(brain.assess(JOB, 1))
    assert fake_claude.requests == []


def test_long_proposals_are_cut_on_a_sentence_boundary():
    text = "Первое предложение. " * 120
    short = _shorten(text, 1500)
    assert len(short) <= 1500 and short.endswith(".")


def test_auth_and_outage_errors_are_not_blamed_on_the_job(tmp_path, fake_claude):
    import httpx2 as httpx

    from freelance_agent.llm import LLMUnavailable

    def unauthorized(request):
        return httpx.Response(401, json={"type": "error", "error": {
            "type": "authentication_error", "message": "invalid x-api-key"}})

    fake_claude.handler = unauthorized  # before the client is built: the transport binds it
    _, store, brain = setup(tmp_path, fake_claude)
    with pytest.raises(LLMUnavailable, match="401"):
        asyncio.run(brain.assess(JOB, 1))
    assert store.llm_spend() == 0


def test_unscoped_key_is_a_setup_problem_not_a_job_failure(tmp_path, fake_claude):
    import httpx2 as httpx

    from freelance_agent.llm import LLMUnavailable

    def not_scoped(request):
        return httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "This API key is not scoped to a workspace, so this request must include "
                       "the anthropic-workspace-id header with the ID of the workspace to use."}})

    fake_claude.handler = not_scoped
    _, _, brain = setup(tmp_path, fake_claude)
    with pytest.raises(LLMUnavailable, match="workspace"):
        asyncio.run(brain.assess(JOB, 1))
