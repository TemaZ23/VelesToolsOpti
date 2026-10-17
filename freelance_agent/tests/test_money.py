from datetime import timedelta

import pytest

from freelance_agent import prefilter
from freelance_agent.agents.schemas import Assessment
from freelance_agent.budget import BudgetExceeded, BudgetGuard, format_report
from freelance_agent.config import Budget, Economics, Filters, Profile
from freelance_agent.economics import decide, platform_fee, win_probability
from freelance_agent.models import Job
from freelance_agent.pricing import cost_usd

from .conftest import ASSESSMENT, NOW, make_store

PROFILE = Profile(name="n", headline="h", about="a", skills=["Python"], min_project_usd=30)


def job(**kw) -> Job:
    base = dict(source="freelancer", external_id="1", url="u", title="Python scraper",
                description="Need a Python script that scrapes 500 products into CSV file.",
                budget_min=100, budget_max=250, currency="USD", usd_rate=1.0, bid_count=12,
                posted_at=NOW - timedelta(hours=1))
    base.update(kw)
    return Job(**base)


def assessment(**kw) -> Assessment:
    return Assessment.model_validate({**ASSESSMENT, **kw})


# ---- pricing -----------------------------------------------------------------------------

def test_cost_per_model():
    assert cost_usd("claude-opus-5-5", input_tokens=1_000_000, output_tokens=0) == pytest.approx(4.0)
    assert cost_usd("claude-opus-5-5", input_tokens=0, output_tokens=1_000_000) == pytest.approx(20.0)
    assert cost_usd("claude-haiku-5-5", input_tokens=2000, output_tokens=500) == pytest.approx(0.00045)
    # cache reads are billed at the cache rate
    assert cost_usd("claude-opus-5-5", input_tokens=0, output_tokens=0,
                    cache_read_tokens=1_000_000) == pytest.approx(0.20)


def test_haiku_long_prompt_rate_card():
    short = cost_usd("claude-haiku-5-5", input_tokens=100_000, output_tokens=1000)
    long = cost_usd("claude-haiku-5-5", input_tokens=100_001, output_tokens=1000)
    assert short == pytest.approx(0.0105)
    assert long == pytest.approx(100_001 * 0.5 / 1e6 + 1000 * 2.5 / 1e6)


def test_unknown_model_is_priced_conservatively():
    assert cost_usd("claude-future-9", input_tokens=0, output_tokens=1_000_000) >= 50.0


# ---- prefilter ---------------------------------------------------------------------------

def test_prefilter_stop_words_use_word_prefixes():
    f, e = Filters(), Economics()
    ok, why = prefilter.check(job(title="Написать отзывы о магазине"), f, e, NOW)
    assert not ok and "отзыв" in why
    # "exam" style needles must not hit longer unrelated words
    ok, _ = prefilter.check(job(title="Example dashboard in Python"), f, e, NOW)
    assert ok
    ok, why = prefilter.check(job(description="Need a voice over for my video, English native"), f, e, NOW)
    assert not ok


def test_prefilter_budget_bids_age_hourly():
    f, e = Filters(min_budget_usd=30, max_existing_bids=60, max_age_hours=48), Economics()
    assert not prefilter.check(job(budget_min=5, budget_max=10), f, e, NOW)[0]
    assert not prefilter.check(job(currency="RUB", usd_rate=None, budget_min=None, budget_max=1000), f, e, NOW)[0]
    assert prefilter.check(job(currency="RUB", usd_rate=None, budget_min=None, budget_max=10000), f, e, NOW)[0]
    assert not prefilter.check(job(bid_count=61), f, e, NOW)[0]
    assert not prefilter.check(job(posted_at=NOW - timedelta(hours=49)), f, e, NOW)[0]
    assert not prefilter.check(job(raw={"type": "hourly"}), f, e, NOW)[0]
    assert prefilter.check(job(budget_min=None, budget_max=None), f, e, NOW)[0]


def test_prefilter_include_keywords():
    f = Filters(include_keywords=["telegram"])
    assert not prefilter.check(job(), f, Economics(), NOW)[0]
    assert prefilter.check(job(title="Telegram bot"), f, Economics(), NOW)[0]


# ---- economics ---------------------------------------------------------------------------

def test_profitable_job_gets_a_bid():
    d = decide(job(), assessment(), PROFILE, Economics(), win_probability(0, 0, 12, Economics()))
    assert d.bid, d.reason
    assert d.price_usd == 120
    assert d.fee_usd == 12.0
    assert d.execution_budget_usd == pytest.approx(2.25)
    assert d.profit_if_won_usd == pytest.approx(120 - 12 - 2.25)
    assert d.expected_profit_usd > 0


@pytest.mark.parametrize("kw,reason", [
    ({"ethical": False}, "неэтичный"),
    ({"ai_deliverable": False}, "ИИ не может"),
    ({"red_flags": ["pay via telegram"]}, "мошенничества"),
])
def test_rejections_from_assessment(kw, reason):
    d = decide(job(), assessment(**kw), PROFILE, Economics(), 0.05)
    assert not d.bid and reason in d.reason


def test_budget_below_minimum_and_execution_too_expensive():
    d = decide(job(budget_min=10, budget_max=25), assessment(), PROFILE, Economics(), 0.05)
    assert not d.bid and "ниже вашего минимума" in d.reason
    d = decide(job(budget_min=30, budget_max=40), assessment(complexity="large"), PROFILE, Economics(), 0.05)
    assert not d.bid and "выполнение ИИ" in d.reason


def test_price_is_clamped_to_budget_and_converted_to_local_currency():
    rub = job(source="flru", currency="RUB", usd_rate=None, budget_min=None, budget_max=12000, bid_count=None)
    d = decide(rub, assessment(suggested_price_usd=400), PROFILE, Economics(), 0.05)
    assert d.bid and d.price_usd == 144.0 and d.bid_amount == 12000 and d.currency == "RUB"
    assert d.fee_usd == 0.0


def test_low_expected_value_is_not_worth_a_proposal():
    d = decide(job(), assessment(), PROFILE, Economics(), win_prob=0.001)
    assert not d.bid and "не окупает" in d.reason


def test_win_probability_learns_and_penalises_crowds():
    e = Economics()
    assert win_probability(0, 0, None, e) == pytest.approx(0.05)
    assert win_probability(40, 10, None, e) > 0.15
    assert win_probability(0, 0, 100, e) < win_probability(0, 0, 5, e)


def test_platform_fee_minimum():
    e = Economics()
    assert platform_fee(30, "freelancer", e) == 5.0
    assert platform_fee(200, "freelancer", e) == 20.0
    assert platform_fee(200, "unknown", e) == 0.0


# ---- budget guard ------------------------------------------------------------------------

def test_daily_limit_applies_to_finding_work_only():
    store = make_store()
    guard = BudgetGuard(store, Budget(daily_llm_limit_usd=1.0, monthly_llm_limit_usd=10.0))
    store.add_llm_call(purpose="triage", model="m", cost_usd=0.9)
    with pytest.raises(BudgetExceeded, match="дневной"):
        guard.check(0.2, "proposal")
    guard.check(5.0, "execute")  # paid work is not blocked by the daily acquisition limit
    store.add_llm_call(purpose="execute", model="m", cost_usd=8.5)
    with pytest.raises(BudgetExceeded, match="месячный"):
        guard.check(1.0, "execute")


def test_stop_loss_stops_bidding_until_money_comes_in():
    store = make_store()
    guard = BudgetGuard(store, Budget(stop_loss_usd=5.0))
    store.add_llm_call(purpose="triage", model="m", cost_usd=6.0)
    allowed, why = guard.bidding_allowed()
    assert not allowed and "стоп-лосс" in why
    store.add_ledger("revenue", 50.0)
    store.add_ledger("fee", 5.0)
    assert guard.bidding_allowed()[0]


def test_report_counts_subscription_in_break_even():
    store = make_store()
    guard = BudgetGuard(store, Budget(fixed_monthly_costs_usd={"claude": 20.0}))
    store.add_llm_call(purpose="triage", model="m", cost_usd=3.0)
    store.add_ledger("revenue", 100.0)
    store.add_ledger("fee", 10.0)
    r = guard.report()
    assert r.profit_month == pytest.approx(100 - 10 - 3 - 20)
    assert r.to_break_even == 0
    text = format_report(r)
    assert "ПРИБЫЛЬ" in text and "67.00" in text
