"""Russia-friendly pieces: CBR rates, rubles, self-employed tax, Pchel.net, the setup wizard."""

import asyncio
import json
import stat

import httpx2 as httpx
import pytest
import yaml

from freelance_agent import fx
from freelance_agent.agents.schemas import Assessment
from freelance_agent.config import Economics, Profile, Settings
from freelance_agent.economics import decide, record_payment
from freelance_agent.models import Job
from freelance_agent.setup_wizard import Answers, render_config, run_setup, write_env
from freelance_agent.sources.rss import parse_feed

from .conftest import ASSESSMENT, fixture_bytes, make_store

PROFILE = Profile(name="n", headline="h", about="a", skills=["Python"], min_project_usd=30)


def test_cbr_rates_from_real_snapshot():
    rates = fx.parse_cbr(fixture_bytes("cbr_daily.xml"))
    assert rates["USD"] == 1.0
    assert rates["RUB"] == pytest.approx(1 / 83.6177)
    assert rates["EUR"] == pytest.approx(93.7187 / 83.6177)
    assert len(rates) > 40


def test_rates_refresh_once_a_day_and_survive_restart():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200, content=fixture_bytes("cbr_daily.xml"))

    async def run() -> None:
        store, econ = make_store(), Economics()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            assert await fx.refresh_daily(http, store, econ)
            assert not await fx.refresh_daily(http, store, econ)
        assert econ.fx_to_usd["RUB"] == pytest.approx(1 / 83.6177)
        fresh = Economics()
        fx.apply_cached(store, fresh)
        assert fresh.fx_to_usd["RUB"] == pytest.approx(1 / 83.6177)

    asyncio.run(run())
    assert len(calls) == 1


def test_rates_outage_keeps_previous_rates():
    async def run() -> float:
        econ = Economics()
        transport = httpx.MockTransport(lambda r: httpx.Response(503))
        async with httpx.AsyncClient(transport=transport) as http:
            assert not await fx.refresh_daily(http, make_store(), econ)
        return econ.fx_to_usd["RUB"]

    assert asyncio.run(run()) == 0.012


def test_pchel_real_feed_snapshot():
    jobs = parse_feed(fixture_bytes("pchel.xml"), "pchel")
    assert len(jobs) == 50
    assert all(j.url.startswith("https://pchel.net/jobs/") for j in jobs)
    assert all("<p>" not in j.description for j in jobs)
    assert len({j.external_id for j in jobs}) == 50


def test_budget_written_in_the_text_caps_the_price():
    job = Job(source="pchel", external_id="1", url="u", title="Дизайн-проект",
              description="Оплата 5 000 рублей за весь проект", currency=None)
    a = Assessment.model_validate({**ASSESSMENT, "suggested_price_usd": 300,
                                   "stated_budget_amount": 5000, "stated_budget_currency": "RUB"})
    d = decide(job, a, PROFILE, Economics(), 0.05)
    assert d.price_usd == pytest.approx(60.0)  # 5000 RUB * 0.012, not the $300 the model suggested


def test_self_employed_tax_reduces_profit_and_is_booked():
    econ = Economics(tax_percent=6)
    job = Job(source="flru", external_id="1", url="u", title="t", description="d",
              budget_max=10000, currency="RUB")
    a = Assessment.model_validate(ASSESSMENT)
    with_tax = decide(job, a, PROFILE, econ, 0.05)
    without = decide(job, a, PROFILE, Economics(), 0.05)
    assert with_tax.fee_usd == pytest.approx(without.fee_usd + with_tax.price_usd * 0.06)

    store = make_store()
    job_id, _ = store.upsert_job(job)
    record_payment(store, econ, store.get_job(job_id), 120.0)
    assert store.ledger_sum("revenue") == 120.0
    assert store.ledger_sum("expense") == pytest.approx(7.2)


def test_rendered_config_is_valid_and_quotes_user_text():
    a = Answers(name='Иван "Ваня": dev', skills=["Python", "C#"], chat_id=12345, tax_percent=6,
                freelancer_enabled=True, pchel_enabled=False)
    data = yaml.safe_load(render_config(a))
    s = Settings.model_validate(data)
    assert s.profile.name == 'Иван "Ваня": dev' and s.profile.skills == ["Python", "C#"]
    assert s.telegram.chat_id == 12345 and s.economics.tax_percent == 6
    assert s.sources.freelancer.enabled and s.sources.rss == []
    default = Settings.model_validate(yaml.safe_load(render_config(Answers())))
    assert not default.sources.freelancer.enabled  # withdrawals from Russia are not guaranteed
    assert [r.name for r in default.sources.rss] == ["pchel"]


def test_write_env_keeps_other_keys_and_is_private(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nANTHROPIC_API_KEY=old\nOTHER=1\n")
    write_env(env, {"ANTHROPIC_API_KEY": "new", "TELEGRAM_BOT_TOKEN": "t"})
    text = env.read_text()
    assert "ANTHROPIC_API_KEY=new" in text and "OTHER=1" in text and "TELEGRAM_BOT_TOKEN=t" in text
    assert "# comment" in text
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_setup_wizard_end_to_end(tmp_path):
    sent: list[dict] = []
    updates = [[], [{"update_id": 7, "message": {"chat": {"id": 555, "type": "private"}, "text": "/start"}}]]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.cbr.ru":
            return httpx.Response(200, content=fixture_bytes("cbr_daily.xml"))
        method = request.url.path.rsplit("/", 1)[-1]
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {"username": "my_agent_bot"}})
        if method == "getUpdates":
            offset = json.loads(request.content).get("offset")
            batch = updates.pop(0) if updates and offset is None else []
            return httpx.Response(200, json={"ok": True, "result": batch})
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    answers = iter(["Артём", "Python-разработчик", "Python, парсинг", "Делаю парсеры.",
                    "3000", "1500", "д", "н", "д", "н", "д"])
    secrets = iter(["sk-ant-test", "123:tok"])
    checked = []

    async def check_key(key: str) -> str | None:
        checked.append(key)
        return None

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await run_setup(tmp_path, lambda q, d: next(answers), lambda q: next(secrets),
                                   lambda line: None, http, check_key)

    config = asyncio.run(run())
    s = Settings.model_validate(yaml.safe_load(config.read_text()))
    assert checked == ["sk-ant-test"]
    assert s.telegram.chat_id == 555 and s.profile.name == "Артём"
    assert s.profile.min_project_usd == pytest.approx(3000 / 83.6177, abs=0.01)
    assert s.economics.tax_percent == 6 and s.sources.flru.enabled and not s.sources.freelancer.enabled
    env = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-test" in env and "TELEGRAM_BOT_TOKEN=123:tok" in env
    assert sent and sent[-1]["chat_id"] == 555 and "подключён" in sent[-1]["text"]
