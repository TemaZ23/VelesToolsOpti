"""End-to-end: the whole order lifecycle with real pipeline code; only the network edges are fake."""

import asyncio
import json
import zipfile
from datetime import timedelta
from pathlib import Path

import pytest

from freelance_agent.agents import executor as executor_mod
from freelance_agent.agents.brain import Brain
from freelance_agent.agents.executor import Executor
from freelance_agent.budget import BudgetGuard
from freelance_agent.llm import LLM
from freelance_agent.models import Job, JobStatus
from freelance_agent.orchestrator import Agent
from freelance_agent.sources.freelancer_com import ThreadMessage

from .conftest import NOW, FakeAgentSDK, FakeClaude, FakeTelegram, make_settings, make_store


class ListSource:
    def __init__(self, name: str, jobs: list[Job]) -> None:
        self.name = name
        self.jobs = jobs

    async def fetch(self) -> list[Job]:
        return list(self.jobs)


class FakeFreelancer:
    can_write = True

    def __init__(self) -> None:
        self.bids: list[dict] = []
        self.award_status = "pending"
        self.accepted: list[int] = []
        self.posted: list[str] = []
        self.attachments: list[Path] = []
        self.thread: list[ThreadMessage] = []
        self.milestone_status: str | None = None
        self.release_requests: list[int] = []

    async def place_bid(self, project_id, amount, period_days, description):
        self.bids.append(dict(project_id=project_id, amount=amount, period=period_days, text=description))
        return 9001

    async def get_bids(self, bid_ids):
        return [{"id": b, "award_status": self.award_status} for b in bid_ids]

    async def accept_bid(self, bid_id):
        self.accepted.append(bid_id)

    async def project_thread_id(self, project_id):
        return 5001

    async def self_id(self):
        return 777

    async def thread_messages(self, thread_id):
        return list(self.thread)

    async def post_message(self, thread_id, text):
        self.posted.append(text)
        self.thread.append(ThreadMessage(f"out{len(self.posted)}", 777, text, 99))

    async def post_attachment(self, thread_id, path):
        self.attachments.append(path)

    async def milestones(self, project_id):
        if not self.milestone_status:
            return []
        return [{"id": 1, "amount": 120.0, "status": self.milestone_status, "bidder_id": 777}]

    async def request_release(self, milestone_id):
        self.release_requests.append(milestone_id)


GOOD_FLRU = Job(source="flru", external_id="100", url="https://www.fl.ru/projects/100/x.html",
                title="Парсер каталога в CSV",
                description="Нужен скрипт на Python: собрать названия и цены 500 товаров в CSV.",
                budget_max=12000, currency="RUB", posted_at=NOW - timedelta(hours=1))
BAD_FLRU = Job(source="flru", external_id="101", url="https://www.fl.ru/projects/101/y.html",
               title="Написать отзывы на картах",
               description="Нужно 30 отзывов о нашем кафе на картах, оплата за каждый.",
               budget_max=3000, currency="RUB", posted_at=NOW - timedelta(hours=1))
GOOD_FREELANCER = Job(source="freelancer", external_id="40772781",
                      url="https://www.freelancer.com/projects/x", title="Python scraper to CSV",
                      description="Scrape 500 products with prices from a catalogue into a CSV file.",
                      budget_min=100, budget_max=250, currency="USD", usd_rate=1.0, bid_count=8,
                      posted_at=NOW - timedelta(hours=1), raw={"type": "fixed"})


def build(tmp_path, claude: FakeClaude, sources, fl=None, sdk=None, **overrides):
    settings = make_settings(tmp_path, **overrides)
    store = make_store()
    guard = BudgetGuard(store, settings.budget)
    brain = Brain(LLM(claude.client(), store, guard), settings)
    sdk = sdk or FakeAgentSDK()
    executor = Executor(settings, store, guard, sdk)
    tg = FakeTelegram()
    agent = Agent(settings, store, guard, brain, executor, sources, tg, fl)  # type: ignore[arg-type]
    return agent, store, tg, sdk


def card(tg: FakeTelegram, action: str) -> int:
    for _label, data in tg.last_buttons():
        name, pid = json.loads(data)
        if name == action:
            return pid
    raise AssertionError(f"no {action!r} button in {tg.last_buttons()}")


async def drain(agent: Agent) -> None:
    while agent._tasks:
        await asyncio.gather(*list(agent._tasks))


def test_flru_order_from_post_to_payment(tmp_path, fake_claude):
    async def scenario():
        agent, store, tg, sdk = build(tmp_path, fake_claude, [ListSource("flru", [GOOD_FLRU, BAD_FLRU])])

        stats = await agent.cycle()
        assert (stats.new, stats.filtered, stats.assessed, stats.proposals) == (2, 1, 1, 1)
        good = store.get_job_by_uid(GOOD_FLRU.uid)
        assert good.status == JobStatus.AWAITING_APPROVAL
        assert store.get_job_by_uid(BAD_FLRU.uid).status == JobStatus.FILTERED
        assert good.bid_amount == 10000 and good.decision["execution_budget_usd"] == 2.25
        assert "Я откликнулся" in tg.last_buttons()[0][0]

        # 1. You paste the proposal on FL.ru and tap "I responded"
        assert await agent.on_button("send", card(tg, "send")) == "Отмечено как отправленное"
        assert store.get_job(good.id).status == JobStatus.SUBMITTED
        followup = card(tg, "won")

        # 2. The client answers on FL.ru; you forward it as a reply to the follow-up card
        await agent.on_reply(followup, "Колонки нужны name и price")
        assert "Колонки CSV: name, price." in store.get_job(good.id).requirements
        assert await agent.on_button("reply_send", card(tg, "reply_send")) == "Отмечено"

        # 3. The client picks you: work runs, gets reviewed and packed
        assert await agent.on_button("won", followup) == "Принято, начинаю работу"
        await drain(agent)
        rec = store.get_job(good.id)
        assert rec.status == JobStatus.REVIEW
        [call] = sdk.calls
        assert call["options"].max_budget_usd == 2.25
        task = (Path(call["options"].cwd) / "TASK.md").read_text(encoding="utf-8")
        assert "Колонки CSV: name, price." in task and "<job_post>" in task
        [archive] = tg.documents
        assert set(zipfile.ZipFile(archive).namelist()) == {"scraper.py", "README.md", "DELIVERY_NOTES.md"}

        # 4. You send the files to the client and later confirm the payment
        assert await agent.on_button("deliver", card(tg, "deliver")) == "Сдано"
        assert store.get_job(good.id).status == JobStatus.DELIVERED
        assert await agent.on_button("paid", card(tg, "paid")) == "Оплата учтена"
        assert store.get_job(good.id).status == JobStatus.PAID

        r = agent.guard.report()
        assert r.revenue_month == pytest.approx(120.0) and r.fees_month == 0
        assert r.llm_month == pytest.approx(store.llm_spend())
        assert 0.9 < r.llm_month < 1.0  # $0.90 execution + four cheap model calls
        assert r.profit_month == pytest.approx(120.0 - r.llm_month - 20.0)
        assert fake_claude.purposes() == ["triage", "proposal", "reply", "review"]
        assert await agent.on_button("paid", card(tg, "paid")) == "Уже обработано"  # double tap

    asyncio.run(scenario())


def test_freelancer_fully_automatic_order(tmp_path, fake_claude):
    async def scenario():
        fl = FakeFreelancer()
        agent, store, tg, sdk = build(tmp_path, fake_claude, [ListSource("freelancer", [GOOD_FREELANCER])],
                                      fl=fl, autonomy={"mode": "auto"})
        await agent.cycle()
        rec = store.get_job_by_uid(GOOD_FREELANCER.uid)
        assert rec.status == JobStatus.SUBMITTED and rec.external_bid_id == "9001"
        [bid] = fl.bids
        assert bid["project_id"] == 40772781 and bid["amount"] == 120 and bid["period"] == 2

        # client writes; the reply goes out automatically, exactly once
        fl.thread.append(ThreadMessage("m1", 123, "Can you include a price column?", 10))
        await agent.sync_freelancer()
        await agent.sync_freelancer()
        assert fl.posted == ["Принято, сделаю выгрузку в CSV с колонками name, price."]

        # award -> accept -> execute -> review -> deliver
        fl.award_status = "awarded"
        await agent.sync_freelancer()
        await drain(agent)
        assert fl.accepted == [9001]
        assert store.get_job(rec.id).status == JobStatus.DELIVERED
        assert len(fl.attachments) == 1 and fl.posted[-1] == "Готово: scraper.py и README.md."

        # escrow released -> revenue and platform fee booked, job closed
        fl.milestone_status = "cleared"
        await agent.sync_freelancer()
        assert store.get_job(rec.id).status == JobStatus.PAID
        r = agent.guard.report()
        assert r.revenue_month == 120.0 and r.fees_month == 12.0
        await agent.sync_freelancer()
        assert agent.guard.report().revenue_month == 120.0  # not double-counted
        history = [(m.direction, m.text) for m in store.messages_for(rec.id)]
        assert history == [
            ("in", "Can you include a price column?"),
            ("out", "Принято, сделаю выгрузку в CSV с колонками name, price."),
            ("out", "Готово: scraper.py и README.md."),
        ]

    asyncio.run(scenario())


def test_semi_mode_asks_before_bidding_and_edited_text_is_sent(tmp_path, fake_claude):
    async def scenario():
        fl = FakeFreelancer()
        agent, store, tg, _ = build(tmp_path, fake_claude, [ListSource("freelancer", [GOOD_FREELANCER])],
                                    fl=fl, autonomy={"mode": "semi", "auto_bid_min_score": 90})
        await agent.cycle()
        assert fl.bids == []  # score 85 < 90: needs your tap
        pid = card(tg, "send")
        edited = "My own proposal text. " * 6
        await agent.on_reply(pid, edited)
        assert await agent.on_button("send", card(tg, "send")) == "Ставка отправлена"
        assert fl.bids[0]["text"] == edited

    asyncio.run(scenario())


def test_stop_loss_blocks_paid_triage(tmp_path, fake_claude):
    async def scenario():
        agent, store, _, _ = build(tmp_path, fake_claude, [ListSource("flru", [GOOD_FLRU])],
                                   budget={"stop_loss_usd": 1.0})
        store.add_llm_call(purpose="triage", model="claude-haiku-5-5", cost_usd=2.0)
        await agent.cycle()
        assert fake_claude.requests == []
        assert store.get_job_by_uid(GOOD_FLRU.uid).status == JobStatus.NEW

    asyncio.run(scenario())


def test_failed_execution_is_reported_and_can_be_retried(tmp_path, fake_claude):
    async def scenario():
        sdk = FakeAgentSDK(is_error=True)
        agent, store, tg, _ = build(tmp_path, fake_claude, [ListSource("flru", [GOOD_FLRU])], sdk=sdk)
        await agent.cycle()
        job = store.get_job_by_uid(GOOD_FLRU.uid)
        await agent.on_button("send", card(tg, "send"))
        await agent.on_button("won", card(tg, "won"))
        await drain(agent)
        assert store.get_job(job.id).status == JobStatus.WON
        assert f"/work {job.id}" in tg.sent[-1][1]
        assert store.llm_spend(job_id=job.id, purposes=("execute",)) == pytest.approx(0.9)

    asyncio.run(scenario())


def test_executor_refuses_to_run_without_sandbox(tmp_path, fake_claude, monkeypatch):
    async def scenario():
        monkeypatch.setattr(executor_mod, "sandbox_problem", lambda: "bwrap: Operation not permitted")
        sdk = FakeAgentSDK()
        agent, store, tg, _ = build(tmp_path, fake_claude, [], sdk=sdk,
                                    executor={"sandbox": True, "allow_unsandboxed": False})
        job_id, _ = store.upsert_job(GOOD_FLRU)
        res = await agent.executor.run(job_id, GOOD_FLRU, "", [], 1.0)
        assert not res.ok and "Operation not permitted" in res.error and sdk.calls == []

    asyncio.run(scenario())


def test_executor_options_are_locked_down(tmp_path, fake_claude):
    agent, *_ = build(tmp_path, fake_claude, [], executor={"sandbox": True})
    opts = agent.executor.options(tmp_path, 2.5, None)
    assert opts.max_budget_usd == 2.5
    assert opts.tools == ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
    assert opts.permission_mode == "dontAsk" and opts.setting_sources == []
    assert opts.sandbox["enabled"] and not opts.sandbox["allowUnsandboxedCommands"]
    assert opts.env["FREELANCER_OAUTH_TOKEN"] == "" and opts.env["TELEGRAM_BOT_TOKEN"] == ""
    assert opts.model == "claude-opus-5-5" and opts.effort == "high"


def test_telegram_updates_from_strangers_are_ignored(tmp_path, fake_claude):
    async def scenario():
        agent, store, tg, _ = build(tmp_path, fake_claude, [ListSource("flru", [GOOD_FLRU])])
        await agent.cycle()
        pid = card(tg, "skip")
        foreign = {"callback_query": {"id": "1", "data": json.dumps(["skip", pid]),
                                      "message": {"message_id": 1, "chat": {"id": 666}}}}
        await agent.handle_update(foreign)
        assert store.get_job_by_uid(GOOD_FLRU.uid).status == JobStatus.AWAITING_APPROVAL
        mine = {"callback_query": {"id": "2", "data": json.dumps(["skip", pid]),
                                   "message": {"message_id": 1, "chat": {"id": 42}}}}
        await agent.handle_update(mine)
        assert store.get_job_by_uid(GOOD_FLRU.uid).status == JobStatus.SKIPPED
        await agent.handle_update({"message": {"chat": {"id": 42}, "text": "/stats"}})
        assert "ПРИБЫЛЬ" in tg.sent[-1][1]

    asyncio.run(scenario())


def test_bad_api_key_leaves_jobs_for_the_next_cycle(tmp_path, fake_claude):
    import httpx2 as httpx

    def unauthorized(request):
        return httpx.Response(401, json={"type": "error", "error": {
            "type": "authentication_error", "message": "invalid x-api-key"}})

    async def scenario():
        fake_claude.handler = unauthorized
        agent, store, tg, _ = build(tmp_path, fake_claude, [ListSource("flru", [GOOD_FLRU])])
        await agent.cycle()
        assert store.get_job_by_uid(GOOD_FLRU.uid).status == JobStatus.NEW
        assert any("Claude API недоступен" in text for _, text, _ in tg.sent)
        await agent.cycle()  # alerted only once a day
        assert sum("Claude API недоступен" in text for _, text, _ in tg.sent) == 1

    asyncio.run(scenario())
