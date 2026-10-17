import asyncio
import json

import httpx2 as httpx
import pytest

from freelance_agent.sources.base import make_http
from freelance_agent.sources.flru import FlruSource, clean_title, parse_budget, parse_feed
from freelance_agent.sources.freelancer_com import FreelancerClient, project_to_job
from freelance_agent.sources.rss import parse_feed as parse_rss

from .conftest import fixture_bytes


def test_flru_budget_and_title_parsing():
    assert parse_budget("Дизайн скриншотов (Бюджет: 1 000  ₽)") == (1000.0, "RUB")
    assert parse_budget("Игра (Бюджет: 10 000  ₽, для всех)") == (10000.0, "RUB")
    assert parse_budget("Создать приложение (для всех)") == (None, None)
    assert clean_title("Игра для свиданий (Бюджет: 10 000  ₽, для всех)") == "Игра для свиданий"
    assert clean_title("Создать легкое приложение для ПК (для всех)") == "Создать легкое приложение для ПК"


def test_flru_real_feed_snapshot():
    jobs = parse_feed(fixture_bytes("flru_all.xml"))
    assert len(jobs) == 59  # every <item> of the captured feed
    by_id = {j.external_id: j for j in jobs}
    game = by_id["5525681"]
    assert game.title == "Попробуй игру для свиданий и сними видео"
    assert game.budget_max == 10000 and game.currency == "RUB"
    assert game.raw["open_to_all"] is True
    assert game.posted_at is not None
    assert all(j.url.startswith("https://www.fl.ru/projects/") for j in jobs)
    assert all(j.description for j in jobs)


def test_flru_source_skips_pro_only_without_pro_account():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=fixture_bytes("flru_all.xml"))

    async def run(pro: bool) -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return len(await FlruSource(http, ["https://x/rss"], pro_account=pro).fetch())

    assert asyncio.run(run(True)) == 59
    open_jobs = asyncio.run(run(False))
    assert 0 < open_jobs < 59


def test_freelancer_project_conversion_uses_live_exchange_rate():
    data = json.loads(fixture_bytes("freelancer_active.json"))
    jobs = [project_to_job(p) for p in data["result"]["projects"]]
    assert len(jobs) == 5
    inr = next(j for j in jobs if j.currency == "INR")
    assert inr.usd_rate and inr.usd_rate < 0.05
    assert inr.budget_max and inr.budget_max * inr.usd_rate < 1000
    assert all(j.url.startswith("https://www.freelancer.com/projects/") for j in jobs)
    assert all(j.raw["type"] == "fixed" for j in jobs)
    assert all(j.bid_count is not None for j in jobs)


def test_freelancer_bid_request_shape():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/users/0.1/self/"):
            return httpx.Response(200, json={"status": "success", "result": {"id": 777}})
        return httpx.Response(200, json={"status": "success", "result": {"id": 555}})

    async def run() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await FreelancerClient(http, "tok").place_bid(123, 90.0, 3, "x" * 120)

    assert asyncio.run(run()) == 555
    bid = seen[-1]
    assert bid.method == "POST" and bid.url.path == "/api/projects/0.1/bids/"
    assert bid.headers["Freelancer-OAuth-V1"] == "tok"
    body = json.loads(bid.content)
    assert body == {"project_id": 123, "bidder_id": 777, "amount": 90.0, "period": 3,
                    "milestone_percentage": 100, "description": "x" * 120}


def test_freelancer_error_is_raised_with_api_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"status": "error", "error_code": "NOT_AUTHENTICATED",
                                         "message": "You must be logged in"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await FreelancerClient(http, "bad").self_id()

    with pytest.raises(Exception, match="NOT_AUTHENTICATED"):
        asyncio.run(run())


def test_generic_rss():
    xml = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
    <item><title>Build a bot</title><link>https://x/1</link><guid>https://x/1</guid>
    <description>&lt;p&gt;Need a &lt;b&gt;Telegram&lt;/b&gt; bot&lt;/p&gt;</description>
    <pubDate>Sat, 17 Oct 2026 06:10:29 GMT</pubDate></item></channel></rss>"""
    [job] = parse_rss(xml, "wwr")
    assert job.title == "Build a bot" and job.description == "Need a Telegram bot"
    assert job.source == "wwr" and len(job.external_id) == 16


@pytest.mark.live
def test_live_job_boards_respond():
    """Read-only requests to the real job boards (no LLM, no money)."""

    async def run() -> tuple[int, int]:
        http = make_http()
        try:
            fl = await FlruSource(http, ["https://www.fl.ru/rss/all.xml"], pro_account=True).fetch()
            fr = await FreelancerClient(http, None).search_active("python", limit=5)
            return len(fl), len(fr)
        finally:
            await http.aclose()

    flru, freelancer = asyncio.run(run())
    assert flru > 0 and freelancer > 0
