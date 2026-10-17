"""The agent as `freelance-agent run` builds it: real sources, real Telegram client and poll loop,
real Anthropic SDK. Only the network is simulated (job boards, CBR, Bot API, Claude API)."""

import asyncio
import json
from html.parser import HTMLParser

import httpx2 as httpx

from freelance_agent.app import build
from freelance_agent.config import Secrets
from freelance_agent.models import JobStatus

from .conftest import FakeAgentSDK, FakeClaude, fixture_bytes, make_settings, make_store

TELEGRAM_TAGS = {"b", "strong", "i", "em", "u", "s", "a", "code", "pre", "blockquote"}


class TelegramHtmlCheck(HTMLParser):
    """Rejects what Telegram rejects with 'can't parse entities': unknown or unbalanced tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        assert tag in TELEGRAM_TAGS, f"tag <{tag}> not supported by Telegram"
        self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack.pop() == tag, f"unbalanced </{tag}>"


class BotApi:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.calls: list[str] = []
        self.queue: list[dict] = []
        self.next_id = 1000
        self.update_id = 0

    def push(self, **update) -> None:
        self.update_id += 1
        self.queue.append({"update_id": self.update_id, **update})

    async def handle(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        self.calls.append(method)
        body = json.loads(request.content) if request.content and method != "sendDocument" else {}
        if method == "getUpdates":
            offset = body.get("offset") or 0
            ready = [u for u in self.queue if u["update_id"] >= offset]
            if not ready:
                await asyncio.sleep(0.01)
            return httpx.Response(200, json={"ok": True, "result": ready})
        if method == "sendMessage":
            assert body["chat_id"] == 42 and len(body["text"]) <= 4096
            checker = TelegramHtmlCheck()
            checker.feed(body["text"])
            checker.close()
            assert not checker.stack, f"unclosed tags {checker.stack}"
            for row in (body.get("reply_markup") or {}).get("inline_keyboard", []):
                for button in row:
                    assert len(button["callback_data"].encode()) <= 64
            self.next_id += 1
            self.sent.append({**body, "message_id": self.next_id})
            return httpx.Response(200, json={"ok": True, "result": {"message_id": self.next_id}})
        if method == "sendDocument":
            assert b'name="document"; filename="job-' in request.content
            assert b'name="chat_id"' in request.content
            self.next_id += 1
            return httpx.Response(200, json={"ok": True, "result": {"message_id": self.next_id}})
        return httpx.Response(200, json={"ok": True, "result": True})

    def last_card(self) -> dict:
        return next(m for m in reversed(self.sent) if m.get("reply_markup"))

    def button(self, message: dict, action: str) -> str:
        for row in message["reply_markup"]["inline_keyboard"]:
            for b in row:
                if json.loads(b["callback_data"])[0] == action:
                    return b["callback_data"]
        raise AssertionError(action)


def network(bot: BotApi, claude: FakeClaude):
    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "www.fl.ru":
            return httpx.Response(200, content=fixture_bytes("flru_all.xml"))
        if host == "pchel.net":
            return httpx.Response(200, content=fixture_bytes("pchel.xml"))
        if host == "www.cbr.ru":
            return httpx.Response(200, content=fixture_bytes("cbr_daily.xml"))
        if host == "api.telegram.org":
            return await bot.handle(request)
        raise AssertionError(f"unexpected request to {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_run_mode_wiring_with_telegram_buttons_and_replies(tmp_path):
    async def scenario() -> None:
        bot, claude = BotApi(), FakeClaude()
        settings = make_settings(
            tmp_path,
            max_new_jobs_per_cycle=2,
            sources={"freelancer": {"enabled": False},
                     "rss": [{"name": "pchel", "url": "https://pchel.net/rss/"}]},
        )
        app = build(settings, Secrets(anthropic_api_key="k", telegram_bot_token="T"),
                    store=make_store(), anthropic_client=claude.client(),
                    http=network(bot, claude), query_fn=FakeAgentSDK())
        agent, store = app.agent, app.store

        stats = await agent.cycle()
        assert stats.fetched == 11 + 50 and stats.proposals == 2  # FL.ru «для всех» + Pchel
        assert settings.economics.fx_to_usd["RUB"] < 0.0125  # CBR rate applied
        cards = [m for m in bot.sent if m.get("reply_markup")]
        assert len(cards) == 2

        poller = asyncio.create_task(agent.poll_telegram())

        async def settle() -> None:
            for _ in range(100):
                await asyncio.sleep(0.02)
                if not agent._tasks - {poller}:
                    break

        # tap "I responded" on the first card
        first = cards[0]
        bot.push(callback_query={"id": "cb1", "data": bot.button(first, "send"),
                                 "message": {"message_id": first["message_id"], "chat": {"id": 42}}})
        await settle()
        assert "answerCallbackQuery" in bot.calls and "editMessageReplyMarkup" in bot.calls
        submitted = store.jobs_by_status(JobStatus.SUBMITTED)
        assert len(submitted) == 1
        followup = bot.last_card()

        # forward the client's message as a reply to the follow-up card
        bot.push(message={"message_id": 1, "chat": {"id": 42}, "text": "Нужны колонки name и price",
                          "reply_to_message": {"message_id": followup["message_id"]}})
        await settle()
        assert "Черновик ответа" in bot.last_card()["text"]

        # commands from you work, messages from other chats are ignored
        bot.push(message={"message_id": 2, "chat": {"id": 42}, "text": "/stats"})
        bot.push(message={"message_id": 3, "chat": {"id": 999}, "text": "/pause"})
        await settle()
        assert "ПРИБЫЛЬ" in bot.sent[-1]["text"]
        assert store.kv_get("paused") != "1"

        # the client picks you: work, review, zip to Telegram, delivery card
        bot.push(callback_query={"id": "cb2", "data": bot.button(followup, "won"),
                                 "message": {"message_id": followup["message_id"], "chat": {"id": 42}}})
        await settle()
        await asyncio.gather(*(agent._tasks - {poller}))
        assert "sendDocument" in bot.calls
        assert store.get_job(submitted[0].id).status == JobStatus.REVIEW
        assert "готово" in bot.last_card()["text"]

        poller.cancel()
        await app.aclose()

    asyncio.run(scenario())
