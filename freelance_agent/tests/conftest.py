"""Test doubles. The Anthropic SDK is exercised for real: only its HTTP transport is mocked."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import httpx2 as httpx
import pytest
from claude_agent_sdk import ResultMessage

from freelance_agent.agents import prompts
from freelance_agent.config import Settings
from freelance_agent.db import Store

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    data: dict[str, Any] = {
        "data_dir": str(tmp_path / "data"),
        "profile": {
            "name": "Test Dev",
            "headline": "Python developer",
            "about": "Writes scrapers and bots.",
            "skills": ["Python", "scraping"],
        },
        "executor": {"sandbox": False, "allow_unsandboxed": True},
        "telegram": {"enabled": True, "chat_id": 42},
        "filters": {"max_age_hours": 10_000},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return Settings.model_validate(data)


def make_store() -> Store:
    return Store(":memory:", clock=lambda: NOW)


ASSESSMENT = {
    "category": "software",
    "ai_deliverable": True,
    "ethical": True,
    "fit_score": 85,
    "complexity": "small",
    "estimated_human_hours": 3,
    "suggested_price_usd": 120,
    "language": "ru",
    "requirements_summary": "Python script that scrapes product names and prices to CSV.",
    "clarifying_questions": ["Which site?"],
    "red_flags": [],
    "reasoning": "Clear scraping task.",
}
PROPOSAL = {
    "text": "Сделаю парсер каталога в CSV: соберу названия и цены со всех 20 страниц, "
            "добавлю повторы при ошибках и README. Срок 2 дня. Какой формат цены нужен?",
    "period_days": 2,
}
REPLY = {
    "intent": "requirements",
    "reply": "Принято, сделаю выгрузку в CSV с колонками name, price.",
    "requirements_update": "Колонки CSV: name, price.",
    "acceptance_criteria": ["CSV содержит name и price для всех товаров"],
    "needs_human": False,
    "human_reason": "",
}
REVIEW = {"passed": True, "issues": [], "delivery_message": "Готово: scraper.py и README.md."}


class FakeClaude:
    """Routes Messages API calls to canned structured outputs by system prompt."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: dict[str, dict[str, Any]] = {
            prompts.TRIAGE_SYSTEM: ASSESSMENT,
            prompts.PROPOSAL_SYSTEM: PROPOSAL,
            prompts.REPLY_SYSTEM: REPLY,
            prompts.REVIEW_SYSTEM: REVIEW,
        }
        self.stop_reason = "end_turn"
        self.usage = {"input_tokens": 2000, "output_tokens": 500,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"body": body, "headers": dict(request.headers),
                              "url": str(request.url)})
        system = body["system"][0]["text"]
        payload = self.responses[system]
        return httpx.Response(200, json={
            "id": f"msg_{len(self.requests)}",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "stop_reason": self.stop_reason,
            "stop_sequence": None,
            "usage": self.usage,
        })

    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx.MockTransport(self.handler)),
        )

    def purposes(self) -> list[str]:
        names = {
            prompts.TRIAGE_SYSTEM: "triage",
            prompts.PROPOSAL_SYSTEM: "proposal",
            prompts.REPLY_SYSTEM: "reply",
            prompts.REVIEW_SYSTEM: "review",
        }
        return [names[r["body"]["system"][0]["text"]] for r in self.requests]


class FakeTelegram:
    def __init__(self) -> None:
        self.chat_id = 42
        self.sent: list[tuple[int, str, Any]] = []
        self.documents: list[Path] = []
        self.cleared: list[int] = []
        self._next = 100

    async def send(self, text_html: str, buttons: Any = None, reply_to: int | None = None) -> int:
        self._next += 1
        self.sent.append((self._next, text_html, buttons))
        return self._next

    async def send_document(self, path: Path, caption_html: str = "") -> int:
        self.documents.append(path)
        self._next += 1
        return self._next

    async def clear_buttons(self, message_id: int) -> None:
        self.cleared.append(message_id)

    async def answer_callback(self, callback_id: str, text: str = "") -> None:
        pass

    def last_buttons(self) -> list[tuple[str, str]]:
        for _, _, buttons in reversed(self.sent):
            if buttons:
                return [b for row in buttons for b in row]
        return []

    def last_card_id(self) -> int:
        for mid, _, buttons in reversed(self.sent):
            if buttons:
                return mid
        raise AssertionError("no card sent")


class FakeAgentSDK:
    """Stands in for claude_agent_sdk.query: writes deliverables like the real agent would."""

    def __init__(self, cost: float = 0.9, files: dict[str, str] | None = None,
                 is_error: bool = False) -> None:
        self.cost = cost
        self.files = files if files is not None else {
            "scraper.py": "print('ok')\n",
            "README.md": "# Scraper\nRun: python scraper.py\n",
            "DELIVERY_NOTES.md": "Done, verified with a demo run.\n",
        }
        self.is_error = is_error
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *, prompt: str, options: Any) -> Any:
        self.calls.append({"prompt": prompt, "options": options})
        out = Path(options.cwd) / "deliverables"
        for name, content in self.files.items():
            (out / name).write_text(content, encoding="utf-8")
        yield ResultMessage(
            subtype="error_max_budget_usd" if self.is_error else "success",
            duration_ms=1000,
            duration_api_ms=900,
            is_error=self.is_error,
            num_turns=5,
            session_id="sess-1",
            total_cost_usd=self.cost,
            usage={"input_tokens": 10000, "output_tokens": 3000},
            result="Implemented scraper.py",
        )


@pytest.fixture
def fake_claude() -> FakeClaude:
    return FakeClaude()


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


Handler = Callable[[httpx.Request], httpx.Response]
