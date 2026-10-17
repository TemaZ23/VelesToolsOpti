"""Minimal Telegram Bot API client (long polling, inline buttons, documents)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import httpx2 as httpx

LIMIT = 4000

Buttons = list[list[tuple[str, str]]]


class TelegramError(RuntimeError):
    pass


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def split_text(text: str, limit: int = LIMIT) -> list[str]:
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
            amp = text.rfind("&", cut - 8, cut)
            if amp != -1 and ";" not in text[amp:cut]:
                cut = amp  # do not split an HTML entity such as &amp;
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    chunks.append(text)
    return chunks


class Telegram:
    def __init__(self, http: httpx.AsyncClient, token: str, chat_id: int | None) -> None:
        self.http = http
        self.base = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id

    async def _call(self, method: str, _http_timeout: float | None = None, **params: Any) -> Any:
        # Bot API parameters go in **params (getUpdates has its own `timeout` parameter).
        files = params.pop("_files", None)
        if files:
            r = await self.http.post(f"{self.base}/{method}", data=params, files=files,
                                     timeout=_http_timeout or 120)
        else:
            r = await self.http.post(f"{self.base}/{method}", json=params,
                                     timeout=_http_timeout or 30)
        data = r.json()
        if not data.get("ok"):
            raise TelegramError(f"{method}: {data.get('description')}")
        return data["result"]

    async def get_me(self) -> dict[str, Any]:
        return await self._call("getMe")

    async def get_updates(self, offset: int | None, timeout: int = 30) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            params["offset"] = offset
        return await self._call("getUpdates", _http_timeout=timeout + 15, **params)

    async def send(self, text_html: str, buttons: Buttons | None = None,
                   reply_to: int | None = None) -> int:
        if self.chat_id is None:
            raise TelegramError("telegram.chat_id is not set; send /start to the bot first")
        chunks = split_text(text_html)
        message_id = 0
        for i, chunk in enumerate(chunks):
            params: dict[str, Any] = {
                "chat_id": self.chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": True},
            }
            if i == 0 and reply_to:
                params["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
            if i == len(chunks) - 1 and buttons:
                params["reply_markup"] = {
                    "inline_keyboard": [
                        [{"text": label, "callback_data": data} for label, data in row]
                        for row in buttons
                    ]
                }
            message_id = (await self._call("sendMessage", **params))["message_id"]
        return int(message_id)

    async def clear_buttons(self, message_id: int) -> None:
        try:
            await self._call("editMessageReplyMarkup", chat_id=self.chat_id,
                             message_id=message_id, reply_markup={"inline_keyboard": []})
        except TelegramError:
            pass  # message too old or already edited

    async def answer_callback(self, callback_id: str, text: str = "") -> None:
        try:
            await self._call("answerCallbackQuery", callback_query_id=callback_id, text=text[:190])
        except TelegramError:
            pass

    async def send_document(self, path: Path, caption_html: str = "") -> int:
        if self.chat_id is None:
            raise TelegramError("telegram.chat_id is not set")
        params: dict[str, Any] = {"chat_id": str(self.chat_id), "parse_mode": "HTML",
                                  "caption": caption_html[:1000]}
        result = await self._call("sendDocument", _files={"document": (path.name, path.read_bytes())},
                                  **params)
        return int(result["message_id"])


def callback(action: str, pending_id: int) -> str:
    return json.dumps([action, pending_id], separators=(",", ":"))


def parse_callback(data: str) -> tuple[str, int] | None:
    try:
        action, pending_id = json.loads(data)
        return str(action), int(pending_id)
    except (ValueError, TypeError):
        return None
