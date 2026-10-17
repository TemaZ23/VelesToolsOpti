import asyncio
import json
import zipfile

import httpx2 as httpx

from freelance_agent.agents.executor import files_for_review, pack
from freelance_agent.cli import main
from freelance_agent.notify.telegram import Telegram, callback, esc, parse_callback, split_text


def test_split_keeps_entities_and_lines_intact():
    text = ("строка &amp; ещё\n" * 600)
    chunks = split_text(text, limit=4000)
    assert "".join(c + "\n" for c in chunks).count("&amp;") == 600
    assert all(len(c) <= 4000 for c in chunks)
    assert split_text("x" * 9000)[0] == "x" * 4000


def test_callback_roundtrip_fits_telegram_limit():
    data = callback("reply_send", 123456789)
    assert len(data.encode()) <= 64
    assert parse_callback(data) == ("reply_send", 123456789)
    assert parse_callback("garbage") is None
    assert esc("<b>&") == "&lt;b&gt;&amp;"


def test_send_puts_buttons_on_last_chunk_only():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(calls)}})

    async def run() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await Telegram(http, "T", 42).send("a\n" * 3000, [[("OK", "x")]])

    assert asyncio.run(run()) == 2
    assert "reply_markup" not in calls[0]
    assert calls[1]["reply_markup"]["inline_keyboard"][0][0] == {"text": "OK", "callback_data": "x"}
    assert all(c["parse_mode"] == "HTML" and c["chat_id"] == 42 for c in calls)


def test_review_listing_names_what_it_omits_and_pack(tmp_path):
    out = tmp_path / "deliverables"
    (out / "node_modules").mkdir(parents=True)
    (out / "node_modules" / "junk.js").write_text("x")
    (out / "DELIVERY_NOTES.md").write_text("notes")
    (out / "big.py").write_text("y" * 200)
    (out / "logo.png").write_bytes(b"\x89PNG")
    listing, content = files_for_review(tmp_path, max_chars=100)
    assert "node_modules" not in listing
    assert content.startswith('<file path="DELIVERY_NOTES.md">')
    assert "big.py (not shown: review size limit)" in content and "logo.png (binary)" in content
    names = zipfile.ZipFile(pack(tmp_path, "job-1")).namelist()
    assert sorted(names) == ["DELIVERY_NOTES.md", "big.py", "logo.png"]


def test_cli_init_and_estimate(tmp_path, capsys):
    cfg = tmp_path / "config.yaml"
    main(["-c", str(cfg), "init"])
    assert cfg.exists() and (tmp_path / ".env").exists()
    main(["-c", str(cfg), "estimate", "--price", "100", "--complexity", "small"])
    out = capsys.readouterr().out
    assert "margin per won job" in out and "break-even at" in out


def test_get_updates_long_polls_with_api_timeout():
    seen: list[tuple[dict, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content), request.extensions.get("timeout")))
        return httpx.Response(200, json={"ok": True, "result": [{"update_id": 1}]})

    async def run() -> list:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await Telegram(http, "T", 42).get_updates(5, timeout=30)

    assert asyncio.run(run()) == [{"update_id": 1}]
    body, http_timeout = seen[0]
    assert body["timeout"] == 30 and body["offset"] == 5
    assert http_timeout["read"] == 45  # HTTP waits longer than Telegram holds the request
