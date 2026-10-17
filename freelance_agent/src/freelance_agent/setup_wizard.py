"""Interactive first-run setup: keys, Telegram chat, profile, platforms -> config.yaml and .env."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx2 as httpx
import yaml

from . import fx
from .config import Settings
from .notify.telegram import Telegram, TelegramError

TEMPLATES = Path(__file__).resolve().parent / "templates"
PCHEL = {"name": "pchel", "url": "https://pchel.net/rss/"}

Ask = Callable[[str, str], str]
AskSecret = Callable[[str], str]
Print = Callable[[str], None]
KeyCheck = Callable[[str], Awaitable[str | None]]


@dataclass
class Answers:
    name: str = "Иван Петров"
    headline: str = "Python-разработчик: парсеры, Telegram-боты, автоматизация"
    about: str = ("Пишу на Python парсеры, интеграции с API, Telegram-ботов, обработку Excel/CSV. "
                  "Работаю с ИИ-инструментами, весь результат проверяю сам.")
    skills: list[str] = field(default_factory=lambda: [
        "Python", "парсинг", "Telegram-боты", "API", "Excel", "SQL", "автоматизация"])
    hourly_rate_usd: float = 18.0
    min_project_usd: float = 30.0
    tax_percent: float = 0.0
    flru_enabled: bool = True
    flru_pro: bool = False
    pchel_enabled: bool = True
    freelancer_enabled: bool = False
    chat_id: int | None = None


def _yaml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid YAML scalar


def render_config(a: Answers) -> str:
    text = (TEMPLATES / "config.example.yaml").read_text(encoding="utf-8")
    values = {
        "name": _yaml_str(a.name),
        "headline": _yaml_str(a.headline),
        "about": _yaml_str(a.about),
        "skills": json.dumps(a.skills, ensure_ascii=False),
        "hourly_rate_usd": f"{a.hourly_rate_usd:g}",
        "min_project_usd": f"{a.min_project_usd:g}",
        "tax_percent": f"{a.tax_percent:g}",
        "flru_enabled": str(a.flru_enabled).lower(),
        "flru_pro": str(a.flru_pro).lower(),
        "freelancer_enabled": str(a.freelancer_enabled).lower(),
        "rss": json.dumps([PCHEL] if a.pchel_enabled else []),
        "chat_id": "null" if a.chat_id is None else str(a.chat_id),
    }
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ValueError("unfilled placeholder in config template")
    Settings.model_validate(yaml.safe_load(text))
    return text


def write_env(path: Path, values: dict[str, str]) -> None:
    """Set keys in .env, keeping comments and other keys; the file is readable only by you."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else \
        (TEMPLATES / "env.example").read_text(encoding="utf-8").splitlines()
    pending = dict(values)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in pending:
            lines[i] = f"{key}={pending.pop(key)}"
    lines += [f"{k}={v}" for k, v in pending.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)


async def check_anthropic_key(key: str) -> str | None:
    """None if the key works (free Models API call), else the error."""
    client = anthropic.AsyncAnthropic(api_key=key, max_retries=1)
    try:
        await client.models.retrieve("claude-haiku-5-5")
        return None
    except anthropic.APIStatusError as e:
        return f"{e.status_code}: {e.message}"
    except anthropic.APIConnectionError as e:
        return f"нет соединения с api.anthropic.com: {e}"
    finally:
        await client.close()


async def wait_for_chat(tg: Telegram, out: Print, attempts: int = 9) -> int | None:
    """Wait until you press Start in the bot; returns your chat id."""
    offset: int | None = None
    for _ in range(attempts):
        try:
            updates = await tg.get_updates(offset, timeout=20)
        except (TelegramError, httpx.HTTPError) as e:
            out(f"  Telegram: {e}")
            return None
        for u in updates:
            offset = int(u["update_id"]) + 1
            chat = (u.get("message") or {}).get("chat") or {}
            if chat.get("type") == "private" and chat.get("id"):
                await tg.get_updates(offset, timeout=0)  # acknowledge, so the agent starts clean
                return int(chat["id"])
        out("  ...жду нажатия «Start» в боте")
    return None


def _yes(answer: str) -> bool:
    return answer.strip().lower() in {"y", "yes", "д", "да", "1", "true"}


async def run_setup(home: Path, ask: Ask, ask_secret: AskSecret, out: Print,
                    http: httpx.AsyncClient, check_key: KeyCheck = check_anthropic_key) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    config_path, env_path = home / "config.yaml", home / ".env"
    env: dict[str, str] = {}
    a = Answers()

    out("\n== 1/4. Claude API ==")
    out("Ключ создаётся в консоли Anthropic: https://platform.claude.com/settings/keys")
    while True:
        key = ask_secret("ANTHROPIC_API_KEY (Enter — пропустить и вписать позже в .env): ").strip()
        if not key:
            break
        problem = await check_key(key)
        if problem is None:
            out("  ключ работает")
            env["ANTHROPIC_API_KEY"] = key
            break
        out(f"  ключ не принят: {problem}")

    out("\n== 2/4. Telegram ==")
    out("Откройте @BotFather в Telegram -> /newbot -> скопируйте токен.")
    while True:
        token = ask_secret("TELEGRAM_BOT_TOKEN (Enter — пропустить): ").strip()
        if not token:
            break
        tg = Telegram(http, token, None)
        try:
            me = await tg.get_me()
        except (TelegramError, httpx.HTTPError) as e:
            out(f"  токен не принят: {e}")
            continue
        env["TELEGRAM_BOT_TOKEN"] = token
        out(f"  бот @{me.get('username')}. Откройте https://t.me/{me.get('username')} и нажмите «Start».")
        a.chat_id = await wait_for_chat(tg, out)
        if a.chat_id is None:
            out("  не дождался. Впишите telegram.chat_id в config.yaml позже или запустите setup снова.")
        else:
            await Telegram(http, token, a.chat_id).send("✅ Фриланс-агент подключён. Здесь будут "
                                                         "карточки заказов и отчёты. /help — команды.")
            out(f"  chat_id {a.chat_id} сохранён, в бот отправлено приветствие")
        break

    out("\n== 3/4. Профиль (только правда — отклики цитируют его) ==")
    rub = 0.012
    try:
        r = await http.get(fx.CBR_URL)
        rub = fx.parse_cbr(r.content)["RUB"]
    except Exception:  # noqa: BLE001 - offline is fine, the default rate is used
        pass
    a.name = ask("Имя, как на площадках", a.name)
    a.headline = ask("Кто вы, одной строкой", a.headline)
    a.skills = [s.strip() for s in ask("Навыки через запятую", ", ".join(a.skills)).split(",") if s.strip()]
    a.about = ask("О себе, 1-3 предложения", a.about)
    min_rub = float(ask("Минимальная цена заказа, ₽", "2500").replace(" ", ""))
    hour_rub = float(ask("Ваша ставка за час, ₽", "1500").replace(" ", ""))
    a.min_project_usd = round(min_rub * rub, 2)
    a.hourly_rate_usd = round(hour_rub * rub, 2)

    out("\n== 4/4. Площадки и налоги ==")
    a.flru_enabled = _yes(ask("FL.ru (оплата на российскую карту) — включить? [Д/н]", "д"))
    if a.flru_enabled:
        a.flru_pro = _yes(ask("Есть PRO-аккаунт на FL.ru? [д/Н]", "н"))
    a.pchel_enabled = _yes(ask("Pchel.net — включить? [Д/н]", "д"))
    a.freelancer_enabled = _yes(ask(
        "Freelancer.com (вывод денег из России может быть недоступен) — включить? [д/Н]", "н"))
    if _yes(ask("Вы самозанятый (НПД)? Налог 6% будет учтён в прибыли [д/Н]", "н")):
        a.tax_percent = 6.0

    config_path.write_text(render_config(a), encoding="utf-8")
    write_env(env_path, env)
    out(f"\nГотово: {config_path} и {env_path}")
    if "ANTHROPIC_API_KEY" not in env and not os.environ.get("ANTHROPIC_API_KEY"):
        out("Не забудьте вписать ANTHROPIC_API_KEY в .env")
    return config_path
