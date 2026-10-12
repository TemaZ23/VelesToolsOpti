"""
Человекочитаемые настройки бота в терминах интерфейса Veles (как в статье LONG_STRATEGY).
"""

from __future__ import annotations

from .strategy import SL_FIXED, SL_NONE, SL_SIGNAL, TP_FIXED, TP_FIXED_OR_SIGNAL, BotConfig, Condition, Sleeve

TF_RU = {"1m": "1 минута", "5m": "5 минут", "15m": "15 минут", "30m": "30 минут", "1h": "1 час", "4h": "4 часа"}

VALUE_NAMES = {
    "RSI": "RSI",
    "CCI": "CCI",
    "CMO": "CMO",
    "WR": "Williams %R",
    "ATRP": "ATR%",
    "ADX": "ADX",
    "MFI": "MFI",
    "VOL": "Объём (номинальный)",
    "MACD": "MACD",
}

# (название, пояснение) для сигнальных серий; reverse — нужен ли «Обратный сигнал»
SIGNAL_NAMES: dict[str, tuple[str, str]] = {
    "RSI_LEVELS": ("Уровни RSI", "RSI пересёк 30 снизу вверх"),
    "CCI_LEVELS": ("Уровни CCI", "CCI пересёк −100 снизу вверх"),
    "STOCH_LEVELS": ("Стохастик уровни", "K вышла из зоны < 20 вверх"),
    "RSI_LEVELS_DN": ("Уровни RSI", "бот SHORT: RSI пересёк 70 сверху вниз"),
    "CCI_LEVELS_DN": ("Уровни CCI", "бот SHORT: CCI пересёк +100 сверху вниз"),
    "STOCH_LEVELS_DN": ("Стохастик уровни", "бот SHORT: K вышла из зоны > 80 вниз"),
    "MACD_ZERO_DOWN": ("Пересечение MACD и нулевого уровня", "MACD пересёк 0 сверху вниз"),
    "MACD_ZERO_UP": ("Пересечение MACD и нулевого уровня", "MACD пересёк 0 снизу вверх"),
}

CHANNELS: dict[str, tuple[str, str, str]] = {
    # серия: (индикатор, параметры, событие)
    "BBL": ("Полосы Боллинджера", "период 20, K 2", "закрытие ниже нижней полосы"),
    "BBU": ("Полосы Боллинджера", "период 20, K 2", "закрытие выше верхней полосы"),
    "BBL3": ("Полосы Боллинджера", "период 20, K 3", "закрытие ниже нижней полосы"),
    "BBU3": ("Полосы Боллинджера", "период 20, K 3", "закрытие выше верхней полосы"),
    "KCL": ("Канал Кельтнера", "период 20, K 2", "закрытие ниже нижней полосы"),
    "KCU": ("Канал Кельтнера", "период 20, K 2", "закрытие выше верхней полосы"),
    "KCL4": ("Канал Кельтнера", "период 20, K 4", "закрытие ниже нижней полосы"),
    "KCU4": ("Канал Кельтнера", "период 20, K 4", "закрытие выше верхней полосы"),
    "TZU": ("Turtle Zone", "период 200, InnerAmp 5.6", "закрытие выше верхней зоны"),
    "TZL": ("Turtle Zone", "период 200, InnerAmp 5.6", "закрытие ниже нижней зоны"),
}


def _fmt_num(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") if abs(v) >= 1000 else f"{v:g}"


def condition_ru(c: Condition, direction: int, context: str) -> str:
    """context: entry | order | tp | sl — определяет, нужен ли «обратный сигнал»."""
    kind, tf = c.series.split("@")
    tf_ru = TF_RU[tf]
    if kind in VALUE_NAMES:
        op = "Больше" if c.op == ">" else "Меньше"
        return f"{VALUE_NAMES[kind]} {tf_ru} {op} {_fmt_num(c.value)}"
    if kind in SIGNAL_NAMES:
        name, note = SIGNAL_NAMES[kind]
        reverse = ""
        if kind in ("MACD_ZERO_DOWN",) and direction > 0:
            reverse = " ! (обратный сигнал), от средней цены"
        if kind in ("MACD_ZERO_UP",) and direction < 0:
            reverse = " ! (обратный сигнал), от средней цены"
        return f"{name} {tf_ru}{reverse} — {note}"
    if kind in CHANNELS:
        name, params, event = CHANNELS[kind]
        # Для лонга «нормальный» сигнал входа/усреднения — ниже нижней полосы, выхода — выше верхней.
        normal_low = direction > 0 and context in ("entry", "order")
        normal_high = direction > 0 and context in ("tp",)
        if direction < 0:
            normal_low = context in ("tp",)
            normal_high = context in ("entry", "order")
        is_low = kind.endswith("L") or kind.endswith("L3") or kind.endswith("L4")
        reverse = "" if (is_low and normal_low) or (not is_low and normal_high) else " ! (обратный сигнал)"
        return f"{name} {tf_ru} ({params}){reverse} — {event}"
    return c.label()


def bot_ru(bot: BotConfig) -> str:
    d = bot.direction
    lines = [f"### {bot.name} — {'LONG' if d > 0 else 'SHORT'}", "", "**Условия открытия сделки (на закрытии бара):**"]
    lines += [f"{i}. {condition_ru(c, d, 'entry')}" for i, c in enumerate(bot.entry, 1)]
    lines += ["", f"**Сетка** (режим «Сигнал», отступ {'вниз' if d > 0 else 'вверх'} от цены входа):", ""]
    lines += ["| Ордер | Отступ | Объём | Сигналы (на закрытии бара) |", "|---|---|---|---|"]
    for i, o in enumerate(bot.orders, 1):
        conds = "; ".join(condition_ru(c, d, "order") for c in o.conditions) or (
            "— (первый ордер)" if i == 1 else "лимитный, без сигнала"
        )
        lines.append(f"| {i} | {o.indent:g}% | {o.volume:g}% | {conds} |")
    lines += ["", f"Подтяжка сетки ордеров: **{bot.pull_up:g}%**", ""]
    if bot.tp_mode == TP_FIXED:
        lines.append(f"**Тейк-профит:** простой, {bot.tp_pct:g}% от средней цены")
    else:
        head = "**Тейк-профит:** по сигналу (на закрытии бара)"
        if bot.tp_mode == TP_FIXED_OR_SIGNAL:
            head += f" + простой {bot.tp_pct:g}%"
        lines.append(f"{head}, минимальный P&L **{bot.tp_min_pnl:g}%**:")
        lines += [f"- {condition_ru(c, d, 'tp')}" for c in bot.tp_conditions]
    lines.append("")
    if bot.sl_mode == SL_NONE:
        lines.append("**Стоп-лосс:** нет")
    elif bot.sl_mode == SL_FIXED:
        lines.append(f"**Стоп-лосс:** {bot.sl_pct:g}% от средней цены")
    elif bot.sl_mode == SL_SIGNAL:
        extra = f", отступ от средней {bot.sl_indent:g}%" if bot.sl_indent > 0 else ""
        lines.append(f"**Стоп-лосс по сигналу** (на закрытии бара{extra}):")
        lines += [f"- {condition_ru(c, d, 'sl')}" for c in bot.sl_conditions]
    return "\n".join(lines)


def sleeve_ru(sl: Sleeve, deposit_example: float = 1000.0) -> str:
    per_bot = deposit_example * sl.bot_share / 100
    dyn = (
        f"при наборе {sl.dyn_block_order}-го ордера любой позицией блокировку −1, после закрытия этой сделки +1"
        if sl.dyn_block_order
        else "нет"
    )
    head = [
        bot_ru(sl.bot),
        "",
        f"**Депозит на бота:** {sl.bot_share:g}% от общего депозита (пример: {deposit_example:g} USDT → "
        f"{per_bot:g} USDT), плечо **{sl.leverage:g}x**, маржа «Кросс», режим позиции «Хеджирование».",
        f"**Блокировка:** {sl.block} («По позиции» и «По ботам»). Динамическая блокировка: {dyn}.",
    ]
    return "\n".join(head)
