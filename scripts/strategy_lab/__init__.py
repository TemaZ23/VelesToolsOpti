"""
Strategy Lab — Самоулучшающаяся торговая система.

NSGA-II многокритериальный оптимизатор ищет Pareto-оптимальные
комбинации индикаторов и параметров по 3 объективам:
  - median_return (profitability)
  - min_split_return (robustness)
  - worst_drawdown (risk)

Модули:
    indicators  — 50+ индикаторов/сигналов
    strategy    — параметризованная стратегия (long/short)
    backtest    — быстрый векторизованный бэктестер с leverage
    optimizer   — NSGA-II многокритериальный оптимизатор + эволюция индикаторов
    report      — визуализация и отчёты

Запуск:
    python scripts/strategy_lab/run.py
    python scripts/strategy_lab/run.py --target-monthly 30
"""
