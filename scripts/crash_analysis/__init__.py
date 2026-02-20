"""
BTC Crash Analysis — Advanced ML Pipeline
==========================================

Автономный CLI-инструмент для предсказания ликвидационных проливов BTC.
Работает напрямую с Binance API, без браузера.

Модули:
    data_loader   — Загрузка данных с Binance + кеширование
    features      — Продвинутый feature engineering (200+ признаков)
    models        — ML pipeline (XGBoost, LightGBM, CatBoost, Ensemble)
    analysis      — SHAP, regime detection, anomaly detection, отчёты

Запуск:
    python scripts/crash_analysis/run.py
    python scripts/crash_analysis/run.py --fast
    python scripts/crash_analysis/run.py --help
"""

from .data_loader import load_all_data
from .features import build_features
from .models import ModelConfig, PipelineResult, run_pipeline
from .analysis import (
    detect_regimes,
    detect_anomalies,
    extract_rules,
    analyze_by_regime,
    generate_plots,
    generate_report,
)

__all__ = [
    "load_all_data",
    "build_features",
    "ModelConfig",
    "PipelineResult",
    "run_pipeline",
    "detect_regimes",
    "detect_anomalies",
    "extract_rules",
    "analyze_by_regime",
    "generate_plots",
    "generate_report",
]
