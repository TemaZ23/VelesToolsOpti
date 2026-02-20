#!/usr/bin/env python3
"""
🔴 VELES CRASH ANALYZER — CLI

Автономный анализ вероятности краша BTC с 200+ фичами,
мульти-моделями (XGBoost/LightGBM/CatBoost) и SHAP.

Одна команда — полный пайплайн:
  python scripts/crash_analysis/run.py

Опции:
  --symbol BTCUSDT     Символ (default: BTCUSDT)
  --years 5            Лет данных (default: 5)
  --crash-pct 5.0      Порог краша в % (default: 5.0)
  --crash-window 48    Окно краша в барах 15m (default: 48 = 12ч)
  --no-cache           Не использовать кеш
  --force-refresh      Перезагрузить все данные
  --fast               Быстрый режим (без complexity features, без Optuna)
  --optuna-trials 50   Количество trials Optuna (default: 50)
  --models xgboost,lightgbm,catboost
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="🔴 Veles Crash Analyzer — предсказание крашей BTC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Торговый символ")
    parser.add_argument("--years", type=int, default=5, help="Лет исторических данных")
    parser.add_argument("--crash-pct", type=float, default=5.0, help="Порог краша (%%)")
    parser.add_argument("--crash-window", type=int, default=48, help="Окно краша (баров 15m)")
    parser.add_argument("--no-cache", action="store_true", help="Не использовать кеш")
    parser.add_argument("--force-refresh", action="store_true", help="Перезагрузить все данные")
    parser.add_argument("--fast", action="store_true", help="Быстрый режим")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--models", default="xgboost,lightgbm,catboost", help="Модели (через запятую)")
    parser.add_argument("--no-plots", action="store_true", help="Не создавать графики")

    args = parser.parse_args()

    start_time = time.time()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              🔴 VELES CRASH ANALYZER                               ║")
    print("║              Продвинутый анализ вероятности краша                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"   Symbol:       {args.symbol}")
    print(f"   Years:        {args.years}")
    print(f"   Crash:        ≥{args.crash_pct}% за {args.crash_window} баров ({args.crash_window * 15 / 60:.0f}ч)")
    print(f"   Models:       {args.models}")
    print(f"   Mode:         {'FAST' if args.fast else 'FULL'}")
    print(f"   Optuna:       {'OFF' if args.fast else f'{args.optuna_trials} trials'}")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("📡 STEP 1/5: Загрузка данных с Binance")
    print("=" * 70)

    from crash_analysis.data_loader import load_all_data

    data = load_all_data(
        symbol=args.symbol,
        years=args.years,
        use_cache=not args.no_cache,
        force_refresh=args.force_refresh,
    )

    step1_time = time.time() - start_time
    print(f"\n   ⏱️  Шаг 1: {step1_time:.0f}с")
    for key, df in data.items():
        print(f"   {key}: {len(df):,} строк")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("🔧 STEP 2/5: Feature Engineering")
    print("=" * 70)

    from crash_analysis.features import build_features

    features_df = build_features(
        data=data,
        crash_threshold_pct=args.crash_pct,
        crash_window_bars=args.crash_window,
        include_complexity=not args.fast,
    )

    step2_time = time.time() - start_time - step1_time
    print(f"\n   ⏱️  Шаг 2: {step2_time:.0f}с")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: ML PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("🧠 STEP 3/5: ML Training & Validation")
    print("=" * 70)

    from crash_analysis.models import ModelConfig, run_pipeline

    model_config = ModelConfig(
        models=args.models.split(","),
        run_optuna=not args.fast,
        optuna_n_trials=args.optuna_trials,
        optuna_timeout=600 if not args.fast else 60,
        use_ensemble=len(args.models.split(",")) >= 2,
    )

    pipeline_result = run_pipeline(features_df, model_config)

    step3_time = time.time() - start_time - step1_time - step2_time
    print(f"\n   ⏱️  Шаг 3: {step3_time:.0f}с")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("🔍 STEP 4/5: Post-Analysis")
    print("=" * 70)

    from crash_analysis.analysis import (
        analyze_by_regime,
        detect_anomalies,
        detect_regimes,
        extract_rules,
    )

    # Regime detection
    print("\n   📊 Regime Detection...")
    df_15m = data.get("futures_15m")
    regimes = detect_regimes(df_15m) if df_15m is not None and len(df_15m) > 0 else None

    # Performance by regime
    regime_analysis = None
    if regimes is not None:
        print("\n   📊 Performance by Regime...")
        regime_analysis = analyze_by_regime(pipeline_result.predictions, regimes)

    # Anomaly detection
    print("\n   📊 Anomaly Detection...")
    anomalies = detect_anomalies(features_df)

    # Rule extraction
    print("\n   📊 Rule Extraction...")
    rules = extract_rules(pipeline_result, top_n=20)
    for i, rule in enumerate(rules[:10], 1):
        print(f"   {i:2d}. {rule['feature']} ({rule['importance']:.4f})")
        print(f"       → {rule['description']}")

    step4_time = time.time() - start_time - step1_time - step2_time - step3_time

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: REPORT & VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("📊 STEP 5/5: Report & Visualization")
    print("=" * 70)

    from crash_analysis.analysis import generate_plots, generate_report

    # Report
    report_path = generate_report(pipeline_result, rules, regime_analysis)

    # Plots
    if not args.no_plots:
        print("\n   📊 Generating plots...")
        try:
            plots = generate_plots(pipeline_result, data)
            print(f"   ✅ {len(plots)} графиков сохранено в output/")
        except Exception as e:
            print(f"   ⚠️  Plot generation failed: {e}")

    # Save predictions
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    pred_path = output_dir / "predictions.csv"
    pipeline_result.predictions.to_csv(pred_path, index=False)
    print(f"   💾 Predictions: {pred_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("✅ DONE")
    print(f"{'=' * 70}")
    print(f"   Общее время:      {total_time:.0f}с ({total_time / 60:.1f}мин)")
    print(f"   Данные:           {step1_time:.0f}с")
    print(f"   Features:         {step2_time:.0f}с")
    print(f"   ML Training:      {step3_time:.0f}с")
    print(f"   Analysis:         {step4_time:.0f}с")
    print(f"   Best Model:       {pipeline_result.best_model_name}")
    best_auc = (
        pipeline_result.models.get(pipeline_result.best_model_name, next(iter(pipeline_result.models.values())))
        .metrics.get("roc_auc", 0)
    )
    print(f"   Best AUC:         {best_auc:.4f}")
    print(f"   Features:         {pipeline_result.n_features}")
    print(f"   Output:           output/")
    print()


if __name__ == "__main__":
    main()
