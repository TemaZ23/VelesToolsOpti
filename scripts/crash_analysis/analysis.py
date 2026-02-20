"""
Пост-анализ: генерация отчётов, визуализаций и интерпретация.

- Regime detection (HMM или эвристический)
- Anomaly detection (Isolation Forest)
- Rule extraction из деревьев
- Temporal pattern analysis
- Визуализации: SHAP, feature importance, prediction timeline
- HTML-отчёт
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .models import ModelResult, PipelineResult


OUTPUT_DIR = Path("output")


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regimes(df_15m: pd.DataFrame, n_regimes: int = 3) -> pd.Series:
    """
    Определение рыночных режимов: Bull / Bear / Ranging.

    Использует кластеризацию на основе:
    - rolling return
    - rolling volatility
    - volume z-score

    Falls back to simple percentile-based classification если HMM недоступен.
    """
    close = df_15m["close"]
    vol = df_15m["volume"]

    ret_96 = close.pct_change(96)  # 24h return
    vol_96 = np.log(close / close.shift(1)).rolling(96).std()
    vol_z = (vol - vol.rolling(96).mean()) / vol.rolling(96).std()

    # Try HMM first
    try:
        from hmmlearn.hmm import GaussianHMM

        features = pd.DataFrame({
            "ret": ret_96,
            "vol": vol_96,
            "vol_z": vol_z,
        }).dropna()

        hmm = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        hmm.fit(features.values)
        hidden_states = hmm.predict(features.values)

        # Map states to Bull/Bear/Range by mean return
        state_returns = {
            state: features["ret"].iloc[hidden_states == state].mean()
            for state in range(n_regimes)
        }
        sorted_states = sorted(state_returns, key=lambda x: state_returns[x])
        label_map = {}
        if n_regimes >= 3:
            label_map[sorted_states[0]] = "Bear"
            label_map[sorted_states[-1]] = "Bull"
            for s in sorted_states[1:-1]:
                label_map[s] = "Range"
        else:
            label_map[sorted_states[0]] = "Bear"
            label_map[sorted_states[-1]] = "Bull"

        regimes = pd.Series("Unknown", index=df_15m.index)
        regimes.iloc[features.index] = [label_map[s] for s in hidden_states]

        print(f"   ✅ HMM: {n_regimes} режимов обнаружено")
        for label in ["Bull", "Bear", "Range"]:
            pct = (regimes == label).mean() * 100
            if pct > 0:
                print(f"      {label}: {pct:.1f}%")

        return regimes

    except ImportError:
        print("   ⚠️  hmmlearn не установлен, используем percentile-based")

    # Fallback: percentile-based
    regimes = pd.Series("Range", index=df_15m.index)
    p33 = ret_96.quantile(0.33)
    p67 = ret_96.quantile(0.67)
    regimes[ret_96 > p67] = "Bull"
    regimes[ret_96 < p33] = "Bear"

    for label in ["Bull", "Bear", "Range"]:
        pct = (regimes == label).mean() * 100
        print(f"      {label}: {pct:.1f}%")

    return regimes


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(features_df: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    """
    Isolation Forest для поиска аномальных рыночных состояний.

    Returns: Series с 1 (аномалия) / 0 (нормально)
    """
    from sklearn.ensemble import IsolationForest

    exclude = {"timestamp", "target"}
    feat_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in ["float64", "float32", "int64"]]

    X = features_df[feat_cols].fillna(0).values

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X)

    anomalies = (labels == -1).astype(int)
    result = pd.Series(anomalies, index=features_df.index)

    n_anomalies = anomalies.sum()
    print(f"   🔴 Аномалий: {n_anomalies} ({n_anomalies / len(anomalies) * 100:.1f}%)")

    # Check overlap with actual crashes
    if "target" in features_df.columns:
        target = features_df["target"].fillna(0).astype(int)
        overlap = ((result == 1) & (target == 1)).sum()
        if target.sum() > 0:
            print(f"   🔴 Аномалии → краши: {overlap}/{int(target.sum())} "
                  f"({overlap / target.sum() * 100:.1f}% отзыв)")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# RULE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_rules(result: PipelineResult, top_n: int = 20) -> list[dict[str, Any]]:
    """
    Извлечь человекочитаемые правила из модели.

    Основано на top SHAP features и пороговых значениях.
    """
    rules: list[dict[str, Any]] = []

    best = result.models.get(result.best_model_name)
    if best is None:
        best = next(iter(result.models.values()), None)

    if best is None:
        return rules

    fi = best.feature_importance.head(top_n)

    for _, row in fi.iterrows():
        feat = row["feature"]
        imp = row["importance"]
        rules.append({
            "feature": feat,
            "importance": round(float(imp), 4),
            "description": _describe_feature(feat),
        })

    return rules


def _describe_feature(name: str) -> str:
    """Человекочитаемое описание фичи."""
    descriptions: dict[str, str] = {
        "funding_rate": "Ставка финансирования (высокая = overleveraged longs)",
        "basis": "Спот-фьючерс базис (высокий = бычий excess)",
        "fear_greed": "Индекс страха/жадности (>80 = экстремальная жадность)",
        "hurst_96": "Экспонента Хёрста (<0.5 = mean-revert, >0.5 = тренд)",
        "vol_ratio_24_96": "Отношение волатильности (>1 = рост волатильности)",
        "taker_delta": "Дельта покупателей/продавцов (сильный bias = потенциальный разворот)",
        "sample_entropy": "Энтропия рынка (низкая = предсказуемость, высокая = хаос)",
        "overlevered": "Комбо: высокий грид + funding + basis = опасность",
        "squeeze": "Сжатие Боллинджера + низкий объём = готовится движение",
    }

    for key, desc in descriptions.items():
        if key in name:
            return desc

    if name.startswith("ret_"):
        return f"Доходность за {name.split('_')[1]} баров"
    if name.startswith("vol_zscore_"):
        return f"Z-score объёма (окно {name.split('_')[-1]})"
    if name.startswith("rsi_"):
        return f"RSI период {name.split('_')[-1]}"
    if "zscore" in name:
        return f"Z-score отклонение: {name}"
    if "ema" in name:
        return f"Расстояние до EMA: {name}"

    return name


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE BY REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_by_regime(
    predictions: pd.DataFrame,
    regimes: pd.Series,
) -> dict[str, dict[str, float]]:
    """Анализ качества модели по рыночным режимам."""
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

    predictions = predictions.copy()
    predictions["regime"] = regimes.values[:len(predictions)]

    results: dict[str, dict[str, float]] = {}
    for regime in predictions["regime"].unique():
        if regime == "Unknown":
            continue
        mask = predictions["regime"] == regime
        subset = predictions[mask].dropna(subset=["pred_proba", "actual"])
        if len(subset) < 10:
            continue

        y_true = subset["actual"].astype(int).values
        y_proba = subset["pred_proba"].values
        y_pred = subset["pred_label"].astype(int).values

        metrics: dict[str, float] = {
            "n_samples": int(len(subset)),
            "crash_rate": float(y_true.mean()),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = 0.0

        results[regime] = metrics
        print(f"   {regime:8s}: N={metrics['n_samples']:>6,} | CrashRate={metrics['crash_rate']:.3f} | "
              f"AUC={metrics['roc_auc']:.3f} | F1={metrics['f1']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_plots(result: PipelineResult, data: dict[str, pd.DataFrame], output_dir: Optional[Path] = None) -> list[Path]:
    """Генерация визуализаций."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    plots: list[Path] = []

    # ── 1. Feature Importance (Top 30) ──
    fig, ax = plt.subplots(figsize=(12, 10))
    best = result.models.get(result.best_model_name)
    if best is None:
        best = next(iter(result.models.values()))

    top30 = best.feature_importance.head(30)
    ax.barh(range(len(top30)), top30["importance"].values, color="#1f77b4")
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Top 30 Features ({result.best_model_name})", fontsize=14)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plots.append(path)
    print(f"   📊 Saved: {path}")

    # ── 2. SHAP Summary ──
    if best.shap_values is not None:
        try:
            import shap
            fig, ax = plt.subplots(figsize=(12, 10))
            shap.summary_plot(
                best.shap_values,
                feature_names=best.shap_feature_names,
                show=False,
                max_display=30,
            )
            path = output_dir / "shap_summary.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            plots.append(path)
            print(f"   📊 Saved: {path}")
        except Exception as e:
            print(f"   ⚠️  SHAP plot failed: {e}")

    # ── 3. Prediction Timeline ──
    preds = result.predictions.dropna(subset=["pred_proba"])
    if len(preds) > 0 and "futures_15m" in data:
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

        ts = pd.to_datetime(preds["timestamp"])
        price = data["futures_15m"].set_index("timestamp")["close"].reindex(preds["timestamp"])

        # Price
        axes[0].plot(ts, price.values, color="#333", linewidth=0.5, alpha=0.8)
        crash_mask = preds["actual"] == 1
        axes[0].scatter(ts[crash_mask], price.values[crash_mask], color="red", s=5, alpha=0.5, label="Actual Crash")
        axes[0].set_ylabel("Price")
        axes[0].set_title("BTC Price + Crash Events")
        axes[0].legend()

        # Probability
        axes[1].fill_between(ts, 0, preds["pred_proba"].values, alpha=0.3, color="orange")
        axes[1].plot(ts, preds["pred_proba"].values, color="orange", linewidth=0.5)
        axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Crash Probability")
        axes[1].set_title("Model Prediction")
        axes[1].set_ylim(0, 1)

        # Rolling accuracy
        window = 96 * 7  # 1 week
        correct = (preds["pred_label"] == preds["actual"]).astype(float)
        rolling_acc = correct.rolling(window, min_periods=100).mean()
        axes[2].plot(ts, rolling_acc.values, color="green", linewidth=1)
        axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        axes[2].set_ylabel("Rolling Accuracy (1w)")
        axes[2].set_title("Prediction Quality Over Time")
        axes[2].set_ylim(0, 1)

        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()

        fig.tight_layout()
        path = output_dir / "prediction_timeline.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)
        print(f"   📊 Saved: {path}")

    # ── 4. Model Comparison ──
    if len(result.models) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        model_names = list(result.models.keys())
        metrics_list = ["roc_auc", "f1", "precision", "recall"]
        x = np.arange(len(model_names))
        width = 0.2
        for i, metric in enumerate(metrics_list):
            vals = [result.models[mn].metrics.get(metric, 0) for mn in model_names]
            ax.bar(x + i * width, vals, width, label=metric)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        path = output_dir / "model_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)
        print(f"   📊 Saved: {path}")

    return plots


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(
    result: PipelineResult,
    rules: list[dict[str, Any]],
    regime_analysis: Optional[dict[str, dict[str, float]]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Генерация текстового отчёта."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  CRASH ANALYSIS REPORT")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    lines.append(f"\n📊 DATASET")
    lines.append(f"   Samples:  {result.n_samples:,}")
    lines.append(f"   Features: {result.n_features}")
    lines.append(f"   Crashes:  {result.n_crashes} ({result.crash_rate * 100:.2f}%)")

    lines.append(f"\n🧠 MODELS")
    for name, mr in result.models.items():
        lines.append(f"\n   {name.upper()}")
        lines.append(f"   {'─' * 40}")
        m = mr.metrics
        lines.append(f"   ROC-AUC:   {m.get('roc_auc', 0):.4f}")
        lines.append(f"   F1:        {m.get('f1', 0):.4f}")
        lines.append(f"   Precision: {m.get('precision', 0):.4f}")
        lines.append(f"   Recall:    {m.get('recall', 0):.4f}")
        lines.append(f"   Threshold: {mr.optimal_threshold:.3f}")

        if mr.fold_metrics:
            lines.append(f"   Walk-Forward folds:")
            for i, fm in enumerate(mr.fold_metrics):
                lines.append(f"     Fold {i + 1}: AUC={fm.get('roc_auc', 0):.4f} F1={fm.get('f1', 0):.4f}")

    if result.ensemble_metrics:
        lines.append(f"\n   ENSEMBLE")
        lines.append(f"   {'─' * 40}")
        em = result.ensemble_metrics
        lines.append(f"   ROC-AUC:   {em.get('roc_auc', 0):.4f}")
        lines.append(f"   F1:        {em.get('f1', 0):.4f}")

    lines.append(f"\n🏆 BEST: {result.best_model_name}")

    # Rules
    if rules:
        lines.append(f"\n📋 TOP CRASH SIGNALS (by importance)")
        lines.append(f"   {'─' * 40}")
        for i, rule in enumerate(rules[:15], 1):
            lines.append(f"   {i:2d}. {rule['feature']}")
            lines.append(f"       Importance: {rule['importance']:.4f}")
            lines.append(f"       {rule['description']}")

    # Regime analysis
    if regime_analysis:
        lines.append(f"\n📊 PERFORMANCE BY MARKET REGIME")
        lines.append(f"   {'─' * 40}")
        for regime, metrics in regime_analysis.items():
            lines.append(f"\n   {regime}:")
            lines.append(f"     Samples:    {metrics['n_samples']:,}")
            lines.append(f"     Crash Rate: {metrics['crash_rate']:.3f}")
            lines.append(f"     AUC:        {metrics['roc_auc']:.3f}")
            lines.append(f"     F1:         {metrics['f1']:.3f}")

    lines.append(f"\n{'=' * 70}")
    lines.append("  END OF REPORT")
    lines.append(f"{'=' * 70}")

    report_text = "\n".join(lines)

    path = output_dir / "crash_analysis_report.txt"
    path.write_text(report_text, encoding="utf-8")
    print(f"   📄 Report: {path}")

    # Also print to console
    print("\n" + report_text)

    return path
