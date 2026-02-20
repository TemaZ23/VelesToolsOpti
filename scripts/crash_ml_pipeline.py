#!/usr/bin/env python3
"""
BTC Crash Prediction ML Pipeline
=================================

Выявление рыночных состояний, предшествующих ликвидационным проливам BTC.

Тип задачи: Binary classification с сильным дисбалансом классов.
Таймфрейм: 15 минут.

Использование:
    1. Экспортировать CSV из Veles Tools (Crash Analysis → Export CSV)
    2. python crash_ml_pipeline.py --input crash_data.csv

Зависимости:
    pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn

Автор: Veles Tools
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Попытка импорта опциональных библиотек
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost не установлен. Установите: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM не установлен. Установите: pip install lightgbm")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP не установлен. Установите: pip install shap")


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Конфигурация ML пайплайна"""
    # Target variable
    crash_threshold_pct: float = 5.0  # Порог падения цены для crash
    crash_window_bars: int = 48       # Окно поиска crash (48 баров = 12 часов)
    
    # Feature engineering windows
    zscore_24h_bars: int = 96         # 24h = 96 баров по 15m
    zscore_72h_bars: int = 288        # 72h = 288 баров
    zscore_7d_bars: int = 672         # 7d = 672 бара
    
    # Walk-forward validation
    train_years: int = 2              # Лет для обучения
    test_months: int = 6              # Месяцев для теста
    
    # Model params
    random_state: int = 42
    n_estimators: int = 200
    
    # Output
    output_dir: str = "crash_analysis_results"


# ═══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """Загрузка CSV датасета"""
    print(f"\n📂 Загрузка данных из {filepath}...")
    
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   Загружено {len(df):,} строк")
    print(f"   Период: {df['timestamp'].min()} — {df['timestamp'].max()}")
    print(f"   Колонки: {list(df.columns)}")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_zscore(series: pd.Series, window: int) -> pd.Series:
    """Расчёт rolling Z-score"""
    rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
    rolling_std = series.rolling(window=window, min_periods=window//2).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def calculate_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Расчёт процентного изменения"""
    return series.pct_change(periods=periods) * 100


def calculate_delta(series: pd.Series, periods: int) -> pd.Series:
    """Расчёт абсолютного изменения"""
    return series.diff(periods=periods)


def engineer_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Feature Engineering
    
    Рассчитывает производные признаки для ML модели.
    """
    print("\n🔧 Feature Engineering...")
    
    df = df.copy()
    
    # ═══════════════════════════════════════════════════════════════════════
    # БАЗОВЫЕ ПРИЗНАКИ (если есть в данных)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Taker Delta Ratio (отношение sell/buy объёмов)
    if 'takerSellVolume' in df.columns and 'takerBuyVolume' in df.columns:
        df['taker_delta_ratio'] = df['takerSellVolume'] / df['takerBuyVolume'].replace(0, np.nan)
        df['taker_imbalance'] = (df['takerSellVolume'] - df['takerBuyVolume']) / (
            df['takerSellVolume'] + df['takerBuyVolume']
        ).replace(0, np.nan)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Z-SCORES (нормализованные отклонения от среднего)
    # ═══════════════════════════════════════════════════════════════════════
    
    # OI Z-score 24h
    if 'openInterest' in df.columns:
        df['oi_zscore_24h'] = calculate_zscore(df['openInterest'], config.zscore_24h_bars)
        df['oi_change_pct_4h'] = calculate_pct_change(df['openInterest'], 16)
        df['oi_change_pct_24h'] = calculate_pct_change(df['openInterest'], config.zscore_24h_bars)
    
    # Funding Z-score 24h
    if 'fundingRate' in df.columns:
        df['funding_zscore_24h'] = calculate_zscore(df['fundingRate'], config.zscore_24h_bars)
        # Экстремальный funding (> 0.1% за 8h = перегретый лонг)
        df['funding_extreme'] = (df['fundingRate'].abs() > 0.001).astype(int)
    
    # ATR Z-score 72h (волатильность относительно нормы)
    if 'atr14' in df.columns:
        df['atr_zscore_72h'] = calculate_zscore(df['atr14'], config.zscore_72h_bars)
        df['atr_expanding'] = (df['atr14'] > df['atr14'].rolling(24).mean()).astype(int)
    
    # Volume Z-score 24h
    if 'priceVolume' in df.columns:
        df['volume_zscore_24h'] = calculate_zscore(df['priceVolume'], config.zscore_24h_bars)
        df['volume_spike'] = (df['volume_zscore_24h'] > 2).astype(int)
    
    # Fear & Greed Z-score 7d
    if 'fearGreedIndex' in df.columns:
        df['feargreed_zscore_7d'] = calculate_zscore(df['fearGreedIndex'], config.zscore_7d_bars)
        df['feargreed_extreme_fear'] = (df['fearGreedIndex'] <= 20).astype(int)
        df['feargreed_extreme_greed'] = (df['fearGreedIndex'] >= 80).astype(int)
    
    # Basis Z-score 24h
    if 'spotFuturesBasis' in df.columns:
        df['basis_zscore_24h'] = calculate_zscore(df['spotFuturesBasis'], config.zscore_24h_bars)
        df['basis_negative'] = (df['spotFuturesBasis'] < 0).astype(int)
        df['basis_extreme'] = (df['spotFuturesBasis'].abs() > df['spotFuturesBasis'].abs().quantile(0.9)).astype(int)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ЦЕНОВЫЕ ПРИЗНАКИ
    # ═══════════════════════════════════════════════════════════════════════
    
    if 'priceClose' in df.columns:
        # Изменения цены за разные периоды
        df['price_change_pct_1h'] = calculate_pct_change(df['priceClose'], 4)
        df['price_change_pct_4h'] = calculate_pct_change(df['priceClose'], 16)
        df['price_change_pct_24h'] = calculate_pct_change(df['priceClose'], 96)
        
        # Расстояние от локального максимума
        df['price_from_high_24h'] = (
            df['priceClose'] / df['priceClose'].rolling(96).max() - 1
        ) * 100
        
        # Расстояние от локального минимума
        df['price_from_low_24h'] = (
            df['priceClose'] / df['priceClose'].rolling(96).min() - 1
        ) * 100
        
        # RSI-подобный индикатор
        delta = df['priceClose'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Перекупленность/перепроданность
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    
    # ═══════════════════════════════════════════════════════════════════════
    # КОМБИНИРОВАННЫЕ ПРИЗНАКИ
    # ═══════════════════════════════════════════════════════════════════════
    
    # Признак "перегретый рынок"
    overheated_conditions = []
    if 'funding_extreme' in df.columns:
        overheated_conditions.append(df['funding_extreme'])
    if 'feargreed_extreme_greed' in df.columns:
        overheated_conditions.append(df['feargreed_extreme_greed'])
    if 'rsi_overbought' in df.columns:
        overheated_conditions.append(df['rsi_overbought'])
    
    if overheated_conditions:
        df['market_overheated'] = sum(overheated_conditions)
    
    # Признак "накопление давления"
    pressure_conditions = []
    if 'oi_zscore_24h' in df.columns:
        pressure_conditions.append((df['oi_zscore_24h'] > 1.5).astype(int))
    if 'volume_spike' in df.columns:
        pressure_conditions.append(df['volume_spike'])
    if 'atr_expanding' in df.columns:
        pressure_conditions.append(df['atr_expanding'])
    
    if pressure_conditions:
        df['pressure_building'] = sum(pressure_conditions)
    
    # Признак "медвежья дивергенция" (цена растёт, но OI падает)
    if 'price_change_pct_4h' in df.columns and 'oi_change_pct_4h' in df.columns:
        df['bearish_divergence'] = (
            (df['price_change_pct_4h'] > 1) & (df['oi_change_pct_4h'] < -2)
        ).astype(int)
    
    print(f"   Создано {len([c for c in df.columns if c not in ['timestamp']])} признаков")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET VARIABLE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_target(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Расчёт целевой переменной crash_next_Xh
    
    crash = 1, если в следующих N барах цена падает на X% или более
    """
    print(f"\n🎯 Расчёт target: crash ≥{config.crash_threshold_pct}% в течение {config.crash_window_bars} баров...")
    
    df = df.copy()
    
    # Находим минимальную цену в следующих N барах
    df['future_min_price'] = df['priceClose'].shift(-1).rolling(
        window=config.crash_window_bars,
        min_periods=1
    ).min().shift(-config.crash_window_bars + 1)
    
    # Расчёт падения
    df['future_drawdown_pct'] = (
        (df['future_min_price'] - df['priceClose']) / df['priceClose'] * 100
    )
    
    # Target: 1 если падение >= threshold
    df['crash_target'] = (df['future_drawdown_pct'] <= -config.crash_threshold_pct).astype(int)
    
    # Убираем последние N баров (нет future data)
    df.loc[df.index[-config.crash_window_bars:], 'crash_target'] = np.nan
    
    # Статистика
    valid_rows = df['crash_target'].notna()
    crash_rate = df.loc[valid_rows, 'crash_target'].mean()
    crash_count = df.loc[valid_rows, 'crash_target'].sum()
    
    print(f"   Всего crash событий: {int(crash_count):,} из {valid_rows.sum():,} ({crash_rate*100:.2f}%)")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    df: pd.DataFrame,
    config: PipelineConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Подготовка данных для обучения
    """
    print("\n📊 Подготовка данных...")
    
    # Список признаков для модели
    feature_cols = [
        col for col in df.columns 
        if col not in [
            'timestamp', 'priceClose', 'priceVolume', 'openInterest', 
            'fundingRate', 'takerBuyVolume', 'takerSellVolume', 'atr14',
            'fearGreedIndex', 'spotFuturesBasis',
            'future_min_price', 'future_drawdown_pct', 'crash_target',
            # Исключаем raw колонки
            'bidDepthDeltaPct1h', 'exchangeInflowZscore24h', 'reserveDelta24h',
            'liquidationDensityRatio', 'crashNext6h'
        ]
    ]
    
    # Убираем колонки с большим количеством NaN
    for col in feature_cols.copy():
        if df[col].isna().mean() > 0.5:
            feature_cols.remove(col)
            print(f"   ⚠️  Удалена колонка {col} (>{50}% NaN)")
    
    print(f"   Используется {len(feature_cols)} признаков")
    
    # Удаляем строки с NaN в target
    df = df.dropna(subset=['crash_target'])
    
    # Заполняем NaN в признаках
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    print(f"   Финальный датасет: {len(df):,} строк")
    
    return df, feature_cols


def walk_forward_split(
    df: pd.DataFrame,
    config: PipelineConfig
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward split для временных рядов
    
    Не используем random split — это приведёт к look-ahead bias!
    """
    print("\n📅 Walk-forward split...")
    
    splits = []
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Определяем периоды
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    # Начинаем с train_years лет данных
    train_end = min_date + pd.DateOffset(years=config.train_years)
    
    while train_end + pd.DateOffset(months=config.test_months) <= max_date:
        test_end = train_end + pd.DateOffset(months=config.test_months)
        
        train_mask = df['timestamp'] < train_end
        test_mask = (df['timestamp'] >= train_end) & (df['timestamp'] < test_end)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(train_df) > 1000 and len(test_df) > 100:
            splits.append((train_df, test_df))
            print(f"   Split {len(splits)}: Train {train_df['timestamp'].min().date()} — {train_df['timestamp'].max().date()} ({len(train_df):,} rows)")
            print(f"            Test  {test_df['timestamp'].min().date()} — {test_df['timestamp'].max().date()} ({len(test_df):,} rows)")
        
        # Сдвигаем окно
        train_end += pd.DateOffset(months=config.test_months)
    
    print(f"\n   Всего {len(splits)} fold(s)")
    
    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """Результат обучения модели"""
    name: str
    model: object
    roc_auc: float
    precision: float
    recall: float
    f1: float
    feature_importance: Optional[pd.DataFrame] = None
    y_pred_proba: Optional[np.ndarray] = None


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    config: PipelineConfig
) -> ModelResult:
    """Обучение Logistic Regression (baseline)"""
    
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=config.random_state
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Метрики
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Feature importance (коэффициенты)
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        name="Logistic Regression",
        model=model,
        roc_auc=roc_auc,
        precision=report.get('1', {}).get('precision', 0),
        recall=report.get('1', {}).get('recall', 0),
        f1=report.get('1', {}).get('f1-score', 0),
        feature_importance=importance,
        y_pred_proba=y_pred_proba
    )


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    config: PipelineConfig
) -> ModelResult:
    """Обучение Random Forest"""
    
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        class_weight='balanced',
        max_depth=10,
        min_samples_leaf=20,
        random_state=config.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        name="Random Forest",
        model=model,
        roc_auc=roc_auc,
        precision=report.get('1', {}).get('precision', 0),
        recall=report.get('1', {}).get('recall', 0),
        f1=report.get('1', {}).get('f1-score', 0),
        feature_importance=importance,
        y_pred_proba=y_pred_proba
    )


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    config: PipelineConfig
) -> Optional[ModelResult]:
    """Обучение XGBoost"""
    
    if not HAS_XGBOOST:
        return None
    
    # Расчёт scale_pos_weight для дисбаланса классов
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=config.random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        name="XGBoost",
        model=model,
        roc_auc=roc_auc,
        precision=report.get('1', {}).get('precision', 0),
        recall=report.get('1', {}).get('recall', 0),
        f1=report.get('1', {}).get('f1-score', 0),
        feature_importance=importance,
        y_pred_proba=y_pred_proba
    )


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    config: PipelineConfig
) -> Optional[ModelResult]:
    """Обучение LightGBM"""
    
    if not HAS_LIGHTGBM:
        return None
    
    model = lgb.LGBMClassifier(
        n_estimators=config.n_estimators,
        max_depth=6,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=config.random_state,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return ModelResult(
        name="LightGBM",
        model=model,
        roc_auc=roc_auc,
        precision=report.get('1', {}).get('precision', 0),
        recall=report.get('1', {}).get('recall', 0),
        f1=report.get('1', {}).get('f1-score', 0),
        feature_importance=importance,
        y_pred_proba=y_pred_proba
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_shap(
    model: object,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    model_name: str
) -> Optional[pd.DataFrame]:
    """SHAP анализ для интерпретации модели"""
    
    if not HAS_SHAP:
        print("   ⚠️  SHAP не установлен, пропускаем анализ")
        return None
    
    print(f"\n🔍 SHAP анализ для {model_name}...")
    
    try:
        # Создаём explainer
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_test[:1000])
        
        # Расчёт SHAP values (на подвыборке для скорости)
        sample_size = min(1000, len(X_test))
        X_sample = X_test[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        
        # Для бинарной классификации берём класс 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / f'shap_summary_{model_name.lower().replace(" ", "_")}.png', dpi=150)
        plt.close()
        
        # Feature importance from SHAP
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        print(f"   Сохранён SHAP summary plot")
        
        return shap_importance
        
    except Exception as e:
        print(f"   ⚠️  Ошибка SHAP: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# RULE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_high_risk_rules(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'crash_target',
    min_probability: float = 0.15,  # Минимальная вероятность для правила
    min_support: int = 50           # Минимальное количество примеров
) -> List[Dict]:
    """
    Извлечение правил для высокого риска crash
    
    Находит комбинации условий, при которых вероятность crash выше порога.
    """
    print(f"\n📋 Извлечение правил (P(crash) > {min_probability*100:.0f}%, support ≥ {min_support})...")
    
    rules = []
    base_crash_rate = df[target_col].mean()
    
    # Одиночные условия
    for col in feature_cols:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Пробуем разные квантили
            for q in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]:
                threshold = df[col].quantile(q)
                
                # Условие >
                mask = df[col] > threshold
                if mask.sum() >= min_support:
                    prob = df.loc[mask, target_col].mean()
                    lift = prob / base_crash_rate if base_crash_rate > 0 else 0
                    
                    if prob >= min_probability and lift > 1.5:
                        rules.append({
                            'conditions': [(col, '>', threshold)],
                            'probability': prob,
                            'support': mask.sum(),
                            'lift': lift
                        })
                
                # Условие <
                mask = df[col] < threshold
                if mask.sum() >= min_support:
                    prob = df.loc[mask, target_col].mean()
                    lift = prob / base_crash_rate if base_crash_rate > 0 else 0
                    
                    if prob >= min_probability and lift > 1.5:
                        rules.append({
                            'conditions': [(col, '<', threshold)],
                            'probability': prob,
                            'support': mask.sum(),
                            'lift': lift
                        })
    
    # Комбинации из 2 условий (топ правил)
    rules.sort(key=lambda x: -x['lift'])
    top_rules = rules[:20]
    
    combined_rules = []
    for i, rule1 in enumerate(top_rules):
        for rule2 in top_rules[i+1:]:
            # Проверяем что признаки разные
            if rule1['conditions'][0][0] != rule2['conditions'][0][0]:
                # Применяем оба условия
                cond1 = rule1['conditions'][0]
                cond2 = rule2['conditions'][0]
                
                if cond1[1] == '>':
                    mask1 = df[cond1[0]] > cond1[2]
                else:
                    mask1 = df[cond1[0]] < cond1[2]
                
                if cond2[1] == '>':
                    mask2 = df[cond2[0]] > cond2[2]
                else:
                    mask2 = df[cond2[0]] < cond2[2]
                
                combined_mask = mask1 & mask2
                
                if combined_mask.sum() >= min_support:
                    prob = df.loc[combined_mask, target_col].mean()
                    lift = prob / base_crash_rate if base_crash_rate > 0 else 0
                    
                    if prob >= min_probability and lift > 2.0:
                        combined_rules.append({
                            'conditions': [cond1, cond2],
                            'probability': prob,
                            'support': combined_mask.sum(),
                            'lift': lift
                        })
    
    # Объединяем и сортируем
    all_rules = rules + combined_rules
    all_rules.sort(key=lambda x: (-x['lift'], -x['probability']))
    
    print(f"   Найдено {len(all_rules)} правил")
    
    return all_rules[:30]  # Топ-30 правил


def format_rules(rules: List[Dict], base_crash_rate: float) -> str:
    """Форматирование правил в читаемый вид"""
    
    output = []
    output.append("\n" + "="*80)
    output.append("ПРАВИЛА ВЫСОКОГО РИСКА CRASH")
    output.append(f"Базовая вероятность crash: {base_crash_rate*100:.2f}%")
    output.append("="*80)
    
    for i, rule in enumerate(rules, 1):
        conditions_str = " И ".join([
            f"{c[0]} {c[1]} {c[2]:.4f}" for c in rule['conditions']
        ])
        
        output.append(f"\n#{i}. {conditions_str}")
        output.append(f"    → P(crash) = {rule['probability']*100:.1f}%")
        output.append(f"    → Lift = {rule['lift']:.2f}x")
        output.append(f"    → Support = {rule['support']} примеров")
    
    return "\n".join(output)


# ═══════════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(
    results: List[ModelResult],
    y_test: np.ndarray,
    output_dir: Path
):
    """Визуализация результатов"""
    
    print("\n📊 Создание визуализаций...")
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for result in results:
        if result.y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, result.y_pred_proba)
            plt.plot(fpr, tpr, label=f'{result.name} (AUC={result.roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150)
    plt.close()
    
    # Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for result in results:
        if result.y_pred_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, result.y_pred_proba)
            plt.plot(recall, precision, label=f'{result.name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curves.png', dpi=150)
    plt.close()
    
    # Feature Importance (лучшая модель)
    best_result = max(results, key=lambda x: x.roc_auc)
    if best_result.feature_importance is not None:
        plt.figure(figsize=(12, 10))
        top_features = best_result.feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top 20 Feature Importance ({best_result.name})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150)
        plt.close()
    
    print(f"   Графики сохранены в {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(input_file: str, config: PipelineConfig):
    """Запуск полного ML пайплайна"""
    
    print("\n" + "="*80)
    print("🚀 BTC CRASH PREDICTION ML PIPELINE")
    print("="*80)
    
    # Создаём output директорию
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Загрузка данных
    df = load_data(input_file)
    
    # 2. Feature Engineering
    df = engineer_features(df, config)
    
    # 3. Target Variable
    df = calculate_target(df, config)
    
    # 4. Подготовка данных
    df, feature_cols = prepare_data(df, config)
    
    # 5. Walk-forward splits
    splits = walk_forward_split(df, config)
    
    if not splits:
        print("❌ Недостаточно данных для walk-forward validation")
        return
    
    # 6. Обучение моделей на каждом fold
    all_results = []
    
    for fold_idx, (train_df, test_df) in enumerate(splits):
        print(f"\n{'='*40}")
        print(f"FOLD {fold_idx + 1}/{len(splits)}")
        print(f"{'='*40}")
        
        # Prepare features
        X_train = train_df[feature_cols].values
        y_train = train_df['crash_target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['crash_target'].values
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        results = []
        
        print("\n🤖 Обучение моделей...")
        
        # Logistic Regression
        lr_result = train_logistic_regression(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_cols, config
        )
        results.append(lr_result)
        print(f"   {lr_result.name}: AUC={lr_result.roc_auc:.3f}, Precision={lr_result.precision:.3f}, Recall={lr_result.recall:.3f}")
        
        # Random Forest
        rf_result = train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_cols, config
        )
        results.append(rf_result)
        print(f"   {rf_result.name}: AUC={rf_result.roc_auc:.3f}, Precision={rf_result.precision:.3f}, Recall={rf_result.recall:.3f}")
        
        # XGBoost
        xgb_result = train_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_cols, config
        )
        if xgb_result:
            results.append(xgb_result)
            print(f"   {xgb_result.name}: AUC={xgb_result.roc_auc:.3f}, Precision={xgb_result.precision:.3f}, Recall={xgb_result.recall:.3f}")
        
        # LightGBM
        lgb_result = train_lightgbm(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_cols, config
        )
        if lgb_result:
            results.append(lgb_result)
            print(f"   {lgb_result.name}: AUC={lgb_result.roc_auc:.3f}, Precision={lgb_result.precision:.3f}, Recall={lgb_result.recall:.3f}")
        
        all_results.append((fold_idx, results, y_test))
    
    # 7. Aggregate results
    print("\n" + "="*80)
    print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ (усреднённые по всем folds)")
    print("="*80)
    
    model_names = set()
    for _, results, _ in all_results:
        for r in results:
            model_names.add(r.name)
    
    for model_name in sorted(model_names):
        aucs = []
        precisions = []
        recalls = []
        
        for _, results, _ in all_results:
            for r in results:
                if r.name == model_name:
                    aucs.append(r.roc_auc)
                    precisions.append(r.precision)
                    recalls.append(r.recall)
        
        if aucs:
            print(f"\n{model_name}:")
            print(f"   ROC AUC:   {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
            print(f"   Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
            print(f"   Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    
    # 8. SHAP analysis на последнем fold
    last_fold_idx, last_results, last_y_test = all_results[-1]
    best_result = max(last_results, key=lambda x: x.roc_auc)
    
    # Prepare test data for SHAP
    _, test_df = splits[-1]
    X_test = test_df[feature_cols].values
    scaler = StandardScaler()
    X_train_last = splits[-1][0][feature_cols].values
    scaler.fit(X_train_last)
    X_test_scaled = scaler.transform(X_test)
    
    shap_importance = analyze_shap(
        best_result.model,
        X_test_scaled,
        feature_cols,
        output_dir,
        best_result.name
    )
    
    # 9. Extract rules
    rules = extract_high_risk_rules(df, feature_cols)
    base_crash_rate = df['crash_target'].mean()
    rules_text = format_rules(rules, base_crash_rate)
    print(rules_text)
    
    # Save rules to file
    with open(output_dir / 'crash_rules.txt', 'w', encoding='utf-8') as f:
        f.write(rules_text)
    
    # 10. Visualizations
    plot_results(last_results, last_y_test, output_dir)
    
    # 11. Summary
    print("\n" + "="*80)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("="*80)
    print(f"\nРезультаты сохранены в: {output_dir.absolute()}")
    print(f"  - roc_curves.png")
    print(f"  - precision_recall_curves.png")
    print(f"  - feature_importance.png")
    print(f"  - crash_rules.txt")
    if HAS_SHAP:
        print(f"  - shap_summary_*.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='BTC Crash Prediction ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовый запуск
  python crash_ml_pipeline.py --input crash_data.csv

  # С кастомными параметрами
  python crash_ml_pipeline.py --input crash_data.csv --crash-threshold 7 --crash-window 24

  # Только анализ признаков
  python crash_ml_pipeline.py --input crash_data.csv --output-dir my_results
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Путь к CSV файлу с данными'
    )
    
    parser.add_argument(
        '--crash-threshold',
        type=float,
        default=5.0,
        help='Порог падения цены для crash (%%). Default: 5.0'
    )
    
    parser.add_argument(
        '--crash-window',
        type=int,
        default=48,
        help='Окно поиска crash (в барах по 15m). Default: 48 (12 часов)'
    )
    
    parser.add_argument(
        '--train-years',
        type=int,
        default=2,
        help='Количество лет для обучения. Default: 2'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='crash_analysis_results',
        help='Директория для результатов. Default: crash_analysis_results'
    )
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        crash_threshold_pct=args.crash_threshold,
        crash_window_bars=args.crash_window,
        train_years=args.train_years,
        output_dir=args.output_dir
    )
    
    run_pipeline(args.input, config)


if __name__ == '__main__':
    main()
