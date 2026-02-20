"""
ML Pipeline для предсказания крашей.

Модели:
- XGBoost
- LightGBM
- CatBoost
- Stacking Ensemble (Ridge meta-learner)

Валидация:
- Walk-Forward (Time Series Split)
- Purged Cross-Validation (gap between train/test)

Интерпретация:
- SHAP values
- Feature importance
- Partial Dependence

Гиперпараметры:
- Optuna (Bayesian optimization)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Конфигурация ML пайплайна."""

    # Walk-Forward validation
    n_splits: int = 5
    purge_gap: int = 96  # 24 часа (15m bars) gap между train/test

    # General
    scale_features: bool = True
    fill_na_strategy: str = "median"  # median, zero
    max_features_corr: float = 0.95  # drop features with higher correlation

    # Class imbalance
    use_smote: bool = False  # SMOTE для Oversampling
    class_weight: str = "balanced"  # balanced, None

    # Optuna
    run_optuna: bool = True
    optuna_n_trials: int = 50
    optuna_timeout: int = 600  # seconds

    # Models to train
    models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "catboost"])
    use_ensemble: bool = True

    # Threshold tuning
    optimize_threshold: bool = True


@dataclass
class ModelResult:
    """Результат одной модели."""

    name: str
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]]
    feature_importance: pd.DataFrame
    best_params: dict[str, Any]
    shap_values: Optional[np.ndarray] = None
    shap_feature_names: Optional[list[str]] = None
    optimal_threshold: float = 0.5


@dataclass
class PipelineResult:
    """Результат всего пайплайна."""

    models: dict[str, ModelResult]
    ensemble_metrics: Optional[dict[str, float]]
    feature_names: list[str]
    n_features: int
    n_samples: int
    n_crashes: int
    crash_rate: float
    removed_correlated: list[str]
    best_model_name: str
    predictions: pd.DataFrame  # timestamp, actual, pred_proba, pred_label


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(
    df: pd.DataFrame, config: ModelConfig
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series]:
    """
    Предобработка: NA, scaling, correlation filter.

    Returns:
        X, y, feature_names, timestamps
    """
    # Separate features
    exclude = {"timestamp", "target"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    y = df["target"].values
    timestamps = df["timestamp"]

    # Fill NaN
    if config.fill_na_strategy == "median":
        medians = X.median()
        X = X.fillna(medians)
    else:
        X = X.fillna(0)

    # Remove constant columns
    constant_cols = X.columns[X.nunique() <= 1].tolist()
    if constant_cols:
        print(f"   🗑️  Удалено {len(constant_cols)} константных фичей")
        X = X.drop(columns=constant_cols)

    # Remove highly correlated features
    removed_corr: list[str] = []
    if config.max_features_corr < 1.0:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > config.max_features_corr)]
        if to_drop:
            print(f"   🗑️  Удалено {len(to_drop)} коррелированных фичей (r>{config.max_features_corr})")
            removed_corr = to_drop
            X = X.drop(columns=to_drop)

    feature_names = X.columns.tolist()
    print(f"   📐 Итого фичей для обучения: {len(feature_names)}")

    return X.values, y, feature_names, timestamps


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _purged_time_series_split(
    n_samples: int, n_splits: int, purge_gap: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time Series Split с purge gap для предотвращения data leakage."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_idx, test_idx in tscv.split(np.zeros(n_samples)):
        # Remove purge_gap rows from end of train
        if len(train_idx) > purge_gap:
            train_idx = train_idx[:-purge_gap]
        splits.append((train_idx, test_idx))

    return splits


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Вычислить метрики."""
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics["roc_auc"] = 0.0
    return metrics


def _optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Найти оптимальный порог по F1-score."""
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
    optuna_params: Optional[dict] = None,
) -> Any:
    """Обучить XGBoost."""
    import xgboost as xgb

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight if config.class_weight == "balanced" else 1.0,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
    }
    if optuna_params:
        params.update(optuna_params)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
    optuna_params: Optional[dict] = None,
) -> Any:
    """Обучить LightGBM."""
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": config.class_weight == "balanced",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
    }
    if optuna_params:
        params.update(optuna_params)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def _train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ModelConfig,
    optuna_params: Optional[dict] = None,
) -> Any:
    """Обучить CatBoost."""
    from catboost import CatBoostClassifier

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "scale_pos_weight": scale_pos_weight if config.class_weight == "balanced" else 1.0,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "AUC",
    }
    if optuna_params:
        params.update(optuna_params)

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=0,
    )
    return model


TRAINERS = {
    "xgboost": _train_xgboost,
    "lightgbm": _train_lightgbm,
    "catboost": _train_catboost,
}


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _optuna_tune(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: ModelConfig,
) -> dict[str, Any]:
    """Bayesian hyperparameter optimization с Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("      ⚠️  Optuna не установлен, используются дефолтные параметры")
        return {}

    def objective(trial: optuna.Trial) -> float:
        if model_name == "xgboost":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        elif model_name == "lightgbm":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 256),
            }
        elif model_name == "catboost":
            params = {
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "iterations": trial.suggest_int("iterations", 100, 1500),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            }
        else:
            return 0.0

        scores: list[float] = []
        for train_idx, val_idx in splits[:3]:  # use fewer folds for speed
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            try:
                trainer = TRAINERS[model_name]
                model = trainer(X_tr, y_tr, X_val, y_val, config, params)
                y_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_proba)
                scores.append(auc)
            except Exception:
                scores.append(0.5)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.optuna_n_trials, timeout=config.optuna_timeout)

    print(f"      🏆 Best AUC: {study.best_value:.4f}")
    return study.best_params


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_shap(model: Any, X: np.ndarray, feature_names: list[str], model_name: str) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Вычислить SHAP values."""
    try:
        import shap

        if model_name in ("xgboost", "lightgbm", "catboost"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))

        # Use subsample for speed
        sample_size = min(5000, len(X))
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]

        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1

        return shap_vals, feature_names
    except Exception as e:
        print(f"      ⚠️  SHAP failed: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def _train_ensemble(
    models_results: dict[str, ModelResult],
    all_oof_probas: dict[str, np.ndarray],
    y: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Stacking Ensemble: берём OOF-предсказания каждой модели,
    обучаем Ridge meta-learner.
    """
    # Stack OOF predictions
    model_names = sorted(all_oof_probas.keys())
    stack_matrix = np.column_stack([all_oof_probas[mn] for mn in model_names])

    # Mask: only where all models have predictions
    mask = ~np.isnan(stack_matrix).any(axis=1) & ~np.isnan(y)
    if mask.sum() < 100:
        print("      ⚠️  Недостаточно OOF для ensemble")
        return np.full(len(y), np.nan), {}

    X_stack = stack_matrix[mask]
    y_stack = y[mask]

    # Simple average (fallback)
    avg_proba = np.nanmean(stack_matrix, axis=1)

    # Try Ridge meta-learner
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_stack)
        meta = RidgeClassifier(alpha=1.0)
        meta.fit(X_scaled, y_stack)

        # Decision function -> probability-like score
        decision = meta.decision_function(X_scaled)
        # Normalize to [0, 1]
        ensemble_proba = np.full(len(y), np.nan)
        d_min, d_max = decision.min(), decision.max()
        if d_max > d_min:
            ensemble_proba[mask] = (decision - d_min) / (d_max - d_min)
        else:
            ensemble_proba = avg_proba

        thr = _optimal_threshold(y_stack, ensemble_proba[mask])
        y_pred = (ensemble_proba[mask] >= thr).astype(int)
        metrics = _compute_metrics(y_stack, y_pred, ensemble_proba[mask])
        print(f"      🎯 Ensemble: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
        return ensemble_proba, metrics

    except Exception as e:
        print(f"      ⚠️  Ridge ensemble failed ({e}), используем avg")
        thr = _optimal_threshold(y[mask], avg_proba[mask])
        y_pred = (avg_proba[mask] >= thr).astype(int)
        metrics = _compute_metrics(y[mask], y_pred, avg_proba[mask])
        return avg_proba, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    df: pd.DataFrame,
    config: Optional[ModelConfig] = None,
) -> PipelineResult:
    """
    Запустить полный ML пайплайн.

    Args:
        df: DataFrame с features + target от features.build_features()
        config: Конфигурация

    Returns:
        PipelineResult с метриками, SHAP, predictions
    """
    if config is None:
        config = ModelConfig()

    print("\n" + "=" * 70)
    print("🧠 ML PIPELINE")
    print("=" * 70)

    # ── Preprocessing ──
    print("\n📦 Предобработка...")
    X, y, feature_names, timestamps = preprocess(df, config)
    n_crashes = int(y.sum())
    crash_rate = y.mean()
    print(f"   Crash events: {n_crashes} ({crash_rate * 100:.2f}%)")

    # Scale if needed
    scaler = None
    if config.scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # ── Walk-Forward Splits ──
    splits = _purged_time_series_split(len(X), config.n_splits, config.purge_gap)
    print(f"   Walk-Forward: {len(splits)} splits, purge gap={config.purge_gap} bars")

    # ── Train Models ──
    models_results: dict[str, ModelResult] = {}
    all_oof_probas: dict[str, np.ndarray] = {}
    removed_correlated: list[str] = []

    for model_name in config.models:
        if model_name not in TRAINERS:
            print(f"   ⚠️  Неизвестная модель: {model_name}")
            continue

        print(f"\n🔬 {model_name.upper()}")
        print("-" * 50)

        # Optuna
        best_params: dict[str, Any] = {}
        if config.run_optuna:
            print(f"   🔍 Optuna optimization ({config.optuna_n_trials} trials)...")
            best_params = _optuna_tune(model_name, X, y, splits, config)

        # Walk-Forward validation
        oof_proba = np.full(len(y), np.nan)
        fold_metrics: list[dict[str, float]] = []
        last_model = None

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            trainer = TRAINERS[model_name]
            model = trainer(X_tr, y_tr, X_val, y_val, config, best_params)

            y_proba = model.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = y_proba

            thr = 0.5
            y_pred = (y_proba >= thr).astype(int)
            fold_m = _compute_metrics(y_val, y_pred, y_proba)
            fold_metrics.append(fold_m)

            print(f"   Fold {fold_i + 1}: AUC={fold_m['roc_auc']:.4f} | F1={fold_m['f1']:.4f} | "
                  f"Prec={fold_m['precision']:.4f} | Rec={fold_m['recall']:.4f}")

            last_model = model

        # Overall OOF metrics
        valid_mask = ~np.isnan(oof_proba)
        if valid_mask.sum() > 0:
            opt_thr = _optimal_threshold(y[valid_mask], oof_proba[valid_mask]) if config.optimize_threshold else 0.5
            oof_pred = (oof_proba[valid_mask] >= opt_thr).astype(int)
            overall_metrics = _compute_metrics(y[valid_mask], oof_pred, oof_proba[valid_mask])
            print(f"\n   📊 OOF Overall (thr={opt_thr:.3f}):")
            print(f"      AUC={overall_metrics['roc_auc']:.4f} | F1={overall_metrics['f1']:.4f} | "
                  f"Prec={overall_metrics['precision']:.4f} | Rec={overall_metrics['recall']:.4f}")
        else:
            overall_metrics = {"roc_auc": 0, "f1": 0, "precision": 0, "recall": 0, "accuracy": 0}
            opt_thr = 0.5

        # Feature Importance
        if hasattr(last_model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feature_names,
                "importance": last_model.feature_importances_,
            }).sort_values("importance", ascending=False)
        else:
            fi = pd.DataFrame({"feature": feature_names, "importance": 0.0})

        # SHAP
        print(f"   🔍 SHAP analysis...")
        shap_vals, shap_names = _compute_shap(last_model, X, feature_names, model_name)

        # Build SHAP importance if available
        if shap_vals is not None:
            shap_importance = np.abs(shap_vals).mean(axis=0)
            fi_shap = pd.DataFrame({
                "feature": feature_names,
                "importance": shap_importance / shap_importance.sum(),
            }).sort_values("importance", ascending=False)
            # Use SHAP-based importance (more reliable)
            fi = fi_shap

        models_results[model_name] = ModelResult(
            name=model_name,
            metrics=overall_metrics,
            fold_metrics=fold_metrics,
            feature_importance=fi,
            best_params=best_params,
            shap_values=shap_vals,
            shap_feature_names=shap_names,
            optimal_threshold=opt_thr,
        )
        all_oof_probas[model_name] = oof_proba

    # ── Ensemble ──
    ensemble_metrics: Optional[dict[str, float]] = None
    if config.use_ensemble and len(models_results) >= 2:
        print(f"\n🏗️ STACKING ENSEMBLE")
        print("-" * 50)
        ensemble_proba, ensemble_metrics = _train_ensemble(models_results, all_oof_probas, y)
    else:
        ensemble_proba = None

    # ── Best Model ──
    best_name = max(models_results, key=lambda k: models_results[k].metrics.get("roc_auc", 0))
    if ensemble_metrics and ensemble_metrics.get("roc_auc", 0) > models_results[best_name].metrics.get("roc_auc", 0):
        best_name = "ensemble"

    print(f"\n🏆 Best Model: {best_name}")

    # ── Predictions DataFrame ──
    best_proba = ensemble_proba if (best_name == "ensemble" and ensemble_proba is not None) else all_oof_probas.get(best_name, np.full(len(y), np.nan))
    best_thr = models_results.get(best_name, models_results[list(models_results.keys())[0]]).optimal_threshold if best_name != "ensemble" else 0.5

    valid = ~np.isnan(best_proba)
    predictions = pd.DataFrame({
        "timestamp": timestamps.values,
        "actual": y,
        "pred_proba": best_proba,
        "pred_label": np.where(valid, (best_proba >= best_thr).astype(int), np.nan),
    })

    return PipelineResult(
        models=models_results,
        ensemble_metrics=ensemble_metrics,
        feature_names=feature_names,
        n_features=len(feature_names),
        n_samples=len(y),
        n_crashes=n_crashes,
        crash_rate=crash_rate,
        removed_correlated=removed_correlated,
        best_model_name=best_name,
        predictions=predictions,
    )
