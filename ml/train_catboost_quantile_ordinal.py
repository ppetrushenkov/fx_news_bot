"""
Обучение CatBoost для:
  1) multi-quantile регрессии (предсказание ренджа: q0.1, q0.5, q0.9)
  2) ordinal-задачи (количество разворотов после новости)

Оба — с walk-forward валидацией по кварталам и логированием в MLflow.

Требуется: pip install catboost mlflow scikit-learn --break-system-packages
"""

import numpy as np
import pandas as pd
import mlflow
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score,
)


# ============================================================================
# 1. MULTI-QUANTILE РЕГРЕССИЯ (предсказание ренджа)
# ============================================================================

def pinball_loss(y_true, y_pred, quantile):
    """Pinball loss (quantile loss) для одного квантиля — основная метрика качества квантильной регрессии."""
    diff = y_true - y_pred
    return np.mean(np.where(diff >= 0, quantile * diff, (quantile - 1) * diff))


def coverage_rate(y_true, q_low_pred, q_high_pred):
    """
    Доля случаев, когда реальное значение попало в интервал [q_low, q_high].
    Для q0.1/q0.9 ожидаемое покрытие ~80%. Если сильно ниже — интервалы слишком узкие (модель оверконфидентна).
    """
    return float(np.mean((y_true >= q_low_pred) & (y_true <= q_high_pred)))


def compute_quantile_metrics(y_true, preds_dict, quantiles):
    """
    preds_dict: {0.1: array, 0.5: array, 0.9: array}
    Возвращает метрики по каждому квантилю + интервальные метрики.
    """
    metrics = {}

    for q in quantiles:
        y_pred_q = preds_dict[q]
        metrics[f"pinball_q{q}"] = pinball_loss(y_true, y_pred_q, q)
        metrics[f"mae_q{q}"] = mean_absolute_error(y_true, y_pred_q)
        # доля случаев, когда реальное значение НИЖЕ предсказанного квантиля
        # (для q0.1 ожидаем ~10%, для q0.5 ~50%, для q0.9 ~90% — проверка калибровки квантиля)
        metrics[f"empirical_coverage_below_q{q}"] = float(np.mean(y_true <= y_pred_q))

    if 0.1 in quantiles and 0.9 in quantiles:
        metrics["interval_coverage_80pct"] = coverage_rate(y_true, preds_dict[0.1], preds_dict[0.9])
        width = preds_dict[0.9] - preds_dict[0.1]
        metrics["mean_interval_width"] = float(np.mean(width))
        metrics["median_interval_width"] = float(np.median(width))
        # доля отрицательной ширины (q0.9 < q0.1) — модель должна быть монотонной, если нет — проблема
        metrics["pct_non_monotonic"] = float(np.mean(width < 0))

    if 0.5 in quantiles:
        metrics["mae_median"] = mean_absolute_error(y_true, preds_dict[0.5])
        metrics["rmse_median"] = float(np.sqrt(mean_squared_error(y_true, preds_dict[0.5])))
        # MAPE может взлетать на около-нулевых значениях ренджа — смотрите осторожно
        nonzero_mask = y_true != 0
        if nonzero_mask.sum() > 0:
            metrics["mape_median"] = float(
                np.mean(np.abs((y_true[nonzero_mask] - preds_dict[0.5][nonzero_mask]) / y_true[nonzero_mask])) * 100
            )

    return metrics


def train_catboost_multiquantile_walkforward(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    quarter_col: str,
    mlflow_experiment_name: str,
    quantiles: list = (0.1, 0.5, 0.9),
    run_name: str = "catboost_multiquantile",
    n_min_train_quarters: int = 1,
    catboost_params: dict = None,
    cat_features: list = None,
    log_models_per_fold: bool = False,
):
    """
    Walk-forward обучение CatBoost с MultiQuantile loss.

    target_col должен быть непрерывной величиной (например, range = high - low за горизонт).

    CatBoost с loss_function='MultiQuantile:alpha=0.1,0.5,0.9' возвращает в predict()
    массив shape (n_samples, n_quantiles) в том же порядке, что заданы alpha.
    """
    quantiles = list(quantiles)
    alpha_str = ",".join(str(q) for q in quantiles)

    default_params = {
        "loss_function": f"MultiQuantile:alpha={alpha_str}",
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "verbose": False,
    }
    if catboost_params:
        default_params.update(catboost_params)

    quarters_sorted = sorted(df[quarter_col].unique())
    if len(quarters_sorted) <= n_min_train_quarters:
        raise ValueError("Недостаточно кварталов для walk-forward валидации")

    mlflow.set_experiment(mlflow_experiment_name)

    oof_true, oof_quarters = [], []
    oof_preds = {q: [] for q in quantiles}

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(default_params)
        mlflow.log_param("quantiles", alpha_str)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_quarters_total", len(quarters_sorted))

        for i, val_quarter in enumerate(quarters_sorted[n_min_train_quarters:], start=n_min_train_quarters):
            train_mask = df[quarter_col].isin(quarters_sorted[:i])
            val_mask = df[quarter_col] == val_quarter

            X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
            X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]

            if len(X_val) == 0:
                continue

            fold_model = CatBoostRegressor(**default_params)
            fold_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
            )

            fold_pred_raw = fold_model.predict(X_val)  # shape (n, n_quantiles)
            fold_preds = {q: fold_pred_raw[:, idx] for idx, q in enumerate(quantiles)}

            for q in quantiles:
                oof_preds[q].extend(fold_preds[q])
            oof_true.extend(y_val.values)
            oof_quarters.extend([str(val_quarter)] * len(y_val))

            fold_metrics = compute_quantile_metrics(y_val.values, fold_preds, quantiles)
            with mlflow.start_run(run_name=f"fold_{val_quarter}", nested=True):
                mlflow.log_param("val_quarter", str(val_quarter))
                mlflow.log_param("n_train", len(X_train))
                mlflow.log_param("n_val", len(X_val))
                mlflow.log_metrics({f"fold_{k}": v for k, v in fold_metrics.items()})
                if log_models_per_fold:
                    mlflow.catboost.log_model(fold_model, artifact_path="model")

        oof_true = np.array(oof_true)
        for q in quantiles:
            oof_preds[q] = np.array(oof_preds[q])

        oof_metrics = compute_quantile_metrics(oof_true, oof_preds, quantiles)
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})

        oof_table = pd.DataFrame({
            "quarter": oof_quarters,
            "y_true": oof_true,
            **{f"pred_q{q}": oof_preds[q] for q in quantiles},
        })
        mlflow.log_table(oof_table, artifact_file="oof_predictions.json")

        # финальная модель на всех данных
        final_model = CatBoostRegressor(**default_params)
        final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": final_model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        mlflow.log_table(importances, artifact_file="feature_importances.json")

        print(f"OOF interval coverage (80%): {oof_metrics.get('interval_coverage_80pct', float('nan')):.3f}")
        print(f"OOF MAE median: {oof_metrics.get('mae_median', float('nan')):.4f}")
        print(f"OOF non-monotonic %: {oof_metrics.get('pct_non_monotonic', 0)*100:.2f}%")
        print(f"MLflow run_id: {parent_run.info.run_id}")

    return {
        "final_model": final_model,
        "quantiles": quantiles,
        "oof_table": oof_table,
        "oof_metrics": oof_metrics,
        "feature_importances": importances,
        "run_id": parent_run.info.run_id,
    }


# ============================================================================
# 2. ORDINAL-ЗАДАЧА (количество разворотов цены)
# ============================================================================

def compute_ordinal_metrics(y_true, y_pred_continuous, max_class=None):
    """
    y_true: целые классы (0, 1, 2, 3, ...)
    y_pred_continuous: непрерывное предсказание регрессора (до округления)

    Метрики ordinal-задачи отличаются от обычной регрессии и обычной классификации:
    - QWK (Quadratic Weighted Kappa) — главная метрика для ordinal, штрафует за "далёкие" ошибки
      сильнее, чем за соседние (предсказать 2 вместо 3 — не страшно, 0 вместо 4 — очень страшно)
    - Accuracy и Accuracy±1 (попадание в соседний класс — для ordinal это почти успех)
    - MAE на округлённых классах
    """
    if max_class is None:
        max_class = int(max(y_true.max(), np.round(y_pred_continuous).max()))

    y_pred_rounded = np.clip(np.round(y_pred_continuous), 0, max_class).astype(int)

    qwk = cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic")
    mae = mean_absolute_error(y_true, y_pred_rounded)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_rounded)))
    mae_continuous = mean_absolute_error(y_true, y_pred_continuous)  # MAE до округления — честнее показывает качество модели

    exact_match = float(np.mean(y_true == y_pred_rounded))
    within_1 = float(np.mean(np.abs(y_true - y_pred_rounded) <= 1))

    # распределение ошибок по классам — полезно смотреть, не "залипает" ли модель на частом классе
    per_class_mae = {}
    for c in sorted(np.unique(y_true)):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_mae[f"mae_class_{c}"] = float(mean_absolute_error(y_true[mask], y_pred_rounded[mask]))

    metrics = {
        "qwk": qwk,
        "mae_rounded": mae,
        "mae_continuous": mae_continuous,
        "rmse_rounded": rmse,
        "exact_match_accuracy": exact_match,
        "within_1_accuracy": within_1,
    }
    metrics.update(per_class_mae)
    return metrics


def train_catboost_ordinal_walkforward(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    quarter_col: str,
    mlflow_experiment_name: str,
    run_name: str = "catboost_ordinal_reversals",
    n_min_train_quarters: int = 1,
    catboost_params: dict = None,
    cat_features: list = None,
    loss_function: str = "RMSE",
    log_models_per_fold: bool = False,
):
    """
    Walk-forward обучение для ordinal-задачи (количество разворотов: 0, 1, 2, 3...).

    Подход: регрессия с последующим округлением + clip в допустимый диапазон классов.
    Это проще и обычно устойчивее, чем строить N-1 бинарных cumulative-классификаторов,
    и хорошо работает, когда классы упорядочены и расстояния между ними имеют смысл
    (что для "количества разворотов" так и есть).

    loss_function: 'RMSE' (штрафует большие ошибки сильнее) или 'Poisson' (если таргет —
    счётная величина с правым хвостом, что обычно верно для count-данных).
    Можно протестировать оба и сравнить qwk/mae в MLflow.
    """
    default_params = {
        "loss_function": loss_function,
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "verbose": False,
    }
    if catboost_params:
        default_params.update(catboost_params)

    quarters_sorted = sorted(df[quarter_col].unique())
    if len(quarters_sorted) <= n_min_train_quarters:
        raise ValueError("Недостаточно кварталов для walk-forward валидации")

    max_class = int(df[target_col].max())

    mlflow.set_experiment(mlflow_experiment_name)

    oof_true, oof_pred_continuous, oof_quarters = [], [], []

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(default_params)
        mlflow.log_param("max_class", max_class)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_quarters_total", len(quarters_sorted))

        for i, val_quarter in enumerate(quarters_sorted[n_min_train_quarters:], start=n_min_train_quarters):
            train_mask = df[quarter_col].isin(quarters_sorted[:i])
            val_mask = df[quarter_col] == val_quarter

            X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
            X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]

            if len(X_val) == 0:
                continue

            fold_model = CatBoostRegressor(**default_params)
            fold_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
            )

            fold_pred = fold_model.predict(X_val)

            oof_pred_continuous.extend(fold_pred)
            oof_true.extend(y_val.values)
            oof_quarters.extend([str(val_quarter)] * len(y_val))

            fold_metrics = compute_ordinal_metrics(y_val.values, fold_pred, max_class=max_class)
            with mlflow.start_run(run_name=f"fold_{val_quarter}", nested=True):
                mlflow.log_param("val_quarter", str(val_quarter))
                mlflow.log_param("n_train", len(X_train))
                mlflow.log_param("n_val", len(X_val))
                mlflow.log_metrics({f"fold_{k}": v for k, v in fold_metrics.items() if isinstance(v, (int, float))})
                if log_models_per_fold:
                    mlflow.catboost.log_model(fold_model, artifact_path="model")

        oof_true = np.array(oof_true)
        oof_pred_continuous = np.array(oof_pred_continuous)

        oof_metrics = compute_ordinal_metrics(oof_true, oof_pred_continuous, max_class=max_class)
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items() if isinstance(v, (int, float))})

        oof_table = pd.DataFrame({
            "quarter": oof_quarters,
            "y_true": oof_true,
            "y_pred_continuous": oof_pred_continuous,
            "y_pred_rounded": np.clip(np.round(oof_pred_continuous), 0, max_class).astype(int),
        })
        mlflow.log_table(oof_table, artifact_file="oof_predictions.json")

        # confusion-like таблица: true vs rounded — полезно глазами оценить, где модель путает классы
        confusion_df = pd.crosstab(
            oof_table["y_true"], oof_table["y_pred_rounded"],
            rownames=["true"], colnames=["pred"]
        )
        mlflow.log_table(confusion_df.reset_index(), artifact_file="confusion_true_vs_pred.json")

        final_model = CatBoostRegressor(**default_params)
        final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": final_model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        mlflow.log_table(importances, artifact_file="feature_importances.json")

        print(f"OOF QWK: {oof_metrics['qwk']:.4f}")
        print(f"OOF MAE (rounded / continuous): {oof_metrics['mae_rounded']:.4f} / {oof_metrics['mae_continuous']:.4f}")
        print(f"OOF exact match / within±1: {oof_metrics['exact_match_accuracy']:.3f} / {oof_metrics['within_1_accuracy']:.3f}")
        print(f"MLflow run_id: {parent_run.info.run_id}")

    return {
        "final_model": final_model,
        "max_class": max_class,
        "oof_table": oof_table,
        "oof_metrics": oof_metrics,
        "feature_importances": importances,
        "run_id": parent_run.info.run_id,
    }


# ============================================================================
# Пример использования
# ============================================================================
if __name__ == "__main__":
    # --- Multi-quantile (рендж) ---
    # result_range = train_catboost_multiquantile_walkforward(
    #     df=df,
    #     feature_cols=[...],
    #     target_col="range_1h",   # например, high_1h - low_1h после новости
    #     quarter_col="quarter",
    #     mlflow_experiment_name="forex_range_prediction",
    #     quantiles=(0.1, 0.5, 0.9),
    # )
    # raw_pred = result_range["final_model"].predict(new_data[feature_cols])  # shape (n, 3)
    # q10, q50, q90 = raw_pred[:, 0], raw_pred[:, 1], raw_pred[:, 2]

    # --- Ordinal (кол-во разворотов) ---
    # result_reversals = train_catboost_ordinal_walkforward(
    #     df=df,
    #     feature_cols=[...],
    #     target_col="n_reversals",
    #     quarter_col="quarter",
    #     mlflow_experiment_name="forex_reversal_count",
    #     loss_function="Poisson",  # попробуйте также 'RMSE' и сравните qwk в MLflow
    # )
    pass
