import numpy as np
import pandas as pd
import mlflow
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score,
)


# ============================================================================
# Общая логика фолдов
# ============================================================================

class SlidingQuarterBlockCV:
    """
    Expanding validation by calendar quarter (uses ``time_col``, not row counts).

    Each fold:
    - **train**: rows with ``t_min <= t < val_start`` (all data from the beginning).
    - **val**: ``val_start <= t < val_end`` with length ``val_quarters``.

    First fold: ``val_start = t_min + train_quarters`` (so the first train block is
    exactly ``train_quarters`` long).

    Each next fold: ``val_start`` advances by ``step_quarters`` (default 1 quarter).

    Yields integer positions (0..n-1) suitable for ``df.iloc[train_idx]`` on the
    same dataframe **sorted by** ``time_col``.
    """

    def __init__(
        self,
        time_col: str,
        train_quarters: int = 4*6,  # 4 quarters * 6 years
        val_quarters: int = 1,
        step_quarters: int = 1,
    ):
        self.time_col = time_col
        self.train_quarters = train_quarters
        self.val_quarters = val_quarters
        self.step_quarters = step_quarters

    def split(self, df: pd.DataFrame):
        df_sorted = df.sort_values(self.time_col).reset_index(drop=True)
        t = pd.to_datetime(df_sorted[self.time_col], utc=True, errors="coerce")
        if t.isna().any():
            raise ValueError(f"SlidingQuarterBlockCV: non-datetime values in {self.time_col!r}")

        pos = np.arange(len(df_sorted))
        t_min = t.iloc[0]
        t_end_data = t.iloc[-1]

        val_start = t_min + pd.DateOffset(months=self.train_quarters * 3)
        step = pd.DateOffset(months=self.step_quarters * 3)
        val_span = pd.DateOffset(months=self.val_quarters * 3)

        while True:
            val_end = val_start + val_span
            train_start = t_min  # Use all data from the beginning

            if val_start > t_end_data:
                break

            train_mask = (t >= train_start) & (t < val_start)
            val_mask = (t >= val_start) & (t < val_end)

            tr_idx = pos[train_mask.to_numpy()]
            va_idx = pos[val_mask.to_numpy()]

            if len(va_idx) > 0 and len(tr_idx) > 0:
                yield tr_idx, va_idx

            val_start = val_start + step

    def get_n_splits(self, df: pd.DataFrame) -> int:
        return sum(1 for _ in self.split(df))
    

def build_walkforward_quarter_folds(quarters_sorted, n_val_quarters=6):
    """
    Стандартный expanding window walk-forward по кварталам.
    Последние n_val_quarters кварталов становятся валидационными фолдами,
    train на каждом шаге — все кварталы строго до текущего val (только прошлое).

    Возвращает список (step_idx, val_quarter, train_quarters), step_idx с 0,
    используется как mlflow `step` для построения временного графика по шагам.
    """
    if len(quarters_sorted) <= n_val_quarters:
        raise ValueError(
            f"Недостаточно кварталов: всего {len(quarters_sorted)}, "
            f"запрошено {n_val_quarters} валидационных"
        )

    val_quarters = quarters_sorted[-n_val_quarters:]  # Добавить :, если нужно валидировать на всех следующих кварталах
    folds = []
    for step_idx, val_quarter in enumerate(val_quarters):
        train_quarters = [q for q in quarters_sorted if q < val_quarter]
        folds.append((step_idx, val_quarter, train_quarters))
    return folds


def _log_step_metrics(metrics: dict, step: int, prefix: str = ""):
    """Логирует метрики фолда с step=step, чтобы MLflow строил line chart по шагам/кварталам."""
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mlflow.log_metric(f"{prefix}{k}", float(v), step=step)


# ============================================================================
# 1. BINARY CLASSIFICATION
# ============================================================================

def select_best_threshold(y_true, y_prob, beta=2.0, min_recall=None):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    if min_recall is not None:
        mask = recalls >= min_recall
        best_idx = np.argmax(recalls) if mask.sum() == 0 else np.argmax(np.where(mask, precisions, -1))
    else:
        f_beta = (1 + beta**2) * precisions * recalls / (beta**2 * precisions + recalls + 1e-9)
        best_idx = np.argmax(f_beta)
    return float(thresholds[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "base_rate": float(np.mean(y_true)),
    }


def train_catboost_binary(
    df, feature_cols, target_col, quarter_col,
    mlflow_experiment_name, run_name="catboost_binary",
    catboost_params=None, calibration_method="isotonic",
    threshold_beta=2.0, min_recall=None, cat_features=None, n_val_quarters=6,
):
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    default_params = {
        "loss_function": "Logloss", "eval_metric": "PRAUC",
        "auto_class_weights": "Balanced", "iterations": 1000,
        "learning_rate": 0.03, "depth": 6, "random_seed": 42,
        "early_stopping_rounds": 50, "verbose": False,
        "task_type": "GPU", "devices": "0",
    }
    if catboost_params:
        default_params.update(catboost_params)

    quarters_sorted = sorted(df[quarter_col].unique())
    folds = build_walkforward_quarter_folds(quarters_sorted, n_val_quarters=n_val_quarters)

    mlflow.set_experiment(mlflow_experiment_name)
    oof_probs, oof_true, oof_quarters = [], [], []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(default_params)
        mlflow.log_param("calibration_method", calibration_method)
        mlflow.log_param("n_quarters_total", len(quarters_sorted))
        mlflow.log_param("fold_order", ",".join(str(q) for _, q, _ in folds))  # видно порядок шагов

        for step_idx, val_quarter, train_quarters in folds:
            train_mask = df[quarter_col].isin(train_quarters)
            val_mask = df[quarter_col] == val_quarter
            X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
            X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]
            if y_val.nunique() < 2:
                continue

            model = CatBoostClassifier(**default_params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)
            probs = model.predict_proba(X_val)[:, 1]

            oof_probs.extend(probs)
            oof_true.extend(y_val.values)
            oof_quarters.extend([str(val_quarter)] * len(y_val))

            fold_metrics = compute_binary_metrics(y_val.values, probs, threshold=0.5)
            # step = step_idx, но логируем и сам квартал как метрику-метку оси для читаемости
            mlflow.log_metric("val_quarter_numeric", step_idx, step=step_idx)
            _log_step_metrics(fold_metrics, step=step_idx, prefix="fold_")

        oof_probs, oof_true = np.array(oof_probs), np.array(oof_true)

        if calibration_method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(oof_probs, oof_true)
            calibrate_fn = lambda p: calibrator.predict(np.asarray(p))
        else:
            calibrator = LogisticRegression()
            calibrator.fit(oof_probs.reshape(-1, 1), oof_true)
            calibrate_fn = lambda p: calibrator.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]

        oof_probs_cal = calibrate_fn(oof_probs)
        threshold, thr_p, thr_r = select_best_threshold(oof_true, oof_probs_cal, beta=threshold_beta, min_recall=min_recall)

        metrics_raw = compute_binary_metrics(oof_true, oof_probs, threshold=0.5)
        metrics_cal = compute_binary_metrics(oof_true, oof_probs_cal, threshold=threshold)
        mlflow.log_metrics({f"oof_raw_{k}": v for k, v in metrics_raw.items()})
        mlflow.log_metrics({f"oof_calibrated_{k}": v for k, v in metrics_cal.items()})
        mlflow.log_metric("selected_threshold", threshold)

        oof_table = pd.DataFrame({"quarter": oof_quarters, "y_true": oof_true, "p_raw": oof_probs, "p_calibrated": oof_probs_cal})
        mlflow.log_table(oof_table, artifact_file="oof_predictions.json")

        final_model = CatBoostClassifier(**default_params)
        final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        print(f"OOF PR-AUC (calibrated): {metrics_cal['pr_auc']:.4f}, threshold={threshold:.3f} (p={thr_p:.3f}, r={thr_r:.3f})")
        print(f"MLflow run_id: {run.info.run_id}")

    return {
        "final_model": final_model, "calibrator": calibrator, "calibrate_fn": calibrate_fn,
        "threshold": threshold, "oof_table": oof_table, "run_id": run.info.run_id,
    }


# ============================================================================
# 2. MULTI-QUANTILE REGRESSION
# ============================================================================

def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))


def compute_quantile_metrics(y_true, preds_dict, quantiles):
    metrics = {}
    for q in quantiles:
        yp = preds_dict[q]
        metrics[f"pinball_q{q}"] = pinball_loss(y_true, yp, q)
        metrics[f"mae_q{q}"] = mean_absolute_error(y_true, yp)
        metrics[f"empirical_coverage_below_q{q}"] = float(np.mean(y_true <= yp))
    if 0.1 in quantiles and 0.9 in quantiles:
        metrics["interval_coverage_80pct"] = float(np.mean((y_true >= preds_dict[0.1]) & (y_true <= preds_dict[0.9])))
        width = preds_dict[0.9] - preds_dict[0.1]
        metrics["mean_interval_width"] = float(np.mean(width))
        metrics["pct_non_monotonic"] = float(np.mean(width < 0))
    if 0.5 in quantiles:
        metrics["mae_median"] = mean_absolute_error(y_true, preds_dict[0.5])
        metrics["rmse_median"] = float(np.sqrt(mean_squared_error(y_true, preds_dict[0.5])))
    return metrics


def train_catboost_multiquantile(
    df, feature_cols, target_col, quarter_col,
    mlflow_experiment_name, quantiles=(0.1, 0.5, 0.9),
    run_name="catboost_multiquantile", catboost_params=None, cat_features=None, n_val_quarters=6,
):
    quantiles = list(quantiles)
    alpha_str = ",".join(str(q) for q in quantiles)
    default_params = {
        "loss_function": f"MultiQuantile:alpha={alpha_str}",
        "iterations": 1000, "learning_rate": 0.03, "depth": 6,
        "random_seed": 42, "early_stopping_rounds": 50, "verbose": False,
    }
    if catboost_params:
        default_params.update(catboost_params)

    quarters_sorted = sorted(df[quarter_col].unique())
    folds = build_walkforward_quarter_folds(quarters_sorted, n_val_quarters=n_val_quarters)

    mlflow.set_experiment(mlflow_experiment_name)
    oof_true, oof_quarters = [], []
    oof_preds = {q: [] for q in quantiles}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(default_params)
        mlflow.log_param("quantiles", alpha_str)
        mlflow.log_param("fold_order", ",".join(str(q) for _, q, _ in folds))

        for step_idx, val_quarter, train_quarters in folds:
            train_mask = df[quarter_col].isin(train_quarters)
            val_mask = df[quarter_col] == val_quarter
            X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
            X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]
            if len(X_val) == 0:
                continue

            model = CatBoostRegressor(**default_params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)
            pred_raw = model.predict(X_val)
            fold_preds = {q: pred_raw[:, idx] for idx, q in enumerate(quantiles)}

            for q in quantiles:
                oof_preds[q].extend(fold_preds[q])
            oof_true.extend(y_val.values)
            oof_quarters.extend([str(val_quarter)] * len(y_val))

            fold_metrics = compute_quantile_metrics(y_val.values, fold_preds, quantiles)
            _log_step_metrics(fold_metrics, step=step_idx, prefix="fold_")

        oof_true = np.array(oof_true)
        for q in quantiles:
            oof_preds[q] = np.array(oof_preds[q])

        oof_metrics = compute_quantile_metrics(oof_true, oof_preds, quantiles)
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})

        oof_table = pd.DataFrame({"quarter": oof_quarters, "y_true": oof_true, **{f"pred_q{q}": oof_preds[q] for q in quantiles}})
        mlflow.log_table(oof_table, artifact_file="oof_predictions.json")

        final_model = CatBoostRegressor(**default_params)
        final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        print(f"OOF interval coverage 80%: {oof_metrics.get('interval_coverage_80pct', float('nan')):.3f}")
        print(f"MLflow run_id: {run.info.run_id}")

    return {"final_model": final_model, "quantiles": quantiles, "oof_table": oof_table, "run_id": run.info.run_id}


# ============================================================================
# 3. ORDINAL (количество разворотов)
# ============================================================================

def compute_ordinal_metrics(y_true, y_pred_continuous, max_class):
    y_pred_rounded = np.clip(np.round(y_pred_continuous), 0, max_class).astype(int)
    return {
        "qwk": cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic"),
        "mae_rounded": mean_absolute_error(y_true, y_pred_rounded),
        "mae_continuous": mean_absolute_error(y_true, y_pred_continuous),
        "rmse_rounded": float(np.sqrt(mean_squared_error(y_true, y_pred_rounded))),
        "exact_match_accuracy": float(np.mean(y_true == y_pred_rounded)),
        "within_1_accuracy": float(np.mean(np.abs(y_true - y_pred_rounded) <= 1)),
    }


def train_catboost_ordinal(
    df, feature_cols, target_col, quarter_col,
    mlflow_experiment_name, run_name="catboost_ordinal_reversals",
    catboost_params=None, cat_features=None, loss_function="RMSE", n_val_quarters=6,
):
    default_params = {
        "loss_function": loss_function, "iterations": 1000,
        "learning_rate": 0.03, "depth": 6, "random_seed": 42,
        "early_stopping_rounds": 50, "verbose": False,
    }
    if catboost_params:
        default_params.update(catboost_params)

    quarters_sorted = sorted(df[quarter_col].unique())
    folds = build_walkforward_quarter_folds(quarters_sorted, n_val_quarters=n_val_quarters)
    max_class = int(df[target_col].max())

    mlflow.set_experiment(mlflow_experiment_name)
    oof_true, oof_pred, oof_quarters = [], [], []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(default_params)
        mlflow.log_param("max_class", max_class)
        mlflow.log_param("fold_order", ",".join(str(q) for _, q, _ in folds))

        for step_idx, val_quarter, train_quarters in folds:
            train_mask = df[quarter_col].isin(train_quarters)
            val_mask = df[quarter_col] == val_quarter
            X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
            X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]
            if len(X_val) == 0:
                continue

            model = CatBoostRegressor(**default_params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)
            pred = model.predict(X_val)

            oof_pred.extend(pred)
            oof_true.extend(y_val.values)
            oof_quarters.extend([str(val_quarter)] * len(y_val))

            fold_metrics = compute_ordinal_metrics(y_val.values, pred, max_class)
            _log_step_metrics(fold_metrics, step=step_idx, prefix="fold_")

        oof_true, oof_pred = np.array(oof_true), np.array(oof_pred)
        oof_metrics = compute_ordinal_metrics(oof_true, oof_pred, max_class)
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})

        oof_table = pd.DataFrame({
            "quarter": oof_quarters, "y_true": oof_true, "y_pred_continuous": oof_pred,
            "y_pred_rounded": np.clip(np.round(oof_pred), 0, max_class).astype(int),
        })
        mlflow.log_table(oof_table, artifact_file="oof_predictions.json")

        final_model = CatBoostRegressor(**default_params)
        final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        print(f"OOF QWK: {oof_metrics['qwk']:.4f}, MAE rounded: {oof_metrics['mae_rounded']:.4f}")
        print(f"MLflow run_id: {run.info.run_id}")

    return {"final_model": final_model, "max_class": max_class, "oof_table": oof_table, "run_id": run.info.run_id}
