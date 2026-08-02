import os
import warnings
from typing import Literal

import mlflow
import mlflow.catboost
import mlflow.xgboost

import mlflow.shap
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import coverage, normalized_mae, within_band_accuracy

import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoost

import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_pinball_loss,
    average_precision_score,
    cohen_kappa_score
)
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import BaseCrossValidator

from metrics import ordinal_gradation_metrics


# Splitting
def time_series_splits(
    df: pd.DataFrame,
    time_col: str,
    n_splits: int = 5,
    val_period: str = "180D",   # длина одного validation окна
    min_train_period: str = "365D"
):
    """
    Expanding-window time series CV.

    Parameters
    ----------
    df : DataFrame
    time_col : str
        Datetime column (tz-aware allowed)
    n_splits : int
    val_period : str
        Pandas offset alias, e.g. "90D", "180D", "1Y"
    min_train_period : str
        Minimal size of train window

    Yields
    ------
    train_idx, val_idx : np.ndarray
    """

    df = df.sort_values(time_col).reset_index(drop=True)

    times = df[time_col]
    min_time = times.min()
    max_time = times.max()

    val_delta = pd.Timedelta(val_period)
    train_delta = pd.Timedelta(min_train_period)

    # последняя возможная точка начала val
    last_val_start = max_time - val_delta

    # равномерно размещаем начала val-окон
    val_starts = pd.date_range(
        start=min_time + train_delta,
        end=last_val_start,
        periods=n_splits
    )

    for val_start in val_starts:
        val_end = val_start + val_delta

        train_mask = times < val_start
        val_mask = (times >= val_start) & (times < val_end)

        train_idx = df.index[train_mask].to_numpy()
        val_idx = df.index[val_mask].to_numpy()

        if len(val_idx) == 0:
            continue

        yield train_idx, val_idx


class RollingWindowCV(BaseCrossValidator):
    def __init__(self, window, horizon, step=1):
        self.window, self.horizon, self.step = window, horizon, step

    def split(self, X, y=None, groups=None):
        n = len(X)
        for start in range(0, n - self.window - self.horizon + 1, self.step):
            train = np.arange(start, start + self.window)
            test  = np.arange(start + self.window, start + self.window + self.horizon)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return (len(X) - self.window - self.horizon) // self.step + 1


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


class SlidingYearBlockCV:
    """
    Sliding validation by calendar time (uses ``time_col``, not row counts).

    Each fold:
    - **train**: rows with ``train_start <= t < val_start`` where the span is
      ``train_years`` (so the last train timestamp is just before validation).
    - **val**: ``val_start <= t < val_end`` with length ``val_years``.

    First fold: ``val_start = t_min + train_years`` (so the first train block is
    exactly ``train_years`` back from the start of validation).

    Each next fold: ``val_start`` advances by ``step_years`` (default 1 year).

    Yields integer positions (0..n-1) suitable for ``df.iloc[train_idx]`` on the
    same dataframe **sorted by** ``time_col``.
    """

    def __init__(
        self,
        time_col: str,
        train_years: int = 6,
        val_years: int = 1,
        step_years: int = 1,
    ):
        self.time_col = time_col
        self.train_years = train_years
        self.val_years = val_years
        self.step_years = step_years

    def split(self, df: pd.DataFrame):
        df_sorted = df.sort_values(self.time_col).reset_index(drop=True)
        t = pd.to_datetime(df_sorted[self.time_col], utc=True, errors="coerce")
        if t.isna().any():
            raise ValueError(f"SlidingYearBlockCV: non-datetime values in {self.time_col!r}")

        pos = np.arange(len(df_sorted))
        t_min = t.iloc[0]
        t_end_data = t.iloc[-1]

        val_start = t_min + pd.DateOffset(years=self.train_years)
        step = pd.DateOffset(years=self.step_years)
        train_span = pd.DateOffset(years=self.train_years)
        val_span = pd.DateOffset(years=self.val_years)

        while True:
            val_end = val_start + val_span
            train_start = val_start - train_span

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


def log_table(table: pd.DataFrame, filename: str) -> None:
    # save fold table
    table.to_csv(f"{filename}.csv", index=False)
    mlflow.log_artifact(f"{filename}.csv")
    os.remove(f"{filename}.csv")


# MLFLOW Logging
def run_time_series_cv_catboost(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    task_type: str,
    experiment_name: str,
    run_name: str,
    description: str,
    cat_features: list[str] | None = None,
    model_params: dict | None = None,
    n_splits: int = 5,
    save_model: bool = False,
    model_name: str = None
):
    """
    Run time series CV for CatBoost with MLflow logging.

    Cross-validation uses :class:`SlidingYearBlockCV`: 6 calendar years of training
    data immediately before each validation year; validation is 1 calendar year;
    the validation window moves forward by 1 year each fold. Row selection is by
    ``time_col``, not a fixed bar count (robust to missing bars).

    The ``n_splits`` argument is retained for API compatibility only (e.g. notebooks);
    the number of folds is ``n_folds`` = how many full year blocks fit in the data.

    ``task_type``:
    - ``"regression"`` — CatBoostRegressor, standard regression metrics.
    - ``"classification"`` — CatBoostClassifier (binary metrics as currently logged).
    - ``"ordinal"`` — ordered labels (e.g. volatility gradation 0..7): CatBoostRegressor
      with RMSE so large rank errors cost more than small ones; metrics via
      :func:`ordinal_gradation_metrics`.
    """

    df = df.sort_values(by=time_col, ascending=True)

    mlflow.set_experiment(experiment_name)

    if model_params is None:
        model_params = {}

    metrics_per_fold = []
    

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("task_type", task_type)
        mlflow.log_param("description", description)
        mlflow.log_param("features", ",".join(features))

        # CV scheme (keep consistent with your existing quantile CV)
        quarter_cv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=4 * 6 * 2,
            val_quarters=1,
            step_quarters=3,
        )
        splits = list(quarter_cv.split(df))
        mlflow.log_param(
            "cv_scheme",
            "SlidingQuarterBlockCV_train_48Q_val_1Q_step_3Q",
        )
        mlflow.log_param("n_folds", len(splits))
        mlflow.log_param("n_splits_arg", n_splits)

        pbar = tqdm(enumerate(splits, 1), total=len(splits), desc='Processing splits', leave=False)
        for fold, (train_idx, val_idx) in pbar:

            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]


            X_train = train_df[features]
            y_train = train_df[target]
            X_val = val_df[features]
            y_val = val_df[target]


            # -------------------------
            # MODEL
            # -------------------------
            gpu_kw = {
                "task_type": "GPU",
                "devices": "0",
                "verbose": False,
                "rsm": model_params.get("rsm", 1.0),
                "l2_leaf_reg": model_params.get("l2_leaf_reg", 3.0),
                "model_size_reg": model_params.get("model_size_reg", 0.5),
            }

            if task_type == "regression":
                model = CatBoostRegressor(**gpu_kw, **model_params)
            elif task_type == "ordinal":
                # RMSE: ошибка 0 vs 7 штрафуется сильнее, чем 0 vs 3 (квадрат расстояния по шкале)
                ord_kw = {
                    "loss_function": "RMSE",
                    "eval_metric": "RMSE",
                    **gpu_kw,
                }
                # Remove keys already in gpu_kw to avoid duplicates when updating with model_params
                clean_params = {k: v for k, v in model_params.items() if k not in gpu_kw}
                ord_kw.update(clean_params)
                ord_kw.update(model_params)
                model = CatBoostRegressor(
                    **ord_kw
                    )
            elif task_type == "classification":
                model = CatBoostClassifier(
                    **gpu_kw,
                    **model_params,
                    auto_class_weights='Balanced',
                    )

            else:
                model = CatBoost(**gpu_kw, **model_params)

            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
                early_stopping_rounds=10,
            )

            # -------------------------
            # METRICS
            # -------------------------
            train_dates = f"{train_df[time_col].min().date()} - {train_df[time_col].max().date()}"
            val_dates = f"{val_df[time_col].min().date()} - {val_df[time_col].max().date()}"

            if task_type == "regression":
                y_pred = model.predict(X_val)

                fold_metrics = {
                    "fold": fold,
                    "Train dates": train_dates,
                    "Val dates": val_dates,
                    "MAE": mean_absolute_error(y_val, y_pred),
                    "MedianAE": median_absolute_error(y_val, y_pred),
                    "RMSE": mean_squared_error(y_val, y_pred) ** 0.5,
                    "R2": r2_score(y_val, y_pred),
                    "nMAE": normalized_mae(y_val, y_pred),
                    "Within_30pct": within_band_accuracy(y_val, y_pred),
                }

                for name, val in [
                    ("MAE", fold_metrics["MAE"]),
                    ("RMSE", fold_metrics["RMSE"]),
                    ("MedianAE", fold_metrics["MedianAE"]),
                    ("R2", fold_metrics["R2"]),
                    ("nMAE", fold_metrics["nMAE"]),
                    ("Within_30pct", fold_metrics["Within_30pct"]),
                ]:
                    mlflow.log_metric(name, val, step=fold)

            elif task_type == "ordinal":
                y_pred = model.predict(X_val)
                om = ordinal_gradation_metrics(y_val, y_pred)
                fold_metrics = {
                    "fold": fold,
                    "Train dates": train_dates,
                    "Val dates": val_dates,
                    **om,
                }
                for name, val in om.items():
                    mlflow.log_metric(name, val, step=fold)

            elif task_type == "multiclassification":
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)

                fold_metrics = {
                    "Train dates": train_dates,
                    "Val dates": val_dates,
                    "fold": fold,
                    "Accuracy": accuracy_score(y_val, y_pred),
                    "Precision": precision_score(y_val, y_pred, zero_division=0, average='macro'),
                    "Recall": recall_score(y_val, y_pred, zero_division=0, average='macro'),
                    "F1": f1_score(y_val, y_pred, zero_division=0, average='macro'),
                    "ROC_AUC": roc_auc_score(
                                    y_val, 
                                    y_proba,              # Передаем целиком, БЕЗ [:, 1]
                                    multi_class='ovr',    # One-vs-Rest режим
                                    average='macro', 
                                    labels=model.classes_ # Задаем правильный порядок классов
                                ),
                    "LogLoss": log_loss(y_val, y_proba),
                }

                mlflow.log_metric("Accuracy", fold_metrics["Accuracy"], step=fold)
                mlflow.log_metric("Precision", fold_metrics["Precision"], step=fold)
                mlflow.log_metric("Recall", fold_metrics["Recall"], step=fold)
                mlflow.log_metric("F1", fold_metrics["F1"], step=fold)
                mlflow.log_metric("ROC_AUC", fold_metrics["ROC_AUC"], step=fold)
                mlflow.log_metric("LogLoss", fold_metrics["LogLoss"], step=fold)

            else:
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

                fold_metrics = {
                    "Train dates": train_dates,
                    "Val dates": val_dates,
                    "fold": fold,
                    "Accuracy": accuracy_score(y_val, y_pred),
                    "Precision": precision_score(y_val, y_pred, zero_division=0, average='macro'),
                    "Recall": recall_score(y_val, y_pred, zero_division=0, average='macro'),
                    "F1": f1_score(y_val, y_pred, zero_division=0, average='macro'),
                    "ROC_AUC": roc_auc_score(y_val, y_proba),
                    "Average_PRAUC": average_precision_score(y_val, y_proba),
                    "LogLoss": log_loss(y_val, y_proba),
                }

                mlflow.log_metric("Accuracy", fold_metrics["Accuracy"], step=fold)
                mlflow.log_metric("Precision", fold_metrics["Precision"], step=fold)
                mlflow.log_metric("Recall", fold_metrics["Recall"], step=fold)
                mlflow.log_metric("F1", fold_metrics["F1"], step=fold)
                mlflow.log_metric("ROC_AUC", fold_metrics["ROC_AUC"], step=fold)
                mlflow.log_metric("LogLoss", fold_metrics["LogLoss"], step=fold)

            metrics_per_fold.append(fold_metrics)

        # -------------------------
        # AGGREGATED METRICS
        # -------------------------
        metrics_df = pd.DataFrame(metrics_per_fold)
        log_table(metrics_df, "cv_metrics_per_fold")

        # ------------------------+
        # FEATURE IMPORTANCE PLOT |
        # ------------------------+
        feature_importance = model.get_feature_importance()
        feature_names = X_val.columns

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 8))
        top_n = 20
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        plt.close()

        # Log plot to MLflow
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        print("\n" + "=" * 80)
        print("Training final model on all data...")
        print("=" * 80)

        X = df[features]
        y = df[target]

        if task_type in ("regression", "ordinal"):
            loss = "RMSE" if task_type == "ordinal" else model_params.get(
                "loss_function", "RMSE"
            )
            eval_m = (
                "RMSE"
                if task_type == "ordinal"
                else model_params.get("eval_metric", loss)
            )
            gpu_kw.update(model_params)
            final_model = CatBoostRegressor(
                **gpu_kw
            )
            final_model.fit(X, y, cat_features=cat_features)
            y_pred_final = final_model.predict(X)
            print("\nFinal model metrics (full data, in-sample):")

            # Save model as a single file
            model_path = f"{model_name or 'catboost_ordinal_model'}.cbm"
            final_model.save_model(model_path)
            mlflow.log_artifact(model_path)
            os.remove(model_path)

            if task_type == "ordinal":
                fm = ordinal_gradation_metrics(y, y_pred_final)
                for k, v in fm.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"  MAE:  {mean_absolute_error(y, y_pred_final):.4f}")
                print(f"  RMSE: {mean_squared_error(y, y_pred_final) ** 0.5:.4f}")

        elif task_type == "multiclassification":
            fin_kw = {}
            fin_kw.update(gpu_kw)
            fin_kw.update(model_params)
            final_model = CatBoostClassifier(
                **fin_kw,
            )
            final_model.fit(X, y, cat_features=cat_features)

            y_pred_final = final_model.predict(X)
            # y_pred_proba_final = final_model.predict_proba(X)[:, 1]
            y_pred_proba_final = final_model.predict_proba(X)

            final_accuracy = accuracy_score(y, y_pred_final)
            final_auc = roc_auc_score(
                y, y_pred_proba_final,
                multi_class='ovr',    # One-vs-Rest режим
                average='macro', 
                labels=model.classes_
                )

            print("\nFinal Model Metrics (on full dataset):")
            print(f"  Accuracy:  {final_accuracy:.4f}")
            print(f"  AUC:       {final_auc:.4f}")

            print("\nClassification Report:")
            report = classification_report(y, y_pred_final, output_dict=True)
            report = pd.DataFrame(report).transpose().reset_index()
            print(report)
            log_table(report, "classification_report")

            # Save model as a single file
            model_path = f"{model_name or 'catboost_mclassification_model'}.cbm"
            final_model.save_model(model_path)
            mlflow.log_artifact(model_path)
            os.remove(model_path)

        else:
            fin_kw = {}
            fin_kw.update(gpu_kw)
            fin_kw.update(model_params)
            final_model = CatBoostClassifier(
                **fin_kw,
            )
            final_model.fit(X, y, cat_features=cat_features)

            y_pred_final = final_model.predict(X)
            y_pred_proba_final = final_model.predict_proba(X)[:, 1]

            final_accuracy = accuracy_score(y, y_pred_final)
            final_auc = roc_auc_score(y, y_pred_proba_final)

            print("\nFinal Model Metrics (on full dataset):")
            print(f"  Accuracy:  {final_accuracy:.4f}")
            print(f"  AUC:       {final_auc:.4f}")

            print("\nClassification Report:")
            report = classification_report(y, y_pred_final, output_dict=True)
            report = pd.DataFrame(report).transpose().reset_index()
            print(report)
            log_table(report, "classification_report")

            # Save model as a single file
            model_path = f"{model_name or 'catboost_classification_model'}.cbm"
            final_model.save_model(model_path)
            mlflow.log_artifact(model_path)
            os.remove(model_path)

    return metrics_df



def run_time_series_cv_catboost_quantile(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    quantiles: list[float],
    experiment_name: str,
    run_name: str,
    description: str,
    hour: int,
    cat_features: list[str] | None = None,
    model_params: dict | None = None,
    n_splits: int = 6,
    save_model: bool = False,
    model_name: str = None
):
    """
    Time-series CV for CatBoost Multi-Quantile Regression with MLflow logging.
    Uses expanding window on last N quarters.
    """
    df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    mlflow.set_experiment(experiment_name)

    if model_params is None:
        model_params = {}

    metrics_per_fold = []

    alphas = ",".join(str(q) for q in quantiles)
    loss_fn = f"MultiQuantile:alpha={alphas}"

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    with mlflow.start_run(run_name=run_name + '_+saved_model' if save_model else run_name):
        # -------------------------
        # META
        # -------------------------
        mlflow.log_param("task_type", "multi_quantile_regression")
        mlflow.log_param("quantiles", alphas)
        mlflow.log_param("description", description)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("hour", hour)

        # Get all unique quarters in the data
        df['quarter'] = df[time_col].dt.to_period('Q')
        unique_quarters = sorted(df['quarter'].unique())

        # Take last n_splits quarters
        val_quarters = unique_quarters[-n_splits:]
        
        splits = []
        for i, val_q in enumerate(val_quarters):
            # Train on all data before this val quarter
            train_mask = df['quarter'] < val_q
            val_mask = df['quarter'] == val_q
            splits.append((df.index[train_mask].tolist(), df.index[val_mask].tolist()))

        mlflow.log_param("cv_scheme", f"ExpandingWindow_last_{n_splits}_quarters")
        mlflow.log_param("n_folds", len(splits))

        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            train_min_date = train_df[time_col].min()
            train_max_date = train_df[time_col].max()
            val_min_date = val_df[time_col].min()
            val_max_date = val_df[time_col].max()

            print(f'[INFO] FOLD: {fold}')
            print(f'[INFO] Train date range: {train_min_date} to {train_max_date}')
            print(f'[INFO] Val date range: {val_min_date} to {val_max_date}')
            print('[INFO] Train count:', train_df.shape[0])
            print('[INFO] Val count:', val_df.shape[0])

            X_train = train_df[features]
            y_train = train_df[target]
            X_val = val_df[features]
            y_val = val_df[target]

            model = CatBoostRegressor(
                loss_function=loss_fn,
                eval_metric=loss_fn,
                iterations=1000,
                learning_rate=0.05,
                task_type="CPU",
                verbose=False,
                **{k: v for k, v in model_params.items() if k not in ("loss_function", "eval_metric", "iterations", "learning_rate", "task_type", "verbose")}
            )

            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True
            )

            y_pred = model.predict(X_val)

            fold_metrics = {
                "fold": fold,
                "train_min_date": train_min_date,
                "train_max_date": train_max_date,
                "val_min_date": val_min_date,
                "val_max_date": val_max_date
            }

            for i, q in enumerate(quantiles):
                q_pred = y_pred[:, i]
                fold_metrics[f"pinball_q{q}"] = mean_pinball_loss(y_true=y_val, y_pred=q_pred, alpha=q)
                fold_metrics[f"coverage_q{q}"] = coverage(y_val, q_pred)
                mlflow.log_metric(f"pinball_q{q}", fold_metrics[f"pinball_q{q}"], step=fold)
                mlflow.log_metric(f"coverage_q{q}", fold_metrics[f"coverage_q{q}"], step=fold)

            fold_metrics["interval_width"] = np.mean(y_pred[:, -1] - y_pred[:, 0])
            mlflow.log_metric("interval_width", fold_metrics["interval_width"], step=fold)
            metrics_per_fold.append(fold_metrics)

        # -------------------------
        # AGGREGATION
        # -------------------------
        metrics_df = pd.DataFrame(metrics_per_fold)
        metrics_path = f"models/cv_quantile_metrics_{hour}h.csv"
        metrics_df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(metrics_path)

        mean_metrics = metrics_df.drop(columns=["fold", "train_min_date", "train_max_date", "val_min_date", "val_max_date"]).mean().to_dict()
        std_metrics = metrics_df.drop(columns=["fold", "train_min_date", "train_max_date", "val_min_date", "val_max_date"]).std().to_dict()

        for k, v in mean_metrics.items():
            mlflow.log_metric(f"{k}_mean", v)
        for k, v in std_metrics.items():
            mlflow.log_metric(f"{k}_std", v)

        # Save model on full dataset
        if save_model:
            print("\n" + "=" * 80)
            print("Training final model on all data...")
            print("=" * 80)
            X = df[features]
            y = df[target]

            final_model = CatBoostRegressor(
                loss_function=loss_fn,
                eval_metric=loss_fn,
                iterations=1000,
                learning_rate=0.05,
                task_type="CPU",
                verbose=False,
                **{k: v for k, v in model_params.items() if k not in ("loss_function", "eval_metric", "iterations", "learning_rate", "task_type", "verbose")}
            )

            final_model.fit(X, y, cat_features=cat_features)

            # Save model
            if model_name is None:
                model_name = f"quantile_model_{hour}h"
            model_save_path = f"models/{model_name}.cbm"
            final_model.save_model(model_save_path)
            mlflow.log_artifact(model_save_path)

            # ------------------------
            # FEATURE IMPORTANCE PLOT
            # ------------------------
            feature_importances = final_model.get_feature_importance(prettified=True)
            if len(feature_importances) > 0:
                importance_df = pd.DataFrame(feature_importances, columns=["Feature", "Importance"])
                importance_df = importance_df.sort_values(by="Importance", ascending=False)

                plt.figure(figsize=(10, 6))
                plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
                plt.xlabel("Importance")
                plt.ylabel("Features")
                plt.title(f"Feature Importance ({hour}h)")
                plt.gca().invert_yaxis()
                plt.tight_layout()

                plot_path = f"models/feature_importance_{hour}h.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(plot_path)
            else:
                print("[WARNING] No feature importances available")

    return metrics_df


def run_time_series_cv_catboost_quantile_gpu(
    df: pd.DataFrame, features: list[str], target: str, time_col: str,
    experiment_name: str, run_name: str, description: str,
    hour: int, quantiles: list[float],
    model_params: dict | None = None,
    cat_features: list[str] | None = None,
    n_splits: int = 6,
    save_model: bool = False,
    model_name: str | None = None,
    verbose: bool = False
):
    """
    Time-series CV for CatBoost quantile regression with GPU support and MLflow logging.

    CatBoost ``MultiQuantile`` does not support GPU, so this function trains
    **separate** models for each quantile using ``Quantile:alpha=q`` on GPU and
    aggregates predictions/metrics in a single MLflow experiment/run.
    Uses expanding window on last N quarters.
    """
    mlflow.set_experiment(experiment_name)

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)


    # +========= Basic parameters ============+
    if model_params is None:
        model_params = {}
    
    quantiles = [float(q) for q in quantiles]
    quantiles_sorted = sorted(quantiles)
    alphas = ",".join(str(q) for q in quantiles_sorted)

    metrics_per_fold: list[dict] = []

    devices = str(model_params.get("devices", "0"))
    verbose = bool(model_params.get("verbose", False))

    # parameters that belong to loss/eval per-quantile model, not shared
    forbidden_loss_keys = {"loss_function", "eval_metric"}
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    with mlflow.start_run(run_name=run_name + "_+saved_model" if save_model else run_name):
        # Get all unique quarters in the data
        df['yyyy-qq'] = df[time_col].dt.to_period('Q')
        unique_quarters = sorted(df['yyyy-qq'].unique())

        # Take last n_splits quarters
        val_quarters = unique_quarters[-n_splits:]
        
        splits = []
        for i, val_q in enumerate(val_quarters):
            # Train on all data before this val quarter
            train_mask = df['yyyy-qq'] < val_q
            val_mask = df['yyyy-qq'] == val_q
            splits.append((df.index[train_mask].tolist(), df.index[val_mask].tolist()))

        # Log params
        mlflow.log_param("task_type", "quantile_regression_gpu_separate_models")
        mlflow.log_param("quantiles", alphas)
        mlflow.log_param("description", description)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("catboost_task_type", "GPU")
        mlflow.log_param("catboost_devices", devices)
        mlflow.log_param("hour", hour)
        mlflow.log_param("cv_scheme", f"ExpandingWindow_last_{n_splits}_quarters")
        mlflow.log_param("n_folds", len(splits))

        pbar = tqdm(enumerate(splits, 1), total=len(splits), desc='Processing splits', leave=False)
        for fold, (train_idx, val_idx) in pbar:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            train_min_date = train_df[time_col].min()
            train_max_date = train_df[time_col].max()
            val_min_date = val_df[time_col].min()
            val_max_date = val_df[time_col].max()


            pbar.set_postfix({
                "Fold": fold,
                "start train": train_min_date.date(), 
                "end train": train_max_date.date(),
                "start val": val_min_date.date(), 
                "end val": val_max_date.date()
                }
            )

            X_train = train_df[features]
            y_train = train_df[target]
            X_val = val_df[features]
            y_val = val_df[target]

            # Predict each quantile with its own GPU model
            preds_by_q: dict[float, np.ndarray] = {}

            for q in quantiles_sorted:
                loss_fn = f"Quantile:alpha={q}"

                params = {
                    "loss_function": loss_fn,
                    "eval_metric": loss_fn,
                    "task_type": "GPU",
                    "devices": devices,
                    "iterations": model_params.get("iterations", 5000),
                    "learning_rate": model_params.get("learning_rate", 0.05),
                    "depth": model_params.get("depth", 6),
                    "random_seed": model_params.get("random_seed", 42),
                    "verbose": verbose,
                }
                # allow advanced params from model_params (but never override loss/eval)
                for k, v in model_params.items():
                    if k in params or k in forbidden_loss_keys:
                        continue
                    params[k] = v

                model = CatBoostRegressor(**params)

                early_stopping_rounds = model_params.get("early_stopping_rounds", 50)
                use_best_model = bool(model_params.get("use_best_model", True))

                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    cat_features=cat_features,
                    use_best_model=use_best_model,
                    early_stopping_rounds=early_stopping_rounds,
                )

                preds_by_q[q] = model.predict(X_val).reshape(-1)

            fold_metrics: dict = {
                "fold": fold,
                "train_min_date": train_min_date,
                "train_max_date": train_max_date,
                "val_min_date": val_min_date,
                "val_max_date": val_max_date
            }

            # Per-quantile metrics
            for q in quantiles_sorted:
                q_pred = preds_by_q[q]
                pinball = mean_pinball_loss(y_true=y_val, y_pred=q_pred, alpha=q)
                cov = coverage(y_val, q_pred)

                fold_metrics[f"pinball_q{q}"] = pinball
                fold_metrics[f"coverage_q{q}"] = cov

                mlflow.log_metric(f"pinball_q{q}", pinball, step=fold)
                mlflow.log_metric(f"coverage_q{q}", cov, step=fold)

            # Interval metrics (between min & max quantile)
            if len(quantiles_sorted) >= 2:
                q_lo, q_hi = quantiles_sorted[0], quantiles_sorted[-1]
                interval_width = float(np.mean(preds_by_q[q_hi] - preds_by_q[q_lo]))
            else:
                interval_width = float("nan")
            fold_metrics["interval_width"] = interval_width
            mlflow.log_metric("interval_width", interval_width, step=fold)

            metrics_per_fold.append(fold_metrics)

        # -------------------------
        # AGGREGATION
        # -------------------------
        metrics_df = pd.DataFrame(metrics_per_fold)
        metrics_path = f"models/cv_catboost_quantile_gpu_metrics_{hour}h.csv"
        metrics_df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(metrics_path)

        if not metrics_df.empty:
            mean_metrics = metrics_df.drop(columns=["fold", "train_min_date", "train_max_date", "val_min_date", "val_max_date"]).mean(numeric_only=True).to_dict()
            std_metrics = metrics_df.drop(columns=["fold", "train_min_date", "train_max_date", "val_min_date", "val_max_date"]).std(numeric_only=True).to_dict()

            for k, v in mean_metrics.items():
                mlflow.log_metric(f"{k}_mean", float(v))

            for k, v in std_metrics.items():
                mlflow.log_metric(f"{k}_std", float(v))

        # -------------------------
        # SAVE FINAL MODELS (per quantile)
        # -------------------------
        if save_model:
            print("\n" + "=" * 80)
            print("Training final GPU quantile models on all data...")
            print("=" * 80)

            if not model_name:
                model_name = f"catboost_quantile_gpu_{hour}h"

            X_full = df[features]
            y_full = df[target]

            # Train a MultiQuantile CatBoostRegressor model on all data and save a single model file
            alphas = ",".join(str(q) for q in quantiles_sorted)
            loss_fn = f"MultiQuantile:alpha={alphas}"
            params = {
                "loss_function": loss_fn,
                "eval_metric": loss_fn,
                "task_type": "CPU",
                "devices": devices,
                "iterations": model_params.get("iterations", 5000),
                "learning_rate": model_params.get("learning_rate", 0.05),
                "depth": model_params.get("depth", 6),
                "random_seed": model_params.get("random_seed", 42),
                "verbose": verbose,
            }
            for k, v in model_params.items():
                # Ignore conflicting or restricted keys
                if k in params or k in forbidden_loss_keys:
                    continue
                params[k] = v

            multiq_model = CatBoostRegressor(**params)
            multiq_model.fit(X_full, y_full, cat_features=cat_features)

            # Feature importance plot for representative quantile (closest to 0.5)
            feature_importances = multiq_model.get_feature_importance(prettified=True)
            if len(feature_importances) > 0:
                importance_df = pd.DataFrame(feature_importances, columns=["Feature", "Importance"])
                importance_df = importance_df.sort_values(by="Importance", ascending=False)

                plt.figure(figsize=(10, 6))
                plt.barh(importance_df["Feature"].head(30), importance_df["Importance"].head(30), color="skyblue")
                plt.xlabel("Importance")
                plt.ylabel("Features")
                plt.title(f"Feature Importance (MultiQuantile, GPU, {hour}h)")
                plt.gca().invert_yaxis()
                plt.tight_layout()

                plot_path = f"models/feature_importance_multiq_{hour}h.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(plot_path)
            else:
                print("[WARNING] No feature importances available")

            # Save model as a single file
            model_save_path = f"models/{model_name}.cbm"
            multiq_model.save_model(model_save_path)
            mlflow.log_artifact(model_save_path)

    return metrics_df

DEFAULT_MULTI_QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)


def _normalized_quantiles(quantiles: list[float] | None) -> list[float]:
    if quantiles is None or len(quantiles) == 0:
        return list(DEFAULT_MULTI_QUANTILES)
    return sorted(float(q) for q in quantiles)


def _copy_features_as_tree_categories(
    sub_df: pd.DataFrame,
    features: list[str],
    cat_features: list[str] | None,
) -> pd.DataFrame:
    """Category dtype for categorical columns — XGBoost / LightGBM."""
    out = sub_df.loc[:, features].copy()
    for c in cat_features or []:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def _cv_split_indices(
    df: pd.DataFrame,
    time_col: str,
    *,
    cv: Literal["sliding_quarters", "time_series_split"],
    train_quarters: int,
    val_quarters: int,
    step_quarters: int,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    df_sorted = df.sort_values(by=time_col, ascending=True)
    if cv == "sliding_quarters":
        qcv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=train_quarters,
            val_quarters=val_quarters,
            step_quarters=step_quarters,
        )
        return list(qcv.split(df_sorted))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [(tr, va) for tr, va in tscv.split(df_sorted)]


def _merge_xgboost_quantile_booster_params(
    quantiles_sorted: list[float],
    model_params: dict,
    *,
    device: str,
) -> dict:
    """Params for ``xgb.train`` with objective ``reg:quantileerror`` (multi-alpha)."""
    alpha = np.asarray(quantiles_sorted, dtype=np.float64)
    reserved = {"objective", "quantile_alpha", "eval_metric", "device", "verbosity"}
    _max_leaves = model_params.get("max_leaves")
    out: dict = {
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "tree_method": str(model_params.get("tree_method", "hist")),
        "device": device,
        "learning_rate": float(model_params.get("learning_rate", 0.05)),
        "max_depth": int(model_params.get("max_depth", 6)),
        "max_leaves": None if _max_leaves is None else int(_max_leaves),
        "subsample": float(model_params.get("subsample", 0.8)),
        "colsample_bytree": float(model_params.get("colsample_bytree", 1.0)),
        "reg_lambda": float(model_params.get("reg_lambda", 1.0)),
        "reg_alpha": float(model_params.get("reg_alpha", 0.0)),
        "verbosity": int(model_params.get("verbosity", 0)),
        "multi_strategy": str(model_params.get("multi_strategy", "one_output_per_tree")),
        "seed": int(model_params.get("random_seed", model_params.get("seed", model_params.get("random_state", 42)))),
    }
    # drop unused None knobs XGBoost may reject before merge
    if out["max_leaves"] is None:
        del out["max_leaves"]
    for k, v in model_params.items():
        if k in reserved or k in out:
            continue
        if k in {
            "iterations",
            "early_stopping_rounds",
            "verbose",
            "random_seed",
            "random_state",
            "prefer_gpu",
        }:
            continue
        out[k] = v
    return out


def _xgb_predict_quantile_matrix(
    booster: xgb.Booster,
    dm: xgb.DMatrix,
    *,
    n_rows: int,
    n_quantiles: int,
) -> np.ndarray:
    best_it = getattr(booster, "best_iteration", None)
    if best_it is not None:
        pred = booster.predict(dm, iteration_range=(0, int(best_it) + 1))
    else:
        pred = booster.predict(dm)
    arr = np.asarray(pred)
    if arr.ndim == 2 and arr.shape[1] == n_quantiles:
        return arr
    if arr.size == n_rows * n_quantiles:
        return arr.reshape(n_rows, n_quantiles)
    if n_quantiles == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(-1, n_quantiles)


def _xgb_fit_quantile_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles_sorted: list[float],
    model_params: dict,
    *,
    prefer_gpu: bool,
    cat_features: list[str] | None,
) -> tuple[xgb.Booster, str, xgb.QuantileDMatrix]:
    """
    One native multi-quantile XGBoost booster.
    Prefer GPU (``device=cuda``); fall back to CPU if training fails (e.g. no CUDA build).
    """
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    y_va = np.asarray(y_val, dtype=np.float64).ravel()
    dtrain = xgb.QuantileDMatrix(X_train, y_tr, enable_categorical=bool(cat_features))
    dval = xgb.QuantileDMatrix(X_val, y_va, ref=dtrain, enable_categorical=bool(cat_features))
    evals = [(dtrain, "train"), (dval, "val")]

    n_round = int(model_params.get("iterations", 5000))
    esr = int(model_params.get("early_stopping_rounds", 50))
    verbose = bool(model_params.get("verbose", False))

    devices = ["cuda", "cpu"] if prefer_gpu else ["cpu"]
    last_err: str | None = None
    for dev in devices:
        params = _merge_xgboost_quantile_booster_params(quantiles_sorted, model_params, device=dev)
        try:
            train_kw = dict(
                params=params,
                dtrain=dtrain,
                num_boost_round=n_round,
                evals=evals,
                verbose_eval=min(50, verbose) if verbose else False,
            )
            if esr > 0:
                train_kw["early_stopping_rounds"] = esr
            booster = xgb.train(**train_kw)
            return booster, dev, dval
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if dev == "cuda":
                warnings.warn(
                    f"XGBoost quantile CUDA attempt failed ({last_err}); retrying on CPU.",
                )
                continue
            raise RuntimeError(last_err) from e
    raise RuntimeError(last_err or "xgb.train failed")


def run_time_series_cv_xgboost_multi_quantile(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    experiment_name: str,
    run_name: str,
    description: str,
    quantiles: list[float] | None = None,
    model_params: dict | None = None,
    train_quarters: int = 4 * 6 * 2,
    val_quarters: int = 1,
    step_quarters: int = 1,
    cv: Literal["sliding_quarters", "time_series_split"] = "sliding_quarters",
    n_splits: int = 5,
    save_model: bool = False,
    model_name: str | None = None,
    verbose: bool = False,
    prefer_gpu: bool = True,
    cat_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Time-series CV for **native multi-quantile** XGBoost (``reg:quantileerror`` with multiple
    ``quantile_alpha`` values). Matches the sliding-quarter scheme used in CatBoost GPU quantile CV.

    Tries GPU (``device=cuda``) first when ``prefer_gpu``; falls back to CPU if CUDA is unsupported.
    Default quantiles are **0.1, 0.5, 0.9**.
    """
    df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
    quantiles_sorted = _normalized_quantiles(quantiles)
    if model_params is None:
        model_params = {}

    mlflow.set_experiment(experiment_name)
    alpha_str = ",".join(str(q) for q in quantiles_sorted)
    metrics_per_fold: list[dict] = []

    splits = _cv_split_indices(
        df,
        time_col,
        cv=cv,
        train_quarters=train_quarters,
        val_quarters=val_quarters,
        step_quarters=step_quarters,
        n_splits=n_splits,
    )

    with mlflow.start_run(run_name=run_name + "_+saved_model" if save_model else run_name):
        mlflow.log_param("task_type", "xgboost_native_multi_quantile")
        mlflow.log_param("quantiles", alpha_str)
        mlflow.log_param("description", description)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("prefer_gpu", prefer_gpu)
        mlflow.log_param("cv_scheme", cv)
        mlflow.log_param("n_folds", len(splits))
        if cv == "sliding_quarters":
            mlflow.log_param("train_quarters", train_quarters)
            mlflow.log_param("val_quarters", val_quarters)
            mlflow.log_param("step_quarters", step_quarters)
        else:
            mlflow.log_param("time_series_splits", n_splits)

        last_booster: xgb.Booster | None = None
        device_used_fold: str = "unknown"

        pbar = tqdm(enumerate(splits, 1), total=len(splits), desc="xgboost mq CV", leave=False)
        for fold, (train_idx, val_idx) in pbar:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            X_train = _copy_features_as_tree_categories(train_df, features, cat_features)
            X_val = _copy_features_as_tree_categories(val_df, features, cat_features)
            y_train = train_df[target]
            y_val = val_df[target]

            if verbose:
                print(f"[XGB mq] Fold {fold} train [{train_df[time_col].min()} … {train_df[time_col].max()}]")
                print(f"[XGB mq] Fold {fold} val   [{val_df[time_col].min()} … {val_df[time_col].max()}]")

            booster, device_used_fold, dval = _xgb_fit_quantile_fold(
                X_train,
                y_train,
                X_val,
                y_val,
                quantiles_sorted,
                model_params,
                prefer_gpu=prefer_gpu,
                cat_features=cat_features,
            )
            last_booster = booster

            y_pred = _xgb_predict_quantile_matrix(
                booster,
                dval,
                n_rows=len(y_val),
                n_quantiles=len(quantiles_sorted),
            )

            fold_metrics: dict = {"fold": fold, "xgboost_device": device_used_fold}
            for qi, q in enumerate(quantiles_sorted):
                q_pred = y_pred[:, qi]
                pb = mean_pinball_loss(y_true=y_val, y_pred=q_pred, alpha=q)
                cov = coverage(y_val, q_pred)
                fold_metrics[f"pinball_q{q}"] = pb
                fold_metrics[f"coverage_q{q}"] = cov
                mlflow.log_metric(f"pinball_q{q}", pb, step=fold)
                mlflow.log_metric(f"coverage_q{q}", cov, step=fold)

            width = float(np.mean(y_pred[:, -1] - y_pred[:, 0])) if len(quantiles_sorted) >= 2 else float("nan")
            fold_metrics["interval_width"] = width
            mlflow.log_metric("interval_width", width, step=fold)
            metrics_per_fold.append(fold_metrics)

        metrics_df = pd.DataFrame(metrics_per_fold)
        if not metrics_df.empty:
            drop_cols = [c for c in ("fold", "xgboost_device") if c in metrics_df.columns]
            agg = metrics_df.drop(columns=drop_cols, errors="ignore")
            mean_metrics = agg.mean(numeric_only=True).to_dict()
            std_metrics = agg.std(numeric_only=True).to_dict()
            for k, v in mean_metrics.items():
                mlflow.log_metric(f"{k}_mean", float(v))
            for k, v in std_metrics.items():
                mlflow.log_metric(f"{k}_std", float(v))

        metrics_path = "cv_xgboost_multi_quantile_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        mlflow.log_artifact(metrics_path)
        os.remove(metrics_path)

        if last_booster is not None and len(features) <= 128:
            try:
                imp = last_booster.get_score(importance_type="gain") or {}
                imp_df = pd.DataFrame([{"Feature": k, "Importance": v} for k, v in imp.items()])
                if not imp_df.empty:
                    imp_df = imp_df.sort_values(by="Importance", ascending=False)
                    plt.figure(figsize=(10, 6))
                    plt.barh(imp_df["Feature"].head(30), imp_df["Importance"].head(30), color="skyblue")
                    plt.xlabel("Gain")
                    plt.title(f"XGBoost multi-quantile (fold last, device={device_used_fold})")
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plot_path = "xgboost_multi_quantile_feature_importance.png"
                    plt.savefig(plot_path)
                    plt.close()
                    mlflow.log_artifact(plot_path)
                    os.remove(plot_path)
            except Exception as e:
                warnings.warn(f"Could not compute XGBoost feature importance: {e}")

        if save_model:
            if not model_name:
                model_name = "xgboost_multi_quantile"
            X_full = _copy_features_as_tree_categories(df, features, cat_features)
            y_full = np.asarray(df[target], dtype=np.float64).ravel()
            dq = xgb.QuantileDMatrix(X_full, y_full, enable_categorical=bool(cat_features))
            final_params = _merge_xgboost_quantile_booster_params(
                quantiles_sorted,
                model_params,
                device="cuda" if prefer_gpu else "cpu",
            )
            try:
                final = xgb.train(
                    final_params,
                    dq,
                    num_boost_round=int(model_params.get("iterations", 5000)),
                    verbose_eval=False,
                )
            except Exception:
                final_params = _merge_xgboost_quantile_booster_params(
                    quantiles_sorted, model_params, device="cpu"
                )
                final = xgb.train(
                    final_params,
                    dq,
                    num_boost_round=int(model_params.get("iterations", 5000)),
                    verbose_eval=False,
                )
            out_path = f"{model_name}.json"
            final.save_model(out_path)
            mlflow.log_artifact(out_path)
            os.remove(out_path)

    return metrics_df


def run_time_series_cv_xgboost_quantile(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    quantiles: list[float],
    experiment_name: str,
    run_name: str,
    description: str,
    model_params: dict | None = None,
    n_splits: int = 5,
    save_model: bool = False,
) -> pd.DataFrame:
    """Backward-compatible shim: wraps :func:`run_time_series_cv_xgboost_multi_quantile` with ``TimeSeriesSplit``."""
    return run_time_series_cv_xgboost_multi_quantile(
        df,
        features,
        target,
        time_col,
        experiment_name,
        run_name,
        description,
        quantiles=quantiles,
        model_params=model_params,
        cv="time_series_split",
        n_splits=n_splits,
        save_model=save_model,
        model_name="xgboost_multi_quantile",
    )


def run_time_series_cv_xgboost_quantile_quarters(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    quantiles: list[float],
    experiment_name: str,
    run_name: str,
    description: str,
    model_params: dict | None = None,
    train_quarters: int = 4,
    val_quarters: int = 1,
    step_quarters: int = 1,
    save_model: bool = False,
) -> pd.DataFrame:
    """Backward-compatible shim: sliding-quarter CV with the smaller default train span (original helper)."""
    return run_time_series_cv_xgboost_multi_quantile(
        df,
        features,
        target,
        time_col,
        experiment_name,
        run_name,
        description,
        quantiles=quantiles,
        model_params=model_params,
        cv="sliding_quarters",
        train_quarters=train_quarters,
        val_quarters=val_quarters,
        step_quarters=step_quarters,
        save_model=save_model,
        model_name="xgboost_multi_quantile_quarters",
    )


def _merge_lgbm_regressor_params(model_params: dict, *, alpha: float, device: str) -> dict:
    forbid = {"objective", "alpha", "device", "verbosity", "verbose"}
    n_estimators = int(model_params.get("iterations", model_params.get("n_estimators", 500)))
    out: dict = {
        "objective": "quantile",
        "alpha": alpha,
        "n_estimators": n_estimators,
        "learning_rate": float(model_params.get("learning_rate", 0.05)),
        "max_depth": int(model_params.get("max_depth", -1)),
        "subsample": float(model_params.get("subsample", 0.8)),
        "colsample_bytree": float(model_params.get("colsample_bytree", 1.0)),
        "reg_lambda": float(model_params.get("reg_lambda", 0.0)),
        "reg_alpha": float(model_params.get("reg_alpha", 0.0)),
        "random_state": int(model_params.get("random_seed", model_params.get("random_state", 42))),
        "verbosity": int(model_params.get("verbosity", -1)),
        "device": device,
    }
    for k, v in model_params.items():
        if k in forbid or k in out:
            continue
        if k in {
            "iterations",
            "early_stopping_rounds",
            "verbose",
            "random_seed",
            "prefer_gpu",
        }:
            continue
        out[k] = v
    return out


def _lightgbm_fit_quantile_bundle(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles_sorted: list[float],
    model_params: dict,
    *,
    prefer_gpu: bool,
    cat_features: list[str] | None,
    verbose: bool,
) -> tuple[list[lgb.LGBMRegressor], list[str]]:
    """
    LightGBM has **no** built-in simultaneous multi-alpha quantile objective like XGBoost.
    This bundles one ``LGBMRegressor`` per quantile; tries ``device=cuda`` (when GPU build exists),
    falls back on failure to ``cpu``.
    """
    callbacks: list = []
    esr = int(model_params.get("early_stopping_rounds", 50))
    if esr > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=esr, verbose=False))
    if not verbose:
        callbacks.append(lgb.log_evaluation(period=0))

    fitted: list[lgb.LGBMRegressor] = []
    devices_logged: list[str] = []

    for q in quantiles_sorted:
        last_err = None
        dev_order = ["cuda", "cpu"] if prefer_gpu else ["cpu"]
        model: lgb.LGBMRegressor | None = None
        picked = "cpu"

        for dev in dev_order:
            try:
                kw = _merge_lgbm_regressor_params(model_params, alpha=q, device=dev)
                model = lgb.LGBMRegressor(**kw)
                fit_kw: dict = {
                    "eval_set": [(X_val, y_val)],
                    "categorical_feature": "auto",
                }
                if callbacks:
                    fit_kw["callbacks"] = callbacks
                model.fit(X_train, y_train, **fit_kw)
                picked = dev
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if dev == "cuda":
                    warnings.warn(f"LightGBM quantile α={q}: CUDA failed ({last_err}); retrying CPU.")
                    continue
                raise RuntimeError(last_err) from e

        if model is None:
            raise RuntimeError(last_err or "LightGBM fit failed")
        fitted.append(model)
        devices_logged.append(picked)

    return fitted, devices_logged


def _lgbm_predict_with_best_iter(model: lgb.LGBMRegressor, X: pd.DataFrame) -> np.ndarray:
    bi = getattr(model, "best_iteration_", None)
    if bi is not None and bi > 0:
        return model.predict(X, num_iteration=int(bi))
    return model.predict(X)


def run_time_series_cv_lightgbm_multi_quantile(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    experiment_name: str,
    run_name: str,
    description: str,
    quantiles: list[float] | None = None,
    model_params: dict | None = None,
    train_quarters: int = 4 * 6 * 2,
    val_quarters: int = 1,
    step_quarters: int = 1,
    cv: Literal["sliding_quarters", "time_series_split"] = "sliding_quarters",
    n_splits: int = 5,
    save_model: bool = False,
    model_name: str | None = None,
    verbose: bool = False,
    prefer_gpu: bool = True,
    cat_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Same CV / MLflow shape as CatBoost / XGBoost quantile helpers. LightGBM needs **three separate**
    quantile boosters internally; bundled here so inference returns one matrix shaped ``(n, 3)``.

    Preference order when ``prefer_gpu`` is ``True``: try CUDA device per quantile booster, otherwise CPU.

    Default quantiles **0.1, 0.5, 0.9**.
    """
    df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
    quantiles_sorted = _normalized_quantiles(quantiles)
    if model_params is None:
        model_params = {}

    mlflow.set_experiment(experiment_name)
    alpha_str = ",".join(str(q) for q in quantiles_sorted)
    splits = _cv_split_indices(
        df,
        time_col,
        cv=cv,
        train_quarters=train_quarters,
        val_quarters=val_quarters,
        step_quarters=step_quarters,
        n_splits=n_splits,
    )
    metrics_per_fold: list[dict] = []
    last_med_model: lgb.LGBMRegressor | None = None

    with mlflow.start_run(run_name=run_name + "_+saved_model" if save_model else run_name):
        mlflow.log_param("task_type", "lightgbm_quantile_bundle_3_boosters")
        mlflow.log_param("quantiles", alpha_str)
        mlflow.log_param("description", description)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("prefer_gpu", prefer_gpu)
        mlflow.log_param("cv_scheme", cv)
        mlflow.log_param("n_folds", len(splits))

        pbar = tqdm(enumerate(splits, 1), total=len(splits), desc="lgb mq CV", leave=False)
        for fold, (train_idx, val_idx) in pbar:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            X_train = _copy_features_as_tree_categories(train_df, features, cat_features)
            X_val = _copy_features_as_tree_categories(val_df, features, cat_features)
            y_train, y_val = train_df[target], val_df[target]

            models, devs_used = _lightgbm_fit_quantile_bundle(
                X_train,
                y_train,
                X_val,
                y_val,
                quantiles_sorted,
                model_params,
                prefer_gpu=prefer_gpu,
                cat_features=cat_features,
                verbose=verbose,
            )
            preds = np.column_stack([_lgbm_predict_with_best_iter(m, X_val) for m in models])
            if 0.5 in quantiles_sorted:
                idx_med = quantiles_sorted.index(0.5)
                last_med_model = models[idx_med]

            fold_metrics: dict = {"fold": fold, "lightgbm_devices_used": ";".join(devs_used)}
            for qi, q in enumerate(quantiles_sorted):
                q_pred = preds[:, qi]
                pb = mean_pinball_loss(y_true=y_val, y_pred=q_pred, alpha=q)
                cov = coverage(y_val, q_pred)
                fold_metrics[f"pinball_q{q}"] = pb
                fold_metrics[f"coverage_q{q}"] = cov
                mlflow.log_metric(f"pinball_q{q}", pb, step=fold)
                mlflow.log_metric(f"coverage_q{q}", cov, step=fold)

            fold_metrics["interval_width"] = float(np.mean(preds[:, -1] - preds[:, 0])) if len(quantiles_sorted) >= 2 else float("nan")
            mlflow.log_metric("interval_width", fold_metrics["interval_width"], step=fold)
            metrics_per_fold.append(fold_metrics)

        metrics_df = pd.DataFrame(metrics_per_fold)
        if not metrics_df.empty:
            drop_cols = [c for c in ("fold", "lightgbm_devices_used") if c in metrics_df.columns]
            agg = metrics_df.drop(columns=drop_cols, errors="ignore")
            for k, v in agg.mean(numeric_only=True).to_dict().items():
                mlflow.log_metric(f"{k}_mean", float(v))
            for k, v in agg.std(numeric_only=True).to_dict().items():
                mlflow.log_metric(f"{k}_std", float(v))

        path_csv = "cv_lightgbm_multi_quantile_metrics.csv"
        metrics_df.to_csv(path_csv, index=False)
        mlflow.log_artifact(path_csv)
        os.remove(path_csv)

        if last_med_model is not None and hasattr(last_med_model, "feature_importances_"):
            try:
                imp = last_med_model.feature_importances_
                names = features
                imp_df = pd.DataFrame({"Feature": names, "Importance": imp[: len(names)]}).sort_values(
                    by="Importance", ascending=False
                )
                plt.figure(figsize=(10, 6))
                plt.barh(imp_df["Feature"].head(30), imp_df["Importance"].head(30), color="lightgreen")
                plt.xlabel("Importance (gain / split)")
                plt.title("LightGBM — median-quantile (α=0.5) booster, last fold")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                pth = "lightgbm_median_quantile_importance.png"
                plt.savefig(pth)
                plt.close()
                mlflow.log_artifact(pth)
                os.remove(pth)
            except Exception as e:
                warnings.warn(f"LightGBM importance plot skipped: {e}")

        if save_model:
            if not model_name:
                model_name = "lightgbm_quantile_bundle"
            X_full = _copy_features_as_tree_categories(df, features, cat_features)
            y_full = df[target]
            log_cb = [lgb.log_evaluation(period=0)]

            for q in quantiles_sorted:
                last_exc: BaseException | None = None
                fitted = False
                for dev in (["cuda", "cpu"] if prefer_gpu else ["cpu"]):
                    try:
                        kw = _merge_lgbm_regressor_params(model_params, alpha=q, device=dev)
                        m_final = lgb.LGBMRegressor(**kw)
                        m_final.fit(
                            X_full,
                            y_full,
                            categorical_feature="auto",
                            callbacks=log_cb,
                        )
                        outp = f"{model_name}_q{q}.txt"
                        m_final.booster_.save_model(outp)
                        mlflow.log_artifact(outp)
                        os.remove(outp)
                        fitted = True
                        break
                    except BaseException as e:
                        last_exc = e
                        if dev == "cuda":
                            warnings.warn(
                                f"LightGBM final τ={q}: CUDA/train failed ({e}); retrying on CPU.",
                            )
                            continue
                        raise RuntimeError(last_exc) from last_exc
                if not fitted:
                    raise RuntimeError(f"Final LightGBM fit failed for τ={q}: {last_exc!r}") from last_exc

    return metrics_df

