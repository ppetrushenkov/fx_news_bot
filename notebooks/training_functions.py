import os
import mlflow
import mlflow.catboost
import mlflow.xgboost

import mlflow.shap
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import coverage

import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoost

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

        # # 6 calendar years train → 1 year val; slide val start 1 year forward each fold
        # year_cv = SlidingYearBlockCV(
        #     time_col=time_col,
        #     train_years=6,
        #     val_years=1,
        #     step_years=1,
        # )
        # splits = list(year_cv.split(df))
        # mlflow.log_param("cv_scheme", "SlidingYearBlockCV_6Y_train_1Y_val_step_1Y")
        # mlflow.log_param("n_folds", len(splits))

        # CV scheme (keep consistent with your existing quantile CV)
        quarter_cv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=4 * 6 * 2,
            val_quarters=1,
            step_quarters=1,
        )
        splits = list(quarter_cv.split(df))
        mlflow.log_param(
            "cv_scheme",
            "SlidingQuarterBlockCV_train_48Q_val_1Q_step_1Q",
        )
        mlflow.log_param("n_folds", len(splits))
        mlflow.log_param("n_splits_arg", n_splits)

        pbar = tqdm(enumerate(splits, 1), total=len(splits), desc='Processing splits', leave=False)
        for fold, (train_idx, val_idx) in pbar:

        # for fold, (train_idx, val_idx) in enumerate(
        #     TimeSeriesSplit(n_splits=n_splits).split(df), 1
        # ):
        # for fold, (train_idx, val_idx) in enumerate(splits, 1):
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
                model = CatBoostRegressor(**ord_kw)
            elif task_type == "classification":
                model = CatBoostClassifier(
                    **gpu_kw,
                    **model_params,
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

            else:
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

                fold_metrics = {
                    "Train dates": train_dates,
                    "Val dates": val_dates,
                    "fold": fold,
                    "Accuracy": accuracy_score(y_val, y_pred),
                    "Precision": precision_score(y_val, y_pred, zero_division=0),
                    "Recall": recall_score(y_val, y_pred, zero_division=0),
                    "F1": f1_score(y_val, y_pred, zero_division=0),
                    "ROC_AUC": roc_auc_score(y_val, y_proba),
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
            final_model = CatBoostRegressor(
                iterations=model_params.get("iterations", 1000),
                learning_rate=model_params.get("learning_rate", 0.1),
                depth=model_params.get("depth", 6),
                loss_function=loss,
                eval_metric=eval_m,
                task_type="GPU",
                devices="0",
                rsm=model_params.get("rsm", 1.0),
                l2_leaf_reg=model_params.get("l2_leaf_reg", 3.0),
                model_size_reg=model_params.get("model_size_reg", 0.5),
                random_seed=model_params.get("random_seed", 42),
                verbose=100,
                **{
                    k: v
                    for k, v in model_params.items()
                    if k
                    not in (
                        "iterations",
                        "learning_rate",
                        "depth",
                        "loss_function",
                        "eval_metric",
                        "random_seed",
                        "rsm",
                        "l2_leaf_reg",
                        "model_size_reg",
                    )
                },
            )
            final_model.fit(X, y, cat_features=cat_features)
            y_pred_final = final_model.predict(X)
            print("\nFinal model metrics (full data, in-sample):")
            if task_type == "ordinal":
                fm = ordinal_gradation_metrics(y, y_pred_final)
                for k, v in fm.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"  MAE:  {mean_absolute_error(y, y_pred_final):.4f}")
                print(f"  RMSE: {mean_squared_error(y, y_pred_final) ** 0.5:.4f}")
        else:
            final_model = CatBoostClassifier(
                iterations=1000,
                task_type="GPU",
                devices="0",
                learning_rate=0.1,
                depth=6,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=100,
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
    cat_features: list[str] | None = None,
    model_params: dict | None = None,
    n_splits: int = 5,
    save_model: bool = False,
    model_name: str = None
):
    """
    Time-series CV for CatBoost Multi-Quantile Regression with MLflow logging.
    """
    df = df.sort_values(by=time_col, ascending=True)

    # mlflow.set_experiment(experiment_name + '_+saved_model' if save_model else experiment_name)
    mlflow.set_experiment(experiment_name)

    if model_params is None:
        model_params = {}

    metrics_per_fold = []

    alphas = ",".join(str(q) for q in quantiles)
    loss_fn = f"MultiQuantile:alpha={alphas}"

    # with mlflow.start_run(run_name=run_name):
    with mlflow.start_run(run_name=run_name + '_+saved_model' if save_model else run_name):

        # -------------------------
        # META
        # -------------------------
        mlflow.log_param("task_type", "multi_quantile_regression")
        mlflow.log_param("quantiles", alphas)
        mlflow.log_param("description", description)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("features", ",".join(features))

        # for fold, (train_idx, val_idx) in enumerate(
        #     TimeSeriesSplit(n_splits=n_splits).split(df), 1
        # ):
        # for fold, (train_idx, val_idx) in enumerate(
        #     QuantileSplitter(n_splits=n_splits).split(df), 1
        # ):
        quarter_cv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=4*6*2,
            val_quarters=1,
            step_quarters=1
        )
        splits = list(quarter_cv.split(df))
        mlflow.log_param("cv_scheme", "SlidingYearBlockCV_6Y_train_1Y_val_step_1Y")
        mlflow.log_param("n_folds", len(splits))

        # for fold, (train_idx, val_idx) in enumerate(
        #     TimeSeriesSplit(n_splits=n_splits).split(df), 1
        # ):
        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            print(f'[INFO] FOLD: {fold}')
            print('[INFO] Min train date:', train_df[time_col].min())
            print('[INFO] Max train date:', train_df[time_col].max())
            print('[INFO] Min val date:', val_df[time_col].min())
            print('[INFO] Max val date:', val_df[time_col].max())
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
                # rsm=model_params.get("rsm", 1.0),
                # l2_leaf_reg=model_params.get("l2_leaf_reg", 3.0),
                # model_size_reg=model_params.get("model_size_reg", 0.8),
                # **{k: v for k, v in model_params.items() if k not in ("rsm", "l2_leaf_reg", "model_size_reg")}
            )

            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True
            )

            y_pred = model.predict(X_val)  # shape: (n, n_quantiles)

            fold_metrics = {"fold": fold}

            for i, q in enumerate(quantiles):
                q_pred = y_pred[:, i]

                fold_metrics[f"pinball_q{q}"] = mean_pinball_loss(y_true=y_val, y_pred=q_pred, alpha=q)
                fold_metrics[f"coverage_q{q}"] = coverage(y_val, q_pred)

                mlflow.log_metric(f"pinball_q{q}", mean_pinball_loss(y_true=y_val, y_pred=q_pred, alpha=q), step=fold)
                mlflow.log_metric(f"coverage_q{q}", coverage(y_val, q_pred), step=fold)

            # interval metrics (between min & max quantile)
            fold_metrics["interval_width"] = np.mean(
                y_pred[:, -1] - y_pred[:, 0]
            )
            mlflow.log_metric("interval_width", np.mean(
                y_pred[:, -1] - y_pred[:, 0]
            ), step=fold)

            metrics_per_fold.append(fold_metrics)

        # -------------------------
        # AGGREGATION
        # -------------------------
        metrics_df = pd.DataFrame(metrics_per_fold)

        mean_metrics = metrics_df.drop(columns="fold").mean().to_dict()
        std_metrics = metrics_df.drop(columns="fold").std().to_dict()

        for k, v in mean_metrics.items():
            mlflow.log_metric(f"{k}_mean", v)

        for k, v in std_metrics.items():
            mlflow.log_metric(f"{k}_std", v)

        metrics_df.to_csv("cv_quantile_metrics.csv", index=False)
        mlflow.log_artifact("cv_quantile_metrics.csv")
        os.remove("cv_quantile_metrics.csv")


        # ------------------------
        # ML SHAP
        # ------------------------
        # mlflow.shap.log_explanation(model, X_val)

        # explainer = shap.Explainer(model, X_val)
        # shap_values = explainer(X_val)

        # # SHAP bar plot for feature importance
        # shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
        # shap_plot_path = "shap_feature_importance.png"
        # plt.savefig(shap_plot_path)
        # plt.close()

        # # Log SHAP plot to MLflow
        # mlflow.log_artifact(shap_plot_path)
        # os.remove(shap_plot_path)

        # Save model
        if save_model:
            print("\n" + "=" * 80)
            print("Training final model on all data...")
            print("=" * 80)
            X = df[features]
            y = df[target]

            model = CatBoostRegressor(
                loss_function=loss_fn,
                eval_metric=loss_fn,
                iterations=1000,
                learning_rate=0.05,
                task_type="CPU",
                verbose=False
            )

            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True
            )

            # Log model to MLflow
            model_path = f"{model_name}.cbm"
            model.save_model(model_path)
            mlflow.log_artifact(model_path)
            os.remove(model_path)

            # ------------------------
            # FEATURE IMPORTANCE PLOT
            # ------------------------
            feature_importances = model.get_feature_importance(prettified=True)
            importance_df = pd.DataFrame(feature_importances, columns=["Feature", "Importance"])
            importance_df = importance_df.sort_values(by="Importance", ascending=False)

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.title("Feature Importance")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # Save plot
            plot_path = "feature_importance.png"
            plt.savefig(plot_path)
            plt.close()

            # Log plot to MLflow
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

    return metrics_df


def run_time_series_cv_catboost_quantile_gpu(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    quantiles: list[float],
    experiment_name: str,
    run_name: str,
    description: str,
    cat_features: list[str] | None = None,
    model_params: dict | None = None,
    n_splits: int = 5,
    save_model: bool = False,
    model_name: str | None = None,
    verbose: bool = False
):
    """
    Time-series CV for CatBoost quantile regression with GPU support and MLflow logging.

    CatBoost ``MultiQuantile`` does not support GPU, so this function trains
    **separate** models for each quantile using ``Quantile:alpha=q`` on GPU and
    aggregates predictions/metrics in a single MLflow experiment/run.
    """
    df = df.sort_values(by=time_col, ascending=True)

    mlflow.set_experiment(experiment_name)

    if model_params is None:
        model_params = {}

    # basic validation / stable ordering
    quantiles = [float(q) for q in quantiles]
    quantiles_sorted = sorted(quantiles)
    alphas = ",".join(str(q) for q in quantiles_sorted)

    metrics_per_fold: list[dict] = []

    devices = str(model_params.get("devices", "0"))
    verbose = bool(model_params.get("verbose", False))

    # parameters that belong to loss/eval per-quantile model, not shared
    forbidden_loss_keys = {"loss_function", "eval_metric"}

    with mlflow.start_run(run_name=run_name + "_+saved_model" if save_model else run_name):
        # -------------------------
        # META
        # -------------------------
        mlflow.log_param("task_type", "quantile_regression_gpu_separate_models")
        mlflow.log_param("quantiles", alphas)
        mlflow.log_param("description", description)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("catboost_task_type", "GPU")
        mlflow.log_param("catboost_devices", devices)

        # CV scheme (keep consistent with your existing quantile CV)
        quarter_cv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=4 * 6 * 2,
            val_quarters=1,
            step_quarters=1,
        )
        splits = list(quarter_cv.split(df))
        mlflow.log_param(
            "cv_scheme",
            "SlidingQuarterBlockCV_train_48Q_val_1Q_step_1Q",
        )
        mlflow.log_param("n_folds", len(splits))
        mlflow.log_param("n_splits_arg", n_splits)

        pbar = tqdm(enumerate(splits, 1), total=len(splits), desc='Processing splits', leave=False)
        for fold, (train_idx, val_idx) in pbar:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            if verbose:
                print(f"[INFO] FOLD: {fold}")
                print("[INFO] Min train date:", train_df[time_col].min())
                print("[INFO] Max train date:", train_df[time_col].max())
                print("[INFO] Min val date:", val_df[time_col].min())
                print("[INFO] Max val date:", val_df[time_col].max())
                print("[INFO] Train count:", train_df.shape[0])
                print("[INFO] Val count:", val_df.shape[0])

            pbar.set_postfix({
                "Fold": fold,
                "start train": train_df[time_col].min().date(), 
                "end train": train_df[time_col].max().date(),
                "start val": val_df[time_col].min().date(), 
                "end val": val_df[time_col].max().date()
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

            fold_metrics: dict = {"fold": fold}

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

        if not metrics_df.empty:
            mean_metrics = metrics_df.drop(columns="fold").mean(numeric_only=True).to_dict()
            std_metrics = metrics_df.drop(columns="fold").std(numeric_only=True).to_dict()

            for k, v in mean_metrics.items():
                mlflow.log_metric(f"{k}_mean", float(v))

            for k, v in std_metrics.items():
                mlflow.log_metric(f"{k}_std", float(v))

        metrics_df.to_csv("cv_catboost_quantile_gpu_metrics.csv", index=False)
        mlflow.log_artifact("cv_catboost_quantile_gpu_metrics.csv")
        os.remove("cv_catboost_quantile_gpu_metrics.csv")

        # -------------------------
        # SAVE FINAL MODELS (per quantile)
        # -------------------------
        if save_model:
            print("\n" + "=" * 80)
            print("Training final GPU quantile models on all data...")
            print("=" * 80)

            if not model_name:
                model_name = "catboost_quantile_gpu"

            X_full = df[features]
            y_full = df[target]

            # pick a representative quantile for feature importance plot (closest to 0.5)
            rep_q = min(quantiles_sorted, key=lambda x: abs(x - 0.5))
            rep_model: CatBoostRegressor | None = None

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
                for k, v in model_params.items():
                    if k in params or k in forbidden_loss_keys:
                        continue
                    params[k] = v

                m = CatBoostRegressor(**params)
                m.fit(X_full, y_full, cat_features=cat_features)

                if q == rep_q:
                    rep_model = m

                model_path = f"{model_name}_q{q}.cbm"
                m.save_model(model_path)
                mlflow.log_artifact(model_path)
                os.remove(model_path)

            # Feature importance plot for representative quantile model
            if rep_model is not None:
                feature_importances = rep_model.get_feature_importance(prettified=True)
                importance_df = pd.DataFrame(feature_importances, columns=["Feature", "Importance"])
                importance_df = importance_df.sort_values(by="Importance", ascending=False)

                plt.figure(figsize=(10, 6))
                plt.barh(importance_df["Feature"].head(30), importance_df["Importance"].head(30), color="skyblue")
                plt.xlabel("Importance")
                plt.ylabel("Features")
                plt.title(f"Feature Importance (Quantile {rep_q}, GPU)")
                plt.gca().invert_yaxis()
                plt.tight_layout()

                plot_path = f"feature_importance_q{rep_q}.png"
                plt.savefig(plot_path)
                plt.close()

                mlflow.log_artifact(plot_path)
                os.remove(plot_path)

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
    save_model: bool = False
):
    """
    Time-series CV for XGBoost Multi-Quantile Regression with GPU support and MLflow logging.

    Uses XGBoost's MultiQuantile loss for predicting multiple quantiles (0.1, 0.5, 0.9) simultaneously.
    GPU is enabled via tree_method='hist' with device='cuda'.
    """
    df = df.sort_values(by=time_col, ascending=True)

    mlflow.set_experiment(experiment_name)

    if model_params is None:
        model_params = {}

    metrics_per_fold = []

    alphas = ",".join(str(q) for q in quantiles)
    loss_fn = f"multi:softmax"  # XGBoost uses multi:softmax for MultiQuantile

    with mlflow.start_run(run_name=run_name + '_+saved_model' if save_model else run_name):

        # -------------------------
        # META
        # -------------------------
        mlflow.log_param("task_type", "xgboost_multi_quantile_regression")
        mlflow.log_param("quantiles", alphas)
        mlflow.log_param("description", description)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("features", ",".join(features))

        for fold, (train_idx, val_idx) in enumerate(
            TimeSeriesSplit(n_splits=n_splits).split(df), 1
        ):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            print(f'FOLD {fold}')
            print('Min train date', train_df[time_col].min())
            print('Max train date', train_df[time_col].max())
            print('Min val date', val_df[time_col].min())
            print('Max val date', val_df[time_col].max())
            print('Train count', train_df.shape[0])
            print('Val count', val_df.shape[0])

            X_train = train_df[features]
            y_train = train_df[target]
            X_val = val_df[features]
            y_val = val_df[target]

            # XGBoost MultiQuantile requires target to be stacked with quantiles
            # Shape for MultiQuantile: (n_samples, n_quantiles)
            # We create y_train_multi and y_val_multi for the multi-output objective
            y_train_multi = np.column_stack([y_train.values] * len(quantiles))
            y_val_multi = np.column_stack([y_val.values] * len(quantiles))

            # Build XGBoost DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train_multi, enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val_multi, enable_categorical=True)

            # XGBoost parameters for MultiQuantile with GPU
            params = {
                "objective": "reg:quantileerror",
                "num_target": len(quantiles),
                "quantile_alpha": np.array(quantiles),
                "tree_method": "hist",  # Use hist for faster training
                "device": "cuda",  # Enable GPU
                "learning_rate": model_params.get("learning_rate", 0.05),
                "max_depth": model_params.get("max_depth", 6),
                "subsample": model_params.get("subsample", 0.8),
                "colsample_bytree": model_params.get("colsample_bytree", 1.0),
                "reg_lambda": model_params.get("reg_lambda", 1.0),
                "reg_alpha": model_params.get("reg_alpha", 0.0),
                "random_state": model_params.get("random_seed", 42),
                "verbosity": 0,
            }

            # Train with evaluation set
            evals = [(dtrain, "train"), (dval, "val")]
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=model_params.get("iterations", 5000),
                evals=evals,
                early_stopping_rounds=model_params.get("early_stopping_rounds", 100),
                verbose_eval=model_params.get("verbose_eval", 100),
            )

            # Predict - returns shape (n_samples, n_quantiles)
            y_pred = model.predict(dval)

            fold_metrics = {"fold": fold}

            for i, q in enumerate(quantiles):
                q_pred = y_pred[:, i]

                fold_metrics[f"pinball_q{q}"] = mean_pinball_loss(y_val, q_pred, alpha=q)
                fold_metrics[f"coverage_q{q}"] = coverage(y_val, q_pred)

                mlflow.log_metric(f"pinball_q{q}", mean_pinball_loss(y_val, q_pred, alpha=q), step=fold)
                mlflow.log_metric(f"coverage_q{q}", coverage(y_val, q_pred), step=fold)

            # interval metrics (between min & max quantile)
            fold_metrics["interval_width"] = np.mean(
                y_pred[:, -1] - y_pred[:, 0]
            )
            mlflow.log_metric("interval_width", np.mean(
                y_pred[:, -1] - y_pred[:, 0]
            ), step=fold)

            metrics_per_fold.append(fold_metrics)

        # -------------------------
        # AGGREGATION
        # -------------------------
        metrics_df = pd.DataFrame(metrics_per_fold)

        mean_metrics = metrics_df.drop(columns="fold").mean().to_dict()
        std_metrics = metrics_df.drop(columns="fold").std().to_dict()

        for k, v in mean_metrics.items():
            mlflow.log_metric(f"{k}_mean", v)

        for k, v in std_metrics.items():
            mlflow.log_metric(f"{k}_std", v)

        metrics_df.to_csv("cv_xgboost_quantile_metrics.csv", index=False)
        mlflow.log_artifact("cv_xgboost_quantile_metrics.csv")
        os.remove("cv_xgboost_quantile_metrics.csv")

        # ------------------------
        # FEATURE IMPORTANCE PLOT
        # ------------------------
        # Get feature importance from XGBoost
        importance = model.get_score(importance_type="gain")
        importance_df = pd.DataFrame([
            {"Feature": k, "Importance": v} for k, v in importance.items()
        ])
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("XGBoost Feature Importance (Gain)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        plot_path = "xgboost_feature_importance.png"
        plt.savefig(plot_path)
        plt.close()

        # Log plot to MLflow
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        print("\n" + "=" * 80)
        print("Training final XGBoost model on all data...")
        print("=" * 80)

        # Train final model on full data
        X_full = df[features]
        y_full = df[target]
        y_full_multi = np.column_stack([y_full.values] * len(quantiles))

        dtrain_full = xgb.DMatrix(X_full, label=y_full_multi)

        final_model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=model_params.get("iterations", 5000),
            verbose_eval=model_params.get("verbose_eval", 100),
        )

        y_pred_final = final_model.predict(dtrain_full)

        print("\nFinal XGBoost model metrics (full data, in-sample):")
        for i, q in enumerate(quantiles):
            q_pred_final = y_pred_final[:, i]
            print(f"  Quantile {q} - Pinball: {mean_pinball_loss(y_full, q_pred_final, alpha=q):.4f}, "
                  f"Coverage: {coverage(y_full, q_pred_final):.4f}")

    return metrics_df


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
    save_model: bool = False
):
    """
    Time-series CV for XGBoost Multi-Quantile Regression with Quarter-by-Quarter sliding window.

    Uses XGBoost's MultiQuantile loss for predicting multiple quantiles (0.1, 0.5, 0.9) simultaneously.
    GPU is enabled via tree_method='hist' with device='cuda'.

    Quarter sliding logic:
    - Train on train_quarters, validate on val_quarters
    - First fold: val_start = t_min + train_quarters
    - Each next fold: val_start advances by step_quarters
    """
    df = df.sort_values(by=time_col, ascending=True)

    mlflow.set_experiment(experiment_name)

    if model_params is None:
        model_params = {}

    metrics_per_fold = []

    alphas = ",".join(str(q) for q in quantiles)

    # Initialize the quarter-based CV
    quarter_cv = SlidingQuarterBlockCV(
        time_col=time_col,
        train_quarters=train_quarters,
        val_quarters=val_quarters,
        step_quarters=step_quarters,
    )
    splits = list(quarter_cv.split(df))

    with mlflow.start_run(run_name=run_name + '_+saved_model' if save_model else run_name):

        # -------------------------
        # META
        # -------------------------
        mlflow.log_param("task_type", "xgboost_multi_quantile_quarterly_cv")
        mlflow.log_param("quantiles", alphas)
        mlflow.log_param("description", description)
        mlflow.log_param("cv_scheme", f"SlidingQuarterBlockCV_{train_quarters}Q_train_{val_quarters}Q_val_step_{step_quarters}Q")
        mlflow.log_param("n_folds", len(splits))
        mlflow.log_param("features", ",".join(features))

        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            print(f'FOLD {fold}')
            print('Min train date', train_df[time_col].min())
            print('Max train date', train_df[time_col].max())
            print('Min val date', val_df[time_col].min())
            print('Max val date', val_df[time_col].max())
            print('Train count', train_df.shape[0])
            print('Val count', val_df.shape[0])

            X_train = train_df[features]
            y_train = train_df[target]
            X_val = val_df[features]
            y_val = val_df[target]

            # XGBoost MultiQuantile requires target to be stacked with quantiles
            y_train_multi = np.column_stack([y_train.values] * len(quantiles))
            y_val_multi = np.column_stack([y_val.values] * len(quantiles))

            # Build XGBoost DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train_multi)
            dval = xgb.DMatrix(X_val, label=y_val_multi)

            # XGBoost parameters for MultiQuantile with GPU
            params = {
                "objective": "multi:quantile",
                "num_target": len(quantiles),
                "quantile_alpha": quantiles,
                "tree_method": "hist",
                "device": "cuda",
                "learning_rate": model_params.get("learning_rate", 0.05),
                "max_depth": model_params.get("max_depth", 6),
                "subsample": model_params.get("subsample", 0.8),
                "colsample_bytree": model_params.get("colsample_bytree", 1.0),
                "reg_lambda": model_params.get("reg_lambda", 1.0),
                "reg_alpha": model_params.get("reg_alpha", 0.0),
                "random_state": model_params.get("random_seed", 42),
                "verbosity": 0,
            }

            # Train with evaluation set
            evals = [(dtrain, "train"), (dval, "val")]
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=model_params.get("iterations", 5000),
                evals=evals,
                early_stopping_rounds=model_params.get("early_stopping_rounds", 100),
                verbose_eval=model_params.get("verbose_eval", 100),
            )

            # Predict - returns shape (n_samples, n_quantiles)
            y_pred = model.predict(dval)

            fold_metrics = {"fold": fold}

            for i, q in enumerate(quantiles):
                q_pred = y_pred[:, i]

                fold_metrics[f"pinball_q{q}"] = mean_pinball_loss(y_val, q_pred, alpha=q)
                fold_metrics[f"coverage_q{q}"] = coverage(y_val, q_pred)

                mlflow.log_metric(f"pinball_q{q}", mean_pinball_loss(y_val, q_pred, alpha=q), step=fold)
                mlflow.log_metric(f"coverage_q{q}", coverage(y_val, q_pred), step=fold)

            # interval metrics (between min & max quantile)
            fold_metrics["interval_width"] = np.mean(
                y_pred[:, -1] - y_pred[:, 0]
            )
            mlflow.log_metric("interval_width", np.mean(
                y_pred[:, -1] - y_pred[:, 0]
            ), step=fold)

            metrics_per_fold.append(fold_metrics)

        # -------------------------
        # AGGREGATION
        # -------------------------
        metrics_df = pd.DataFrame(metrics_per_fold)

        mean_metrics = metrics_df.drop(columns="fold").mean().to_dict()
        std_metrics = metrics_df.drop(columns="fold").std().to_dict()

        for k, v in mean_metrics.items():
            mlflow.log_metric(f"{k}_mean", v)

        for k, v in std_metrics.items():
            mlflow.log_metric(f"{k}_std", v)

        metrics_df.to_csv("cv_xgboost_quarterly_metrics.csv", index=False)
        mlflow.log_artifact("cv_xgboost_quarterly_metrics.csv")
        os.remove("cv_xgboost_quarterly_metrics.csv")

        # ------------------------
        # FEATURE IMPORTANCE PLOT
        # ------------------------
        importance = model.get_score(importance_type="gain")
        importance_df = pd.DataFrame([
            {"Feature": k, "Importance": v} for k, v in importance.items()
        ])
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("XGBoost Feature Importance (Gain) - Quarterly CV")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_path = "xgboost_quarterly_feature_importance.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        print("\n" + "=" * 80)
        print("Training final XGBoost model on all data...")
        print("=" * 80)

        # Train final model on full data
        X_full = df[features]
        y_full = df[target]
        y_full_multi = np.column_stack([y_full.values] * len(quantiles))

        dtrain_full = xgb.DMatrix(X_full, label=y_full_multi)

        final_model = xgb.train(
            params,
            dtrain_full,
            num_boost_round=model_params.get("iterations", 5000),
            verbose_eval=model_params.get("verbose_eval", 100),
        )

        y_pred_final = final_model.predict(dtrain_full)

        print("\nFinal XGBoost model metrics (full data, in-sample):")
        for i, q in enumerate(quantiles):
            q_pred_final = y_pred_final[:, i]
            print(f"  Quantile {q} - Pinball: {mean_pinball_loss(y_full, q_pred_final, alpha=q):.4f}, "
                  f"Coverage: {coverage(y_full, q_pred_final):.4f}")

    return metrics_df
