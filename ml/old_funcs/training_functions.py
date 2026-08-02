import os
import mlflow
import mlflow.catboost

import mlflow.shap
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, CatBoostClassifier

import shap
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_pinball_loss
)

from sklearn.model_selection import TimeSeriesSplit


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
):
    """
    Run expanding-window time series CV for CatBoost with MLflow logging.
    """

    df = df.sort_values(by=time_col, ascending=True)

    mlflow.set_experiment(experiment_name)

    if model_params is None:
        model_params = {}

    metrics_per_fold = []
    

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("task_type", task_type)
        mlflow.log_param("description", description)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("features", ",".join(features))

        # for fold, (train_idx, val_idx) in enumerate(
        #     time_series_splits(df, time_col, n_splits=n_splits), 1
        # ):
        for fold, (train_idx, val_idx) in enumerate(
            TimeSeriesSplit(n_splits=n_splits).split(df), 1
        ):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            # print(f'FOLD {fold}')
            # print('Min train date', train_df['custom_event_time'].min())
            # print('Max train date', train_df['custom_event_time'].max())
            # print('Min val date', val_df['custom_event_time'].min())
            # print('Max val date', val_df['custom_event_time'].max())

            X_train = train_df[features]
            y_train = train_df[target]
            X_val = val_df[features]
            y_val = val_df[target]


            # -------------------------
            # MODEL
            # -------------------------
            if task_type == "regression":
                model = CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    task_type="GPU",
                    devices='0',
                    verbose=False,
                    **model_params
                )
            else:
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    task_type="GPU",
                    devices='0',
                    verbose=False,
                    **model_params
                )

            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True
            )
            # -------------------------
            # METRICS
            # -------------------------
            if task_type == "regression":
                y_pred = model.predict(X_val)

                fold_metrics = {
                    "fold": fold,
                    "MAE": mean_absolute_error(y_val, y_pred),
                    "MedianAE": median_absolute_error(y_val, y_pred),
                    "RMSE": mean_squared_error(y_val, y_pred)**0.5,
                    "R2": r2_score(y_val, y_pred),
                    "nMAE": normalized_mae(y_val, y_pred),
                    "Within_30pct": within_band_accuracy(y_val, y_pred),
                }

                mlflow.log_metric("MAE", mean_absolute_error(y_val, y_pred), step=fold)
                mlflow.log_metric("RMSE", mean_squared_error(y_val, y_pred)**0.5, step=fold)
                mlflow.log_metric("MedianAE", median_absolute_error(y_val, y_pred), step=fold)
                mlflow.log_metric("R2", r2_score(y_val, y_pred), step=fold)
                mlflow.log_metric("nMAE", normalized_mae(y_val, y_pred), step=fold)
                mlflow.log_metric("Within_30pct", within_band_accuracy(y_val, y_pred), step=fold)
            else:
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

                fold_metrics = {
                    "fold": fold,
                    "Accuracy": accuracy_score(y_val, y_pred),
                    "F1": f1_score(y_val, y_pred),
                    "ROC_AUC": roc_auc_score(y_val, y_proba),
                    "LogLoss": log_loss(y_val, y_proba),
                }

                mlflow.log_metric("Accuracy", accuracy_score(y_val, y_pred), step=fold)
                mlflow.log_metric("F1", f1_score(y_val, y_pred), step=fold)
                mlflow.log_metric("ROC_AUC", roc_auc_score(y_val, y_proba), step=fold)
                mlflow.log_metric("LogLoss", log_loss(y_val, y_proba), step=fold)

            metrics_per_fold.append(fold_metrics)

        # -------------------------
        # AGGREGATED METRICS
        # -------------------------
        metrics_df = pd.DataFrame(metrics_per_fold)

        # mean_metrics = metrics_df.drop(columns="fold").mean().to_dict()
        # std_metrics = metrics_df.drop(columns="fold").std().to_dict()

        # for k, v in mean_metrics.items():
        #     mlflow.log_metric(f"{k}_mean", v)

        # for k, v in std_metrics.items():
        #     mlflow.log_metric(f"{k}_std", v)

        # save fold table
        metrics_df.to_csv("cv_metrics_per_fold.csv", index=False)
        mlflow.log_artifact("cv_metrics_per_fold.csv")
        os.remove("cv_metrics_per_fold.csv")

        # ------------------------
        # FEATURE IMPORTANCE PLOT
        # ------------------------
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

        # ------------------------
        # ML SHAP
        # ------------------------
        mlflow.shap.log_explanation(model, X_val)

        # SHAP explanation for CatBoost model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_val)

        # SHAP bar plot for feature importance
        shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
        shap_plot_path = "shap_feature_importance.png"
        plt.savefig(shap_plot_path)  # Save the current figure
        plt.close()

        # Log SHAP plot to MLflow
        mlflow.log_artifact(shap_plot_path)
        os.remove(shap_plot_path)

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
):
    """
    Time-series CV for CatBoost Multi-Quantile Regression with MLflow logging.
    """

    mlflow.set_experiment(experiment_name)

    if model_params is None:
        model_params = {}

    metrics_per_fold = []

    alphas = ",".join(str(q) for q in quantiles)
    loss_fn = f"MultiQuantile:alpha={alphas}"

    with mlflow.start_run(run_name=run_name):

        # -------------------------
        # META
        # -------------------------
        mlflow.log_param("task_type", "multi_quantile_regression")
        mlflow.log_param("quantiles", alphas)
        mlflow.log_param("description", description)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("features", ",".join(features))

        for fold, (train_idx, val_idx) in enumerate(
            time_series_splits(df, time_col, n_splits=n_splits), 1
        ):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            print(f'FOLD {fold}')
            print('Min train date', train_df['custom_event_time'].min())
            print('Max train date', train_df['custom_event_time'].max())
            print('Min val date', val_df['custom_event_time'].min())
            print('Max val date', val_df['custom_event_time'].max())
            print('Train count', train_df.shape[0])
            print('Val count', val_df.shape[0])

            X_train = train_df[features]
            y_train = train_df[target]
            X_val = val_df[features]
            y_val = val_df[target]

            model = CatBoostRegressor(
                loss_function=loss_fn,
                eval_metric=loss_fn,
                verbose=False,
                **model_params
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

                fold_metrics[f"pinball_q{q}"] = mean_pinball_loss(y_val, q_pred, q)
                fold_metrics[f"coverage_q{q}"] = coverage(y_val, q_pred)

                mlflow.log_metric(f"pinball_q{q}", mean_pinball_loss(y_val, q_pred, q), step=fold)
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
        mlflow.shap.log_explanation(model, X_val)

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

        # ------------------------
        # FEATURE IMPORTANCE PLOT
        # ------------------------
        feature_importances = model.get_feature_importance(prettified=True)
        importance_df = pd.DataFrame(feature_importances, columns=["Feature", "Importance"])
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        # Plotting
        import matplotlib.pyplot as plt
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
