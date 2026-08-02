import numpy as np
import pandas as pd
import mlflow 
from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,  # PR-AUC
    roc_auc_score,
    f1_score,
    fbeta_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def select_best_threshold(y_true, y_prob, beta=2.0, min_recall=None):
    """
    Подбирает порог по F-beta (beta>1 => recall важнее precision).
    Если задан min_recall, ищет порог с максимальным precision при recall >= min_recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve возвращает на 1 элемент больше, чем thresholds — выравниваем
    precisions, recalls = precisions[:-1], recalls[:-1]

    if min_recall is not None:
        mask = recalls >= min_recall
        if mask.sum() == 0:
            best_idx = np.argmax(recalls)  # fallback: берём максимальный recall, что есть
        else:
            # среди порогов, дающих нужный recall, берём тот, что даёт максимальный precision
            best_idx = np.argmax(np.where(mask, precisions, -1))
    else:
        f_beta = (1 + beta**2) * precisions * recalls / (beta**2 * precisions + recalls + 1e-9)
        best_idx = np.argmax(f_beta)

    return float(thresholds[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])


def calibrate_probabilities(oof_probs, oof_true, method="isotonic"):
    """
    Обучает калибратор на OOF-предсказаниях, накопленных через walk-forward.
    method: 'isotonic' (нужно больше данных) или 'sigmoid' (Platt, безопаснее при малых данных).
    """
    oof_probs = np.asarray(oof_probs).reshape(-1, 1) if method == "sigmoid" else np.asarray(oof_probs)
    oof_true = np.asarray(oof_true)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_probs, oof_true)
        predict_fn = lambda p: calibrator.predict(np.asarray(p))
    elif method == "sigmoid":
        calibrator = LogisticRegression()
        calibrator.fit(oof_probs, oof_true)
        predict_fn = lambda p: calibrator.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]
    else:
        raise ValueError("method должен быть 'isotonic' или 'sigmoid'")

    return calibrator, predict_fn


def compute_classification_metrics(y_true, y_prob, threshold):
    """Считает полный набор метрик для заданного порога и вероятностей."""
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])

    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "log_loss": logloss,
        "f1": f1,
        "f2": f2,
        "precision": precision,
        "recall": recall,
        "threshold_used": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "base_rate": float(np.mean(y_true)),
    }


def train_catboost_binary_walkforward(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    quarter_col: str,
    mlflow_experiment_name: str,
    run_name: str = "catboost_binary",
    n_min_train_quarters: int = 1,
    catboost_params: dict = None,
    calibration_method: str = "isotonic",
    threshold_beta: float = 2.0,
    min_recall: float = None,
    cat_features: list = None,
    log_models_per_fold: bool = False,
):
    """
    Walk-forward обучение CatBoost по кварталам с накоплением OOF-предсказаний,
    калибровкой и логированием в MLflow.

    Параметры
    ---------
    df : DataFrame со всеми данными (фичи + таргет + колонка квартала)
    feature_cols : список колонок-фичей
    target_col : имя колонки с бинарным таргетом (0/1)
    quarter_col : имя колонки с кварталом (сортируемый тип: int, str вида '2024Q1' с лекс. порядком, или Period)
    n_min_train_quarters : минимальное число кварталов в обучающей выборке перед первой валидацией
    catboost_params : словарь параметров CatBoostClassifier (auto_class_weights ставится по умолчанию)
    calibration_method : 'isotonic' или 'sigmoid'
    threshold_beta : beta для подбора порога (2.0 => recall важнее precision, что обычно нужно для алертов)
    min_recall : если задан, порог подбирается по max precision при recall >= min_recall (приоритетнее threshold_beta)
    cat_features : список категориальных фичей (имена колонок), если есть
    log_models_per_fold : логировать ли модель каждого fold в MLflow (по умолчанию False, логируется только финальная)

    Возвращает
    ----------
    dict с финальной моделью, калибратором, порогом и таблицей OOF-предсказаний
    """
    default_params = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",          # PR-AUC уместнее accuracy/AUC при дисбалансе
        "auto_class_weights": "Balanced",  # аналог class_weight='balanced'
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "verbose": False,
        "task_type": "GPU",
        "devices": "0",
    }
    if catboost_params:
        default_params.update(catboost_params)

    quarters_sorted = sorted(df[quarter_col].unique())
    if len(quarters_sorted) <= n_min_train_quarters:
        raise ValueError("Недостаточно кварталов для walk-forward валидации")

    mlflow.set_experiment(mlflow_experiment_name)

    oof_probs, oof_true, oof_quarters = [], [], []

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params(default_params)
        mlflow.log_param("calibration_method", calibration_method)
        mlflow.log_param("threshold_beta", threshold_beta)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_quarters_total", len(quarters_sorted))

        # --- Walk-forward цикл по валидационным кварталам ---
        for i, val_quarter in enumerate(quarters_sorted[n_min_train_quarters:], start=n_min_train_quarters):
            train_mask = df[quarter_col].isin(quarters_sorted[:i])
            val_mask = df[quarter_col] == val_quarter

            X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
            X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]

            if y_val.nunique() < 2:
                # в этом квартале нет обоих классов — пропускаем, метрики будут не информативны
                continue

            fold_model = CatBoostClassifier(**default_params)
            fold_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
            )

            fold_probs = fold_model.predict_proba(X_val)[:, 1]

            oof_probs.extend(fold_probs)
            oof_true.extend(y_val.values)
            oof_quarters.extend([str(val_quarter)] * len(y_val))

            # метрики по фолду — на сырых (некалиброванных) вероятностях, порог 0.5 как ориентир
            fold_metrics = compute_classification_metrics(y_val.values, fold_probs, threshold=0.5)
            with mlflow.start_run(run_name=f"fold_{val_quarter}", nested=True):
                mlflow.log_param("val_quarter", str(val_quarter))
                mlflow.log_param("train_quarters", len(quarters_sorted[:i]))
                mlflow.log_param("n_train", len(X_train))
                mlflow.log_param("n_val", len(X_val))
                mlflow.log_metrics({f"fold_{k}": v for k, v in fold_metrics.items() if isinstance(v, (int, float))})
                if log_models_per_fold:
                    mlflow.catboost.log_model(fold_model, artifact_path="model")

        oof_probs = np.array(oof_probs)
        oof_true = np.array(oof_true)

        # --- Калибровка на полном OOF-пуле ---
        calibrator, calibrate_fn = calibrate_probabilities(oof_probs, oof_true, method=calibration_method)
        oof_probs_calibrated = calibrate_fn(oof_probs)

        # --- Подбор порога на калиброванных OOF-вероятностях ---
        threshold, thr_precision, thr_recall = select_best_threshold(
            oof_true, oof_probs_calibrated, beta=threshold_beta, min_recall=min_recall
        )

        # --- Метрики OOF: до и после калибровки ---
        metrics_raw = compute_classification_metrics(oof_true, oof_probs, threshold=0.5)
        metrics_calibrated = compute_classification_metrics(oof_true, oof_probs_calibrated, threshold=threshold)

        mlflow.log_metrics({f"oof_raw_{k}": v for k, v in metrics_raw.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"oof_calibrated_{k}": v for k, v in metrics_calibrated.items() if isinstance(v, (int, float))})
        mlflow.log_metric("selected_threshold", threshold)

        # --- Calibration curve (для проверки качества калибровки) ---
        prob_true_raw, prob_pred_raw = calibration_curve(oof_true, oof_probs, n_bins=10, strategy="quantile")
        prob_true_cal, prob_pred_cal = calibration_curve(oof_true, oof_probs_calibrated, n_bins=10, strategy="quantile")

        calib_df = pd.DataFrame({
            "bin_pred_raw": prob_pred_raw,
            "bin_true_raw": prob_true_raw,
        })
        calib_df_cal = pd.DataFrame({
            "bin_pred_calibrated": prob_pred_cal,
            "bin_true_calibrated": prob_true_cal,
        })
        mlflow.log_table(calib_df, artifact_file="calibration_curve_raw.json")
        mlflow.log_table(calib_df_cal, artifact_file="calibration_curve_calibrated.json")

        # --- Таблица всех OOF-предсказаний (для дальнейшего анализа/построения графиков) ---
        oof_table = pd.DataFrame({
            "quarter": oof_quarters,
            "y_true": oof_true,
            "p_raw": oof_probs,
            "p_calibrated": oof_probs_calibrated,
        })
        mlflow.log_table(oof_table, artifact_file="oof_predictions.json")

        # --- Финальная модель — обучаем на всех данных ---
        final_model = CatBoostClassifier(**default_params)
        final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)

        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        # фиче-импортансы — полезно смотреть отдельно
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": final_model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
        mlflow.log_table(importances, artifact_file="feature_importances.json")

        print(f"OOF PR-AUC (calibrated): {metrics_calibrated['pr_auc']:.4f}")
        print(f"OOF Brier (raw -> calibrated): {metrics_raw['brier_score']:.4f} -> {metrics_calibrated['brier_score']:.4f}")
        print(f"Selected threshold: {threshold:.3f} (precision={thr_precision:.3f}, recall={thr_recall:.3f})")
        print(f"MLflow run_id: {parent_run.info.run_id}")

    return {
        "final_model": final_model,
        "calibrator": calibrator,
        "calibrate_fn": calibrate_fn,
        "threshold": threshold,
        "oof_table": oof_table,
        "metrics_raw": metrics_raw,
        "metrics_calibrated": metrics_calibrated,
        "feature_importances": importances,
        "run_id": parent_run.info.run_id,
    }


# --------------------------------------------------------------------------
# Пример использования
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # df = pd.read_parquet("your_data.parquet")
    # result = train_catboost_binary_walkforward(
    #     df=df,
    #     feature_cols=[c for c in df.columns if c not in ("target", "quarter")],
    #     target_col="target",
    #     quarter_col="quarter",
    #     mlflow_experiment_name="forex_pinbar_spike",
    #     run_name="pinbar_v1",
    #     calibration_method="isotonic",
    #     threshold_beta=2.0,   # recall важнее precision для алертов
    # )
    #
    # # Инференс на новых данных:
    # raw_p = result["final_model"].predict_proba(new_data[feature_cols])[:, 1]
    # calibrated_p = result["calibrate_fn"](raw_p)
    # is_alert = calibrated_p >= result["threshold"]
    pass
