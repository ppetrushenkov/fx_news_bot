import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    # classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    # PrecisionRecallDisplay,
    precision_recall_curve,
    auc,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support
)

from tqdm import tqdm
from dataclasses import dataclass
import seaborn as sns


@dataclass
class BestThreshold:
    threshold: float
    precision: float
    recall: float


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


# ============================================================================
# Общая логика фолдов
# ============================================================================

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

def select_best_threshold(precisions, recalls, thresholds, beta=2.0, min_recall=None):
    """Returns best threshold, precision and recall"""
    precisions, recalls = precisions[:-1], recalls[:-1]
    if min_recall is not None:
        mask = recalls >= min_recall
        best_idx = np.argmax(recalls) if mask.sum() == 0 else np.argmax(np.where(mask, precisions, -1))
    else:
        f_beta = (1 + beta**2) * precisions * recalls / (beta**2 * precisions + recalls + 1e-9)
        best_idx = np.argmax(f_beta)
    
    return BestThreshold(
        threshold=float(thresholds[best_idx]), 
        precision=float(precisions[best_idx]), 
        recall=float(recalls[best_idx])
    )


def compute_binary_metrics(y_true, y_prob, name: str, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "name": name,
        "threshold": threshold,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "f0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "f3": fbeta_score(y_true, y_pred, beta=3.0, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "base_rate": float(np.mean(y_true)),
    }


def train_catboost_binary(
    df,
    time_col,
    feature_cols,
    target_col,
    mlflow_experiment_name,
    run_name="catboost_binary",
    catboost_params=None,
    cat_features=None,
):
    df = df.sort_values(by=time_col, ascending=True)

    default_params = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "auto_class_weights": "Balanced",
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

    mlflow.set_experiment(mlflow_experiment_name)

    # Здесь будем копить предсказания и реальные таргеты СО ВСЕХ фолдов для финального подбора
    all_y_true = []
    all_y_probs = []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(default_params)

        quarter_cv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=4 * 6 * 2,
            val_quarters=1,
            step_quarters=3,
        )
        splits = list(quarter_cv.split(df))

        pbar = tqdm(
            enumerate(splits, 1),
            total=len(splits),
            desc="Processing splits",
            leave=False,
        )
        for fold, (train_idx, val_idx) in pbar:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]

            model = CatBoostClassifier(**default_params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
            )
            y_probs = model.predict_proba(X_val)[:, 1]

            # Сохраняем для финального анализа
            all_y_true.extend(y_val.values)
            all_y_probs.extend(y_probs)

            # --- Логируем базовые метрики фолда по порогу 0.5 ---
            fold_metrics = compute_binary_metrics(
                y_val.values, y_probs, name="default_0.5", threshold=0.5
            )

            for metric_key, metric_value in fold_metrics.items():
                if metric_key == "name":
                    continue
                # Имена в UI будут вида: fold_f1, fold_precision, fold_pr_auc
                mlflow.log_metric(f"fold_{metric_key}", metric_value, step=fold)

        # --- КОНЕЦ ЦИКЛА: Финальный подбор порогов по всей OOF (Out-Of-Fold) выборке ---
        all_y_true = np.array(all_y_true)
        all_y_probs = np.array(all_y_probs)

        # Строим общую PR-кривую
        precisions, recalls, thresholds = precision_recall_curve(
            all_y_true, all_y_probs
        )
        final_pr_auc = auc(recalls, precisions)
        mlflow.log_metric("oof_pr_auc", final_pr_auc)

        # Сохраняем итоговый график PR-кривой
        fig, ax = plt.subplots()
        ax.plot(
            recalls, precisions, label=f"OOF PR AUC = {final_pr_auc:.4f}", color="red"
        )
        ax.set_title("Overall OOF PR Curve")
        ax.legend()
        mlflow.log_figure(fig, artifact_file="plots/overall_pr_curve.png")
        plt.close(fig)

        # Подбираем лучшие пороги под разные беты по всей истории
        th_f05 = select_best_threshold(
            precisions, recalls, thresholds, beta=0.5
        )
        th_f1 = select_best_threshold(precisions, recalls, thresholds, beta=1.0)
        th_f2 = select_best_threshold(precisions, recalls, thresholds, beta=2.0)
        th_f3 = select_best_threshold(precisions, recalls, thresholds, beta=3.0)

        # Считаем чистые метрики для каждого оптимального порога
        m_f05 = compute_binary_metrics(
            all_y_true, all_y_probs, threshold=th_f05.threshold, name="best_f0.5"
        )
        m_f1 = compute_binary_metrics(
            all_y_true, all_y_probs, threshold=th_f1.threshold, name="best_f1"
        )
        m_f2 = compute_binary_metrics(
            all_y_true, all_y_probs, threshold=th_f2.threshold, name="best_f2"
        )
        m_f3 = compute_binary_metrics(
            all_y_true, all_y_probs, threshold=th_f3.threshold, name="best_f3"
        )

        # Формируем красивую финальную таблицу порогов
        summary_metrics_df = pd.concat(
            [
                pd.DataFrame([m_f05]),
                pd.DataFrame([m_f1]),
                pd.DataFrame([m_f2]),
                pd.DataFrame([m_f3]),
            ],
            axis=0,
        ).reset_index(drop=True)

        # Отправляем таблицу в артефакты
        mlflow.log_table(
            data=summary_metrics_df, artifact_file="tables/best_thresholds_summary.json"
        )

        # Превращаем датафрейм в HTML-строку со стилями
        html_table = summary_metrics_df.to_html(index=False, classes="table table-striped")

        # Оборачиваем в базовый HTML-шаблон, чтобы шрифт был приятным
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            {html_table}
        </body>
        </html>
        """

        # Пишем во временный файл и логируем как артефакт
        with open("best_thresholds_summary.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        mlflow.log_artifact("best_thresholds_summary.html", artifact_path="tables")

        # Логируем ключевые агрегаты в UI для сравнения экспериментов
        mlflow.log_metric("best_threshold_f0.5", th_f05.threshold)
        mlflow.log_metric("max_precision_at_f0.5", th_f05.precision)

        # Обучение финальной модели на всем датасете
        final_model = CatBoostClassifier(**default_params)
        final_model.fit(
            df[feature_cols],
            df[target_col],
            cat_features=cat_features,
            verbose=False,
        )
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        print(f"MLflow run_id: {run.info.run_id}")

    return {
        "final_model": final_model,
        "run_id": run.info.run_id,
    }

# ============================================================================
# 2. MULTICLASS CLASSIFICATION
# ============================================================================

def compute_multiclass_metrics(y_true, y_pred, y_prob, name: str, labels=None):
    """Возвращает словарь основных метрик для мультиклассовой классификации"""
    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_prob, labels=labels)
    
    # Считаем macro, micro и weighted усреднения
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "name": name,
        "accuracy": acc,
        "log_loss": loss,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
    }

def log_confusion_matrix(y_true, y_pred, labels, artifact_file="plots/confusion_matrix.png"):
    """Генерирует и логирует тепловую карту матрицы ошибок в MLflow"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Overall OOF Confusion Matrix')
    plt.tight_layout()
    mlflow.log_figure(fig, artifact_file=artifact_file)
    plt.close(fig)


def train_catboost_multiclass(
    df,
    time_col,
    feature_cols,
    target_col,
    mlflow_experiment_name,
    run_name="catboost_multiclass",
    catboost_params=None,
    cat_features=None,
):
    df = df.sort_values(by=time_col, ascending=True)
    
    # Определяем уникальные классы в таргете
    unique_labels = sorted(df[target_col].unique())
    num_classes = len(unique_labels)

    default_params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "auto_class_weights": "Balanced",
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

    mlflow.set_experiment(mlflow_experiment_name)

    # Накопители для Out-Of-Fold предсказаний
    all_y_true = []
    all_y_preds = []
    all_y_probs = [] # Будет хранить матрицы вероятностей (N, num_classes)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(default_params)

        # Используем ваш кросс-валидатор на основе кварталов
        quarter_cv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=4 * 6 * 2,
            val_quarters=1,
            step_quarters=3,
        )
        splits = list(quarter_cv.split(df))

        pbar = tqdm(
            enumerate(splits, 1),
            total=len(splits),
            desc="Processing multiclass splits",
            leave=False,
        )
        
        for fold, (train_idx, val_idx) in pbar:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]

            model = CatBoostClassifier(**default_params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
            )
            
            # Получаем вероятности для всех классов и жесткие предсказания классов
            y_probs = model.predict_proba(X_val)
            y_preds = model.predict(X_val).flatten()

            # Сохраняем для финального OOF-анализа
            all_y_true.extend(y_val.values)
            all_y_preds.extend(y_preds)
            all_y_probs.extend(y_probs)

            # --- Логируем базовые метрики фолда ---
            fold_metrics = compute_multiclass_metrics(
                y_val.values, y_preds, y_probs, name=f"fold_{fold}", labels=unique_labels
            )

            for metric_key, metric_value in fold_metrics.items():
                if metric_key == "name":
                    continue
                mlflow.log_metric(f"fold_{metric_key}", metric_value, step=fold)

        # --- КОНЕЦ ЦИКЛА: Финальный анализ по всей OOF выборке ---
        all_y_true = np.array(all_y_true)
        all_y_preds = np.array(all_y_preds)
        all_y_probs = np.array(all_y_probs)

        # Расчет итоговых метрик по всей OOF-выборке
        oof_metrics = compute_multiclass_metrics(
            all_y_true, all_y_preds, all_y_probs, name="overall_oof", labels=unique_labels
        )
        
        # Логируем ключевые агрегаты в UI
        mlflow.log_metric("oof_accuracy", oof_metrics["accuracy"])
        mlflow.log_metric("oof_f1_macro", oof_metrics["f1_macro"])
        mlflow.log_metric("oof_f1_weighted", oof_metrics["f1_weighted"])

        # Визуализируем и логируем матрицу ошибок
        log_confusion_matrix(all_y_true, all_y_preds, labels=unique_labels, artifact_file="plots/overall_confusion_matrix.png")

        # Детальный разбор метрик в разрезе каждого класса отдельно
        p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
            all_y_true, all_y_preds, labels=unique_labels, zero_division=0
        )
        
        per_class_records = []
        for i, class_label in enumerate(unique_labels):
            per_class_records.append({
                "class_label": class_label,
                "precision": p_class[i],
                "recall": r_class[i],
                "f1_score": f1_class[i],
                "support": int(support_class[i])
            })
            
        summary_metrics_df = pd.DataFrame(per_class_records)

        # Отправляем JSON-таблицу в артефакты
        mlflow.log_table(
            data=summary_metrics_df, artifact_file="tables/per_class_metrics_summary.json"
        )

        # Обучение финальной модели на всем доступном датасете
        final_model = CatBoostClassifier(**default_params)
        final_model.fit(
            df[feature_cols],
            df[target_col],
            cat_features=cat_features,
            verbose=False,
        )
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        print(f"MLflow run_id: {run.info.run_id}")

    return {
        "final_model": final_model,
        "run_id": run.info.run_id,
    }


# ============================================================================
# 3. MULTI-QUANTILE REGRESSION
# ============================================================================

def compute_quantile_metrics(y_true, preds_dict, quantiles):
    """
    Рассчитывает специфичные метрики для оценки квантилей и интервалов.
    """
    metrics = {}
    
    # 1. Считаем Pinball Loss для каждого квантиля отдельно
    for q in quantiles:
        error = y_true - preds_dict[q]
        pinball = np.mean(np.maximum(q * error, (q - 1) * error))
        metrics[f"pinball_q{q}"] = pinball
        
        # Для медианы (0.5) также полезно посчитать классический MAE
        if q == 0.5:
            metrics["mae_q0.5"] = np.mean(np.abs(error))

    # 2. Если в наборе есть квантили 0.1 и 0.9, оцениваем 80%-й интервал
    if 0.1 in quantiles and 0.9 in quantiles:
        q_low = preds_dict[0.1]
        q_high = preds_dict[0.9]
        
        # Покрытие (сколько фактов попало внутрь прогнозируемого ренджа)
        coverage = np.mean((y_true >= q_low) & (y_true <= q_high))
        metrics["interval_coverage_80pct"] = coverage
        
        # Средняя ширина интервала (насколько модель "уверена" в прогнозе)
        mean_width = np.mean(q_high - q_low)
        metrics["interval_mean_width"] = mean_width
        
        # Winkler Score (метрика качества интервальных прогнозов)
        # Штрафует за широкие интервалы + жестко штрафует, если факт вылетел за границы
        alpha = 0.2  # для 80% интервала (1 - 0.8)
        winkler_scores = []
        for i in range(len(y_true)):
            width = q_high[i] - q_low[i]
            if y_true[i] < q_low[i]:
                score = width + (2 / alpha) * (q_low[i] - y_true[i])
            elif y_true[i] > q_high[i]:
                score = width + (2 / alpha) * (y_true[i] - q_high[i])
            else:
                score = width
            winkler_scores.append(score)
        metrics["winkler_score"] = np.mean(winkler_scores)

    return metrics


def train_catboost_multiquantile(
    df,
    time_col,  # переименовали quarter_col в time_col для соответствия сигнатуре
    feature_cols,
    target_col,
    mlflow_experiment_name,
    quantiles=(0.1, 0.5, 0.9),
    run_name="catboost_multiquantile",
    catboost_params=None,
    cat_features=None,
):
    df = df.sort_values(by=time_col, ascending=True)
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

    mlflow.set_experiment(mlflow_experiment_name)
    
    oof_true, oof_quarters = [], []
    oof_preds = {q: [] for q in quantiles}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(default_params)
        mlflow.log_param("quantiles", alpha_str)

        # Твой кросс-валидатор SlidingQuarterBlockCV
        quarter_cv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=4 * 6 * 2,
            val_quarters=1,
            step_quarters=3,
        )
        splits = list(quarter_cv.split(df))

        # Настройка tqdm точно такая же, как в бинарной классификации
        pbar = tqdm(
            enumerate(splits, 1),
            total=len(splits),
            desc="Processing quantile splits",
            leave=False,
        )

        for fold, (train_idx, val_idx) in pbar:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]

            model = CatBoostRegressor(**default_params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
            )
            
            pred_raw = model.predict(X_val)
            fold_preds = {q: pred_raw[:, idx] for idx, q in enumerate(quantiles)}

            for q in quantiles:
                oof_preds[q].extend(fold_preds[q])
            oof_true.extend(y_val.values)
            # Сохраняем информацию о времени/квартале для OOF таблицы
            oof_quarters.extend(val_df[time_col].astype(str).values)

            # Логируем метрики фолда
            fold_metrics = compute_quantile_metrics(y_val.values, fold_preds, quantiles)
            for metric_key, metric_value in fold_metrics.items():
                mlflow.log_metric(f"fold_{metric_key}", metric_value, step=fold)

        # --- КОНЕЦ ЦИКЛА: Финальный расчет OOF ---
        oof_true = np.array(oof_true)
        for q in quantiles:
            oof_preds[q] = np.array(oof_preds[q])

        oof_metrics = compute_quantile_metrics(oof_true, oof_preds, quantiles)
        
        for k, v in oof_metrics.items():
            mlflow.log_metric(f"oof_{k}", v)

        oof_table = pd.DataFrame({
            "quarter": oof_quarters,
            "y_true": oof_true,
            **{f"pred_q{q}": oof_preds[q] for q in quantiles}
        })
        mlflow.log_table(oof_table, artifact_file="tables/oof_predictions.json")

        # HTML Отчет
        quantile_summary_records = []
        for q in quantiles:
            quantile_summary_records.append({
                "Type": f"Quantile {q}",
                "Metric / Property": "Pinball Loss",
                "Value": round(oof_metrics[f"pinball_q{q}"], 5),
                "Description": "Точность оценки конкретного квантиля"
            })
        
        if "interval_coverage_80pct" in oof_metrics:
            quantile_summary_records.extend([
                {"Type": "Interval 10%-90%", "Metric / Property": "Coverage (Target: 0.80)", "Value": round(oof_metrics["interval_coverage_80pct"], 4), "Description": "Идеал = 0.80."},
                {"Type": "Interval 10%-90%", "Metric / Property": "Mean Width", "Value": round(oof_metrics["interval_mean_width"], 4), "Description": "Средняя ширина ценового диапазона."},
                {"Type": "Interval 10%-90%", "Metric / Property": "Winkler Score", "Value": round(oof_metrics["winkler_score"], 4), "Description": "Штраф за ширину и пробои границ."}
            ])
            
        summary_df = pd.DataFrame(quantile_summary_records)
        html_table = summary_df.to_html(index=False, classes="table table-striped")
        html_content = f"<html><head><style>body {{ font-family: Arial, sans-serif; margin: 20px; }} table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }} th {{ background-color: #f2f2f2; font-weight: bold; }} tr:nth-child(even) {{ background-color: #f9f9f9; }}</style></head><body><h3>Overall Out-of-Fold MultiQuantile Analysis</h3>{html_table}</body></html>"

        with open("quantile_metrics_summary.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        mlflow.log_artifact("quantile_metrics_summary.html", artifact_path="tables")

        final_model = CatBoostRegressor(**default_params)
        final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        if "interval_coverage_80pct" in oof_metrics:
            print(f"OOF interval coverage 80%: {oof_metrics['interval_coverage_80pct']:.3f}")
        print(f"MLflow run_id: {run.info.run_id}")

    return {
        "final_model": final_model,
        "quantiles": quantiles,
        "oof_table": oof_table,
        "run_id": run.info.run_id
    }


# def pinball_loss(y_true, y_pred, q):
#     diff = y_true - y_pred
#     return float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))


# def compute_quantile_metrics(y_true, preds_dict, quantiles):
#     metrics = {}
#     for q in quantiles:
#         yp = preds_dict[q]
#         metrics[f"pinball_q{q}"] = pinball_loss(y_true, yp, q)
#         metrics[f"mae_q{q}"] = mean_absolute_error(y_true, yp)
#         metrics[f"empirical_coverage_below_q{q}"] = float(np.mean(y_true <= yp))
#     if 0.1 in quantiles and 0.9 in quantiles:
#         metrics["interval_coverage_80pct"] = float(np.mean((y_true >= preds_dict[0.1]) & (y_true <= preds_dict[0.9])))
#         width = preds_dict[0.9] - preds_dict[0.1]
#         metrics["mean_interval_width"] = float(np.mean(width))
#         metrics["pct_non_monotonic"] = float(np.mean(width < 0))
#     if 0.5 in quantiles:
#         metrics["mae_median"] = mean_absolute_error(y_true, preds_dict[0.5])
#         metrics["rmse_median"] = float(np.sqrt(mean_squared_error(y_true, preds_dict[0.5])))
#     return metrics


# def train_catboost_multiquantile(
#     df, feature_cols, target_col, quarter_col,
#     mlflow_experiment_name, quantiles=(0.1, 0.5, 0.9),
#     run_name="catboost_multiquantile", catboost_params=None, cat_features=None, n_val_quarters=6,
# ):
#     quantiles = list(quantiles)
#     alpha_str = ",".join(str(q) for q in quantiles)
#     default_params = {
#         "loss_function": f"MultiQuantile:alpha={alpha_str}",
#         "iterations": 1000, "learning_rate": 0.03, "depth": 6,
#         "random_seed": 42, "early_stopping_rounds": 50, "verbose": False,
#     }
#     if catboost_params:
#         default_params.update(catboost_params)

#     quarters_sorted = sorted(df[quarter_col].unique())
#     folds = build_walkforward_quarter_folds(quarters_sorted, n_val_quarters=n_val_quarters)

#     mlflow.set_experiment(mlflow_experiment_name)
#     oof_true, oof_quarters = [], []
#     oof_preds = {q: [] for q in quantiles}

#     with mlflow.start_run(run_name=run_name) as run:
#         mlflow.log_params(default_params)
#         mlflow.log_param("quantiles", alpha_str)
#         mlflow.log_param("fold_order", ",".join(str(q) for _, q, _ in folds))

#         for step_idx, val_quarter, train_quarters in folds:
#             train_mask = df[quarter_col].isin(train_quarters)
#             val_mask = df[quarter_col] == val_quarter
#             X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
#             X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]
#             if len(X_val) == 0:
#                 continue

#             model = CatBoostRegressor(**default_params)
#             model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)
#             pred_raw = model.predict(X_val)
#             fold_preds = {q: pred_raw[:, idx] for idx, q in enumerate(quantiles)}

#             for q in quantiles:
#                 oof_preds[q].extend(fold_preds[q])
#             oof_true.extend(y_val.values)
#             oof_quarters.extend([str(val_quarter)] * len(y_val))

#             fold_metrics = compute_quantile_metrics(y_val.values, fold_preds, quantiles)
#             _log_step_metrics(fold_metrics, step=step_idx, prefix="fold_")

#         oof_true = np.array(oof_true)
#         for q in quantiles:
#             oof_preds[q] = np.array(oof_preds[q])

#         oof_metrics = compute_quantile_metrics(oof_true, oof_preds, quantiles)
#         mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})

#         oof_table = pd.DataFrame({"quarter": oof_quarters, "y_true": oof_true, **{f"pred_q{q}": oof_preds[q] for q in quantiles}})
#         mlflow.log_table(oof_table, artifact_file="oof_predictions.json")

#         final_model = CatBoostRegressor(**default_params)
#         final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
#         mlflow.catboost.log_model(final_model, artifact_path="final_model")

#         print(f"OOF interval coverage 80%: {oof_metrics.get('interval_coverage_80pct', float('nan')):.3f}")
#         print(f"MLflow run_id: {run.info.run_id}")

#     return {"final_model": final_model, "quantiles": quantiles, "oof_table": oof_table, "run_id": run.info.run_id}


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
    df,
    time_col,  # переименовали quarter_col в time_col для унификации
    feature_cols,
    target_col,
    mlflow_experiment_name,
    run_name="catboost_ordinal_reversals",
    catboost_params=None,
    cat_features=None,
    loss_function="RMSE",
):
    df = df.sort_values(by=time_col, ascending=True)
    max_class = int(df[target_col].max())

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

    mlflow.set_experiment(mlflow_experiment_name)
    
    oof_true, oof_pred, oof_quarters = [], [], []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(default_params)
        mlflow.log_param("max_class", max_class)

        # Твой стандартный кросс-валидатор на кварталах
        quarter_cv = SlidingQuarterBlockCV(
            time_col=time_col,
            train_quarters=4 * 6 * 2,
            val_quarters=1,
            step_quarters=3,
        )
        splits = list(quarter_cv.split(df))

        # Настройка tqdm точно так же, как в остальных функциях пайплайна
        pbar = tqdm(
            enumerate(splits, 1),
            total=len(splits),
            desc="Processing ordinal splits",
            leave=False,
        )

        for fold, (train_idx, val_idx) in pbar:
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]

            model = CatBoostRegressor(**default_params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                use_best_model=True,
            )
            
            pred = model.predict(X_val)

            oof_pred.extend(pred)
            oof_true.extend(y_val.values)
            oof_quarters.extend(val_df[time_col].astype(str).values)

            # Логируем метрики фолда на шаге step=fold
            fold_metrics = compute_ordinal_metrics(y_val.values, pred, max_class)
            for metric_key, metric_value in fold_metrics.items():
                mlflow.log_metric(f"fold_{metric_key}", metric_value, step=fold)

        # --- КОНЕЦ ЦИКЛА: Финальный расчет OOF ---
        oof_true = np.array(oof_true)
        oof_pred = np.array(oof_pred)
        
        oof_metrics = compute_ordinal_metrics(oof_true, oof_pred, max_class)
        
        # Логируем ключевые агрегированные OOF-метрики в интерфейс MLflow
        for k, v in oof_metrics.items():
            mlflow.log_metric(f"oof_{k}", v)

        # Сохраняем OOF таблицу предсказаний (с непрерывными и округленными значениями)
        oof_table = pd.DataFrame({
            "quarter": oof_quarters,
            "y_true": oof_true,
            "y_pred_continuous": oof_pred,
            "y_pred_rounded": np.clip(np.round(oof_pred), 0, max_class).astype(int),
        })
        mlflow.log_table(oof_table, artifact_file="tables/oof_predictions.json")

        # --- Формируем красивую HTML-таблицу с результатами анализа ---
        ordinal_summary_records = [
            {
                "Metric": "QWK (Quadratic Weighted Kappa)",
                "Value": round(oof_metrics["qwk"], 4),
                "Description": "Качество ранжирования. Жестко штрафует за сильные промахи."
            },
            {
                "Metric": "MAE (Rounded Predictions)",
                "Value": round(oof_metrics["mae_rounded"], 4),
                "Description": "Средняя абсолютная ошибка в количестве разворотов после округления."
            },
            {
                "Metric": "MAE (Continuous Predictions)",
                "Value": round(oof_metrics["mae_continuous"], 4),
                "Description": "Чистая средняя ошибка модели до округления до ближайшего класса."
            },
            {
                "Metric": "RMSE (Rounded Predictions)",
                "Value": round(oof_metrics["rmse_rounded"], 4),
                "Description": "Среднеквадратичная ошибка (чувствительна к выбросам и крупным промахам)."
            },
            {
                "Metric": "Exact Match Accuracy",
                "Value": round(oof_metrics["exact_match_accuracy"], 4),
                "Description": "Процент случаев, когда модель угадала точное число разворотов."
            },
            {
                "Metric": "Within-1 Accuracy",
                "Value": round(oof_metrics["within_1_accuracy"], 4),
                "Description": "Доля прогнозов с ошибкой не более чем на ±1 разворот."
            }
        ]
            
        summary_df = pd.DataFrame(ordinal_summary_records)
        
        # Оборачиваем в HTML-шаблон для рендеринга в UI
        html_table = summary_df.to_html(index=False, classes="table table-striped")
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h3>Overall Out-of-Fold Ordinal Regression Analysis</h3>
            <p>Задача: прогнозирование количества разворотов рынка после новостного события.</p>
            {html_table}
        </body>
        </html>
        """

        with open("ordinal_metrics_summary.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        mlflow.log_artifact("ordinal_metrics_summary.html", artifact_path="tables")

        # Обучение финальной модели на всей доступной истории
        final_model = CatBoostRegressor(**default_params)
        final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
        mlflow.catboost.log_model(final_model, artifact_path="final_model")

        print(f"OOF QWK: {oof_metrics['qwk']:.4f}, MAE rounded: {oof_metrics['mae_rounded']:.4f}")
        print(f"MLflow run_id: {run.info.run_id}")

    return {
        "final_model": final_model,
        "max_class": max_class,
        "oof_table": oof_table,
        "run_id": run.info.run_id
    }


# def train_catboost_ordinal(
#     df, feature_cols, target_col, quarter_col,
#     mlflow_experiment_name, run_name="catboost_ordinal_reversals",
#     catboost_params=None, cat_features=None, loss_function="RMSE", n_val_quarters=6,
# ):
#     default_params = {
#         "loss_function": loss_function, "iterations": 1000,
#         "learning_rate": 0.03, "depth": 6, "random_seed": 42,
#         "early_stopping_rounds": 50, "verbose": False,
#     }
#     if catboost_params:
#         default_params.update(catboost_params)

#     quarters_sorted = sorted(df[quarter_col].unique())
#     folds = build_walkforward_quarter_folds(quarters_sorted, n_val_quarters=n_val_quarters)
#     max_class = int(df[target_col].max())

#     mlflow.set_experiment(mlflow_experiment_name)
#     oof_true, oof_pred, oof_quarters = [], [], []

#     with mlflow.start_run(run_name=run_name) as run:
#         mlflow.log_params(default_params)
#         mlflow.log_param("max_class", max_class)
#         mlflow.log_param("fold_order", ",".join(str(q) for _, q, _ in folds))

#         for step_idx, val_quarter, train_quarters in folds:
#             train_mask = df[quarter_col].isin(train_quarters)
#             val_mask = df[quarter_col] == val_quarter
#             X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
#             X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]
#             if len(X_val) == 0:
#                 continue

#             model = CatBoostRegressor(**default_params)
#             model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)
#             pred = model.predict(X_val)

#             oof_pred.extend(pred)
#             oof_true.extend(y_val.values)
#             oof_quarters.extend([str(val_quarter)] * len(y_val))

#             fold_metrics = compute_ordinal_metrics(y_val.values, pred, max_class)
#             _log_step_metrics(fold_metrics, step=step_idx, prefix="fold_")

#         oof_true, oof_pred = np.array(oof_true), np.array(oof_pred)
#         oof_metrics = compute_ordinal_metrics(oof_true, oof_pred, max_class)
#         mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})

#         oof_table = pd.DataFrame({
#             "quarter": oof_quarters, "y_true": oof_true, "y_pred_continuous": oof_pred,
#             "y_pred_rounded": np.clip(np.round(oof_pred), 0, max_class).astype(int),
#         })
#         mlflow.log_table(oof_table, artifact_file="oof_predictions.json")

#         final_model = CatBoostRegressor(**default_params)
#         final_model.fit(df[feature_cols], df[target_col], cat_features=cat_features, verbose=False)
#         mlflow.catboost.log_model(final_model, artifact_path="final_model")

#         print(f"OOF QWK: {oof_metrics['qwk']:.4f}, MAE rounded: {oof_metrics['mae_rounded']:.4f}")
#         print(f"MLflow run_id: {run.info.run_id}")

#     return {"final_model": final_model, "max_class": max_class, "oof_table": oof_table, "run_id": run.info.run_id}
