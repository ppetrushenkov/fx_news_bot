from typing import Dict, Literal

# from config import Config
from db.models import Predictions

from datetime import datetime, timedelta

from catboost import CatBoostClassifier, CatBoostRegressor

import pandas as pd
import numpy as np
import os
import joblib

from tqdm import tqdm
from pathlib import Path

from ml.preprocessing import EventPreprocessingTransformer, PricePreprocessingTransformer


def load_model(model_path: Path, model_type: str):
    """
    Load CatBoost model from a file path.
    """
    if model_type == 'regression':
        model = CatBoostRegressor()
    elif model_type == 'classification':
        model = CatBoostClassifier()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    model.load_model(str(model_path))
    return model


class FxRangePredictor:
    def __init__(self, period: int = 21, timeframe: Literal['hourly', 'daily'] = 'hourly'):
        """Initialize the predictor with a trained models."""
        self.period = period

        CATEGORICAL_FEATURES = ['main_event', 'instrument', 'base_currency', 'quote_currency', 'prev_hour_main_event',
                                'next_hour_main_event']
        self.cat_features = CATEGORICAL_FEATURES

        hourly_model_path = Path('models/ml/hourly')
        daily_model_path = Path('models/ml/daily')

        if timeframe == 'hourly':
            # Hourly models
            # ------------------- Range models ----------------------
            range_model_path = hourly_model_path / 'TotalRangePrediction'
            trm_model_name = "range_prediction_model_"
            self.trm_1h = load_model(range_model_path / (trm_model_name + '1h.cbm'), 'regression')
            self.trm_3h = load_model(range_model_path / (trm_model_name + '3h.cbm'), 'regression')
            self.trm_6h = load_model(range_model_path / (trm_model_name + '6h.cbm'), 'regression')
            self.trm_24h = load_model(range_model_path / (trm_model_name + '24h.cbm'), 'regression')

            # ------------------- Binary classification models -------------------
            self.chaos_pred_model = load_model(hourly_model_path / 'chaos_model.cb', 'classification')
            self.big_doji_model = load_model(hourly_model_path / 'big_doji_model.cb', 'classification')
            self.channel_expansion_model = load_model(hourly_model_path / 'double_extremum_breakout_model.cb',
                                                      'classification')
            self.sfp_model = load_model(hourly_model_path / 'sfp_model.cb', 'classification')

            # ------------------- Multiclassification models ---------------------
            self.regime_model_1day = load_model(hourly_model_path / 'trend_or_flat_model_1day.cb', 'classification')
            self.regime_model_2days = load_model(hourly_model_path / 'trend_or_flat_model_2days.cb', 'classification')

            # ------------------- Ordinal model --------------------
            self.dir_changes_model = load_model(hourly_model_path / 'direction_count_model.cb', 'regression')


        elif timeframe == 'daily':
            # Daily models
            self.impulse_bar_model_d = load_model(daily_model_path / 'impulse_bar_model_d.cbm', 'classification')
            self.bar_type_model_d = load_model(daily_model_path / 'bar_type_model_d.cbm', 'classification')

        # LOAD TRANSFORMERS FOR FEATURES
        path_to_transformers = Path('models/feature_transformers')
        self.event_transformer = joblib.load(path_to_transformers / 'event_transformer.pkl')
        self.price_transformer = joblib.load(path_to_transformers / 'price_transformer.pkl')

        print("[Predictor] Models loaded successfully.")

    def get_daily_predictions(self, data: pd.DataFrame, ml_threshs: dict = None) -> Dict:
        if data.empty:
            return {}

        names_v_models = {
            'Impulse_bar': self.impulse_bar_model_d,
            'Bar_type': self.bar_type_model_d,
        }

        total_models = len(names_v_models)
        print(f"[Predictor] Starting predictions for {total_models} models...")
        predictions = {'tickers': data['ticker'].tolist()}

        with tqdm(total=total_models, desc="[Predictor] Generating predictions") as pbar:
            for name, model in names_v_models.items():
                pbar.set_postfix(current_model=name)
                if name == "Impulse_bar":
                    probs = model.predict_proba(data)[:, 1]
                    preds = (probs > 0.5).astype(bool)
                elif name == "Bar_type":
                    preds = model.predict(data)
                    preds = [i[0] for i in preds]
                else:
                    preds = model.predict(data)
                predictions[name] = preds
                pbar.update(1)

        print("[Predictor] All predictions completed successfully!")
        return predictions

    def get_hourly_predictions(self, data: pd.DataFrame, ml_threshs: dict = None) -> dict:
        """
        The `get_hourly_predictions` function is designed to generate predictions using multiple machine learning models.
        It accepts a processed DataFrame containing featured events and prices as input and returns a dictionary 
        where the keys are model names (tasks e.g. is_breakout) and the values are arrays of predictions.

        ## Parameters

        - **data**: A pandas DataFrame containing the processed event and price data.

        ## Returns

        A dictionary where each key is a model names and each value is an array of predictions made by different models.

        ## Models

        - `"total_range_Nh"`: Predicts future total range within N hours.
        - `"Big Spike"`: Predicts whether a given event is a big doji pattern.
        - `"Extremum Breakout"`: Predicts if there is a breakout in the price in channel in both places.
        - `"Direction changes"`: Predicts the direction changes within 24 hours.
        - `"Chaos"`: Predicts chaotic behavior within 24 hours using a probability threshold of 0.3.
        - `"Regime 1 day / 2 days"`: Predicts Trend / Flat / None in 24 and 48 hours.
        """
        if data.empty:
            return {}

        names_v_models = {
            'total_range_1h': self.trm_1h,
            'total_range_3h': self.trm_3h,
            'total_range_6h': self.trm_6h,
            'total_range_24h': self.trm_24h,

            'Big Spike': self.big_doji_model,
            'Extremum Breakout': self.channel_expansion_model,
            'Direction Changes': self.dir_changes_model,
            'Chaos': self.chaos_pred_model,
            'SFP': self.sfp_model,

            'Regime in 1 day': self.regime_model_1day,
            'Regime in 2 days': self.regime_model_2days
        }
        print(names_v_models)

        total_models = len(names_v_models)

        print(f"[Predictor] Starting predictions for {total_models} models...")
        predictions = {'tickers': data['ticker'].tolist()}

        conservative, base, aggressive = ml_threshs['conservative'], ml_threshs['base'], ml_threshs['aggressive']

        with tqdm(total=total_models, desc="[Predictor] Generating predictions") as pbar:
            for model_name, model in names_v_models.items():
                pbar.set_postfix(current_model=model_name)

                if model_name in ['Chaos', 'SFP', 'Extremum Breakout', 'Big Spike']:
                    probs = model.predict_proba(data)[:, 1]
                    preds = {
                        'Conservative': ((probs > conservative.get(model_name)).astype(bool)),
                        'Base': ((probs > base.get(model_name)).astype(bool)),
                        'Aggressive': ((probs > aggressive.get(model_name)).astype(bool))
                    }
                elif model_name.startswith("Regime"):
                    preds = model.predict(data)
                    preds = [i[0] for i in preds]

                elif model_name.startswith('total_range'):
                    preds = model.predict(data)
                    preds = np.sort(preds, axis=1)

                elif model_name.startswith("Direction Changes"):
                    preds = model.predict(data)
                    preds = [np.round(i, 2) for i in preds]

                else:
                    preds = model.predict(data)

                predictions[model_name] = preds
                pbar.update(1)

        print("[Predictor] All predictions completed successfully!")
        return predictions

# TODO: Add methods for saving predictions to the database and any other utility functions as needed.
