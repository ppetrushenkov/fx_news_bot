from typing import Dict

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


class FxRangePredictor:
    def __init__(self, period: int = 21):
        """Initialize the predictor with a trained models."""
        self.period = period

        CATEGORICAL_FEATURES = ['main_event', 'instrument', 'base_currency', 'quote_currency', 'prev_hour_main_event', 'next_hour_main_event']
        self.cat_features = CATEGORICAL_FEATURES

        model_path = Path('models/ml/')
        
        # ------------------- Range models ----------------------
        range_model_path = model_path / 'TotalRangePrediction'
        trm_model_name = "range_prediction_model_"
        self.trm_1h = self.load_model(range_model_path / (trm_model_name + '1h.cbm'), 'regression')
        self.trm_3h = self.load_model(range_model_path / (trm_model_name + '3h.cbm'), 'regression')
        self.trm_6h = self.load_model(range_model_path / (trm_model_name + '6h.cbm'), 'regression')
        self.trm_24h = self.load_model(range_model_path / (trm_model_name + '24h.cbm'), 'regression')

        # ------------------- Binary classification models -------------------
        self.chaos_pred_model = self.load_model(model_path / 'chaos_model.cb', 'classification')
        self.big_doji_model = self.load_model(model_path / 'big_doji_model.cb', 'classification')
        self.channel_expansion_model = self.load_model(model_path / 'double_extremum_breakout_model.cb', 'classification')
        self.sfp_model = self.load_model(model_path / 'sfp_model.cb', 'classification')

        # ------------------- Multiclassification models ---------------------
        self.regime_model_1day = self.load_model(model_path / 'trend_or_flat_model_1day.cb', 'classification')
        self.regime_model_2days = self.load_model(model_path / 'trend_or_flat_model_2days.cb', 'classification')

        # ------------------- Ordinal model --------------------
        self.dir_changes_model = self.load_model(model_path / 'direction_count_model.cb', 'regression')

        # LOAD TRANSFORMERS FOR FEATURES
        path_to_transformers = Path('models/feature_transformers')
        self.event_transformer = joblib.load(path_to_transformers / 'event_transformer.pkl')
        self.price_transformer = joblib.load(path_to_transformers / 'price_transformer.pkl')

        print("[Predictor] Models loaded successfully.")

    
    def load_model(self, model_path: Path, model_type: str):
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

    
    def get_predictions(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        The `get_predictions` function is designed to generate predictions using multiple machine learning models. 
        It accepts a processed DataFrame containing featured events and prices as input and returns a dictionary 
        where the keys are model names (tasks e.g. is_breakout) and the values are arrays of predictions.

        ## Parameters

        - **data**: A pandas DataFrame containing the processed event and price data.

        ## Returns

        A dictionary where each key is a model names and each value is an array of predictions made by different models.

        ## Models

        - `"total_range_Nh"`: Predicts future total range within N hours.
        - `"is_big_doji"`: Predicts whether a given event is a big doji pattern.
        - `"is_breakout"`: Predicts if there is a breakout in the price.
        - `"dir_count_24h"`: Predicts the direction changes within 24 hours.
        - `"is_chaos_24h"`: Predicts chaotic behavior within 24 hours using a probability threshold of 0.3.
        - `"is_regime_24h"`: Predicts Trend / Flat / None.
        """
        if data.empty:
            return {}

        all_models = []
        model_names = [
            'total_range_1h', 'total_range_3h', 'total_range_6h', 'total_range_24h',
            'big_doji', 'expansion', 'dir_changes', 'chaos', 'sfp',
            'regime_1day', 'regime_2days'
            ]
        models = [
            self.trm_1h, self.trm_3h, self.trm_6h, self.trm_24h,
            self.big_doji_model, self.channel_expansion_model, 
            self.dir_changes_model, self.chaos_pred_model, self.sfp_model,
            self.regime_model_1day, self.regime_model_2days
            ]
        
        for model_name, model in zip(model_names, models):
            all_models.append((model_name, model))

        total_models = len(all_models)

        print(f"[Predictor] Starting predictions for {total_models} models...")
        predictions = {}
        predictions['tickers'] = data['ticker'].tolist()
        
        with tqdm(total=total_models, desc="[Predictor] Generating predictions") as pbar:
            for model_name, model in all_models:
                pbar.set_postfix(current_model=model_name)
                if model_name == "chaos":
                    probs = model.predict_proba(data)[:, 1]
                    preds = (probs > 0.3).astype(bool)
                if model_name.startswith("regime"):
                    preds = model.predict(data)
                    preds = [i[0] for i in preds]
                if model_name.startswith('total_range'):
                    preds = model.predict(data)
                    preds = np.sort(preds, axis=1)
                if model_name.startswith("dir_changes"):
                    preds = model.predict(data)
                    preds = [np.round(i, 2) for i in preds]
                else:
                    preds = model.predict(data)
                predictions[model_name] = preds
                pbar.update(1)
        
        print("[Predictor] All predictions completed successfully!")
        return predictions

# TODO: Add methods for saving predictions to the database and any other utility functions as needed.