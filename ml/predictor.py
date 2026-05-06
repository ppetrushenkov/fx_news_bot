from datetime import datetime, timedelta
from sys import prefix

from db.database import SessionLocal
from db.models import TodayEconomicNews, Prediction

from news_featuring import classify_news, classify_event_type, period_extraction, floor_or_ceil, build_hour_features
from price_featuring import calculate_chaos_features, add_features as add_price_features

from catboost import CatBoostClassifier, CatBoostRegressor

from config import Config

import pandas as pd
import numpy as np


class VolatilityPredictor:
    def __init__(self, 
                 volatility_model_path=None,
                 range_model_path=None,
                 chaos_model_path=None
                 ):
        """Initialize the predictor with a trained models."""
        if volatility_model_path is None:
            volatility_model_path = Config.VOLATILITY_MODEL_PATH
        if range_model_path is None:
            range_model_path = Config.RANGE_MODEL_PATH
        if chaos_model_path is None:
            chaos_model_path = Config.CHAOS_MODEL_PATH
        
        try:
            self.volatility_model = CatBoostClassifier()
            self.volatility_model.load_model(volatility_model_path)

            self.range_model = CatBoostRegressor()
            self.range_model.load_model(range_model_path)

            self.chaos_model = CatBoostClassifier()
            self.chaos_model.load_model(chaos_model_path)
        
        except Exception as e:
            ValueError('Can not load CatBoost models')
            print(e)

    def get_predictions(self, news: pd.DataFrame, prices: pd.DataFrame):
        """Main function. Takes news and prices, do preprocessing and return predictions for data"""
        data = self.preprocess_data_for_ml_predictions(news, prices)
        
        # In a real app, these would call actual model prediction methods.
        # For now, we return dummy predictions as requested.
        return self.get_dummy_predictions(data)

    def get_dummy_predictions(self, data: pd.DataFrame):
        """Generates dummy predictions for requested targets."""
        tickers = data['ticker'].unique() if 'ticker' in data.columns else ['EURUSD']
        
        predictions = []
        for ticker in tickers:
            ticker_data = data[data['ticker'] == ticker] if 'ticker' in data.columns else data
            if ticker_data.empty:
                continue
                
            pred = {
                'ticker': ticker,
                'datetime': ticker_data['cropped_date'].iloc[0] if 'cropped_date' in ticker_data.columns else datetime.now(),
                
                # Quantile ranges for future movement (regression/quantile prediction)
                'trg_future_range_1h': np.random.uniform(10, 50),  # dummy value in pips
                'trg_future_range_3h': np.random.uniform(20, 80),
                'trg_future_range_6h': np.random.uniform(30, 120),
                'trg_future_range_24h': np.random.uniform(50, 200),
                
                # Boolean chaos indicators
                'trg_is_chaos_1h': bool(np.random.choice([True, False])),
                'trg_is_chaos_3h': bool(np.random.choice([True, False])),
                'trg_is_chaos_6h': bool(np.random.choice([True, False])),
                'trg_is_chaos_24h': bool(np.random.choice([True, False])),
                
                # Movement type indicators
                'trg_is_trend': bool(np.random.choice([True, False])),
                'trg_is_flat': bool(np.random.choice([True, False]))
            }
            predictions.append(pred)
            
        return pd.DataFrame(predictions)

    def preprocess_data_for_ml_predictions(self, news: pd.DataFrame, prices: pd.DataFrame):
        # 1. Preprocess News
        news_features = self.add_features_for_news(news)
        
        # 2. Preprocess Prices
        prices_features = self.add_features_for_prices(prices)
        
        # 3. Merge
        # Ensure timestamps are in the same format for merging
        news_features['cropped_date'] = pd.to_datetime(news_features['cropped_date'])
        prices_features['utc_dt'] = pd.to_datetime(prices_features['utc_dt'])
        
        data = news_features.merge(prices_features, left_on='cropped_date', right_on='utc_dt', how='inner')
        
        # 4. Add time features (week, dayofweek, etc.)
        data = self.add_time_features(data, 'cropped_date')
        
        # 5. Add stable hour feature
        data = self.add_stable_hour_feature(data, news)
        
        return data

    def add_stable_hour_feature(self, data: pd.DataFrame, original_news: pd.DataFrame) -> pd.DataFrame:
        """Adds stable hour based on the local timezone of the news currency."""
        tz_map = {
            'USD': 'US/Eastern', 'EUR': 'Europe/Berlin', 'GBP': 'Europe/London',
            'JPY': 'Asia/Tokyo', 'AUD': 'Australia/Sydney', 'CAD': 'America/Toronto',
            'CHF': 'Europe/Zurich', 'NZD': 'Pacific/Auckland'
        }
        
        news_copy = original_news.copy()
        news_copy['date_dt'] = pd.to_datetime(news_copy['date'])
        if news_copy['date_dt'].dt.tz is None:
            news_copy['date_dt'] = news_copy['date_dt'].dt.localize('UTC')
        
        def get_local_hour(row):
            tz = tz_map.get(row['currency'], 'UTC')
            return row['date_dt'].tz_convert(tz).hour
            
        news_copy['local_hour'] = news_copy.apply(get_local_hour, axis=1)
        news_copy['cropped_date'] = news_copy['date_dt'].apply(floor_or_ceil)
        
        # Take the local hour of the most important event for each hour
        stable_hours = (
            news_copy.sort_values(['cropped_date', 'importance'], ascending=[True, False])
            .drop_duplicates('cropped_date')
            .set_index('cropped_date')[['local_hour']]
            .rename(columns={'local_hour': 'stable_hour'})
        )
        
        return data.merge(stable_hours, left_on='cropped_date', right_index=True, how='left')
        
    def add_features_for_news(self, news) -> pd.DataFrame:
        df = news.copy()
        # Basic features
        df['class'] = df['comment'].astype(str).apply(classify_news)
        df['event_type'] = df['title'].apply(classify_event_type)
        df['period_extracted'] = df['period'].astype(str).apply(period_extraction)
        df['impact_rank'] = df['importance'] + 1
        
        # Standardize date
        df['date_dt'] = pd.to_datetime(df['date'])
        if df['date_dt'].dt.tz is None:
            df['date_dt'] = df['date_dt'].dt.localize('UTC')
        else:
            df['date_dt'] = df['date_dt'].dt.tz_convert('UTC')
            
        df['cropped_date'] = df['date_dt'].apply(floor_or_ceil)

        key_events = {
            "NFP", "CPI", "CORE_CPI", "PCE", "FOMC_RATE", 
            "FOMC_PRES_CONF", "PMI_MANUFACTURING", "PMI_SERVICES", "GDP"
        }
        
        # Aggregate features by hour
        news_agg = build_hour_features(df, key_event_types=key_events)
        return news_agg

    def add_features_for_prices(self, prices) -> pd.DataFrame:
        df = prices.copy()
        # Localize to UTC if needed
        df['utc_dt'] = pd.to_datetime(df['time'])
        if df['utc_dt'].dt.tz is None:
            df['utc_dt'] = df['utc_dt'].dt.localize('UTC')
        else:
            df['utc_dt'] = df['utc_dt'].dt.tz_convert('UTC')
            
        # Add price features from pipeline
        df = add_price_features(df, period=21)
        return df

    def add_time_features(self, data: pd.DataFrame, dt_col: str):
        df = data.copy()
        df[dt_col] = pd.to_datetime(df[dt_col])
        df['week'] = df[dt_col].dt.isocalendar().week.astype(int)
        df['dayofweek'] = df[dt_col].dt.dayofweek
        df['day'] = df[dt_col].dt.day
        df['hour'] = df[dt_col].dt.hour
        df['minute'] = df[dt_col].dt.minute
        return df
        
    def predict_volatility(self, data):
        """Predict volatility expansion (gradation 0-7)."""
        # Feature list from notebook for volatility
        features = [
            'news_count', 'high_impact_count', 'key_event_count', 'sum_impact', 
            'sum_event_weight', 'max_event_weight', 'dominant_event_type', 'event_entropy', 
            'has_nfp', 'has_gdp', 'has_cpi', 'has_fomc_rate', 'has_fomc_pres_conf',
            'has_core_cpi', 'has_pmi_services', 'has_pce', 'has_pmi_manufacturing',
            'last_key_event_name', 'last_key_event_hours_ago', 
            'base_currency', 'quote_currency', 'instrument',
            'kaufman_efficiency_ratio', 'custom_efficiency_ratio', 'wick_ratio', 
            'relative_range', 'relative_atr', 'normalized_bb_width', 
            'prev_5_mean_range_norm', 'prev_5_min_range_norm', 'prev_5_max_range_norm',
            'dayofweek', 'stable_hour', 'minute'
        ]
        
        X = data[features]
        prediction = self.volatility_model.predict(X)
        return prediction

    def predict_range(self, data: pd.DataFrame):
        """Predict future range (regression)."""
        # Similar features as volatility
        features = [
            'news_count', 'high_impact_count', 'key_event_count', 'sum_impact', 
            'sum_event_weight', 'max_event_weight', 'dominant_event_type', 'event_entropy', 
            'has_nfp', 'has_gdp', 'has_cpi', 'has_fomc_rate', 'has_fomc_pres_conf',
            'has_core_cpi', 'has_pmi_services', 'has_pce', 'has_pmi_manufacturing',
            'last_key_event_name', 'last_key_event_hours_ago', 
            'base_currency', 'quote_currency', 'instrument',
            'kaufman_efficiency_ratio', 'custom_efficiency_ratio', 'wick_ratio', 
            'relative_range', 'relative_atr', 'normalized_bb_width', 
            'prev_5_mean_range_norm', 'prev_5_min_range_norm', 'prev_5_max_range_norm',
            'dayofweek', 'stable_hour', 'minute'
        ]
        X = data[features]
        prediction = self.range_model.predict(X)
        return prediction

    def predict_chaos(self, data: pd.DataFrame):
        """Predict direction changes, flat, trend, and chaos."""
        features = [
            'news_count', 'high_impact_count', 'key_event_count', 'sum_impact', 
            'sum_event_weight', 'max_event_weight', 'dominant_event_type', 'event_entropy', 
            'has_nfp', 'has_gdp', 'has_cpi', 'has_fomc_rate', 'has_fomc_pres_conf',
            'has_core_cpi', 'has_pmi_services', 'has_pce', 'has_pmi_manufacturing',
            'last_key_event_name', 'last_key_event_hours_ago', 
            'base_currency', 'quote_currency', 'instrument',
            'kaufman_efficiency_ratio', 'custom_efficiency_ratio', 'wick_ratio', 
            'relative_range', 'relative_atr', 'normalized_bb_width', 
            'prev_5_mean_range_norm', 'prev_5_min_range_norm', 'prev_5_max_range_norm',
            'dayofweek', 'stable_hour', 'minute'
        ]
        X = data[features]
        
        # Predict the main chaos class
        chaos_pred = self.chaos_model.predict(X)
        chaos_proba = self.chaos_model.predict_proba(X)[:, 1]
        
        # Since we might not have separate models for direction changes, flat, and trend,
        # we can derive them from the chaos prediction if the model was trained on a multi-class target,
        # or return the same if it's a binary chaos classifier.
        # Based on the user input, these are the targets we want to return:
        return {
            "is_chaos": chaos_pred,
            "chaos_probability": chaos_proba,
            # Placeholder for other targets if separate models are not provided
            "direction_changes": None, 
            "is_flat": None,
            "is_trend": None
        }


# def check_for_new_events():
#     """Check for new economic events and run predictions."""
#     # Create database session
#     db = SessionLocal()
    
#     try:
#         # Get recent events (last hour)
#         one_hour_ago = datetime.now() - timedelta(hours=1)
        
#         recent_events = db.query(TodayEconomicNews).filter(
#             TodayEconomicNews.created_at >= one_hour_ago,
#             TodayEconomicNews.event_weight >= 3
#         ).order_by(TodayEconomicNews.created_at.desc()).all()
        
#         if recent_events:
#             # Convert to DataFrame for compatibility with existing code
#             events_data = [{
#                 'id': event.id,
#                 'date': event.date,
#                 'currency': event.currency,
#                 'importance': event.importance,
#                 'title': event.title,
#                 'indicator': event.indicator,
#                 'country': event.country,
#                 'category': event.category,
#                 'event_type': event.event_type,
#                 'is_key_event': event.is_key_event,
#                 'event_weight': event.event_weight,
#                 'custom_event_time': event.custom_event_time,
#                 'created_at': event.created_at
#             } for event in recent_events]
            
#             recent_events_df = pd.DataFrame(events_data)
            
#             # Load trained model
#             predictor = NewsPredictor()
            
#             # Run predictions
#             predictions, probabilities = predictor.predict_volatility(recent_events_df)
            
#             # Store predictions in database
#             store_predictions(recent_events_df, predictions, probabilities)
            
#             # Send alerts if needed
#             send_alerts(recent_events_df, predictions, probabilities)
#     except Exception as e:
#         print(f"Error checking for new events: {e}")
#     finally:
#         db.close()


# def store_predictions(news_data, predictions, probabilities):
#     """Store model predictions in database."""
#     db = SessionLocal()
    
#     try:
#         for i, (_, row) in enumerate(news_data.iterrows()):
#             prediction = Prediction(
#                 news_id=row['id'],
#                 model_type='volatility_classifier',
#                 prediction_value=float(predictions[i]),
#                 prediction_probability=float(probabilities[i])
#             )
#             db.add(prediction)
        
#         db.commit()
#         print(f"Stored {len(predictions)} predictions in database")
#     except Exception as e:
#         print(f"Error storing predictions: {e}")
#         db.rollback()
#     finally:
#         db.close()


# def send_alerts(news_data, predictions, probabilities):
#     """Send alerts to subscribed users."""
#     # Implementation would integrate with your Telegram bot
#     # to send notifications about high-impact events
#     print(f"Would send alerts for {len(news_data)} events")
#     pass


## Initialize database tables if not already created
# from db.database import create_tables
# create_tables()