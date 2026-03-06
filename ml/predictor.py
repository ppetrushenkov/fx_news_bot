from datetime import datetime, timedelta
from sys import prefix

from db.database import SessionLocal
from db.models import TodayEconomicNews, Prediction

from news_featuring import classify_news, classify_event_type, period_extraction, floor_or_ceil, build_hour_features

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
        volatility_predictions = self.predict_volatility(data)
        range_predictions = self.predict_range(data)
        chaos_predictions = self.predict_chaos(data)
        return (volatility_predictions, range_predictions, chaos_predictions)

    def preprocess_data_for_ml_predictions(self, news: pd.DataFrame, prices: pd.DataFrame):
        # news = self.add_features_for_news(news)  # News already preprocessed
        prices = self.add_features_for_prices(prices)
        data = news.merge(prices, on='utc_datetime', how='inner')
        # TODO: add features, that should be added only here if I can't add it earlier
        return data
        
    def add_features_for_news(self, news) -> pd.DataFrame:
        # Add features
        news['class'] = news['comment'].astype(str).apply(classify_news)
        news['event_type'] = news['title'].apply(classify_event_type)
        news['period'] = news['period'].astype(str).apply(period_extraction)

        news['impact_rank'] = news['importance'] + 1
        # TODO: Add source feature

        # Dummy features
        news = self.add_dummies(news, 'period', prefix='per', drop_list=None)
        news = self.add_dummies(news, 'class', prefix='cat', drop_list=['cat_OTHER'])
        news = self.add_dummies(news, 'event_type', prefix='e', drop_list=['e_OTHER'])
        news = self.add_dummies(news, 'currency', prefix='cur', drop_list=None)

        # TODO: Consider convert datetime to moscow timezone here
        news['cropped_event_time'] = news['utc_dt'].progress_apply(floor_or_ceil)

        key_events = {
            "NFP", "CPI", "CORE_CPI", "PCE", "FOMC_RATE", "FOMC_PRES_CONF",
            "PMI_MANUFACTURING", "PMI_SERVICES", "GDP"
        }
        news_agg = build_hour_features(news, key_event_types=key_events)
        return news_agg

    def add_dummies(self, news: pd.DataFrame, col: str, prefix=None, drop_list=None):
        dummy = pd.get_dummies(news[col], prefix=prefix).astype(int)
        if drop_list:
            dummy = dummy.drop(drop_list, axis=1)
        return pd.concat([news, dummy], axis=1)

    def add_features_for_prices(self, prices) -> pd.DataFrame:
        return None
        
    def predict_volatility(self, news_data):
        """Predict volatility based on news data."""
        features = self.prepare_features(news_data)
        if features.size > 0:
            prediction = self.model.predict(features)
            probability = self.model.predict_proba(features)[:, 1]
        else:
            # Return dummy values for now
            prediction = np.zeros(len(news_data))
            probability = np.zeros(len(news_data))
        
        return prediction, probability

    def predict_range(self, data: pd.DataFrame):
        pass

    def predict_chaos(self, data: pd.DataFrame):
        pass


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