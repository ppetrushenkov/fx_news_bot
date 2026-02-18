import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sqlalchemy.orm import sessionmaker
from db.database import engine, get_db
from db.models import EconomicNews, Prediction

from catboost import CatBoostClassifier, CatBoostRegressor

from config import Config

# Create session factory for model executor
ModelSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class NewsPredictor:
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
        
    def prepare_features(self, news_data):
        """Prepare features for model prediction."""
        # Implementation based on your existing feature engineering
        # This would include:
        # - News features (importance, event type, etc.)
        # - Market features (if available)
        # - Time-based features
        # For now, return empty array as placeholder
        return np.array([]).reshape(len(news_data), 0)
        
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


def check_for_new_events():
    """Check for new economic events and run predictions."""
    # Create database session
    db = ModelSession()
    
    try:
        # Get recent events (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        recent_events = db.query(EconomicNews).filter(
            EconomicNews.created_at >= one_hour_ago,
            EconomicNews.event_weight >= 3
        ).order_by(EconomicNews.created_at.desc()).all()
        
        if recent_events:
            # Convert to DataFrame for compatibility with existing code
            events_data = [{
                'id': event.id,
                'date': event.date,
                'currency': event.currency,
                'importance': event.importance,
                'title': event.title,
                'indicator': event.indicator,
                'country': event.country,
                'category': event.category,
                'event_type': event.event_type,
                'is_key_event': event.is_key_event,
                'event_weight': event.event_weight,
                'custom_event_time': event.custom_event_time,
                'created_at': event.created_at
            } for event in recent_events]
            
            recent_events_df = pd.DataFrame(events_data)
            
            # Load trained model
            predictor = NewsPredictor()
            
            # Run predictions
            predictions, probabilities = predictor.predict_volatility(recent_events_df)
            
            # Store predictions in database
            store_predictions(recent_events_df, predictions, probabilities)
            
            # Send alerts if needed
            send_alerts(recent_events_df, predictions, probabilities)
    except Exception as e:
        print(f"Error checking for new events: {e}")
    finally:
        db.close()


def store_predictions(news_data, predictions, probabilities):
    """Store model predictions in database."""
    db = ModelSession()
    
    try:
        for i, (_, row) in enumerate(news_data.iterrows()):
            prediction = Prediction(
                news_id=row['id'],
                model_type='volatility_classifier',
                prediction_value=float(predictions[i]),
                prediction_probability=float(probabilities[i])
            )
            db.add(prediction)
        
        db.commit()
        print(f"Stored {len(predictions)} predictions in database")
    except Exception as e:
        print(f"Error storing predictions: {e}")
        db.rollback()
    finally:
        db.close()


def send_alerts(news_data, predictions, probabilities):
    """Send alerts to subscribed users."""
    # Implementation would integrate with your Telegram bot
    # to send notifications about high-impact events
    print(f"Would send alerts for {len(news_data)} events")
    pass


# Initialize database tables if not already created
from db.database import create_tables
create_tables()