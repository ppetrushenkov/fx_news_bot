from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func


Base = declarative_base()

class EconomicNews(Base):
    __tablename__ = 'economic_news'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False)
    currency = Column(String)
    importance = Column(Integer)
    title = Column(String)
    indicator = Column(String)
    country = Column(String)
    category = Column(String)
    event_type = Column(String)
    is_key_event = Column(Boolean)
    event_weight = Column(Integer)
    custom_event_time = Column(String)
    created_at = Column(DateTime, default=func.now())

class TodayEconomicNews(Base):
    __tablename__ = 'today_economic_news'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    country = Column(String)
    indicator = Column(String)
    category = Column(String)
    period = Column(String)
    referenceDate = Column(DateTime)
    source = Column(String)
    source_url = Column(String)
    actual = Column(String)
    previous = Column(String)
    forecast = Column(String)
    actualRaw = Column(Float)
    previousRaw = Column(Float)
    forecastRaw = Column(Float)
    currency = Column(String)
    importance = Column(Integer)
    date = Column(DateTime)
    ticker = Column(String)
    unit = Column(String)
    comment = Column(String)
    scale = Column(String)

class Prices(Base):
    __tablename__ = 'prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    datetime = Column(DateTime, nullable=False)
    ticker = Column(String, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    news_id = Column(Integer, ForeignKey('economic_news.id'))
    model_type = Column(String)
    prediction_value = Column(Float)
    prediction_probability = Column(Float)
    volatility_category = Column(Integer)
    predicted_at = Column(DateTime, default=func.now())

class UserSubscription(Base):
    __tablename__ = 'user_subscriptions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, unique=True)
    chat_id = Column(Integer)
    subscribed_to_alerts = Column(Boolean, default=True)
    subscribed_to_daily_summary = Column(Boolean, default=False)
    user_tz = Column(String, default='Europe/Moscow')  # TODO: CHANGE WHEN BUILD THE APP
    created_at = Column(DateTime, default=func.now())