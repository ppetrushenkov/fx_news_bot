from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func


Base = declarative_base()

class AllEconomicNews(Base):
    __tablename__ = 'all_economic_news'

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


class TodayEventsAggregated(Base):
    __tablename__ = 'today_events_aggregated'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agg_time = Column(DateTime)
    news_cnt = Column(Integer)
    high_imp_cnt = Column(Integer)
    sum_imp_cnt = Column(Integer)
    max_imp_rnk = Column(Integer)
    main_event_type = Column(String)
    is_core_cpi = Column(Boolean)
    is_cpi = Column(Boolean)
    is_fomc = Column(Boolean)
    is_gdp = Column(Boolean)
    is_nfp = Column(Boolean)
    is_pce = Column(Boolean)
    is_pmi_mnf = Column(Boolean)
    is_pmi_srv = Column(Boolean)
    is_retail = Column(Boolean)
    is_cat_bonds = Column(Boolean)
    is_cat_sb_speech = Column(Boolean)
    is_cat_commodities = Column(Boolean)
    is_cat_consumer_housing = Column(Boolean)
    is_cat_economic_activity = Column(Boolean)
    is_cat_gdp = Column(Boolean)
    is_cat_inflation = Column(Boolean)
    is_cat_labor_market = Column(Boolean)
    is_cat_manufactoring = Column(Boolean)
    is_cat_monetary_policy = Column(Boolean)
    is_cat_sentiment = Column(Boolean)
    is_cat_trade_finance = Column(Boolean)
    is_cur_eur = Column(Boolean)
    is_cur_gpb = Column(Boolean)
    is_cur_jpy = Column(Boolean)
    is_cur_aud = Column(Boolean)
    is_cur_usd = Column(Boolean)
    prev_main_event_type = Column(String)
    hours_from_last_key_event = Column(Integer)
    hours_from_last_max_impact = Column(Integer)


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