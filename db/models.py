from numpy import integer
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func


Base = declarative_base()


class Events(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer)
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



# class TodayEconomicEvents(Base):
#     __tablename__ = 'today_economic_events'
    
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     title = Column(String, nullable=False)
#     country = Column(String)
#     indicator = Column(String)
#     category = Column(String)
#     period = Column(String)
#     referenceDate = Column(DateTime)
#     source = Column(String)
#     source_url = Column(String)
#     actual = Column(String)
#     previous = Column(String)
#     forecast = Column(String)
#     actualRaw = Column(Float)
#     previousRaw = Column(Float)
#     forecastRaw = Column(Float)
#     currency = Column(String)
#     importance = Column(Integer)
#     date = Column(DateTime)
#     ticker = Column(String)
#     unit = Column(String)
#     comment = Column(String)
#     scale = Column(String)


class Prices(Base):
    __tablename__ = 'prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    datetime = Column(DateTime, nullable=False)
    ticker = Column(String, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)

    __table_args__ = (
        UniqueConstraint("datetime", "ticker", name="uq_prices_datetime_ticker"),
    )


class Ranges(Base):
    """
    Stores realized future ranges after an event (computed from price bars).

    Required structure from user:
    - main_event name
    - datetime
    - future_range_1h/3h/6h/24h
    """

    __tablename__ = "ranges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    datetime = Column(DateTime, nullable=False)
    main_event = Column(String, nullable=False)  # or "No main event"

    future_range_1h = Column(Float)
    future_range_3h = Column(Float)
    future_range_6h = Column(Float)
    future_range_24h = Column(Float)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint("main_event", "datetime", "ticker", name="uq_event_ranges_main_event_datetime"),
    )


class Predictions(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    rounded_date = Column(DateTime)
    ticker = Column(String)
    trg_future_range_1h = Column(String)  # Will be stored as string splitted with "-" like "54 - 76"
    trg_future_range_3h = Column(String) 
    trg_future_range_6h = Column(String) 
    trg_future_range_24h = Column(String)
    trg_future_overall_range_1h = Column(String)
    trg_future_overall_range_3h = Column(String)
    trg_future_overall_range_6h = Column(String)
    trg_future_overall_range_24h = Column(String)
    trg_expect_big_doji = Column(Integer)  # 1 if in the next 4 hours we expect the big doji
    trg_future_dir_changes = Column(Integer)
    trg_is_flat = Column(Integer)
    trg_is_trend = Column(Integer)
    trg_is_chaos_1h = Column(Integer)
    trg_is_chaos_3h = Column(Integer)
    trg_is_chaos_6h = Column(Integer)
    trg_is_chaos_24h = Column(Integer)

    __table_args__ = (
        UniqueConstraint("rounded_date", "ticker", name="uq_predictions_date_ticker"),
    )


class UserSubscriptions(Base):
    __tablename__ = 'user_subscriptions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, unique=True)
    chat_id = Column(Integer)
    subscribed_to_alerts = Column(Boolean, default=True)
    subscribed_to_daily_summary = Column(Boolean, default=False)
    user_tz = Column(String, default='Etc/UTC')
    created_at = Column(DateTime, default=func.now())