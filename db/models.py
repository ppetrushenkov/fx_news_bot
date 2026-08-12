# from numpy import integer
from sqlalchemy import (
    Column,
    Integer,
    Numeric,
    String,
    Boolean,
    Float,
    DateTime,
    Time,
    ForeignKey,
    UniqueConstraint, Date
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

import datetime


Base = declarative_base()


class Events(Base):
    __tablename__ = 'events'
    _id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer)
    title = Column(String, nullable=False)
    country = Column(String, nullable=True)
    indicator = Column(String)
    category = Column(String, nullable=True)
    period = Column(String)
    referenceDate = Column(DateTime, nullable=True)
    source = Column(String)
    source_url = Column(String)
    actual = Column(Float, nullable=True)
    previous = Column(Float, nullable=True)
    forecast = Column(Float, nullable=True)
    actualRaw = Column(Float, nullable=True)
    previousRaw = Column(Float, nullable=True)
    forecastRaw = Column(Float, nullable=True)
    currency = Column(String)
    importance = Column(Integer)
    date = Column(DateTime)
    ticker = Column(String, nullable=True)
    unit = Column(String, nullable=True)
    comment = Column(String, nullable=True)
    scale = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("event_id", name="uq_event_id"),
    )


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


class DailyPrices(Base):
    __tablename__ = 'daily_prices'

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


class UserSettings(Base):
    __tablename__ = 'user_settings'

    user_id = Column(Integer, primary_key=True)
    user_timezone = Column(Numeric(3, 1))

    # --- Alerts settings ---
    daily_alerts = Column(Boolean, default=False)
    weekly_alerts = Column(Boolean, default=False)
    chaos_alerts = Column(Boolean, default=False)

    # --- Time to alert ---
    daily_alerts_time = Column(Time, default=datetime.time(8, 0))
    weekly_alerts_time = Column(Time, default=datetime.time(8, 0))

    # --- Importance filters for events ---
    show_low_importance = Column(Boolean, default=False)
    show_medium_importance = Column(Boolean, default=False)
    show_high_importance = Column(Boolean, default=True)

    # --- ML risk settings ---
    ml_risk_level = Column(String, default="base")  # can be "conservative", "medium", or "aggressive"

    updated_at = Column(DateTime, default=func.now())
