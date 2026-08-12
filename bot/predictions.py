from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd
from pandas import DataFrame
from sqlalchemy import func, select

from bot.parameters import ml_thresholds
from db.data_handler import DBHandler
from db.models import DailyPrices, Prices
from ml.predictor import FxRangePredictor
from utils.datetime_utils import utc_now


async def get_predictions_for_next_hour() -> tuple[DataFrame, DataFrame, dict]:
    print('[PREDICTIONS] Start the function to checking the market for next hour')
    db = DBHandler()
    predictor = FxRangePredictor()

    # Preparing Events and Prices
    agg_events, events = await get_next_hour_event_features(db, predictor)
    agg_prices_latest = await get_last_price_features(db, predictor)

    # Unite all together
    df = agg_prices_latest.merge(agg_events, how='cross')

    # Add range features for each event
    df = db.add_last_ranges_for_each_event(df)

    # Prepare data for predictions
    features = [
        i for i in df.columns if
        i not in ['time', 'time_to_check', 'rounded_time', 'datetime', 'open', 'high', 'low', 'close']
    ]
    df = df[features]

    # Get predictions
    print('[PREDICTIONS] Add predictions to DataFrame for further processing')
    predictions = predictor.get_hourly_predictions(df, ml_threshs=ml_thresholds)  # TODO: Add thresholds for different risk types
    print('[PREDICTIONS] Predictions for the next hour')
    print(predictions)
    return agg_events, events, predictions


async def get_daily_predictions():
    db = DBHandler()
    predictor = FxRangePredictor()

    # Preparing Events and Prices
    agg_events, events = await get_events_features_for_today(db, predictor)
    agg_prices_latest = await get_daily_price_features(db, predictor)

    # Join
    df = agg_prices_latest.merge(agg_events, how='cross')

    # Prepare data for predictions
    features = [
        i for i in df.columns if
        i not in ['time', 'time_to_check', 'rounded_time', 'datetime', 'open', 'high', 'low', 'close']
    ]
    df = df[features]

    # Get predictions
    print('[PREDICTIONS] Add predictions to DataFrame for further processing')
    predictions = predictor.get_daily_predictions(df)  # TODO: Add thresholds for different risk types
    print('[PREDICTIONS] Predictions for the next hour')
    print(predictions)
    return agg_events, events, predictions


async def get_daily_price_features(db: DBHandler, predictor: FxRangePredictor) -> DataFrame:
    yesterday = utc_now().date() - timedelta(days=1)
    if db.sess.execute(select(func.max(DailyPrices.datetime))).scalar() < yesterday:
        db.update_daily_prices()

    prices = db.get_last_prices(table=DailyPrices, period=21 * 2 + 1)
    agg_prices = predictor.price_transformer.transform(prices, timeframe='daily')
    agg_prices_latest = agg_prices[agg_prices['datetime'].dt.date == pd.to_datetime(yesterday).date()]
    return agg_prices_latest


async def get_last_price_features(db: DBHandler, predictor: FxRangePredictor) -> Any:
    db.update_hourly_prices()

    prices = db.get_last_prices(table=Prices, period=21 * 24 + 1)  # 21 days + 1 hour for ATR calculation
    agg_prices = predictor.price_transformer.transform(prices)
    agg_prices_latest = agg_prices.groupby('ticker').tail(1)
    return agg_prices_latest


async def get_events_features_for_today(db: DBHandler, predictor: FxRangePredictor) -> tuple[DataFrame, DataFrame]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    events = db.get_events_for_range(
        start=now - timedelta(days=2),
        end=now + timedelta(days=2)
    )
    print(events[['date', 'title']].head())
    events['rounded_time'] = pd.to_datetime(events['date'].dt.date)
    agg_events = predictor.event_transformer.transform(events, round_method='daily')
    print(agg_events.head())

    agg_events = agg_events[agg_events['rounded_time'] == now.date()]
    agg_events = agg_events.iloc[[0], :]
    return agg_events, events


async def get_next_hour_event_features(db: DBHandler, predictor: FxRangePredictor) -> tuple[Any, DataFrame]:
    # TODO: Do more precise filtering of events to only those that are relevant for the next hour
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    events = db.get_events_for_range(
        start=now - timedelta(days=2),
        end=now + timedelta(days=2)
    )
    print(events[['date', 'title']].head())
    events['rounded_time'] = events['date'].apply(lambda x: x.floor('1h'))
    print(events.head())
    agg_events = predictor.event_transformer.transform(events)


    print(agg_events.head())

    agg_events = agg_events[agg_events['rounded_time'] > pd.to_datetime(now)]
    agg_events = agg_events.iloc[[0], :]
    return agg_events, events
