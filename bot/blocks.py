from __future__ import annotations
from typing import Optional
from config import Config

from tqdm import tqdm
from time import sleep
from datetime import timedelta, datetime, timezone
from twelvedata import TDClient
from sqlalchemy import text, select, func, insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

import numpy as np
import pandas as pd
import pytz

from db.database import SessionLocal, create_tables
from db.models import Events, Prices, Ranges
from db.data_handler import DBHandler

from bot.data_loader import get_economic_events, get_historical_prices
from bot.feature_engineer import (
    get_most_important_events, 
    _utc_now, _next_sunday_utc, 
    _df_standardize_event_dates,
    _df_standardize_prices
)

from ml.event_categories import EVENT_WEIGHTS_D
from ml.news_featuring import get_max_weight_event
from ml.targets import get_future_range_from_now



def create_block() -> None:
    """
    Block 1: Create tables:
    - Events: Past and future events (UTC)
    - Prices: TwelveData historical prices
    - Ranges: Ranges, that produced every event in 1, 3, 6 and 24 hours
    - Predictions: Store predictions for every hour, that contains at least one event
    - UserSubscriptions: Store telegram user ids, that subscribed on alerts
    """
    create_tables()


def _get_default_tickers(tickers: Optional[list[str]]) -> list[str]:
    return tickers or getattr(Config, "SUPPORTED_TICKERS", None) or [
        "EUR/USD",
        "GBP/USD",
        "USD/CHF",
        "USD/JPY",
        "USD/CAD",
        "AUD/USD",
        "NZD/USD",
    ]


def _populate_past_events(*, db: DBHandler, now: datetime, countries: Optional[list[str]]) -> None:
    print("[UPDATE BLOCK] Populating past events...")

    # If the table is empty
    if db.sess.execute(select(Events)).first() is None:
        start_datetime = now - timedelta(days=45)
    else:
        # Check max date in table
        q = select(func.max(Events.date))
        start_datetime = db.sess.execute(q).scalar()
        start_datetime = start_datetime.astimezone(timezone.utc)
        print("[UPDATE BLOCK] Start date =", start_datetime)
        print("[UPDATE BLOCK] Start date type is", type(start_datetime))

    today = now.date()
    start_date = start_datetime.date()
    days_delta = (now - start_datetime).days

    if start_date < today:
        start_date_s = start_date.strftime("%Y-%m-%d")
        end_date_s = now.strftime("%Y-%m-%d")
        past_events = get_economic_events(
            start_date=start_date_s,
            end_date=end_date_s,
            countries=countries,
        )
        past_events = _df_standardize_event_dates(past_events)
        existed_event_ids = db.sess.execute(select(Events.id)).scalars().all()
        past_events = past_events[~past_events["id"].isin(existed_event_ids)]

        if len(past_events) > 0:
            db.write_into(past_events, Events.__tablename__, if_exists="append")
            print("[UPDATE BLOCK] Past events lenght:", len(past_events))
            print("[UPDATE BLOCK] Uploaded from:", start_date_s)
            print("[UPDATE BLOCK] Days uploaded:", days_delta)
        else:
            print("[UPDATE BLOCK] Past events: Nothing to append")
    else:
        print("[UPDATE BLOCK] The past events has no need to update")


def _populate_future_events(*, db: DBHandler, countries: Optional[list[str]]) -> None:
    print("[UPDATE BLOCK] Populating future events...")

    # Check max date in table
    q = select(func.max(Events.date))
    start_date = db.sess.execute(q).scalar()
    start_date = start_date.date()

    # Get date to start from and date to
    end_date = _next_sunday_utc()
    days_delta = (end_date - start_date).days

    print("[UPDATE BLOCK] Start date =", start_date)

    if start_date < end_date:
        start_date_s = start_date.strftime("%Y-%m-%d")
        end_date_s = end_date.strftime("%Y-%m-%d")
        future_events = get_economic_events(
            start_date=start_date_s,
            end_date=end_date_s,
            countries=countries,
        )
        future_events = _df_standardize_event_dates(future_events)
        existed_event_ids = db.sess.execute(select(Events.id)).scalars().all()
        future_events = future_events[~future_events["id"].isin(existed_event_ids)]

        if len(future_events) > 0:
            db.write_into(future_events, Events.__tablename__, if_exists="append")
            print("[UPDATE BLOCK] Future events lenght:", len(future_events))
            print("[UPDATE BLOCK] Uploaded from:", start_date_s)
            print("[UPDATE BLOCK] Days uploaded:", days_delta)
        else:
            print("[UPDATE BLOCK] Future events: Nothing to append")
    else:
        print("[UPDATE BLOCK] The future events has no need to be updated")


def _populate_prices(
    *,
    db: DBHandler,
    now: datetime,
    tickers: list[str],
    prices_interval: str,
) -> None:
    print("[UPDATE BLOCK] Populating prices...")
    client = TDClient(apikey=Config.TWELVE_API, timezone=pytz.timezone("Etc/UTC"))

    if db.sess.execute(select(Prices)).first() is None:
        print("[UPDATE BLOCK] No rows in table Prices. Populating data for the last 50 days...")
        start_date_dt = now - timedelta(days=50)
    else:
        print(
            '[UPDATE BLOCK] The table Prices already have some data. Get last date and populating data with a fresh prices...'
        )
        q = select(func.max(Prices.datetime))
        start_date_dt = db.sess.execute(q).scalar()
        start_date_dt += timedelta(hours=1)

    start_date_s = start_date_dt.strftime("%Y-%m-%d %H:%M:%S")
    pbar = tqdm(tickers, postfix="")

    for ticker in pbar:
        pbar.set_description(f"Processing {ticker} data")
        q = select(func.max(Prices.datetime)).where(Prices.ticker == ticker)
        last_date = db.sess.execute(q).scalar()

        if last_date is not None:
            last_date = last_date.replace(minute=0, second=0, microsecond=0)
            now_rounded = now.replace(minute=0, second=0, microsecond=0)

            last_date = last_date.astimezone(timezone.utc).replace(tzinfo=None)
            now_rounded = now_rounded.astimezone(timezone.utc).replace(tzinfo=None)

            if now_rounded == last_date:
                pbar.set_postfix(
                    {"status": f"{ticker} up-to-date (last_date={last_date}), nothing to download."}
                )
                continue

            ticker_start_date = (last_date + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            ticker_start_date = start_date_s

        pbar.set_postfix({"status": "Downloading..."})
        p = get_historical_prices(
            client,
            ticker,
            start_date=ticker_start_date,
            interval=prices_interval,
        )
        p = _df_standardize_prices(p, ticker)
        pbar.set_postfix({"status": "Write into the Prices table..."})
        db.write_into(p, Prices.__tablename__, "append")
        pbar.set_postfix({"status": "Completed. Sleep for 1 sec..."})
        sleep(1)

    print("[UPDATE BLOCK] Table Prices were populated.")


def _populate_event_ranges(*, db: DBHandler, now: datetime) -> None:
    print("[UPDATE BLOCK] Populating event ranges...")

    # 1. Get Events
    if db.sess.execute(select(Ranges)).first() is None:
        print('[UPDATE BLOCK] The table "Ranges" is empty. Take all data for processing...')
        past_events_query = db.sess.query(Events.title, Events.date).filter(Events.date <= now)
        ranges_start_dt = None
    else:
        # Check max date in table
        q = select(func.max(Ranges.datetime))
        ranges_start_dt = db.sess.execute(q).scalar()
        print(f'[UPDATE BLOCK] The table "Ranges" is not empty. Take the events from the {ranges_start_dt.date()}')
        past_events_query = (
            db.sess.query(Events.title, Events.date).filter(Events.date > ranges_start_dt, Events.date <= now)
        )

    past_events = pd.read_sql(past_events_query.statement, db.sess.bind)
    past_events["date"] = pd.to_datetime(past_events["date"], errors="coerce")

    min_event_dt = past_events["date"].dropna().min()
    if pd.isna(min_event_dt):
        # No (new) events in this window; still compute ranges for "No main events"
        start_date_dt = ranges_start_dt or (now - timedelta(days=50))
    else:
        start_date_dt = min_event_dt.to_pydatetime()

    # 2. Get prices
    prices_query = (
        db.sess.query(Prices.datetime, Prices.ticker, Prices.open, Prices.high, Prices.low, Prices.close)
        .filter(Prices.datetime >= start_date_dt)
    )
    prices = pd.read_sql(prices_query.statement, db.sess.bind)

    # 3. Preprocess events
    past_events["rounded_time"] = past_events["date"].apply(lambda x: x.floor("1h"))
    past_events["main_event"] = past_events["title"].astype(str).apply(get_most_important_events)
    past_events["weights"] = past_events["main_event"].apply(lambda x: EVENT_WEIGHTS_D.get(x, 0))
    past_events["main_event"].fillna("No main events", inplace=True)

    if len(past_events) > 0 and past_events["rounded_time"].notna().any():
        main_event_by_time = past_events.groupby("rounded_time")[["weights", "main_event"]].apply(
            lambda x: get_max_weight_event(x, mie_col="main_event")
        )
        main_event_by_time.name = "main_event"
    else:
        # No events at all -> every hour is "No main events"
        main_event_by_time = pd.Series(dtype="object", name="main_event")

    # 4. Preprocess prices
    prices["datetime"] = pd.to_datetime(prices["datetime"], errors="coerce")

    for h in [1, 3, 6, 24]:
        prices[f"future_range_{h}"] = prices.groupby("ticker", group_keys=False).apply(
            lambda df: get_future_range_from_now(df, h)
        )
        prices[f"future_range_{h}_log"] = prices[f"future_range_{h}"].apply(lambda x: np.log1p(x))

    # 5. Join prices to the events
    # Include hours with NO events: they become "No main events"
    price_times = pd.Index(pd.Series(prices["datetime"].dropna().unique()).sort_values(), name="rounded_time")
    max_weight_events_by_time = (
        main_event_by_time.reindex(price_times).fillna("No main events").to_frame()
    )

    df = pd.merge(max_weight_events_by_time, prices, left_on="rounded_time", right_on="datetime")

    # 6. Get future ranges for each main_event
    future_ranges_by_ticker_and_event = (
        df.groupby(["main_event", "ticker"], group_keys=True).apply(lambda x: x.tail(50)).reset_index()
    )

    # Prepare rows as list of dictionaries for bulk insert
    insert_rows = []
    for _, row in future_ranges_by_ticker_and_event.iterrows():
        fr_1h = row.get("future_range_1_log", row.get("future_range_1h_log", None))
        fr_3h = row.get("future_range_3_log", row.get("future_range_3h_log", None))
        fr_6h = row.get("future_range_6_log", row.get("future_range_6h_log", None))
        fr_24h = row.get("future_range_24_log", row.get("future_range_24h_log", None))

        # SQLite doesn't like NaN for bound params in some cases; use NULL instead
        fr_1h = None if pd.isna(fr_1h) else float(fr_1h)
        fr_3h = None if pd.isna(fr_3h) else float(fr_3h)
        fr_6h = None if pd.isna(fr_6h) else float(fr_6h)
        fr_24h = None if pd.isna(fr_24h) else float(fr_24h)

        insert_rows.append(
            {
                "datetime": row["datetime"],
                "ticker": row["ticker"],
                "main_event": row["main_event"],
                "future_range_1h": fr_1h,
                "future_range_3h": fr_3h,
                "future_range_6h": fr_6h,
                "future_range_24h": fr_24h,
            }
        )

    if insert_rows:
        # Upsert to handle existing (main_event, datetime, ticker)
        stmt = sqlite_insert(Ranges).values(insert_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["main_event", "datetime", "ticker"],
            set_={
                "future_range_1h": stmt.excluded.future_range_1h,
                "future_range_3h": stmt.excluded.future_range_3h,
                "future_range_6h": stmt.excluded.future_range_6h,
                "future_range_24h": stmt.excluded.future_range_24h,
            },
        )
        db.sess.execute(stmt)
    db.sess.commit()
    print("[UPDATE BLOCK] Ranges were populated")


def update_block(
    *,
    countries: Optional[list[str]] = None,
    tickers: Optional[list[str]] = None,
    prices_interval: str = "1h",
) -> None:
    """
    Block 2: Check if any data in DataBase. If not, populate with all data for the last 2 months and next week further.
    Else check the last data in tables and populate with a fresh data.
    For the events also check the future data to ensure the data still the same.
    """
    db = DBHandler()

    now = _utc_now()
    tickers = _get_default_tickers(tickers)

    _populate_past_events(db=db, now=now, countries=countries)
    _populate_future_events(db=db, countries=countries)
    _populate_prices(db=db, now=now, tickers=tickers, prices_interval=prices_interval)
    _populate_event_ranges(db=db, now=now)
    
    print('\n////////////////////////////////////')
    print('[INFO] The database has been updated')
    print('////////////////////////////////////\n')
    return
