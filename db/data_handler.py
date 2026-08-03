from datetime import datetime, timedelta
from time import sleep
from typing import Literal

from pytz import timezone
import pytz
from tqdm import tqdm

from config import Config

# from bot.app import start
from bot.data_loader import get_economic_events, get_historical_prices

from bot.feature_engineer import (
    get_most_important_events, 
    _utc_now, _next_sunday_utc, 
    _df_standardize_event_dates,
    _df_standardize_prices
)

from db.database import SessionLocal
from db.models import Events, Prices, Ranges

from sqlalchemy import func, select, desc, case, extract
from sqlalchemy.dialects.sqlite import insert

from twelvedata import TDClient

import numpy as np
import pandas as pd

from ml.event_categories import EVENT_WEIGHTS_D
from ml.news_featuring import get_max_weight_event
from ml.targets import get_future_range_from_now



class DBHandler:
    """
    This class simply the work with retrieving the data from different sources. 
    This class do:
    - retrieve events from the database;
    - aggregate events by specified datetime (e.g. 30min or hour);
    - fetch prices from different sources (TwelveData and maybe more);
    """
    def __init__(self, sess = None):
        self.sess = sess if sess is not None else SessionLocal()
        self.td = TDClient(apikey=Config.TWELVE_API, timezone=pytz.timezone("Etc/UTC"))
        self.countries = ['US', 'EU', 'GB', 'JP', 'DE', 'FR', 'CA', 'AU', 'SG', 'SE']
        self.supported_tickers = ['EUR/USD', 'GBP/USD', 'USD/CHF', 'USD/JPY', 'USD/CAD', 'AUD/USD', 'NZD/USD']
        self.days_for_new_dataset = 50
        self.prices_interval = '1h'

# +---------------  WRITE  --------------------+ #
    def write_into(self, df: pd.DataFrame, table_name: str, if_exists: Literal['replace', 'append'] = 'replace'):
        """Write DataFrame to database."""
        try:
            df.to_sql(table_name, self.sess.bind, if_exists=if_exists, index=False)
            print(f"Added {len(df)} news items to database")

        except Exception as e:
            print(f"Error populating database: {e}")
            self.sess.rollback()

# +---------------  GET  --------------------+ #
    def get_events_for_range(self, start: datetime, end: datetime):
        stmt = select(Events).where(
            Events.date.between(
                start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
                )
            )
        events = pd.read_sql(stmt, self.sess.bind)
        return events

    # def get_events_for_next_hour(self) -> pd.DataFrame:
    #     # 1. Calculate boundaries (Using timezone-aware UTC is best practice)
    #     now = _utc_now()
    #     one_hour_later = now + timedelta(hours=12)

    #     # 2. Build the query
    #     # This fetches records where the 'start_time' is between now and +1 hour
    #     stmt = select(Events).where(
    #         Events.date > now,
    #         Events.date <= one_hour_later
    #     )
        
    #     events = pd.read_sql_query(stmt, self.sess.connection())
    #     return events
    
    def get_last_prices(self, period: int):
        is_weekday = extract('dow', Prices.datetime).notin_([0, 6])

        window_func = func.row_number().over(
            partition_by=Prices.ticker,
            order_by=desc(Prices.datetime)
        ).label("row_num")

        subq = (
            select(Prices, window_func)
            .where(is_weekday)
            .subquery()
        )

        stmt = select(subq).where(subq.c.row_num <= period)

        prices = pd.read_sql(stmt, self.sess.bind)
        prices = prices.sort_values(by=['ticker', 'datetime'], ascending=[True, True])
        return prices

    def get_ranges_for_each_event(
        self,
        windows: tuple[int, ...] = (5, 20),
    ) -> pd.DataFrame:
        """
        For each (instrument, main_event) pair, calculate statistics using SQL window functions.
        This is more efficient than loading all data into memory.
        """
        range_cols = {
            "future_range_1h": "1h",
            "future_range_3h": "3h",
            "future_range_6h": "6h",
            "future_range_24h": "24h"
        }

        # 1. Subquery to rank ranges per (ticker, main_event)
        subq = select(
            Ranges.ticker,
            Ranges.main_event,
            Ranges.future_range_1h,
            Ranges.future_range_3h,
            Ranges.future_range_6h,
            Ranges.future_range_24h,
            func.row_number().over(
                partition_by=[Ranges.ticker, Ranges.main_event],
                order_by=desc(Ranges.datetime)
            ).label("rn")
        ).subquery()

        # 2. Aggregations
        agg_cols = [subq.c.ticker, subq.c.main_event]

        # Most recent range (previous_range) where rn = 1
        for col_name, suffix in range_cols.items():
            agg_cols.append(
                func.max(case((subq.c.rn == 1, getattr(subq.c, col_name)))).label(f"prev_range_{suffix}")
            )

        # Statistics for windows
        for w in windows:
            for col_name, suffix in range_cols.items():
                c = getattr(subq.c, col_name)
                agg_cols.append(func.min(case((subq.c.rn <= w, c))).label(f"min_range_{suffix}_{w}"))
                agg_cols.append(func.avg(case((subq.c.rn <= w, c))).label(f"mean_range_{suffix}_{w}"))
                agg_cols.append(func.max(case((subq.c.rn <= w, c))).label(f"max_range_{suffix}_{w}"))

        stmt = select(*agg_cols).group_by(subq.c.ticker, subq.c.main_event)
        
        # 3. Execute and return
        df = pd.read_sql(stmt, self.sess.bind)
        
        # Define column order
        col_order = ["ticker", "main_event"]
        for suffix in range_cols.values():
            col_order.append(f"prev_range_{suffix}")
        for w in windows:
            for suffix in range_cols.values():
                col_order.extend([f"min_range_{suffix}_{w}", f"mean_range_{suffix}_{w}", f"max_range_{suffix}_{w}"])
        
        if df.empty:
            return pd.DataFrame(columns=col_order)
            
        return df[col_order]

    def add_ranges_for_each_event(self, event_x_prices: pd.DataFrame):
        """
        Add ranges for each event in ``event_x_prices``.

        Args:
            event_x_prices (pd.DataFrame): DataFrame with columns:
                - main_event: Event name.
                - ticker: Ticker symbol.
                - ...
        """
        ranges = self.get_ranges_for_each_event()
        print('[INFO] Merging historical ranges by (ticker, main_event)...')
        
        # Ensure 'ticker' column names match for merging
        # If event_x_prices has 'instrument' instead of 'ticker', we handle it
        merge_on = ["ticker", "main_event"]
        
        event_x_prices = pd.merge(
            event_x_prices,
            ranges,
            on=merge_on,
            how="left",
        )
        return event_x_prices


# +---------------  UPDATE DATA  --------------------+ #
    def update_events(self):
        if self.sess.execute(select(Events)).first() is None:  # If no data in Events
            start_datetime = _utc_now() - timedelta(days=self.days_for_new_dataset)
        else:
            q = select(func.max(Events.date))
            start_datetime = self.sess.execute(q).scalar()
            start_datetime = start_datetime.astimezone(timezone('UTC'))

        start_date = start_datetime.date()
        end_date = _utc_now().date() + timedelta(days=7)  # Fetch events for the next week
        # end_date = _next_sunday_utc()

        if start_date < end_date:
            start_date_s = start_date.strftime("%Y-%m-%d")
            end_date_s = end_date.strftime("%Y-%m-%d")
            days_delta = (datetime.strptime(end_date_s, "%Y-%m-%d") - datetime.strptime(start_date_s, "%Y-%m-%d")).days
            events = get_economic_events(
                start_date=start_date_s,
                end_date=end_date_s,
                countries=self.countries,
            )
            events = _df_standardize_event_dates(events)

            if len(events) > 0:
                events = events.to_dict(orient='records')

                # INSERT VALUES
                stmt = insert(Events).values(events)
                on_conflict_stmt = stmt.on_conflict_do_nothing(
                    index_elements=["event_id"],
                ) 
                self.sess.execute(on_conflict_stmt)
                self.sess.commit()

                print("[UPDATE] Events lenght:", len(events))
                print("[UPDATE] Uploaded from:", start_date_s)
                print("[UPDATE] Days uploaded:", days_delta)
                print('[UPDATE] The events were populated (updated)')
            
            else:
                print('[UPDATE] No need to update the Events table')
    
    def update_prices(self):
        print("[UPDATE] Populating prices...")
        now = _utc_now()

        if self.sess.execute(select(Prices)).first() is None:
            print("[UPDATE] No rows in table Prices. Populating data for the last 50 days...")
            start_date_dt = now - timedelta(days=self.days_for_new_dataset)
        else:
            print(
                '[UPDATE] The table Prices already have some data. Get last date and populating data with a fresh prices...'
            )
            q = select(func.max(Prices.datetime))
            start_date_dt = self.sess.execute(q).scalar()
            start_date_dt += timedelta(hours=1)

        start_date_s = start_date_dt.strftime("%Y-%m-%d %H:%M:%S")
        pbar = tqdm(self.supported_tickers, postfix="")

        for ticker in pbar:
            pbar.set_description(f"Processing {ticker} data")
            q = select(func.max(Prices.datetime)).where(Prices.ticker == ticker)
            last_date = self.sess.execute(q).scalar()

            if last_date is not None:
                last_date = last_date.replace(minute=0, second=0, microsecond=0)
                now_rounded = now.replace(minute=0, second=0, microsecond=0)

                last_date = last_date.astimezone(timezone('UTC')).replace(tzinfo=None)
                now_rounded = now_rounded.astimezone(timezone('UTC')).replace(tzinfo=None)

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
                self.td,
                ticker,
                start_date=ticker_start_date,
                interval=self.prices_interval,
            )
            p = _df_standardize_prices(p, ticker)
            pbar.set_postfix({"status": "Write into the Prices table..."})
            self.write_into(p, Prices.__tablename__, "append")
            pbar.set_postfix({"status": "Completed. Sleep for 1 sec..."})
            sleep(1)

        print("[UPDATE] Table Prices were populated.")
        return None

    def update_ranges(self):
        print("[UPDATE] Populating event ranges...")
        # TODO: Почему у некоторых событий future_range_1h, future_range_3h, future_range_6h, future_range_24h = NaN?
        # +================================+
        # 1. Get Events
        # +================================+
        # If empty
        if self.sess.execute(select(Ranges)).first() is None:
            print('[UPDATE] The table "Ranges" is empty. Take all data for processing...')
            past_events_query = self.sess.query(Events.title, Events.date)
            ranges_start_dt = None
        else:
            # Check max date in Ranges
            q = select(func.max(Ranges.datetime))
            ranges_start_dt = self.sess.execute(q).scalar()
            print(f'[UPDATE] The table "Ranges" is not empty. Take the events from the {ranges_start_dt.date()}')
            # Get events, that are greater than the max date in Ranges
            past_events_query = self.sess.query(Events.title, Events.date).filter(Events.date > ranges_start_dt)

        past_events = pd.read_sql(past_events_query.statement, self.sess.bind)
        
        if past_events.empty:
            print("[UPDATE] No new events. Skipping.")
            return
        
        # +========================+
        # 2. Get prices
        # +========================+
        start_date_dt = past_events["date"].min()

        prices_query = (
            self.sess.query(Prices.ticker, Prices.datetime, Prices.open, Prices.high, Prices.low, Prices.close)
            .filter(Prices.datetime >= start_date_dt)
        )
        prices = pd.read_sql(prices_query.statement, self.sess.bind)

        if prices.empty:
            print("[UPDATE] No prices for the given date range. Skipping.")
            return

        # +=======================+
        # 3. Preprocess events
        # +=======================+
        past_events["rounded_time"] = past_events["date"].apply(lambda x: x.floor("1h"))
        past_events["most_important_event"] = past_events["title"].astype(str).apply(get_most_important_events)
        past_events["weights"] = past_events["most_important_event"].apply(lambda x: EVENT_WEIGHTS_D.get(x, 0))

        main_event = past_events \
            .groupby("rounded_time")[["weights", "most_important_event"]] \
                .apply(lambda x: get_max_weight_event(x))
        main_event.name = "most_important_event"
        main_event.fillna('No main events', inplace=True)

        # +=========================+
        # 4. Preprocess prices
        # +=========================+
        prices = prices.sort_values(['ticker', 'datetime'])

        for h in [1, 3, 6, 24]:
            prices[f"future_range_{h}h"] = prices \
                .groupby('ticker', group_keys=False) \
                .apply(lambda x: get_future_range_from_now(x, h))

        # +================================+
        # 5. Join prices to the events
        # +================================+
        main_event = main_event.to_frame().reset_index()
        events_x_prices = pd.merge(prices, main_event, left_on='datetime', right_on='rounded_time', how='inner')
        print(events_x_prices.head())

        # +=======================================+
        # 6. Get future ranges for each main_event
        # +=======================================+
        future_ranges_by_ticker_and_event = events_x_prices.groupby(["ticker", "rounded_time", "most_important_event"], group_keys=True) \
               .apply(lambda x: x.tail(50)).reset_index()

        # Prepare rows as list of dictionaries for bulk insert
        insert_rows = []
        for _, row in future_ranges_by_ticker_and_event.iterrows():
            fr_1h = row.get("future_range_1h", None)
            fr_3h = row.get("future_range_3h", None)
            fr_6h = row.get("future_range_6h", None)
            fr_24h = row.get("future_range_24h", None)

            insert_rows.append(
                {
                    "datetime": row["datetime"],
                    "ticker": row["ticker"],
                    "main_event": row["most_important_event"],
                    "future_range_1h": fr_1h,
                    "future_range_3h": fr_3h,
                    "future_range_6h": fr_6h,
                    "future_range_24h": fr_24h,
                }
            )

        if insert_rows:
            # Upsert to handle existing (main_event, datetime, ticker)
            stmt = insert(Ranges).values(insert_rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["main_event", "datetime", "ticker"],
                set_={
                    "future_range_1h": stmt.excluded.future_range_1h,
                    "future_range_3h": stmt.excluded.future_range_3h,
                    "future_range_6h": stmt.excluded.future_range_6h,
                    "future_range_24h": stmt.excluded.future_range_24h,
                },
            )
            self.sess.execute(stmt)
        self.sess.commit()
        print("[UPDATE] Ranges were populated")
