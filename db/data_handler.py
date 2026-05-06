from datetime import datetime
from typing import Literal

from config import Config

from db.database import SessionLocal
from db.models import Events

from sqlalchemy import func, select

from bot.utils import custom_datetime_crop

from twelvedata import TDClient

import pandas as pd



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
        self.td = TDClient(apikey=Config.TWELVE_API)
        self.supported_tickers = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD']

# +---------------  WRITE  --------------------+ #
    def write_into(self, df: pd.DataFrame, table_name: str, if_exists: Literal['replace', 'append'] = 'replace'):
        """Write DataFrame to database."""
        try:
            df.to_sql(table_name, self.sess.bind, if_exists=if_exists, index=False)
            print(f"Added {len(df)} news items to database")

        except Exception as e:
            print(f"Error populating database: {e}")
            self.sess.rollback()
        
        # finally:
        #     self.sess.close()
    
    # def populate_aggregation_table(self, events_today: pd.DataFrame):
    #     """Populate aggregation table with events for today."""
    #     events_today['cropped_datetime'] = custom_datetime_crop(events_today['date'])
    #     aggregated_events = events_today.groupby('cropped_datetime').agg({'event': ' '.join})
        
    #     self.write_into(aggregated_events, TodayEventsAggregated.__tablename__)

# +---------------  GET  --------------------+ #
    def get_all_records_from_table(self, db_model) -> pd.DataFrame:
        """Select all records from the database and return them as a pandas DataFrame."""
        query = self.sess.query(db_model)
        return pd.read_sql_query(query.statement, self.sess.bind, params=query.statement.compile().params)

    def get_events_for_today(self) -> pd.DataFrame:
        """Get events for today."""
        events_today = self.get_all_records_from_table(Events)
        return events_today
    
    # def get_aggregated_events_for_coming_hour(self) -> pd.DataFrame:
    #     """Return aggregated events for the next hour."""
    #     now = datetime.now()
    #     hour_from_now = now + pd.Timedelta(hours=1)
        
    #     query = self.sess.query(TodayEventsAggregated).filter(
    #         TodayEventsAggregated.agg_time >= now,
    #         TodayEventsAggregated.agg_time <= hour_from_now
    #     )
    #     df = pd.read_sql_query(query.statement, self.sess.bind, params=query.statement.compile().params)
    #     return df

    def get_last_prices(self) -> pd.DataFrame:
        """Fetch last prices for supported tickers from Twelve Data source"""
        # Fetch data for all supported tickers
        prices_list = []
        for ticker in self.supported_tickers:
            try:
                ts = self.td.time_series(
                    symbol=ticker,
                    interval="30min",
                    outputsize=15
                )
                df = ts.as_pandas()
                df['ticker'] = ticker
                df.reset_index(inplace=True)
                # Ensure column names match what the predictor expects
                df.columns = ['time', 'open', 'high', 'low', 'close', 'ticker']
                prices_list.append(df)
            except Exception as e:
                print(f"Error fetching prices for {ticker}: {e}")
                
        if not prices_list:
            return pd.DataFrame()
            
        return pd.concat(prices_list, ignore_index=True)

# +---------------  PREPROCESSING  --------------------+ #
    def unite_events_and_prices(self):
        # TODO: Unite aggregated news with prices
        merged = pd.merge(self.aggregated_events, self.prices, on='dt', how='left')
        pass
        return None
    
    def get_data_for_ml_prediction(self):
        aggregated_events = self.get_aggregated_events_for_now()
        prices = self.get_last_prices()
        united = self.unite_events_and_prices(aggregated_events, prices)
        preprocessed = self.preprocess_events_and_prices(united)
        return preprocessed

    def preprocess_events_and_prices(self):
        # TODO: Add price and news featuring. Need to set feature lists
        pass

    def check_the_market(self):
        """Generate predictions for each ticker in supported list.
        
        The list of supported tickers: 
        ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY' 'USDCAD', 'AUDUSD', 'NZDUSD']
        """
        predictions = []
        data = self.get_data_for_ml_prediction()
        for ticker in data['tickers']:
            ticker_data = data[data['ticker'] == ticker]
            volatility_cls = self.classify_volatility(ticker_data)
            predicted_range = self.predict_range(ticker_data)
            is_chaos_prediction = self.classify_chaos(ticker_data)
            predictions.append({
                'Ticker': ticker,
                'Volatility class': volatility_cls,
                'Min predicted range': predicted_range[0],
                'Max predicted range': predicted_range[1],
                'Chaos probability': is_chaos_prediction
            })
        return predictions
    
    def is_need_to_update_today_events(self) -> bool:
        """Return True if current date is not the same as the min date """
        q = select(func.min(Events.date))
        min_date = self.sess.execute(q).scalar()  # TODO: check this method or use .scalars().all()[0]
        min_date = pd.to_datetime(min_date).date()
        date_now = datetime.now().date()
        if min_date == date_now:
            return False
        return True
    