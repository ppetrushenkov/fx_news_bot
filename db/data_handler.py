from datetime import datetime

from config import Config

from db.database import SessionLocal
from db.models import TodayEconomicNews, TodayEventsAggregated

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
    def __init__(self):
        self.db = SessionLocal()
        self.td = TDClient(apikey=Config.TWELVE_API)
        self.supported_tickers = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD']

# +---------------  WRITE  --------------------+ #
    def write_table_to_database(self, df: pd.DataFrame, table_name: str):
        """Write DataFrame to database."""
        try:
            df.to_sql(table_name, self.db.bind, if_exists='replace', index=False)
            print(f"Added {len(df)} news items to database")

        except Exception as e:
            print(f"Error populating database: {e}")
            self.db.rollback()
        
        finally:
            self.db.close()
    
    def populate_aggregation_table(self, events_today: pd.DataFrame):
        """Populate aggregation table with events for today."""
        events_today['cropped_datetime'] = custom_datetime_crop(events_today['date'])
        aggregated_events = events_today.groupby('cropped_datetime').agg({'event': ' '.join})
        
        self.write_table_to_database(aggregated_events, TodayEventsAggregated.__tablename__)

# +---------------  GET  --------------------+ #
    def get_all_records_from_table(self, db_model) -> pd.DataFrame:
        """Select all records from the database and return them as a pandas DataFrame."""
        query = self.db.query(db_model)
        return pd.read_sql_query(query.statement, self.db.bind, params=query.statement.compile().params)

    def get_events_for_today(self) -> pd.DataFrame:
        """Get events for today."""
        events_today = self.get_all_records_from_table(TodayEconomicNews)
        return events_today
    
    def get_aggregated_events_for_today(self) -> pd.DataFrame:
        """Return events for today."""
        events_today = self.get_all_records_from_table(TodayEventsAggregated)
        return events_today

    def get_aggregated_events_for_now(self, event_time: str) -> pd.DataFrame:
        agg_news = self.get_aggregated_events_for_today()
        upcoming_aggregated_events = agg_news[agg_news['cropped_datetime'] == event_time]
        return upcoming_aggregated_events

    def get_last_prices(self) -> pd.DataFrame:
        """Fetch last prices for supported tickers from Twelve Data source"""
        # TODO: Define, how may output size do I need for predictions
        ts = self.td.time_series(
            symbol=self.supported_tickers,
            interval="30min",
            outputsize=15
        )
        # Returns pandas.DataFrame
        prices = ts.as_pandas()
        prices.reset_index(inplace=True)
        prices.columns = ['ticker', 'datetime', 'open', 'high', 'low', 'close']
        return prices

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
    
    def check_if_need_to_update_today_events(self) -> bool:
        """Return True if current date is not the same as the min date """
        q = select(func.min(TodayEconomicNews.date))
        min_date = self.db.execute(q).scalar()  # TODO: check this method or use .scalars().all()[0]
        min_date = pd.to_datetime(min_date).date()
        date_now = datetime.now().date()
        if min_date == date_now:
            return False
        return True
    