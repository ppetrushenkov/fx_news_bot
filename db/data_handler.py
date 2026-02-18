from config import Config

from db.database import get_db
from db.models import TodayEconomicNews, Base
from bot.utils import custom_datetime_crop

from twelvedata import TDClient

import pandas as pd



class DataRetriever:
    """
    This class simply the work with retrieving the data from different sources. 
    This class do:
    - retrieve events from the database;
    - aggregate events by specified datetime (e.g. 30min or hour);
    - fetch prices from different sources (TwelveData and maybe more);
    """
    def __init__(self, aggregation_time: str = '30min'):
        self.db = next(get_db)
        self.aggregation_time = aggregation_time
        self.td = TDClient(apikey=Config.TWELVE_API)
        self.supported_tickers = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD']

    def select_all_to_df(self, db_model) -> pd.DataFrame:
        """Select all records from the database and return them as a pandas DataFrame.

        Args:
            db_model (Base): The database model to select from.
        
        Returns:
            pd.DataFrame: The selected records as a pandas DataFrame.
        """
        query = self.db.query(db_model)
        return pd.read_sql_query(query.statement, self.db.bind, params=query.statement.compile().params)

    def get_events_for_today(self) -> pd.DataFrame:
        """Get events for today."""
        events_today = self.select_all_to_df(TodayEconomicNews)
        return events_today

    def get_aggregated_events_for_now(self, event_time: str) -> pd.DataFrame:
        # TODO: Write aggregation function
        events_today = self.get_events_for_today()  # May be take whole dataset, not just today
        events_today['cropped_datetime'] = custom_datetime_crop(events_today['date'])
        
        upcoming_events = events_today[events_today['cropped_datetime'] == event_time]
        
        aggregated_news = None
        return aggregated_news

    def get_last_prices(self) -> pd.DataFrame:
        """Fetch last prices for supported tickers from Twelve Data source"""
        # TODO: Define, how may outputsize do I need for predictions
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


class Predictor:
    def __init__(self):
        self.volatility_clf_model = None
        self.range_predictor_model = None
        self.chaos_clf_model = None

    def get_model_predictions(self):
        pass


