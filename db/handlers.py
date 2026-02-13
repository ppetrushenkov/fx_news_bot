from config import Config
from db.database import get_db
from db.models import TodayEconomicNews, Base
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

    def select_all_to_df(self, db_model) -> pd.DataFrame:
        query = self.db.query(db_model)
        return pd.read_sql_query(query.statement, self.db.bind, params=query.statement.compile().params)

    def get_events_for_today(self) -> pd.DataFrame:
        events_today = self.select_all_to_df(TodayEconomicNews)
        return events_today

    def get_aggregated_events_for_now(self, event_time: str) -> pd.DataFrame:
        # TODO: Write aggregation function
        events_today = self.get_events_for_today()  # May be take whole dataset, not just today
        events_today['cropped_datetime'] = self._custom_datetime_crop(events_today['date'])
        
        upcoming_events = events_today[events_today['cropped_datetime'] == event_time]
        
        aggregated_news = None
        return aggregated_news

    def get_last_prices(self) -> pd.DataFrame:
        # TODO: Add Twelve Data support retrieving
        prices = None
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

    def _convert_tz_to_moscow(self, dt: pd.Series):
        return pd.to_datetime(dt).dt.tz_convert('Europe/Moscow')

    def _custom_datetime_crop(self, dtime: pd.Series):
        dtime = pd.to_datetime(dtime)
        minutes = dtime.minute

        if self.aggregation_time == '30min':
            if (59 >= minutes >= 45) or (29 >= minutes >= 15):
                return dtime.ceil('30min')
            else:
                return dtime.floor('30min')
        elif self.aggregation_time == 'h':
            if minutes >= 30:
                return dtime.ceil(self.aggregation_time)
            else:
                return dtime.floor(self.aggregation_time)



class Predictor:
    def __init__(self):
        self.volatility_clf_model = None
        self.range_predictor_model = None
        self.chaos_clf_model = None

    def get_model_predictions(self):
        pass


