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
    def __init__(self):
        self.db = next(get_db)

    def select_all_to_df(self, db_model) -> pd.DataFrame:
        query = self.db.query(db_model)
        return pd.read_sql_query(query.statement, self.db.bind, params=query.statement.compile().params)

    def get_events_for_today(self) -> pd.DataFrame:
        events_today = self.select_all_to_df(TodayEconomicNews)
        return events_today

    def get_aggregated_event_info(self):
        events_today = self.get_events_for_today()
        events_today['cropped_datetime'] = self._custom_datetime_crop(events_today['date'])
        pass

    def get_prices(self):
        # TODO: Add Twelve Data support retrieving
        pass

    def unite_events_and_prices(self):
        # TODO: Unite news and prices
        pass

    def preprocess_events_and_prices(self):
        pass

    def _convert_tz_to_moscow(self, dt: pd.Series):
        return pd.to_datetime(dt).dt.tz_convert('Europe/Moscow')

    def _custom_datetime_crop(self, dtime: pd.Series):
        dtime = pd.to_datetime(dtime)
        minutes = dtime.minute
        
        if (59 >= minutes >= 45) or (29 >= minutes >= 15):
            return dtime.ceil('30min')
        else:
            return dtime.floor('30min')



class Predictor:
    def __init__(self):
        self.volatility_clf_model = None
        self.range_predictor_model = None
        self.chaos_clf_model = None

    def get_model_predictions(self):
        pass


