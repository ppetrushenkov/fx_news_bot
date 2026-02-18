from db.models import TodayEconomicNews
from bot.scheduler import populate_database, fetch_economic_news
from db.database import create_tables, get_db
import pandas as pd


db = next(get_db())

def custom_datetime_crop_to_closest_half_hour(dtime: pd.Series):
    dtime = pd.to_datetime(dtime)
    minutes = dtime.minute
    
    if (59 >= minutes >= 45) or (29 >= minutes >= 15):
        return dtime.ceil('30min')
    else:
        return dtime.floor('30min')
    

def get_times_to_set_scheduler(event_times: pd.Series, delta: str = '30min') -> pd.Series:
    crop_dates = event_times.apply(custom_datetime_crop_to_closest_half_hour)
    unique_dates = crop_dates.unique()
    unique_dates_shifted = unique_dates - pd.Timedelta(delta)
    return unique_dates_shifted
    

def main():
    """Fetch unique times for today's news"""
    query = db.query(TodayEconomicNews)
    df = pd.read_sql_query(query.statement, db.bind, params=query.statement.compile().params)
    unique_dates_shifted = get_times_to_set_scheduler(df['date'])
    print(unique_dates_shifted)

if __name__ == '__main__':
    main()

