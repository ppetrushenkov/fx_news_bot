import pandas as pd
import requests
from datetime import date, datetime, timedelta

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from bot.data_loader import get_events_till_next_sunday
from db.models import Base, Events, Prices


def convert_tz_to_moscow(dt: pd.Series):
    return pd.to_datetime(dt).dt.tz_convert('Europe/Moscow')


def custom_datetime_crop_to_closest_half_hour(dtime: pd.Series):
    dtime = pd.to_datetime(dtime)
    minutes = dtime.minute
    
    if (59 >= minutes >= 45) or (29 >= minutes >= 15):
        return dtime.ceil('30min')
    else:
        return dtime.floor('30min')


def fetch_economic_news(delta: int = 1) -> pd.DataFrame:
    """
    Fetch economic news for the current day from the TradingView economic calendar.
    """
    countries = ['US', 'FR', 'GB', 'EU', 'AU', 'DE', 'JP']  # TODO: add more countries
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=delta)).strftime('%Y-%m-%d')
    
    url = 'https://economic-calendar.tradingview.com/events'
    headers = {'Origin': 'https://in.tradingview.com'}
    payload = {
        'from': f'{today}T00:00:00.000Z',
        'to': f'{tomorrow}T00:00:00.000Z',
        'countries': ','.join(countries)
    }
    
    try:
        response = requests.get(url, headers=headers, params=payload)
        data = response.json()
        news_df = pd.DataFrame(data['result'])
        news_df.sort_values('date', inplace=True)
        
        if 'scale' not in news_df.columns:
            news_df['scale'] = None
        return news_df
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame()
    

def fetch_economic_news_for_today() -> pd.DataFrame:
    return fetch_economic_news(1)

def fetch_economic_news_for_two_days() -> pd.DataFrame:
    return fetch_economic_news(2)

def fetch_economic_news_for_a_week() -> pd.DataFrame:
    return fetch_economic_news(7)


def custom_datetime_crop(self, dtime: pd.Series, aggregation_time: str = '30min'):
    dtime = pd.to_datetime(dtime)
    minutes = dtime.minute

    if aggregation_time == '30min':
        if (59 >= minutes >= 45) or (29 >= minutes >= 15):
            return dtime.ceil('30min')
        else:
            return dtime.floor('30min')
    elif aggregation_time in ['1h', 'h']:
        if minutes >= 30:
            return dtime.ceil(aggregation_time)
        else:
            return dtime.floor(aggregation_time)


def get_datetime_list_to_set_scheduler(event_times: pd.Series, delta: str = '30min') -> pd.Series:
    """
    Takes the raw datetime column from TodayEconomicNews table
    and return the list of unique datetime values, when the events comes out
    and shift it on `delta` back.
    
    :param event_times: Pandas datetime column from TodayEconomicNews table
    :type event_times: pd.Series
    :param delta: The value that the datetime will move back to
    :type delta: str
    :return: The pd.Series of datetime values, shifted back on `delta` value
    :rtype: Series[Any]
    """
    crop_dates = event_times.apply(custom_datetime_crop_to_closest_half_hour)
    unique_dates = crop_dates.unique()
    unique_dates_shifted = unique_dates - pd.Timedelta(delta)
    return unique_dates_shifted

def get_max_date_from_table(table: Base):
    """
    Gets the max date from the passed table using SQLAlchemy.
    Returns a tuple: (min_date, max_date)
    """
    engine = create_engine('sqlite:///forex_news_bot.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        max_date = session.query(
            func.max(table.date)
        ).first()
        return max_date
    finally:
        session.close()


def get_next_sunday_date() -> date:
    """
    Returns the date of the next Sunday.
    If today is Sunday, returns the date of the following Sunday (next week).
    """
    today = datetime.now().date()
    # weekday(): Monday is 0, Sunday is 6
    days_ahead = 6 - today.weekday()
    if days_ahead == 0:
        # Today is Sunday, so add 7 days
        next_sunday = today + timedelta(days=7)
    else:
        # Otherwise, add days to reach next Sunday
        next_sunday = today + timedelta(days=days_ahead)
    return next_sunday


def get_prev_sunday_date() -> date:
    """
    Returns the date of the previous Sunday.
    If today is Sunday, returns the date of the previous Sunday (last week).
    """
    today = datetime.now().date()
    # Monday is 0, Sunday is 6
    days_since_sunday = (today.weekday() + 1) % 7
    prev_sunday = today - timedelta(days=days_since_sunday)
    return prev_sunday
