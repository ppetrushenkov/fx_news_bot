import pandas as pd
import requests
from datetime import datetime, timedelta


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
        return news_df
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame()
    

def fetch_economic_news_for_today() -> pd.DataFrame:
    return fetch_economic_news(1)

def fetch_economic_news_for_two_days() -> pd.DataFrame:
    return fetch_economic_news(2)

def fetch_economic_news_for_a_week() -> pd.DataFrame:
    return fetch_economic_news(5)
    

