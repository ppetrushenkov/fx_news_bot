"""
Parsing new data into database
"""
from typing import List
from datetime import datetime, date, timedelta
import pandas as pd
import requests

from config import Config
try:
    from twelvedata import TDClient
except Exception:  # pragma: no cover
    TDClient = None  # type: ignore
import pytz


def get_economic_events(start_date: str, end_date: str, countries: List[str] = None) -> pd.DataFrame:
    """
    Get economic events using TradingView API

    ### Args:
        start_date (str): Start date
        end_date (str): Till date
        countries (list[str]): list of countries to filter by

    ### Returns:
        pd.DataFrame: DataFrame with economic events
    """
    if countries is None:
        countries = ['US', 'EU', 'GB', 'JP', 'DE', 'FR', 'CA', 'AU', 'SG', 'SE']
    countries = [country.upper() for country in countries]

    start_date += 'T00:00:00'
    end_date += 'T00:00:00'

    url = 'https://economic-calendar.tradingview.com/events'
    headers = {
        'Origin': 'https://in.tradingview.com',
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    payload = {
        'from': start_date + '.000Z',
        'to': end_date + '.000Z',
        'countries': ','.join(countries or []),
    }

    try:
        data = requests.get(url, headers=headers, params=payload).json()
        data = pd.DataFrame(data['result'])
        data.sort_values('date', inplace=True)
            
        if 'scale' not in data.columns:
            data['scale'] = None

        return data
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame()


def get_historical_prices(client, ticker: str, interval: str = "1h", start_date: str = None, end_date: str = None, outputsize: int = None, **kwargs) -> pd.DataFrame:
    """
    Get historical prices for a given ticker using TwelveData API

    ### Args:
       client: TDClient object
       ticker (str): Ticker to get prices for
       interval (str, optional): Interval for the data. Defaults to "1h".
       outputsize (int, optional): Number of rows to retrieve. Defaults (max) to 5000.
    ### Returns:
        pd.DataFrame: DataFrame with historical prices for the ticker
    """
    if client is None:
        raise RuntimeError(
            "TwelveData client is not available. Install dependency `twelvedata` and set `TWELVE_API_KEY`."
        )

    bars = client.time_series(
        symbol=ticker,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        outputsize=outputsize,
        **kwargs
    )
    return bars.as_pandas()


def get_events_till_next_sunday() -> pd.DataFrame:
    """
    Get economic events for a week till next Sunday using the TradingView API.
    """
    now = datetime.now(tz=pytz.timezone(Config.TZ))
    days_until_sunday = 7 - now.isoweekday() if now.isoweekday() != 7 else 7
    next_sunday = now + timedelta(days=days_until_sunday)

    start_date = now.strftime("%Y-%m-%d")
    end_date = next_sunday.strftime("%Y-%m-%d")

    weekly_events = get_economic_events(start_date=start_date, end_date=end_date)
    return weekly_events


def get_events_for_today() -> pd.DataFrame:
    """
    Get economic events for today using the TradingView API.
    """
    now = datetime.now(tz=pytz.timezone(Config.TZ))
    start_date = now.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")
    
    today_events = get_economic_events(start_date, end_date)
    return today_events


def get_events_for_tomorrow() -> pd.DataFrame:
    """
    Get economic events for tomorrow using the TradingView API.
    """
    now = datetime.now(tz=pytz.timezone(Config.TZ))
    tomorrow = now + timedelta(days=1)
    
    start_date = tomorrow.strftime("%Y-%m-%d")
    end_date = tomorrow.strftime("%Y-%m-%d")
    
    today_events = get_economic_events(start_date, end_date)
    return today_events
