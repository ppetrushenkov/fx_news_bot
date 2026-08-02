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


# def get_economic_events(start_date: str, end_date: str, countries: List[str] = None) -> pd.DataFrame:
#     """
#     Get economic events using TradingView API

#     ### Args:
#         start_date (str): Start date
#         end_date (str): Till date
#         countries (list[str]): list of countries to filter by

#     ### Returns:
#         pd.DataFrame: DataFrame with economic events
#     """
#     if countries is None:
#         countries = ['US', 'EU', 'GB', 'JP', 'DE', 'FR', 'CA', 'AU', 'SG', 'SE']
#     countries = [country.upper() for country in countries]

#     start_date += 'T00:00:00'
#     end_date += 'T00:00:00'

#     url = 'https://economic-calendar.tradingview.com/events'
#     headers = {
#         'Origin': 'https://in.tradingview.com',
#         "User-Agent": "Mozilla/5.0",
#         "Accept": "application/json"
#     }
#     payload = {
#         'from': start_date + '.000Z',
#         'to': end_date + '.000Z',
#         'countries': ','.join(countries or []),
#     }

#     try:
#         data = requests.get(url, headers=headers, params=payload).json()
#         data = pd.DataFrame(data['result'])
#         data.sort_values('date', inplace=True)
            
#         if 'scale' not in data.columns:
#             data['scale'] = None

#         return data
    
#     except Exception as e:
#         print(f"Error fetching news: {e}")
#         return pd.DataFrame()

def get_economic_events(start_date: str, end_date: str, countries: List[str] = None) -> pd.DataFrame:
    """
    Get economic events using TradingView API with recursive splitting if API returns too many results.

    ### Args:
        start_date (str): Start date ("YYYY-MM-DD")
        end_date (str): End date ("YYYY-MM-DD")
        countries (list[str]): list of countries to filter by

    ### Returns:
        pd.DataFrame: DataFrame with economic events
    """
    # Helper to split the range by midpoint (by day)
    def split_date_range(start: str, end: str):
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        if end_dt <= start_dt:
            return None, None
        delta_days = (end_dt - start_dt).days
        mid_dt = start_dt + timedelta(days=delta_days // 2)
        mid = mid_dt.strftime("%Y-%m-%d")
        return (start, mid), (mid, end)

    if countries is None:
        countries = ['US', 'EU', 'GB', 'JP', 'DE', 'FR', 'CA', 'AU', 'SG', 'SE']
    countries = [country.upper() for country in countries]

    url = 'https://economic-calendar.tradingview.com/events'
    headers = {
        'Origin': 'https://in.tradingview.com',
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    def fetch_events(s, e, rec_level=0):
        start_ts = s + 'T00:00:00'
        end_ts = e + 'T00:00:00'
        payload = {
            'from': start_ts + '.000Z',
            'to': end_ts + '.000Z',
            'countries': ','.join(countries or []),
        }

        try:
            resp = requests.get(url, headers=headers, params=payload)
            resp.raise_for_status()
            data_json = resp.json()

            # Some calendar outages return missing 'result'
            if 'result' not in data_json:
                print(f"[get_economic_events] 'result' not in response for {s} to {e}")
                return pd.DataFrame()

            data = pd.DataFrame(data_json['result'])

            if not data.empty:
                data.sort_values('date', inplace=True)
                if 'scale' not in data.columns:
                    data['scale'] = None

            if len(data) >= 2000:
                range1, range2 = split_date_range(s, e)
                if not range1 or not range2 or range1 == range2:
                    print(f"[get_economic_events] Unable to split further {s} - {e}, returning {len(data)} rows")
                    return data

                print(f"[get_economic_events] Splitting range: {s} to {e} due to {len(data)} rows")
                df1 = fetch_events(range1[0], range1[1], rec_level=rec_level+1)
                df2 = fetch_events(range2[0], range2[1], rec_level=rec_level+1)
                combined = pd.concat([df1, df2], ignore_index=True)

                if 'id' in combined.columns:
                    combined = combined.drop_duplicates(subset='id')
                else:
                    combined = combined.drop_duplicates()
                return combined
            
            else:
                return data

        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()

    return fetch_events(start_date, end_date)


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
