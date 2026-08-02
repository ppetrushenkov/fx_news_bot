"""
Calculate features for each economic event and write it into DB
"""
# from db.data_handler import DBHandler
from typing import Optional
from datetime import datetime, timedelta, timezone, date as date_type
import pandas as pd
import numpy as np


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _next_sunday_utc(d: Optional[date_type] = None) -> date_type:
    d = d or _utc_now().date()
    days_ahead = 6 - d.weekday()  # Monday=0 .. Sunday=6
    return d + timedelta(days=days_ahead if days_ahead != 0 else 7)


def _df_standardize_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize event dates to naive UTC DateTime."""
    if df.empty:
        return df
    out = df.copy()

    for dt_col in ("date", "referenceDate"):
        if dt_col in out.columns:
            s = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)  # naive UTC
            out[dt_col] = s.apply(lambda x: None if pd.isna(x) else x.to_pydatetime())

    for int_col in ("id", "importance"):
        if int_col in out.columns:
            s = pd.to_numeric(out[int_col], errors="coerce")
            out[int_col] = s.apply(lambda x: None if pd.isna(x) else int(x))

    out = out.dropna(subset=["date"])
    out["id"] = out["id"].astype(int)

    out.rename(columns={'id': 'event_id'}, inplace=True)

    # out = out.where(pd.notna(out), None)
    out = out.replace({np.nan: None})
    
    return out


def _df_standardize_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.reset_index()

    # TwelveData sometimes returns index name as 'datetime'/'time'
    if "datetime" in out.columns:
        dt_col = "datetime"
    elif "time" in out.columns:
        dt_col = "time"
    else:
        dt_col = out.columns[0]

    out["datetime"] = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime"])
    out["datetime"] = out["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)

    for c in ["open", "high", "low", "close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = None

    out["ticker"] = ticker
    return out[["datetime", "ticker", "open", "high", "low", "close"]].dropna(subset=["datetime"])



def get_most_important_events(title: pd.Series):
    title = str(title).upper()
    
    # Priority mappings for specific events
    if 'BALANCE OF TRADE' in title:
        return 'Balance_of_Trade'
    if 'CPI' in title or 'INFLATION RATE' in title or 'PPI' in title:
        if 'CORE' in title:
            return 'Core_Inflation_rate'
        return 'Inflation_rate'
    if 'INTEREST RATE DECISION' in title or 'DEPOSIT FACILITY RATE' in title:
        return 'Interest_Rate_Decision'
    if 'NON FARM PAYROLLS' in title or 'NONFARM PAYROLLS' in title:
        return 'NFP'
    if 'GDP' in title:
        return 'GDP'
    if 'FOMC' in title:
        return 'FOMC'
    if 'PMI' in title:
        if 'MANUFACTURING' in title:
            return 'PMI_Manufacturing'
        if 'SERVICES' in title:
            return 'PMI_Services'
        return 'PMI'
    if 'RETAIL SALES' in title:
        return 'Retail_Sales'
    if 'UNEMPLOYMENT RATE' in title:
        return 'Unemployment_rate'
    
    return None
    