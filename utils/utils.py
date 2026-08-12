import numpy as np

import pandas as pd


def df_standardize_event_dates(df: pd.DataFrame) -> pd.DataFrame:
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


def df_standardize_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
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


