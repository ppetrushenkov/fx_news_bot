from typing import Literal

import numpy as np
import pandas as pd
import pandas_ta as ta

# from hurst import compute_Hc
# from scipy.stats import entropy
from IPython.display import display


def get_trading_session(ticker: str, hour: int, month: int) -> dict:
    """
    Determine trading sessions (Europe, USA, Asia) based on ticker, hour, and month,
    considering daylight saving time (DST).
    
    Args:
        ticker: Currency pair ticker (e.g., 'EURUSD', 'USDJPY')
        hour: Hour of the day (0-23, UTC)
        month: Month of the year (1-12)
    
    Returns:
        Dictionary with one-hot encoded sessions: {'europe': 0/1, 'usa': 0/1, 'asia': 0/1}
    """
    is_dst = 3 <= month <= 10
    
    europe_start = 7 if is_dst else 8
    europe_end = 16 if is_dst else 17
    
    usa_start = 13 if is_dst else 14
    usa_end = 22 if is_dst else 23
    
    asia_start = 0
    asia_end = 9
    
    sessions = {
        'europe': 1 if europe_start <= hour < europe_end else 0,
        'usa': 1 if usa_start <= hour < usa_end else 0,
        'asia': 1 if asia_start <= hour < asia_end else 0
    }
    
    return sessions


def get_base_and_quote_currency(pair: str):
    return pair[:-3], pair[-3:]


# def kaufman_efficiency_ratio(data: pd.DataFrame, window: int):
#     direction = data.close.diff(window).abs()
#     sum_range = data.trange.rolling(window).sum()
#     return direction / sum_range


def kaufman_efficiency_ratio(df, window=24):
    """Рассчитывает Kaufman Efficiency Ratio (ER)

    ER = |Цена(t) - Цена(t-n)| / Сумма(|Цена(i) - Цена(i-1)|)
    """
    # Абсолютное изменение цены за весь период (Direction)
    direction = (df["close"] - df["close"].shift(window)).abs()

    # Сумма абсолютных изменений между соседними барами (Volatility)
    bar_to_bar_change = (df["close"] - df["close"].shift(1)).abs()
    volatility = bar_to_bar_change.rolling(window=window).sum()

    # Защита от деления на ноль
    er = np.where(volatility != 0, direction / volatility, 0)
    return pd.Series(er, index=df.index)


def custom_efficiency_ratio(data: pd.DataFrame, window: int = 21):
    total_range = data.high.rolling(window).max() - data.low.rolling(window).min()
    sum_range = data.trange.rolling(window=window).sum()
    return sum_range / total_range
    # return total_range / sum_range


def noise_inside_the_bars(data: pd.DataFrame, eps: float = 1e-9):
    tr = data.trange
    body = (data['open'] - data['close']).abs()
    return body / tr + eps


def relative_range(bar_range: pd.Series, window: int):
    v_mean = bar_range.rolling(window*5).mean()
    v_std = bar_range.rolling(window*5).std()
    return (bar_range - v_mean) / (v_std + 1e-9)


def zscore_atr(atr: pd.Series, window: int):
    return (atr - atr.rolling(window).mean()) / atr.rolling(window).std()

def fast_vs_slow_atr(hi, lo, cl, fast_period: int = 21, slow_period: int = 100):
    fast_atr = ta.atr(hi, lo, cl, fast_period)
    slow_atr = ta.atr(hi, lo, cl, slow_period)
    return fast_atr / slow_atr


def relative_volume(volume: pd.Series, window: int):
    v_mean = volume.rolling(window).mean()
    v_std = volume.rolling(window).std()
    return (volume - v_mean) / (v_std + 1e-9)


def calculate_bb_width(data: pd.DataFrame, window: int):
    bbands = ta.bbands(
        data['close'], 
        length=window, 
        lower_std=3,
        upper_std=3,
        mamode=0
    )
    upper_band, lower_band = bbands[f'BBU_{window}_3_3'], bbands[f'BBL_{window}_3_3']
    return upper_band - lower_band


def normalized_bb_width(bb_width, atr):
    return bb_width / atr


def create_daily_data(data: pd.DataFrame) -> pd.DataFrame:
    daily_data = data \
        .set_index('datetime') \
        .resample('D').agg(
        {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
    ).dropna()
    return daily_data.reset_index()


# def add_daily_atr(data: pd.DataFrame, period=14) -> pd.DataFrame:
#     """
#     Добавляет ATR для каждого дня в датафрейме.
#     """
#     daily_data = create_daily_data(data)
#     data.reset_index(inplace=True)

#     daily_data['daily_atr'] = ta.atr(daily_data['high'], daily_data['low'], daily_data['close'], period).shift(1)
#     data = pd.merge(data, daily_data[['datetime', 'daily_atr']], on='datetime', how='left')
#     data['daily_atr'] = data['daily_atr'].ffill()
#     return data


def distance_from_sma_normalized(data: pd.DataFrame, period: int = 240):
    sma = ta.sma(data['close'], length=period)
    atr = ta.atr(data['high'], data['low'], data['close'], length=period)
    distance = np.where(data['close'] > sma, data['high'] - sma, sma - data['low'])
    return distance / atr


def rolling_shannon_entropy(window_data, bins=10):
        # Убираем NaN и строим гистограмму для получения частот
        counts, _ = np.histogram(window_data.dropna(), bins=bins)
        # Считаем энтропию Шеннона (base=2 для бит)
        return entropy(counts, base=2)


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling stdev of log returns — классическая оценка реализованной волатильности."""
    lr = np.log(close / close.shift(1))
    return lr.rolling(window).std()


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Range-based оценка волатильности (high/low), полезна на интрадее."""
    ratio = (high / low.replace(0, np.nan)).clip(lower=1e-12)
    hl = np.log(ratio)
    return np.sqrt(hl.pow(2).rolling(window).mean() / (4.0 * np.log(2.0)))


def add_time_features(data: pd.DataFrame, dt_col: str):
    data[dt_col] = pd.to_datetime(data[dt_col])
    data['week'] = data[dt_col].dt.isocalendar().week
    data['month'] = data[dt_col].dt.month
    data['quarter'] = data[dt_col].dt.quarter
    data['dayofweek'] = data[dt_col].dt.dayofweek
    data['day'] = data[dt_col].dt.day
    data['hour'] = data[dt_col].dt.hour
    return data


def generate_smart_lags(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Генерирует лаги и статистики, привязанные к торговым сессиям (8ч) и суткам (24ч/48ч).
    Включает Rolling Min/Max для определения экстремального сжатия.
    """
    res_df = df.copy()
    
    # 1. Сессионные и суточные лаги
    lags = [1, 8, 24, 48]
    for lag in lags:
        res_df[f'{feature_name}_lag_{lag}'] = res_df[feature_name].shift(lag)
        
    # 2. Скользящие средние (4 часа и 8 часов)
    res_df[f'{feature_name}_roll_mean_4'] = res_df[feature_name].rolling(window=4).mean()
    res_df[f'{feature_name}_roll_mean_8'] = res_df[feature_name].rolling(window=8).mean()
    
    # 3. Rolling Min и Max за суточный цикл (24 бара)
    res_df[f'{feature_name}_roll_min_24'] = res_df[feature_name].rolling(window=24).min()
    res_df[f'{feature_name}_roll_max_24'] = res_df[feature_name].rolling(window=24).max()
    
    # 4. УМНАЯ ФИЧА: Положение индикатора внутри суточного диапазона (от 0 до 1)
    # Показывает, насколько близко мы к историческому минимуму флэта за сутки
    denom = res_df[f'{feature_name}_roll_max_24'] - res_df[f'{feature_name}_roll_min_24']
    # Защита от деления на ноль, если индикатор был идеально плоским
    denom = np.where(denom == 0, 1e-6, denom) 
    
    res_df[f'{feature_name}_relative_position_24'] = (res_df[feature_name] - res_df[f'{feature_name}_roll_min_24']) / denom
    
    # 5. Скорость изменения за одну торговую сессию (8 часов)
    res_df[f'{feature_name}_diff_8'] = res_df[feature_name] - res_df[f'{feature_name}_lag_8']
    
    return res_df


def add_features(data: pd.DataFrame, period: int = 21, timeframe: Literal['hourly', 'daily'] = 'hourly'):
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)

    h, l, c = data["high"], data["low"], data["close"]

    data["trange"] = ta.true_range(h, l, c)
    data["atr"] = ta.atr(h, l, c, period)
    data["atr_normalized"] = data["atr"] / data["close"]
    # atr = data["atr"]

    if timeframe == 'hourly':
        # Daily data
        daily_data = create_daily_data(data)

        # Daily range
        daily_data['day_range_1'] = (daily_data['high'] - daily_data['low']).shift(1)
        daily_data['day_range_2'] = (daily_data['high'] - daily_data['low']).shift(2)
        daily_data['day_range_3'] = (daily_data['high'] - daily_data['low']).shift(3)
        daily_data['day_range_4'] = (daily_data['high'] - daily_data['low']).shift(4)

        # Daily pct_change
        daily_data['day_change_1'] = daily_data['close'].pct_change(1)
        daily_data['day_change_2'] = daily_data['close'].pct_change(2)
        daily_data['day_change_3'] = daily_data['close'].pct_change(3)
        daily_data['day_change_4'] = daily_data['close'].pct_change(4)

        # Daily ATR
        daily_data['prev_daily_atr'] = ta.atr(daily_data['high'], daily_data['low'], daily_data['close'], period).shift(1)

        # Merge daily data
        data = data.merge(daily_data.drop(['open', 'high', 'low', 'close'], axis=1), on='datetime', how='left')

        for c in ['day_range_1', 'day_range_2', 'day_range_3', 'day_range_4', 'day_change_1', 'day_change_2', 'day_change_3', 'day_change_4', 'prev_daily_atr']:
            data[c] = data[c].ffill()

    # +---------------- Start featuring ---------------+
    # 1. Efficiency Ratio
    data["kaufman_efficiency_ratio"] = kaufman_efficiency_ratio(data, window=period)
    data = generate_smart_lags(data, 'kaufman_efficiency_ratio')

    data["custom_efficiency_ratio"] = custom_efficiency_ratio(data, window=period)

    # 2. Wick-to-Body Ratio (Шум внутри баров)
    data["wick_ratio"] = noise_inside_the_bars(data)
    data = generate_smart_lags(data, 'wick_ratio')

    # 3. Relative Range
    data["relative_range"] = relative_range(data["trange"], window=period)
    data = generate_smart_lags(data, 'relative_range')

    # 4. Relative ATR
    data["zscore_atr"] = zscore_atr(data["atr"], window=period)
    data = generate_smart_lags(data, 'zscore_atr')

    # 5. ATR Ratio
    slow_period = 100 if timeframe == 'hourly' else period*2
    data["fast_vs_slow_atr"] = fast_vs_slow_atr(data['high'], data['low'], data['close'], fast_period=period, slow_period=slow_period)
    data = generate_smart_lags(data, 'fast_vs_slow_atr')
    
    # 6. BB width
    bb_width = calculate_bb_width(data, window=period)
    data["normalized_bb_width"] = normalized_bb_width(bb_width, data['atr'])
    data = generate_smart_lags(data, 'normalized_bb_width')

    # 7. Distance from SMA
    period_for_distance = 240 if timeframe == 'hourly' else period
    data["distance_from_sma"] = distance_from_sma_normalized(data, period=period_for_distance)
    data = generate_smart_lags(data, 'distance_from_sma')

    # 8. ADX (+ DI spread для направленного давления перед новостью)
    adx_df = ta.adx(data['high'], data['low'], data['close'], length=period)
    adx_df.columns = ['adx', 'adxr', 'dmp', 'dmn']
    data = pd.concat([data, adx_df], axis=1)
    data['di_spread'] = adx_df['dmp'] - adx_df['dmn']
    data = generate_smart_lags(data, 'adx')
    data = generate_smart_lags(data, 'di_spread')

    # 9. Текущий бар vs ATR (насколько уже «растянут» рынок)
    data["tr_over_atr"] = data["trange"] / (data["atr"] + 1e-9)
    data = generate_smart_lags(data, 'tr_over_atr')

    # 10. Realized volatility
    short_w = max(3, period // 3)
    long_w = max(short_w + 1, period * 3)
    rv_s = realized_volatility(data['close'], short_w)
    rv_l = realized_volatility(data['close'], long_w)
    data["realized_vol_short"] = rv_s
    data["realized_vol_long"] = rv_l
    data["realized_vol_ratio"] = rv_s / (rv_l + 1e-9)

    # 11. Parkinson vol (по high/low) и нормировка на ATR — доп. сигнал до новости
    pk = parkinson_volatility(data['high'], data['low'], period)
    data["parkinson_vol"] = pk
    data["parkinson_vol_over_atr"] = pk / (data["atr"] + 1e-9)

    # 12. Short vs Long ATR. Diapason squeeze and expansion
    atr_short_len = max(2, period // 2)
    atr_long_len = max(period + 1, period * 2)
    atr_short = ta.atr(data['high'], data['low'], data['close'], atr_short_len)
    atr_long = ta.atr(data['high'], data['low'], data['close'], atr_long_len)
    data["atr_short_over_long"] = atr_short / (atr_long + 1e-9)
    data = generate_smart_lags(data, 'atr_short_over_long')

    # 13. Недавняя «активность» — сумма |log-return| (импульс перед событием)
    data["abs_log_return_sum"] = (
        np.log(data['close'] / data['close'].shift(1)).abs().rolling(short_w).sum()
    )
    data = generate_smart_lags(data, 'abs_log_return_sum')

    # 14. Volatility of Volatility (VVoV)
    data["vvov"] = data["atr"].rolling(period).std() / (data["atr"].rolling(period).mean() + 1e-9)
    data = generate_smart_lags(data, 'vvov')

    # 15. Add time features
    data = add_time_features(data, 'datetime')

    # 16. Log return
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))

    # 17. Sessions
    if timeframe == 'hourly':
        session_data = data.apply(lambda row: get_trading_session(row['ticker'], row['hour'], row['month']), axis=1)
        data['is_europe'] = session_data.apply(lambda x: x['europe'])
        data['is_usa'] = session_data.apply(lambda x: x['usa'])
        data['is_asia'] = session_data.apply(lambda x: x['asia'])

        # 18. Sine time features
        data['sin_hour'] = np.sin(data['hour'] * 2*np.pi/24)

    return data
