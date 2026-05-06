import numpy as np
import pandas as pd
import pandas_ta as ta

# from hurst import compute_Hc
from scipy.stats import entropy


def get_base_and_quote_currency(pair: str):
    return pair[:-3], pair[-3:]


def kaufman_efficiency_ratio(data: pd.DataFrame, window: int):
    direction = data.close.diff(window).abs()
    sum_range = data.trange.rolling(window).sum()
    return direction / sum_range


def custom_efficiency_ratio(data: pd.DataFrame, window: int = 21) -> float:
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


def relative_atr(atr: pd.Series, window: int):
    return (atr - atr.rolling(window*5).mean()) / atr.rolling(window*5).std()


def relative_volume(volume: pd.Series, window: int):
    v_mean = volume.rolling(window*5).mean()
    v_std = volume.rolling(window*5).std()
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


def add_daily_atr(data: pd.DataFrame, period=14) -> pd.DataFrame:
    """
    Добавляет ATR для каждого дня в датафрейме.
    """
    data.set_index('time', inplace=True)
    daily_data = data.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
        }
    ).dropna()

    data.reset_index(inplace=True)
    daily_data.reset_index(inplace=True)

    daily_data['daily_atr'] = ta.atr(daily_data['high'], daily_data['low'], daily_data['close'], period).shift(1)
    data = pd.merge(data, daily_data[['time', 'daily_atr']], on='time', how='left')
    data['daily_atr'] = data['daily_atr'].ffill()
    return data


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


def add_features(data: pd.DataFrame, period: int = 21):
    data['time'] = pd.to_datetime(data['time'], utc=True)

    # atr = ta.atr(h, l, c, period)
    h, l, c = data["high"], data["low"], data["close"]
    data = add_daily_atr(data, period)  # Calculate Daily ATR

    data["trange"] = ta.true_range(h, l, c)
    data["atr"] = ta.atr(h, l, c, period)
    atr = data["atr"]

    # 1. Kaufman Efficiency Ratio
    data["kaufman_efficiency_ratio"] = kaufman_efficiency_ratio(data, window=period)

    # 2. Custom Range efficiency
    data["custom_efficiency_ratio"] = custom_efficiency_ratio(data, window=period)

    # 3. Wick-to-Body Ratio (Шум внутри баров)
    data["wick_ratio"] = noise_inside_the_bars(data)

    # 4. Relative Range
    data["relative_range"] = relative_range(data["trange"], window=period)

    # 5. Relative ATR
    data["relative_atr"] = relative_atr(atr, window=period)

    # 6. BB width
    bb_width = calculate_bb_width(data, window=period)
    data["normalized_bb_width"] = normalized_bb_width(bb_width, atr)

    # 7. Distance from SMA
    data["distance_from_sma"] = distance_from_sma_normalized(data, period=240)

    # 8. ADX (+ DI spread для направленного давления перед новостью)
    adx_df = ta.adx(h, l, c, length=period)
    data = pd.concat([data, adx_df], axis=1)
    dmp_cols = [col for col in adx_df.columns if col.startswith("DMP_")]
    dmn_cols = [col for col in adx_df.columns if col.startswith("DMN_")]
    if dmp_cols and dmn_cols:
        data["di_spread"] = adx_df[dmp_cols[0]] - adx_df[dmn_cols[0]]

    # 9. Текущий бар vs ATR (насколько уже «растянут» рынок)
    data["tr_over_atr"] = data["trange"] / (atr + 1e-9)

    # 10. Реализованная волатильность и всплеск относительно более длинного окна
    short_w = max(3, period // 3)
    long_w = max(short_w + 1, period * 3)
    rv_s = realized_volatility(c, short_w)
    rv_l = realized_volatility(c, long_w)
    data["realized_vol_short"] = rv_s
    data["realized_vol_long"] = rv_l
    data["realized_vol_ratio"] = rv_s / (rv_l + 1e-9)

    # 11. Parkinson vol (по high/low) и нормировка на ATR — доп. сигнал до новости
    pk = parkinson_volatility(h, l, period)
    data["parkinson_vol"] = pk
    data["parkinson_vol_over_atr"] = pk / (atr + 1e-9)

    # 12. Режим ATR: короткий / длинный (сжатие vs расширение диапазона)
    atr_short_len = max(2, period // 2)
    atr_long_len = max(period + 1, period * 2)
    atr_short = ta.atr(h, l, c, atr_short_len)
    atr_long = ta.atr(h, l, c, atr_long_len)
    data["atr_short_over_long"] = atr_short / (atr_long + 1e-9)

    # 13. Недавняя «активность» — сумма |log-return| (импульс перед событием)
    data["abs_log_return_sum"] = (
        np.log(c / c.shift(1)).abs().rolling(short_w).sum()
    )

    # 14. Volatility of Volatility (VVoV)
    data["vvov"] = atr.rolling(period).std() / (atr.rolling(period).mean() + 1e-9)

    # 15. Add time features
    data = add_time_features(data, 'time')

    # 16. Log return
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))

    return data

def get_target_range(data: pd.DataFrame, atr: pd.Series, future_n: int = 8) -> pd.Series:
    """Return range value in future 4 hours"""
    future_max = data['high'].shift(-future_n).rolling(future_n).max()
    future_min = data['low'].shift(-future_n).rolling(future_n).min()
    future_range = future_max - future_min
    return future_range / atr


def stopped_out_by_volatility_category(data, future_n: int = 8):  # 8 is 4 hours (8 bars in M30 timeframe)
    """Return the category of volatility.
    There is 5 levels of volatility:
    0 - no 24 bar min/max values was broke out simultaniously
    Returns a Series with values 0-4 indicating volatility category.
    """
    ranges = [24, 48, 96, 144]  # 'half_day', '1_day', '2_day', '3_day'
    
    future_max = data['high'].shift(-future_n).rolling(future_n).max()
    future_min = data['low'].shift(-future_n).rolling(future_n).min()

    # Initialize result Series with zeros
    result = pd.Series(0, index=data.index)
    
    levels = {
        24: 1,
        48: 2,
        96: 3,
        144: 4
    }

    # Check each range level, starting from smallest to largest
    # The largest matching range will overwrite previous values
    for v in ranges:
        prev_max_high = data['high'].rolling(v).max()
        prev_min_low = data['low'].rolling(v).min()
        
        # Create boolean mask for rows where condition is True
        condition = (future_max > prev_max_high) & (future_min < prev_min_low)
        
        # Update result for rows where condition is True
        result.loc[condition] = levels[v]
    
    return result


def atr_volatility_expansion(data, atr, stop_mult, future_n: int = 8):
    current_close = data['close']
    future_max = data['high'].shift(-future_n).rolling(future_n).max()
    future_min = data['low'].shift(-future_n).rolling(future_n).min()
    future_range = future_max - future_min

    # Find stop levels
    upper_stop = current_close + (atr * stop_mult)
    lower_stop = current_close - (atr * stop_mult)

    # Targets
    target_volatility_expantion = future_range > (atr * 3)
    target_whipsaw = (future_max > upper_stop) & (future_min < lower_stop)
    target_chaos = target_volatility_expantion | target_whipsaw
    return target_chaos.astype(int)


# def label_future_chaos(data: pd.DataFrame, N=5, stop_mult=1.5):
#     """
#     Генерирует сигналы о будущем хаосе на N баров вперед.
#     stop_mult: множитель ATR для определения "выноса стопов"
#     """
#     atr = ta.ATR(data['high'], data['low'], data['close'], N*3)

#     # Classify big volatility in future (if future range in 4 hours will be greater ATR, then return 1)
#     data['target_is_volatile'] = is_volatile_in_future(data, atr, future_n=N)

#     # Classify volatility category
#     data['target_volatility_category'] = stopped_out_by_volatility_category(data, future_n=N)
    
#     # Target range in 4 hours
#     data[f'target_range_normalized'] = get_target_range(data, atr, future_n=N)

#     # Target Bool if 
#     data['target_chaos'] = atr_volatility_expansion(data, atr, stop_mult, future_n=N)

#     return data