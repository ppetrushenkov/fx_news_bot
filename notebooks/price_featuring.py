import numpy as np
import pandas as pd
import talib as ta
from hurst import compute_Hc
from skyfield.elementslib import period
from tqdm.auto import tqdm

tqdm.pandas(desc="Processing DataFrame")


def kaufman_efficiency_ratio(data: pd.DataFrame, window: int):
    change = data['close'].diff(window).abs()
    volatility = data['close'].diff().abs().rolling(window).sum()
    # volatility = data.trange.abs().rolling(window).sum()
    return change / volatility


def custom_range_efficiency(data: pd.DataFrame, window: int = 3) -> float:
    total_range = data.high.rolling(window).max() - data.low.rolling(window).min()
    sum_range = data.trange.rolling(window=window).sum()
    return total_range / sum_range


def noise_inside_the_bars(data: pd.DataFrame):
    body = (data['open'] - data['close']).abs()
    bar_range = data['high'] - data['low']
    return body / (bar_range + 1e-9)


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
    upper_band, middle_band, lower_band = ta.BBANDS(
        data['close'].values,
        timeperiod=window, 
        nbdevup=3,
        nbdevdn=3,
        matype=0
    )
    return upper_band - lower_band

def normalized_bb_width(bb_width, atr):
    return bb_width / atr



def rolling_hurst(series: pd.Series, window=100) -> pd.Series:
    def get_h(sub_series):
        if len(sub_series) < 50:
            return np.nan
        H, _, _ = compute_Hc(sub_series, kind='price', simplified=True)
        return H
    
    return series.rolling(window=window).apply(get_h, raw=False)


def calculate_chaos_features(data: pd.DataFrame, window: int = 14):
    atr = ta.ATR(data['high'], data['low'], data['close'], window)

    data['trange'] = ta.TRANGE(data['high'], data['low'], data['close'])

    # 1. Kaufman Efficiency Ratio
    data['kaufman_efficiency_ratio'] = kaufman_efficiency_ratio(data, window=window)
    data['kaufman_efficiency_ratio_future'] = data['kaufman_efficiency_ratio'].shift(-window-1)

    # 2. Custom Range efficiency
    data['custom_range_efficiency'] = custom_range_efficiency(data, window=window)
    data['custom_range_efficiency_future'] = data['custom_range_efficiency'].shift(-window-1)

    # 3. Wick-to-Body Ratio (Шум внутри баров)
    data['wick_ratio'] = noise_inside_the_bars(data)

    # 4. Relative Range
    data['relative_range'] = relative_range(data['trange'], window=window)
    
    # 5. Relative ATR
    data['relative_atr'] = relative_atr(atr, window=window)

    # 6. Anomaly volume
    data['relative_volume'] = relative_volume(data['volume'], window)

    # 7. BB width
    bb_width = calculate_bb_width(data, window=window)
    data['normalized_bb_width'] = normalized_bb_width(bb_width, atr)

    # 8. Price to Volume
    data['price_volume_ratio'] = data.trange / (data.volume + 1e-9)

    # 8. Hurst Exponent
    # data['hurst'] = rolling_hurst(data['close'], window=100)
    
    return data

# +============== TARGETS ==============+

def is_volatile_in_future(data: pd.DataFrame, atr: pd.Series, future_n: int = 8, coef: float = 1) -> pd.Series:
    """Return 1 if the future range will be greater than ATR * coef, else 0."""
    future_max = data['high'].shift(-future_n).rolling(future_n).max()
    future_min = data['low'].shift(-future_n).rolling(future_n).min()
    future_range = future_max - future_min
    return (future_range > atr * coef).astype(int)


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
    target_volatility_expantion = future_range > (atr * stop_mult)
    target_whipsaw = (future_max > upper_stop) & (future_min < lower_stop)
    target_chaos = target_volatility_expantion | target_whipsaw
    return target_chaos.astype(int)


def add_daily_atr(data: pd.DataFrame, period=14) -> pd.DataFrame:
    """
    Добавляет ATR для каждого дня в датафрейме.
    """
    daily_data = data.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
        }
    ).dropna()
    data.reset_index(inplace=True)
    daily_data['date'] = daily_data.index.date
    daily_data['daily_atr'] = ta.ATR(daily_data['high'], daily_data['low'], daily_data['close'], period).shift(1)
    return pd.merge(data, daily_data[['date', 'daily_atr']], on='date', how='left')


def label_future_chaos(data: pd.DataFrame, N=5, stop_mult=0.5, period: int = 14):
    """
    Генерирует сигналы о будущем хаосе на N баров вперед.
    stop_mult: множитель ATR для определения "выноса стопов"
    """
    # atr = ta.ATR(data['high'], data['low'], data['close'], N*3)

    data['time'] = pd.to_datetime(data['time'])
    data['date'] = data['time'].dt.date
    data.set_index('time', inplace=True)

    # Calculate Daily ATR
    data = add_daily_atr(data, period)
    # print(data.head(1))

    # Classify big volatility in future (if future range in 4 hours will be greater ATR, then return 1)
    data['target_is_volatile'] = is_volatile_in_future(data, data['daily_atr'], future_n=N, coef=stop_mult)

    # Classify volatility category
    data['target_volatility_category'] = stopped_out_by_volatility_category(data, future_n=N)
    
    # Target range in 4 hours
    data[f'target_range_normalized'] = get_target_range(data, data['daily_atr'], future_n=N)

    # Target Bool if 
    data['target_is_chaos'] = atr_volatility_expansion(data, data['daily_atr'], stop_mult, future_n=N)

    return data