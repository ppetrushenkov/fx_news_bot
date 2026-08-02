import pandas as pd
import numpy as np
import pandas_ta as ta
from tqdm import tqdm

from numba import njit


# TODO: !!! Добавить Swing Failure (Ложный пробой)


def is_swing_failure(data: pd.DataFrame, n_forward: int = 24):
    return None


def get_bar_volatility_gradation(data: pd.DataFrame, n_forward: int = 4) -> pd.Series:
    """Максимальный коэффициент k ∈ {0,…,n_forward} на баре t.

    Для каждого k ≥ 1 смотрим **следующие k баров** (t+1 … t+k): совокупный диапазон
    ``max(high) − min(low)`` по этим барам. Если он **строго больше** ``ATR[t] * k``,
    то k достижим. Итоговая метка — **наибольший** такой k; если ни один не выполнен — 0.

    Нужна колонка ``data.atr`` (ATR на текущем баре), выровненная по индексу.
    """
    atr_v = data.atr
    
    n = len(data)
    grads = np.zeros(n, dtype=np.int64)

    max_high = data['high'].shift(-n_forward).rolling(n_forward).max()
    min_low = data['low'].shift(-n_forward).rolling(n_forward).min()
    bar_range = max_high - min_low

    for k in range(1, 7 + 1):  # 7 - GRADATIONS
        grads = np.where(bar_range > atr_v * k, k, grads)

    return pd.Series(grads, index=data.index)


def get_future_range(data: pd.DataFrame, n_forward: int) -> pd.Series:
    channel = ta.donchian(data.high, data.low, lower_length=n_forward, upper_length=n_forward, offset=-n_forward)
    channel.columns = ['lower', 'middle', 'upper']
    future_range = channel['upper'] - channel['lower']
    return future_range


def extremum_breakout(data: pd.DataFrame, channel_period: int, future_n: int) -> pd.Series:
    """Return boolean series if extremum breakout happened in future n hours"""
    channel = ta.donchian(data.high, data.low, lower_length=channel_period, upper_length=channel_period)
    channel.columns = ['lower', 'middle', 'upper']

    future_max = data['high'].shift(-future_n).rolling(future_n).max()
    future_min = data['low'].shift(-future_n).rolling(future_n).min()

    return (future_max > channel['upper']) & (future_min < channel['lower'])


def get_future_range_from_now(data: pd.DataFrame, n_forward: int) -> pd.Series:
    channel = ta.donchian(data.high, data.low, lower_length=n_forward, upper_length=n_forward, offset=-n_forward)
    if channel is None:
        return pd.Series(np.nan, index=data.index)
    channel.columns = ['lower', 'middle', 'upper']
    future_range = channel['upper'] - channel['lower']
    return future_range


def get_future_range_from_next_bar(data: pd.DataFrame, n_forward: int) -> pd.Series:
    channel = ta.donchian(data.high, data.low, lower_length=n_forward, upper_length=n_forward, offset=-n_forward-1)
    if channel is None:
        return pd.Series(np.nan, index=data.index)
    channel.columns = ['lower', 'middle', 'upper']
    future_range = channel['upper'] - channel['lower']
    return future_range


def get_overall_future_range(data: pd.DataFrame, n_forward: int):
    return data['trange'].shift(-n_forward).rolling(n_forward).sum()


def get_future_custom_efficiency_ratio(data: pd.DataFrame, n_forward: int = 21) -> pd.Series:
    total_range = data.high.shift(-n_forward).rolling(n_forward).max() - data.low.shift(-n_forward).rolling(n_forward).min()
    rng = data.high - data.low
    sum_range = rng.shift(-n_forward).rolling(n_forward).sum()
    return sum_range / (total_range + 1e-6)
    # sum_range = data.trange.shift(-n_forward).rolling(n_forward).sum()


def get_future_adx(adx: pd.Series, n_forward: int = 24) -> pd.Series:
    return adx.shift(-n_forward).rolling(n_forward).apply(lambda x: x.max() if any(x > 30) else x.min())


def get_big_wick_candles(
    data: pd.DataFrame,
    n_forward: int = 4,
    wick_ratio: float = 0.5,
    atr_coef: float = 1
) -> pd.Series:
    """Returns a binary series indicating whether the candle has a big wick.

    A candle is considered to have a big wick if the ratio of the body to the total range (trange)
    is less than the given wick_ratio and the trange is greater than the atr multiplied by atr_coef.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing 'open', 'high', 'low' and 'close' columns.
        n_forward (int, optional): Number of forward bars to consider for the big wick condition. Defaults to 4.
        wick_ratio (float, optional): The ratio threshold for considering a wick as big. Defaults to 0.5.
        atr_coef (float, optional): The coefficient to multiply the ATR by when checking if trange is large enough. Defaults to 1.

    Returns:
        pd.Series: A binary series where 1 indicates a candle with a big wick and 0 indicates otherwise.
    """
    tr = data.trange
    atr = data.atr

    body = (data["open"] - data["close"]).abs()
    ratio = body / tr.replace(0, np.nan)
    per_bar = ((tr > atr * atr_coef) & (ratio < wick_ratio)).astype(int)
    stacked = pd.concat([per_bar.shift(-k) for k in range(1, n_forward + 1)], axis=1)
    return stacked.max(axis=1).fillna(0).astype(int)


def calculate_direction_changes(data: pd.DataFrame, look_forward: int = 6):
    def get_dir_changes(chunk: pd.DataFrame):
        chunk.reset_index(inplace=True)
        changes = 0
        cmin, cmax = chunk.loc[0, 'low'], chunk.loc[0, 'high']
        cdir = 'bull' if (chunk.loc[0, 'close'] - chunk.loc[0, 'open']) > 0 else 'bear'
        
        for i, row in chunk.iterrows():
            cl, ch = row.low, row.high
            if cdir == 'bear' and ch > cmax:
                cmax = ch
                cdir = 'bull'
                changes += 1
                
            elif cdir == 'bull' and cl < cmin:
                cmin = cl
                cdir = 'bear'
                changes += 1   

        return changes

    changes = []
    
    for i in tqdm(range(len(data) - look_forward + 1)):
        chunk = data.iloc[i:i+look_forward, :]
        chg = get_dir_changes(chunk)
        changes.append(chg)

    return pd.Series(changes)


@njit
def get_dir_changes_numba(open_, high, low, close):
    changes = 0
    
    cmin = low[0]
    cmax = high[0]
    
    if close[0] - open_[0] > 0:
        cdir = 1  # bull
    else:
        cdir = -1  # bear

    for i in range(len(open_)):
        cl = low[i]
        ch = high[i]

        if cdir == -1 and ch > cmax:
            cmax = ch
            cdir = 1
            changes += 1

        elif cdir == 1 and cl < cmin:
            cmin = cl
            cdir = -1
            changes += 1

    return changes


@njit
def calculate_direction_changes_numba(open_, high, low, close, look_forward):
    n = len(open_)
    result = np.zeros(n - look_forward + 1)

    for i in range(n - look_forward + 1):
        result[i] = get_dir_changes_numba(
            open_[i:i+look_forward],
            high[i:i+look_forward],
            low[i:i+look_forward],
            close[i:i+look_forward]
        )

    return result


def calculate_direction_changes_fast(data: pd.DataFrame, look_forward: int = 6):
    open_ = data['open'].values
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values

    res = calculate_direction_changes_numba(open_, high, low, close, look_forward)
    
    return pd.Series(res, index=data.index[:len(res)])


def clip_values(ser: pd.Series, up_level: float = None, down_level: float = None):
    return ser.clip(lower=down_level, upper=up_level)


def classify_trend_or_flat(adx: pd.Series, n_forward: int, trend_thresh: int, flat_thresh: int):
    max_adx = adx.shift(-n_forward).rolling(n_forward).max()
    min_adx = adx.shift(-n_forward).rolling(n_forward).min()

    return np.where(max_adx > trend_thresh, 'Trend', np.where(min_adx < flat_thresh, 'Flat', 'None'))


def set_targets(data: pd.DataFrame, look_forward_bars: int):
    # Prerequisites
    atr = data['atr']

    # er = data['custom_efficiency_ratio']
    # er25 = er.quantile(0.25)
    # er75 = er.quantile(0.75)

    # max_future_efficiency = er.shift(-look_forward_bars).rolling(look_forward_bars).max()
    # min_future_efficiency = er.shift(-look_forward_bars).rolling(look_forward_bars).min()

    # Volatility Ranges (TARGET #2)
    future_range_1h = get_future_range_from_next_bar(data, n_forward=1)
    future_range_3h = get_future_range_from_next_bar(data, n_forward=3)
    future_range_6h = get_future_range_from_next_bar(data, n_forward=6)
    future_range_24h = get_future_range_from_next_bar(data, n_forward=24)

    future_range_1h /= atr
    future_range_3h /= atr
    future_range_6h /= atr
    future_range_24h /= atr

    future_range_1h = clip_values(future_range_1h, up_level=10)
    future_range_3h = clip_values(future_range_3h, up_level=15)
    future_range_6h = clip_values(future_range_6h, up_level=20)
    future_range_24h = clip_values(future_range_24h, up_level=25)

    # Overall volatility (sum of ranges)
    overall_volatility_1h = get_future_custom_efficiency_ratio(data, n_forward=1)
    overall_volatility_3h = get_future_custom_efficiency_ratio(data, n_forward=3)
    overall_volatility_6h = get_future_custom_efficiency_ratio(data, n_forward=6)
    overall_volatility_24h = get_future_custom_efficiency_ratio(data, n_forward=24)

    ## Big wick candles
    big_wick_candles = get_big_wick_candles(data, n_forward=6, wick_ratio=0.3, atr_coef=1.5)
    dir_changes = calculate_direction_changes_fast(data, look_forward=look_forward_bars)
    is_extremum_breakout = extremum_breakout(data, channel_period=24, future_n=look_forward_bars)
    is_regime = classify_trend_or_flat(data['adx'], n_forward=look_forward_bars, trend_thresh=25, flat_thresh=15)


    # Set targets into DataFrame
    data['trg_future_range_1h'] = future_range_1h
    data['trg_future_range_3h'] = future_range_3h
    data['trg_future_range_6h'] = future_range_6h
    data['trg_future_range_24h'] = future_range_24h

    ## Overall range. (SUM of ranges)
    data['trg_overall_future_range_1h'] = overall_volatility_1h
    data['trg_overall_future_range_3h'] = overall_volatility_3h
    data['trg_overall_future_range_6h'] = overall_volatility_6h
    data['trg_overall_future_range_24h'] = overall_volatility_24h

    # Chaos classification
    data['trg_big_doji'] = big_wick_candles
    data['trg_dir_changes'] = dir_changes
    data['trg_is_extremum_breakout'] = is_extremum_breakout
    data['trg_regime'] = is_regime

    # data['trg_is_flat_flg'] = (max_future_efficiency >= er75).astype(int)
    # data['trg_is_trend_flg'] = (min_future_efficiency <= er25).astype(int)

    data['trg_is_chaos_1h'] = (overall_volatility_1h >= overall_volatility_1h.quantile(0.5)) & (future_range_1h >= 1) & (dir_changes > 1)
    data['trg_is_chaos_3h'] = (overall_volatility_3h >= overall_volatility_3h.quantile(0.6)) & (future_range_3h >= 1) & (dir_changes > 1)
    data['trg_is_chaos_6h'] = (overall_volatility_6h >= overall_volatility_6h.quantile(0.6)) & (future_range_6h >= 2) & (dir_changes > 1)
    data['trg_is_chaos_24h'] = (overall_volatility_24h >= overall_volatility_24h.quantile(0.75)) & (future_range_24h >= 3) & (dir_changes > 1)

    for trg in ['trg_future_range_1h', 'trg_future_range_3h', 'trg_future_range_6h', 'trg_future_range_24h']:
        data[trg + '_log'] = data[trg].apply(lambda x: np.log1p(x))

    return data
