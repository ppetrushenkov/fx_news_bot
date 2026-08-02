import pandas as pd
import numpy as np
import pandas_ta as ta
from tqdm import tqdm

from numba import njit


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


# def get_future_range(data: pd.DataFrame, n_forward: int) -> pd.Series:
#     future_max = data.high.shift(-n_forward).rolling(n_forward).max()
#     future_min = data.low.shift(-n_forward).rolling(n_forward).min()
#     future_range = future_max - future_min
#     return future_range

def get_future_range(data: pd.DataFrame, n_forward: int) -> pd.Series:
    channel = ta.donchian(data.high, data.low, lower_length=n_forward, upper_length=n_forward, offset=-n_forward)
    channel.columns = ['lower', 'middle', 'upper']
    future_range = channel['upper'] - channel['lower']
    return future_range


def get_overall_future_range(data: pd.DataFrame, n_forward: int):
    return data['trange'].shift(-n_forward).rolling(n_forward).sum()


def get_future_custom_efficiency_ratio(data: pd.DataFrame, n_forward: int = 21) -> float:
    total_range = data.high.shift(-n_forward).rolling(n_forward).max() - data.low.shift(-n_forward).rolling(n_forward).min()
    rng = data.high - data.low
    sum_range = rng.shift(-n_forward).rolling(n_forward).sum()
    return sum_range / (total_range + 1e-6)
    # sum_range = data.trange.shift(-n_forward).rolling(n_forward).sum()


def get_big_wick_candles(
    data: pd.DataFrame,
    n_forward: int = 4,
    wick_ratio: float = 0.5,
    atr_coef: float = 1
) -> pd.Series:
    """1 на баре t, если среди следующих n_forward баров (t+1 … t+n_forward) есть свеча с
    true range > ATR и малым телом: |open−close| / trange < wick_ratio."""
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


def set_targets(data: pd.DataFrame, look_forward_bars: int):
    """
    Sets the targets for data. It sets the following targets:
    (VOLATILITY CLASSIFICATION)
    - `trg_bar_volatility_expansion_gradation` - возвращает максимальную степень волатильности
        для любого из следующих N баров;
    - `trg_is_greater_daily_atr` - возвращает True, если рендж следующих N свечей (в совокупности)
        формирует диапазон больше Дневного АТРа;  (СПОРНО)
    (RANGE REGRESSION)
    - `trg_future_range_n(hour)` - рендж следующих баров:
        - 1 час;
        - 3 часа;
        - 6 часов
    (CHAOS CLASSIFICATION)
    - `trg_big_doji` - возвращает True, если в ближайшие 2 часа будет большая Доджи (или пин бар) свеча.
        По сути свеча с отношением тела ко всей свече ниже 0.2 и размер свечи > АТРа;
    - `trg_dir_changes` - подсчет количества смен направлений после выхода новости
        (количество поочередных обновлений экстремумов).
        Так же может называться как поочередное выбивание стопов в обе стороны;
    - `trg_is_flat_flg` - минимальное значение KER (Kaufman Efficiency Ratio) на следующих N (от 8 до 24) баров.
        Должно быть меньше 0.1;
    - `trg_is_trend_flg` - максимальное значение KER на следующих N (от 8 до 24) баров. Должно быть больше 0.5;
    - `trg_is_chaos` - комплексная метка, означающая количество смен направлений (trg_dir_changes) > 1,
        волатильность > 3х АТР и одновременно должно быть и тренд и флэт.
    """
    # Prerequisites
    atr = data['atr']

    # er = custom_efficiency_ratio(data, window=period)
    er = data['custom_efficiency_ratio']
    er25 = er.quantile(0.25)
    er75 = er.quantile(0.75)
    # er90 = er.quantile(0.90)

    max_future_efficiency = er.shift(-look_forward_bars).rolling(look_forward_bars).max()
    min_future_efficiency = er.shift(-look_forward_bars).rolling(look_forward_bars).min()

    # Volatility Bar Gradation (TARGET #1)  DO NOT NEED THIS BECAUSE OF THE TARGET 2
    # volatility_gradation = get_bar_volatility_gradation(data, n_forward=look_forward_bars)

    # Volatility Ranges (TARGET #2)
    volatility_range_1h = get_future_range(data, n_forward=2)
    volatility_range_3h = get_future_range(data, n_forward=6)
    volatility_range_6h = get_future_range(data, n_forward=12)
    volatility_range_24h = get_future_range(data, n_forward=48)

    overall_volatility_1h = get_future_custom_efficiency_ratio(data, n_forward=2)
    overall_volatility_3h = get_future_custom_efficiency_ratio(data, n_forward=6)
    overall_volatility_6h = get_future_custom_efficiency_ratio(data, n_forward=12)
    overall_volatility_24h = get_future_custom_efficiency_ratio(data, n_forward=48)

    # Chaos classification
    ## Big wick candles
    big_wick_candles = get_big_wick_candles(data, n_forward=look_forward_bars, wick_ratio=0.4, atr_coef=1.5)
    dir_changes = calculate_direction_changes_fast(data, look_forward=look_forward_bars)

    # -----------------------------------------------------------------------------------
    # Set targets
    # -----------------------------------------------------------------------------------
    
    ## Volatility Classification
    # data['trg_volatility_expansion_gradation'] = volatility_gradation

    # Regression
    ## Whole range. Predict the whole distance, that the price will reach in the next N (1h, 3h, 6h, 24h) bars.
    data['trg_future_range_1h'] = clip_values(volatility_range_1h / atr, up_level=10)
    data['trg_future_range_3h'] = clip_values(volatility_range_3h / atr, up_level=15)
    data['trg_future_range_6h'] = clip_values(volatility_range_6h / atr, up_level=20)
    data['trg_future_range_24h'] = clip_values(volatility_range_24h / atr, up_level=25)

    ## Overall range. (SUM of ranges)
    data['trg_overall_future_range_1h'] = overall_volatility_1h
    data['trg_overall_future_range_3h'] = overall_volatility_3h
    data['trg_overall_future_range_6h'] = overall_volatility_6h
    data['trg_overall_future_range_24h'] = overall_volatility_24h

    # Chaos classification
    data['trg_big_doji'] = big_wick_candles
    data['trg_dir_changes'] = dir_changes
    data['trg_is_flat_flg'] = (max_future_efficiency >= er75).astype(int)
    data['trg_is_trend_flg'] = (min_future_efficiency <= er25).astype(int)

    data['trg_is_chaos_1h'] = (overall_volatility_1h >= overall_volatility_1h.quantile(0.75))
    data['trg_is_chaos_3h'] = (overall_volatility_3h >= overall_volatility_3h.quantile(0.75))
    data['trg_is_chaos_6h'] = (overall_volatility_6h >= overall_volatility_6h.quantile(0.75))
    data['trg_is_chaos_24h'] = (overall_volatility_24h >= overall_volatility_24h.quantile(0.75))

    # print('[INFO] Efficiency Ratio (Instrument: {instrument}): {er25}, {er75}'.format(instrument=data['instrument'].iloc[0], er25=er25, er75=er75))
    # data['trg_is_flat_flg'] = (min_future_efficiency <= er25).astype(int)
    # data['trg_is_trend_flg'] = (max_future_efficiency >= er75).astype(int)

    return data
