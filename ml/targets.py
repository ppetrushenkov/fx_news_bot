import pandas as pd
import numpy as np
import pandas_ta as ta
from tqdm import tqdm
from scipy.stats import linregress

from numba import njit


# def get_slope(y):
#     x = np.arange(len(y))
#     return np.polyfit(x, y, 1)[0]


# def get_percentage_angle(y):
#     x = np.arange(len(y))
#     slope, _ = np.polyfit(x, y, 1)
    
#     # Calculate slope as a percentage of the average price in this window
#     mean_price = np.mean(y)
#     slope_pct = slope / mean_price if mean_price != 0 else 0
    
#     # Convert to degrees (np.arctan2 takes rise, run)
#     # Multiplying by a scale factor helps map percentage to a visual 45° angle
#     scale_factor = 1000  # Adjust this to match your visual preference
#     angle_rad = np.arctan2(slope_pct * scale_factor, 1)
    
#     return np.degrees(angle_rad)


# def get_normalized_angle(y):
#     y_min = np.min(y)
#     y_max = np.max(y)
    
#     # If the window is completely flat, the angle is 0
#     if y_max == y_min:
#         return 0.0
    
#     # Scale Y to be between 0 and 1
#     y_norm = (y - y_min) / (y_max - y_min)
    
#     # Scale X to be between 0 and 1 across the 24 bars
#     x_norm = np.linspace(0, 1, len(y))
    
#     # Calculate slope on the normalized square grid
#     slope, _ = np.polyfit(x_norm, y_norm, 1)
    
#     # Convert slope directly to degrees
#     return np.degrees(np.arctan(slope))


def get_angle(y, target_move=1.0):
    start_price = y[0]
    end_price = y[-1]
    
    price_change = end_price - start_price
    
    return np.degrees(np.arctan(price_change / target_move))


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
    channel = ta.donchian(data.high, data.low, lower_length=channel_period, upper_length=channel_period, offset=-1)
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


# def classify_trend_or_flat(adx: pd.Series, n_forward: int, trend_thresh: int, flat_thresh: int):
#     max_adx = adx.shift(-n_forward).rolling(n_forward).max()
#     min_adx = adx.shift(-n_forward).rolling(n_forward).min()

#     return np.where(max_adx > trend_thresh, 'Trend', np.where(min_adx < flat_thresh, 'Flat', 'None'))

def classify_trend_or_flat(adx: pd.Series, n_forward: int, trend_thresh: int, flat_thresh: int):
    next_forward_adx = adx.shift(-n_forward)
    return np.where(next_forward_adx > trend_thresh, 'Trend', np.where(next_forward_adx < flat_thresh, 'Flat', 'None'))


# def detect_sfp(df: pd.DataFrame, period: int, n_forward: int) -> pd.Series:
#     """Находит паттерн Swing Failure Pattern (SFP) в DataFrame.

#     Parameters:
#     df (pd.DataFrame): Данные с колонками 'High', 'Low', 'Close'
#     period (int): Период для расчета ценового канала (Donchian Channel)
#     n_forward (int): На сколько баров вперед смотреть для подтверждения паттерна

#     Returns:
#     pd.Series: Серия из 1 (паттерн обнаружен) и 0 (паттерна нет)
#     """
#     # Проверяем наличие необходимых колонок
#     required_cols = ["high", "low", "close"]
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"DataFrame должен содержать колонку {col}")

#     # Создаем копию, чтобы не портить исходный DataFrame
#     df_copy = df.copy()

#     # Расчет каналов по ПРЕДЫДУЩИМ барам (сдвиг на 1 бар назад)
#     df_copy["Upper_Channel"] = (
#         df_copy["high"].shift(1).rolling(window=period).max()
#     )
#     df_copy["Lower_Channel"] = (
#         df_copy["low"].shift(1).rolling(window=period).min()
#     )
#     df_copy["Mid_Channel"] = (
#         df_copy["Upper_Channel"] + df_copy["Lower_Channel"]
#     ) / 2

#     # Массив для записи сигналов (заполнен нулями)
#     signals = np.zeros(len(df_copy), dtype=int)

#     # Итерируемся по строкам. Ограничиваем цикл, чтобы не выйти за пределы массива при заглядывании вперед
#     for i in range(len(df_copy) - n_forward + 1):
#         uc = df_copy["Upper_Channel"].iloc[i]
#         lc = df_copy["Lower_Channel"].iloc[i]
#         mid = df_copy["Mid_Channel"].iloc[i]

#         # Если канал еще не сформировался (недостаточно истории), пропускаем бар
#         if pd.isna(uc) or pd.isna(lc):
#             continue

#         # Вырезаем окно "вперед" размером n_forward баров, начиная с текущего i
#         forward_window = df_copy.iloc[i : i + n_forward]

#         # Последняя цена закрытия в этом окне forward
#         last_close = forward_window["close"].iloc[-1]

#         # 1. Медвежий SFP (Пробой верхнего канала, закрытие внутри, итог — ниже середины)
#         bearish_breakout = (
#             (forward_window["high"] > uc) & (forward_window["close"] <= uc)
#         ).any()
#         bearish_signal = bearish_breakout and (last_close < mid)

#         # 2. Бычий SFP (Пробой нижнего канала, закрытие внутри, итог — выше середины)
#         bullish_breakout = (
#             (forward_window["low"] < lc) & (forward_window["close"] >= lc)
#         ).any()
#         bullish_signal = bullish_breakout and (last_close > mid)

#         # Если выполнилось хотя бы одно условие, фиксируем точку зарождения паттерна (бар i)
#         if bearish_signal or bullish_signal:
#             signals[i] = 1

#     return pd.Series(signals, index=df.index, name="SFP_Signal")


# @njit
# def _detect_sfp_numba_core(
#     high: np.ndarray,
#     low: np.ndarray,
#     close: np.ndarray,
#     period: int,
#     n_forward: int,
# ) -> np.ndarray:
#     """Ускоренное ядро расчета SFP с помощью Numba."""
#     n = len(high)
#     signals = np.zeros(n, dtype=np.int32)

#     # Начинаем с period, так как до этого момента история канала не сформирована
#     # И заканчиваем так, чтобы n_forward не выходил за границы массива
#     for i in range(period, n - n_forward + 1):

#         # 1. Расчет границ канала по ПРЕДЫДУЩИМ period барам (от i-period до i-1 включительно)
#         uc = high[i - period : i].max()
#         lc = low[i - period : i].min()
#         mid = (uc + lc) / 2.0

#         # Флаги пробития для текущего окна forward
#         bearish_breakout = False
#         bullish_breakout = False

#         # 2. Проверяем бары в окне n_forward (от i до i + n_forward - 1)
#         for j in range(i, i + n_forward):
#             if high[j] > uc and close[j] <= uc:
#                 bearish_breakout = True
#             if low[j] < lc and close[j] >= lc:
#                 bullish_breakout = True

#         # Последняя цена закрытия в окне n_forward
#         last_close = close[i + n_forward - 1]

#         # 3. Финальная проверка условий
#         if bearish_breakout and last_close < mid:
#             signals[i] = 1
#         elif bullish_breakout and last_close > mid:
#             signals[i] = 1

#     return signals

@njit
def _detect_sfp_combined_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    n_forward: int,
) -> np.ndarray:
    n = len(high)
    signals = np.zeros(n, dtype=np.int32)

    for i in range(period, n - n_forward + 1):
        # Расчет канала по предыдущим барам
        uc = high[i - period : i].max()
        lc = low[i - period : i].min()
        mid = (uc + lc) / 2.0

        bearish_breakout = False
        bullish_breakout = False

        # Сканируем окно n_forward полностью для каждого типа паттерна
        for j in range(i, i + n_forward):

            # --- МЕДВЕЖИЙ SFP (Ложный пробой ВВЕРХ) ---
            # Вариант 1: Классический (пробой хвостом, закрытие внутри)
            if high[j] > uc and close[j] <= uc:
                bearish_breakout = True

            # Вариант 2: 2-барный (закрытие выше, но следующий закрывается внутри)
            # Проверяем, что j + 1 не выходит за границы общего массива данных
            elif j + 1 < n and close[j] > uc and close[j + 1] <= uc:
                bearish_breakout = True

            # --- БЫЧИЙ SFP (Ложный пробой ВНИЗ) ---
            # Вариант 1: Классический (пробой хвостом, закрытие внутри)
            if low[j] < lc and close[j] >= lc:
                bullish_breakout = True

            # Вариант 2: 2-барный (закрытие ниже, но следующий закрывается внутри)
            elif j + 1 < n and close[j] < lc and close[j + 1] >= lc:
                bullish_breakout = True

        # Последняя цена закрытия в окне forward
        last_close = close[i + n_forward - 1]

        # Финальная проверка: ушла ли цена за середину канала
        if bearish_breakout and last_close < mid:
            signals[i] = 1
        elif bullish_breakout and last_close > mid:
            signals[i] = 1

    return signals


def detect_sfp_fast(
    df: pd.DataFrame, period: int, n_forward: int
) -> pd.Series:
    """Публичная обертка, принимающая DataFrame и возвращающая pd.Series."""
    # Извлекаем numpy-массивы (Numba работает с ними мгновенно)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    # Запускаем JIT-компилированное ядро
    signals = _detect_sfp_combined_core(high, low, close, period, n_forward)

    return pd.Series(signals, index=df.index, name="SFP_Signal")


# def calc_adx(high, low, close, period=14):
#     adx = ta.adx(high, low, close, length=period)
#     adx.columns = ['adx', 'adxr', 'dmp', 'dmn']
#     return adx['adx']

# def calc_atr(high, low, close, period=14):
#     atr = ta.atr(high, low, close, length=period)
#     return atr


def get_direction_1day(data: pd.DataFrame, horizon: int = 24) -> pd.Series:
    er = data['kaufman_efficiency_ratio']
    
    # 1. Рассчитываем будущее среднее значение ER (строго вперед)
    past_mean_er = er.rolling(window=horizon).mean()
    future_mean_er = past_mean_er.shift(-(horizon - 1))
    
    # Квантили для ER
    er25 = future_mean_er.quantile(0.25)
    er75 = future_mean_er.quantile(0.75)

    # 2. Добавляем фильтр по ATR на БУДУЩИЕ 24 бара
    # Вычисляем чистое движение цены от текущего момента на 24 бара вперед
    future_movement = data['close'].shift(-(horizon - 1)) - data['close']
    
    # Требуем, чтобы движение составило хотя бы 70% от вчерашнего дневного ATR
    # (Вы можете изменить 0.70 на 0.50 или 1.00 в зависимости от жесткости фильтра)
    atr_condition = future_movement.abs() > (data['prev_daily_atr'] * 0.70)
    
    # 3. Логика разметки: Тренд подтверждается сильным ER И достаточным ходом цены
    is_trend = (future_mean_er > er75) & atr_condition
    is_flat = future_mean_er < er25

    # Сборка финальных меток
    labels = np.where(
        is_trend, "Trend",
        np.where(is_flat, "Flat", "None")
    )
    
    return pd.Series(labels, index=data.index)


def get_direction_2day(data: pd.DataFrame, horizon: int = 48) -> pd.Series:
    # 1. Авто-калибровка шага на основе истории (остается по прошлым данным)
    historical_changes = data['close'].diff(horizon).abs()
    calibrated_target = historical_changes.quantile(0.75)

    # 2. Сначала считаем угол «назад» (за прошлые 48 баров)
    past_slope = data['close'].rolling(window=horizon).apply(
        get_angle, 
        raw=True, 
        kwargs={'target_move': calibrated_target}
    )
    
    # Сдвигаем угол назад по индексу, чтобы для текущего бара это стал угол БУДУЩИХ 48 баров
    # (horizon - 1), так как текущий бар включается в это окно как первый элемент
    future_slope = past_slope.shift(-(horizon - 1))

    # 3. Изменение цены строго за БУДУЩИЕ 48 баров (от текущего Close до Close через 48 баров)
    future_movement = data['close'].shift(-(horizon - 1)) - data['close']
    atr_condition = future_movement.abs() > data['prev_daily_atr']
    
    # 4. Логика разметки: теперь ВСЕ условия смотрят только в будущее
    is_trend = (future_slope.abs() >= 35) & atr_condition
    is_flat = future_slope.abs() < 20

    # Сборка финальных меток
    labels = np.where(
        is_trend, "Trend",
        np.where(is_flat, "Flat", "None")
    )

    return pd.Series(labels, index=data.index)


def set_targets(data: pd.DataFrame, look_forward_bars: int):
    # Prerequisites
    atr = data['atr']
    adx_1day = ta.adx(data.high, data.low, data.close, length=24)
    adx_1day.columns = ['adx', 'adxr', 'dmp', 'dmn']
    adx_2day = ta.adx(data.high, data.low, data.close, length=48)
    adx_2day.columns = ['adx', 'adxr', 'dmp', 'dmn']

    adx_1day_25 = adx_1day['adx'].quantile(0.25)
    adx_1day_75 = adx_1day['adx'].quantile(0.75)

    adx_2day_25 = adx_2day['adx'].quantile(0.25)
    adx_2day_75 = adx_2day['adx'].quantile(0.75)

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

    # Volatility Pct Changes (TARGET #2)
    # future_pct_change_1h = get_future_pct_change(data, n_forward=1)

    # Overall volatility (sum of ranges)
    overall_volatility_1h = get_future_custom_efficiency_ratio(data, n_forward=1)
    overall_volatility_3h = get_future_custom_efficiency_ratio(data, n_forward=3)
    overall_volatility_6h = get_future_custom_efficiency_ratio(data, n_forward=6)
    overall_volatility_24h = get_future_custom_efficiency_ratio(data, n_forward=24)

    ## Big wick candles
    big_wick_candles = get_big_wick_candles(data, n_forward=6, wick_ratio=0.3, atr_coef=1.5)
    dir_changes = calculate_direction_changes_fast(data, look_forward=look_forward_bars)
    is_extremum_breakout = extremum_breakout(data, channel_period=48, future_n=look_forward_bars)

    direction_1day = get_direction_1day(data, horizon=24)
    direction_2day = get_direction_2day(data, horizon=48)
    
    # is_regime = classify_trend_or_flat(data['adx'], n_forward=look_forward_bars, trend_thresh=25, flat_thresh=15)
    # is_regime_1day = classify_trend_or_flat(adx_1day['adx'], n_forward=24, trend_thresh=25, flat_thresh=15)
    # is_regime_2day = classify_trend_or_flat(adx_2day['adx'], n_forward=48, trend_thresh=25, flat_thresh=15)
    # is_regime = label_regime(data, period=48, n_forward=48)


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
    data['trg_direction_1day'] = direction_1day
    data['trg_direction_2day'] = direction_2day
    data['trg_is_sfp'] = detect_sfp_fast(data, period=24, n_forward=8)

    # data['trg_is_flat_flg'] = (max_future_efficiency >= er75).astype(int)
    # data['trg_is_trend_flg'] = (min_future_efficiency <= er25).astype(int)

    # data['trg_is_chaos_1h'] = (overall_volatility_1h >= overall_volatility_1h.quantile(0.5)) & (future_range_1h >= 1) & (dir_changes > 1)
    data['trg_is_chaos_3h'] = (dir_changes > 2) & (future_range_3h >= 2)
    data['trg_is_chaos_6h'] = (overall_volatility_6h >= overall_volatility_6h.quantile(0.75)) & (future_range_6h >= 2) & (dir_changes > 1)
    data['trg_is_chaos_24h'] = (overall_volatility_24h >= overall_volatility_24h.quantile(0.75)) & (future_range_24h >= 3) & (dir_changes > 1)

    for trg in ['trg_future_range_1h', 'trg_future_range_3h', 'trg_future_range_6h', 'trg_future_range_24h']:
        data[trg + '_log'] = data[trg].apply(lambda x: np.log1p(x))

    return data
