import pandas as pd
import numpy as np
import pandas_ta as ta

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
    daily_data.reset_index(inplace=True)

    daily_data['daily_atr'] = ta.atr(daily_data['high'], daily_data['low'], daily_data['close'], period).shift(1)
    data = pd.merge(data, daily_data[['time', 'daily_atr']], on='time', how='left')
    data['daily_atr'] = data['daily_atr'].ffill()
    return data

def get_bar_volatility_gradation(data: pd.DataFrame, atr: pd.Series, daily_atr: pd.Series, period: int = 24, n_forward: int = 4) -> pd.Series:
    """Возвращает максимальную степень волатильности для любого из следующих N баров.
    Волатильность должна быть больше 2‑х АТР текущего таймфрейма.
    Градация:
    - 0 - меньше 66% перцентиля АТР;
    - 1 - АТР от 66% до 75% перцентиля;
    - 2 - АТР от 75% до 90%;
    - 3 - АТР от 90% перцентиля до дневного АТРа;
    - 4 - АТР больше дневного АТРа;
    """
    q66 = atr.rolling(window=period).quantile(0.66)
    q75 = atr.rolling(window=period).quantile(0.75)
    q90 = atr.rolling(window=period).quantile(0.90)

    tr = data.trange.values
    datr = daily_atr.values

    grads = np.where(
        tr > datr, 4,
            np.where(
                datr > tr >= q90, 3,
                    np.where(
                        q90 > tr >= q75, 2,
                            np.where(
                                q75 > tr > q66, 1,
                                    0
                )
            )
        )
    )
    grads = pd.Series(grads).shift(-n_forward).rolling(n_forward).max()
    return grads

def get_future_range(data: pd.DataFrame, n_forward: int = 4) -> pd.Series:
    future_max = data['high'].shift(-n_forward).rolling(n_forward).max()
    future_min = data['low'].shift(-n_forward).rolling(n_forward).min()
    future_range = future_max - future_min
    return future_range

def get_big_wick_candles(data: pd.DataFrame, n_forward: int = 4, wick_ratio: float = 0.2) -> pd.Series:
    body = (data['Open'] - data['Close']).abs()
    ratio = body / data.trange  # add forward looking
    return (ratio < wick_ratio).astype(int)

def set_targets(data: pd.DataFrame, look_forward_bars: int = 5, period: int = 24):
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
        По сути свеча с отношением тела ко всей свече ниже 0.2;
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
    atr = ta.atr(data['high'], data['low'], data['close'], length=period)
    data = add_daily_atr(data, period)  # Calculate Daily ATR
    daily_atr = data['daily_atr']

    # Volatility Bar Gradation (TARGET #1)
    volatility_gradation = get_bar_volatility_gradation(data, atr, daily_atr, period, n_forward=look_forward_bars)

    # Volatility Ranges
    volatility_range_1h = get_future_range(data, n_forward=2)
    volatility_range_3h = get_future_range(data, n_forward=6)
    volatility_range_6h = get_future_range(data, n_forward=12)

    # Chaos classification
    ## Big wick candles
    big_wick_candles = get_big_wick_candles(data, n_forward = 4, wick_ratio = 0.2)

    # Set targets
    data['trg_bar_volatility_expansion_gradation'] = volatility_gradation
    data['trg_is_greater_daily_atr'] = (data['high'].shift(-look_forward_bars).rolling(look_forward_bars).max() > daily_atr)
    data['trg_future_range_1h'] = get_future_range(data, n_forward=2)
    data['trg_future_range_3h'] = get_future_range(data, n_forward=6)
    data['trg_future_range_6h'] = get_future_range(data, n_forward=12)
    data['trg_big_doji'] = big_wick_candles
    data['trg_dir_changes'] = (data['high'].diff().abs() > 0).cumsum()
    data['trg_is_flat_flg'] = (ta.ker(data['open'], data['close'], data['high'], data['low']) < 0.1)
    data['trg_is_trend_flg'] = (ta.ker(data['open'], data['close'], data['high'], data['low']) > 0.5)
    data['trg_is_chaos'] = ((data['high'].diff().abs() > 0).cumsum() > 1) & (volatility_gradation > 3) & ((data['trg_is_flat_flg'] | data['trg_is_trend_flg']))

    return data
