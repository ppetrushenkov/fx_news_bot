import pandas as pd
import numpy as np

from ml.source_categories import SOURCE_CATS
from ml.event_categories import CLASS_KEYWORDS, EVENT_WEIGHTS_D


def classify_by_dict(d: dict, source: str) -> str:
    source = str(source).lower()
    scores = {k: 0 for k in d.keys()}
    for sk, sv in d.items():
        overlaps = sum([i in source for i in sv])
        scores[sk] = overlaps

    if max(scores.values()) == 0:
        return "OTHER"

    return max(scores, key=scores.get)


def classify_multi(d: dict, row: pd.Series) -> dict:
    row = str(row).lower()
    
    result = {}
    for sk, sv in d.items():
        overlaps = sum([kw in row for kw in sv])
        # result[sk] = int(overlaps > 0)  # 1 если есть хотя бы одно совпадение
        result[sk] = overlaps
    
    return result


def extract_period(period):
    """Return 'Q' and 'M' periods"""
    period = str(period).upper()
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']

    if any([i for i in months if i in period]):
        return 'M'
    
    elif any([i for i in quarters if i in period]):
        return 'Q'
    
    else:
        return None


def extract_stage_release(title: pd.Series) -> pd.Series:
    """Return 'Preliminary', 'Flash', 'Final' stage of release from the title."""
    title = str(title).upper()
    if 'PREL' in title:
        return 'Preliminary'
    if 'FLASH' in title:
        return 'Flash'
    if 'FINAL' in title:
        return 'Final'
    return None


def extract_calculation_period(title: pd.Series) -> pd.Series:
    title = str(title).upper()
    if 'MOM' in title:
        return 'MoM'
    if 'QOQ' in title:
        return 'QoQ'
    if 'YOY' in title:
        return 'YoY'
    return None


def get_report_period(ser: pd.Series):
    pass


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


def get_dummies(data: pd.DataFrame, column: str, prefix: str):
    """Create dummy variables for a single column."""
    return pd.get_dummies(data, columns=[column], prefix=prefix, prefix_sep='_', dtype=int)


def prepare_for_dummy(data: pd.DataFrame):
    data['stage_release'] = data['title'].apply(extract_stage_release)
    data['calc_period'] = data['title'].apply(extract_calculation_period)
    data['source'] = data['source'].apply(lambda x: classify_by_dict(SOURCE_CATS, x))

    # Add event category
    multi_labels = data['title'].apply(lambda x: classify_multi(CLASS_KEYWORDS, str(x).lower()))
    multi_df = pd.DataFrame(list(multi_labels))
    data = pd.concat([data, multi_df.add_prefix('event_')], axis=1)

    # Add most important events
    data['most_important_event'] = data['title'].apply(get_most_important_events)
    data['mie'] = data['most_important_event'].copy()
    data['is_calendar'] = data['indicator'].apply(lambda x: 1 if x == 'Calendar' else 0)
    
    for col in ['president', 'election']:
        data[f"is_{col}"] = data['title'].apply(lambda x: 1 if col in str(x).lower() else 0)

    return data


def extract_news_features_pipeline(data: pd.DataFrame):
    data['stage_release'] = data['title'].apply(extract_stage_release)
    data['calc_period'] = data['title'].apply(extract_calculation_period)
    data['source'] = data['source'].apply(lambda x: classify_by_dict(SOURCE_CATS, x))

    # Add event category
    multi_labels = data['title'].apply(lambda x: classify_multi(CLASS_KEYWORDS, str(x).lower()))
    multi_df = pd.DataFrame(list(multi_labels))
    data = pd.concat([data, multi_df.add_prefix('event_')], axis=1)

    # Add most important events
    data['most_important_event'] = data['title'].apply(get_most_important_events)
    data['mie'] = data['most_important_event'].copy()

    for col in ['category', 'currency', 'country', 'source', 'stage_release', 'calc_period', 'scale', 'mie']:
        data = get_dummies(data, column=col, prefix=col)

    for col in ['president', 'election']:
        data[f"is_{col}"] = data['title'].apply(lambda x: 1 if col in str(x).lower() else 0)

    # Add calendar flag
    data['is_calendar'] = data['indicator'].apply(lambda x: 1 if x == 'Calendar' else 0)

    # Remove redundant columns
    data = data.drop(columns=[i for i in data.columns if i.lower().endswith('other')] + ['source_url'])
    return data


def floor_or_ceil(x: str, freq: str='h'):
    """
    Если значения минут ближе к 30 или 00, значит округляем в большую сторону, иначе в меньшую.
    """
    x_dt = pd.to_datetime(x)
    if freq == '30min':
        if (29 >= x_dt.minute >= 20) or (59 >= x_dt.minute >= 50):  # TODO: May be change to last 5 minutes (20 -> 25 and 50 -> 55)
            return x_dt.ceil(freq, ambiguous='NaT', nonexistent='shift_forward')
        else:
            return x_dt.floor(freq)
    elif freq == 'h':
        if x_dt.minute <= 29:
            return x_dt.floor(freq, ambiguous='NaT', nonexistent='shift_forward')
        else:
            return x_dt.ceil(freq, ambiguous='NaT', nonexistent='shift_forward')
    return None


def get_max_weight_event(x: pd.DataFrame):
    x.reset_index(inplace=True, drop=True)
    # return x.loc[x['weights'].argmax(), 'most_important_event']
    return x.loc[x['weights'].argmax(), 'main_event']


def aggregate_events(df: pd.DataFrame, dt_col: str = 'time_to_check'):
    # 1. Basic counts
    agg = df \
        .groupby(dt_col) \
            .agg(
                news_count=("title", "count"),
                high_impact_count=("importance", lambda x: (x == 1).sum()),
                key_event_count=("most_important_event", "count"),
            )
    
    # 2. Main event
    df['weights'] = df['most_important_event'].apply(lambda x: EVENT_WEIGHTS_D.get(x, 0))
    main_event = df.groupby('rounded_time')[['weights', 'most_important_event']].apply(lambda x: get_max_weight_event(x))
    main_event.name = 'main_event'
    main_event.fillna('No main events', inplace=True)

    agg = agg.join(main_event, how='left')

    # 3. Add previous and next hour features
    agg['prev_hour_news_count'] = agg['news_count'].shift(1)
    agg['next_hour_news_count'] = agg['news_count'].shift(-1)
    agg['prev_hour_high_impact_count'] = agg['high_impact_count'].shift(1)
    agg['next_hour_high_impact_count'] = agg['high_impact_count'].shift(-1)
    agg['prev_hour_main_event'] = agg['main_event'].shift(1)
    agg['next_hour_main_event'] = agg['main_event'].shift(-1)
    
    # 4. Sum features
    features = [
        'category', 'currency', 'country', 'source', 'event', 
        'stage_release', 'calc_period', 'is_calendar', 'is_president', 'mie_'
    ]
    for ftr in features:
        ftr_agg_d = {c: 'sum' for c in df.columns if c.startswith(ftr)}
        feature_sum = df.groupby(dt_col).agg(ftr_agg_d).astype(int)
        agg = pd.concat([agg, feature_sum], axis=1)
    
    agg = agg.reset_index()

    # 5. Add what hours left for prev and next event
    agg['time_passed_from_last_events'] = (agg[dt_col] - agg[dt_col].shift(1)).dt.total_seconds() / (60 * 60 / 2)
    agg['time_left_to_next_events'] = (agg[dt_col].shift(-1) - agg[dt_col]).dt.total_seconds() / (60 * 60 / 2)

    # 5. Fill Nans for numeric columns
    num_cols = agg.select_dtypes(include=[np.number]).columns
    agg.fillna({c: 0 for c in num_cols}, inplace=True)

    # 6. How long ago was the last key event?
    key_event_mask = ~agg['main_event'].isna()
    last_important_event_dt = agg[dt_col].where(key_event_mask).shift(1).ffill()
    agg['last_important_event_in_hours'] = (agg[dt_col] - last_important_event_dt).dt.total_seconds() / 3600

    return agg