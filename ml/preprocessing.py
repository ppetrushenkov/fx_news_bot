import pandas as pd
from scipy.stats import entropy

from economy_classes import CLASS_KEYWORDS


def get_event_period(period: str) -> str:
    """
    Return the period of the event.
    
    Possible periods: 
    - M
    - Q
    """
    period = str(period).lower()
    
    monthes = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthes = [i.lower() for i in monthes]
    quarters = ['q1', 'q2', 'q3', 'q4']

    if any([i for i in monthes if i in period]):
        return 'M'
    elif any([i for i in quarters if i in period]):
        return 'Q'
    else:
        return None


def get_event_category(event: str):
    """
    Return the category of the event.
    
    Possible categories: 
    - MONETARY_POLICY
    - CB_SPEECH
    - INFLATION
    - LABOR_MARKET
    - GDP
    - ECONOMIC_ACTIVITY
    - SENTIMENT
    - CONSUMER_HOUSING
    - MANUFACTORING
    - TRADE_FINANCE
    - COMMODITIES
    - BONDS
    """
    event = str(event).lower()
    scores = {}

    for cls, keywords in CLASS_KEYWORDS.items():
        scores[cls] = sum([1 for k in keywords if k in event])

    return max(scores, key=scores.get)


def get_event_type(event: str):
    """
    Return the type of the event.
    
    Possible types: 
    - NFP
    - INTEREST_RATE
    - CPI_CORE
    - CPI
    - GDP
    - PMI_MANUFACTURING
    - PMI_SERVICES
    - RETAIL_SALES
    - OTHER
    """
    t = event.lower()

    if any([i for i in ["non farm", "nonfarm", "payrolls"] if i in t]):
        return "NFP"
    elif "interest rate" in t:
        return "INTEREST_RATE"
    elif "cpi" in t and "core" in t:
        return "CPI_CORE"
    elif "cpi" in t:
        return "CPI"
    elif "gdp" in t:
        return "GDP"
    elif "pmi" in t and 'manufacturing' in t:
        return "PMI_MANUFACTURING"
    elif "pmi" in t and 'services' in t:
        return "PMI_SERVICES"
    elif "retail" in t:
        return "RETAIL_SALES"
    
    return "OTHER"


def build_hour_features(
    news_df: pd.DataFrame,
    key_event_types: list[str] | None = None,
    event_time_column: str = 'custom_event_time'
) -> pd.DataFrame:
    """
    Build hour-level aggregated features from event-level news dataframe.

    Parameters
    ----------
    news_df : pd.DataFrame
        Event-level news data. Must contain:
        ['event_hour', 'event_type', 'news_class',
         'impact_rank', 'is_key_event', 'event_weight']

    key_event_types : list[str], optional
        List of key event types to build presence flags for
        (e.g. ['NFP', 'CPI', 'PMI_MANUFACTURING']).
        If None, inferred from is_key_event == True.

    Returns
    -------
    pd.DataFrame
        Hour-level feature table indexed by event_hour.
    """

    df = news_df.copy()

    # -----------------------------
    # 1. BASIC COUNTS & INTENSITY
    # -----------------------------
    agg = df.groupby(event_time_column).agg(
        news_count=("event_type", "count"),
        high_impact_count=("impact_rank", lambda x: (x == 2).sum()),
        key_event_count=("is_key_event", "sum"),
        sum_impact_rank=("impact_rank", "sum"),
        sum_event_weight=("event_weight", "sum"),
        max_event_weight=("event_weight", "max"),
    )

    # -----------------------------
    # 2. DOMINANT EVENT TYPE
    # -----------------------------
    # dominant_event = (
    #     df.groupby([event_time_column, "event_type"])
    #       .size()
    #       .reset_index(name="cnt")
    #       .sort_values([event_time_column, "cnt"], ascending=[True, False])
    #       .drop_duplicates(event_time_column)
    #       .set_index(event_time_column)[["event_type"]]
    #       .rename(columns={"event_type": "dominant_event_type"})
    # )
    dominant_event = (
        df.groupby([event_time_column, 'event_type', 'event_weight']) \
        .size()
        .reset_index(name="cnt")
        .sort_values([event_time_column, "cnt"], ascending=[True, False])
        .drop_duplicates(event_time_column)
        .set_index(event_time_column)[["event_type"]]
        .rename(columns={"event_type": "dominant_event_type"})
    )

    # -----------------------------
    # 3. EVENT ENTROPY (STRUCTURAL NOISE)
    # -----------------------------
    def event_entropy(x: pd.Series) -> float:
        probs = x.value_counts(normalize=True)
        return entropy(probs) if len(probs) > 1 else 0.0

    entropy_df = (
        df.groupby(event_time_column)["event_type"]
          .apply(event_entropy)
          .to_frame("event_entropy")
    )

    # -----------------------------
    # 4. EVENT PRESENCE
    # -----------------------------
    event_agg_dict = {col: 'sum' for col in df.columns if col.startswith('e_')}
    cat_agg_dict = {col: 'sum' for col in df.columns if col.startswith('class_')}
    
    # Perform the groupby and aggregation
    event_presence_sum = df.groupby(event_time_column).agg(event_agg_dict)
    category_presence_sum = df.groupby(event_time_column).agg(cat_agg_dict)

    # Add period column
    period_agg_dict = {col: 'sum' for col in df.columns if col.startswith('period_')}
    period_presence_sum = df.groupby(event_time_column).agg(period_agg_dict)

    # -----------------------------
    # 5. FINAL MERGE
    # -----------------------------
    hour_features = (
        agg
        .join(dominant_event, how="left")
        .join(entropy_df, how="left")
        # .join(presence_df, how="left")
        .join(event_presence_sum, how="left")
        .join(category_presence_sum, how="left")
        .join(period_presence_sum, how="left")
        .reset_index()
    )

    # Fill NaNs (safety)
    num_cols = hour_features.select_dtypes(include=[np.number]).columns
    hour_features[num_cols] = hour_features[num_cols].fillna(0)

    hour_features["dominant_event_type"] = (
        hour_features["dominant_event_type"]
        .fillna("NONE")
        .astype("category")
    )

    # Create field 'Last_key_event_bars_ago' (example last important event was N bars ago and it was NFP)
    # Find last key event name and how many hours ago it was for each hour row

    # Prepare a mask for rows with actual key events (using is_key_event)
    key_event_mask = hour_features['dominant_event_type'].isin(key_event_types)

    # Build up last key event name using a forward fill
    hour_features['last_key_event_name'] = (
        hour_features['dominant_event_type'].where(key_event_mask).shift(1)
        .ffill()
    )

    # For time calculations, get event_hour of previous key events
    last_key_event_time = hour_features[event_time_column].where(key_event_mask).shift(1).ffill()

    hour_features['last_key_event_hours_ago'] = (
        (hour_features[event_time_column] - last_key_event_time).dt.total_seconds() / 3600
    )

    return hour_features