from economy_classes import CLASS_KEYWORDS

import re
import pandas as pd
import numpy as np


def extract_speaker_major_and_name(speaker: str) -> pd.DataFrame:
    """
    Extract the major (organization/institution) and name from Speaker column.
    
    Parameters
    ----------
    speaker : str
        Speaker string from news dataframe (e.g., "ECB President Jean-Claude Trichet;")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with two columns: 'Major' and 'Name'
    """
    if pd.isna(speaker) or speaker is None or speaker == '':
        return pd.DataFrame({'Major': [None], 'Name': [None]})
    
    # Remove trailing semicolon and whitespace
    speaker = str(speaker).strip().rstrip(';').strip()
    
    # Common central bank organizations and their variations (ordered by specificity)
    organizations = {
        'FED': ['Federal Reserve Bank of New York', 'Federal Reserve Bank', 'Federal Reserve', 'Fed', 'FOMC'],
        'ECB': ['ECB', 'European Central Bank'],
        'BOJ': ['BOJ', 'Bank of Japan'],
        'BOE': ['BOE', 'Bank of England'],
        'RBA': ['RBA', 'Reserve Bank of Australia'],
        'RBNZ': ['RBNZ', 'Reserve Bank of New Zealand'],
        'BOC': ['BOC', 'Bank of Canada'],
        'SNB': ['SNB', 'Swiss National Bank'],
        'PBOC': ['PBOC', 'People\'s Bank of China'],
    }
    
    # Common titles (ordered by length to match longer titles first)
    titles = ['Governor and Chairman', 'Vice President', 'Deputy Governor', 
              'President', 'Chairman', 'Chair', 'Governor', 'Member']
    
    major = None
    name = None
    
    # Step 1: Identify organization
    speaker_upper = speaker.upper()
    for org_key, org_variations in organizations.items():
        # Check variations from longest to shortest
        sorted_variations = sorted(org_variations, key=len, reverse=True)
        for variation in sorted_variations:
            if variation.upper() in speaker_upper:
                major = org_key
                break
        if major:
            break
    
    # Step 2: Find title position
    title_found = None
    title_pos = -1
    for title in titles:
        title_lower = title.lower()
        if title_lower in speaker.lower():
            title_pos = speaker.lower().find(title_lower)
            title_found = title
            break
    
    # Step 3: Extract name (comes after title)
    if title_found and title_pos >= 0:
        # Get text after title
        after_title = speaker[title_pos + len(title_found):].strip()
        
        if after_title:
            # Extract name pattern: capitalized words (1-3 words, may include hyphens)
            # Pattern: Start with capitalized word, may have hyphenated part, may have another capitalized word
            name_match = re.match(r'^([A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?)', after_title)
            if name_match:
                potential_name = name_match.group(1).strip()
                
                # Filter out common words that aren't names
                exclude_words = {'And', 'The', 'Of', 'To', 'In', 'At', 'On', 'Bank', 
                               'Federal', 'Reserve', 'European', 'Central', 'England', 
                               'Japan', 'Australia', 'New', 'Zealand', 'Canada', 
                               'Swiss', 'National', 'People', 'York'}
                
                name_words = potential_name.split()
                # Check if all words are valid name components
                if all(word not in exclude_words for word in name_words):
                    name = potential_name.rstrip('.,;:')
    
    # Step 4: If no organization found but title exists, try to infer from context
    if not major and title_found:
        # Look for organization before title
        before_title = speaker[:title_pos].strip() if title_pos > 0 else ''
        if before_title:
            # Try to match known organizations
            for org_key, org_variations in organizations.items():
                for variation in org_variations:
                    if variation.upper() in before_title.upper():
                        major = org_key
                        break
                if major:
                    break
            
            # If still no match, use first significant capitalized word as major
            if not major:
                words = before_title.split()
                for word in words:
                    if word and word[0].isupper() and word.isalpha():
                        # Skip common words
                        if word.lower() not in ['the', 'of', 'and']:
                            major = word
                            break
    
    # Step 5: If still no major, use first significant capitalized word
    if not major:
        words = speaker.split()
        skip_words = {'and', 'the', 'of', 'to', 'in', 'at', 'on'}
        for word in words:
            if word and word[0].isupper() and word.lower() not in skip_words:
                major = word
                break
    
    return pd.DataFrame({'Major': [major], 'Name': [name]})


def extract_speaker_features(news_df: pd.DataFrame, speaker_col: str = 'Speaker') -> pd.DataFrame:
    """
    Extract speaker major and name features from news dataframe.
    
    Parameters
    ----------
    news_df : pd.DataFrame
        News dataframe containing Speaker column
    speaker_col : str, default 'Speaker'
        Name of the column containing speaker information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with two columns: 'Major' and 'Name'
    """
    if speaker_col not in news_df.columns:
        raise ValueError(f"Column '{speaker_col}' not found in dataframe")
    
    results = news_df[speaker_col].apply(extract_speaker_major_and_name)
    result_df = pd.concat(results.tolist(), ignore_index=True)
    
    return result_df


def classify_news(title: str):
    title = str(title).lower()
    scores = {}

    for cls, keywords in CLASS_KEYWORDS.items():
        scores[cls] = sum(1 for k in keywords if k in title)

    if max(scores.values()) == 0:
        return "OTHER"

    return max(scores, key=scores.get)


def get_specific_event_type(title: str) -> str:
    t = title.lower()

    if "non farm" in t or "nonfarm" in t:
        return "NFP"
    if "cpi" in t and "core" in t:
        return "CORE_CPI"
    if "cpi" in t:
        return "CPI"
    if "pce" in t:
        return "PCE"
    if "fomc" in t and "rate" in t:
        return "FOMC_RATE"
    if "fomc" in t and "press" in t:
        return "FOMC_PRES_CONF"
    if "pmi" in t and "manufacturing" in t:
        return "PMI_MANUFACTURING"
    if "pmi" in t and "services" in t:
        return "PMI_SERVICES"
    if "gdp" in t:
        return "GDP"
    if "retail" in t:
        return "RETAIL_SALES"

    return "OTHER"


def period_extraction(period):
    period = str(period).lower()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months = [i.lower() for i in months]
    quarters = ['q1', 'q2', 'q3', 'q4']

    if any([i for i in months if i in period]):
        return 'M'

    elif any([i for i in quarters if i in period]):
        return 'Q'

    else:
        return None


def floor_or_ceil(x: str):
    """
    Если значения минут ближе к 30 или 00, значит округляем в большую сторону, иначе в меньшую.
    """
    x_dt = pd.to_datetime(x)
    if (29 >= x_dt.minute >= 20) or (59 >= x_dt.minute >= 50):  # TODO: May be change to last 5 minutes (20 -> 25 and 50 -> 55)
        return x_dt.ceil('30min', ambiguous='NaT', nonexistent='shift_forward')
    else:
        return x_dt.floor('30min')


def build_hour_features(
        news_df: pd.DataFrame,
        key_event_types: set[str] | None = None,
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
        base_news_count=("event_type", "count"),
        base_high_impact_count=("impact_rank", lambda x: (x == 2).sum()),
        base_sum_impact_rank=("impact_rank", "sum"),
        base_max_impact_rank=("impact_rank", "max"),
    )

    # -----------------------------
    # 2. DOMINANT EVENT TYPE
    # -----------------------------
    dominant_event = (
        df.groupby([event_time_column, "event_type"])
        .size()
        .reset_index(name="cnt")
        .sort_values([event_time_column, "cnt"], ascending=[True, False])
        .drop_duplicates(event_time_column)
        .set_index(event_time_column)[["event_type"]]
        .rename(columns={"event_type": "extra_dominant_event_type"})
    )

    # -----------------------------
    # 3. EVENT ENTROPY (STRUCTURAL NOISE)
    # -----------------------------
    # def event_entropy(x: pd.Series) -> float:
    #     probs = x.value_counts(normalize=True)
    #     return entropy(probs) if len(probs) > 1 else 0.0
    #
    # entropy_df = (
    #     df.groupby(event_time_column)["event_type"]
    #     .apply(event_entropy)
    #     .to_frame("event_entropy")
    # )

    # -----------------------------
    # 4. AGG ONE HOT FEATURES
    # -----------------------------
    event_agg_dict = {col: 'sum' for col in df.columns if col.startswith('e_')}
    cat_agg_dict = {col: 'sum' for col in df.columns if col.startswith('cat_')}
    cur_agg_dict = {col: 'sum' for col in df.columns if col.startswith('cur_')}

    # Perform the groupby and aggregation
    event_presence_sum = df.groupby(event_time_column).agg(event_agg_dict)
    category_presence_sum = df.groupby(event_time_column).agg(cat_agg_dict)
    currency_presence_sum = df.groupby(event_time_column).agg(cur_agg_dict)

    # -----------------------------
    # 5. FINAL MERGE
    # -----------------------------
    hour_features = (
        agg
        .join(dominant_event, how="left")
        # .join(entropy_df, how="left")
        # .join(presence_df, how="left")
        .join(event_presence_sum, how="left")
        .join(category_presence_sum, how="left")
        .join(currency_presence_sum, how="left")
        .reset_index()
    )

    # Fill NaNs (safety)
    num_cols = hour_features.select_dtypes(include=[np.number]).columns
    hour_features[num_cols] = hour_features[num_cols].fillna(0)

    hour_features["extra_dominant_event_type"] = (
        hour_features["extra_dominant_event_type"]
        .fillna("NONE")
        .astype("category")
    )

    # Create field 'Last_key_event_bars_ago' (example last important event was N bars ago and it was NFP)
    # Find last key event name and how many hours ago it was for each hour row

    # Prepare a mask for rows with actual key events (using is_key_event)
    key_event_mask = hour_features['extra_dominant_event_type'].isin(key_event_types)
    key_max_rank_mask = hour_features['base_max_impact_rank'] == 2

    # Build up last key event name using a forward fill
    hour_features['extra_last_key_event_name'] = (
        hour_features['extra_dominant_event_type'].where(key_event_mask).shift(1)
        .ffill()
    )

    # For time calculations, get event_hour of previous key events
    last_key_event_time = hour_features[event_time_column].where(key_event_mask).shift(1).ffill()
    last_max_impact_time = hour_features[event_time_column].where(key_max_rank_mask).shift(1).ffill()

    hour_features['extra_last_key_event_hours_ago'] = (
            (hour_features[event_time_column] - last_key_event_time).dt.total_seconds() / 3600
    )
    hour_features['extra_last_max_impact_hours_ago'] = (
            (hour_features[event_time_column] - last_max_impact_time).dt.total_seconds() / 3600
    )

    return hour_features