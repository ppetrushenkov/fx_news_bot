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


def build_hour_features(
    news_df: pd.DataFrame,
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
        base_news_count=("importance", "count"),
        base_high_impact_count=("importance", lambda x: (x == 2).sum()),
        base_sum_impact_rank=("importance", "sum"),
        base_max_impact_rank=("importance", "max"),
    )

    return agg
