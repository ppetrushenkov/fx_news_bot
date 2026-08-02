from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

import sys
import os

from ml.source_categories import SOURCE_CATS
from ml.event_categories import CLASS_KEYWORDS
from ml.news_featuring import (
    aggregate_events,
    classify_multi,
    extract_calculation_period,
    extract_stage_release,
    floor_or_ceil,
    get_most_important_events,
    classify_by_dict
)
from ml.price_featuring import add_features, get_base_and_quote_currency
# from targets import set_targets


# class EventPreprocessingTransformer(BaseEstimator, TransformerMixin):
#     cols_to_onehot = ['category', 'currency', 'country', 'source', 'stage_release', 'calc_period', 'scale', 'mie']

#     def __init__(self, datetime_crop_method: str = '1st'):
#         self.datetime_crop_method = datetime_crop_method
#         self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         self.final_columns_ = None

#     def _crop_time(self, X: pd.DataFrame) -> pd.DataFrame:
#         if self.datetime_crop_method == '1st':
#             X['date'] = pd.to_datetime(X['date'], utc=True)
#             X['rounded_time'] = X['date'].apply(lambda x: x.floor('1h'))
#         elif self.datetime_crop_method == '2nd':
#             X['date'] = pd.to_datetime(X['date'], utc=True)
#             X['rounded_time'] = X['date'].apply(lambda x: floor_or_ceil(x, freq='h'))

#         return X

#     def _encode_categoricals(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
#         events = prepare_for_dummy(X)
#         cat_data = events[self.cols_to_onehot].astype(str).fillna('missing')

#         if fit:
#             encoded = self.onehot_encoder.fit_transform(cat_data)
#         else:
#             encoded = self.onehot_encoder.transform(cat_data)

#         feature_names = self.onehot_encoder.get_feature_names_out(self.cols_to_onehot)
#         encoded_df = pd.DataFrame(encoded, columns=feature_names, index=events.index)

#         other_cols = [c for c in events.columns if c not in self.cols_to_onehot]
#         events = pd.concat([events[other_cols], encoded_df], axis=1)

#         drop_cols = [c for c in events.columns if c.lower().endswith('other') or c.lower().endswith('none')] + ['source_url']
#         events = events.drop(columns=[c for c in drop_cols if c in events.columns])

#         return events

#     def _preprocess(self, X: pd.DataFrame, fit_encoder: bool = False) -> pd.DataFrame:
#         events_features = self._encode_categoricals(X, fit=fit_encoder)
#         events_features = self._crop_time(events_features)

#         agg_events = aggregate_events(events_features, dt_col='rounded_time')
#         agg_events['time_to_check'] = agg_events['rounded_time'] - pd.Timedelta(hours=1)

#         return agg_events

#     def fit(self, X: pd.DataFrame, y=None):
#         agg_events = self._preprocess(X, fit_encoder=True)
#         self.final_columns_ = list(agg_events.columns)
#         print('[INFO] Fit done!')

#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         agg_events = self._preprocess(X, fit_encoder=False)
#         agg_events = agg_events.reindex(columns=self.final_columns_, fill_value=0)
#         print('[INFO] Transform done!')

#         return agg_events


# class EventPreprocessingTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, datetime_crop_method: str = '1st'):     
#         self.datetime_crop_method = datetime_crop_method
#         self.final_columns_ = None

#     def _crop_time(self, X: pd.DataFrame) -> pd.DataFrame:
#         if self.datetime_crop_method == '1st':
#             X['date'] = pd.to_datetime(X['date'], utc=True)
#             X['rounded_time'] = X['date'].apply(lambda x: x.floor('1h'))
#         elif self.datetime_crop_method == '2nd':
#             X['date'] = pd.to_datetime(X['date'], utc=True)
#             X['rounded_time'] = X['date'].apply(lambda x: floor_or_ceil(x, freq='h'))
        
#         return X

#     def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
#         events = X.copy()
#         events_features = extract_news_features_pipeline(events)
#         events_features = self._crop_time(events_features)
#         agg_events = aggregate_events(events_features, dt_col='rounded_time')
#         agg_events['time_to_check'] = agg_events['rounded_time'] - pd.Timedelta(hours=1)

#         return agg_events

#     def fit(self, X: pd.DataFrame, y=None):
#         agg_events = self._preprocess(X)
        
#         self.final_columns_ = list(agg_events.columns)
#         print('[INFO] Fit done!')
        
#         return self
     
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         agg_events = self._preprocess(X)
        
#         agg_events = agg_events.reindex(columns=self.final_columns_, fill_value=0)
#         print('[INFO] Transform done!')
                
#         return agg_events

# def classify_source(d: dict, source: str) -> str:
#     source = str(source).lower()
#     scores = {k: 0 for k in d.keys()}
#     for sk, sv in d.items():
#         overlaps = sum([i in source for i in sv])
#         scores[sk] = overlaps

#     if max(scores.values()) == 0:
#         return "OTHER"

#     return max(scores, key=scores.get)


class EventPreprocessingTransformer(BaseEstimator, TransformerMixin):
    cols_to_onehot = ['category', 'currency', 'country', 'source', 'stage_release', 'calc_period', 'scale', 'mie']

    def __init__(self, dt_crop_method: str = '1st'):
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.dt_crop_method = dt_crop_method
        self.final_columns_ = None

    def _crop_time(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.dt_crop_method == '1st':
            X['date'] = pd.to_datetime(X['date'], utc=True)
            X['rounded_time'] = X['date'].apply(lambda x: x.floor('1h'))
        elif self.dt_crop_method == '2nd':
            X['date'] = pd.to_datetime(X['date'], utc=True)
            X['rounded_time'] = X['date'].apply(lambda x: floor_or_ceil(x, freq='h'))

        return X

    def extract_features(self, X: pd.DataFrame) -> pd.DataFrame:
        events = X.copy()
        events['source'] = events['source'].apply(lambda x: classify_by_dict(SOURCE_CATS, x))
        events['stage_release'] = events['title'].apply(extract_stage_release)
        events['calc_period'] = events['title'].apply(extract_calculation_period)

        multi_labels = events['title'].apply(lambda x: classify_multi(CLASS_KEYWORDS, str(x).lower()))
        multi_df = pd.DataFrame(list(multi_labels))
        events = pd.concat([events, multi_df.add_prefix('event_')], axis=1)

        events['most_important_event'] = events['title'].apply(get_most_important_events)
        events['mie'] = events['most_important_event']
        events['is_calendar'] = events['indicator'].apply(lambda x: 1 if x == 'Calendar' else 0)

        for col in ['president', 'election']:
            events[f'is_{col}'] = events['title'].apply(lambda x: 1 if col in str(x).lower() else 0)

        return events

    def _encode_categoricals(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        events = self.extract_features(X)
        cat_data = events[self.cols_to_onehot].astype(str).fillna('missing')

        if fit:
            encoded = self.onehot_encoder.fit_transform(cat_data)
        else:
            encoded = self.onehot_encoder.transform(cat_data)

        feature_names = self.onehot_encoder.get_feature_names_out(self.cols_to_onehot)
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=events.index)

        other_cols = [c for c in events.columns if c not in self.cols_to_onehot]
        events = pd.concat([events[other_cols], encoded_df], axis=1)

        drop_cols = [c for c in events.columns if c.lower().endswith('other') or c.lower().endswith('none')] + ['source_url']
        events = events.drop(columns=[c for c in drop_cols if c in events.columns])

        return events

    def _preprocess(self, X: pd.DataFrame, fit_encoder: bool = False) -> pd.DataFrame:
        events_features = self._encode_categoricals(X, fit=fit_encoder)
        events_features = self._crop_time(events_features)

        agg_events = aggregate_events(events_features, dt_col='rounded_time')
        agg_events['time_to_check'] = agg_events['rounded_time'] - pd.Timedelta(hours=1)

        return agg_events

    def fit(self, X: pd.DataFrame, y=None):
        agg_events = self._preprocess(X, fit_encoder=True)
        self.final_columns_ = list(agg_events.columns)
        print('[INFO] Fit done!')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        agg_events = self._preprocess(X, fit_encoder=False)
        agg_events = agg_events.reindex(columns=self.final_columns_, fill_value=0)
        print('[INFO] Transform done!')
        return agg_events


class PricePreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, period: int = 21, drop_raw_cols: bool = False):
        self.period = period
        self.drop_raw_cols = drop_raw_cols
        self.final_columns_ = None

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:

        if 'datetime' not in X.columns and 'time' in X.columns:
            X.rename(columns={'time': 'datetime'}, inplace=True)

        if 'ticker' not in X.columns:
            raise ValueError("DataFrame must contain 'ticker' column.")

        X[['base_currency', 'quote_currency']] = get_base_and_quote_currency(X['ticker'].iloc[0])
        prices = add_features(X, period=self.period)

        return prices
        
    def fit(self, X: pd.DataFrame, y=None):
        prices = self._preprocess(X)        
        self.final_columns_ = list(prices.columns)
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        prices = self._preprocess(X)
        prices = prices.reindex(columns=self.final_columns_, fill_value=0)
        return prices


class TargetPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, look_forward_bars: int = 4):
        self.look_forward_bars = look_forward_bars
        self.final_columns_ = None

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        prices = X.copy()
        prices = set_targets(prices, look_forward_bars=self.look_forward_bars)
        prices.drop(['open', 'high', 'low', 'close'], axis=1, errors='ignore', inplace=True)
        prices.dropna(inplace=True)
        return prices
        
    def fit(self, X: pd.DataFrame, y=None):
        prices = self._preprocess(X)
        self.final_columns_ = list(prices.columns)
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: 
        prices = self._preprocess(X)
        prices = prices.reindex(columns=self.final_columns_, fill_value=0)
        return prices


class DataMergerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: dict) -> pd.DataFrame:
        """
        Merges aggregated events and price data.
        X should be a dictionary with keys:
            'events': aggregated events DataFrame
            'prices': price features DataFrame
        """
        if not isinstance(X, dict):
            raise ValueError("DataMerger expects a dictionary with keys 'events' and 'prices'")
            
        if 'events' not in X or 'prices' not in X:
            raise ValueError("Dictionary must contain 'events' and 'prices' keys")
            
        agg_events = X['events']
        prices = X['prices']
        
        # Merge on time or rounded_time
        if 'time_to_check' in agg_events.columns and 'datetime' in prices.columns:
            merged = pd.merge(
                prices, agg_events, 
                left_on='datetime', 
                right_on='time_to_check', 
                how='left'
            )
        else:
            raise ValueError("No matching time columns for merge")
            
        return merged


class RangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    # TODO: add ranges


def get_full_preprocess_pipeline(train: bool = False):
    if train:
        return Pipeline([
            ('event_preprocessing', EventPreprocessingTransformer()),
            ('price_preprocessing', PricePreprocessingTransformer()),
            ('target_preprocessing', TargetPreprocessingTransformer()),
            ('data_merger', DataMergerTransformer()),
        ])
    else:
        return Pipeline([
            ('event_preprocessing', EventPreprocessingTransformer()),
            ('price_preprocessing', PricePreprocessingTransformer()),
            ('data_merger', DataMergerTransformer()),
        ])
