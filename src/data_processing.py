"""
Data processing and feature engineering module for Credit Risk Model.

This module provides all data engineering needed to prepare raw transactional data for modeling:
- RFM/aggregate features
- Extracted time features
- WoE/IV transformation
- Encoding & normalization
- Proxy target label creation

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from xverse.transformer import WOETransformer
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Core pipeline components
# --------------------------

class RFMFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', date_col='TransactionStartTime', amount_col='Amount', snapshot_date=None):
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        if self.snapshot_date is None:
            snapshot = df[self.date_col].max() + pd.Timedelta(days=1)
        else:
            snapshot = pd.to_datetime(self.snapshot_date)
        rfm = df.groupby(self.customer_id_col).agg(
            Recency=(self.date_col, lambda x: (snapshot - x.max()).days),
            Frequency=(self.customer_id_col, 'count'),
            Monetary=(self.amount_col, 'sum'),
            MedianAmt=(self.amount_col, 'median'),
            AvgAmt=(self.amount_col, 'mean'),
            StdAmt=(self.amount_col, 'std')
        ).reset_index().fillna(0)
        return rfm

class TransactionDateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df['Transaction_Hour'] = df[self.date_col].dt.hour
        df['Transaction_Day'] = df[self.date_col].dt.day
        df['Transaction_Month'] = df[self.date_col].dt.month
        df['Transaction_Year'] = df[self.date_col].dt.year
        return df

class NumericFeatureAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        agg = df.groupby(self.customer_id_col).agg(
            Total_Amount=(self.amount_col, 'sum'),
            Avg_Amount=(self.amount_col, 'mean'),
            Transaction_Count=(self.amount_col, 'count'),
            Std_Amount=(self.amount_col, 'std')
        ).reset_index().fillna(0)
        return agg

# Categorical/encoding pipeline
class CategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, encoding_type='onehot'):
        self.cols = cols
        self.encoding_type = encoding_type
        self.encoders = {}
        self.encoder=None
    def fit(self, X, y=None):
        if self.encoding_type == 'onehot':
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(X[self.cols])
        else:
            for col in self.cols:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders[col] = le
        return self
    def transform(self, X):
        X = X.copy()
        if self.encoding_type == 'onehot':
            arr = self.encoder.transform(X[self.cols])
            cats = self.encoder.get_feature_names_out(self.cols)
            onehot_df = pd.DataFrame(arr, columns=cats, index=X.index)
            X.reset_index(drop=True, inplace=True)
            return pd.concat([X.drop(self.cols, axis=1).reset_index(drop=True), onehot_df], axis=1)
        else:
            for col in self.cols:
                X[col] = self.encoders[col].transform(X[col].astype(str))
            return X

# WoE Transformer using xverse
class WoEIVFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_features: list, target_col: str):
        self.cat_features = cat_features
        self.target_col = target_col
        self.woe = WOETransformer(features=self.cat_features)
    def fit(self, X, y=None):
        self.woe.fit(X, X[self.target_col])
        return self
    def transform(self, X):
        X = self.woe.transform(X)
        return X

# --------------------------
# Loader and builder
# --------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    """Load transaction data from CSV file."""
    return pd.read_csv(data_path)

def create_feature_pipeline(customer_id_col='CustomerId', date_col='TransactionStartTime', amount_col='Amount', cat_cols=None, encoding_type='onehot'):
    if cat_cols is None:
        cat_cols = ['ProductCategory', 'ChannelId']
    pipeline = Pipeline([
        ("date_feats", TransactionDateFeatureEngineer(date_col=date_col)),
        ("agg_numeric", NumericFeatureAggregator(customer_id_col, amount_col)),
        ("encode_cat", CategoryEncoder(cat_cols, encoding_type=encoding_type)),
        ("impute", SimpleImputer(strategy='median')),
        ("scale", StandardScaler()),
    ])
    return pipeline

# --------------------------
# RFM & Proxy Target Creation
# --------------------------

def engineer_rfm_features(df: pd.DataFrame, customer_id_col: str = 'CustomerId') -> pd.DataFrame:
    """
    Engineer RFM features at customer level
    """
    snapshot = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby(customer_id_col).agg(
        Recency = ('TransactionStartTime', lambda x: (snapshot - pd.to_datetime(x.max())).days),
        Frequency = (customer_id_col, 'count'),
        Monetary = ('Amount', 'sum'),
        MedianAmt=('Amount','median'),
        AvgAmt=('Amount','mean'),
        StdAmt=('Amount','std')
    ).reset_index().fillna(0)
    return rfm

def create_proxy_variable(df: pd.DataFrame, rfm_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Create proxy (high-risk) variable based on RFM clustering.
    Returns: main DataFrame with is_high_risk column merged
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    # prepare RFM for clustering
    rfm_vars = ['Recency','Frequency','Monetary','MedianAmt','AvgAmt','StdAmt']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[rfm_vars])
    # clustering
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['cluster'] = km.fit_predict(rfm_scaled)
    # Assign high risk: cluster with highest Recency, lowest Frequency & Monetary
    stats = rfm_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = stats.sort_values(['Recency','Frequency','Monetary'], ascending=[False,True,True]).index[0]
    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
    # merge proxy variable to main df
    df = df.merge(rfm_df[[customer_id_col, 'is_high_risk']], how='left', left_on=customer_id_col, right_on=customer_id_col)
    return df
