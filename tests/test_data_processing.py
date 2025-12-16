"""
Unit tests for data processing module.
"""

import pandas as pd
import numpy as np
import pytest
from src.data_processing import engineer_rfm_features, create_proxy_variable


def test_engineer_rfm_features_columns():
    data = pd.DataFrame({
        'CustomerId': ['c1','c1','c2'],
        'TransactionStartTime': ['2024-01-01','2024-01-05','2024-01-03'],
        'Amount': [100, 50, 25]
    })
    rfm = engineer_rfm_features(data, customer_id_col='CustomerId')
    expected_cols = {'CustomerId','Recency','Frequency','Monetary','MedianAmt','AvgAmt','StdAmt'}
    assert expected_cols.issubset(set(rfm.columns)), "RFM columns missing"
    assert len(rfm) == 2


def test_create_proxy_variable_merges_label():
    data = pd.DataFrame({
        'CustomerId': ['c1','c1','c2','c3'],
        'TransactionStartTime': ['2024-01-01','2024-01-05','2024-01-03','2024-01-02'],
        'Amount': [100, 50, 25, 10]
    })
    rfm = engineer_rfm_features(data, customer_id_col='CustomerId')
    out = create_proxy_variable(data, rfm_df=rfm, n_clusters=2, random_state=0)
    assert 'is_high_risk' in out.columns, "Proxy label not merged"
    # Ensure all customers have a label
    assert out['is_high_risk'].notnull().all()
    # Binary label
    assert set(out['is_high_risk'].unique()).issubset({0,1})
