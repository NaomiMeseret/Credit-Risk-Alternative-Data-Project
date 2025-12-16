"""
Smoke tests for training pipeline.
"""

import pandas as pd
from src.train import train_with_mlflow


def test_train_with_mlflow_smoke(tmp_path):
    # minimal dataset with target
    df = pd.DataFrame({
        'cat_feature': ['a','b','a','b'],
        'num_feature': [1,2,3,4],
        'is_high_risk': [0,1,0,1]
    })
    tracking_uri = f"file:{tmp_path}/mlruns"
    model, metrics = train_with_mlflow(
        df=df,
        target_col='is_high_risk',
        model_type='logistic',
        tracking_uri=tracking_uri,
        save_local_path=tmp_path / "best_model"
    )
    assert 'roc_auc' in metrics
    assert metrics['roc_auc'] >= 0.5  # sanity check
    # ensure model saved
    assert (tmp_path / "best_model").exists()
