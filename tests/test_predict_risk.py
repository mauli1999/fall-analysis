import sys
import os
# ensure project root is on sys.path so imports like `import predict_risk` work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pytest
from predict_risk import train_risk_model, predict_risk

def make_df(rows=10, single_class=None):
    # Helper to create synthetic fall data
    data = {
        "location": ["bathroom", "hall", "bedroom", "kitchen"] * ((rows // 4) + 1),
        "cause": ["slippery floor", "tripped on object", "medical", "unknown"] * ((rows // 4) + 1),
        "injury": ["minor", "major", "none", "minor"] * ((rows // 4) + 1),
        "fall_time": ["08:30", "14:20", "22:10", "03:45"] * ((rows // 4) + 1),
        "fall_status": [1, 0, 1, 0] * ((rows // 4) + 1),
    }
    df = pd.DataFrame({k: v[:rows] for k, v in data.items()})
    if single_class is not None:
        # set all fall_status to given class (0 or 1)
        df["fall_status"] = [single_class] * len(df)
    return df

def test_train_and_predict_basic():
    df = make_df(rows=20)
    model, encoders = train_risk_model(df)
    scores = predict_risk(model, encoders, df)
    assert isinstance(scores, np.ndarray)
    assert scores.shape[0] == df.shape[0]
    # probabilities should be numeric
    assert np.all(np.isfinite(scores))

def test_predict_handles_unknown_categories():
    df_train = make_df(rows=20)
    model, encoders = train_risk_model(df_train)
    # create new data with a category unseen during training
    df_new = df_train.copy()
    df_new.loc[0, "cause"] = "brand new cause"
    scores = predict_risk(model, encoders, df_new)
    assert scores.shape[0] == df_new.shape[0]

def test_single_class_model_produces_finite_scores():
    # train model on only one class (all zeros)
    df_single = make_df(rows=20, single_class=0)
    model, encoders = train_risk_model(df_single)
    scores = predict_risk(model, encoders, df_single)
    assert scores.shape[0] == df_single.shape[0]
    # When model has single class, predict_risk should return finite numbers (0 or 1 or probabilities)
    assert np.all(np.isfinite(scores))