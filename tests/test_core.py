import pytest
import pandas as pd
import numpy as np
import os
from data_generation import generate_synthetic_fall_data
from predict_risk import predict_risk, train_model

def test_synthetic_data_shape():
    """Test that synthetic data has expected shape."""
    data = generate_synthetic_fall_data(n_samples=100)
    assert data.shape == (100, 8), f"Expected shape (100, 8), got {data.shape}"

def test_synthetic_data_value_ranges():
    """Test that synthetic data values are within expected ranges and allow missingness."""
    data = generate_synthetic_fall_data(n_samples=1000)
    
    # Age: integers 18-100 where present, allow NaNs
    age_clean = data['age'].dropna()
    assert age_clean.between(18, 100).all(), f"Age values out of range: {age_clean.min()} to {age_clean.max()}"
    assert data['age'].dtype == 'float64'  # Due to NaNs
    
    # Gender: only 'Male', 'Female', or NaN allowed
    assert data['gender'].dropna().isin(['Male', 'Female']).all(), "Invalid gender values found"
    
    # Fall occurred should be 0 or 1 (no missing)
    assert data['fall_occurred'].isin([0, 1]).all(), "Fall occurred not binary"

def test_synthetic_data_reproducibility():
    """Test that synthetic data is reproducible with fixed seeds."""
    data1 = generate_synthetic_fall_data(n_samples=100)
    data2 = generate_synthetic_fall_data(n_samples=100)
    pd.testing.assert_frame_equal(data1, data2, check_dtype=True)

def test_model_training_does_not_crash():
    """Test that model training does not raise exceptions."""
    data = generate_synthetic_fall_data(n_samples=200)
    data_path = 'temp_test_data.csv'
    data.to_csv(data_path, index=False)
    try:
        model, encoders, imputers = train_model(data_path)
        assert model is not None
        assert isinstance(encoders, dict)
        assert isinstance(imputers, dict)
        assert os.path.exists('fall_risk_model.pkl')
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists('fall_risk_model.pkl'):
            os.remove('fall_risk_model.pkl')

def test_predict_risk_function():
    """Test the predict_risk function works end-to-end."""
    data = generate_synthetic_fall_data(n_samples=200)
    data_path = 'temp_test_data.csv'
    data.to_csv(data_path, index=False)
    
    try:
        model, encoders, imputers = train_model(data_path)
        
        # Test on a small subset
        test_subset = data.head(10).copy()
        
        # Introduce a missing value and unseen category to test robustness
        test_subset.loc[0, 'age'] = np.nan
        test_subset.loc[0, 'gender'] = np.nan
        test_subset.loc[0, 'location'] = 'Outpatient Clinic'  # Unseen category
        
        scores = predict_risk(model, encoders, imputers, test_subset)
        
        assert len(scores) == 10
        assert scores.min() >= 0 and scores.max() <= 1
        assert not np.any(np.isnan(scores))
        
    finally:
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists('fall_risk_model.pkl'):
            os.remove('fall_risk_model.pkl')