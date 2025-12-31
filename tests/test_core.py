import pytest
import pandas as pd
import numpy as np
from data_generation import generate_synthetic_fall_data
from predict_risk import train_model

def test_synthetic_data_shape():
    """Test that synthetic data has expected shape."""
    data = generate_synthetic_fall_data(n_samples=100)
    assert data.shape == (100, 8), f"Expected shape (100, 8), got {data.shape}"

def test_synthetic_data_value_ranges():
    """Test that synthetic data values are within expected ranges."""
    data = generate_synthetic_fall_data(n_samples=1000)
    # Age should be between 18 and 100 (clipped)
    assert data['age'].min() >= 18, f"Min age {data['age'].min()} below 18"
    assert data['age'].max() <= 100, f"Max age {data['age'].max()} above 100"
    # Gender should be 'Male' or 'Female'
    assert data['gender'].isin(['Male', 'Female', None]).all(), "Gender values invalid"
    # Fall occurred should be 0 or 1
    assert data['fall_occurred'].isin([0, 1]).all(), "Fall occurred not binary"

def test_synthetic_data_reproducibility():
    """Test that synthetic data is reproducible with fixed seeds."""
    data1 = generate_synthetic_fall_data(n_samples=100)
    data2 = generate_synthetic_fall_data(n_samples=100)
    pd.testing.assert_frame_equal(data1, data2), "Data not reproducible"

def test_model_training_does_not_crash():
    """Test that model training does not raise exceptions."""
    data = generate_synthetic_fall_data(n_samples=200)
    data_path = 'temp_test_data.csv'
    data.to_csv(data_path, index=False)
    try:
        model = train_model(data_path)
        assert model is not None, "Model training failed"
    except Exception as e:
        pytest.fail(f"Model training crashed: {e}")
    finally:
        import os
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists('fall_risk_model.pkl'):
            os.remove('fall_risk_model.pkl')

def test_model_predictions_shape():
    """Test that model produces predictions of correct shape."""
    data = generate_synthetic_fall_data(n_samples=200)
    data_path = 'temp_test_data.csv'
    data.to_csv(data_path, index=False)
    model = train_model(data_path)
    # Simulate test data
    X_test = data.drop('fall_occurred', axis=1).select_dtypes(include=[np.number])  # Assuming numeric features for simplicity
    predictions = model.predict(X_test)
    assert predictions.shape == (200,), f"Predictions shape {predictions.shape} != (200,)"
    # Clean up
    import os
    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists('fall_risk_model.pkl'):
        os.remove('fall_risk_model.pkl')