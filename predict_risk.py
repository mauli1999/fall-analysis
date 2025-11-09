import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple, Any

def train_risk_model(df: pd.DataFrame) -> Tuple[Any, dict]:
    """Train a risk prediction model and return it with encoders."""
    # Prepare features (exclude target 'fall_status' to avoid leakage)
    features = ['location', 'cause', 'injury']
    X = df[features].copy()
    y = (df['fall_status'] == 1).astype(int)  # Binary target: 1 for fall, 0 for no-fall
    
    # Encode categorical variables
    encoders = {}
    for col in features:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # Add time-based feature (hour from fall_time)
    X['Hour'] = pd.to_datetime(df['fall_time'], format='%H:%M', errors='coerce').dt.hour.fillna(12)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, encoders

def predict_risk(model: Any, encoders: dict, new_data: pd.DataFrame) -> np.ndarray:
    """Predict risk scores for new data."""
    # Prepare features (exclude target 'fall_status')
    features = ['location', 'cause', 'injury']
    X_new = new_data[features].copy()
    
    # Encode categorical variables using existing encoders
    for col in features:
        if col in encoders:
            le = encoders[col]
            X_new[col] = X_new[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    
    # Add time-based feature (hour from fall_time)
    X_new['Hour'] = pd.to_datetime(new_data['fall_time'], format='%H:%M', errors='coerce').dt.hour.fillna(12)
    
    # Predict probabilities (risk score as probability of fall)
    try:
        proba = model.predict_proba(X_new)
    except Exception as e:
        logger.exception("predict_proba failed, falling back to predict: %s", e)
        # fallback: if predict exists, use it as hard 0/1 probabilities
        try:
            preds = model.predict(X_new)
            return np.array(preds, dtype=float)
        except Exception:
            # final fallback: return zeros
            return np.zeros(len(X_new), dtype=float)

    # Robust handling of returned probability shapes
    proba = np.asarray(proba)
    if proba.ndim == 1:
        # some classifiers may return 1d array
        risk_scores = proba.ravel()
    elif proba.shape[1] == 1:
        # only one column returned => model fitted with a single class
        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == 1:
            single_class = classes[0]
            if single_class == 1:
                risk_scores = np.ones(proba.shape[0], dtype=float)
            else:
                risk_scores = np.zeros(proba.shape[0], dtype=float)
        else:
            # treat the single column as probability of class 1
            risk_scores = proba.ravel()
    else:
        # multi-column: try to find column for class '1', else use second column
        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                idx = list(classes).index(1)
            except ValueError:
                idx = 1 if proba.shape[1] > 1 else 0
        else:
            idx = 1 if proba.shape[1] > 1 else 0
        risk_scores = proba[:, idx]

    # ensure numeric numpy array
    risk_scores = np.asarray(risk_scores, dtype=float)
    
    return risk_scores

if __name__ == "__main__":
    # Example usage (for testing)
    from ehr_fetcher import EHRFetcher
    fetcher = EHRFetcher()
    falls = fetcher.fetch_falls(max_results=50)
    df = pd.DataFrame([fall.dict() for fall in falls])
    if df.empty:
        # Fallback to synthetic data
        df = pd.read_csv('synthetic_fall_data.csv')
        df['fall_date'] = pd.to_datetime(df['fall_date'])
        df.fillna('Unknown', inplace=True)
    model, encoders = train_risk_model(df)
    joblib.dump((model, encoders), 'fall_risk_model.pkl')
    scores = predict_risk(model, encoders, df)
    print(f"Sample risk scores: {scores[:5]}")