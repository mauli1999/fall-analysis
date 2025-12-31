import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
from typing import Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Preprocesses the dataset by imputing missing values, encoding categoricals, and adding features.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with columns age, gender, location, cause, time_of_day, fall_occurred, fall_date, fall_time, injury.
    
    Returns:
    Tuple[pd.DataFrame, dict]: Preprocessed dataframe and dictionary of encoders.
    """
    df = df.copy()
    
    # Impute missing age (numerical) with mean
    age_imputer = SimpleImputer(strategy='mean')
    df['age'] = age_imputer.fit_transform(df[['age']])
    
    # Impute missing gender (categorical) with most frequent
    gender_imputer = SimpleImputer(strategy='most_frequent')
    df['gender'] = gender_imputer.fit_transform(df[['gender']]).ravel()
    
    # Encode categorical features
    categorical_features = ['location', 'cause']
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Add Hour feature from fall_time
    df['Hour'] = pd.to_datetime(df['fall_time'], format='%H:%M', errors='coerce').dt.hour.fillna(12)
    
    return df, encoders

def train_model(data_path: str) -> Tuple[Any, dict]:
    """
    Trains the risk prediction model and baseline, evaluates them, and saves the model.
    
    Parameters:
    data_path (str): Path to the CSV file.
    
    Returns:
    Tuple[Any, dict]: Trained model and encoders.
    """
    try:
        logging.info("Starting data loading from %s", data_path)
        df = pd.read_csv(data_path)
        logging.info("Data loading completed: %d rows, %d columns", df.shape[0], df.shape[1])
        
        # Preprocess data
        logging.info("Starting data preprocessing")
        df, encoders = preprocess_data(df)
        logging.info("Data preprocessing completed")
        
        # Prepare features and target
        features = ['location', 'cause', 'Hour']
        X = df[features]
        y = df['fall_occurred']
        
        # Split data into train/validation/test (60/20/20 for example, but prompt says train/validation/test)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # Train baseline
        logging.info("Starting baseline model training")
        baseline = DummyClassifier(strategy='most_frequent')
        baseline.fit(X_train, y_train)
        logging.info("Baseline model training completed")
        
        # Evaluate baseline on test
        logging.info("Evaluating baseline model")
        y_pred_baseline = baseline.predict(X_test)
        y_proba_baseline = baseline.predict_proba(X_test)[:, 1]
        print("Baseline Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline)}")
        print(f"Precision: {precision_score(y_test, y_pred_baseline)}")
        print(f"Recall: {recall_score(y_test, y_pred_baseline)}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_baseline)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_baseline)}")
        
        # Train main model
        logging.info("Starting main model training")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        logging.info("Main model training completed")
        
        # Evaluate main model on test
        logging.info("Evaluating main model")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print("Main Model Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Precision: {precision_score(y_test, y_pred)}")
        print(f"Recall: {recall_score(y_test, y_pred)}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        # Save model and encoders
        joblib.dump((model, encoders), 'fall_risk_model.pkl')
        logging.info("Model and encoders saved to fall_risk_model.pkl")
        
        return model, encoders
    
    except FileNotFoundError as e:
        logging.error("Critical error: Data file not found - %s", str(e))
        raise
    except Exception as e:
        logging.error("Unexpected error during training - %s", str(e))
        raise

def predict_risk(model: Any, encoders: dict, new_data: pd.DataFrame) -> np.ndarray:
    """
    Predicts risk scores for new data.
    
    Parameters:
    model (Any): Trained model.
    encoders (dict): Dictionary of label encoders.
    new_data (pd.DataFrame): New data to predict on.
    
    Returns:
    np.ndarray: Risk scores (probabilities of fall).
    """
    try:
        df = new_data.copy()
        # Ensure Hour exists
        df['Hour'] = pd.to_datetime(df['fall_time'], format='%H:%M', errors='coerce').dt.hour.fillna(12)
        
        # Encode categorical features using saved encoders
        for col in ['location', 'cause']:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        features = ['location', 'cause', 'Hour']
        X_new = df[features]
        
        # Predict probabilities
        risk_scores = model.predict_proba(X_new)[:, 1]
        return risk_scores
    except Exception as e:
        logging.error("Error during prediction - %s", str(e))
        raise


if __name__ == "__main__":
    # Example usage
    model, encoders = train_model('synthetic_fall_data.csv')
    # Load some data for prediction
    df = pd.read_csv('synthetic_fall_data.csv')
    scores = predict_risk(model, encoders, df.head(10))
    print(f"Sample risk scores: {scores}")