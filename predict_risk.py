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

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Preprocesses the dataset by imputing missing values, encoding categoricals, and adding features.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with columns age, gender, location, cause, time_of_day, fall_occurred, fall_date, fall_time.
    
    Returns:
    Tuple[pd.DataFrame, dict, dict]: Preprocessed dataframe and dictionaries of encoders and imputers.
    """
    df = df.copy()
    
    # Impute and save imputers
    age_imputer = SimpleImputer(strategy='mean')
    df['age'] = age_imputer.fit_transform(df[['age']])
    
    gender_imputer = SimpleImputer(strategy='most_frequent')
    df['gender'] = gender_imputer.fit_transform(df[['gender']]).ravel()
    
    # Encode categorical features
    categorical_features = ['location', 'cause', 'gender']
    encoders = {}
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown').astype(str)
        le = LabelEncoder()
        le.fit(df[col])
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['Unknown'])[0])
        encoders[col] = le
    
    # Add Hour feature
    df['Hour'] = pd.to_datetime(df['fall_time'], format='%H:%M', errors='coerce').dt.hour.fillna(12)
    
    # Save imputers in dict
    imputers = {
        'age': age_imputer,
        'gender': gender_imputer
    }
    
    return df, encoders, imputers

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
        df, encoders, imputers = preprocess_data(df)
        logging.info("Data preprocessing completed")
        
        # Prepare features and target
        features = ['age','location', 'cause', 'gender', 'Hour']
        X = df[features]
        y = df['fall_occurred']
        
        # Split data into train/validation/test
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
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,          # Limit depth
            min_samples_leaf=5,    # Prevent tiny leaves
            random_state=42
        )
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
        
        joblib.dump((model, encoders, imputers), 'fall_risk_model.pkl')
        logging.info("Model, encoders, and imputers saved to fall_risk_model.pkl")

        return model, encoders, imputers
    
    except FileNotFoundError as e:
        logging.error("Critical error: Data file not found - %s", str(e))
        raise
    except Exception as e:
        logging.error("Unexpected error during training - %s", str(e))
        raise

def predict_risk(model: Any, encoders: dict, imputers: dict, new_data: pd.DataFrame) -> np.ndarray:
    """
    Predicts risk scores for new data.
    
    Parameters:
    model (Any): Trained model.
    encoders (dict): Dictionary of label encoders.
    imputers: dict containing fitted 'age' and 'gender' SimpleImputer objects
    new_data (pd.DataFrame): New data to predict on.
    
    Returns:
    np.ndarray: Risk scores (probabilities of fall).

    """
    try:
        df = new_data.copy()
        
        # === REAPPLY IMPUTATION ===
        if 'age' in imputers:
            df['age'] = imputers['age'].transform(df[['age']])
        if 'gender' in imputers:
            df['gender'] = imputers['gender'].transform(df[['gender']]).ravel()
        
        # Add Hour
        df['Hour'] = pd.to_datetime(df['fall_time'], format='%H:%M', errors='coerce').dt.hour.fillna(12)
        
        # Encode categoricals
        for col in ['location', 'cause', 'gender']:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['Unknown'])[0]
                )
        
        features = ['age', 'location', 'cause', 'gender', 'Hour']
        X_new = df[features]       
        risk_scores = model.predict_proba(X_new)[:, 1]
        return risk_scores
    except Exception as e:
        logging.error("Error during prediction - %s", str(e))
        raise

if __name__ == "__main__":
    model, encoders, imputers = train_model('synthetic_fall_data.csv')
    # Load some data for prediction
    df = pd.read_csv('synthetic_fall_data.csv')
    scores = predict_risk(model, encoders, imputers, df.head(10))
    print(f"Sample risk scores: {scores}")