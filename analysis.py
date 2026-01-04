import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from io import BytesIO
from PIL import Image
import joblib
from predict_risk import train_model, predict_risk
from datetime import timedelta
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_TOKEN")
client = InferenceClient(api_key=hf_api_key) if hf_api_key else None

def load_data(file_path: str = 'synthetic_fall_data.csv', **kwargs) -> pd.DataFrame:
    """Synchronously load data from CSV file."""
    
    df = pd.read_csv(file_path)
    df['fall_date'] = pd.to_datetime(df['fall_date'])
    return df

def analyze_falls(df: pd.DataFrame = None):
    """
    Analyzes fall data. Accepts either a DataFrame or a file path to compute statistics, generate visualizations, and produce a narrative summary.
    
    This function loads data from the specified CSV file or gets the filtered data, performs statistical analysis on fall occurrences,
    trains a risk prediction model, computes risk scores, and generates a bar chart visualization.
    It uses only synthetic data and does not involve real EHR data.
    
    Parameters:
    data_path (str): Path to the CSV file containing the synthetic fall data.
    
    Returns:
    tuple: (stats dict, narrative str, img_buffer BytesIO) - Statistics dictionary, narrative summary, and image buffer for visualization.
    """
    """
    Analyzes fall data. Accepts either a DataFrame or a file path.
    """
    try:
        data_path = 'synthetic_fall_data.csv'
        
        if df is None:
            logging.info("Starting data loading from %s", data_path)
            df = pd.read_csv(data_path)
            logging.info("Data loading completed: %d rows, %d columns", df.shape[0], df.shape[1])
        else:
            logging.info("Using provided DataFrame with %d rows", len(df))
        if df.empty:
            return {'total_falls': 0, 'falls_by_location': {}, 'falls_by_cause': {}, 'falls_by_hour': {}, 'risk_by_location': {}, 'avg_risk': 0}, "No data available for analysis.", BytesIO()

        stats = {
            'total_falls': int(df['fall_occurred'].sum()),  # Fixed: actual falls, not total rows
            'falls_by_location': df['location'].value_counts().to_dict(),
            'falls_by_cause': df['cause'].value_counts().to_dict(),
            'falls_by_hour': pd.to_datetime(df['fall_time'], format='%H:%M', errors='coerce').dt.hour.value_counts().sort_index().to_dict()
        }
        
        # Train model and compute risk scores
        model_file = 'fall_risk_model.pkl'
        if os.path.exists(model_file):
            logging.info("Loading existing model from %s", model_file)
            model, encoders, imputers = joblib.load(model_file)
        else:
            logging.info("No saved model found; training new model")
            model, encoders, imputers = train_model(data_path) 
        
        # Compute risk scores on the provided DataFrame
        risk_scores = predict_risk(model, encoders, imputers, df)
        df = df.copy()
        df['risk_score'] = risk_scores
        
        stats['risk_by_location'] = df.groupby('location')['risk_score'].mean().fillna(0).to_dict()
        stats['avg_risk'] = df['risk_score'].mean()

        # Generate narrative
        if client:
            try:
                prompt = f"Summarize the following fall data trends concisely: {str(stats)}"
                narrative = client.summarization(prompt, model="facebook/bart-large-cnn")
                # narrative = client.text_generation(prompt, model="facebook/bart-large-cnn", max_new_tokens=150)
                narrative = narrative if isinstance(narrative, str) else str(narrative)
            except Exception as e:
                narrative = f"Error generating narrative: {str(e)}"
        else:
            max_location = max(stats['falls_by_location'], key=stats['falls_by_location'].get, default='N/A')
            max_cause = max(stats['falls_by_cause'], key=stats['falls_by_cause'].get, default='N/A')
            max_hour = max(stats['falls_by_hour'], key=stats['falls_by_hour'].get, default='N/A')
            narrative = f"""
            Analysis Summary:
            - Total falls: {stats['total_falls']}
            - Most common location: {max_location} ({stats['falls_by_location'].get(max_location, 0)} falls)
            - Most common cause: {max_cause} ({stats['falls_by_cause'].get(max_cause, 0)} falls)
            - Peak hour: {max_hour}:00 ({stats['falls_by_hour'].get(max_hour, 0)} falls)
            - Average risk score: {stats['avg_risk']:.2f}
            """
        
        logging.info("Statistical analysis completed")
        
        # Generate visualizations
        logging.info("Starting visualization generation")
        fig, ax = plt.subplots(figsize=(8, 6))
        df['location'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Falls by Location')
        ax.set_xlabel('Location')
        ax.set_ylabel('Number of Falls')
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close(fig)
        img_buffer.seek(0)
        logging.info("Visualization generation completed")
        
        # Any metric computations
        logging.info("Computing summary metrics")
        # Additional metrics if needed
        logging.info("Metric computation completed")
        
        return stats, narrative, img_buffer
    
    except FileNotFoundError as e:
        logging.error("Critical error: Data file not found - %s", str(e))
        raise  # Fail fast
    except ValueError as e:
        logging.error("Critical error: Data analysis failed - %s", str(e))
        raise  # Fail fast
    except Exception as e:
        logging.error("Unexpected error during analysis - %s", str(e))
        raise  # Fail fast