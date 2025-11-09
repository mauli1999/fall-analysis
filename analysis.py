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
from predict_risk import train_risk_model, predict_risk
from ehr_fetcher import EHRFetcher, FallData
from datetime import timedelta
import streamlit as st

load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_TOKEN")
client = InferenceClient(api_key=hf_api_key) if hf_api_key else None

def load_data(use_ehr: bool = False, file_path: str = 'synthetic_fall_data.csv', **kwargs) -> pd.DataFrame:
    """Synchronously load data from EHR or CSV, fallback to CSV if EHR fails or empty."""
    if use_ehr:
        fetcher = EHRFetcher()
        try:
            falls = fetcher.fetch_falls(**kwargs)
            data_rows = [fall.dict() for fall in falls]
            df = pd.DataFrame(data_rows)
            if df.empty:
                st.warning("No fall data found in EHR. Falling back to synthetic data.")
                df = pd.read_csv(file_path)
            df['fall_date'] = pd.to_datetime(df['fall_date'])
            df.fillna('Unknown', inplace=True)
            return df
        except ValueError as e:
            st.error(f"Fetch failed: {str(e)}. Falling back to synthetic data.")
            df = pd.read_csv(file_path)
            df['fall_date'] = pd.to_datetime(df['fall_date'])
            df.fillna('Unknown', inplace=True)
            return df
    else:
        df = pd.read_csv(file_path)
        df['fall_date'] = pd.to_datetime(df['fall_date'])
        df.fillna('Unknown', inplace=True)
        return df

def analyze_falls(df: pd.DataFrame) -> tuple[dict, str, BytesIO]:
    if df.empty:
        return {'total_falls': 0, 'falls_by_location': {}, 'falls_by_cause': {}, 'falls_by_injury': {}, 'falls_by_hour': {}, 'risk_by_location': {}, 'avg_risk': 0}, "No data available for analysis.", BytesIO()

    stats = {
        'total_falls': len(df),
        'falls_by_location': df['location'].value_counts().to_dict(),
        'falls_by_cause': df['cause'].value_counts().to_dict(),
        'falls_by_injury': df['injury'].value_counts().to_dict(),
        'falls_by_hour': pd.to_datetime(df['fall_time'], format='%H:%M', errors='coerce').dt.hour.value_counts().sort_index().to_dict()
    }
    
    model_path = 'fall_risk_model.pkl'
    model, encoders = train_risk_model(df)
    joblib.dump((model, encoders), model_path)
    
    risk_scores = predict_risk(model, encoders, df)
    df['risk_score'] = risk_scores
    stats['risk_by_location'] = df.groupby('location')['risk_score'].mean().fillna(0).to_dict()
    stats['avg_risk'] = df['risk_score'].mean()

    if client:
        try:
            prompt = f"Summarize the following fall data trends concisely: {str(stats)}"
            narrative = client.summarization(prompt, model="facebook/bart-large-cnn")
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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    df['location'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Falls by Location')
    ax.set_xlabel('Location')
    ax.set_ylabel('Number of Falls')
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close(fig)
    img = Image.open(img_buffer)
    img_buffer.seek(0)
    return stats, narrative, img_buffer