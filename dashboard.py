import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import joblib
from analysis import analyze_falls
from predict_risk import predict_risk
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import sendgrid
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

@st.cache_data
def load_data():
    """Load the synthetic fall data."""
    try:
        logging.info("Loading data in dashboard")
        df = pd.read_csv('synthetic_fall_data.csv')
        df['fall_date'] = pd.to_datetime(df['fall_date'])
        logging.info("Data loaded successfully")
        return df
    except Exception as e:
        logging.error("Error loading data: %s", str(e))
        st.error("Failed to load data.")
        return pd.DataFrame()

def generate_pdf_report(stats, narrative, img_buffer):
    """Generate PDF report."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Fall Detection Report")
    c.drawString(100, 730, f"Total Falls: {stats.get('total_falls', 0)}")
    c.drawString(100, 710, f"Average Risk: {stats.get('avg_risk', 0):.2f}")
    c.drawString(100, 690, "Narrative:")
    text = narrative
    lines = text.split('\n')
    y = 670
    for line in lines:
        c.drawString(100, y, line)
        y -= 20

    # In generate_pdf_report():
    if img_buffer:
        img_buffer.seek(0)
        c.drawImage(ImageReader(img_buffer), 100, 400, width=400, height=300)

    # if img_buffer:
    #     c.drawImage(img_buffer, 100, 400, width=400, height=300)
    c.save()
    buffer.seek(0)
    return buffer

async def scheduled_report(email, stats, narrative, img_buffer):
    """Send email report asynchronously."""
    try:
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        message = Mail(
            from_email='******@gmail.com',
            to_emails=email,
            subject='Scheduled Fall Detection Report',
            html_content=f'<p>{narrative}</p>'
        )
        # Attach PDF
        pdf_buffer = generate_pdf_report(stats, narrative, img_buffer)
        encoded_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
        attachment = Attachment(
            FileContent(encoded_pdf),
            FileName('report.pdf'),
            FileType('application/pdf'),
            Disposition('attachment')
        )
        message.attachment = attachment
        response = sg.send(message)
        logging.info("Email sent successfully")
    except Exception as e:
        logging.error("Error sending email: %s", str(e))

def main():
    st.title("Fall Detection Dashboard")
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data available.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    locations = st.sidebar.multiselect("Location", options=df['location'].unique(), default=df['location'].unique())
    causes = st.sidebar.multiselect("Cause", options=df['cause'].unique(), default=df['cause'].unique())
    date_range = st.sidebar.date_input("Date Range", [df['fall_date'].min(), df['fall_date'].max()])
    
    # Filter data
    filtered_df = df[
        (df['location'].isin(locations)) &
        (df['cause'].isin(causes)) &
        (df['fall_date'] >= pd.to_datetime(date_range[0])) &
        (df['fall_date'] <= pd.to_datetime(date_range[1]))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # Load model
    try:
        logging.info("Loading model")
        model, encoders, imputers = joblib.load('fall_risk_model.pkl')
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error("Error loading model: %s", str(e))
        st.error("Failed to load model.")
        return
    
    # Compute risk scores
    try:
        risk_scores = predict_risk(model, encoders, imputers, filtered_df)
        filtered_df['risk_score'] = risk_scores
    except Exception as e:
        logging.error("Error computing risk scores: %s", str(e))
        st.error("Failed to compute risk scores.")
        return
    
    # Analyze falls
    try:
        logging.info("Running analysis")
        stats, narrative, img_buffer = analyze_falls(df=filtered_df)
        logging.info("Analysis completed")
    except Exception as e:
        logging.error("Error in analysis: %s", str(e))
        st.error("Failed to analyze data.")
        return
    
    # Display metrics
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Falls", stats.get('total_falls', 0))
    col2.metric("Average Risk", f"{stats.get('avg_risk', 0):.2f}")
    col3.metric("Filtered Falls", len(filtered_df))
    
    # Visualizations
    st.header("Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Falls by Location")
        if not filtered_df.empty:
            fig, ax = plt.subplots()
            filtered_df['location'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        else:
            st.write("No data to display.")
    
    with col2:
        st.subheader("Falls by Cause")
        if not filtered_df.empty:
            fig, ax = plt.subplots()
            filtered_df['cause'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        else:
            st.write("No data to display.")
    
    st.subheader("Falls by Hour")
    if not filtered_df.empty:
        fig, ax = plt.subplots()
        filtered_df['fall_time'].apply(lambda x: pd.to_datetime(x, format='%H:%M').hour).value_counts().sort_index().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No data to display.")
    
    st.subheader("Risk Heatmap by Location")
    if not filtered_df.empty:
        pivot = filtered_df.pivot_table(values='risk_score', index='location', aggfunc='mean')
        fig, ax = plt.subplots()
        sns.heatmap(pivot, annot=True, ax=ax)
        st.pyplot(fig)
    else:
        st.write("No data to display.")
    
    # Narrative
    st.header("Analysis Narrative")
    st.write(narrative)
    
    # Download CSV
    st.header("Download Filtered Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "filtered_fall_data.csv", "text/csv")
    
    # Email report
    st.header("Email Report")
    email = st.text_input("Email Address")
    if st.button("Send Report"):
        if email:
            asyncio.run(scheduled_report(email, stats, narrative, img_buffer))
            st.success("Report sent!")
        else:
            st.error("Please enter an email address.")

if __name__ == "__main__":
    main()