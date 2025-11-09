import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64
from analysis import load_data, analyze_falls
from apscheduler.schedulers.background import BackgroundScheduler
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from PIL import Image as PILImage
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from analysis import load_data, analyze_falls
from apscheduler.schedulers.background import BackgroundScheduler
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from PIL import Image as PILImage
import atexit
import datetime
import os
import atexit
import asyncio
import logging


st.set_page_config(layout="wide", page_title="Fall Analysis Dashboard", page_icon="📊")

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f5f5; }
    .stHeader { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; text-align: center; font-size: 24px; font-weight: bold; }
    .stButton>button { background-color: #3498db; color: white; border: none; padding: 5px 15px; border-radius: 5px; }
    .stButton>button:hover { background-color: #2980b9; }
    .stDownloadButton>button { background-color: #2ecc71; }
    .stDownloadButton>button:hover { background-color: #27ae60; }
    </style>
    """,
    unsafe_allow_html=True
)

# initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load env early and read API key
load_dotenv()
sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
if not sendgrid_api_key:
    logger.warning("SENDGRID_API_KEY is not set. Email sending will fail until you set it in the environment.")

# scheduler setup
sched = BackgroundScheduler()
atexit.register(lambda: sched.shutdown(wait=False))

@st.cache_data(ttl=3600, max_entries=1, persist=False)
def get_data(data_source):
    """Load data synchronously (FHIR or synthetic)."""
    return load_data(use_ehr=(data_source == "Public FHIR (HAPI)"))

df = get_data(st.session_state.get('data_source', "Synthetic Data"))

@st.cache_data(ttl=3600)
def generate_falls_by_location(filtered_df):
    if filtered_df.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, 'No data available for selected filters', ha='center', va='center', fontsize=12)
        ax.set_title('Falls by Location')
        ax.set_xlabel('Location')
        ax.set_ylabel('Number of Falls')
        ax.set_xticks([])
        ax.set_yticks([])
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer
    fig, ax = plt.subplots(figsize=(6, 5))
    filtered_df['location'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Falls by Location')
    ax.set_xlabel('Location')
    ax.set_ylabel('Number of Falls')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    plt.tight_layout()
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    return img_buffer

@st.cache_data(ttl=3600)
def generate_falls_by_hour(filtered_df):
    if filtered_df.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, 'No data available for selected filters', ha='center', va='center', fontsize=12)
        ax.set_title('Falls by Hour')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Number of Falls')
        ax.set_xticks([])
        ax.set_yticks([])
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer
    fig, ax = plt.subplots(figsize=(6, 5))
    filtered_df['hour'] = pd.to_datetime(filtered_df['fall_time'], format='%H:%M', errors='coerce').dt.hour
    filtered_df['hour'].value_counts().sort_index().plot(kind='bar', ax=ax, color='lightgreen')
    ax.set_title('Falls by Hour')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Number of Falls')
    ax.tick_params(axis='x', labelsize=10)
    plt.tight_layout()
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    return img_buffer

@st.cache_data(ttl=3600)
def generate_falls_by_cause(filtered_df):
    if filtered_df.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, 'No data available for selected filters', ha='center', va='center', fontsize=12)
        ax.set_title('Falls by Cause')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('equal')
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer
    fig, ax = plt.subplots(figsize=(6, 5))
    cause_counts = filtered_df['cause'].value_counts()
    ax.pie(cause_counts, labels=cause_counts.index, autopct='%1.1f%%', colors=plt.cm.Pastel1.colors, textprops={'fontsize': 8})
    ax.axis('equal')
    ax.set_title('Falls by Cause')
    plt.tight_layout()
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    return img_buffer

@st.cache_data(ttl=3600)
def generate_risk_heatmap(stats):
    if not stats.get('risk_by_location'):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, 'No data available for selected filters', ha='center', va='center', fontsize=12)
        ax.set_title('Risk Score by Location')
        ax.set_xticks([])
        ax.set_yticks([])
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer
    fig, ax = plt.subplots(figsize=(6, 5))
    risk_data = pd.Series(stats['risk_by_location']).sort_index()
    heatmap = ax.imshow([risk_data.values], cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(risk_data.index)))
    ax.set_xticklabels(risk_data.index, rotation=45, fontsize=10)
    ax.set_yticks([])
    ax.set_title('Risk Score by Location')
    plt.colorbar(heatmap)
    plt.tight_layout()
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)
    return img_buffer

def generate_pdf_report(stats: dict, narrative: str, chart_buffer: BytesIO, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle(name='Title', parent=styles['Heading1'], fontSize=16, spaceAfter=12)
    story.append(Paragraph("Monthly Fall Analysis Report", title_style))
    narrative_style = ParagraphStyle(name='Narrative', parent=styles['Normal'], fontSize=12, leading=14)
    narrative_text = narrative.replace("'", "").replace("SummarizationOutput(summary_text=", "").strip()
    story.append(Paragraph(narrative_text, narrative_style))
    story.append(Paragraph("<br/><br/>", narrative_style))
    stats_style = ParagraphStyle(name='Stats', parent=styles['Normal'], fontSize=10, leading=12)
    story.append(Paragraph("Key Statistics:", stats_style))
    for key, value in stats.items():
        stat_text = f"- {key}: {value}" if not isinstance(value, dict) else f"- {key}: {dict(value)}"
        story.append(Paragraph(stat_text, stats_style))
    img_pil = PILImage.open(chart_buffer)
    aspect = img_pil.size[0] / img_pil.size[1]
    img_width = 4 * inch
    img_height = img_width / aspect
    if img_height > 3 * inch:
        img_height = 3 * inch
        img_width = img_height * aspect
    chart_buffer.seek(0)
    from reportlab.platypus import Image
    story.append(Image(chart_buffer, width=img_width, height=img_height))
    doc.build(story)


def _run_async_coroutine(coro, *args, **kwargs):
    """Run an async coroutine from sync context and log exceptions."""
    try:
        return asyncio.run(coro(*args, **kwargs))
    except Exception:
        logger.exception("Error running coroutine through asyncio.run")

def schedule_async_job(async_func, trigger, **trigger_args):
    """Schedule an async function with BackgroundScheduler by wrapping it."""
    def _job_wrapper():
        _run_async_coroutine(async_func)
    sched.add_job(_job_wrapper, trigger, **trigger_args)
    logger.info("Scheduled async job %s with %s %s", async_func.__name__, trigger, trigger_args)


def scheduled_report(recipient_email, run_date=None):
    current_month = datetime.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    filtered_df = df[df['fall_date'] >= current_month]
    if not filtered_df.empty:
        stats, narrative, chart_buffer = analyze_falls(filtered_df)
        temp_pdf_path = "monthly_fall_report.pdf"
        generate_pdf_report(stats, narrative, chart_buffer, temp_pdf_path)
        with open(temp_pdf_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            message = Mail(
                from_email="mauli3003@gmail.com",
                to_emails=recipient_email,
                subject="Monthly Fall Analysis Report",
                plain_text_content="Attached is the monthly fall analysis report."
            )
            attachedFile = Attachment(
                FileContent(encoded),
                FileName("monthly_fall_report.pdf"),
                FileType("application/pdf"),
                Disposition("attachment")
            )
            message.attachment = attachedFile
            sg = SendGridAPIClient(sendgrid_api_key)
            response = sg.send(message)
            st.success(f"Email sent! Status: {response.status_code}")
        os.remove(temp_pdf_path)
    else:
        st.warning("No fall data for the current month.")

# Sidebar
st.sidebar.header("Filters")
locations = ['All'] + sorted(df['location'].unique())
selected_location = st.sidebar.selectbox("Filter by Location", locations)
causes = ['All'] + sorted(df['cause'].unique())
selected_cause = st.sidebar.selectbox("Filter by Cause", causes)
date_min = df['fall_date'].min().to_pydatetime()
date_max = df['fall_date'].max().to_pydatetime()
selected_dates = st.sidebar.slider("Select Date Range", min_value=date_min, max_value=date_max, value=(date_min, date_max))

st.sidebar.markdown("---")
st.sidebar.subheader("Report Actions")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download CSV",
    data=csv,
    file_name="fall_report.csv",
    mime="text/csv"
)
recipient_email = st.sidebar.text_input("Recipient Email")
if st.sidebar.button("Send Now") and recipient_email and sendgrid_api_key:
    scheduled_report(recipient_email)
elif not sendgrid_api_key:
    st.sidebar.warning("SendGrid API key not found. Set SENDGRID_API_KEY in .env.")
schedule_type = st.sidebar.selectbox("Frequency", ["None", "Daily", "Weekly", "Monthly"])
if schedule_type != "None":
    if st.sidebar.button(f"Set {schedule_type} Schedule"):
        sched.remove_all_jobs()
        if schedule_type == "Daily":
            sched.add_job(lambda: scheduled_report("mauli3003@gmail.com"), 'interval', days=1)
        elif schedule_type == "Weekly":
            sched.add_job(lambda: scheduled_report("mauli3003@gmail.com"), 'interval', weeks=1)
        elif schedule_type == "Monthly":
            sched.add_job(lambda: scheduled_report("mauli3003@gmail.com"), 'cron', day='1', hour=0, minute=0)
        st.sidebar.success(f"Report scheduled to run {schedule_type.lower()} to mauli3003@gmail.com")

st.sidebar.markdown("---")
data_source = st.sidebar.radio("Data Source", ["Synthetic Data", "Public FHIR (HAPI)"], index=0)
if 'data_source' not in st.session_state or st.session_state['data_source'] != data_source:
    st.session_state['data_source'] = data_source
    st.cache_data.clear()
    df = get_data(data_source)
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    df = load_data(use_ehr=(data_source == "Public FHIR (HAPI)"))
    st.sidebar.success(f"Data refreshed from {data_source}.")

# Apply Filters
filtered_df = df.copy()
if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]
if selected_cause != 'All':
    filtered_df = filtered_df[filtered_df['cause'] == selected_cause]
filtered_df = filtered_df[(filtered_df['fall_date'] >= selected_dates[0]) & (filtered_df['fall_date'] <= selected_dates[1])]

# Main content
st.markdown('<div class="stHeader">Fall Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

st.header("Summary Statistics")
stats, narrative, _ = analyze_falls(filtered_df)
col1, col3 = st.columns(2)
with col1:
    st.metric("Total Falls", stats['total_falls'])
with col3:
    st.metric("Avg Risk Score", f"{stats['avg_risk']:.2f}")

st.header("Visualizations")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Falls by Location")
    falls_loc_buffer = generate_falls_by_location(filtered_df)
    st.image(falls_loc_buffer, use_container_width=True)
with col2:
    st.subheader("Falls by Hour")
    falls_hour_buffer = generate_falls_by_hour(filtered_df)
    st.image(falls_hour_buffer, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Falls by Cause")
    falls_cause_buffer = generate_falls_by_cause(filtered_df)
    st.image(falls_cause_buffer, use_container_width=True)
with col4:
    st.subheader("Risk by Location")
    risk_heatmap_buffer = generate_risk_heatmap(stats)
    st.image(risk_heatmap_buffer, use_container_width=True)

st.header("AI Narrative Summary")
clean_narrative = narrative.replace("SummarizationOutput(summary_text=", "").replace(")", "").strip().replace("'", "")
st.markdown(clean_narrative)

if not sched.running:
    sched.start()