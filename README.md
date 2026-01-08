# Fall Detection AI Agent

This is a research prototype for prototyping machine learning pipelines on synthetic data resembling fall incident scenarios.

## What This Project Does (Current State Only)

The project generates synthetic data with features like age, gender, location, cause, and time of day, along with a binary target. It trains a Random Forest Classifier model to predict the binary target, provides a Streamlit dashboard for basic data visualization, and evaluates model performance on a held-out test set.

## Data Source (Synthetic) and Why

All data is fully synthetic, created programmatically without any basis in real-world data. Synthetic data is used to enable controlled testing of machine learning code without requiring access to external datasets.

## Modeling Approach and Evaluation Metrics

The approach uses Random Forest Classifier for binary classification, with a train/validation/test split (60/20/20). A simple baseline model (dummy classifier) is trained for comparison. Evaluation metrics include accuracy, precision, recall, ROC-AUC, and confusion matrix, computed on the test set.

## Key Limitations and Non-Goals

- The entire project operates on synthetic data only, with no connection to real events or data.
- Synthetic data includes artificial correlations and imbalances but does not simulate any real-world phenomena.
- This prototype does not aim to produce usable models, handle predictions beyond the synthetic dataset, or apply to any practical scenarios.
- No advanced techniques like hyperparameter optimization, complex feature engineering, or model ensembles are included.
- The dashboard is basic and for illustrative purposes only.

## Ethical Considerations and Disclaimer

This project is strictly for educational prototyping of machine learning concepts. It does not involve or reference real data, individuals, or events. It should not be used for any decision-making, analysis, or application outside of code development and testing. No ethical, privacy, or regulatory considerations are addressed, as the project is entirely synthetic.

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. Clone the repository:
   https://github.com/yourusername/fall_detection_ai_agent.git

2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

### Usage

1. Generate synthetic data (optional, as the dashboard can load existing data):
   python data_generation.py

2. Train the model:
   python predict_risk.py
   python analysis.py

3. Run the dashboard:
   streamlit run dashboard.py

4. Run tests:
   pytest tests.py
