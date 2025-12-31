import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_fall_data(n_samples=1000):
    """
    Generates synthetic data simulating fall incidents for machine learning prototyping.
    
    This function creates a dataset with features such as age, gender, location, cause, and time,
    along with a binary target indicating fall occurrence. The data is designed to mimic basic
    structures for testing predictive models, including correlations, imbalances, and noise.
    
    What it explicitly does NOT represent:
    - Real patient data from electronic health records.
    - Clinically validated fall risk factors or outcomes.
    - Any real-world distributions, correlations, or statistics from healthcare settings.
    - Ethical or privacy considerations of actual medical data.
    
    Assumptions and Limitations:
    - Age is modeled as a normal distribution centered at 70 years, with correlations to fall occurrence
      based on a linear probability increase, but this does not reflect empirical clinical data.
    - Class imbalance is introduced with falls occurring at lower rates, simulating rarity, though
      without grounding in observed prevalence.
    - Missing values and noise are randomly applied to a subset of features, representing potential
      data quality issues in hypothetical scenarios, but not based on real-world error patterns.
    - All features are generated independently except for the specified correlations, and the dataset
      lacks temporal dependencies, confounding variables, or external validations.
    - This synthetic data is solely for illustrative and prototyping purposes; it should not be used
      to draw conclusions about real fall incidents or inform clinical decisions.
    
    Parameters:
    n_samples (int): Number of samples to generate.
    
    Returns:
    pd.DataFrame: Synthetic dataset.
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate base features with some correlations
    age = np.random.randint(18, 101, size=n_samples).clip(lower=18, upper=100) # Age around 70, correlated with risk
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.4, 0.6])  # Slight imbalance
    location = np.random.choice(['Home', 'Hospital', 'Nursing Home'], n_samples)
    cause = np.random.choice(['Slip', 'Trip', 'Dizziness', 'Other'], n_samples)
    time_of_day = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples)
    
    # Introduce correlation: higher age increases fall probability
    fall_prob = 0.1 + (age - 18) / (100 - 18) * 0.3  # Base 10%, up to 40% for age 100
    fall_occurred = np.random.binomial(1, fall_prob)  # Class imbalance: falls are rarer
    
    # Generate fall_date: random dates within last 12 months
    today = datetime.now()
    start_date = today - timedelta(days=365)
    fall_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    fall_date = [d.strftime('%Y-%m-%d') for d in fall_dates]
    
    # Generate fall_time based on time_of_day
    def get_random_time(time_of_day):
        if time_of_day == 'Morning':
            hour = np.random.randint(6, 12)
        elif time_of_day == 'Afternoon':
            hour = np.random.randint(12, 18)
        elif time_of_day == 'Evening':
            hour = np.random.randint(18, 24)
        elif time_of_day == 'Night':
            hour = np.random.randint(0, 6)
        minute = np.random.randint(0, 60)
        return f"{hour:02d}:{minute:02d}"
    
    fall_time = [get_random_time(tod) for tod in time_of_day]
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'location': location,
        'cause': cause,
        'time_of_day': time_of_day,
        'fall_occurred': fall_occurred,
        'fall_date': fall_date,
        'fall_time': fall_time
    })
    
    data['gender'] = data['gender'].where(pd.notna(data['gender']), None)
    
    return data

def summarize_dataset_properties(data):
    """
    Summarizes key properties of the dataset, including distributions, correlations, and missingness.
    
    Parameters:
    data (pd.DataFrame): The dataset to summarize.
    """
    print("Dataset Shape:", data.shape)
    print("\nColumn Data Types:\n", data.dtypes)
    print("\nSummary Statistics:\n", data.describe(include='all'))
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nClass Distribution (fall_occurred):\n", data['fall_occurred'].value_counts(normalize=True))
    
    # Correlations (only numeric columns)
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        print("\nCorrelations:\n", numeric_data.corr())
    else:
        print("\nNo numeric columns for correlation analysis.")

# Generate synthetic fall data
synthetic_fall_data = generate_synthetic_fall_data(1000)

# Summarize dataset properties
summarize_dataset_properties(synthetic_fall_data)

# Save to CSV
synthetic_fall_data.to_csv('synthetic_fall_data.csv', index=False)
print(f"\nSynthetic CSV generated with {len(synthetic_fall_data)} records! (Falls: {sum(synthetic_fall_data['fall_occurred'] == 1)}, No-Falls: {sum(synthetic_fall_data['fall_occurred'] == 0)})")