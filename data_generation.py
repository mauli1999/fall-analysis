import pandas as pd
import random
from datetime import datetime, timedelta

# Generate synthetic fall and no-fall records with lowercase column names
data = []
for i in range(50):  # Generate 50 fall records
    fall_date = datetime.now() - timedelta(days=random.randint(1, 30))
    fall_hour = random.randint(0, 23)
    row = {
        'patient_id': f'P{random.randint(1000, 9999)}',
        'fall_date': fall_date.strftime('%Y-%m-%d'),
        'fall_time': f'{fall_hour:02d}:{random.randint(0, 59):02d}',
        'location': random.choice(['Bathroom', 'Bedroom', 'Hallway', 'Kitchen', 'Stairs']),
        'cause': random.choice(['Slippery floor', 'Medication side effect', 'Poor lighting', 'Tripped on object', 'Dizziness']),
        'injury': random.choice(['None', 'Bruise', 'Fracture', 'Head injury']),
        'notes': random.choice(['Patient was alone', 'Assisted by staff', 'Post-meal incident']),
        'fall_status': 1  # 1 indicates a fall
    }
    data.append(row)

# Duplicate data for no-fall instances and assign based on heuristic
no_fall_data = []
for row in data:
    # Heuristic: Less likely to fall between 2 AM and 5 AM (low activity)
    hour = int(row['fall_time'].split(':')[0])
    if hour >= 2 and hour <= 5 and random.random() < 0.7:  # 70% chance of no-fall in low-activity hours
        new_row = row.copy()
        new_row['fall_status'] = 0  # 0 indicates no-fall
        new_row['injury'] = 'None'  # No injury for no-fall
        new_row['notes'] = 'No incident reported'
        no_fall_data.append(new_row)
    else:
        # For other times, randomly assign some no-fall instances
        if random.random() < 0.3:  # 30% chance of no-fall outside low-activity hours
            new_row = row.copy()
            new_row['fall_status'] = 0
            new_row['injury'] = 'None'
            new_row['notes'] = 'No incident reported'
            no_fall_data.append(new_row)

data.extend(no_fall_data)

# Create DataFrame and save
df = pd.DataFrame(data)
df.to_csv('synthetic_fall_data.csv', index=False)
print(f"Synthetic CSV generated with {len(df)} records! (Falls: {sum(df['fall_status'] == 1)}, No-Falls: {sum(df['fall_status'] == 0)})")