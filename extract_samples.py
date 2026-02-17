#!/usr/bin/env python
"""Extract real NORMAL and RECOVERING samples from baseline training data."""
import pandas as pd
import json
import numpy as np

df = pd.read_csv('data/baseline_model/sensor_data/sensor.csv')
print('Total rows:', len(df))
print('Labels:', df['machine_status'].value_counts().to_dict())

# Get samples for NORMAL (5 consecutive rows)
normal_mask = df['machine_status'] == 'NORMAL'
normal_idx = np.where(normal_mask)[0][:5]
normal_sample = df.iloc[normal_idx].drop(['machine_status', 'timestamp'], axis=1, errors='ignore')
# Keep only sensor columns
sensor_cols = [c for c in normal_sample.columns if c.startswith('sensor_')]
normal_sample = normal_sample[sensor_cols]

# Get samples for RECOVERING (5 consecutive rows)
recovering_mask = df['machine_status'] == 'RECOVERING'
recovering_idx = np.where(recovering_mask)[0][:5]
recovering_sample = df.iloc[recovering_idx].drop(['machine_status', 'timestamp'], axis=1, errors='ignore')
# Keep only sensor columns
recovering_sample = recovering_sample[sensor_cols]

# Convert to JSON
normal_json = normal_sample.to_dict(orient='records')
recovering_json = recovering_sample.to_dict(orient='records')

print("\n=== NORMAL (5 rows) ===")
print("First row sample:")
print(json.dumps(normal_json[0], indent=2)[:300])

print("\n=== RECOVERING (5 rows) ===")
print("First row sample:")
print(json.dumps(recovering_json[0], indent=2)[:300])

# Save for use in streamlit
with open('artifacts/sample_data.json', 'w') as f:
    json.dump({
        'normal': normal_json,
        'recovering': recovering_json
    }, f, indent=2)
print("\n✓ Saved to artifacts/sample_data.json")
