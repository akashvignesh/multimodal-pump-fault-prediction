#!/usr/bin/env python
"""Extract example data for all 3 classes."""
import pandas as pd
import json
import numpy as np

df = pd.read_csv('data/baseline_model/sensor_data/sensor.csv')

# Get samples for all 3 classes (3 of each)
normal_mask = df['machine_status'] == 'NORMAL'
normal_idx = np.where(normal_mask)[0][:3]
normal_sample = df.iloc[normal_idx].drop(['machine_status', 'timestamp'], axis=1, errors='ignore')
sensor_cols = [c for c in normal_sample.columns if c.startswith('sensor_')]
normal_sample = normal_sample[sensor_cols]

recovering_mask = df['machine_status'] == 'RECOVERING'
recovering_idx = np.where(recovering_mask)[0][:3]
recovering_sample = df.iloc[recovering_idx].drop(['machine_status', 'timestamp'], axis=1, errors='ignore')
recovering_sample = recovering_sample[sensor_cols]

broken_mask = df['machine_status'] == 'BROKEN'
broken_idx = np.where(broken_mask)[0][:3]
if len(broken_idx) > 0:
    broken_sample = df.iloc[broken_idx].drop(['machine_status', 'timestamp'], axis=1, errors='ignore')
    broken_sample = broken_sample[sensor_cols]
else:
    broken_sample = pd.DataFrame()

# Convert to JSON
normal_json = normal_sample.to_dict(orient='records')
recovering_json = recovering_sample.to_dict(orient='records')
broken_json = broken_sample.to_dict(orient='records') if len(broken_sample) > 0 else []

# Save for use in streamlit
with open('artifacts/example_data.json', 'w') as f:
    json.dump({
        'normal': normal_json,
        'recovering': recovering_json,
        'broken': broken_json
    }, f, indent=2)

print('✓ Saved example data (3 of each class) to artifacts/example_data.json')
print(f'  NORMAL: {len(normal_json)} examples')
print(f'  RECOVERING: {len(recovering_json)} examples')
print(f'  BROKEN: {len(broken_json)} examples')
