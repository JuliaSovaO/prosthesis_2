# test_data.py
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('data/all_data3.csv', header=None, names=['ch1', 'ch2', 'ch3', 'ch4', 'gesture'])

print("Data statistics by gesture:")
for gesture in df['gesture'].unique():
    subset = df[df['gesture'] == gesture]
    print(f"\n{gesture} (n={len(subset)}):")
    print(f"  ch1: {subset['ch1'].mean():.1f} ± {subset['ch1'].std():.1f}")
    print(f"  ch2: {subset['ch2'].mean():.1f} ± {subset['ch2'].std():.1f}")
    print(f"  ch3: {subset['ch3'].mean():.1f} ± {subset['ch3'].std():.1f}")
    print(f"  ch4: {subset['ch4'].mean():.1f} ± {subset['ch4'].std():.1f}")

# Check for any anomalies
print("\n\nChecking for anomalies:")
print(f"Total samples: {len(df)}")
print(f"Unique values in each column:")
for col in ['ch1', 'ch2', 'ch3', 'ch4']:
    print(f"  {col}: {df[col].nunique()} unique values")