# ml/analyze_new_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your new data
df = pd.read_csv('data/all_data3.csv', header=None, 
                 names=['ch1', 'ch2', 'ch3', 'ch4', 'gesture'])

print("=== NEW DATA STATISTICS ===")
print(f"Total samples: {len(df)}")
print(f"Gestures collected: {df['gesture'].unique()}")

print("\n=== SAMPLES PER GESTURE ===")
print(df['gesture'].value_counts())

print("\n=== MEAN VALUES PER GESTURE ===")
for gesture in df['gesture'].unique():
    subset = df[df['gesture'] == gesture]
    print(f"\n{gesture.upper()} (n={len(subset)}):")
    print(f"  ch1: {subset['ch1'].mean():.1f} ± {subset['ch1'].std():.1f}")
    print(f"  ch2: {subset['ch2'].mean():.1f} ± {subset['ch2'].std():.1f}")
    print(f"  ch3: {subset['ch3'].mean():.1f} ± {subset['ch3'].std():.1f}")
    print(f"  ch4: {subset['ch4'].mean():.1f} ± {subset['ch4'].std():.1f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
channels = ['ch1', 'ch2', 'ch3', 'ch4']
titles = ['Flexor Carpi Radialis', 'Brachioradialis', 
          'Flexor Carpi Ulnaris', 'Flexor Digitorum Superficialis']

for idx, (ax, ch, title) in enumerate(zip(axes.flat, channels, titles)):
    for gesture in df['gesture'].unique():
        subset = df[df['gesture'] == gesture]
        ax.hist(subset[ch], alpha=0.5, label=gesture, bins=30)
    ax.set_title(title)
    ax.set_xlabel('ADC Value')
    ax.set_ylabel('Frequency')
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('ml/gesture_distributions.png')
print("\nDistribution plot saved to: ml/gesture_distributions.png")