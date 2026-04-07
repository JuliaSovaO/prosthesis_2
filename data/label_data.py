"""
Label EMG data using manually identified gesture ranges
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Gesture mapping
gesture_map = {
    "rock": 0, "scissors": 1, "paper": 2, "fuck": 3,
    "three": 4, "four": 5, "good": 6, "okay": 7,
    "finger-gun": 8, "rest": 9
}

gesture_names = {v: k for k, v in gesture_map.items()}

# YOUR GESTURE RANGES - based on visual inspection
gesture_ranges = [
    (500, 2700, 'rest'),
    
    # First rock
    (3500, 5000, 'rock'),
    (5500, 6250, 'rock'),
    
    # Scissors  
    (8000, 9000, 'scissors'),
    (9500, 10700, 'scissors'),
    
    # Paper
    (16000, 17000, 'paper'),
    (18250, 19000, 'paper'),
    
    # Fuck (middle finger)
    (26800, 29200, 'fuck'),
    (31450, 32750, 'fuck'),
    
    # Three
    (37000, 38150, 'three'),
    (39050, 41000, 'three'),
    
    # Four
    (43350, 44700, 'four'),
    (45500, 47300, 'four'),
    
    # Good
    (50400, 51400, 'good'),
    (52950, 55300, 'good'),
    
    # Okay
    (56100, 57600, 'okay'),
    (58750, 59200, 'okay'),
    
    # Finger-gun
    (61400, 62650, 'finger-gun'),
    (64800, 66000, 'finger-gun'),

    (66700, 67500, 'rest'),
]

def load_and_label_data(data_path="data/all_g.txt"):
    """Load data and apply labels based on ranges"""
    
    # Load data
    df = pd.read_csv(data_path, header=None, names=['ch0', 'ch1', 'ch2', 'ch3'])
    print(f"Loaded {len(df)} samples ({len(df)/1000:.1f} seconds)")
    
    # Initialize all labels as -1 (unlabeled)
    labels = np.full(len(df), -1, dtype=int)
    
    # Apply labels from ranges
    print("\nApplying labels:")
    for start, end, gesture_name in gesture_ranges:
        # Ensure ranges are within bounds
        start = max(0, start)
        end = min(len(df), end)
        
        gesture_code = gesture_map[gesture_name]
        labels[start:end] = gesture_code
        print(f"  {gesture_name:12} | samples {start:6d} - {end:6d} | {end-start:5d} samples ({(end-start)/1000:.2f}s)")
    
    # Everything unlabeled becomes 'rest'
    unlabeled = np.sum(labels == -1)
    if unlabeled > 0:
        print(f"\n⚠️ {unlabeled} samples were unlabeled - setting to 'rest'")
        labels[labels == -1] = gesture_map['rest']
    
    # Add labels to dataframe
    df['gesture'] = labels
    
    # Verify labeling
    print("\n" + "="*60)
    print("FINAL LABEL DISTRIBUTION:")
    print("="*60)
    for code in sorted(df['gesture'].unique()):
        count = np.sum(df['gesture'] == code)
        if count > 0:
            print(f"  {gesture_names[code]:12}: {count:6d} samples ({count/1000:5.1f} seconds)")
    
    return df

def visualize_labeled_data(df, save_dir="plots"):
    """Create visualization of labeled data"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Create combined signal
    combined = (df['ch1'].values + df['ch2'].values) / 2
    
    colors = {
        'rock': '#E41A1C', 'scissors': '#377EB8', 'paper': '#4DAF4A',
        'fuck': '#984EA3', 'three': '#FF7F00', 'four': '#FFFF33',
        'good': '#A65628', 'okay': '#F781BF', 'finger-gun': '#999999',
        'rest': '#E5E5E5'
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    
    # Section 1: 0-20000
    ax = axes[0]
    end1 = min(20000, len(df))
    for gesture_code in df['gesture'].unique():
        gesture_name = gesture_names[gesture_code]
        mask = df['gesture'][:end1] == gesture_code
        if mask.any():
            diff_mask = np.diff(np.concatenate(([0], mask.astype(int), [0])))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]
            for start, end in zip(starts, ends):
                if end - start > 20:
                    ax.axvspan(start, end, alpha=0.3, color=colors[gesture_name])
    ax.plot(combined[:end1], 'b-', linewidth=0.8, alpha=0.7)
    ax.set_title('Samples 0-20000', fontsize=12)
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    # Section 2: 20000-40000
    ax = axes[1]
    start2, end2 = 20000, min(40000, len(df))
    for gesture_code in df['gesture'].unique():
        gesture_name = gesture_names[gesture_code]
        mask = df['gesture'][start2:end2] == gesture_code
        if mask.any():
            diff_mask = np.diff(np.concatenate(([0], mask.astype(int), [0])))
            starts = np.where(diff_mask == 1)[0] + start2
            ends = np.where(diff_mask == -1)[0] + start2
            for start, end in zip(starts, ends):
                if end - start > 20:
                    ax.axvspan(start, end, alpha=0.3, color=colors[gesture_name])
    ax.plot(combined[start2:end2], 'b-', linewidth=0.8, alpha=0.7)
    ax.set_title('Samples 20000-40000', fontsize=12)
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    # Section 3: 40000-end
    ax = axes[2]
    start3 = 40000
    for gesture_code in df['gesture'].unique():
        gesture_name = gesture_names[gesture_code]
        mask = df['gesture'][start3:] == gesture_code
        if mask.any():
            diff_mask = np.diff(np.concatenate(([0], mask.astype(int), [0])))
            starts = np.where(diff_mask == 1)[0] + start3
            ends = np.where(diff_mask == -1)[0] + start3
            for start, end in zip(starts, ends):
                if end - start > 20:
                    ax.axvspan(start, end, alpha=0.3, color=colors[gesture_name])
    ax.plot(combined[start3:], 'b-', linewidth=0.8, alpha=0.7)
    ax.set_title(f'Samples 40000-{len(df)}', fontsize=12)
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/labeled_data_verification.png", dpi=150)
    plt.show()
    print(f"✅ Saved: {save_dir}/labeled_data_verification.png")

def save_labeled_data(df, output_path="data/all_data_labeled.csv"):
    """Save labeled data to CSV WITHOUT headers"""
    # Save without header to match training script expectation
    df[['ch0', 'ch1', 'ch2', 'ch3', 'gesture']].to_csv(output_path, index=False, header=False)
    print(f"\n✅ Labeled data saved to: {output_path}")
    
    # Also save a sample for quick testing
    sample_size = min(50000, len(df))
    sampled = df[['ch0', 'ch1', 'ch2', 'ch3', 'gesture']].sample(n=sample_size, random_state=42)
    sampled.to_csv("data/all_data_labeled_sample.csv", index=False, header=False)
    print(f"✅ Sample saved to: data/all_data_labeled_sample.csv")

def main():
    print("=== EMG Data Labeling with Manual Ranges ===\n")
    
    # Load and label data
    df = load_and_label_data("data/all_g.txt")
    
    # Visualize results
    visualize_labeled_data(df)
    
    # Save labeled data
    save_labeled_data(df)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. Review the verification plot 'plots/labeled_data_verification.png'")
    print("2. If labels look correct, train the ANN model:")
    print("\n   python src/src_cube/ann/ann_train_realistic.py")
    print("\n   Make sure to update the data path in the training script to:")
    print("   data_path = 'data/all_data_labeled.csv'")
    print("\n3. After training, upload the new model to STM32")

if __name__ == "__main__":
    main()