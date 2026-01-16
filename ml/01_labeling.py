import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load your raw data
print("Loading data...")
data = pd.read_csv("raw_data.csv", header=None, names=["ch1", "ch2", "ch3"])
print(f"Total rows: {len(data)}")
print(f"Duration: {len(data)/332:.1f} seconds")  # 332 Hz sampling

# ===== CALCULATE EXACT TIMING =====
fs = 332  # Your sampling frequency
total_time_seconds = len(data) / fs
print(f"\nTotal recording time: {total_time_seconds:.1f} seconds")
print(f"Expected: ~175 seconds (2:55)")

# ===== CREATE LABELS BASED ON YOUR SEQUENCE =====
# Each gesture: 5 seconds, Rest: 5 or 10 seconds
samples_per_5s = int(5 * fs)  # 5 seconds at 332 Hz = 1660 samples
samples_per_10s = int(10 * fs)  # 10 seconds = 3320 samples

# Initialize labels array (0 = rest, 1-9 = gestures)
labels = np.zeros(len(data), dtype=int)

# Gesture mapping (as in paper)
gesture_codes = {
    "rest": 0,
    "rock": 1,
    "scissors": 2,
    "paper": 3,
    "one": 4,
    "three": 5,
    "four": 6,
    "good": 7,
    "okay": 8,
    "finger_gun": 9,
}

# ===== MANUAL ADJUSTMENT BASED ON YOUR TIMING =====
# You might need to adjust these start positions!
# Watch your video and note exact timestamps

current_sample = 0

# Sequence according to your description:
sequence = [
    ("rest", 10),  # 0-10s: rest
    ("rock", 5),  # 10-15s: rock
    ("rest", 5),  # 15-20s: rest
    ("rock", 5),  # 20-25s: rock
    ("rest", 5),  # 25-30s: rest
    ("scissors", 5),  # 30-35s: scissors
    ("rest", 5),  # 35-40s: rest
    ("scissors", 5),  # 40-45s: scissors
    ("rest", 5),  # 45-50s: rest
    ("paper", 5),  # 50-55s: paper
    ("rest", 5),  # 55-60s: rest
    ("paper", 5),  # 60-65s: paper
    ("rest", 5),  # 65-70s: rest
    ("one", 5),  # 70-75s: one
    ("rest", 5),  # 75-80s: rest
    ("one", 5),  # 80-85s: one
    ("rest", 5),  # 85-90s: rest
    ("three", 5),  # 90-95s: three
    ("rest", 5),  # 95-100s: rest
    ("three", 5),  # 100-105s: three
    ("rest", 5),  # 105-110s: rest
    ("four", 5),  # 110-115s: four
    ("rest", 5),  # 115-120s: rest
    ("four", 5),  # 120-125s: four
    ("rest", 5),  # 125-130s: rest
    ("good", 5),  # 130-135s: good
    ("rest", 5),  # 135-140s: rest
    ("good", 5),  # 140-145s: good
    ("rest", 5),  # 145-150s: rest
    ("okay", 5),  # 150-155s: okay
    ("rest", 5),  # 155-160s: rest
    ("okay", 5),  # 160-165s: okay
    ("rest", 5),  # 165-170s: rest
    ("finger_gun", 5),  # 170-175s: finger gun
    ("rest", 5),  # 175-180s: rest
    ("finger_gun", 5),  # 180-185s: finger gun
    ("rest", 10),  # 185-195s: rest to end
]

# Apply labels
print("\nApplying labels...")
for gesture, duration in sequence:
    samples = int(duration * fs)
    end_sample = current_sample + samples

    if end_sample > len(data):
        end_sample = len(data)  # Truncate if needed

    labels[current_sample:end_sample] = gesture_codes[gesture]

    print(
        f"  {gesture:12s} ({duration:2d}s): samples {current_sample:6d} to {end_sample:6d}"
    )

    current_sample = end_sample

    if current_sample >= len(data):
        break

print(f"\nLabeled {current_sample} of {len(data)} samples")
print(f"Remaining samples: {len(data) - current_sample}")

# ===== VISUALIZE LABELS =====
plt.figure(figsize=(15, 8))

# Plot EMG signals with labels
for ch in range(3):
    plt.subplot(4, 1, ch + 1)
    plt.plot(data.iloc[:5000, ch], alpha=0.7, label=f"CH{ch+1}")
    plt.ylabel(f"CH{ch+1}")
    plt.legend(loc="upper right")

# Plot labels
plt.subplot(4, 1, 4)
unique_labels = np.unique(labels[:5000])
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for label in unique_labels:
    if label == 0:  # Rest
        continue
    mask = labels[:5000] == label
    plt.fill_between(
        np.arange(5000),
        0,
        1,
        where=mask,
        alpha=0.3,
        color=colors[label],
        label=f"Gesture {label}",
    )

plt.xlabel("Sample")
plt.ylabel("Gesture")
plt.ylim(0, 1)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("labeled_data_visualization.png", dpi=150)
plt.show()

# ===== CHECK LABEL DISTRIBUTION =====
print("\n=== LABEL DISTRIBUTION ===")
label_counts = pd.Series(labels).value_counts().sort_index()
label_names = [
    "rest",
    "rock",
    "scissors",
    "paper",
    "one",
    "three",
    "four",
    "good",
    "okay",
    "finger_gun",
]

for idx, count in label_counts.items():
    if idx < len(label_names):
        seconds = count / fs
        print(f"{label_names[idx]:12s}: {count:6d} samples ({seconds:5.1f}s)")

# ===== SAVE LABELED DATA =====
print("\nSaving labeled data...")
labeled_data = pd.DataFrame(
    {"ch1": data["ch1"], "ch2": data["ch2"], "ch3": data["ch3"], "label": labels}
)

labeled_data.to_csv("labeled_emg_data.csv", index=False)
np.savez("emg_dataset.npz", emg=data.values, labels=labels, sampling_rate=fs)

print("\n=== DATA READY FOR ML ===")
print(f"Saved: labeled_emg_data.csv")
print(f"Saved: emg_dataset.npz")
print(f"Sampling rate: {fs} Hz")
print(f"Total samples: {len(data)}")
print(f"Total labels: {len(np.unique(labels))} classes")
print("\nNext: Run feature extraction and training!")
