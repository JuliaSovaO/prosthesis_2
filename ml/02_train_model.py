# 02_quick_train.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load labeled data
print("Loading labeled data...")
data = np.load("emg_dataset.npz")
emg = data["emg"]
labels = data["labels"]
fs = data["sampling_rate"]

print(f"EMG shape: {emg.shape}")
print(f"Labels: {np.unique(labels, return_counts=True)}")


# ===== SIMPLE FEATURE EXTRACTION =====
def extract_simple_features(signal, window_size=250, step_size=25):
    """Extract 6 features per channel as in paper."""
    window_samples = int(window_size * fs / 1000)  # 83 samples at 332Hz
    step_samples = int(step_size * fs / 1000)  # 8 samples

    features = []
    for ch in range(3):  # For each channel
        channel_signal = signal[:, ch]

        channel_feats = []
        for start in range(0, len(channel_signal) - window_samples, step_samples):
            window = channel_signal[start : start + window_samples]

            # 6 time-domain features
            rms = np.sqrt(np.mean(window**2))
            var = np.var(window)
            mav = np.mean(np.abs(window))

            # Simplified SSC and ZC
            diffs = np.diff(window)
            ssc = np.sum(diffs[:-1] * diffs[1:] < 0)
            zc = np.sum(window[:-1] * window[1:] < 0)
            wl = np.sum(np.abs(diffs))

            channel_feats.append([rms, var, mav, ssc, zc, wl])

        features.append(channel_feats)

    # Combine channels: (n_windows, 18 features)
    features = np.array(features)  # (3, n_windows, 6)
    features = np.transpose(features, (1, 0, 2))  # (n_windows, 3, 6)
    features = features.reshape(features.shape[0], -1)  # (n_windows, 18)

    return features


print("\nExtracting features...")
X = extract_simple_features(emg)
print(f"Features shape: {X.shape}")

# Create labels for windows
window_samples = int(250 * fs / 1000)  # 83 samples
step_samples = int(25 * fs / 1000)  # 8 samples
n_windows = X.shape[0]

y = []
for i in range(n_windows):
    start = i * step_samples
    window_labels = labels[start : start + window_samples]
    if len(window_labels) > 0:
        y.append(np.bincount(window_labels).argmax())
    else:
        y.append(0)
y = np.array(y)

print(f"Window labels shape: {y.shape}")

# ===== TRAIN ANN (as in paper) =====
# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train ANN
print("\nTraining ANN...")
ann = MLPClassifier(
    hidden_layer_sizes=(300, 300, 300),
    activation="relu",
    solver="adam",
    alpha=0.001,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
)

ann.fit(X_train, y_train)

# Evaluate
y_pred = ann.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== RESULTS ===")
print(f"Accuracy: {accuracy:.3f}")

# Confusion matrix
label_names = [
    "Rest",
    "Rock",
    "Scissors",
    "Paper",
    "One",
    "Three",
    "Four",
    "Good",
    "Okay",
    "Finger Gun",
]

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred, normalize="true")
sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=label_names[: len(np.unique(y))],
    yticklabels=label_names[: len(np.unique(y))],
)
plt.title(f"ANN Confusion Matrix (Accuracy: {accuracy:.3f})")
plt.tight_layout()
plt.savefig("ann_results.png", dpi=150)
plt.show()

# Save model
import joblib

joblib.dump(ann, "emg_gesture_ann.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

print(f"\nModel saved: emg_gesture_ann.pkl")
print(f"Scaler saved: feature_scaler.pkl")
print(f"\nReady for real-time deployment!")
