import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings

# Suppress the deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Load data
print("Loading data...")
df = pd.read_csv('data/all_data2.csv', header=None, names=['ch1', 'ch2', 'ch3', 'ch4', 'gesture'])

# IMPORTANT: Based on your muscle-sensor mapping:
# ch1 = flexor carpi radialis (a0)
# ch2 = brachioradialis (a1) 
# ch3 = flexor carpi ulnaris (a2)
# ch4 = flexor digitorum superficialis (a3)

# Map gestures to numbers with EXACT definitions:
gesture_mapping = {
    'rock': 0,           # all closed
    'scissors': 1,       # index, middle opened, others closed
    'paper': 2,          # all opened
    'fuck': 3,           # middle finger opened, others closed
    'three': 4,          # index, middle, ring opened, others closed
    'four': 5,           # only thumb closed
    'good': 6,           # only thumb opened
    'okay': 7,           # index and thumb make circle, others opened
    'finger-gun': 8,     # index, thumb opened, others closed
    'rest': 9            # relaxed
}

print("Channel mapping:")
print("ch1 (a0): flexor carpi radialis")
print("ch2 (a1): brachioradialis")
print("ch3 (a2): flexor carpi ulnaris")
print("ch4 (a3): flexor digitorum superficialis")

print("\nGesture mapping:")
for gesture, idx in gesture_mapping.items():
    print(f"{gesture}: {idx}")

df['label'] = df['gesture'].map(gesture_mapping)

# Check for missing labels
missing = df[df['label'].isna()]['gesture'].unique()
if len(missing) > 0:
    print(f"\nWarning: Missing labels for gestures: {missing}")
    print("Available gestures:", df['gesture'].unique())

# Drop rows with missing labels
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"\nTotal samples: {len(df)}")
print("Gesture distribution:")
print(df['label'].value_counts().sort_index())

# Feature extraction functions (optimized for STM32)
def extract_features(window):
    """Extract features for a window of data (4 channels)"""
    features = []
    
    for ch in range(4):  # 4 channels
        signal = window[:, ch]
        
        # 1. Mean Absolute Value (MAV) - simple and effective
        mav = np.mean(np.abs(signal))
        
        # 2. Root Mean Square (RMS) - good for EMG
        rms = np.sqrt(np.mean(signal**2))
        
        # 3. Variance (VAR)
        var = np.var(signal)
        
        # 4. Waveform Length (WL) - sum of absolute differences
        wl = np.sum(np.abs(np.diff(signal)))
        
        # 5. Zero Crossing (ZC) - count sign changes
        zc = np.sum(np.diff(np.sign(signal)) != 0)
        
        features.extend([mav, rms, var, wl, zc])
    
    return features

# Prepare sliding windows
WINDOW_SIZE = 150  # 150ms at 1000Hz = 150 samples
OVERLAP = 100      # 100ms overlap = 67% overlap

X = []
y = []

data_array = df[['ch1', 'ch2', 'ch3', 'ch4']].values
labels = df['label'].values

for i in range(0, len(data_array) - WINDOW_SIZE, WINDOW_SIZE - OVERLAP):
    window = data_array[i:i+WINDOW_SIZE]
    window_label = labels[i+WINDOW_SIZE//2]  # Label for middle of window
    
    features = extract_features(window)
    X.append(features)
    y.append(window_label)

X = np.array(X)
y = np.array(y)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Number of features per sample: {X.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression with OneVsRest strategy
print("\nTraining logistic regression...")
base_model = LogisticRegression(
    C=0.1,  # Strong regularization to prevent overfitting
    penalty='l2',
    solver='liblinear',  # Good for small datasets
    max_iter=1000,
    random_state=42
)

# Wrap with OneVsRestClassifier for multiclass
model = OneVsRestClassifier(base_model)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Train Accuracy: {accuracy_score(y_train, model.predict(X_train_scaled)):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(gesture_mapping.keys())))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model for STM32
print("\nSaving model for STM32...")

# Access coefficients from the underlying estimators
coef_list = []
intercept_list = []

for estimator in model.estimators_:
    coef_list.append(estimator.coef_[0])  # Get the coefficient vector
    intercept_list.append(estimator.intercept_[0])  # Get the intercept

# Convert to arrays
coef = np.array(coef_list)  # shape: (n_classes, n_features)
intercept = np.array(intercept_list)  # shape: (n_classes,)

print(f"Coefficient shape: {coef.shape}")
print(f"Intercept shape: {intercept.shape}")

# Calculate model size
model_size_bytes = (coef.size + intercept.size) * 4  # 4 bytes per float
print(f"Model size: {model_size_bytes} bytes")
print(f"Number of parameters: {coef.size + intercept.size}")

# Save as C header file
with open('ml/emg_model.h', 'w') as f:
    f.write('#ifndef EMG_MODEL_H\n')
    f.write('#define EMG_MODEL_H\n\n')
    
    f.write('#include <stdint.h>\n\n')
    
    # IMPORTANT: Update gesture mapping to match your definitions
    f.write('typedef enum {\n')
    f.write('    // Based on exact definitions:\n')
    f.write('    GESTURE_ROCK = 0,           // all closed\n')
    f.write('    GESTURE_SCISSORS = 1,       // index, middle opened, others closed\n')
    f.write('    GESTURE_PAPER = 2,          // all opened\n')
    f.write('    GESTURE_FUCK = 3,           // middle finger opened, others closed\n')
    f.write('    GESTURE_THREE = 4,          // index, middle, ring opened, others closed\n')
    f.write('    GESTURE_FOUR = 5,           // only thumb closed\n')
    f.write('    GESTURE_GOOD = 6,           // only thumb opened\n')
    f.write('    GESTURE_OKAY = 7,           // index and thumb make circle, others opened\n')
    f.write('    GESTURE_FINGER_GUN = 8,     // index, thumb opened, others closed\n')
    f.write('    GESTURE_REST = 9            // relaxed\n')
    f.write('} GestureType;\n\n')
    
    # Model parameters
    f.write(f'#define NUM_FEATURES {X.shape[1]}\n')
    f.write(f'#define NUM_CLASSES {len(gesture_mapping)}\n\n')
    
    f.write('// Channel mapping (CRITICAL!):\n')
    f.write('// ch1: flexor carpi radialis (a0)\n')
    f.write('// ch2: brachioradialis (a1)\n')
    f.write('// ch3: flexor carpi ulnaris (a2)\n')
    f.write('// ch4: flexor digitorum superficialis (a3)\n\n')
    
    f.write('// Feature scaling parameters\n')
    f.write(f'extern const float scaler_mean[{X.shape[1]}];\n')
    f.write(f'extern const float scaler_scale[{X.shape[1]}];\n\n')
    
    f.write('// Logistic regression coefficients\n')
    f.write(f'extern const float lr_coefficients[NUM_CLASSES][NUM_FEATURES];\n')
    f.write(f'extern const float lr_intercept[NUM_CLASSES];\n\n')
    
    f.write('// Gesture names\n')
    f.write('extern const char* gesture_names[NUM_CLASSES];\n\n')
    
    f.write('#endif // EMG_MODEL_H\n')

# Save as C source file
with open('ml/emg_model.c', 'w') as f:
    f.write('#include "emg_model.h"\n\n')
    f.write('#include <math.h>\n\n')
    
    # Scaling parameters
    f.write('const float scaler_mean[NUM_FEATURES] = {\n')
    for mean_val in scaler.mean_:
        f.write(f'    {mean_val:.6f}f,\n')
    f.write('};\n\n')
    
    f.write('const float scaler_scale[NUM_FEATURES] = {\n')
    for scale_val in scaler.scale_:
        f.write(f'    {scale_val:.6f}f,\n')
    f.write('};\n\n')
    
    # Coefficients
    f.write('const float lr_coefficients[NUM_CLASSES][NUM_FEATURES] = {\n')
    for class_idx in range(len(coef)):
        f.write('    {\n')
        for feat_idx in range(len(coef[class_idx])):
            f.write(f'        {coef[class_idx][feat_idx]:.6f}f,\n')
        f.write('    },\n')
    f.write('};\n\n')
    
    # Intercepts
    f.write('const float lr_intercept[NUM_CLASSES] = {\n')
    for intercept_val in intercept:
        f.write(f'    {intercept_val:.6f}f,\n')
    f.write('};\n\n')
    
    # Gesture names
    f.write('const char* gesture_names[NUM_CLASSES] = {\n')
    for gesture in gesture_mapping.keys():
        f.write(f'    "{gesture}",\n')
    f.write('};\n\n')
    
    # Helper function for prediction
    f.write('''
GestureType predict_gesture(const float* features) {
    float scores[NUM_CLASSES] = {0};
    float max_score = -INFINITY;
    int predicted_class = 0;
    
    // Scale features
    float scaled_features[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
    }
    
    // Calculate scores for each class (One-vs-Rest)
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
        scores[class_idx] = lr_intercept[class_idx];
        
        for (int feat_idx = 0; feat_idx < NUM_FEATURES; feat_idx++) {
            scores[class_idx] += lr_coefficients[class_idx][feat_idx] * scaled_features[feat_idx];
        }
        
        // Keep track of maximum score
        if (scores[class_idx] > max_score) {
            max_score = scores[class_idx];
            predicted_class = class_idx;
        }
    }
    
    return (GestureType)predicted_class;
}
''')

print("Model saved to ml/emg_model.h and ml/emg_model.c")

# Also save Python model for reference
joblib.dump({
    'model': model,
    'scaler': scaler,
    'gesture_mapping': gesture_mapping,
    'channel_mapping': {
        'ch1': 'flexor carpi radialis (a0)',
        'ch2': 'brachioradialis (a1)',
        'ch3': 'flexor carpi ulnaris (a2)',
        'ch4': 'flexor digitorum superficialis (a3)'
    }
}, 'ml/emg_model.pkl')

print("\nDone!")