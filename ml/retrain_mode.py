# ml/retrain_with_new_data_fixed.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== RETRAINING WITH CLEAN NEW DATA ===")

# Load ONLY new data
df = pd.read_csv('data/all_data3.csv', header=None,
                 names=['ch1', 'ch2', 'ch3', 'ch4', 'gesture'])

print(f"Training with {len(df)} clean samples")
print(f"Gesture distribution:")
print(df['gesture'].value_counts())

# Map gestures to numbers
gesture_mapping = {
    'rock': 0,
    'scissors': 1,
    'paper': 2,
    'fuck': 3,
    'three': 4,
    'four': 5,
    'good': 6,
    'okay': 7,
    'finger-gun': 8,
    'rest': 9
}

df['label'] = df['gesture'].map(gesture_mapping)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Feature extraction (same as before)
def extract_features(window):
    features = []
    for ch in range(4):
        signal = window[:, ch]
        mav = np.mean(np.abs(signal))
        rms = np.sqrt(np.mean(signal**2))
        var = np.var(signal)
        wl = np.sum(np.abs(np.diff(signal)))
        zc = np.sum(np.diff(np.sign(signal)) != 0)
        features.extend([mav, rms, var, wl, zc])
    return features

WINDOW_SIZE = 150
OVERLAP = 100

X = []
y = []
data_array = df[['ch1', 'ch2', 'ch3', 'ch4']].values
labels = df['label'].values

for i in range(0, len(data_array) - WINDOW_SIZE, WINDOW_SIZE - OVERLAP):
    window = data_array[i:i+WINDOW_SIZE]
    window_label = labels[i+WINDOW_SIZE//2]
    features = extract_features(window)
    X.append(features)
    y.append(window_label)

X = np.array(X)
y = np.array(y)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features per sample: {X.shape[1]}")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\nTraining logistic regression...")
base_model = LogisticRegression(
    C=0.1,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)

model = OneVsRestClassifier(base_model)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Train Accuracy: {accuracy_score(y_train, model.predict(X_train_scaled)):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(gesture_mapping.keys())))

# Save model
print("\nSaving model...")

coef_list = []
intercept_list = []
for estimator in model.estimators_:
    coef_list.append(estimator.coef_[0])
    intercept_list.append(estimator.intercept_[0])

coef = np.array(coef_list)
intercept = np.array(intercept_list)

# Save as C files (same format as before)
with open('ml/emg_model_new.h', 'w') as f:
    f.write('#ifndef EMG_MODEL_H\n')
    f.write('#define EMG_MODEL_H\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write('typedef enum {\n')
    for gesture, idx in gesture_mapping.items():
        f.write(f'    GESTURE_{gesture.upper().replace("-", "_")} = {idx},\n')
    f.write('} GestureType;\n\n')
    f.write(f'#define NUM_FEATURES {X.shape[1]}\n')
    f.write(f'#define NUM_CLASSES {len(gesture_mapping)}\n\n')
    f.write('extern const float scaler_mean[NUM_FEATURES];\n')
    f.write('extern const float scaler_scale[NUM_FEATURES];\n\n')
    f.write('extern const float lr_coefficients[NUM_CLASSES][NUM_FEATURES];\n')
    f.write('extern const float lr_intercept[NUM_CLASSES];\n\n')
    f.write('extern const char* gesture_names[NUM_CLASSES];\n\n')
    f.write('GestureType predict_gesture(const float* features);\n\n')
    f.write('#endif // EMG_MODEL_H\n')

with open('ml/emg_model_new.c', 'w') as f:
    f.write('#include "emg_model.h"\n\n')
    f.write('#include <math.h>\n\n')
    
    f.write('const float scaler_mean[NUM_FEATURES] = {\n')
    for mean_val in scaler.mean_:
        f.write(f'    {mean_val:.6f}f,\n')
    f.write('};\n\n')
    
    f.write('const float scaler_scale[NUM_FEATURES] = {\n')
    for scale_val in scaler.scale_:
        f.write(f'    {scale_val:.6f}f,\n')
    f.write('};\n\n')
    
    f.write('const float lr_coefficients[NUM_CLASSES][NUM_FEATURES] = {\n')
    for class_idx in range(len(coef)):
        f.write('    {\n')
        for feat_idx in range(len(coef[class_idx])):
            f.write(f'        {coef[class_idx][feat_idx]:.6f}f,\n')
        f.write('    },\n')
    f.write('};\n\n')
    
    f.write('const float lr_intercept[NUM_CLASSES] = {\n')
    for intercept_val in intercept:
        f.write(f'    {intercept_val:.6f}f,\n')
    f.write('};\n\n')
    
    f.write('const char* gesture_names[NUM_CLASSES] = {\n')
    for gesture in gesture_mapping.keys():
        f.write(f'    "{gesture}",\n')
    f.write('};\n\n')
    
    f.write('''
GestureType predict_gesture(const float* features) {
    float scores[NUM_CLASSES] = {0};
    float max_score = -INFINITY;
    int predicted_class = 0;
    
    float scaled_features[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (features[i] - scaler_mean[i]) / scaler_scale[i];
    }
    
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
        scores[class_idx] = lr_intercept[class_idx];
        for (int feat_idx = 0; feat_idx < NUM_FEATURES; feat_idx++) {
            scores[class_idx] += lr_coefficients[class_idx][feat_idx] * scaled_features[feat_idx];
        }
        if (scores[class_idx] > max_score) {
            max_score = scores[class_idx];
            predicted_class = class_idx;
        }
    }
    
    return (GestureType)predicted_class;
}
''')

# Save Python model
joblib.dump({
    'model': model,
    'scaler': scaler,
    'gesture_mapping': gesture_mapping,
    'data_stats': {
        'min_values': df[['ch1', 'ch2', 'ch3', 'ch4']].min().to_dict(),
        'max_values': df[['ch1', 'ch2', 'ch3', 'ch4']].max().to_dict(),
        'mean_values': df[['ch1', 'ch2', 'ch3', 'ch4']].mean().to_dict(),
    }
}, 'ml/emg_model_new.pkl')

print("New model saved to ml/emg_model_new.h and ml/emg_model_new.c")
print("\n=== IMPORTANT ===")
print("1. Replace emg_model.h and emg_model.c with the new files")
print("2. Rebuild and upload to STM32")
print("3. Test with real EMG signals")