# 03_extract_weights.py
import joblib
import numpy as np

# Load your trained model
model = joblib.load('emg_gesture_ann.pkl')
scaler = joblib.load('feature_scaler.pkl')

print("=== EXTRACTING MODEL WEIGHTS FOR STM32 ===")

# Extract ANN parameters (for scikit-learn MLPClassifier)
print("\nModel structure:")
print(f"  Input layer: 18 features")
print(f"  Hidden layers: {model.n_layers_ - 2} layers")
print(f"  Hidden layer sizes: {model.hidden_layer_sizes}")
print(f"  Output: {model.n_outputs_} classes")

# Save weights and biases
print("\nSaving weights to C header file...")

with open('emg_model_weights.h', 'w') as f:
    f.write("/* EMG Gesture Recognition Model - 91.3% Accuracy */\n")
    f.write("/* Generated from Python scikit-learn model */\n\n")
    f.write("#ifndef EMG_MODEL_WEIGHTS_H\n")
    f.write("#define EMG_MODEL_WEIGHTS_H\n\n")
    
    # Save scaler parameters
    f.write("/* Feature scaler (standardization) */\n")
    f.write(f"#define NUM_FEATURES {len(scaler.mean_)}\n")
    f.write("static const float SCALER_MEAN[NUM_FEATURES] = {\n    ")
    f.write(", ".join([f"{x:.6f}f" for x in scaler.mean_]))
    f.write("\n};\n\n")
    
    f.write("static const float SCALER_SCALE[NUM_FEATURES] = {\n    ")
    f.write(", ".join([f"{x:.6f}f" for x in scaler.scale_]))
    f.write("\n};\n\n")
    
    # Save coefs_ and intercepts_ (weights and biases)
    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        layer_type = "HIDDEN" if i < len(model.coefs_) - 1 else "OUTPUT"
        f.write(f"/* Layer {i+1} weights ({layer_type}) */\n")
        f.write(f"static const float WEIGHTS_{i}[{coef.shape[0]}][{coef.shape[1]}] = {{\n")
        
        for row in range(coef.shape[0]):
            f.write("    {")
            f.write(", ".join([f"{x:.6f}f" for x in coef[row]]))
            f.write("}")
            if row < coef.shape[0] - 1:
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        
        f.write(f"/* Layer {i+1} biases */\n")
        f.write(f"static const float BIASES_{i}[{len(intercept)}] = {{\n    ")
        f.write(", ".join([f"{x:.6f}f" for x in intercept]))
        f.write("\n};\n\n")
    
    f.write("#endif /* EMG_MODEL_WEIGHTS_H */\n")

print("✓ Saved: emg_model_weights.h")
print("\nNow copy this file to your STM32 project!")