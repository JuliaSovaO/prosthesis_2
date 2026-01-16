# 08_extract_small_weights.py
import joblib
import numpy as np

# Load small model
model = joblib.load('emg_small_model.pkl')
scaler = joblib.load('feature_scaler_small.pkl')

print("=== EXTRACTING SMALL MODEL WEIGHTS FOR STM32 ===")
print(f"Model structure: {model.hidden_layer_sizes}")
print(f"Total parameters: {sum(c.size for c in model.coefs_) + sum(i.size for i in model.intercepts_)}")

with open('emg_small_weights.h', 'w') as f:
    f.write("/* Small EMG Model for STM32F401 */\n")
    f.write("/* Memory optimized: ~6KB */\n\n")
    f.write("#ifndef EMG_SMALL_WEIGHTS_H\n")
    f.write("#define EMG_SMALL_WEIGHTS_H\n\n")
    
    # Save scaler parameters
    f.write("/* Feature scaler */\n")
    f.write(f"#define NUM_FEATURES {len(scaler.mean_)}\n")
    f.write("#define HIDDEN1_SIZE 32\n")
    f.write("#define HIDDEN2_SIZE 16\n")
    f.write("#define NUM_GESTURES 10\n\n")
    
    f.write("static const float SCALER_MEAN[NUM_FEATURES] = {\n    ")
    f.write(", ".join([f"{x:.6f}f" for x in scaler.mean_]))
    f.write("\n};\n\n")
    
    f.write("static const float SCALER_SCALE[NUM_FEATURES] = {\n    ")
    f.write(", ".join([f"{x:.6f}f" for x in scaler.scale_]))
    f.write("\n};\n\n")
    
    # Save weights and biases
    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        layer_type = "HIDDEN1" if i == 0 else ("HIDDEN2" if i == 1 else "OUTPUT")
        input_size = coef.shape[0]
        output_size = coef.shape[1]
        
        f.write(f"/* Layer {i+1} weights ({layer_type}) */\n")
        f.write(f"static const float WEIGHTS_{i}[{input_size}][{output_size}] = {{\n")
        
        for row in range(input_size):
            f.write("    {")
            f.write(", ".join([f"{x:.6f}f" for x in coef[row]]))
            f.write("}")
            if row < input_size - 1:
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        
        f.write(f"/* Layer {i+1} biases */\n")
        f.write(f"static const float BIASES_{i}[{len(intercept)}] = {{\n    ")
        f.write(", ".join([f"{x:.6f}f" for x in intercept]))
        f.write("\n};\n\n")
    
    f.write("#endif /* EMG_SMALL_WEIGHTS_H */\n")

print("✓ Saved: emg_small_weights.h")