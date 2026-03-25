"""
ANN Training Script for EMG Gesture Classification
Trains a neural network on the collected EMG data and generates C headers
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json
import os

def load_data(csv_path):
    """Load EMG data from CSV file"""
    df = pd.read_csv(csv_path, header=None)
    
    X = df.iloc[:, :4].values.astype(np.float32)
    y = df.iloc[:, 4].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("Gesture classes:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")
    
    return X, y_encoded, label_encoder

# Feature extraction: compute time-domain features from sliding window
def extract_features(X, y, window_size=50, step=25):
    """
    Extract time-domain features from raw EMG signals
    Features: RMS, MAV, VAR, WL, ZC, SSC
    """
    n_samples, n_channels = X.shape
    features = []
    labels = []
    
    for start in range(0, n_samples - window_size, step):
        end = start + window_size
        window = X[start:end, :]
        
        window_labels = y[start:end]
        label = np.bincount(window_labels).argmax()
        
        feat = []
        for ch in range(n_channels):
            ch_data = window[:, ch]
            
            # RMS
            rms = np.sqrt(np.mean(ch_data ** 2))
            
            # MAV
            mav = np.mean(np.abs(ch_data))
            
            # Variance
            var = np.var(ch_data)
            
            # Waveform Length
            wl = np.sum(np.abs(np.diff(ch_data)))
            
            # Zero Crossing
            zc = np.sum((ch_data[:-1] * ch_data[1:]) < 0)
            
            # Slope Sign Change
            diff1 = np.diff(ch_data)
            ssc = 0
            for i in range(1, len(diff1)):
                if diff1[i-1] * diff1[i] > 0:
                    ssc += 1
            
            feat.extend([rms, mav, var, wl, zc, ssc])
        
        features.append(feat)
        labels.append(label)
    
    return np.array(features), np.array(labels)

# Define the ANN model
def create_model(input_dim, num_classes, hidden_sizes=[64, 32, 16], activation='tanh'):
    """
    Create ANN model with specified architecture
    Using tanh activation as requested
    """
    model = models.Sequential()
    
    model.add(layers.Dense(hidden_sizes[0], activation=activation, input_shape=(input_dim,)))
    model.add(layers.Dropout(0.2))
    
    for size in hidden_sizes[1:]:
        model.add(layers.Dense(size, activation=activation))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Generate C header with weights
def generate_c_weights(model, label_encoder, scaler, output_dir='.'):
    """
    Extract weights from Keras model and generate C header files
    """
    weights = model.get_weights()
    
    c_code = []
    
    c_code.append("#ifndef ANN_WEIGHTS_H")
    c_code.append("#define ANN_WEIGHTS_H")
    c_code.append("")
    c_code.append("#include <stdint.h>")
    c_code.append("")
    c_code.append("// Gesture classes")
    c_code.append(f"#define ANN_NUM_CLASSES {len(label_encoder.classes_)}")
    c_code.append("")
    c_code.append("const char* ann_class_names[] = {")
    for name in label_encoder.classes_:
        c_code.append(f'    "{name}",')
    c_code.append("};")
    c_code.append("")
    
    layer_weights = []
    layer_biases = []
    for i, w in enumerate(weights):
        if i % 2 == 0:
            layer_weights.append(w)
        else:
            layer_biases.append(w)
    
    num_layers = len(layer_weights)
    
    layer_sizes = [model.input_shape[1]]
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            layer_sizes.append(layer.units)
    
    c_code.append("// Layer sizes")
    c_code.append(f"#define ANN_NUM_LAYERS {num_layers}")
    c_code.append("const int ann_layer_sizes[] = {")
    for i, sz in enumerate(layer_sizes):
        c_code.append(f"    {sz},  // Layer {i}")
    c_code.append("};")
    c_code.append("")
    
    for layer_idx in range(num_layers):
        w = layer_weights[layer_idx]
        b = layer_biases[layer_idx]
        
        c_code.append(f"// Layer {layer_idx} weights: {w.shape}")
        c_code.append(f"const float ann_weights_{layer_idx}[] = {{")
        flat = w.flatten()
        for j, val in enumerate(flat):
            if j % 8 == 0:
                c_code.append("    ")
            c_code.append(f"{val:.8f}f, ")
            if (j + 1) % 8 == 0 or j == len(flat) - 1:
                c_code.append("\n")
        c_code.append("};")
        c_code.append("")
        
        c_code.append(f"// Layer {layer_idx} bias: {b.shape}")
        c_code.append(f"const float ann_biases_{layer_idx}[] = {{")
        for j, val in enumerate(b):
            if j % 8 == 0:
                c_code.append("    ")
            c_code.append(f"{val:.8f}f, ")
            if (j + 1) % 8 == 0 or j == len(b) - 1:
                c_code.append("\n")
        c_code.append("};")
        c_code.append("")
    
    c_code.append("// Weights pointers array")
    c_code.append("const float* ann_weights_ptrs[] = {")
    for i in range(num_layers):
        c_code.append(f"    ann_weights_{i},")
    c_code.append("};")
    c_code.append("")
    
    c_code.append("// Biases pointers array")
    c_code.append("const float* ann_biases_ptrs[] = {")
    for i in range(num_layers):
        c_code.append(f"    ann_biases_{i},")
    c_code.append("};")
    c_code.append("")
    
    c_code.append("// Input scaling parameters (mean and std for each feature)")
    c_code.append("static const float ann_input_mean[] = {")
    for i, mean in enumerate(scaler.mean_):
        if i % 8 == 0:
            c_code.append("    ")
        c_code.append(f"{mean:.8f}f, ")
        if (i + 1) % 8 == 0 or i == len(scaler.mean_) - 1:
            c_code.append("\n")
    c_code.append("};")
    c_code.append("")
    
    c_code.append("static const float ann_input_std[] = {")
    for i, std in enumerate(scaler.scale_):
        if i % 8 == 0:
            c_code.append("    ")
        c_code.append(f"{std:.8f}f, ")
        if (i + 1) % 8 == 0 or i == len(scaler.scale_) - 1:
            c_code.append("\n")
    c_code.append("};")
    c_code.append("")
    
    c_code.append("#endif // ANN_WEIGHTS_H")
    
    with open(os.path.join(output_dir, 'weights.h'), 'w') as f:
        f.write('\n'.join(c_code))
    
    print(f"Generated weights.h in {output_dir}")

def main():
    print("=== EMG Gesture ANN Training ===\n")
    
    data_path = input("Enter path to all_data3.csv: ").strip()
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return None, None, None
    
    print(f"Loading data from {data_path}...")
    X_raw, y_encoded, label_encoder = load_data(data_path)
    print(f"Loaded {len(X_raw)} samples, {X_raw.shape[1]} channels")
    print(f"Classes: {label_encoder.classes_}")
    
    print("\nExtracting time-domain features...")
    print(f"Window size: 50 samples, Step: 25 samples")
    X_features, y_features = extract_features(X_raw, y_encoded, window_size=50, step=25)
    print(f"Extracted {len(X_features)} feature windows")
    print(f"Feature vector size: {X_features.shape[1]} (6 features × 4 channels)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_features, test_size=0.2, random_state=42, stratify=y_features
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nScaler parameters (for C implementation):")
    print("ann_input_mean = [", end="")
    for i, mean in enumerate(scaler.mean_):
        print(f"{mean:.4f}", end=", " if i < len(scaler.mean_)-1 else "")
    print("]")
    print("ann_input_std = [", end="")
    for i, std in enumerate(scaler.scale_):
        print(f"{std:.4f}", end=", " if i < len(scaler.scale_)-1 else "")
    print("]")
    
    input_dim = X_features.shape[1]
    num_classes = len(label_encoder.classes_)
    
    print(f"\nCreating ANN model: {input_dim} inputs → 64 → 32 → 16 → {num_classes} outputs")
    print("Activation function: tanh for hidden layers, softmax for output")
    
    model = create_model(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=[64, 32, 16],
        activation='tanh'
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
    
    print("\nGenerating C weights...")
    generate_c_weights(model, label_encoder, scaler, output_dir='.')
    
    model.save('emg_ann_model.h5')
    
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'classes': label_encoder.classes_.tolist(),
        'accuracy': float(accuracy),
        'input_dim': input_dim,
        'num_classes': num_classes
    }
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    print("\nDone! Generated:")
    print("  - weights.h (weights, biases, and normalization parameters)")
    print("  - emg_ann_model.h5 (full Keras model)")
    print("  - scaler_params.json (normalization parameters)")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    model, scaler, label_encoder = main()