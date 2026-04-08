import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json
import os

tf.keras.backend.set_floatx('float32')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :4].values.astype(np.float32)
    y = df.iloc[:, 4].values.astype(np.int32)
    
    # Filter out rest samples (class 9)
    gesture_mask = y != 9
    X = X[gesture_mask]
    y = y[gesture_mask]
    
    gesture_names = {
        0: "rock", 1: "scissors", 2: "paper", 3: "fuck",
        4: "three", 5: "four", 6: "good", 7: "okay", 8: "finger-gun"
    }
    
    print(f"Filtered data: {len(X)} gesture samples (rest removed)")
    
    # Re-encode labels to 0-8
    unique_gestures = np.unique(y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    label_encoder.classes_ = np.array([gesture_names[int(c)] for c in unique_gestures])
    
    print("\nGesture classes (9 gestures total):")
    for i, name in enumerate(label_encoder.classes_):
        print(f"  {i}: {name}")
    
    return X, y_encoded, label_encoder

def extract_features_simple(X, y, window_size=50, step=25):
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
            
            rms = np.sqrt(np.mean(ch_data ** 2))
            mav = np.mean(np.abs(ch_data))
            var = np.var(ch_data)
            wl = np.sum(np.abs(np.diff(ch_data)))
            
            # Clip to prevent overflow
            rms = min(rms, 10000)
            mav = min(mav, 10000)
            var = min(var, 1e8)
            wl = min(wl, 1e6)
            
            feat.extend([rms, mav, var, wl])
        
        features.append(feat)
        labels.append(label)
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

def create_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    print("=== EMG ANN Training (Gestures Only - No Rest) ===\n")
    
    data_path = "data/all_data_labeled.csv"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
    
    print(f"Loading data...")
    X_raw, y_encoded, label_encoder = load_and_prepare_data(data_path)
    print(f"Loaded {len(X_raw)} gesture samples")
    
    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(y_encoded, return_counts=True)
    for code, count in zip(unique, counts):
        print(f"  {label_encoder.classes_[code]}: {count} samples")
    
    print("\nExtracting features...")
    X_features, y_features = extract_features_simple(X_raw, y_encoded)
    print(f"Extracted {len(X_features)} windows, {X_features.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_features, test_size=0.2, random_state=42, stratify=y_features
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create model
    input_dim = X_features.shape[1]
    num_classes = len(label_encoder.classes_)
    
    print(f"\nModel: {input_dim} inputs -> 64 -> 32 -> {num_classes}")
    model = create_model(input_dim, num_classes)
    model.summary()
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
        ],
        verbose=1
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
    
    # Generate C headers
    generate_c_headers(model, label_encoder, scaler)
    
    print(f"\nFinal test accuracy: {acc:.4f} ({acc*100:.1f}%)")

def generate_c_headers(model, label_encoder, scaler):
    weights = []
    biases = []
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            w, b = layer.get_weights()
            weights.append(w)
            biases.append(b)
    
    class_names = label_encoder.classes_.tolist()
    
    # Write weights.h
    with open('src/src_cube/ann/weights.h', 'w') as f:
        f.write('#ifndef ANN_WEIGHTS_H\n#define ANN_WEIGHTS_H\n\n')
        f.write('#include <stdint.h>\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n\n')
        
        f.write(f'#define ANN_INPUT_SIZE {scaler.mean_.shape[0]}\n')
        f.write(f'#define ANN_WINDOW_SIZE 50\n')
        f.write(f'#define ANN_NUM_CLASSES {len(class_names)}\n')
        f.write(f'#define ANN_NUM_LAYERS {len(weights)}\n\n')
        
        f.write('extern const char* ann_class_names[];\n')
        f.write('extern const int ann_layer_sizes[];\n\n')
        
        for i in range(len(weights)):
            f.write(f'extern const float ann_weights_{i}[];\n')
            f.write(f'extern const float ann_biases_{i}[];\n\n')
        
        f.write('extern const float* ann_weights_ptrs[];\n')
        f.write('extern const float* ann_biases_ptrs[];\n')
        f.write('extern const float ann_input_mean[];\n')
        f.write('extern const float ann_input_std[];\n\n')
        
        f.write('#ifdef __cplusplus\n}\n#endif\n\n#endif\n')
    
    # Write weights.c
    with open('src/src_cube/ann/weights.c', 'w') as f:
        f.write('#include "weights.h"\n\n')
        
        f.write('const char* ann_class_names[] = {\n')
        for name in class_names:
            f.write(f'    "{name}",\n')
        f.write('};\n\n')
        
        # Layer sizes
        layer_sizes = [model.input_shape[1]]
        for layer in model.layers:
            if isinstance(layer, layers.Dense):
                layer_sizes.append(layer.units)
        f.write('const int ann_layer_sizes[] = {\n')
        for i, sz in enumerate(layer_sizes):
            f.write(f'    {sz},\n')
        f.write('};\n\n')
        
        # Weights and biases
        for i, (w, b) in enumerate(zip(weights, biases)):
            f.write(f'const float ann_weights_{i}[] = {{\n')
            flat = w.flatten()
            for j, val in enumerate(flat):
                if j % 8 == 0:
                    f.write('    ')
                f.write(f'{val:.6f}f')
                if j < len(flat) - 1:
                    f.write(', ')
                if (j + 1) % 8 == 0:
                    f.write('\n')
            if len(flat) % 8 != 0:
                f.write('\n')
            f.write('};\n\n')
            
            f.write(f'const float ann_biases_{i}[] = {{\n')
            for j, val in enumerate(b):
                if j % 8 == 0:
                    f.write('    ')
                f.write(f'{val:.6f}f')
                if j < len(b) - 1:
                    f.write(', ')
                if (j + 1) % 8 == 0:
                    f.write('\n')
            if len(b) % 8 != 0:
                f.write('\n')
            f.write('};\n\n')
        
        # Pointers
        f.write('const float* ann_weights_ptrs[] = {\n')
        for i in range(len(weights)):
            f.write(f'    ann_weights_{i},\n')
        f.write('};\n\n')
        
        f.write('const float* ann_biases_ptrs[] = {\n')
        for i in range(len(weights)):
            f.write(f'    ann_biases_{i},\n')
        f.write('};\n\n')
        
        # Normalization
        f.write('const float ann_input_mean[] = {\n')
        for i, mean in enumerate(scaler.mean_):
            if i % 8 == 0:
                f.write('    ')
            f.write(f'{mean:.6f}f')
            if i < len(scaler.mean_) - 1:
                f.write(', ')
            if (i + 1) % 8 == 0:
                f.write('\n')
        if len(scaler.mean_) % 8 != 0:
            f.write('\n')
        f.write('};\n\n')
        
        f.write('const float ann_input_std[] = {\n')
        for i, std in enumerate(scaler.scale_):
            if i % 8 == 0:
                f.write('    ')
            f.write(f'{std:.6f}f')
            if i < len(scaler.scale_) - 1:
                f.write(', ')
            if (i + 1) % 8 == 0:
                f.write('\n')
        if len(scaler.scale_) % 8 != 0:
            f.write('\n')
        f.write('};\n')
    
    print("\nGenerated C headers in src/src_cube/ann/")

if __name__ == "__main__":
    main()