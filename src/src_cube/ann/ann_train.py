import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json
import os

tf.keras.backend.set_floatx('float32')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WINDOW_SIZE = 50      # 50ms window at ~1000Hz
WINDOW_STEP = 25      # 50% overlap
FEATURES_PER_CHANNEL = 4  # RMS, MAV, VAR, WL
INPUT_SIZE = 16       # 4 channels * 4 features

GESTURE_ORDER = ['finger-gun', 'four', 'fuck', 'good', 'okay', 
                 'paper', 'rest', 'rock', 'scissors', 'three']

def load_balanced_data(csv_path):
    print(f"Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    X = df[['s1', 's2', 's3', 's4']].values.astype(np.float32)
    
    y = df['gesture'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(GESTURE_ORDER)
    y_encoded = label_encoder.transform(y)
    
    print(f"\nGesture classes:")
    for i, name in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {i}: {name} ({count} samples, {count/1000:.1f}s)")
    
    return X, y_encoded, label_encoder

def extract_features_sliding_window(X, y, window_size=50, step=25):
    n_samples, n_channels = X.shape
    features = []
    labels = []
    
    print(f"\nExtracting features with sliding window...")
    print(f"  Window size: {window_size} samples")
    print(f"  Step size: {step} samples")
    print(f"  Total samples: {n_samples}")
    
    total_windows = 0
    for start in range(0, n_samples - window_size, step):
        end = start + window_size
        window = X[start:end, :]
        
        window_labels = y[start:end]
        label = np.bincount(window_labels).argmax()
        
        # Extract 4 features per channel
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
            
            # Clip to prevent overflow
            rms = min(rms, 10000) if not np.isnan(rms) else 0
            mav = min(mav, 10000) if not np.isnan(mav) else 0
            var = min(var, 1e8) if not np.isnan(var) else 0
            wl = min(wl, 1e6) if not np.isnan(wl) else 0
            
            feat.extend([rms, mav, var, wl])
        
        features.append(feat)
        labels.append(label)
        total_windows += 1
        
        if total_windows % 5000 == 0:
            print(f"  Processed {total_windows} windows...")
    
    print(f"  Extracted {total_windows} windows")
    print(f"  Feature vector size: {len(features[0])}")
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

def create_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def generate_c_headers(model, label_encoder, scaler, output_dir='src/src_cube/ann_balanced'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get weights from Dense layers only
    weights = []
    biases = []
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            w, b = layer.get_weights()
            weights.append(w)
            biases.append(b)
    
    class_names = label_encoder.classes_.tolist()
    
    # Generate weights.h
    with open(os.path.join(output_dir, 'weights.h'), 'w') as f:
        f.write('#ifndef ANN_WEIGHTS_H\n#define ANN_WEIGHTS_H\n\n')
        f.write('#include <stdint.h>\n\n#ifdef __cplusplus\nextern "C" {\n#endif\n\n')
        
        f.write(f'#define ANN_INPUT_SIZE {scaler.mean_.shape[0]}\n')
        f.write(f'#define ANN_WINDOW_SIZE {WINDOW_SIZE}\n')
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
    
    # Generate weights.c
    with open(os.path.join(output_dir, 'weights.c'), 'w') as f:
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
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
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
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
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
        
        # Pointer arrays
        f.write('const float* ann_weights_ptrs[] = {\n')
        for i in range(len(weights)):
            f.write(f'    ann_weights_{i},\n')
        f.write('};\n\n')
        
        f.write('const float* ann_biases_ptrs[] = {\n')
        for i in range(len(weights)):
            f.write(f'    ann_biases_{i},\n')
        f.write('};\n\n')
        
        # Normalization parameters
        f.write('const float ann_input_mean[] = {\n')
        for i, mean in enumerate(scaler.mean_):
            if np.isnan(mean) or np.isinf(mean):
                mean = 0.0
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
            if np.isnan(std) or np.isinf(std) or std == 0:
                std = 1.0
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
    
    print(f"\nGenerated C headers in {output_dir}/")

def plot_training_history(history, save_path='training_history_balanced.png'):
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Training history saved to {save_path}")
    except Exception as e:
        print(f"Could not plot: {e}")

def main():
    print("="*60)
    print("EMG ANN Training - Balanced Data (10 Gestures)")
    print(f"Window: {WINDOW_SIZE} samples, Step: {WINDOW_STEP}")
    print("="*60)
    
    # Load data
    data_path = "data/08044/all_data_processed.csv"
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        print("Trying alternative path...")
        data_path = "data/07043/all_data.csv"
        if not os.path.exists(data_path):
            print("File not found!")
            return
    
    X_raw, y_encoded, label_encoder = load_balanced_data(data_path)
    print(f"\nLoaded {len(X_raw)} samples, {X_raw.shape[1]} channels")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Extract features with sliding window
    X_features, y_features = extract_features_sliding_window(
        X_raw, y_encoded, window_size=WINDOW_SIZE, step=WINDOW_STEP
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_features, test_size=0.2, random_state=42, stratify=y_features
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    input_dim = X_features.shape[1]
    num_classes = len(label_encoder.classes_)
    
    print(f"\nCreating ANN model:")
    print(f"  Input: {input_dim} features")
    print(f"  Architecture: 128 → 64 → 32 → {num_classes}")
    
    model = create_model(input_dim, num_classes)
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint('best_model_balanced.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(label_encoder.classes_):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], y_pred_classes[mask])
            print(f"  {name:12}: {class_acc:.4f} ({class_acc*100:.1f}%)")
    
    # Generate C headers
    print("\n" + "="*60)
    print("GENERATING C HEADERS")
    print("="*60)
    generate_c_headers(model, label_encoder, scaler, output_dir='src/src_cube/ann_balanced')
    
    # Save model and scaler
    model.save('emg_ann_model_balanced.keras')
    
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'classes': label_encoder.classes_.tolist(),
        'accuracy': float(accuracy),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'window_size': WINDOW_SIZE,
        'window_step': WINDOW_STEP
    }
    with open('scaler_params_balanced.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    # Plot training history
    plot_training_history(history, 'training_history_balanced.png')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - src/src_cube/ann_balanced/weights.h")
    print("  - src/src_cube/ann_balanced/weights.c")
    print("  - emg_ann_model_balanced.keras")
    print("  - scaler_params_balanced.json")
    print(f"\nFinal test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

if __name__ == "__main__":
    main()