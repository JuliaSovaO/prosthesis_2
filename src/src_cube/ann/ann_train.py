import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import json
import os
import warnings
warnings.filterwarnings('ignore')

tf.keras.backend.set_floatx('float32')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WINDOW_SIZE = 250
WINDOW_STEP = 50
FEATURES_PER_CHANNEL = 8
NUM_CHANNELS = 4
INPUT_SIZE = NUM_CHANNELS * FEATURES_PER_CHANNEL  # 32

GESTURE_ORDER = ['finger-gun', 'four', 'fuck', 'good', 'okay', 
                 'paper', 'rest', 'rock', 'scissors', 'three']

def extract_features(window):
    """Extract 8 features per channel: RMS, MAV, WL, ZC, SSC, IEMG, MeanFreq, MedFreq"""
    features = []
    
    for ch in range(window.shape[1]):
        ch_data = window[:, ch].astype(np.float64)
        
        # Time domain features
        rms = np.sqrt(np.mean(ch_data ** 2))
        mav = np.mean(np.abs(ch_data))
        wl = np.sum(np.abs(np.diff(ch_data)))
        
        # Zero crossings with threshold
        threshold = 0.01 * (np.max(ch_data) - np.min(ch_data) + 1e-6)
        zc = 0
        for i in range(1, len(ch_data)):
            if abs(ch_data[i] - ch_data[i-1]) >= threshold:
                if np.sign(ch_data[i]) != np.sign(ch_data[i-1]):
                    zc += 1
        
        # Slope sign changes
        ssc = 0
        for i in range(1, len(ch_data)-1):
            d1 = ch_data[i] - ch_data[i-1]
            d2 = ch_data[i+1] - ch_data[i]
            if d1 * d2 < 0:
                if abs(d1) >= threshold or abs(d2) >= threshold:
                    ssc += 1
        
        # Integrated EMG
        iemg = np.sum(np.abs(ch_data))
        
        # Frequency features
        n = len(ch_data)
        freqs = np.fft.rfftfreq(n, d=1.0/1000.0)
        spectrum = np.abs(np.fft.rfft(ch_data))
        total_power = np.sum(spectrum) + 1e-10
        mean_freq = np.sum(freqs * spectrum) / total_power
        
        cumulative = np.cumsum(spectrum)
        half_power_idx = np.searchsorted(cumulative, cumulative[-1]/2)
        median_freq = freqs[min(half_power_idx, len(freqs)-1)]
        
        features.extend([
            np.clip(rms, 0, 10000),
            np.clip(mav, 0, 10000),
            np.clip(wl, 0, 500000),
            float(zc),
            float(ssc),
            np.clip(iemg, 0, 5000000),
            np.clip(mean_freq, 0, 500),
            np.clip(median_freq, 0, 500)
        ])
    
    return np.array(features, dtype=np.float32)

def create_model(input_dim, num_classes):
    """Simple but deep MLP - compatible with simple C inference"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32, kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def fuse_batchnorm(model):
    """Fuse BatchNormalization layers into preceding Dense layers for deployment"""
    dense_layers = []
    current_dense = None
    
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            if current_dense is not None:
                w, b = current_dense.get_weights()
                dense_layers.append((w, b))
            current_dense = layer
        elif isinstance(layer, layers.BatchNormalization) and current_dense is not None:
            w, b = current_dense.get_weights()
            gamma, beta, mean, var = layer.get_weights()
            epsilon = layer.epsilon
            
            scale = gamma / np.sqrt(var + epsilon)
            w_fused = w * scale
            b_fused = (b - mean) * scale + beta
            
            dense_layers.append((w_fused, b_fused))
            current_dense = None
    
    if current_dense is not None:
        w, b = current_dense.get_weights()
        dense_layers.append((w, b))
    
    return dense_layers

def write_flat_array(f, name, arr):
    flat = arr.flatten() if hasattr(arr, 'flatten') else np.array(arr)
    f.write(f'const float {name}[] = {{\n')
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

def generate_c_headers(dense_layers, class_names, scaler, output_dir='src/src_cube/ann'):
    os.makedirs(output_dir, exist_ok=True)
    
    num_layers = len(dense_layers)
    
    # Layer sizes: [input_dim, hidden1, hidden2, ..., output]
    layer_sizes = [INPUT_SIZE]
    for w, _ in dense_layers:
        layer_sizes.append(w.shape[1])
    
    print(f"\nGenerating C headers with {num_layers} layers")
    print(f"Layer sizes: {layer_sizes}")
    
    # weights.h
    with open(os.path.join(output_dir, 'weights.h'), 'w') as f:
        f.write('#ifndef ANN_WEIGHTS_H\n#define ANN_WEIGHTS_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write('#ifdef __cplusplus\nextern "C" {\n#endif\n\n')
        
        f.write(f'#define ANN_INPUT_SIZE {INPUT_SIZE}\n')
        f.write(f'#define ANN_WINDOW_SIZE {WINDOW_SIZE}\n')
        f.write(f'#define ANN_NUM_CLASSES {len(class_names)}\n')
        f.write(f'#define ANN_NUM_LAYERS {num_layers}\n')
        f.write(f'#define ANN_FEATURES_PER_CHANNEL {FEATURES_PER_CHANNEL}\n\n')
        
        f.write('extern const char* ann_class_names[];\n')
        f.write('extern const int ann_layer_sizes[];\n\n')
        
        for i in range(num_layers):
            f.write(f'extern const float ann_weights_{i}[];\n')
            f.write(f'extern const float ann_biases_{i}[];\n')
        
        f.write('\nextern const float ann_input_mean[];\n')
        f.write('extern const float ann_input_std[];\n\n')
        
        f.write('#ifdef __cplusplus\n}\n#endif\n\n#endif\n')
    
    # weights.c
    with open(os.path.join(output_dir, 'weights.c'), 'w') as f:
        f.write('#include "weights.h"\n\n')
        
        f.write('const char* ann_class_names[] = {\n')
        for name in class_names:
            f.write(f'    "{name}",\n')
        f.write('};\n\n')
        
        f.write('const int ann_layer_sizes[] = {\n')
        for sz in layer_sizes:
            f.write(f'    {sz},\n')
        f.write('};\n\n')
        
        for i, (w, b) in enumerate(dense_layers):
            write_flat_array(f, f'ann_weights_{i}', w)
            write_flat_array(f, f'ann_biases_{i}', b)
        
        write_flat_array(f, 'ann_input_mean', scaler.mean_)
        write_flat_array(f, 'ann_input_std', scaler.scale_)
    
    print(f"Generated: {output_dir}/weights.h, {output_dir}/weights.c")

def main():
    print("="*60)
    print("EMG ANN Training - Compatible Architecture")
    print(f"Features: {FEATURES_PER_CHANNEL}/ch = {INPUT_SIZE} total")
    print(f"Window: {WINDOW_SIZE}, Step: {WINDOW_STEP}")
    print("="*60)
    
    # Load data
    data_paths = [
        # "data/02051/all_data_processed.csv",
        # "data/02051/all_data.csv",
        # "data/02052/all_data_processed.csv",
        # "data/02052/all_data.csv",
        "data/05051/all_data_processed.csv",
        "data/05051/all_data.csv",
        # "data/02053/all_data_processed.csv",
        # "data/02053/all_data.csv",
        # "data/02054/all_data_processed.csv",
        # "data/02054/all_data.csv",
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("ERROR: No data file found!")
        return
    
    print(f"\nLoading: {data_path}")
    df = pd.read_csv(data_path)
    X_raw = df[['s1', 's2', 's3', 's4']].values.astype(np.float32)
    y_str = df['gesture'].values
    
    le = LabelEncoder()
    le.fit(GESTURE_ORDER)
    y = le.transform(y_str)
    
    print(f"Samples: {len(X_raw)}")
    for i, name in enumerate(le.classes_):
        count = np.sum(y == i)
        print(f"  {name}: {count} ({count/1000:.1f}s)")
    
    # Extract features
    print(f"\nExtracting features (window={WINDOW_SIZE}, step={WINDOW_STEP})...")
    features_list = []
    labels_list = []
    
    for start in range(0, len(X_raw) - WINDOW_SIZE, WINDOW_STEP):
        window = X_raw[start:start + WINDOW_SIZE]
        feat = extract_features(window)
        label = np.bincount(y[start:start + WINDOW_SIZE]).argmax()
        features_list.append(feat)
        labels_list.append(label)
        
        if len(features_list) % 5000 == 0:
            print(f"  {len(features_list)} windows...")
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    print(f"Total: {len(X)} windows")
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=10000, neginf=-10000)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Normalize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Create model
    print(f"\nModel: 256→128→64→32→10 (BN fused for deployment)")
    model = create_model(INPUT_SIZE, len(le.classes_))
    model.summary()
    
    # Use a FLOAT learning rate (fixes the ReduceLROnPlateau crash)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', 
                                  save_best_only=True, verbose=1),
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train_s, y_train,
        epochs=300,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=cb,
        verbose=2
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_test_s, verbose=0).argmax(axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("Per-class accuracy:")
    for i, name in enumerate(le.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            ca = (y_pred[mask] == i).mean()
            print(f"  {name:12}: {ca:.4f} ({ca*100:.0f}%)")
    
    # Fuse BN and generate C headers
    dense_layers = fuse_batchnorm(model)
    generate_c_headers(dense_layers, le.classes_.tolist(), scaler)
    
    # Save model
    model.save('emg_ann_model.keras')
    
    params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'classes': le.classes_.tolist(),
        'accuracy': float(acc),
        'input_dim': INPUT_SIZE,
        'num_classes': len(le.classes_),
        'window_size': WINDOW_SIZE,
        'window_step': WINDOW_STEP,
        'features_per_channel': FEATURES_PER_CHANNEL
    }
    with open('scaler_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    print("\nFiles generated:")
    print("  - src/src_cube/ann/weights.h")
    print("  - src/src_cube/ann/weights.c")
    print("  - emg_ann_model.keras")
    print("  - scaler_params.json")

if __name__ == "__main__":
    main()