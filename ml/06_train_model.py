# 02_train_model.py (updated)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import glob

def load_collected_data():
    """Load data from multiple CSV files"""
    all_data = []
    all_labels = []
    
    # Load all gesture files
    for gesture_id in range(10):
        files = glob.glob(f'gesture_{gesture_id}_*.csv')
        for file in files:
            df = pd.read_csv(file)
            if len(df) > 0:
                all_data.append(df[['ch1', 'ch2', 'ch3']].values)
                all_labels.extend([gesture_id] * len(df))
    
    return np.vstack(all_data), np.array(all_labels)

def extract_features_window(data, window_size=83, step=8):
    """Extract features using sliding window"""
    features_list = []
    labels_list = []
    
    for start in range(0, len(data) - window_size, step):
        window = data[start:start + window_size]
        
        # Extract 6 features per channel (same as paper)
        ch_features = []
        for ch in range(3):
            channel_data = window[:, ch]
            
            # Calculate features
            rms = np.sqrt(np.mean(channel_data**2))
            var = np.var(channel_data)
            mav = np.mean(np.abs(channel_data))
            
            # SSC
            ssc = np.sum([1 for i in range(1, len(channel_data)-1) 
                         if (channel_data[i] - channel_data[i-1]) * 
                            (channel_data[i] - channel_data[i+1]) > 0])
            
            # ZC
            zc = np.sum([1 for i in range(len(channel_data)-1)
                        if channel_data[i] * channel_data[i+1] < 0 and 
                        abs(channel_data[i] - channel_data[i+1]) > 10])
            
            # WL
            wl = np.sum(np.abs(np.diff(channel_data)))
            
            ch_features.extend([rms, var, mav, ssc, zc, wl])
        
        features_list.append(ch_features)
        
        # Assign label (majority vote in window)
        # Note: You need to adjust this based on your data structure
        labels_list.append(0)  # Placeholder
    
    return np.array(features_list), np.array(labels_list)

def main():
    print("Loading collected data...")
    raw_data, labels = load_collected_data()
    
    print(f"Loaded {len(raw_data)} samples")
    
    print("Extracting features...")
    features, window_labels = extract_features_window(raw_data)
    
    print(f"Extracted {len(features)} feature vectors")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, window_labels, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ANN (same architecture as paper)
    print("Training ANN...")
    model = MLPClassifier(
        hidden_layer_sizes=(300, 300, 300),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=2000,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(model, 'emg_gesture_ann.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("\nModel saved to emg_gesture_ann.pkl")
    print("Scaler saved to feature_scaler.pkl")

if __name__ == "__main__":
    main()