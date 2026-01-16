# 04_real_time_test.py
import joblib
import numpy as np
import time
from collections import deque

# Load model
model = joblib.load('emg_gesture_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Simulate real-time prediction
def real_time_simulation():
    # Load some test data
    data = np.load('emg_dataset.npz')
    emg = data['emg']
    fs = 332
    
    window_size = int(0.25 * fs)  # 83 samples
    step_size = int(0.025 * fs)   # 8 samples
    
    buffer = deque(maxlen=window_size)
    
    print("Real-time simulation started...")
    print("Press Ctrl+C to stop\n")
    
    for i in range(0, len(emg), step_size):
        if i + window_size > len(emg):
            break
            
        # Get window
        window = emg[i:i+window_size]
        
        # Extract features (simplified)
        features = []
        for ch in range(3):
            signal = window[:, ch]
            rms = np.sqrt(np.mean(signal**2))
            var = np.var(signal)
            mav = np.mean(np.abs(signal))
            diffs = np.diff(signal)
            ssc = np.sum(diffs[:-1] * diffs[1:] < 0)
            zc = np.sum(signal[:-1] * signal[1:] < 0)
            wl = np.sum(np.abs(diffs))
            features.extend([rms, var, mav, ssc, zc, wl])
        
        # Scale and predict
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][prediction]
        
        gesture_names = ['REST', 'ROCK', 'SCISSORS', 'PAPER', 'ONE',
                        'THREE', 'FOUR', 'GOOD', 'OKAY', 'FINGER_GUN']
        
        print(f"Time: {i/fs:5.1f}s | Gesture: {gesture_names[prediction]:12s} | Confidence: {probability:.2f}")
        
        time.sleep(0.025)  # Simulate real-time
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    real_time_simulation()