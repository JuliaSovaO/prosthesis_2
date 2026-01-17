# ml/01_preprocess_data.py
import pandas as pd
import numpy as np
from scipy import signal
import glob

def extract_td_features(window_data):
    """Extract 6 time-domain features from a window"""
    features = []
    
    # RMS
    rms = np.sqrt(np.mean(window_data**2))
    features.append(rms)
    
    # Variance
    var = np.var(window_data)
    features.append(var)
    
    # MAV
    mav = np.mean(np.abs(window_data))
    features.append(mav)
    
    # SSC
    ssc = np.sum((window_data[1:-1] > window_data[:-2]) & 
                 (window_data[1:-1] > window_data[2:]) |
                 (window_data[1:-1] < window_data[:-2]) & 
                 (window_data[1:-1] < window_data[2:]))
    features.append(ssc)
    
    # ZC with threshold
    threshold = 0.01 * np.max(np.abs(window_data))
    zc = np.sum(np.diff(np.sign(window_data - np.mean(window_data))) != 0)
    features.append(zc)
    
    # WL
    wl = np.sum(np.abs(np.diff(window_data)))
    features.append(wl)
    
    return np.array(features)

def preprocess_data():
    # Load all CSV files
    data_files = glob.glob('data/*.csv') + glob.glob('data/*.txt')
    
    all_features = []
    all_labels = []
    
    for file_path in data_files:
        # Read data
        df = pd.read_csv(file_path, header=None)
        
        # Assuming format: ch1, ch2, ch3, label
        if df.shape[1] >= 4:
            signals = df.iloc[:, :3].values
            labels = df.iloc[:, 3].values
            
            # Apply sliding window (250ms at 1500Hz = 375 samples)
            window_size = 375
            overlap = 0.9
            step = int(window_size * (1 - overlap))
            
            for start in range(0, len(signals) - window_size, step):
                window = signals[start:start + window_size]
                label = labels[start + window_size // 2]  # Middle label
                
                # Extract features for each channel
                channel_features = []
                for ch in range(3):
                    ch_features = extract_td_features(window[:, ch])
                    channel_features.extend(ch_features)
                
                all_features.append(channel_features)
                all_labels.append(label)
    
    # Save processed data
    np.savez('ml/emg_processed.npz', 
             features=np.array(all_features), 
             labels=np.array(all_labels))
    
    print(f"Processed {len(all_features)} samples")
    print(f"Feature shape: {len(all_features[0])}")
    
if __name__ == "__main__":
    preprocess_data()