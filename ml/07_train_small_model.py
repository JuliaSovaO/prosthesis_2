# 07_train_small_model.py
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def create_small_model():
    # Load your existing data
    # Adjust this to load your actual data
    print("Loading data...")
    # data = pd.read_csv('labeled_emg_data.csv')
    # X = data.drop('label', axis=1).values
    # y = data['label'].values
    
    # For testing, create dummy data
    X = np.random.randn(1000, 18)  # 1000 samples, 18 features
    y = np.random.randint(0, 10, 1000)  # 10 gestures
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create MUCH smaller model for STM32
    print("Training small model for STM32...")
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),  # Much smaller layers!
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Save model
    joblib.dump(model, 'emg_small_model.pkl')
    joblib.dump(scaler, 'feature_scaler_small.pkl')
    
    print("\nModel size summary:")
    print(f"  Input: {X.shape[1]} features")
    print(f"  Hidden layers: {model.hidden_layer_sizes}")
    print(f"  Output: {model.n_outputs_} classes")
    
    total_params = 0
    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        params = coef.size + intercept.size
        total_params += params
        print(f"  Layer {i+1}: {params} parameters")
    
    print(f"  Total parameters: {total_params}")
    print(f"  Flash memory required: ~{total_params * 4 / 1024:.1f}KB")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = create_small_model()