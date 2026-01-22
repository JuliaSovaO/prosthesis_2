# ml/quick_test.py
import joblib
import numpy as np

# Load retrained model
model_data = joblib.load('ml/emg_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

# Test with typical values from your new data
test_cases = {
    'rest': [75, 84, 141, 399],
    'rock': [170, 300, 330, 600],
    'good': [250, 135, 644, 493],
    # Add more based on your new data statistics
}

for gesture, values in test_cases.items():
    # Create feature vector (repeat values 5 times for 20 features)
    features = np.array([values * 5])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    
    print(f"{gesture:10s} -> Predicted: {prediction}")