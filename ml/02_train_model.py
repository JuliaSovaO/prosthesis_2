# ml/02_train_regression.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_models():
    # Load processed data
    data = np.load('ml/emg_processed.npz')
    X = data['features']
    y = data['labels'].astype(int)
    
    # Remove any NaN or Inf values
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different models
    models = {
        'logistic': LogisticRegression(
            max_iter=1000, 
            multi_class='ovr',
            solver='liblinear',
            C=0.1
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        ),
        'svm': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True
        )
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
        
        print(f"{name}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")
    
    # Select best model based on test accuracy
    best_model_name = max(results, key=lambda x: results[x]['test_acc'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Test accuracy: {results[best_model_name]['test_acc']:.3f}")
    
    # Save model and scaler
    joblib.dump(best_model, 'ml/emg_best_model.pkl')
    joblib.dump(scaler, 'ml/feature_scaler.pkl')
    
    # Extract model parameters for C implementation
    extract_model_params(best_model, scaler, best_model_name)
    
    return best_model, scaler, best_model_name

def extract_model_params(model, scaler, model_name):
    """Extract model parameters for C implementation"""
    
    if model_name == 'logistic':
        # For logistic regression
        coef = model.coef_  # Shape: (n_classes, n_features)
        intercept = model.intercept_
        
        # Save as C header file
        with open('../src/src_cube/emg_model_params.h', 'w') as f:
            f.write('#ifndef EMG_MODEL_PARAMS_H\n')
            f.write('#define EMG_MODEL_PARAMS_H\n\n')
            
            f.write(f'#define N_CLASSES {coef.shape[0]}\n')
            f.write(f'#define N_FEATURES {coef.shape[1]}\n\n')
            
            # Coefficients
            f.write('static const float COEFF[N_CLASSES][N_FEATURES] = {\n')
            for i in range(coef.shape[0]):
                f.write('    {')
                f.write(', '.join([f'{val:.6f}f' for val in coef[i]]))
                f.write('},\n')
            f.write('};\n\n')
            
            # Intercepts
            f.write('static const float INTERCEPT[N_CLASSES] = {\n')
            f.write('    ' + ', '.join([f'{val:.6f}f' for val in intercept]))
            f.write('\n};\n\n')
            
            # Scaler mean and std
            mean = scaler.mean_
            std = np.sqrt(scaler.var_)
            
            f.write('static const float SCALER_MEAN[N_FEATURES] = {\n')
            f.write('    ' + ', '.join([f'{val:.6f}f' for val in mean]))
            f.write('\n};\n\n')
            
            f.write('static const float SCALER_STD[N_FEATURES] = {\n')
            f.write('    ' + ', '.join([f'{val:.6f}f' for val in std]))
            f.write('\n};\n\n')
            
            f.write('#endif // EMG_MODEL_PARAMS_H\n')
            
    elif model_name == 'random_forest':
        # Simplified implementation - save tree structure
        # Note: This is complex; consider using a simpler model
        pass

if __name__ == "__main__":
    train_models()