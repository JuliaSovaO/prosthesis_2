#!/usr/bin/env python3
"""
Final test of the STM32 classifier logic
"""

import pandas as pd
import numpy as np

def simulate_stm32_classifier(features):
    """Simulate the exact classifier logic running on STM32"""
    ch1_rms = features[0]
    ch2_rms = features[6]
    ch3_rms = features[12]
    
    ratio_ch2_ch1 = ch2_rms / ch1_rms if ch1_rms > 0.001 else 100.0
    
    # SCISSORS: Extremely low ch3
    if ch3_rms < 0.006:
        return "SCISSORS"
    
    # PAPER: Very low ch2
    if ch2_rms < 0.008:
        return "PAPER"
    
    # GOOD: Extreme ratio
    if ratio_ch2_ch1 > 10.0:
        return "GOOD"
    
    # REST: Highest ch2
    if ch2_rms > 0.035:
        return "REST"
    
    # ROCK: High ch2, moderate ch3
    if ch2_rms > 0.030 and ch3_rms > 0.025:
        return "ROCK"
    
    # THREE: High ch1
    if ch1_rms > 0.015:
        return "THREE"
    
    # FOUR: Moderate ch1, ch3 > ch1
    if ch1_rms > 0.012 and ch3_rms > ch1_rms:
        return "FOUR"
    
    # OKAY: Specific range
    if 0.009 < ch1_rms < 0.012 and 2.0 < ratio_ch2_ch1 < 3.0:
        return "OKAY"
    
    # FINGER_GUN
    if ch3_rms > 0.020 and ch2_rms > 0.020:
        return "FINGER_GUN"
    
    # FUCK/ONE
    return "FUCK"

def test_classifier_accuracy():
    df = pd.read_csv('data/all_data.csv')
    df.columns = ['ch1', 'ch2', 'ch3', 'gesture']
    
    # Normalize ADC values 0-1
    for ch in ['ch1', 'ch2', 'ch3']:
        df[ch] = df[ch] / 4095.0
    
    print("Testing STM32 Classifier Accuracy")
    print("="*70)
    
    gesture_mapping = {
        'finger-gun': 'FINGER_GUN',
        'fuck': 'FUCK',
        'good': 'GOOD',
        'okay': 'OKAY',
        'paper': 'PAPER',
        'rest': 'REST',
        'rock': 'ROCK',
        'scissors': 'SCISSORS',
        'three': 'THREE',
        'four': 'FOUR'
    }
    
    results = {}
    
    for gesture in df['gesture'].unique():
        subset = df[df['gesture'] == gesture]
        
        if len(subset) < 200:
            continue
            
        # Test 5 different windows
        predictions = []
        
        for i in range(5):
            start = i * 100
            if start + 200 <= len(subset):
                window = subset.iloc[start:start+200]
                
                # Calculate features like STM32 does
                features = [0] * 18
                for ch_idx, ch in enumerate(['ch1', 'ch2', 'ch3']):
                    signal = window[ch].values
                    dc = np.mean(signal)
                    rms = np.sqrt(np.mean((signal - dc)**2))
                    features[ch_idx * 6] = rms
                
                predicted = simulate_stm32_classifier(features)
                predictions.append(predicted)
        
        # Count accuracy
        expected = gesture_mapping.get(gesture, gesture.upper())
        correct = sum(1 for p in predictions if p == expected)
        accuracy = correct / len(predictions) if predictions else 0
        
        results[gesture] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'expected': expected
        }
        
        print(f"{gesture:12s}: {accuracy:.1%} correct ({correct}/{len(predictions)})")
        print(f"  Predictions: {predictions}")
    
    # Overall accuracy
    total_correct = sum(r['accuracy'] * len(r['predictions']) for r in results.values())
    total_tests = sum(len(r['predictions']) for r in results.values())
    overall_accuracy = total_correct / total_tests if total_tests > 0 else 0
    
    print(f"\nOverall Accuracy: {overall_accuracy:.1%}")
    
    return results

def print_feature_ranges():
    """Print expected feature ranges for debugging"""
    df = pd.read_csv('data/all_data.csv')
    df.columns = ['ch1', 'ch2', 'ch3', 'gesture']
    
    for ch in ['ch1', 'ch2', 'ch3']:
        df[ch] = df[ch] / 4095.0
    
    print("\nFeature Ranges for Each Gesture:")
    print("="*70)
    print(f"{'Gesture':12s} {'ch1 RMS':>8s} {'ch2 RMS':>8s} {'ch3 RMS':>8s} {'ch2/ch1':>8s}")
    print("-"*70)
    
    for gesture in sorted(df['gesture'].unique()):
        subset = df[df['gesture'] == gesture]
        
        if len(subset) > 200:
            window = subset.iloc[:200]
            
            rms_values = []
            for ch in ['ch1', 'ch2', 'ch3']:
                signal = window[ch].values
                dc = np.mean(signal)
                rms = np.sqrt(np.mean((signal - dc)**2))
                rms_values.append(rms)
            
            ratio = rms_values[1] / rms_values[0] if rms_values[0] > 0 else 0
            
            print(f"{gesture:12s} {rms_values[0]:8.4f} {rms_values[1]:8.4f} "
                  f"{rms_values[2]:8.4f} {ratio:8.1f}")

if __name__ == "__main__":
    test_classifier_accuracy()
    print_feature_ranges()