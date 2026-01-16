#include "emg_classifier.h"
#include "emg_weights.h"
#include <math.h>
#include <string.h>

// ===== OPTIMIZED FEATURE EXTRACTION =====

static inline float fast_rms(uint16_t* signal, uint32_t n) {
    uint32_t sum_sq = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint16_t val = signal[i];
        sum_sq += val * val;
    }
    return sqrtf((float)sum_sq / n);
}

static inline float fast_mav(uint16_t* signal, uint32_t n) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += signal[i];
    }
    return (float)sum / n;
}

static inline float fast_wl(uint16_t* signal, uint32_t n) {
    uint32_t wl = 0;
    for (uint32_t i = 0; i < n - 1; i++) {
        int32_t diff = signal[i+1] - signal[i];
        wl += (diff > 0) ? diff : -diff;
    }
    return (float)wl;
}

void extract_features_fast(uint16_t* ch1, uint16_t* ch2, uint16_t* ch3, 
                          uint32_t n_samples, float* features_out) {
    // Extract only the most important features to save computation
    // Channel 1: RMS, MAV, WL
    features_out[0] = fast_rms(ch1, n_samples);
    features_out[1] = fast_mav(ch1, n_samples);
    features_out[2] = fast_wl(ch1, n_samples);
    
    // Channel 2: RMS, MAV, WL
    features_out[3] = fast_rms(ch2, n_samples);
    features_out[4] = fast_mav(ch2, n_samples);
    features_out[5] = fast_wl(ch2, n_samples);
    
    // Channel 3: RMS, MAV, WL
    features_out[6] = fast_rms(ch3, n_samples);
    features_out[7] = fast_mav(ch3, n_samples);
    features_out[8] = fast_wl(ch3, n_samples);
}

// ===== OPTIMIZED INFERENCE =====

static inline float fast_relu(float x) {
    return x > 0 ? x : 0;
}

int classify_gesture_fast(float* features) {
    // Normalize features
    float norm_features[9];  // Only 9 features now
    for (int i = 0; i < 9; i++) {
        norm_features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
    
    // Layer 1: Input -> Hidden1 (9×32)
    float hidden1[HIDDEN1_SIZE];
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        hidden1[i] = BIASES_0[i];
        for (int j = 0; j < 9; j++) {
            hidden1[i] += norm_features[j] * WEIGHTS_0[j][i];
        }
        hidden1[i] = fast_relu(hidden1[i]);
    }
    
    // Layer 2: Hidden1 -> Hidden2 (32×16)
    float hidden2[HIDDEN2_SIZE];
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        hidden2[i] = BIASES_1[i];
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            hidden2[i] += hidden1[j] * WEIGHTS_1[j][i];
        }
        hidden2[i] = fast_relu(hidden2[i]);
    }
    
    // Output layer: Hidden2 -> Output (16×10)
    float max_val = -1e9;
    int max_idx = 0;
    
    for (int i = 0; i < NUM_GESTURES; i++) {
        float val = BIASES_2[i];
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            val += hidden2[j] * WEIGHTS_2[j][i];
        }
        
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    // Simple threshold (no softmax to save computation)
    if (max_val < 0.0f) return 0;  // Return REST if confidence low
    
    return max_idx;
}