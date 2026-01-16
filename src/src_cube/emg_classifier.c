// emg_classifier.c - COMPLETE FIXED VERSION
#include "emg_classifier.h"
#include "emg_model_weights.h" 
#include <math.h>
#include <string.h>

// Gesture names
const char* GESTURE_NAMES[NUM_GESTURES] = {
    "REST", "ROCK", "SCISSORS", "PAPER", "ONE",
    "THREE", "FOUR", "GOOD", "OKAY", "FINGER_GUN"
};

// ===== FEATURE EXTRACTION =====

static float calculate_rms(uint16_t* signal, uint32_t n) {
    float sum_sq = 0;
    for (uint32_t i = 0; i < n; i++) {
        float val = (float)signal[i];
        sum_sq += val * val;
    }
    return sqrtf(sum_sq / n);
}

static float calculate_variance(uint16_t* signal, uint32_t n) {
    float mean = 0;
    for (uint32_t i = 0; i < n; i++) {
        mean += signal[i];
    }
    mean /= n;
    
    float var = 0;
    for (uint32_t i = 0; i < n; i++) {
        float diff = signal[i] - mean;
        var += diff * diff;
    }
    return var / n;
}

static float calculate_mav(uint16_t* signal, uint32_t n) {
    float sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += fabsf((float)signal[i]);
    }
    return sum / n;
}

static float calculate_ssc(uint16_t* signal, uint32_t n) {
    uint32_t ssc = 0;
    for (uint32_t i = 1; i < n - 1; i++) {
        if ((signal[i] - signal[i-1]) * (signal[i] - signal[i+1]) > 0) {
            ssc++;
        }
    }
    return (float)ssc;
}

static float calculate_zc(uint16_t* signal, uint32_t n) {
    uint32_t zc = 0;
    float threshold = 10.0f;
    
    for (uint32_t i = 0; i < n - 1; i++) {
        if ((signal[i] > 0 && signal[i+1] < 0) || (signal[i] < 0 && signal[i+1] > 0)) {
            if (fabsf((float)(signal[i] - signal[i+1])) > threshold) {
                zc++;
            }
        }
    }
    return (float)zc;
}

static float calculate_wl(uint16_t* signal, uint32_t n) {
    float wl = 0;
    for (uint32_t i = 0; i < n - 1; i++) {
        wl += fabsf((float)(signal[i+1] - signal[i]));
    }
    return wl;
}

void extract_all_features(uint16_t* ch1, uint16_t* ch2, uint16_t* ch3, 
                         uint32_t n_samples, float* features_out) {
    // Channel 1 features (0-5)
    features_out[0] = calculate_rms(ch1, n_samples);
    features_out[1] = calculate_variance(ch1, n_samples);
    features_out[2] = calculate_mav(ch1, n_samples);
    features_out[3] = calculate_ssc(ch1, n_samples);
    features_out[4] = calculate_zc(ch1, n_samples);
    features_out[5] = calculate_wl(ch1, n_samples);
    
    // Channel 2 features (6-11)
    features_out[6] = calculate_rms(ch2, n_samples);
    features_out[7] = calculate_variance(ch2, n_samples);
    features_out[8] = calculate_mav(ch2, n_samples);
    features_out[9] = calculate_ssc(ch2, n_samples);
    features_out[10] = calculate_zc(ch2, n_samples);
    features_out[11] = calculate_wl(ch2, n_samples);
    
    // Channel 3 features (12-17)
    features_out[12] = calculate_rms(ch3, n_samples);
    features_out[13] = calculate_variance(ch3, n_samples);
    features_out[14] = calculate_mav(ch3, n_samples);
    features_out[15] = calculate_ssc(ch3, n_samples);
    features_out[16] = calculate_zc(ch3, n_samples);
    features_out[17] = calculate_wl(ch3, n_samples);
}

// ===== NORMALIZATION =====

void normalize_features(float* features) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}

// ===== ACTIVATION FUNCTIONS =====

static inline float relu(float x) {
    return x > 0 ? x : 0;
}

static void softmax(float* x, int size) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exponentials
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// ===== ANN INFERENCE =====

int classify_gesture(float* features) {
    // Normalize features
    normalize_features(features);
    
    // Layer 1: Input -> Hidden (18×300)
    float hidden1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden1[i] = BIASES_0[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            hidden1[i] += features[j] * WEIGHTS_0[j][i];
        }
        hidden1[i] = relu(hidden1[i]);
    }
    
    // Layer 2: Hidden -> Hidden (300×300)
    float hidden2[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden2[i] = BIASES_1[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden2[i] += hidden1[j] * WEIGHTS_1[j][i];
        }
        hidden2[i] = relu(hidden2[i]);
    }
    
    // Layer 3: Hidden -> Hidden (300×300)
    float hidden3[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden3[i] = BIASES_2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden3[i] += hidden2[j] * WEIGHTS_2[j][i];
        }
        hidden3[i] = relu(hidden3[i]);
    }
    
    // Output layer: Hidden -> Output (300×10)
    float output[NUM_GESTURES];
    for (int i = 0; i < NUM_GESTURES; i++) {
        output[i] = BIASES_3[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hidden3[j] * WEIGHTS_3[j][i];
        }
    }
    
    // Apply softmax
    softmax(output, NUM_GESTURES);
    
    // Find gesture with highest probability
    int max_idx = 0;
    for (int i = 1; i < NUM_GESTURES; i++) {
        if (output[i] > output[max_idx]) {
            max_idx = i;
        }
    }
    
    // Optional: Add confidence threshold
    if (output[max_idx] < 0.5) return 0; // Reject if low confidence
    
    return max_idx;
}