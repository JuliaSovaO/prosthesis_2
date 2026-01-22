#include "emg_classifier.h"
#include "emg_model.h"

// Initialize EMG buffer
void emg_buffer_init(EMG_Buffer* buffer) {
    buffer->write_index = 0;
    buffer->is_full = false;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            buffer->data[i][ch] = 0;
        }
    }
}

// Add a new sample to the buffer
void emg_buffer_add_sample(EMG_Buffer* buffer, int16_t ch1, int16_t ch2, int16_t ch3, int16_t ch4) {
    buffer->data[buffer->write_index][0] = ch1;
    buffer->data[buffer->write_index][1] = ch2;
    buffer->data[buffer->write_index][2] = ch3;
    buffer->data[buffer->write_index][3] = ch4;
    
    buffer->write_index++;
    if (buffer->write_index >= WINDOW_SIZE) {
        buffer->write_index = 0;
        buffer->is_full = true;
    }
}

// Extract features from window (optimized for STM32)
void extract_features_from_window(const int16_t window[WINDOW_SIZE][NUM_CHANNELS], float* features) {
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        float sum_abs = 0;
        float sum_sqr = 0;
        float sum = 0;
        float sum_diff = 0;
        int zero_crossings = 0;
        
        // Calculate basic statistics
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float val = window[i][ch];
            sum_abs += fabsf(val);
            sum_sqr += val * val;
            sum += val;
            
            if (i > 0) {
                float diff = val - window[i-1][ch];
                sum_diff += fabsf(diff);
                
                // Zero crossing detection
                if ((val >= 0 && window[i-1][ch] < 0) || (val < 0 && window[i-1][ch] >= 0)) {
                    zero_crossings++;
                }
            }
        }
        
        // IMPORTANT: Feature order MUST match Python training
        // Python order: [mav, rms, var, wl, zc] for each channel
        int base_idx = ch * FEATURES_PER_CHANNEL;
        
        // Feature 1: Mean Absolute Value (MAV)
        features[base_idx + 0] = sum_abs / WINDOW_SIZE;
        
        // Feature 2: Root Mean Square (RMS)
        features[base_idx + 1] = sqrtf(sum_sqr / WINDOW_SIZE);
        
        // Feature 3: Variance
        float mean = sum / WINDOW_SIZE;
        float variance = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float diff = window[i][ch] - mean;
            variance += diff * diff;
        }
        features[base_idx + 2] = variance / WINDOW_SIZE;
        
        // Feature 4: Waveform Length (WL)
        features[base_idx + 3] = sum_diff;
        
        // Feature 5: Zero Crossing (ZC)
        features[base_idx + 4] = zero_crossings;
    }
}

// Classify gesture using logistic regression model
GestureType classify_gesture(const float* features) {
    // Verify feature count
    if (TOTAL_FEATURES != NUM_FEATURES) {
        // Debug error
        return GESTURE_REST;
    }
    return predict_gesture(features);
}

// Process a window and extract features if available
bool emg_buffer_process_window(EMG_Buffer* buffer, float* features) {
    if (!buffer->is_full) {
        return false;
    }
    
    // Create a window starting from the oldest data
    int16_t window[WINDOW_SIZE][NUM_CHANNELS];
    uint16_t start_idx = buffer->write_index;  // Oldest data
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
        uint16_t idx = (start_idx + i) % WINDOW_SIZE;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            window[i][ch] = buffer->data[idx][ch];
        }
    }
    
    // Extract features
    extract_features_from_window(window, features);
    
    return true;
}