#include "ann_inference.h"
#include "weights.h"
#include <string.h>

// Forward pass through the ANN
void ann_forward(const float* input, float* output) {
    // First layer: input -> hidden1 (64 neurons)
    float hidden1[64];
    int input_dim = ann_layer_sizes[0];
    int hidden1_dim = ann_layer_sizes[1];
    
    for (int i = 0; i < hidden1_dim; i++) {
        float sum = ann_biases_ptrs[0][i];
        for (int j = 0; j < input_dim; j++) {
            sum += input[j] * ann_weights_ptrs[0][j * hidden1_dim + i];
        }
        hidden1[i] = tanhf(sum);
    }
    
    // Second layer: hidden1 -> hidden2 (32 neurons)
    float hidden2[32];
    int hidden1_out = ann_layer_sizes[1];
    int hidden2_dim = ann_layer_sizes[2];
    
    for (int i = 0; i < hidden2_dim; i++) {
        float sum = ann_biases_ptrs[1][i];
        for (int j = 0; j < hidden1_out; j++) {
            sum += hidden1[j] * ann_weights_ptrs[1][j * hidden2_dim + i];
        }
        hidden2[i] = tanhf(sum);
    }
    
    // Third layer: hidden2 -> hidden3 (16 neurons)
    float hidden3[16];
    int hidden2_out = ann_layer_sizes[2];
    int hidden3_dim = ann_layer_sizes[3];
    
    for (int i = 0; i < hidden3_dim; i++) {
        float sum = ann_biases_ptrs[2][i];
        for (int j = 0; j < hidden2_out; j++) {
            sum += hidden2[j] * ann_weights_ptrs[2][j * hidden3_dim + i];
        }
        hidden3[i] = tanhf(sum);
    }
    
    // Output layer: hidden3 -> output (num_classes)
    int hidden3_out = ann_layer_sizes[3];
    int num_classes = ann_layer_sizes[4];
    
    for (int i = 0; i < num_classes; i++) {
        output[i] = ann_biases_ptrs[3][i];
        for (int j = 0; j < hidden3_out; j++) {
            output[i] += hidden3[j] * ann_weights_ptrs[3][j * num_classes + i];
        }
    }
    
    // softmax activation
    float max_val = output[0];
    for (int i = 1; i < num_classes; i++) {
        if (output[i] > max_val) max_val = output[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < num_classes; i++) {
        output[i] /= sum;
    }
}

// Normalize features using stored mean and std
void ann_normalize_features(float* features) {
    for (int i = 0; i < ANN_INPUT_SIZE; i++) {
        if (ann_input_std[i] > 1e-6f) {
            features[i] = (features[i] - ann_input_mean[i]) / ann_input_std[i];
        } else {
            features[i] = 0.0f;
        }
    }
}

// Feature extraction from raw EMG data
void ann_extract_features(
    const uint16_t raw_data[][4],
    uint16_t window_size,
    float* features
) {
    int feat_idx = 0;
    
    for (int ch = 0; ch < 4; ch++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        float sum_abs = 0.0f;
        float sum_wl = 0.0f;
        int zero_cross = 0;
        int slope_change = 0;
        
        // Convert to float and compute basic statistics
        float samples[ANN_WINDOW_SIZE];
        for (int i = 0; i < window_size; i++) {
            samples[i] = (float)raw_data[i][ch];
            sum += samples[i];
            sum_sq += samples[i] * samples[i];
            sum_abs += fabsf(samples[i]);
        }
        
        float mean = sum / window_size;
        
        // RMS (Root Mean Square)
        features[feat_idx++] = sqrtf(sum_sq / window_size);
        
        // MAV (Mean Absolute Value)
        features[feat_idx++] = sum_abs / window_size;
        
        // VAR (Variance)
        float var = 0.0f;
        for (int i = 0; i < window_size; i++) {
            float diff = samples[i] - mean;
            var += diff * diff;
        }
        features[feat_idx++] = var / window_size;
        
        // WL (Waveform Length)
        for (int i = 1; i < window_size; i++) {
            sum_wl += fabsf(samples[i] - samples[i-1]);
        }
        features[feat_idx++] = sum_wl;
        
        // ZC (Zero Crossing)
        for (int i = 1; i < window_size; i++) {
            if ((samples[i-1] * samples[i]) < 0) {
                zero_cross++;
            }
        }
        features[feat_idx++] = (float)zero_cross;
        
        // SSC (Slope Sign Change)
        for (int i = 2; i < window_size; i++) {
            float d1 = samples[i-1] - samples[i-2];
            float d2 = samples[i] - samples[i-1];
            if ((d1 * d2) > 0) {
                slope_change++;
            }
        }
        features[feat_idx++] = (float)slope_change;
    }
}

// Get predicted class
uint8_t ann_predict(const float* input) {
    float output[ANN_NUM_CLASSES];
    ann_forward(input, output);
    
    uint8_t best_class = 0;
    float best_prob = output[0];
    for (int i = 1; i < ANN_NUM_CLASSES; i++) {
        if (output[i] > best_prob) {
            best_prob = output[i];
            best_class = i;
        }
    }
    return best_class;
}

// Process a full window of EMG data
uint8_t ann_process_window(const uint16_t emg_buffer[][4], uint16_t buffer_size) {
    if (buffer_size < ANN_WINDOW_SIZE) {
        return 0;  // Not enough data
    }
    
    float features[ANN_INPUT_SIZE];
    ann_extract_features(emg_buffer, ANN_WINDOW_SIZE, features);
    ann_normalize_features(features);
    
    return ann_predict(features);
}

void ann_init(void) {
    // *all initialization is done via static data in weights.h, this function exists for API consistency
}

const char* ann_get_class_name(uint8_t class_idx) {
    if (class_idx < ANN_NUM_CLASSES) {
        return ann_class_names[class_idx];
    }
    return "UNKNOWN";
}