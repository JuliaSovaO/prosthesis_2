#include "ann_inference.h"
#include "weights.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

void ann_forward(const float* input, float* output) {    
    // Layer 1: input (16) -> hidden1 (128)
    float hidden1[128];
    for (int i = 0; i < 128; i++) {
        float sum = ann_biases_0[i];
        for (int j = 0; j < 16; j++) {
            sum += input[j] * ann_weights_0[j * 128 + i];
        }
        hidden1[i] = sum > 0 ? sum : 0;
    }
    
    // Layer 2: hidden1 (128) -> hidden2 (64)
    float hidden2[64];
    for (int i = 0; i < 64; i++) {
        float sum = ann_biases_1[i];
        for (int j = 0; j < 128; j++) {
            sum += hidden1[j] * ann_weights_1[j * 64 + i];
        }
        hidden2[i] = sum > 0 ? sum : 0;
    }
    
    // Layer 3: hidden2 (64) -> hidden3 (32)
    float hidden3[32];
    for (int i = 0; i < 32; i++) {
        float sum = ann_biases_2[i];
        for (int j = 0; j < 64; j++) {
            sum += hidden2[j] * ann_weights_2[j * 32 + i];
        }
        hidden3[i] = sum > 0 ? sum : 0;
    }
    
    // Output layer: hidden3 (32) -> output (10)
    for (int i = 0; i < 10; i++) {
        output[i] = ann_biases_3[i];
        for (int j = 0; j < 32; j++) {
            output[i] += hidden3[j] * ann_weights_3[j * 10 + i];
        }
    }
    
    // Softmax
    float max_val = output[0];
    for (int i = 1; i < 10; i++) {
        if (output[i] > max_val) max_val = output[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < 10; i++) {
        output[i] /= sum;
    }
}

void ann_normalize_features(float* features) {
    for (int i = 0; i < ANN_INPUT_SIZE; i++) {
        // Use the stored mean/std from training if available
        if (ann_input_std[i] > 1e-6f) {
            features[i] = (features[i] - ann_input_mean[i]) / ann_input_std[i];
        } else {
            features[i] = 0.0f;
        }
        
        // Debug
        static int print_count = 0;
        if (print_count < 5 && i < 4) {
            printf("NORM[%d]: raw=%.0f mean=%.0f std=%.0f -> norm=%.3f\n", 
                   i, features[i] * ann_input_std[i] + ann_input_mean[i],
                   ann_input_mean[i], ann_input_std[i], features[i]);
        }
        if (i == 3) print_count++;
        
        // Clip to reasonable range
        if (features[i] > 5.0f) features[i] = 5.0f;
        if (features[i] < -5.0f) features[i] = -5.0f;
    }
}

void ann_extract_features(const uint16_t raw_data[][4], uint16_t window_size, float* features) {
    int feat_idx = 0;
    
    for (int ch = 0; ch < 4; ch++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            float val = (float)raw_data[i][ch];
            sum += val;
            sum_sq += val * val;
        }
        
        // RMS
        features[feat_idx++] = sqrtf(sum_sq / window_size);
        // MAV  
        features[feat_idx++] = sum / window_size;
        // VAR
        float mean = sum / window_size;
        float var = (sum_sq / window_size) - (mean * mean);
        features[feat_idx++] = var;
        // WL
        float wl = 0.0f;
        for (int i = 1; i < window_size; i++) {
            wl += fabsf((float)raw_data[i][ch] - (float)raw_data[i-1][ch]);
        }
        features[feat_idx++] = wl;
    }
}

uint8_t ann_predict(const float* input) {
    float output[10];
    ann_forward(input, output);
    
    uint8_t best_class = 0;
    float best_prob = output[0];
    for (int i = 1; i < 10; i++) {
        if (output[i] > best_prob) {
            best_prob = output[i];
            best_class = i;
        }
    }
    return best_class;
}

uint8_t ann_process_window(const uint16_t emg_buffer[][4], uint16_t buffer_size) {
    if (buffer_size < ANN_WINDOW_SIZE) {
        return 0;
    }
    
    float features[ANN_INPUT_SIZE];
    ann_extract_features(emg_buffer, ANN_WINDOW_SIZE, features);
    ann_normalize_features(features);
    
    return ann_predict(features);
}

float ann_get_confidence_from_buffer(const uint16_t emg_buffer[][4], uint16_t buffer_size) {
    if (buffer_size < ANN_WINDOW_SIZE) {
        return 0.0f;
    }
    
    float features[ANN_INPUT_SIZE];
    ann_extract_features(emg_buffer, ANN_WINDOW_SIZE, features);
    ann_normalize_features(features);
    
    float output[10];
    ann_forward(features, output);
    
    uint8_t pred = ann_predict(features);
    return output[pred];
}

void ann_init(void) {
    printf("ANN INITIALIZED - Clean Model (10 classes)\n");
    printf("Mean values from training:\n");
    printf("  CH0 mean: %.0f, std: %.0f\n", ann_input_mean[0], ann_input_std[0]);
    printf("  CH1 mean: %.0f, std: %.0f\n", ann_input_mean[4], ann_input_std[4]);
    printf("  CH2 mean: %.0f, std: %.0f\n", ann_input_mean[8], ann_input_std[8]);
    printf("  CH3 mean: %.0f, std: %.0f\n", ann_input_mean[12], ann_input_std[12]);
}

const char* ann_get_class_name(uint8_t class_idx) {
    const char* names[] = {"rock", "scissors", "paper", "fuck", "three", 
                           "four", "good", "okay", "finger-gun", "rest"};
    if (class_idx < 10) {
        return names[class_idx];
    }
    return "UNKNOWN";
}