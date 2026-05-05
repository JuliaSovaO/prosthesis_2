#include "ann_inference.h"
#include "weights.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

#define MAX_HIDDEN 256

static float arr_max(const float* arr, int len) {
    float m = arr[0];
    for (int i = 1; i < len; i++)
        if (arr[i] > m) m = arr[i];
    return m;
}

static float arr_min(const float* arr, int len) {
    float m = arr[0];
    for (int i = 1; i < len; i++)
        if (arr[i] < m) m = arr[i];
    return m;
}

void ann_extract_features(const uint16_t raw_data[][4], uint16_t window_size, float* features) {
    int feat_idx = 0;
    
    for (int ch = 0; ch < 4; ch++) {
        float ch_data[ANN_WINDOW_SIZE];
        float sum = 0.0f, sum_sq = 0.0f, sum_abs = 0.0f;
        
        for (int i = 0; i < window_size; i++) {
            ch_data[i] = (float)raw_data[i][ch];
            sum += ch_data[i];
            sum_sq += ch_data[i] * ch_data[i];
            sum_abs += fabsf(ch_data[i]);
        }
        
        float range = arr_max(ch_data, window_size) - arr_min(ch_data, window_size);
        float threshold = 0.01f * (range + 1e-6f);
        
        // 1. RMS
        features[feat_idx++] = sqrtf(sum_sq / window_size);
        // 2. MAV
        features[feat_idx++] = sum_abs / window_size;
        // 3. WL
        float wl = 0.0f;
        for (int i = 1; i < window_size; i++)
            wl += fabsf(ch_data[i] - ch_data[i-1]);
        features[feat_idx++] = wl;
        // 4. ZC
        int zc = 0;
        for (int i = 1; i < window_size; i++) {
            if (fabsf(ch_data[i] - ch_data[i-1]) >= threshold) {
                if ((ch_data[i] >= 0 && ch_data[i-1] < 0) || 
                    (ch_data[i] < 0 && ch_data[i-1] >= 0))
                    zc++;
            }
        }
        features[feat_idx++] = (float)zc;
        // 5. SSC
        int ssc = 0;
        for (int i = 1; i < window_size - 1; i++) {
            float d1 = ch_data[i] - ch_data[i-1];
            float d2 = ch_data[i+1] - ch_data[i];
            if (d1 * d2 < 0 && (fabsf(d1) >= threshold || fabsf(d2) >= threshold))
                ssc++;
        }
        features[feat_idx++] = (float)ssc;
        // 6. IEMG
        features[feat_idx++] = sum_abs;
        // 7-8. Freq
        float mean_freq = (float)zc / (2.0f * window_size / 1000.0f);
        features[feat_idx++] = mean_freq;
        features[feat_idx++] = mean_freq * 0.8f;
    }
}

void ann_forward(const float* input, float* output) {
    float buf1[MAX_HIDDEN];
    float buf2[MAX_HIDDEN];
    float *curr_in = (float*)input;
    float *curr_out = buf1;
    int curr_in_size = ANN_INPUT_SIZE;
    
    // Process hidden layers 0 to ANN_NUM_LAYERS-2
    for (int layer = 0; layer < ANN_NUM_LAYERS - 1; layer++) {
        int out_size = ann_layer_sizes[layer + 1];
        const float* weights;
        const float* biases;
        
        switch (layer) {
            case 0: weights = ann_weights_0; biases = ann_biases_0; break;
            case 1: weights = ann_weights_1; biases = ann_biases_1; break;
            case 2: weights = ann_weights_2; biases = ann_biases_2; break;
            case 3: weights = ann_weights_3; biases = ann_biases_3; break;
            case 4: weights = ann_weights_4; biases = ann_biases_4; break;
            default: return;
        }
        
        for (int i = 0; i < out_size; i++) {
            float sum = biases[i];
            for (int j = 0; j < curr_in_size; j++) {
                sum += curr_in[j] * weights[j * out_size + i];
            }
            curr_out[i] = sum > 0 ? sum : 0; // ReLU
        }
        
        curr_in = curr_out;
        curr_in_size = out_size;
        curr_out = (curr_out == buf1) ? buf2 : buf1;
    }
    
    // Output layer
    int last_layer = ANN_NUM_LAYERS - 1;
    const float* last_weights;
    const float* last_biases;
    
    switch (last_layer) {
        case 0: last_weights = ann_weights_0; last_biases = ann_biases_0; break;
        case 1: last_weights = ann_weights_1; last_biases = ann_biases_1; break;
        case 2: last_weights = ann_weights_2; last_biases = ann_biases_2; break;
        case 3: last_weights = ann_weights_3; last_biases = ann_biases_3; break;
        case 4: last_weights = ann_weights_4; last_biases = ann_biases_4; break;
        default: return;
    }
    
    for (int i = 0; i < ANN_NUM_CLASSES; i++) {
        output[i] = last_biases[i];
        for (int j = 0; j < curr_in_size; j++) {
            output[i] += curr_in[j] * last_weights[j * ANN_NUM_CLASSES + i];
        }
    }
    
    // Softmax
    float max_val = output[0];
    for (int i = 1; i < ANN_NUM_CLASSES; i++)
        if (output[i] > max_val) max_val = output[i];
    
    float exp_sum = 0.0f;
    for (int i = 0; i < ANN_NUM_CLASSES; i++) {
        output[i] = expf(output[i] - max_val);
        exp_sum += output[i];
    }
    for (int i = 0; i < ANN_NUM_CLASSES; i++)
        output[i] /= exp_sum;
}

void ann_normalize_features(float* features) {
    for (int i = 0; i < ANN_INPUT_SIZE; i++) {
        if (ann_input_std[i] > 1e-6f) {
            features[i] = (features[i] - ann_input_mean[i]) / ann_input_std[i];
        } else {
            features[i] = 0.0f;
        }
        if (features[i] > 5.0f) features[i] = 5.0f;
        if (features[i] < -5.0f) features[i] = -5.0f;
    }
}

uint8_t ann_predict(const float* input) {
    float output[ANN_NUM_CLASSES];
    ann_forward(input, output);
    
    uint8_t best = 0;
    float best_prob = output[0];
    for (int i = 1; i < ANN_NUM_CLASSES; i++) {
        if (output[i] > best_prob) {
            best_prob = output[i];
            best = i;
        }
    }
    return best;
}

uint8_t ann_process_window(const uint16_t emg_buffer[][4], uint16_t buffer_size) {
    if (buffer_size < ANN_WINDOW_SIZE) return 0;
    
    float features[ANN_INPUT_SIZE];
    ann_extract_features(emg_buffer, ANN_WINDOW_SIZE, features);
    ann_normalize_features(features);
    return ann_predict(features);
}

float ann_get_confidence_from_buffer(const uint16_t emg_buffer[][4], uint16_t buffer_size) {
    if (buffer_size < ANN_WINDOW_SIZE) return 0.0f;
    
    float features[ANN_INPUT_SIZE];
    ann_extract_features(emg_buffer, ANN_WINDOW_SIZE, features);
    ann_normalize_features(features);
    
    float output[ANN_NUM_CLASSES];
    ann_forward(features, output);
    
    uint8_t pred = ann_predict(features);
    return output[pred];
}

void ann_init(void) {
    printf("ANN INIT - %d layers, %d features\n", ANN_NUM_LAYERS, ANN_INPUT_SIZE);
    printf("  Architecture: ");
    for (int i = 0; i <= ANN_NUM_LAYERS; i++) {
        printf("%d", ann_layer_sizes[i]);
        if (i < ANN_NUM_LAYERS) printf("->");
    }
    printf("\n");
    printf("  Features: RMS,MAV,WL,ZC,SSC,IEMG,MeanFreq,MedFreq\n");
}

const char* ann_get_class_name(uint8_t class_idx) {
    if (class_idx < ANN_NUM_CLASSES) return ann_class_names[class_idx];
    return "UNKNOWN";
}