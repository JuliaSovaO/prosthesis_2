#include "emg_classifier.h"
#include "emg_model.h"
#include <string.h>
#include <math.h>

EMG_Buffer emg_buffer;

void init_emg_buffer(EMG_Buffer* buf) {
    if (buf == NULL) return;
    
    memset(buf, 0, sizeof(EMG_Buffer));
    buf->sample_count = 0;
    buf->idx = 0;
}

void add_emg_sample(EMG_Buffer* buf, uint16_t ch1, uint16_t ch2, uint16_t ch3) {
    if (buf == NULL) return;
    
    // Store normalized values (0-1)
    buf->buffer[0][buf->idx] = (float)ch1 / 4095.0f;
    buf->buffer[1][buf->idx] = (float)ch2 / 4095.0f;
    buf->buffer[2][buf->idx] = (float)ch3 / 4095.0f;
    
    buf->idx = (buf->idx + 1) % WINDOW_SIZE;
    if (buf->sample_count < WINDOW_SIZE) {
        buf->sample_count++;
    }
}

// Simple moving average for DC offset
static float calculate_moving_average(const float* data, int n) {
    if (n <= 0) return 0.0f;
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum / n;
}

// Simple RMS calculation
static float calculate_simple_rms(const float* data, int n) {
    if (n <= 0) return 0.0f;
    
    float dc_offset = calculate_moving_average(data, n);
    float sum_squares = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float val = data[i] - dc_offset;
        sum_squares += val * val;
    }
    
    return sqrtf(sum_squares / n);
}

int classify_current_gesture(void) {
    float features[N_FEATURES] = {0};
    
    // Check if we have enough data
    if (emg_buffer.sample_count < WINDOW_SIZE) {
        return GESTURE_REST;
    }
    
    // Calculate RMS for each channel (only features we actually use)
    features[0] = calculate_simple_rms(emg_buffer.buffer[0], WINDOW_SIZE);  // ch1 RMS
    features[6] = calculate_simple_rms(emg_buffer.buffer[1], WINDOW_SIZE);  // ch2 RMS
    features[12] = calculate_simple_rms(emg_buffer.buffer[2], WINDOW_SIZE); // ch3 RMS
    
    // Fill other features with zeros (not used in current classifier)
    // features[5] = features[0] * features[0];   // ch1 Energy
    // features[11] = features[6] * features[6];  // ch2 Energy
    // features[17] = features[12] * features[12]; // ch3 Energy
    
    // Get prediction
    return predict_gesture(features);
}