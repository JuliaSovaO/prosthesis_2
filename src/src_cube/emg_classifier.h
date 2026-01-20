#ifndef EMG_CLASSIFIER_H
#define EMG_CLASSIFIER_H

#include "common_defs.h"
#include "emg_model.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#define WINDOW_SIZE 150  // 150 samples at 1000Hz = 150ms
#define OVERLAP 100      // 100 samples overlap
#define STEP_SIZE (WINDOW_SIZE - OVERLAP)  // 50 samples step

#define NUM_CHANNELS 4
#define FEATURES_PER_CHANNEL 5
#define TOTAL_FEATURES (NUM_CHANNELS * FEATURES_PER_CHANNEL)

// EMG buffer structure
typedef struct {
    int16_t data[WINDOW_SIZE][NUM_CHANNELS];
    uint16_t write_index;
    bool is_full;
} EMG_Buffer;

// Function prototypes
void emg_buffer_init(EMG_Buffer* buffer);
void emg_buffer_add_sample(EMG_Buffer* buffer, int16_t ch1, int16_t ch2, int16_t ch3, int16_t ch4);
bool emg_buffer_process_window(EMG_Buffer* buffer, float* features);
GestureType classify_gesture(const float* features);
void extract_features_from_window(const int16_t window[WINDOW_SIZE][NUM_CHANNELS], float* features);

#endif // EMG_CLASSIFIER_H