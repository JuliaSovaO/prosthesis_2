// src/src_cube/emg_classifier.h
#ifndef EMG_CLASSIFIER_H
#define EMG_CLASSIFIER_H

#include <stdint.h>
#include <math.h>

#define N_FEATURES 18
#define N_CLASSES 10
#define WINDOW_SIZE 375  // 250ms at 1500Hz

typedef struct {
    float buffer[3][WINDOW_SIZE];
    uint32_t idx;
    uint32_t sample_count;
} EMG_Buffer;

void init_emg_buffer(EMG_Buffer* buf);
void add_sample(EMG_Buffer* buf, uint16_t ch1, uint16_t ch2, uint16_t ch3);
int extract_features(EMG_Buffer* buf, float* features);
int classify_gesture(const float* features);
void scale_features(float* features);

#endif // EMG_CLASSIFIER_H