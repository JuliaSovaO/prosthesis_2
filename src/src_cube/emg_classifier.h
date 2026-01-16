// emg_classifier.h - ADD extern declaration
#ifndef EMG_CLASSIFIER_H
#define EMG_CLASSIFIER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define NUM_FEATURES 18
#define NUM_GESTURES 10
#define HIDDEN_SIZE 300
#define WINDOW_SAMPLES 83  // 250ms at 332Hz

// Feature extraction
void extract_all_features(uint16_t* ch1, uint16_t* ch2, uint16_t* ch3, 
                         uint32_t n_samples, float* features_out);

// Normalize features using scaler
void normalize_features(float* features);

// ANN forward pass
int classify_gesture(float* features);

// Gesture names - DECLARE AS extern (defined in main.cpp)
extern const char* GESTURE_NAMES[NUM_GESTURES];

#ifdef __cplusplus
}
#endif

#endif /* EMG_CLASSIFIER_H */