#ifndef EMG_MODEL_H
#define EMG_MODEL_H

#include <stdint.h>
#include <math.h>

#define N_FEATURES 24  // 4 channels × 6 features
#define N_GESTURES 10

#define GESTURE_REST 0
#define GESTURE_ROCK 1
#define GESTURE_SCISSORS 2
#define GESTURE_PAPER 3
#define GESTURE_FUCK 4     
#define GESTURE_THREE 5
#define GESTURE_FOUR 6
#define GESTURE_GOOD 7
#define GESTURE_OKAY 8
#define GESTURE_FINGER_GUN 9
#define GESTURE_ONE GESTURE_FUCK

// TEMPORARY: Simple classifier for 4 channels
static inline uint8_t predict_gesture(const float* features) {
    // For now, use only first 3 channels as before
    // TODO: Update with 4-channel logic when you have data
    
    float ch1_rms = features[0];
    float ch2_rms = features[6];
    float ch3_rms = features[12];
    // float ch4_rms = features[18];  // Comment out unused variable
    
    // For now, use the same logic as 3 channels
    float ratio_ch2_ch1 = (ch1_rms > 0.001f) ? ch2_rms / ch1_rms : 100.0f;
    
    // SCISSORS: Very low ch3
    if (ch3_rms < 0.005f) {
        return GESTURE_SCISSORS;
    }
    
    // PAPER: Very low ch2
    if (ch2_rms < 0.007f) {
        return GESTURE_PAPER;
    }
    
    // GOOD: Extreme ratio
    if (ratio_ch2_ch1 > 15.0f) {
        return GESTURE_GOOD;
    }
    
    return GESTURE_REST;
}

#endif // EMG_MODEL_H