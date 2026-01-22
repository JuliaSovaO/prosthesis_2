#ifndef EMG_MODEL_H
#define EMG_MODEL_H

#include <stdint.h>

typedef enum {
    GESTURE_ROCK = 0,
    GESTURE_SCISSORS = 1,
    GESTURE_PAPER = 2,
    GESTURE_FUCK = 3,
    GESTURE_THREE = 4,
    GESTURE_FOUR = 5,
    GESTURE_GOOD = 6,
    GESTURE_OKAY = 7,
    GESTURE_FINGER_GUN = 8,
    GESTURE_REST = 9,
} GestureType;

#define NUM_FEATURES 20
#define NUM_CLASSES 10

extern const float scaler_mean[NUM_FEATURES];
extern const float scaler_scale[NUM_FEATURES];

extern const float lr_coefficients[NUM_CLASSES][NUM_FEATURES];
extern const float lr_intercept[NUM_CLASSES];

extern const char* gesture_names[NUM_CLASSES];

GestureType predict_gesture(const float* features);

#endif // EMG_MODEL_H
