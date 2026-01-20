#ifndef EMG_MODEL_H
#define EMG_MODEL_H

#include <stdint.h>

typedef enum {
    GESTURE_ROCK = 0,           // all closed
    GESTURE_SCISSORS = 1,       // index, middle opened, others closed
    GESTURE_PAPER = 2,          // all opened
    GESTURE_FUCK = 3,           // middle finger opened, others closed
    GESTURE_THREE = 4,          // index, middle, ring opened, others closed
    GESTURE_FOUR = 5,           // only thumb closed
    GESTURE_GOOD = 6,           // only thumb opened
    GESTURE_OKAY = 7,           // index and thumb make circle, others opened
    GESTURE_FINGER_GUN = 8,     // index, thumb opened, others closed
    GESTURE_REST = 9            // relaxed
} GestureType;

#define NUM_FEATURES 20
#define NUM_CLASSES 10

// Channel mapping:
// ch1: flexor carpi radialis (a0)
// ch2: brachioradialis (a1)
// ch3: flexor carpi ulnaris (a2)
// ch4: flexor digitorum superficialis (a3)

extern const float scaler_mean[NUM_FEATURES];
extern const float scaler_scale[NUM_FEATURES];

extern const float lr_coefficients[NUM_CLASSES][NUM_FEATURES];
extern const float lr_intercept[NUM_CLASSES];

extern const char* gesture_names[NUM_CLASSES];

GestureType predict_gesture(const float* features);

#endif // EMG_MODEL_H
