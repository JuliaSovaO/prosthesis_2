#ifndef EMG_MODEL_H
#define EMG_MODEL_H

#include <stdint.h>
#include <math.h>

#define N_FEATURES 18  // 3 channels × 6 features
#define N_GESTURES 10

// Gesture definitions - MUST match your Python GESTURE_MAP
#define GESTURE_REST 0
#define GESTURE_ROCK 1
#define GESTURE_SCISSORS 2
#define GESTURE_PAPER 3
#define GESTURE_FUCK 4      // This is "ONE" gesture
#define GESTURE_THREE 5
#define GESTURE_FOUR 6
#define GESTURE_GOOD 7
#define GESTURE_OKAY 8
#define GESTURE_FINGER_GUN 9

// Also define GESTURE_ONE as an alias for GESTURE_FUCK
#define GESTURE_ONE GESTURE_FUCK

// OPTIMIZED CLASSIFIER based on your feature ranges
static inline uint8_t predict_gesture(const float* features) {
    // Extract RMS values
    float ch1_rms = features[0];
    float ch2_rms = features[6];
    float ch3_rms = features[12];
    
    // Calculate simple ratios
    float ratio_ch2_ch1 = (ch1_rms > 0.001f) ? ch2_rms / ch1_rms : 100.0f;
    float ratio_ch3_ch1 = (ch1_rms > 0.001f) ? ch3_rms / ch1_rms : 100.0f;
    
    // DEBUG - Uncomment to see what's happening
    // printf("C1=%.4f C2=%.4f C3=%.4f R21=%.1f R31=%.1f\n", 
    //        ch1_rms, ch2_rms, ch3_rms, ratio_ch2_ch1, ratio_ch3_ch1);
    
    // OPTIMIZED DECISION TREE based on your actual data:
    
    // 1. SCISSORS: Extremely low ch3 (< 0.005) - VERY DISTINCTIVE
    if (ch3_rms < 0.005f) {
        return GESTURE_SCISSORS;
    }
    
    // 2. PAPER: Very low ch2 (< 0.007) - VERY DISTINCTIVE
    if (ch2_rms < 0.007f) {
        return GESTURE_PAPER;
    }
    
    // 3. GOOD: Extreme ch2/ch1 ratio (> 15)
    if (ratio_ch2_ch1 > 15.0f) {
        return GESTURE_GOOD;
    }
    
    // 4. THREE: High ch1 (> 0.018) and ch2 ≈ ch1
    if (ch1_rms > 0.018f && ratio_ch2_ch1 > 0.9f && ratio_ch2_ch1 < 1.2f) {
        return GESTURE_THREE;
    }
    
    // 5. FOUR: High ch1 (> 0.018) and ch2 > ch1
    if (ch1_rms > 0.018f && ratio_ch2_ch1 > 1.2f) {
        return GESTURE_FOUR;
    }
    
    // 6. REST: Highest ch2 (> 0.032) and high ratio
    if (ch2_rms > 0.032f && ratio_ch2_ch1 > 5.0f) {
        return GESTURE_REST;
    }
    
    // 7. ROCK: High ch2 (> 0.032) and moderate ratio
    if (ch2_rms > 0.032f && ratio_ch2_ch1 > 3.5f && ratio_ch2_ch1 < 4.5f) {
        return GESTURE_ROCK;
    }
    
    // 8. OKAY: Specific range from your data
    if (ch1_rms > 0.010f && ch1_rms < 0.014f && 
        ratio_ch2_ch1 > 2.0f && ratio_ch2_ch1 < 3.0f) {
        return GESTURE_OKAY;
    }
    
    // 9. FINGER_GUN: High ch3 relative to ch1
    if (ratio_ch3_ch1 > 4.0f && ratio_ch3_ch1 < 6.0f && ch2_rms > 0.025f) {
        return GESTURE_FINGER_GUN;
    }
    
    // 10. FUCK/ONE: Moderate ratios, high ch3
    if (ratio_ch2_ch1 > 4.0f && ratio_ch2_ch1 < 5.0f && 
        ratio_ch3_ch1 > 4.0f && ratio_ch3_ch1 < 5.0f) {
        return GESTURE_FUCK;
    }
    
    // Default fallback
    return GESTURE_REST;
}

#endif // EMG_MODEL_H