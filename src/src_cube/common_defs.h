// common_defs.h - FIXED
#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>  // ADD THIS for uint16_t, uint32_t, uint8_t

// Sampling configuration
#define ADC_CHANNELS 3
#define SAMPLING_RATE_HZ 2000
#define BUFFER_SIZE_MS 1000  // 1 second buffer
#define BUFFER_SAMPLES ((SAMPLING_RATE_HZ * BUFFER_SIZE_MS) / 1000)

// Gesture definitions
#define NUM_GESTURES 10
extern const char* GESTURE_NAMES[NUM_GESTURES];

// Circular buffer for data collection
typedef struct {
    uint16_t ch1;
    uint16_t ch2;
    uint16_t ch3;
    uint32_t timestamp;
    uint8_t gesture_id;
} EMG_Sample;

// Data collection configuration
#define COLLECTION_DURATION_MS 5000  // 5 seconds per trial
#define SAMPLES_PER_TRIAL ((SAMPLING_RATE_HZ * COLLECTION_DURATION_MS) / 1000)
#define TRIALS_PER_GESTURE 20
#define REST_BETWEEN_TRIALS_MS 10000  // 10 seconds
#define REST_BETWEEN_GESTURES_MS 300000  // 5 minutes

#ifdef __cplusplus
}
#endif

#endif // COMMON_DEFS_H