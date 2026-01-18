#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Sampling configuration - SET TO 1500Hz AS REQUESTED
#define ADC_CHANNELS 3
#define SAMPLING_RATE_HZ 1500  // Changed from 2000 to 1500
#define BUFFER_SIZE_MS 1000
#define BUFFER_SAMPLES ((SAMPLING_RATE_HZ * BUFFER_SIZE_MS) / 1000)

// Gesture definitions - Will be overridden by emg_model.h
#define NUM_GESTURES 10

// Data collection configuration
#define COLLECTION_DURATION_MS 5000
#define SAMPLES_PER_TRIAL ((SAMPLING_RATE_HZ * COLLECTION_DURATION_MS) / 1000)
#define TRIALS_PER_GESTURE 20
#define REST_BETWEEN_TRIALS_MS 10000
#define REST_BETWEEN_GESTURES_MS 300000

// Collection modes
typedef enum {
    MODE_IDLE = 0,
    MODE_COLLECTING,
    MODE_PROCESSING
} CollectionMode;

#ifdef __cplusplus
}
#endif

#endif // COMMON_DEFS_H