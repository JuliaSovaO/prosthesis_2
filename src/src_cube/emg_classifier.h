#ifndef EMG_CLASSIFIER_H
#define EMG_CLASSIFIER_H

#include <stdint.h>
#include <math.h>

// Configuration
#define N_FEATURES 18  // 3 channels × 6 features (simplified to 3 actually used)
#define WINDOW_SIZE 200  // 200 samples at 1000Hz = 200ms window

typedef struct {
    float buffer[3][WINDOW_SIZE];
    uint32_t idx;
    uint32_t sample_count;
} EMG_Buffer;

// Declare the global buffer
extern EMG_Buffer emg_buffer;

// Function prototypes
void init_emg_buffer(EMG_Buffer* buf);
void add_emg_sample(EMG_Buffer* buf, uint16_t ch1, uint16_t ch2, uint16_t ch3);
int classify_current_gesture(void);

#endif // EMG_CLASSIFIER_H