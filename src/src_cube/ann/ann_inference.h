#ifndef ANN_INFERENCE_H
#define ANN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#define ANN_INPUT_SIZE     24  // 4 channels * 6 features
#define ANN_WINDOW_SIZE    50  // samples per window
#define ANN_WINDOW_STEP    25  // samples overlap

void ann_extract_features(
    const uint16_t raw_data[][4],
    uint16_t window_size,
    float* features
);

void ann_forward(const float* input, float* output);
uint8_t ann_predict(const float* input);
uint8_t ann_process_window(const uint16_t emg_buffer[][4], uint16_t buffer_size);
void ann_init(void);
const char* ann_get_class_name(uint8_t class_idx);
void ann_normalize_features(float* features);

#ifdef __cplusplus
}
#endif

#endif // ANN_INFERENCE_H