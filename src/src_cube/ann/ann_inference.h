#ifndef ANN_INFERENCE_H
#define ANN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <math.h>

#ifndef ANN_INPUT_SIZE
#define ANN_INPUT_SIZE     32
#endif
#ifndef ANN_WINDOW_SIZE
#define ANN_WINDOW_SIZE    250
#endif
#ifndef ANN_NUM_CLASSES
#define ANN_NUM_CLASSES    10
#endif

void ann_extract_features(const uint16_t raw_data[][4], uint16_t window_size, float* features);
void ann_forward(const float* input, float* output);
uint8_t ann_predict(const float* input);
uint8_t ann_process_window(const uint16_t emg_buffer[][4], uint16_t buffer_size);
void ann_init(void);
const char* ann_get_class_name(uint8_t class_idx);
void ann_normalize_features(float* features);
float ann_get_confidence_from_buffer(const uint16_t emg_buffer[][4], uint16_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif