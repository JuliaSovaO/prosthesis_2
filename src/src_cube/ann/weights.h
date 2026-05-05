#ifndef ANN_WEIGHTS_H
#define ANN_WEIGHTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ANN_INPUT_SIZE 32
#define ANN_WINDOW_SIZE 250
#define ANN_NUM_CLASSES 10
#define ANN_NUM_LAYERS 5
#define ANN_FEATURES_PER_CHANNEL 8

extern const char* ann_class_names[];
extern const int ann_layer_sizes[];

extern const float ann_weights_0[];
extern const float ann_biases_0[];
extern const float ann_weights_1[];
extern const float ann_biases_1[];
extern const float ann_weights_2[];
extern const float ann_biases_2[];
extern const float ann_weights_3[];
extern const float ann_biases_3[];
extern const float ann_weights_4[];
extern const float ann_biases_4[];

extern const float ann_input_mean[];
extern const float ann_input_std[];

#ifdef __cplusplus
}
#endif

#endif
