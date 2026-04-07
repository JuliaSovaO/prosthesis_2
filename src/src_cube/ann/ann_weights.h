#ifndef ANN_WEIGHTS_H
#define ANN_WEIGHTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Model configuration
#define ANN_INPUT_SIZE     24
#define ANN_WINDOW_SIZE    50
#define ANN_NUM_CLASSES    10
#define ANN_NUM_LAYERS     4

// Layer sizes
extern const int ann_layer_sizes[];

// Weights and biases
extern const float ann_weights_0[];
extern const float ann_biases_0[];
extern const float ann_weights_1[];
extern const float ann_biases_1[];
extern const float ann_weights_2[];
extern const float ann_biases_2[];
extern const float ann_weights_3[];
extern const float ann_biases_3[];

// Pointer arrays
extern const float* ann_weights_ptrs[];
extern const float* ann_biases_ptrs[];

// Normalization parameters
extern const float ann_input_mean[];
extern const float ann_input_std[];

// Class names
extern const char* ann_class_names[];

#ifdef __cplusplus
}
#endif

#endif // ANN_WEIGHTS_H