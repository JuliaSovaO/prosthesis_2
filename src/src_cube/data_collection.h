// data_collection.h - HEADER ONLY
#ifndef DATA_COLLECTION_H
#define DATA_COLLECTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "common_defs.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Data collection modes
typedef enum {
    MODE_IDLE = 0,
    MODE_COLLECTING,
    MODE_PROCESSING,
    MODE_COMPLETE
} CollectionMode;

// Data collection functions
void data_collection_init(void);
void data_collection_start(uint8_t gesture_id);
void data_collection_stop(void);
void data_collection_process(void);
void data_collection_main(void);

// Data export
void export_data_to_serial(uint8_t gesture_id);
void export_data_summary(void);

// Buffer management
uint32_t get_available_samples(void);
uint32_t get_sample_rate(void);

#ifdef __cplusplus
}
#endif

#endif // DATA_COLLECTION_H