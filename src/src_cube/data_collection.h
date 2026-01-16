// data_collection.h
#ifndef DATA_COLLECTION_H
#define DATA_COLLECTION_H

#include "main.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void collect_gesture_data(int gesture_id);
void data_collection_main(void);

#ifdef __cplusplus
}
#endif

#endif // DATA_COLLECTION_H