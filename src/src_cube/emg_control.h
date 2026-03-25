#ifndef EMG_CONTROL_H
#define EMG_CONTROL_H

#include "servo_control.h"
#include "ann/weights.h"
#include <stdbool.h>

// ANN configuration
#define EMG_WINDOW_SIZE     50
#define EMG_WINDOW_STEP     25

// Gesture states (for backward compatibility)
#define STATE_IDLE          0
#define STATE_CLOSE         1
#define STATE_OPEN          2
#define STATE_THUMB         3
#define STATE_INDEX         4
#define STATE_MIDDLE        5
#define STATE_RING          6
#define STATE_PINKY         7
#define STATE_PINCH         8

typedef struct {
    uint8_t current_angle;
    uint8_t target_angle;
    uint8_t min_angle;
    uint8_t max_angle;
} ServoState_t;

extern volatile bool data_rdy_f;
extern uint16_t adc_buffer[];

void EMG_Control_Init(void);
void EMG_Control_Process(void);
void EMG_AutoCalibrate(void);
void EMG_PrintRawData(void);

#endif