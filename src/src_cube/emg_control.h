#ifndef EMG_CONTROL_H
#define EMG_CONTROL_H

#include "servo_control.h"

#define EMG_WINDOW_SIZE     50

#define EMG_THRESHOLD_BASE   400
#define EMG_HYSTERESIS       70
#define STATE_DEBOUNCE_MS    150

#define CH1 0  // PA0 - EMG Sensor 1
#define CH2 1  // PA1 - EMG Sensor 2  
#define CH3 2  // PA2 - EMG Sensor 3
#define CH4 3  // PA3 - EMG Sensor 4

#define STATE_IDLE          0
#define STATE_CLOSE         1  // All fingers close
#define STATE_OPEN          2  // All fingers open
#define STATE_THUMB         3  // Thumb only
#define STATE_INDEX         4  // Index finger only
#define STATE_MIDDLE        5  // Middle finger only
#define STATE_RING          6  // Ring finger only
#define STATE_PINKY         7  // Pinky only
#define STATE_PINCH         8  // Thumb + index pinch

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