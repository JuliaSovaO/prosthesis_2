#ifndef EMG_CONTROL_H
#define EMG_CONTROL_H

#include "servo_control.h"
#include "ann/ann_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EMG_WINDOW_SIZE     250     // 250ms window at ~1000Hz
#define EMG_WINDOW_STEP     50      // 50ms step 
#define EMG_BUFFER_SIZE     300     // must be > WINDOW_SIZE
#define PREDICTION_HISTORY  7      
#define STATE_DEBOUNCE_MS   150    
#define MIN_GESTURE_INTERVAL_MS 1500
#define ACTIVITY_TIMEOUT_MS 500 
#define MIN_CONFIDENCE      0.4f 

typedef enum {
    STATE_ROCK = 0,
    STATE_SCISSORS = 1,
    STATE_PAPER = 2,
    STATE_FUCK = 3,
    STATE_THREE = 4,
    STATE_FOUR = 5,
    STATE_GOOD = 6,
    STATE_OKAY = 7,
    STATE_FINGER_GUN = 8,
    STATE_REST = 9,
    STATE_IDLE = 9,
    STATE_COUNT = 10
} ProsthesisState_t;

typedef struct {
    uint8_t thumb_angle;
    uint8_t index_angle;
    uint8_t middle_angle;
    uint8_t ring_angle;
    uint8_t pinky_angle;
} GestureAngles_t;

extern volatile bool data_rdy_f;
extern uint16_t adc_buffer[];
extern PCA9685_HandleTypeDef pca9685;

void EMG_Control_Init(void);
void EMG_Control_Process(void);
void EMG_AutoCalibrate(void);
void EMG_PrintRawData(void);
void EMG_SetDebugMode(uint8_t enable);
const char* EMG_GetStateName(ProsthesisState_t state);

#ifdef __cplusplus
}
#endif

#endif // EMG_CONTROL_H