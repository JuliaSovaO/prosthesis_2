#ifndef EMG_CONTROL_H
#define EMG_CONTROL_H

#include <stdint.h>
#include <stdbool.h>

// Gesture definitions
typedef enum {
    GESTURE_REST = 0,
    GESTURE_ROCK,
    GESTURE_SCISSORS,
    GESTURE_PAPER,
    GESTURE_ONE,
    GESTURE_THREE,
    GESTURE_FOUR,
    GESTURE_GOOD,
    GESTURE_OKAY,
    GESTURE_FINGER_GUN,
    NUM_GESTURES
} GestureType;

// Feature structure
typedef struct {
    float rms;
    float var;
    float mav;
    float ssc;
    float zc;
    float wl;
} TimeDomainFeatures;

// Data collection functions
void EMG_DataCollection_Init(void);
void EMG_DataCollection_Start(GestureType gesture);
void EMG_DataCollection_Stop(void);
void EMG_DataCollection_ProcessSample(void);
bool EMG_DataCollection_IsRecording(void);
GestureType EMG_DataCollection_GetCurrentGesture(void);
void EMG_DataCollection_ExtractFeatures(uint32_t window_start, 
                                        TimeDomainFeatures* ch1_feat,
                                        TimeDomainFeatures* ch2_feat, 
                                        TimeDomainFeatures* ch3_feat);
void EMG_DataCollection_PrintFeaturesCSV(TimeDomainFeatures* ch1, 
                                         TimeDomainFeatures* ch2, 
                                         TimeDomainFeatures* ch3);
uint32_t EMG_DataCollection_GetSampleIndex(void);
uint32_t EMG_DataCollection_GetSamplesPerWindow(void);

#endif