#include "emg_control.h"
#include "main.h"
#include "ann/ann_inference.h"
#include <stdio.h>
#include <string.h>

extern ADC_HandleTypeDef hadc2;

// EMG buffer for ANN (circular buffer)
#define EMG_BUFFER_SIZE     200  // Enough for multiple windows
static uint16_t emg_buffer[EMG_BUFFER_SIZE][4];
static uint16_t emg_buffer_idx = 0;
static uint16_t emg_buffer_filled = 0;

// Gesture mapping from ANN output to servo states
typedef struct {
    uint8_t gesture_id;
    const char* name;
    uint8_t thumb_angle;
    uint8_t index_angle;
    uint8_t middle_angle;
    uint8_t ring_angle;
    uint8_t pinky_angle;
} GestureMap_t;

// Note: The order here must match the order from training!
static const GestureMap_t gesture_map[] = {
    {0, "finger-gun", 0,   90,  0,   0,   0},   // finger-gun
    {1, "four",       0,   90,  120, 130, 90},  // four
    {2, "fuck",       150, 90,  170, 180, 120}, // fuck (ONE)
    {3, "good",       90,  0,   0,   0,   0},   // good
    {4, "okay",       60,  60,  170, 180, 120}, // okay
    {5, "paper",      0,   0,   10,  20,  0},   // paper
    {6, "rest",       0,   0,   10,  20,  0},   // rest
    {7, "rock",       150, 180, 170, 180, 120}, // rock
    {8, "scissors",   0,   90,  0,   0,   0},   // scissors
    {9, "three",      0,   90,  120, 130, 0},   // three
};

static uint8_t current_gesture = 6;  // Start with rest (index 6)
static uint8_t last_gesture = 6;
static uint32_t last_gesture_time = 0;
static uint32_t gesture_debounce_ms = 200;

// Filter for raw EMG (moving average)
static uint16_t emg_filtered[4] = {0};
static uint32_t emg_sum[4] = {0};
static uint16_t emg_filter_buf[4][ANN_WINDOW_SIZE];
static uint8_t emg_filter_idx = 0;
static uint8_t emg_filter_full = 0;

void EMG_Control_Init(void) {
    // Initialize EMG filter buffers
    memset(emg_filter_buf, 0, sizeof(emg_filter_buf));
    memset(emg_sum, 0, sizeof(emg_sum));
    emg_filter_idx = 0;
    emg_filter_full = 0;
    
    // Initialize ANN
    ann_init();
    
    printf("\r\n=== EMG ANN Gesture Control ===\r\n");
    printf("ANN Input size: %d\r\n", ANN_INPUT_SIZE);
    printf("ANN Classes: %d\r\n", ANN_NUM_CLASSES);
    printf("Window size: %d samples\r\n", ANN_WINDOW_SIZE);
    printf("Gesture debounce: %lu ms\r\n\r\n", (unsigned long)gesture_debounce_ms);
    
    for (int i = 0; i < ANN_NUM_CLASSES; i++) {
        printf("  %d: %s\r\n", i, ann_get_class_name(i));
    }
    printf("\r\n");
    
    // Start with open hand (rest position)
    SetServo1Angle(gesture_map[current_gesture].thumb_angle);
    SetServo2Angle(gesture_map[current_gesture].index_angle);
    SetServo3Angle(gesture_map[current_gesture].middle_angle);
    SetServo4Angle(gesture_map[current_gesture].ring_angle);
    SetServo5Angle(gesture_map[current_gesture].pinky_angle);
}

// Update moving average filter for EMG
static void update_emg_filter(uint16_t raw[4]) {
    if (emg_filter_full) {
        for (int ch = 0; ch < 4; ch++) {
            emg_sum[ch] -= emg_filter_buf[ch][emg_filter_idx];
        }
    }
    
    for (int ch = 0; ch < 4; ch++) {
        emg_filter_buf[ch][emg_filter_idx] = raw[ch];
        emg_sum[ch] += raw[ch];
        emg_filtered[ch] = emg_sum[ch] / (emg_filter_full ? ANN_WINDOW_SIZE : (emg_filter_idx + 1));
    }
    
    emg_filter_idx = (emg_filter_idx + 1) % ANN_WINDOW_SIZE;
    if (emg_filter_idx == 0) {
        emg_filter_full = 1;
    }
}

// Apply gesture to servos with smooth movement
static void apply_gesture(uint8_t gesture_id) {
    if (gesture_id >= ANN_NUM_CLASSES) return;
    
    const GestureMap_t* g = &gesture_map[gesture_id];
    
    // Smooth movement - gradually move to target angles
    static uint8_t current_angles[5] = {0};
    static uint32_t last_update = 0;
    uint32_t now = HAL_GetTick();
    
    if (now - last_update > 10) {  // Update every 10ms
        last_update = now;
        
        #define SMOOTH_STEP 3
        
        // Thumb
        if (current_angles[0] < g->thumb_angle) {
            current_angles[0] += SMOOTH_STEP;
            if (current_angles[0] > g->thumb_angle) current_angles[0] = g->thumb_angle;
        } else if (current_angles[0] > g->thumb_angle) {
            current_angles[0] -= SMOOTH_STEP;
            if (current_angles[0] < g->thumb_angle) current_angles[0] = g->thumb_angle;
        }
        
        // Index
        if (current_angles[1] < g->index_angle) {
            current_angles[1] += SMOOTH_STEP;
            if (current_angles[1] > g->index_angle) current_angles[1] = g->index_angle;
        } else if (current_angles[1] > g->index_angle) {
            current_angles[1] -= SMOOTH_STEP;
            if (current_angles[1] < g->index_angle) current_angles[1] = g->index_angle;
        }
        
        // Middle
        if (current_angles[2] < g->middle_angle) {
            current_angles[2] += SMOOTH_STEP;
            if (current_angles[2] > g->middle_angle) current_angles[2] = g->middle_angle;
        } else if (current_angles[2] > g->middle_angle) {
            current_angles[2] -= SMOOTH_STEP;
            if (current_angles[2] < g->middle_angle) current_angles[2] = g->middle_angle;
        }
        
        // Ring
        if (current_angles[3] < g->ring_angle) {
            current_angles[3] += SMOOTH_STEP;
            if (current_angles[3] > g->ring_angle) current_angles[3] = g->ring_angle;
        } else if (current_angles[3] > g->ring_angle) {
            current_angles[3] -= SMOOTH_STEP;
            if (current_angles[3] < g->ring_angle) current_angles[3] = g->ring_angle;
        }
        
        // Pinky
        if (current_angles[4] < g->pinky_angle) {
            current_angles[4] += SMOOTH_STEP;
            if (current_angles[4] > g->pinky_angle) current_angles[4] = g->pinky_angle;
        } else if (current_angles[4] > g->pinky_angle) {
            current_angles[4] -= SMOOTH_STEP;
            if (current_angles[4] < g->pinky_angle) current_angles[4] = g->pinky_angle;
        }
        
        SetServo1Angle(current_angles[0]);
        SetServo2Angle(current_angles[1]);
        SetServo3Angle(current_angles[2]);
        SetServo4Angle(current_angles[3]);
        SetServo5Angle(current_angles[4]);
    }
}

void EMG_Control_Process(void) {
    // static uint32_t last_window_time = 0;
    static uint16_t samples_since_last_window = 0;
    uint32_t now = HAL_GetTick();
    
    if (!data_rdy_f) {
        return;
    }
    
    // Get latest EMG samples (4 channels)
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    uint16_t raw_emg[4] = {
        adc_buffer[last_idx + 0],
        adc_buffer[last_idx + 1],
        adc_buffer[last_idx + 2],
        adc_buffer[last_idx + 3]
    };
    
    // Update filter
    update_emg_filter(raw_emg);
    
    // Add to circular buffer for ANN processing
    emg_buffer[emg_buffer_idx][0] = emg_filtered[0];
    emg_buffer[emg_buffer_idx][1] = emg_filtered[1];
    emg_buffer[emg_buffer_idx][2] = emg_filtered[2];
    emg_buffer[emg_buffer_idx][3] = emg_filtered[3];
    
    emg_buffer_idx = (emg_buffer_idx + 1) % EMG_BUFFER_SIZE;
    if (emg_buffer_filled < EMG_BUFFER_SIZE) {
        emg_buffer_filled++;
    }
    
    // Process ANN every window step
    samples_since_last_window++;
    
    if (samples_since_last_window >= ANN_WINDOW_STEP && emg_buffer_filled >= ANN_WINDOW_SIZE) {
        samples_since_last_window = 0;
        
        // Get the last ANN_WINDOW_SIZE samples
        uint16_t window_start = (emg_buffer_idx - ANN_WINDOW_SIZE + EMG_BUFFER_SIZE) % EMG_BUFFER_SIZE;
        uint16_t window[ANN_WINDOW_SIZE][4];
        
        for (int i = 0; i < ANN_WINDOW_SIZE; i++) {
            int idx = (window_start + i) % EMG_BUFFER_SIZE;
            window[i][0] = emg_buffer[idx][0];
            window[i][1] = emg_buffer[idx][1];
            window[i][2] = emg_buffer[idx][2];
            window[i][3] = emg_buffer[idx][3];
        }
        
        // Run ANN inference
        uint8_t predicted_gesture = ann_process_window(window, ANN_WINDOW_SIZE);
        
        // Debounce gesture changes
        if (predicted_gesture != last_gesture) {
            if (now - last_gesture_time > gesture_debounce_ms) {
                // Gesture changed
                if (predicted_gesture != current_gesture) {
                    printf("Gesture: %s -> %s\r\n", 
                           ann_get_class_name(current_gesture),
                           ann_get_class_name(predicted_gesture));
                    current_gesture = predicted_gesture;
                }
                last_gesture = predicted_gesture;
                last_gesture_time = now;
            }
        }
        
        // Visual feedback - blink green LED when gesture changes
        static uint8_t last_led_state = 0;
        if (current_gesture != last_led_state) {
            HAL_GPIO_TogglePin(USER_LED_GREEN_GPIO_Port, USER_LED_GREEN_Pin);
            last_led_state = current_gesture;
        }
    }
    
    // Apply current gesture to servos
    apply_gesture(current_gesture);
    
    data_rdy_f = false;
}

void EMG_AutoCalibrate(void) {
    printf("\r\n=== AUTO-CALIBRATION ===\r\n");
    printf("Relax for 3 seconds to calibrate baseline...\r\n");
    
    uint32_t start_time = HAL_GetTick();
    uint32_t sums[4] = {0};
    uint16_t count = 0;
    
    while (HAL_GetTick() - start_time < 3000) {
        if (data_rdy_f) {
            int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
            sums[0] += adc_buffer[last_idx + 0];
            sums[1] += adc_buffer[last_idx + 1];
            sums[2] += adc_buffer[last_idx + 2];
            sums[3] += adc_buffer[last_idx + 3];
            count++;
            data_rdy_f = false;
        }
        HAL_Delay(1);
    }
    
    if (count > 0) {
        printf("Baseline levels:\r\n");
        for (int i = 0; i < 4; i++) {
            uint16_t baseline = sums[i] / count;
            printf("  Channel %d: %d\r\n", i+1, baseline);
        }
    }
    printf("Calibration complete!\r\n");
}

void EMG_PrintRawData(void) {
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    printf("%d,%d,%d,%d\r\n",
           adc_buffer[last_idx + 0],
           adc_buffer[last_idx + 1],
           adc_buffer[last_idx + 2],
           adc_buffer[last_idx + 3]);
}