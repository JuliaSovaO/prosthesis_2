#include "main.h"
#include "emg_control.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// Circular buffer for EMG data (4 channels)
static uint16_t emg_circular_buffer[EMG_BUFFER_SIZE][4];
static uint16_t buffer_write_idx = 0;
static uint16_t buffer_filled = 0;

// Prediction smoothing
static uint8_t prediction_history[PREDICTION_HISTORY];
static uint8_t history_idx = 0;
static uint8_t history_filled = 0;

// State management
static ProsthesisState_t current_state = STATE_IDLE;
static uint32_t last_state_change_time = 0;
static uint32_t last_activity_time = 0;
static uint32_t last_gesture_time = 0;  // Track last gesture change for servo protection

// Debug mode
static uint8_t debug_mode = 1;
static uint32_t last_debug_print = 0;
static uint32_t frame_count = 0;

// Baseline calibration values
static uint16_t baseline[4] = {0};
static uint16_t threshold[4] = {0};

// Servo angles for each gesture
static const GestureAngles_t gesture_angles[] = {
    /* ROCK (fist - all closed) */
    {180, 180, 130, 180, 180},
    
    /* SCISSORS (index + middle open) */
    {180, 10, 10, 180, 180},
    
    /* PAPER (all open) */
    {0, 10, 10, 80, 10},
    
    /* FUCK (middle finger only) */
    {180, 180, 10, 180, 180},
    
    /* THREE (index + middle + ring open) */
    {180, 10, 10, 80, 180},
    
    /* FOUR (only thumb closed) */
    {180, 10, 10, 80, 10},
    
    /* GOOD (only thumb open) */
    {0, 180, 130, 180, 180},
    
    /* OKAY (thumb + index circle) */
    {180, 180, 10, 80, 10},
    
    /* FINGER_GUN (index + thumb) */
    {0, 10, 160, 180, 180},
    
    /* REST (relaxed) */
    {30, 40, 40, 80, 40}
};

static const char* state_names[] = {
    "ROCK", "SCISSORS", "PAPER", "FUCK", "THREE",
    "FOUR", "GOOD", "OKAY", "FINGER_GUN", "REST"
};

void EMG_Control_Init(void) {
    memset(emg_circular_buffer, 0, sizeof(emg_circular_buffer));
    buffer_write_idx = 0;
    buffer_filled = 0;
    
    for (int i = 0; i < PREDICTION_HISTORY; i++) {
        prediction_history[i] = STATE_IDLE;
    }
    history_idx = 0;
    history_filled = 0;
    
    current_state = STATE_IDLE;
    last_state_change_time = HAL_GetTick();
    last_activity_time = HAL_GetTick();
    last_gesture_time = HAL_GetTick();  // Initialize gesture timer
    
    ann_init();
    
    // Set initial servo positions (rest/open hand)
    const GestureAngles_t* idle_angles = &gesture_angles[STATE_REST];
    SetServo1Angle(idle_angles->thumb_angle);
    SetServo2Angle(idle_angles->index_angle);
    SetServo3Angle(idle_angles->middle_angle);
    SetServo4Angle(idle_angles->ring_angle);
    SetServo5Angle(idle_angles->pinky_angle);
    
    printf("\r\n=== 4-CHANNEL EMG ANN CONTROL ===\r\n");
    printf("Window: %d samples, Step: %d samples, History: %d\r\n", 
           EMG_WINDOW_SIZE, EMG_WINDOW_STEP, PREDICTION_HISTORY);
    printf("ANN Input: %d features, Output: %d classes\r\n", ANN_INPUT_SIZE, ANN_NUM_CLASSES);
    printf("Min confidence: %.2f, Min gesture interval: %dms\r\n", MIN_CONFIDENCE, MIN_GESTURE_INTERVAL_MS);
    printf("Activity timeout: %dms\r\n", ACTIVITY_TIMEOUT_MS);
}

void EMG_AutoCalibrate(void) {
    printf("\r\n=== AUTO-CALIBRATION ===\r\n");
    printf("Relax your hand for 3 seconds...\r\n");
    
    uint32_t start_time = HAL_GetTick();
    uint32_t sums[4] = {0};
    uint16_t count = 0;
    
    while (HAL_GetTick() - start_time < 3000) {
        if (data_rdy_f) {
            int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
            for (int i = 0; i < 4; i++) {
                sums[i] += adc_buffer[last_idx + i];
            }
            count++;
            data_rdy_f = false;
        }
        HAL_Delay(1);
    }
    
    if (count > 0) {
        for (int i = 0; i < 4; i++) {
            baseline[i] = sums[i] / count;
            threshold[i] = baseline[i] + 300 + (i * 30);
        }
        
        printf("Calibration complete:\r\n");
        printf("  CH1 (FCR): baseline=%d, threshold=%d\r\n", baseline[0], threshold[0]);
        printf("  CH2 (BR):  baseline=%d, threshold=%d\r\n", baseline[1], threshold[1]);
        printf("  CH3 (FCU): baseline=%d, threshold=%d\r\n", baseline[2], threshold[2]);
        printf("  CH4 (FDS): baseline=%d, threshold=%d\r\n", baseline[3], threshold[3]);
    }
}

static void add_to_buffer(uint16_t ch0, uint16_t ch1, uint16_t ch2, uint16_t ch3) {
    emg_circular_buffer[buffer_write_idx][0] = ch0;
    emg_circular_buffer[buffer_write_idx][1] = ch1;
    emg_circular_buffer[buffer_write_idx][2] = ch2;
    emg_circular_buffer[buffer_write_idx][3] = ch3;
    
    buffer_write_idx++;
    if (buffer_write_idx >= EMG_BUFFER_SIZE) {
        buffer_write_idx = 0;
    }
    
    if (buffer_filled < EMG_BUFFER_SIZE) {
        buffer_filled++;
    }
}

static void get_window_data(uint16_t start_idx, uint16_t window_data[][4]) {
    for (int i = 0; i < EMG_WINDOW_SIZE; i++) {
        int idx = (start_idx + i) % EMG_BUFFER_SIZE;
        window_data[i][0] = emg_circular_buffer[idx][0];
        window_data[i][1] = emg_circular_buffer[idx][1];
        window_data[i][2] = emg_circular_buffer[idx][2];
        window_data[i][3] = emg_circular_buffer[idx][3];
    }
}

static uint8_t smooth_prediction(uint8_t new_prediction) {
    prediction_history[history_idx] = new_prediction;
    history_idx = (history_idx + 1) % PREDICTION_HISTORY;
    
    if (history_idx == 0) {
        history_filled = PREDICTION_HISTORY;
    } else if (!history_filled && history_idx > history_filled) {
        history_filled = history_idx;
    }
    
    int votes[STATE_COUNT] = {0};
    int max_votes = 0;
    uint8_t best_gesture = current_state;
    
    int history_len = history_filled ? PREDICTION_HISTORY : history_idx;
    for (int i = 0; i < history_len; i++) {
        uint8_t pred = prediction_history[i];
        if (pred < STATE_COUNT) {
            votes[pred]++;
            if (votes[pred] > max_votes) {
                max_votes = votes[pred];
                best_gesture = pred;
            }
        }
    }
    
    int required_votes = (PREDICTION_HISTORY / 2) + 1;
    if (max_votes >= required_votes) {
        return best_gesture;
    }
    return current_state;
}

static uint8_t detect_activity(void) {
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    for (int i = 0; i < 4; i++) {
        uint16_t raw = adc_buffer[last_idx + i];
        if (raw > baseline[i] + 100) {
            return 1;
        }
    }
    return 0;
}

static void execute_gesture(ProsthesisState_t state) {
    if (state >= STATE_COUNT) {
        state = STATE_REST;
    }
    
    const GestureAngles_t* angles = &gesture_angles[state];
    
    SetServo1Angle(angles->thumb_angle);
    SetServo2Angle(angles->index_angle);
    SetServo3Angle(angles->middle_angle);
    SetServo4Angle(angles->ring_angle);
    SetServo5Angle(angles->pinky_angle);
    
    if (debug_mode) {
        printf("SERVO: T=%d I=%d M=%d R=%d P=%d\r\n",
               angles->thumb_angle, angles->index_angle,
               angles->middle_angle, angles->ring_angle, angles->pinky_angle);
    }
}

void EMG_Control_Process(void) {
    static uint16_t process_counter = 0;
    static uint32_t debug_counter = 0;
    uint32_t now = HAL_GetTick();
    
    if (!data_rdy_f) {
        return;
    }
    
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    uint16_t ch0 = adc_buffer[last_idx + 0];
    uint16_t ch1 = adc_buffer[last_idx + 1];
    uint16_t ch2 = adc_buffer[last_idx + 2];
    uint16_t ch3 = adc_buffer[last_idx + 3];
    
    add_to_buffer(ch0, ch1, ch2, ch3);
    
    if (buffer_filled >= EMG_WINDOW_SIZE) {
        if (process_counter >= EMG_WINDOW_STEP) {
            process_counter = 0;
            
            uint16_t window_start = (buffer_write_idx + EMG_BUFFER_SIZE - EMG_WINDOW_SIZE) % EMG_BUFFER_SIZE;
            uint16_t window_data[EMG_WINDOW_SIZE][4];
            get_window_data(window_start, window_data);
            
            uint8_t prediction = ann_process_window(window_data, EMG_WINDOW_SIZE);
            float confidence = ann_get_confidence_from_buffer(window_data, EMG_WINDOW_SIZE);
            uint8_t smoothed = smooth_prediction(prediction);
            uint8_t is_active = detect_activity();
            
            if (is_active) {
                last_activity_time = now;
            }
            
            // State change with servo protection - minimum 5 seconds between changes
            if (smoothed != current_state && confidence >= MIN_CONFIDENCE) {
                // Check if enough time has passed since last gesture change
                if (now - last_gesture_time >= MIN_GESTURE_INTERVAL_MS) {
                    if (now - last_state_change_time >= STATE_DEBOUNCE_MS) {
                        printf("GESTURE: %s -> %s (conf=%.3f)\r\n", 
                               state_names[current_state], state_names[smoothed], confidence);
                        current_state = smoothed;
                        last_state_change_time = now;
                        last_gesture_time = now;
                        execute_gesture(current_state);
                    }
                } else {
                    // Print warning occasionally (every 2 seconds max)
                    static uint32_t last_warning = 0;
                    if (now - last_warning > 2000) {
                        printf("WARNING: Gesture change blocked - only %d ms since last change (min %d ms)\r\n",
                               (int)(now - last_gesture_time), MIN_GESTURE_INTERVAL_MS);
                        last_warning = now;
                    }
                }
            } else if (current_state != STATE_REST && !is_active && (now - last_activity_time > ACTIVITY_TIMEOUT_MS)) {
                // Return to rest - this is allowed anytime (doesn't count as gesture change for servo protection)
                printf("GESTURE: %s -> REST (timeout)\r\n", state_names[current_state]);
                current_state = STATE_REST;
                last_state_change_time = now;
                // Note: last_gesture_time NOT updated here - returning to rest is safe
                execute_gesture(STATE_REST);
            }
            
            debug_counter++;
            if (debug_counter % 20 == 0) {
                printf("DEBUG: ADC: %d,%d,%d,%d | Pred: %d (%s) | Conf: %.3f | Active: %d\r\n",
                       ch0, ch1, ch2, ch3, prediction, ann_get_class_name(prediction), confidence, is_active);
            }
        }
        process_counter++;
    }
    
    frame_count++;
    data_rdy_f = false;
}

void EMG_PrintRawData(void) {
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    printf("%d,%d,%d,%d\r\n",
           adc_buffer[last_idx + 0],
           adc_buffer[last_idx + 1],
           adc_buffer[last_idx + 2],
           adc_buffer[last_idx + 3]);
}

void EMG_SetDebugMode(uint8_t enable) {
    debug_mode = enable;
    printf("Debug mode: %s\r\n", enable ? "ON" : "OFF");
}

const char* EMG_GetStateName(ProsthesisState_t state) {
    if (state < STATE_COUNT) {
        return state_names[state];
    }
    return "UNKNOWN";
}