#include "main.h"
#include "emg_control.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static uint16_t emg_circular_buffer[EMG_BUFFER_SIZE][4];
static uint16_t buffer_write_idx = 0;
static uint16_t buffer_filled = 0;

static uint8_t prediction_history[PREDICTION_HISTORY];
static uint8_t history_idx = 0;
static uint8_t history_filled = 0;

static ProsthesisState_t current_state = STATE_REST;
static uint32_t last_state_change_time = 0;
static uint32_t last_activity_time = 0;
static uint32_t last_gesture_time = 0;

static uint8_t debug_mode = 1;
static uint32_t frame_count = 0;

static uint16_t baseline[4] = {0};
static uint16_t threshold[4] = {0};

static const GestureAngles_t gesture_angles[] = {
    // PHYS_ROCK (0)
    {180, 180, 130, 180, 180},
    // PHYS_PAPER (1)
    {0, 10, 10, 80, 10},
    // PHYS_OKAY (2)
    {180, 180, 10, 80, 10},
    // PHYS_REST (3)
    {30, 40, 40, 80, 40},
    // PHYS_FUCK (4)
    {180, 180, 10, 180, 180},
};

static const char* physical_names[] = {
    "ROCK", "PAPER", "OKAY", "REST", "FUCK"
};

static uint8_t map_to_physical(uint8_t ann_class) {
    switch (ann_class) {
        case 0:  return PHYS_REST;   // finger-gun -> rest
        case 1:  return PHYS_PAPER;  // four -> paper
        case 2:  return PHYS_FUCK;   // fuck -> fuck
        case 3:  return PHYS_REST;   // good -> rest
        case 4:  return PHYS_OKAY;   // okay -> okay
        case 5:  return PHYS_PAPER;  // paper -> paper
        case 6:  return PHYS_REST;   // rest -> rest
        case 7:  return PHYS_ROCK;   // rock -> rock
        case 8:  return PHYS_FUCK;   // scissors -> fuck
        case 9:  return PHYS_ROCK;   // three -> rock
        default: return PHYS_REST;
    }
}

void EMG_Control_Init(void) {
    memset(emg_circular_buffer, 0, sizeof(emg_circular_buffer));
    buffer_write_idx = 0;
    buffer_filled = 0;
    
    for (int i = 0; i < PREDICTION_HISTORY; i++) {
        prediction_history[i] = PHYS_REST;
    }
    history_idx = 0;
    history_filled = 0;
    
    current_state = STATE_REST;
    last_state_change_time = HAL_GetTick();
    last_activity_time = HAL_GetTick();
    last_gesture_time = HAL_GetTick();
    
    ann_init();
    
    const GestureAngles_t* idle_angles = &gesture_angles[PHYS_REST];
    SetServo1Angle(idle_angles->thumb_angle);
    SetServo2Angle(idle_angles->index_angle);
    SetServo3Angle(idle_angles->middle_angle);
    SetServo4Angle(idle_angles->ring_angle);
    SetServo5Angle(idle_angles->pinky_angle);
}

void EMG_AutoCalibrate(void) {
    printf("\r\n=== AUTO-CALIBRATION ===\r\n");
    printf("Relax for 3 seconds...\r\n");
    
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
        printf("Baselines: CH1=%d CH2=%d CH3=%d CH4=%d\r\n",
               baseline[0], baseline[1], baseline[2], baseline[3]);
    }
}

static void add_to_buffer(uint16_t ch0, uint16_t ch1, uint16_t ch2, uint16_t ch3) {
    emg_circular_buffer[buffer_write_idx][0] = ch0;
    emg_circular_buffer[buffer_write_idx][1] = ch1;
    emg_circular_buffer[buffer_write_idx][2] = ch2;
    emg_circular_buffer[buffer_write_idx][3] = ch3;
    buffer_write_idx = (buffer_write_idx + 1) % EMG_BUFFER_SIZE;
    if (buffer_filled < EMG_BUFFER_SIZE) buffer_filled++;
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
    if (history_filled < PREDICTION_HISTORY) history_filled++;
    
    int votes[PHYS_COUNT] = {0};
    int max_votes = 0;
    uint8_t best = current_state;
    
    for (int i = 0; i < history_filled; i++) {
        uint8_t p = prediction_history[i];
        if (p < PHYS_COUNT) {
            votes[p]++;
            if (votes[p] > max_votes) {
                max_votes = votes[p];
                best = p;
            }
        }
    }
    
    return (max_votes >= (PREDICTION_HISTORY / 2) + 1) ? best : current_state;
}

static uint8_t detect_activity(void) {
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    for (int i = 0; i < 4; i++) {
        if (adc_buffer[last_idx + i] > baseline[i] + 100) return 1;
    }
    return 0;
}

static void execute_gesture(uint8_t state) {
    if (state >= PHYS_COUNT) state = PHYS_REST;
    const GestureAngles_t* a = &gesture_angles[state];
    SetServo1Angle(a->thumb_angle);
    SetServo2Angle(a->index_angle);
    SetServo3Angle(a->middle_angle);
    SetServo4Angle(a->ring_angle);
    SetServo5Angle(a->pinky_angle);
}

void EMG_Control_Process(void) {
    static uint16_t process_counter = 0;
    static uint32_t debug_counter = 0;
    uint32_t now = HAL_GetTick();
    
    if (!data_rdy_f) return;
    
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    uint16_t ch0 = adc_buffer[last_idx + 0];
    uint16_t ch1 = adc_buffer[last_idx + 1];
    uint16_t ch2 = adc_buffer[last_idx + 2];
    uint16_t ch3 = adc_buffer[last_idx + 3];
    
    add_to_buffer(ch0, ch1, ch2, ch3);
    
    if (buffer_filled >= EMG_WINDOW_SIZE) {
        if (process_counter >= EMG_WINDOW_STEP) {
            process_counter = 0;
            
            uint16_t ws = (buffer_write_idx + EMG_BUFFER_SIZE - EMG_WINDOW_SIZE) % EMG_BUFFER_SIZE;
            uint16_t wd[EMG_WINDOW_SIZE][4];
            get_window_data(ws, wd);
            
            uint8_t ann_pred = ann_process_window(wd, EMG_WINDOW_SIZE);
            float confidence = ann_get_confidence_from_buffer(wd, EMG_WINDOW_SIZE);
            
            // map 10-class ann to 5 physical gestures
            uint8_t physical_pred = map_to_physical(ann_pred);
            uint8_t smoothed = smooth_prediction(physical_pred);
            uint8_t is_active = detect_activity();
            
            if (is_active) last_activity_time = now;
            
            if (smoothed != current_state && confidence >= MIN_CONFIDENCE) {
                if (now - last_gesture_time >= MIN_GESTURE_INTERVAL_MS) {
                    if (now - last_state_change_time >= STATE_DEBOUNCE_MS) {
                        printf("GESTURE: %s -> %s (ANN:%s, conf=%.2f)\r\n",
                               physical_names[current_state], physical_names[smoothed],
                               ann_get_class_name(ann_pred), confidence);
                        current_state = smoothed;
                        last_state_change_time = now;
                        last_gesture_time = now;
                        execute_gesture(current_state);
                    }
                }
            } else if (current_state != PHYS_REST && !is_active && 
                       (now - last_activity_time > ACTIVITY_TIMEOUT_MS)) {
                printf("GESTURE: %s -> REST (timeout)\r\n", physical_names[current_state]);
                current_state = PHYS_REST;
                last_state_change_time = now;
                execute_gesture(PHYS_REST);
            }
            
            debug_counter++;
            if (debug_counter % 10 == 0) {
                printf("ANN:%d(%s) -> PHYS:%d(%s) conf=%.2f\r\n",
                       ann_pred, ann_get_class_name(ann_pred),
                       physical_pred, physical_names[physical_pred], confidence);
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
           adc_buffer[last_idx + 0], adc_buffer[last_idx + 1],
           adc_buffer[last_idx + 2], adc_buffer[last_idx + 3]);
}

void EMG_SetDebugMode(uint8_t enable) {
    debug_mode = enable;
}

const char* EMG_GetStateName(ProsthesisState_t state) {
    if (state < PHYS_COUNT) return physical_names[state];
    return "UNKNOWN";
}