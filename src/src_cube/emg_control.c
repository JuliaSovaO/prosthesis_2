#include "emg_control.h"
#include "main.h"
#include <stdio.h>
#include <string.h>

extern ADC_HandleTypeDef hadc2;

typedef struct {
    uint16_t buf[EMG_WINDOW_SIZE];
    uint32_t sum;
    uint16_t filtered;
    uint16_t index;
    uint8_t full;
    uint16_t baseline;
    uint16_t threshold;
    uint8_t activated;
    uint8_t activation_count;  
    uint8_t deactivation_count;
    uint8_t channel_id;
} EMG_Filter_t;

// 4-channel EMG filters
EMG_Filter_t channels[4];

// Servo states (5 fingers)
ServoState_t servo_states[5] = {
    {0, 0, 0, 150},      // thumb
    {0, 0, 0, 180},      // index
    {10, 10, 10, 170},   // middle
    {20, 20, 20, 180},   // ring
    {0, 0, 0, 120}       // pinky
};

static uint8_t current_state = STATE_IDLE;
static uint32_t state_change_time = 0;
static uint32_t last_emg_activity = 0;
static uint8_t debug_mode = 0;  // 0=normal, 1=raw data output
static uint32_t last_raw_print = 0;
static uint32_t frame_count = 0;

// Filter update with moving average
static void update_filter(EMG_Filter_t *filter, uint16_t raw_value) {
    uint16_t i = filter->index;
    uint16_t old_value = filter->buf[i];
    filter->buf[i] = raw_value;
    
    if (filter->full) {
        filter->sum += raw_value - old_value;
    } else {
        filter->sum += raw_value;
    }
    
    uint16_t count = filter->full ? EMG_WINDOW_SIZE : (filter->index + 1);
    filter->filtered = filter->sum / count;
    
    filter->index = (filter->index + 1) % EMG_WINDOW_SIZE;
    if (filter->index == 0) filter->full = 1;
}

// Update baseline during inactivity
static void update_baseline(EMG_Filter_t *filter) {
    static uint32_t last_update[4] = {0};
    
    if (!filter->activated && (HAL_GetTick() - last_update[filter->channel_id] > 3000)) {
        filter->baseline = (filter->baseline * 15 + filter->filtered) / 16;
        last_update[filter->channel_id] = HAL_GetTick();
    }
}

// Detect muscle activation with hysteresis
static void detect_activation(EMG_Filter_t *filter) {
    int16_t signal_above = (int16_t)filter->filtered - (int16_t)filter->baseline;
    
    if (signal_above > filter->threshold + EMG_HYSTERESIS) {
        filter->activation_count++;
        filter->deactivation_count = 0;
        
        if (filter->activation_count >= 3) {
            filter->activated = 1;
        }
    } 
    else if (signal_above < filter->threshold - EMG_HYSTERESIS) {
        filter->deactivation_count++;
        filter->activation_count = 0;
        
        if (filter->deactivation_count >= 5) {
            filter->activated = 0;
        }
    } else {
        filter->activation_count = 0;
        filter->deactivation_count = 0;
    }
}

// Determine which channels are active
static uint8_t get_active_pattern(void) {
    uint8_t pattern = 0;
    for (int i = 0; i < 4; i++) {
        if (channels[i].activated) {
            pattern |= (1 << i);
        }
    }
    return pattern;
}

// Map activation pattern to hand state
static uint8_t pattern_to_state(uint8_t pattern) {
    switch(pattern) {
        case 0x00: return STATE_IDLE;      // No activation
        case 0x01:                         // CH1 only - close all
        case 0x02:                         // CH2 only - close all (alternate)
            return STATE_CLOSE;
        case 0x04: return STATE_OPEN;       // CH3 only - open all
        case 0x08: return STATE_THUMB;      // CH4 only - thumb
        case 0x03: return STATE_PINCH;      // CH1+CH2 - pinch grip
        case 0x09: return STATE_INDEX;      // CH1+CH4 - index
        case 0x0A: return STATE_MIDDLE;     // CH2+CH4 - middle
        default: return STATE_IDLE;
    }
}

void EMG_Control_Init(void) {
    // Initialize all 4 channels
    for (int i = 0; i < 4; i++) {
        channels[i].channel_id = i;
        channels[i].threshold = EMG_THRESHOLD_BASE + (i * 50);
        channels[i].baseline = 500 + (i * 30);
        
        memset(channels[i].buf, 0, sizeof(channels[i].buf));
        channels[i].sum = 0;
        channels[i].filtered = 0;
        channels[i].index = 0;
        channels[i].full = 0;
        channels[i].activated = 0;
        channels[i].activation_count = 0;
        channels[i].deactivation_count = 0;
    }
    
    current_state = STATE_IDLE;
    state_change_time = HAL_GetTick();
    last_emg_activity = HAL_GetTick();
    debug_mode = 0;
    
    printf("\r\n=== 4-CHANNEL EMG CONTROL ===\r\n");
    printf("Thresholds: CH1=%d, CH2=%d, CH3=%d, CH4=%d\r\n", 
           channels[0].threshold, channels[1].threshold, 
           channels[2].threshold, channels[3].threshold);
    printf("Hysteresis: %d, Debounce: %dms\r\n", EMG_HYSTERESIS, STATE_DEBOUNCE_MS);
    printf("Output format (debug ON): CH1,CH2,CH3,CH4\r\n");
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
        for (int i = 0; i < 4; i++) {
            channels[i].baseline = sums[i] / count;
            channels[i].threshold = channels[i].baseline + 300 + (i * 30);
        }
        
        printf("Calibrated: ");
        printf("CH1=%d(+%d), CH2=%d(+%d), ", 
               channels[0].baseline, channels[0].threshold - channels[0].baseline,
               channels[1].baseline, channels[1].threshold - channels[1].baseline);
        printf("CH3=%d(+%d), CH4=%d(+%d)\r\n",
               channels[2].baseline, channels[2].threshold - channels[2].baseline,
               channels[3].baseline, channels[3].threshold - channels[3].baseline);
    }
}

void EMG_PrintRawData(void) {
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    
    printf("%d,%d,%d,%d\r\n",
           adc_buffer[last_idx + 0],
           adc_buffer[last_idx + 1],
           adc_buffer[last_idx + 2],
           adc_buffer[last_idx + 3]);
}

void EMG_Control_Process(void) {
    static uint32_t last_process = 0;
    uint32_t now = HAL_GetTick();
    
    if (!data_rdy_f) {
        return;
    }
    
    int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
    
    // update all 4 channels
    for (int i = 0; i < 4; i++) {
        update_filter(&channels[i], adc_buffer[last_idx + i]);
        update_baseline(&channels[i]);
        detect_activation(&channels[i]);
    }
    
    frame_count++;
    
    // state machine - update at ~1kHz rate
    if (now - last_process >= 1) {  // 1ms = 1000Hz
        last_process = now;
        
        uint8_t active_pattern = get_active_pattern();
        uint8_t new_state = pattern_to_state(active_pattern);
        
        // check for EMG activity timeout
        uint8_t any_activity = 0;
        for (int i = 0; i < 4; i++) {
            if (channels[i].activated) {
                any_activity = 1;
                last_emg_activity = now;
            }
        }
        
        if (!any_activity && (now - last_emg_activity > 2000)) {
            new_state = STATE_IDLE;
        }
        
        // state change with debounce
        if (new_state != current_state) {
            if (now - state_change_time > STATE_DEBOUNCE_MS) {
                printf("STATE: ");
                switch(new_state) {
                    case STATE_CLOSE: printf("CLOSE ALL"); break;
                    case STATE_OPEN: printf("OPEN ALL"); break;
                    case STATE_THUMB: printf("THUMB"); break;
                    case STATE_INDEX: printf("INDEX"); break;
                    case STATE_MIDDLE: printf("MIDDLE"); break;
                    case STATE_RING: printf("RING"); break;
                    case STATE_PINKY: printf("PINKY"); break;
                    case STATE_PINCH: printf("PINCH"); break;
                    default: printf("IDLE"); break;
                }
                printf(" (Pattern: 0x%X)\r\n", active_pattern);
                
                current_state = new_state;
                state_change_time = now;
            }
        }
        
        // apply state to servos
        switch(current_state) {
            case STATE_CLOSE:
                // all fingers close
                for (int i = 0; i < 5; i++) {
                    servo_states[i].target_angle = servo_states[i].max_angle;
                }
                break;
                
            case STATE_OPEN:
                // all fingers open
                for (int i = 0; i < 5; i++) {
                    servo_states[i].target_angle = servo_states[i].min_angle;
                }
                break;
                
            case STATE_THUMB:
                // thumb only
                servo_states[0].target_angle = 90;
                for (int i = 1; i < 5; i++) {
                    servo_states[i].target_angle = servo_states[i].min_angle;
                }
                break;
                
            case STATE_INDEX:
                // tndex only
                servo_states[1].target_angle = 90;
                servo_states[0].target_angle = servo_states[0].min_angle;
                for (int i = 2; i < 5; i++) {
                    servo_states[i].target_angle = servo_states[i].min_angle;
                }
                break;
                
            case STATE_PINCH:
                // thumb + index pinch
                servo_states[0].target_angle = 90;
                servo_states[1].target_angle = 90;
                for (int i = 2; i < 5; i++) {
                    servo_states[i].target_angle = servo_states[i].min_angle;
                }
                break;
                
            case STATE_IDLE:
            default:
                // slowly return to open position
                for (int i = 0; i < 5; i++) {
                    if (servo_states[i].target_angle > servo_states[i].min_angle) {
                        servo_states[i].target_angle -= 2;
                    }
                    if (servo_states[i].target_angle < servo_states[i].min_angle) {
                        servo_states[i].target_angle = servo_states[i].min_angle;
                    }
                }
                break;
        }
        
        // apply servo commands
        for (int i = 0; i < 5; i++) {
            // smooth movement
            if (servo_states[i].current_angle < servo_states[i].target_angle) {
                servo_states[i].current_angle += 2;
                if (servo_states[i].current_angle > servo_states[i].target_angle) {
                    servo_states[i].current_angle = servo_states[i].target_angle;
                }
            } else if (servo_states[i].current_angle > servo_states[i].target_angle) {
                servo_states[i].current_angle -= 2;
                if (servo_states[i].current_angle < servo_states[i].target_angle) {
                    servo_states[i].current_angle = servo_states[i].target_angle;
                }
            }
        }
        
        SetServo1Angle(servo_states[0].current_angle);
        SetServo2Angle(servo_states[1].current_angle);
        SetServo3Angle(servo_states[2].current_angle);
        SetServo4Angle(servo_states[3].current_angle);
        SetServo5Angle(servo_states[4].current_angle);
    }
    
    static uint32_t last_print = 0;
    if (now - last_print >= 1) {  // 1ms = 1000Hz max
        int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
        
        printf("%d,%d,%d,%d\r\n",
            adc_buffer[last_idx + 0],
            adc_buffer[last_idx + 1],
            adc_buffer[last_idx + 2],
            adc_buffer[last_idx + 3]);
        
        last_print = now;
    }
    
    data_rdy_f = false;
}