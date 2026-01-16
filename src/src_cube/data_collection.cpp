#include "data_collection.h"
#include <string.h>
#include <stdlib.h>

// Make adc_buffer and hadc1 accessible (add extern declarations)
extern ADC_HandleTypeDef hadc1;
extern volatile bool data_rdy_f;
extern __attribute__((aligned(4))) uint16_t adc_buffer[3 * 1];  // ADC_CHANNELS * SAMPLES

#define NUM_GESTURES 10
#define SAMPLES_PER_GESTURE 1000
#define TOTAL_TRIALS 20

const char* gesture_names[NUM_GESTURES] = {
    "REST", "ROCK", "SCISSORS", "PAPER", "ONE",
    "THREE", "FOUR", "GOOD", "OKAY", "FINGER_GUN"
};

void collect_gesture_data(int gesture_id) {
    printf("=== Collecting %s ===\n", gesture_names[gesture_id]);
    printf("Get ready...\n");
    HAL_Delay(3000);
    
    char filename[50];
    sprintf(filename, "gesture_%d_%s.csv", gesture_id, gesture_names[gesture_id]);
    
    // In embedded systems, we usually write to serial, not files
    printf("STARTING COLLECTION...\n");
    printf("CSV_START,%s\n", filename);
    
    uint32_t start_time = HAL_GetTick();
    uint32_t sample_count = 0;
    
    printf("START (5 seconds)...\n");
    
    while (sample_count < SAMPLES_PER_GESTURE) {
        if (data_rdy_f) {
            uint16_t ch1 = adc_buffer[0];
            uint16_t ch2 = adc_buffer[1];
            uint16_t ch3 = adc_buffer[2];
            uint32_t timestamp = HAL_GetTick() - start_time;
            
            // Output CSV line via serial
            printf("DATA,%lu,%u,%u,%u,%d\n", 
                   timestamp, ch1, ch2, ch3, gesture_id);
            
            sample_count++;
            data_rdy_f = false;
        }
        HAL_Delay(3);  // ~332Hz sampling
    }
    
    printf("CSV_END,%s\n", filename);
    printf("Saved %lu samples\n\n", sample_count);
}

void data_collection_main(void) {
    printf("=== EMG Data Collection ===\n");
    printf("Connect electrodes to:\n");
    printf("  CH1: Flexor carpi radialis\n");
    printf("  CH2: Flexor carpi ulnaris\n");
    printf("  CH3: Brachioradialis\n\n");
    
    // Start ADC
    if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_buffer, 3 * 1) != HAL_OK) {
        printf("ADC start failed!\n");
        return;
    }
    
    // Collect each gesture multiple times
    for (int gesture = 0; gesture < NUM_GESTURES; gesture++) {
        printf("Gesture %d/%d: %s\n", gesture+1, NUM_GESTURES, gesture_names[gesture]);
        printf("Press button to start collection...\n");
        
        // Wait for button press (PA0 - adjust based on your hardware)
        while (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0) == GPIO_PIN_SET) {
            HAL_Delay(100);
        }
        
        for (int trial = 0; trial < TOTAL_TRIALS; trial++) {
            printf("Trial %d/%d\n", trial+1, TOTAL_TRIALS);
            collect_gesture_data(gesture);
            
            if (trial < TOTAL_TRIALS - 1) {
                printf("Rest 10 seconds...\n");
                HAL_Delay(10000);
            }
        }
        
        if (gesture < NUM_GESTURES - 1) {
            printf("Rest 5 minutes before next gesture...\n");
            HAL_Delay(300000);  // 5 minutes
        }
    }
    
    printf("=== Data Collection Complete ===\n");
    
    // Stop ADC
    HAL_ADC_Stop_DMA(&hadc1);
}