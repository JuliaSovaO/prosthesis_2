// data_collection.cpp - IMPLEMENTATION ONLY
#include "data_collection.h"
#include "periph_init.h"
#include <string.h>
#include <stdlib.h>

// Forward declarations (these are in main.cpp)
extern "C" {
    extern EMG_Sample sample_buffer[BUFFER_SAMPLES];
    extern volatile uint32_t sample_buffer_idx;
    extern const char* GESTURE_NAMES[NUM_GESTURES]; 
}

// Gesture names (defined in main.cpp)
extern const char* GESTURE_NAMES[NUM_GESTURES];

// Collection state
static CollectionMode collection_mode = MODE_IDLE;
static uint8_t current_gesture = 0;
static uint32_t trial_count = 0;
static uint32_t sample_count = 0;
static uint32_t start_timestamp = 0;
static uint32_t last_sample_time = 0;

void data_collection_init(void) {
    collection_mode = MODE_IDLE;
    current_gesture = 0;
    trial_count = 0;
    sample_count = 0;
    
    // Clear sample buffer
    memset(sample_buffer, 0, sizeof(sample_buffer));
    sample_buffer_idx = 0;
}

void data_collection_start(uint8_t gesture_id) {
    if (collection_mode != MODE_IDLE) {
        printf("Error: Collection already in progress!\n");
        return;
    }
    
    current_gesture = gesture_id;
    collection_mode = MODE_COLLECTING;
    sample_count = 0;
    start_timestamp = HAL_GetTick();
    last_sample_time = 0;
    
    printf("=== Starting collection for %s ===\n", GESTURE_NAMES[gesture_id]);
    printf("Duration: %d ms\n", COLLECTION_DURATION_MS);
    printf("Expected samples: %d\n", SAMPLES_PER_TRIAL);
    printf("Sampling rate: %d Hz\n", SAMPLING_RATE_HZ);
    
    // Start timer for ADC triggering
    HAL_TIM_Base_Start(&htim3);
    
    // Start ADC with DMA - use a simple buffer
    static uint16_t dma_buffer[ADC_CHANNELS * 100];
    if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)dma_buffer, ADC_CHANNELS * 100) != HAL_OK) {
        printf("ADC start failed!\n");
        collection_mode = MODE_IDLE;
        return;
    }
    
    // Start ADC conversions
    HAL_ADC_Start_IT(&hadc1);
}

void data_collection_stop(void) {
    if (collection_mode != MODE_COLLECTING) {
        return;
    }
    
    // Stop ADC and timer
    HAL_ADC_Stop_DMA(&hadc1);
    HAL_TIM_Base_Stop(&htim3);
    
    collection_mode = MODE_PROCESSING;
    printf("Collection stopped. Collected %lu samples.\n", sample_count);
    
    // Process and export data
    export_data_to_serial(current_gesture);
    
    trial_count++;
    collection_mode = MODE_IDLE;
}

void export_data_to_serial(uint8_t gesture_id) {
    printf("\n=== Exporting data for %s ===\n", GESTURE_NAMES[gesture_id]);
    printf("Trial: %lu/%d\n", trial_count + 1, TRIALS_PER_GESTURE);
    printf("Total samples: %lu\n", sample_count);
    
    // Export header
    printf("CSV_START,gesture_%d_%s_trial_%lu.csv\n",
           gesture_id, GESTURE_NAMES[gesture_id], trial_count + 1);
    printf("timestamp_us,ch1,ch2,ch3,gesture_id\n");
    
    // Export all collected samples
    uint32_t start_idx = (sample_buffer_idx > sample_count) ? 
                         (sample_buffer_idx - sample_count) % BUFFER_SAMPLES : 0;
    
    for (uint32_t i = 0; i < sample_count; i++) {
        uint32_t idx = (start_idx + i) % BUFFER_SAMPLES;
        EMG_Sample* sample = &sample_buffer[idx];
        
        printf("%lu,%u,%u,%u,%d\n",
               sample->timestamp,
               sample->ch1,
               sample->ch2,
               sample->ch3,
               sample->gesture_id);
        
        if (i % 100 == 0) {
            HAL_Delay(1);
        }
    }
    
    printf("CSV_END,gesture_%d_%s_trial_%lu.csv\n",
           gesture_id, GESTURE_NAMES[gesture_id], trial_count + 1);
    printf("Export complete.\n\n");
}

void data_collection_process(void) {
    if (collection_mode == MODE_COLLECTING) {
        uint32_t elapsed = HAL_GetTick() - start_timestamp;
        
        if (elapsed > COLLECTION_DURATION_MS + 1000) {
            printf("Timeout! Stopping collection.\n");
            data_collection_stop();
        }
    }
}

void data_collection_main(void) {
    printf("=== High-Speed EMG Data Collection ===\n");
    printf("Sampling rate: %d Hz\n", SAMPLING_RATE_HZ);
    printf("Duration per trial: %d ms\n", COLLECTION_DURATION_MS);
    printf("Trials per gesture: %d\n", TRIALS_PER_GESTURE);
    printf("Expected samples per trial: %d\n", SAMPLES_PER_TRIAL);
    printf("\nConnect electrodes to:\n");
    printf("  CH1: Flexor carpi radialis (PA0)\n");
    printf("  CH2: Flexor carpi ulnaris (PA1)\n");
    printf("  CH3: Brachioradialis (PA2)\n");
    printf("\nPress USER button (PA0) to start...\n");
    
    // Wait for button press
    while (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0) == GPIO_PIN_SET) {
        HAL_Delay(100);
    }
    
    // Initialize data collection
    data_collection_init();
    
    // Collect all gestures
    for (uint8_t gesture = 0; gesture < NUM_GESTURES; gesture++) {
        printf("\n\n=== Gesture %d/%d: %s ===\n", 
               gesture + 1, NUM_GESTURES, GESTURE_NAMES[gesture]);
        
        // Collect multiple trials
        for (uint32_t trial = 0; trial < TRIALS_PER_GESTURE; trial++) {
            printf("\nTrial %lu/%d\n", trial + 1, TRIALS_PER_GESTURE);
            printf("Ready... (3 seconds)\n");
            HAL_Delay(3000);
            
            printf("START!\n");
            data_collection_start(gesture);
            
            // Wait for collection to complete
            while (collection_mode == MODE_COLLECTING) {
                data_collection_process();
                HAL_Delay(10);
            }
            
            // Rest between trials
            if (trial < TRIALS_PER_GESTURE - 1) {
                printf("Rest %d seconds...\n", REST_BETWEEN_TRIALS_MS / 1000);
                HAL_Delay(REST_BETWEEN_TRIALS_MS);
            }
        }
        
        // Rest between gestures
        if (gesture < NUM_GESTURES - 1) {
            printf("\nRest %d minutes before next gesture...\n", 
                   REST_BETWEEN_GESTURES_MS / 60000);
            HAL_Delay(REST_BETWEEN_GESTURES_MS);
        }
    }
    
    printf("\n=== Data Collection Complete ===\n");
    printf("Total gestures: %d\n", NUM_GESTURES);
    printf("Total trials: %d\n", NUM_GESTURES * TRIALS_PER_GESTURE);
    printf("Total samples: ~%lu\n",
           (uint32_t)NUM_GESTURES * TRIALS_PER_GESTURE * SAMPLES_PER_TRIAL);
}

uint32_t get_available_samples(void) {
    return sample_count;
}

uint32_t get_sample_rate(void) {
    if (sample_count == 0 || (HAL_GetTick() - start_timestamp) == 0) {
        return 0;
    }
    return (sample_count * 1000) / (HAL_GetTick() - start_timestamp);
}