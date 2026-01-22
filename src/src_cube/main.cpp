#include "main.h"
#include "periph_init.h"
#include "emg_classifier.h"
#include "signal_validation.h"
#include "gesture.h"
#include <string.h>
#include <stdio.h> 

EMG_Buffer emg_buffer;

GestureType current_gesture = GESTURE_REST;
GestureType last_executed_gesture = GESTURE_REST;

// Feature buffer
float extracted_features[TOTAL_FEATURES];

#define ADC_CHANNELS 4
#define BUF_SIZE 8 
__attribute__((aligned(4))) uint16_t adc_buffer[BUF_SIZE] = {0};

extern ADC_HandleTypeDef hadc1;
extern DMA_HandleTypeDef hdma_adc1;
extern UART_HandleTypeDef huart1;
extern TIM_HandleTypeDef htim3;

// Clean output: sensors + gesture only
void output_sensors_and_gesture(uint16_t ch1, uint16_t ch2, uint16_t ch3, uint16_t ch4, GestureType gesture) {
    // Format: "S1,S2,S3,S4:G" (e.g., "1234,567,890,123:9")
    static char buf[30];
    int len = snprintf(buf, sizeof(buf), "%04d,%04d,%04d,%04d:%d\n", 
                      ch1, ch2, ch3, ch4, gesture);
    HAL_UART_Transmit(&huart1, (uint8_t*)buf, len, 10);
}

// Only output when gesture changes
void output_gesture_change(GestureType old_gesture, GestureType new_gesture) {
    if (old_gesture != new_gesture) {
        char buf[30];
        int len = snprintf(buf, sizeof(buf), "Gesture: %d -> %d\n", old_gesture, new_gesture);
        HAL_UART_Transmit(&huart1, (uint8_t*)buf, len, 10);
    }
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    MX_TIM3_Init();
    MX_I2C1_Init();
    
    emg_buffer_init(&emg_buffer);
    InitAllServos();

    // Startup message only
    const char* startup_msg = "EMG System Ready\n";
    HAL_UART_Transmit(&huart1, (uint8_t*)startup_msg, strlen(startup_msg), 100);
    
    // Header for data stream
    const char* header = "S1,S2,S3,S4:GESTURE\n";
    HAL_UART_Transmit(&huart1, (uint8_t*)header, strlen(header), 100);

    HAL_TIM_Base_Start(&htim3);
    HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buffer, BUF_SIZE);
    
    uint32_t last_dma_pos = 0;
    uint32_t last_classification_time = HAL_GetTick();
    uint32_t last_output_time = HAL_GetTick();
    
    // Gesture history for consistency (reduced from 10 to 5 for faster response)
    GestureType gesture_history[5] = {GESTURE_REST};
    uint8_t history_index = 0;
    uint8_t history_count = 0;
    
    // For output throttling
    uint16_t output_ch1 = 0, output_ch2 = 0, output_ch3 = 0, output_ch4 = 0;
    
    while (1) {
        uint32_t dma_pos = BUF_SIZE - hdma_adc1.Instance->NDTR;
        
        if (dma_pos != last_dma_pos) {
            uint32_t sample_idx = (dma_pos - ADC_CHANNELS + BUF_SIZE) % BUF_SIZE;
            
            if (sample_idx + 3 < BUF_SIZE) {
                // Store sensor readings for output
                output_ch1 = adc_buffer[sample_idx];
                output_ch2 = adc_buffer[sample_idx + 1];
                output_ch3 = adc_buffer[sample_idx + 2];
                output_ch4 = adc_buffer[sample_idx + 3];
                
                // Add sample to EMG buffer
                emg_buffer_add_sample(&emg_buffer, output_ch1, output_ch2, output_ch3, output_ch4);
                
                // Process classification every 50ms (20Hz)
                uint32_t current_time = HAL_GetTick();
                if (current_time - last_classification_time >= 50) {
                    last_classification_time = current_time;
                    
                    // Try to process a window
                    if (emg_buffer_process_window(&emg_buffer, extracted_features)) {
                        
                        // Validate signal
                        if (are_features_valid(extracted_features)) {
                            // Valid signal - classify
                            GestureType new_gesture = classify_gesture(extracted_features);
                            
                            // Update gesture history
                            gesture_history[history_index] = new_gesture;
                            history_index = (history_index + 1) % 5;
                            if (history_count < 5) history_count++;
                            
                            // Check for consistent gesture (3 out of 5)
                            if (history_count >= 3) {
                                int count[10] = {0};
                                for (int i = 0; i < history_count; i++) {
                                    count[gesture_history[i]]++;
                                }
                                
                                // Find most frequent gesture
                                GestureType most_frequent = GESTURE_REST;
                                int max_count = 0;
                                for (int i = 0; i < 10; i++) {
                                    if (count[i] > max_count) {
                                        max_count = count[i];
                                        most_frequent = (GestureType)i;
                                    }
                                }
                                
                                // If gesture is consistent (at least 3 of last 5)
                                if (max_count >= 3 && most_frequent != current_gesture) {
                                    // Output gesture change
                                    output_gesture_change(current_gesture, most_frequent);
                                    
                                    // Update and execute
                                    current_gesture = most_frequent;
                                    if (current_gesture != last_executed_gesture) {
                                        last_executed_gesture = current_gesture;
                                        execute_gesture(last_executed_gesture);
                                    }
                                }
                            }
                        } else {
                            // Invalid signal - reset to REST
                            if (current_gesture != GESTURE_REST) {
                                output_gesture_change(current_gesture, GESTURE_REST);
                                current_gesture = GESTURE_REST;
                                last_executed_gesture = GESTURE_REST;
                                execute_gesture(GESTURE_REST);
                            }
                        }
                    }
                }
                
                // Output sensor data at 10Hz (every 100ms)
                if (current_time - last_output_time >= 100) {
                    last_output_time = current_time;
                    output_sensors_and_gesture(output_ch1, output_ch2, output_ch3, output_ch4, current_gesture);
                }
            }
            
            last_dma_pos = dma_pos;
        }
        
        // Minimal LED blink (once per second)
        static uint32_t led_timer = 0;
        if (HAL_GetTick() - led_timer > 1000) {
            led_timer = HAL_GetTick();
            HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        }
    }
}

void Error_Handler(void) {
    while (1) {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        HAL_Delay(100);
    }
}

extern "C" {
    int _write(int file, char *ptr, int len) {
        // Disable stdio output to reduce noise
        // HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, 1);
        return len;
    }
    
    void DMA2_Stream0_IRQHandler(void) { 
        HAL_DMA_IRQHandler(&hdma_adc1); 
    }
    
    void TIM3_IRQHandler(void) { 
        HAL_TIM_IRQHandler(&htim3); 
    }
}