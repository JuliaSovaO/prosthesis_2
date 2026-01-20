#include "main.h"
#include "periph_init.h"
#include "emg_classifier.h"
#include "gesture.h"
#include <string.h>

EMG_Buffer emg_buffer;

GestureType current_gesture = GESTURE_REST;
GestureType last_gesture = GESTURE_REST;
uint32_t gesture_hold_time = 0;
#define GESTURE_HOLD_THRESHOLD 10  // Number of consistent classifications needed

// Feature buffer
float extracted_features[TOTAL_FEATURES];

#define ADC_CHANNELS 4
#define BUF_SIZE 8 
__attribute__((aligned(4))) uint16_t adc_buffer[BUF_SIZE] = {0};

extern ADC_HandleTypeDef hadc1;
extern DMA_HandleTypeDef hdma_adc1;
extern UART_HandleTypeDef huart1;
extern TIM_HandleTypeDef htim3;

void fast_output(uint16_t a, uint16_t b, uint16_t c, uint16_t d, GestureType gesture) {
    static uint8_t buf[] = "0000,0000,0000,0000,00\n";  // 4 channels + 2-digit gesture
    
    buf[0] = '0' + (a / 1000);
    buf[1] = '0' + ((a / 100) % 10);
    buf[2] = '0' + ((a / 10) % 10);
    buf[3] = '0' + (a % 10);
    
    buf[5] = '0' + (b / 1000);
    buf[6] = '0' + ((b / 100) % 10);
    buf[7] = '0' + ((b / 10) % 10);
    buf[8] = '0' + (b % 10);
    
    buf[10] = '0' + (c / 1000);
    buf[11] = '0' + ((c / 100) % 10);
    buf[12] = '0' + ((c / 10) % 10);
    buf[13] = '0' + (c % 10);
    
    buf[15] = '0' + (d / 1000);
    buf[16] = '0' + ((d / 100) % 10);
    buf[17] = '0' + ((d / 10) % 10);
    buf[18] = '0' + (d % 10);
    
    // Add gesture code
    buf[20] = '0' + (gesture / 10);
    buf[21] = '0' + (gesture % 10);
    
    HAL_UART_Transmit(&huart1, buf, 23, 1);
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

    const char* header = "EMG Gesture Control System\r\n";
    HAL_UART_Transmit(&huart1, (uint8_t*)header, strlen(header), 100);
    
    // TEST MODE: Uncomment to test servos without EMG
    // test_all_gestures();
    // while(1) { HAL_Delay(1000); }

    HAL_TIM_Base_Start(&htim3);
    HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buffer, BUF_SIZE);
    
    uint32_t last_dma_pos = 0;
    uint32_t last_process_time = HAL_GetTick();
    uint32_t classification_count = 0;
    
    GestureType gesture_buffer[10] = {GESTURE_REST};
    uint8_t buffer_index = 0;
    
    while (1) {
        uint32_t dma_pos = BUF_SIZE - hdma_adc1.Instance->NDTR;
        
        if (dma_pos != last_dma_pos) {
            uint32_t sample_idx = (dma_pos - ADC_CHANNELS + BUF_SIZE) % BUF_SIZE;
            
            if (sample_idx + 3 < BUF_SIZE) {
                uint16_t ch1 = adc_buffer[sample_idx];     // PA0: flexor carpi radialis (a0)
                uint16_t ch2 = adc_buffer[sample_idx + 1]; // PA1: brachioradialis (a1)
                uint16_t ch3 = adc_buffer[sample_idx + 2]; // PA2: flexor carpi ulnaris (a2)
                uint16_t ch4 = adc_buffer[sample_idx + 3]; // PA3: flexor digitorum superficialis (a3)
                
                emg_buffer_add_sample(&emg_buffer, ch1, ch2, ch3, ch4);
                
                // Process every 50ms (20Hz classification rate)
                uint32_t current_time = HAL_GetTick();
                if (current_time - last_process_time >= 50) {
                    last_process_time = current_time;
                    
                    // Try to process a window
                    if (emg_buffer_process_window(&emg_buffer, extracted_features)) {
                        current_gesture = classify_gesture(extracted_features);
                        
                        gesture_buffer[buffer_index] = current_gesture;
                        buffer_index = (buffer_index + 1) % 10;
                        
                        // Check if we have consistent classifications
                        bool consistent = true;
                        for (int i = 1; i < 10; i++) {
                            if (gesture_buffer[i] != gesture_buffer[0]) {
                                consistent = false;
                                break;
                            }
                        }
                        
                        // If consistent for 10 samples (500ms), execute gesture
                        if (consistent && gesture_buffer[0] != last_gesture) {
                            last_gesture = gesture_buffer[0];
                            execute_gesture(last_gesture);
                        }
                        
                        // Output for debugging (every 10 classifications)
                        classification_count++;
                        if (classification_count % 10 == 0) {
                            fast_output(ch1, ch2, ch3, ch4, current_gesture);
                        }
                    }
                }
            }
            
            last_dma_pos = dma_pos;
        }
        
        // blink LED 
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
        for(volatile int i = 0; i < 1000000; i++);
    }
}

extern "C" {
    int _write(int file, char *ptr, int len) {
        HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, 1);
        return len;
    }
    
    void DMA2_Stream0_IRQHandler(void) { 
        HAL_DMA_IRQHandler(&hdma_adc1); 
    }
    
    void TIM3_IRQHandler(void) { 
        HAL_TIM_IRQHandler(&htim3); 
    }
}