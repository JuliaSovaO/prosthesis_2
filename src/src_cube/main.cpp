// main.cpp - Application code only
#include "main.h"
#include "periph_init.h"
#include <string.h>

// Constants
#define ADC_CHANNELS 3
#define BUF_SIZE 6  // Small buffer for fast updates

// Global buffer
__attribute__((aligned(4))) uint16_t adc_buffer[BUF_SIZE] = {0};

// External handles
extern ADC_HandleTypeDef hadc1;
extern DMA_HandleTypeDef hdma_adc1;
extern UART_HandleTypeDef huart1;
extern TIM_HandleTypeDef htim3;

// Ultra-fast output function
void fast_output(uint16_t a, uint16_t b, uint16_t c) {
    static uint8_t buf[] = "0000,0000,0000\n";
    
    // Fast update using bit operations
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
    
    HAL_UART_Transmit(&huart1, buf, 15, 1);
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    MX_TIM3_Init();
    
    // Start everything
    HAL_TIM_Base_Start(&htim3);
    HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buffer, BUF_SIZE);
    
    // Send one-time header
    const char* header = "START\n";
    HAL_UART_Transmit(&huart1, (uint8_t*)header, 6, 100);
    
    // Main loop
    uint32_t last_dma_pos = 0;
    uint32_t sample_count = 0;
    
    while (1) {
        // Get current DMA position
        uint32_t dma_pos = BUF_SIZE - hdma_adc1.Instance->NDTR;
        
        // Check if we have new data
        if (dma_pos != last_dma_pos) {
            // Calculate sample position (go back 1 complete sample)
            uint32_t sample_idx = (dma_pos - ADC_CHANNELS + BUF_SIZE) % BUF_SIZE;
            
            if (sample_idx + 2 < BUF_SIZE) {
                // Output immediately
                fast_output(adc_buffer[sample_idx],
                           adc_buffer[sample_idx + 1],
                           adc_buffer[sample_idx + 2]);
                
                sample_count++;
                
                // Flash LED every 2000 samples (1 second at 2000Hz)
                if (sample_count % 2000 == 0) {
                    HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
                }
            }
            
            last_dma_pos = dma_pos;
        }
    }
}

void Error_Handler(void) {
    while (1) {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        for(volatile int i = 0; i < 1000000; i++);
    }
}

// UART write function (for any printf usage)
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