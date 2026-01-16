#ifndef HAL_UART_MODULE_ENABLED
#define HAL_UART_MODULE_ENABLED
#endif

#include "main.h"
#include "periph_init.h"
#include "stm32f4xx_hal.h"
#include <stdio.h>
#include <cstring>

volatile bool data_rdy_f = false;
__attribute__((aligned(4))) uint16_t adc_buffer[ADC_CHANNELS * SAMPLES] = {0};

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    
    // Clear buffer
    memset(adc_buffer, 0, sizeof(adc_buffer));
    
    // Start ADC with DMA
    if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_buffer, ADC_CHANNELS * SAMPLES) != HAL_OK) {
        Error_Handler();
    }
    
    // Wait for first data
    while (!data_rdy_f) {
        HAL_Delay(1);
    }
    
    // Restart for continuous operation
    HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_buffer, ADC_CHANNELS * SAMPLES);
    
    printf("START\n");  // One-time marker
    
    while (1) {
        if (data_rdy_f) {
            int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
            
            // Output format: ch1,ch2,ch3
            printf("%d,%d,%d\n",
                   adc_buffer[last_idx + 0],  // CH1 (PA0)
                   adc_buffer[last_idx + 1],  // CH2 (PA1)
                   adc_buffer[last_idx + 2]); // CH3 (PA2)
            
            data_rdy_f = false;
        }
        HAL_Delay(1);
    }
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
    if (hadc->Instance == ADC1) {
        __DSB();
        data_rdy_f = true;
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
        HAL_UART_Transmit(&huart1, (uint8_t *)ptr, len, HAL_MAX_DELAY);
        return len;
    }
    void DMA2_Stream0_IRQHandler(void) { HAL_DMA_IRQHandler(&hdma_adc1); }
}