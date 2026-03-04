#include "main.h"
#include "pca9685.h"
#include "periph_init.h"
#include "stm32f7xx_hal.h"
#include "servo_control.h"
#include "gestures.h"
#include "emg_control.h"
#include <stdio.h>

// const uint16_t SAMPLES = 512;
// const uint16_t ADC_CHANNELS = 4;

volatile bool data_rdy_f = false;
__attribute__((aligned(4))) uint16_t adc_buffer[ADC_CHANNELS * SAMPLES] = {0};
PCA9685_HandleTypeDef pca9685;

extern UART_HandleTypeDef huart6;

int main(void)
{
    HAL_Init();

    // PA5 blue LED
    __HAL_RCC_GPIOA_CLK_ENABLE();
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
    
    // blink LED 5 times rapidly
    for(int i = 0; i < 5; i++) {
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET); // LED on
        HAL_Delay(200);
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);   // LED off
        HAL_Delay(200);
    }

    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART6_UART_Init(); 
    MX_DMA_Init();
    MX_ADC2_Init();
    MX_I2C2_Init();

    HAL_Delay(500);

    printf("\r\n\n");
    printf("========================================\r\n");
    printf("=== 4-CHANNEL EMG PROSTHESIS CONTROL ===\r\n");
    printf("=== STM32F723E-DISCOVERY            ===\r\n");
    printf("========================================\r\n");
    printf("Sample Rate: ~31,250 sets/sec\r\n\r\n");
    EMG_Control_Init();
    EMG_AutoCalibrate();

    // I2C2
    printf("=== I2C2 DEVICE SCAN ===\r\n");
    for (uint8_t addr = 0x01; addr < 0x7F; addr++) {
        if (HAL_I2C_IsDeviceReady(&hi2c2, (addr << 1), 1, 10) == HAL_OK) {
            printf("Found device at 0x%02X\r\n", addr);
        }
    }

    // Initialize PCA9685 on I2C2
    if (PCA9685_Init(&pca9685, &hi2c2, PCA9685_I2C_ADDRESS, 50.0))
    {
        printf("PCA9685 initialized successfully on I2C2\r\n");
    }
    else
    {
        printf("PCA9685 initialization failed!\r\n");
    }

    // clear buffer
    for (int i = 0; i < ADC_CHANNELS * SAMPLES; i++) {
        adc_buffer[i] = 0;
    }

    printf("\r\n=== Starting ADC2 DMA ===\r\n");

    // start ADC2 with DMA
    if (HAL_ADC_Start_DMA(&hadc2, (uint32_t *)adc_buffer, ADC_CHANNELS * SAMPLES) != HAL_OK)
    {
        Error_Handler();
    }

    uint32_t last_print = 0;

    while (1)
    {
        if (data_rdy_f)
        {
            uint32_t now = HAL_GetTick();
            
            // print at ~1000Hz
            if (now - last_print >= 1) {
                int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
                
                // CH1,CH2,CH3,CH4
                printf("%d,%d,%d,%d\r\n",
                       adc_buffer[last_idx + 0],
                       adc_buffer[last_idx + 1],
                       adc_buffer[last_idx + 2],
                       adc_buffer[last_idx + 3]);
                
                last_print = now;
            }
            
            data_rdy_f = false;
        }
        
        // toggle green LED for activity
        static uint32_t last_led = 0;
        if (HAL_GetTick() - last_led >= 500) {
            HAL_GPIO_TogglePin(USER_LED_GREEN_GPIO_Port, USER_LED_GREEN_Pin);
            last_led = HAL_GetTick();
        }
        
    }
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc->Instance == ADC2)
    {
        __DSB();
        data_rdy_f = true;
    }
}

void Error_Handler(void)
{
    printf("FATAL ERROR\r\n");
    while (1)
    {
        HAL_GPIO_TogglePin(USER_LED_RED_GPIO_Port, USER_LED_RED_Pin);
        HAL_Delay(100);
    }
}

extern "C"
{
    int _write(int file, char *ptr, int len)
    {
        HAL_UART_Transmit(&huart6, (uint8_t *)ptr, len, HAL_MAX_DELAY);
        return len;
    }
}