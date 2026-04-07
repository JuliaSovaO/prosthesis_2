#include "main.h"
#include "pca9685.h"
#include "periph_init.h"
#include "stm32f7xx_hal.h"
#include "servo_control.h"
#include "emg_control.h"
#include <stdio.h>

volatile bool data_rdy_f = false;
__attribute__((aligned(4))) uint16_t adc_buffer[ADC_CHANNELS * SAMPLES] = {0};
PCA9685_HandleTypeDef pca9685;

extern UART_HandleTypeDef huart6;

int main(void)
{
    HAL_Init();

    // Blue LED on PA5 for status
    __HAL_RCC_GPIOA_CLK_ENABLE();
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
    
    // Blink LED 3 times to indicate boot
    for(int i = 0; i < 3; i++) {
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
        HAL_Delay(200);
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
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
    printf("    EMG ANN PROSTHESIS CONTROL v2.0\r\n");
    printf("    STM32F723E-DISCOVERY\r\n");
    printf("========================================\r\n");
    printf("ADC Sampling Rate: ~31,250 Hz\r\n");
    printf("ANN Input: %d features\r\n", ANN_INPUT_SIZE);
    printf("ANN Output: %d gestures\r\n", ANN_NUM_CLASSES);
    printf("Window: %d ms, Step: %d ms\r\n", 
           (EMG_WINDOW_SIZE * 1000) / 1000,
           (EMG_WINDOW_STEP * 1000) / 1000);
    
    // Initialize PCA9685 on I2C2
    printf("\r\n=== INITIALIZING PCA9685 ===\r\n");
    if (PCA9685_Init(&pca9685, &hi2c2, PCA9685_I2C_ADDRESS, 50.0))
    {
        printf("PCA9685 initialized successfully\r\n");
    }
    else
    {
        printf("PCA9685 initialization FAILED!\r\n");
        Error_Handler();
    }
    
    // Initialize EMG control (includes ANN)
    EMG_Control_Init();
    
    // Auto-calibrate baseline
    EMG_AutoCalibrate();
    
    // Clear ADC buffer
    for (int i = 0; i < ADC_CHANNELS * SAMPLES; i++) {
        adc_buffer[i] = 0;
    }

    printf("\r\n=== STARTING ADC2 DMA ===\r\n");
    printf("System ready! Make a gesture to begin...\r\n");
    printf("========================================\r\n\n");

    // Start ADC2 with DMA
    if (HAL_ADC_Start_DMA(&hadc2, (uint32_t *)adc_buffer, ADC_CHANNELS * SAMPLES) != HAL_OK)
    {
        Error_Handler();
    }
    HAL_ADC_Start(&hadc2);

    uint32_t last_heartbeat = 0;

    while (1)
    {
        // Process EMG data
        EMG_Control_Process();
        
        uint32_t now = HAL_GetTick();
        
        // Heartbeat LED - faster blink when active
        int last_idx = (SAMPLES - 1) * ADC_CHANNELS;
        uint8_t is_active = 0;
        for (int i = 0; i < 4; i++) {
            if (adc_buffer[last_idx + i] > 600) {
                is_active = 1;
                break;
            }
        }
        
        uint32_t heartbeat_interval = is_active ? 100 : 500;
        if (now - last_heartbeat >= heartbeat_interval) {
            HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
            last_heartbeat = now;
        }
        
        HAL_Delay(1);
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