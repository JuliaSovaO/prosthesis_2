#ifndef PERIPH_INIT_H
#define PERIPH_INIT_H

#include "stm32f7xx_hal.h"

extern ADC_HandleTypeDef hadc2;  // ADC2
extern UART_HandleTypeDef huart6; // USART2
extern DMA_HandleTypeDef hdma_adc2;
extern I2C_HandleTypeDef hi2c2;   // I2C2

void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_ADC2_Init(void);
void MX_DMA_Init(void);
void MX_USART6_UART_Init(void);
void MX_I2C2_Init(void);

#endif