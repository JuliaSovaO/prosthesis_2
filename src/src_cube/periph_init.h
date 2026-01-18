#ifndef PERIPH_INIT_H
#define PERIPH_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"
#include <stdbool.h>

// Function prototypes
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
void MX_GPIO_Init(void);
void MX_USART1_UART_Init(void);
void MX_DMA_Init(void);
void MX_ADC1_Init(void);
void MX_TIM2_Init(void);
void MX_TIM3_Init(void);
void MX_I2C1_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* PERIPH_INIT_H */