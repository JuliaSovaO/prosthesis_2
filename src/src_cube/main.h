#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stm32f7xx_hal.h"
#include "pca9685.h"
#include "servo_control.h"
#include "gestures.h"
#include "emg_control.h"
#include <stdbool.h>
#include <math.h>

#define SAMPLES (uint16_t)512
#define ADC_CHANNELS (uint16_t)4

extern volatile bool data_rdy_f;
extern uint16_t adc_buffer[];

void Error_Handler(void);
void TestServo(void);
void TestIndividualFingers(void);

// I2C2 for PCA9685
#define I2C2_SCL_Pin        GPIO_PIN_4
#define I2C2_SCL_GPIO_Port  GPIOH
#define I2C2_SDA_Pin        GPIO_PIN_5
#define I2C2_SDA_GPIO_Port  GPIOH

// USART6 for built-in Virtual COM Port (connected to ST-LINK)
#define USART6_TX_Pin        GPIO_PIN_6
#define USART6_TX_GPIO_Port  GPIOC
#define USART6_RX_Pin        GPIO_PIN_7
#define USART6_RX_GPIO_Port  GPIOC

// ADC2 pins for 4 EMG sensors
#define EMG1_PIN            GPIO_PIN_0
#define EMG1_GPIO_PORT      GPIOC
#define EMG1_ADC_CHANNEL    ADC_CHANNEL_10

#define EMG2_PIN            GPIO_PIN_1
#define EMG2_GPIO_PORT      GPIOC
#define EMG2_ADC_CHANNEL    ADC_CHANNEL_11

#define EMG3_PIN            GPIO_PIN_4
#define EMG3_GPIO_PORT      GPIOA
#define EMG3_ADC_CHANNEL    ADC_CHANNEL_4

#define EMG4_PIN            GPIO_PIN_4
#define EMG4_GPIO_PORT      GPIOC
#define EMG4_ADC_CHANNEL    ADC_CHANNEL_14

// User LEDs 
#define USER_LED_RED_Pin    GPIO_PIN_1
#define USER_LED_RED_GPIO_Port GPIOB
#define USER_LED_GREEN_Pin  GPIO_PIN_0
#define USER_LED_GREEN_GPIO_Port GPIOB
#define USER_LED_BLUE_Pin   GPIO_PIN_5
#define USER_LED_BLUE_GPIO_Port GPIOA

// User Button 
#define USER_BUTTON_Pin     GPIO_PIN_0
#define USER_BUTTON_GPIO_Port GPIOA

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */