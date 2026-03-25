#include "main.h"
#include "periph_init.h"
#include "ann/ann_inference.h"
#include <string.h>
#include <stdio.h>

#define BUF_SIZE 8  // 2 complete samples for 4 channels (4*2)

__attribute__((aligned(4))) uint16_t adc_buffer[BUF_SIZE] = {0};
volatile bool data_rdy_f = false;

extern ADC_HandleTypeDef hadc2;
extern DMA_HandleTypeDef hdma_adc2;
extern UART_HandleTypeDef huart6;

PCA9685_HandleTypeDef pca9685;

void fast_output(uint16_t ch1, uint16_t ch2, uint16_t ch3, uint16_t ch4) {
    static uint8_t buf[] = "0000,0000,0000,0000\n";
    
    // Channel 1
    buf[0] = '0' + (ch1 / 1000);
    buf[1] = '0' + ((ch1 / 100) % 10);
    buf[2] = '0' + ((ch1 / 10) % 10);
    buf[3] = '0' + (ch1 % 10);
    
    // Channel 2
    buf[5] = '0' + (ch2 / 1000);
    buf[6] = '0' + ((ch2 / 100) % 10);
    buf[7] = '0' + ((ch2 / 10) % 10);
    buf[8] = '0' + (ch2 % 10);
    
    // Channel 3
    buf[10] = '0' + (ch3 / 1000);
    buf[11] = '0' + ((ch3 / 100) % 10);
    buf[12] = '0' + ((ch3 / 10) % 10);
    buf[13] = '0' + (ch3 % 10);
    
    // Channel 4
    buf[15] = '0' + (ch4 / 1000);
    buf[16] = '0' + ((ch4 / 100) % 10);
    buf[17] = '0' + ((ch4 / 10) % 10);
    buf[18] = '0' + (ch4 % 10);
    
    HAL_UART_Transmit(&huart6, buf, 20, 1);
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART6_UART_Init();
    MX_DMA_Init();
    MX_ADC2_Init();
    MX_I2C2_Init();
    
    huart6.Init.BaudRate = 230400;
    HAL_UART_Init(&huart6);
    
    // Initialize PCA9685 servo controller
    if (!PCA9685_Init(&pca9685, &hi2c2, PCA9685_I2C_ADDRESS, 50.0f)) {
        printf("PCA9685 Init Failed!\r\n");
        Error_Handler();
    }
    
    EMG_Control_Init();
    EMG_AutoCalibrate();
    HAL_ADC_Start_DMA(&hadc2, (uint32_t*)adc_buffer, BUF_SIZE);
    
    printf("=== EMG ANN Gesture Control Started ===\r\n");
    printf("Sampling rate: 1500 Hz\r\n");
    printf("Window size: %d samples\r\n", ANN_WINDOW_SIZE);
    printf("Window step: %d samples\r\n", ANN_WINDOW_STEP);
    printf("\r\n");
    
    // Test finger sequence on button press
    uint32_t last_button_check = 0;
    uint8_t button_pressed = 0;
    
    while (1) {
        EMG_Control_Process();
        
        if (HAL_GetTick() - last_button_check > 100) {
            last_button_check = HAL_GetTick();
            
            if (HAL_GPIO_ReadPin(USER_BUTTON_GPIO_Port, USER_BUTTON_Pin) == GPIO_PIN_SET) {
                if (!button_pressed) {
                    button_pressed = 1;
                    printf("\r\n=== BUTTON PRESSED - Running test sequence ===\r\n");
                    TestFingerSequence();
                }
            } else {
                button_pressed = 0;
            }
        }
        
        // Toggle blue LED to indicate running
        static uint32_t last_led = 0;
        if (HAL_GetTick() - last_led > 1000) {
            last_led = HAL_GetTick();
            HAL_GPIO_TogglePin(USER_LED_BLUE_GPIO_Port, USER_LED_BLUE_Pin);
        }
    }
}

void Error_Handler(void) {
    while (1) {
        HAL_GPIO_TogglePin(USER_LED_RED_GPIO_Port, USER_LED_RED_Pin);
        for(volatile int i = 0; i < 1000000; i++);
    }
}

extern "C" {
    int _write(int file, char *ptr, int len) {
        HAL_UART_Transmit(&huart6, (uint8_t*)ptr, len, 1);
        return len;
    }
}

extern "C" void DMA2_Stream2_IRQHandler(void) {
    uint32_t isr = DMA2->LISR;
    
    // Check Transfer Complete for stream 2 (bit 5 in LISR)
    if (isr & DMA_LISR_TCIF2) {
        DMA2->LIFCR = DMA_LIFCR_CTCIF2;  // Clear flag
        data_rdy_f = true;
    }
    
    // Check Half Transfer for stream 2 (bit 4 in LISR)
    if (isr & DMA_LISR_HTIF2) {
        DMA2->LIFCR = DMA_LIFCR_CHTIF2;
    }
    
    // Check Transfer Error for stream 2 (bit 3 in LISR)
    if (isr & DMA_LISR_TEIF2) {
        DMA2->LIFCR = DMA_LIFCR_CTEIF2;
        Error_Handler();
    }
    
    // Check Direct Mode Error for stream 2 (bit 2 in LISR)
    if (isr & DMA_LISR_DMEIF2) {
        DMA2->LIFCR = DMA_LIFCR_CDMEIF2;
    }
    
    // Check FIFO Error for stream 2 (bit 0 in LISR)
    if (isr & DMA_LISR_FEIF2) {
        DMA2->LIFCR = DMA_LIFCR_CFEIF2;
    }
}