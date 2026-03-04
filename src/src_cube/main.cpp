#include "main.h"
#include "periph_init.h"
#include <string.h>
#include <stdio.h>

#define BUF_SIZE 8  // 2 complete samples for 4 channels (4*2)

__attribute__((aligned(4))) uint16_t adc_buffer[BUF_SIZE] = {0};

extern ADC_HandleTypeDef hadc2;
extern DMA_HandleTypeDef hdma_adc2;
extern UART_HandleTypeDef huart6;

// 4*4 digits + 3 commas + newline = 20 bytes
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
    
    HAL_UART_Transmit(&huart6, buf, 20, 1);  // 20 bytes total
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART6_UART_Init();
    MX_DMA_Init();
    MX_ADC2_Init();
    
    huart6.Init.BaudRate = 230400;
    HAL_UART_Init(&huart6);
    
    // start ADC with DMA (small buffer for low latency)
    HAL_ADC_Start_DMA(&hadc2, (uint32_t*)adc_buffer, BUF_SIZE);
    
    // send one-time header
    const char* header = "START\n";
    HAL_UART_Transmit(&huart6, (uint8_t*)header, 6, 100);
    
    uint32_t last_dma_pos = 0;
    uint32_t sample_count = 0;
    // uint32_t last_stats = HAL_GetTick();
    
    // LEDs
    HAL_GPIO_WritePin(USER_LED_BLUE_GPIO_Port, USER_LED_BLUE_Pin, GPIO_PIN_SET);
    
    while (1) {
        // direct register access to get DMA position
        uint32_t dma_pos = BUF_SIZE - hdma_adc2.Instance->NDTR;
        
        if (dma_pos != last_dma_pos) {
            // calc sample position (go back 1 complete sample)
            uint32_t sample_idx = (dma_pos - ADC_CHANNELS + BUF_SIZE) % BUF_SIZE;
            
            if (sample_idx + 3 < BUF_SIZE) {
                fast_output(adc_buffer[sample_idx],
                           adc_buffer[sample_idx + 1],
                           adc_buffer[sample_idx + 2],
                           adc_buffer[sample_idx + 3]);
                
                sample_count++;
                
                // toggle green LED every 1000 samples (visual indicator)
                if (sample_count % 1000 == 0) {
                    HAL_GPIO_TogglePin(USER_LED_GREEN_GPIO_Port, USER_LED_GREEN_Pin);
                }
            }
            
            last_dma_pos = dma_pos;
        }
        
        // print speed stats every 5 seconds
        // if (HAL_GetTick() - last_stats >= 5000) {
        //     uint32_t speed = sample_count / 5;
            
        //     // use blue LED to indicate stats calculation
        //     HAL_GPIO_WritePin(USER_LED_BLUE_GPIO_Port, USER_LED_BLUE_Pin, GPIO_PIN_RESET);
            
        //     char stats_buf[32];
        //     int len = sprintf(stats_buf, "SPD:%lu Hz\n", speed);
        //     HAL_UART_Transmit(&huart6, (uint8_t*)stats_buf, len, 100);
            
        //     HAL_GPIO_WritePin(USER_LED_BLUE_GPIO_Port, USER_LED_BLUE_Pin, GPIO_PIN_SET);
            
        //     sample_count = 0;
        //     last_stats = HAL_GetTick();
        // }
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