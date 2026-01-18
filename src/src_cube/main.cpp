#include "main.h"
#include "periph_init.h"
#include <string.h>

#define ADC_CHANNELS 4
#define BUF_SIZE 8 

__attribute__((aligned(4))) uint16_t adc_buffer[BUF_SIZE] = {0};

extern ADC_HandleTypeDef hadc1;
extern DMA_HandleTypeDef hdma_adc1;
extern UART_HandleTypeDef huart1;
extern TIM_HandleTypeDef htim3;

void fast_output(uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
    static uint8_t buf[] = "0000,0000,0000,0000\n";  // Changed: 4 channels, 20 bytes
    
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
    
    HAL_UART_Transmit(&huart1, buf, 20, 1);  // Changed: 20 bytes
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    MX_TIM3_Init();
    
    HAL_TIM_Base_Start(&htim3);
    HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buffer, BUF_SIZE);
    
    const char* header = "START_4CH\n";
    HAL_UART_Transmit(&huart1, (uint8_t*)header, strlen(header), 100);
    
    uint32_t last_dma_pos = 0;
    uint32_t sample_count = 0;
    
    while (1) {
        uint32_t dma_pos = BUF_SIZE - hdma_adc1.Instance->NDTR;
        
        if (dma_pos != last_dma_pos) {
            uint32_t sample_idx = (dma_pos - ADC_CHANNELS + BUF_SIZE) % BUF_SIZE;
            
            if (sample_idx + 3 < BUF_SIZE) {
                fast_output(adc_buffer[sample_idx],
                           adc_buffer[sample_idx + 1],
                           adc_buffer[sample_idx + 2],
                           adc_buffer[sample_idx + 3]);  
                
                sample_count++;
                
                if (sample_count % 1500 == 0) {
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