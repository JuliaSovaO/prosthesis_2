// main.cpp - SIMPLIFIED
#include "main.h"
#include "periph_init.h"
#include "stm32f4xx_hal.h"
#include "emg_classifier.h"
#include "data_collection.h"
#include "servo_control.h"
#include <stdio.h>
#include <string.h>

// Global variables
volatile bool data_rdy_f = false;
volatile bool buffer_full = false;
uint32_t buffer_index = 0;
__attribute__((aligned(4))) uint16_t adc_buffer[3] = {0};

// Data collection buffers (defined here)
EMG_Sample sample_buffer[BUFFER_SAMPLES];
volatile uint32_t sample_buffer_idx = 0;

// PCA9685 handle
extern PCA9685_HandleTypeDef pca9685;

// Gesture names (defined here)
const char* GESTURE_NAMES[NUM_GESTURES] = {
    "REST", "ROCK", "SCISSORS", "PAPER", "ONE",
    "THREE", "FOUR", "GOOD", "OKAY", "FINGER_GUN"
};

// Function prototypes
void control_servo(int gesture);
void Error_Handler(void);
bool init_servos(void);

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    MX_I2C1_Init();
    MX_TIM3_Init();  // Timer for 2000Hz sampling
    
    // Initialize servos
    if (!init_servos()) {
        printf("Failed to initialize servos!\n");
    } else {
        printf("Servos initialized\n");
    }
    
    // Clear buffers
    memset(adc_buffer, 0, sizeof(adc_buffer));
    memset(sample_buffer, 0, sizeof(sample_buffer));
    
    printf("=== EMG Gesture Recognition System ===\n");
    printf("STM32F401CC @ 84MHz\n");
    printf("Sampling rate: %d Hz\n", SAMPLING_RATE_HZ);
    printf("\nSelect mode:\n");
    printf("1. Data Collection (2000Hz)\n");
    printf("2. Real-time Classification\n");
    printf("Enter choice (1 or 2): ");
    
    // Wait for mode selection
    int mode = 0;
    while (mode != 1 && mode != 2) {
        char c = 0;
        if (HAL_UART_Receive(&huart1, (uint8_t*)&c, 1, 100) == HAL_OK) {
            mode = c - '0';
        }
    }
    
    if (mode == 1) {
        // Data collection mode
        data_collection_main();
    } else {
        // Real-time classification mode
        printf("\n=== Real-time Classification Mode ===\n");
        printf("Starting classification...\n");
        
        // For now, just test servos
        printf("Testing servos...\n");
        TestServoSequence();
    }
    
    printf("System shutdown.\n");
    while (1) {
        HAL_Delay(1000);
    }
}

bool init_servos(void) {
    // Initialize PCA9685
    if (!PCA9685_Init(&pca9685, &hi2c1, 0x40, 50.0f)) {
        printf("PCA9685 init failed!\n");
        return false;
    }
    
    // Initialize to open position
    OpenHand();
    HAL_Delay(500);
    
    return true;
}

void control_servo(int gesture) {
    // Your servo control code here
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
    void TIM3_IRQHandler(void) { HAL_TIM_IRQHandler(&htim3); }
}