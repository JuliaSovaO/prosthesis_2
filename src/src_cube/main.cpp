// main.cpp - UPDATED VERSION WITH PCA9685 INIT
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
__attribute__((aligned(4))) uint16_t adc_buffer[3 * 1] = {0};

// EMG buffer
uint16_t emg_buffer[3][83];
uint32_t buffer_idx = 0;

// PCA9685 handle (defined in servo_control.c but needs to be accessible)
extern PCA9685_HandleTypeDef pca9685;

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
    MX_I2C1_Init();  // Add this for I2C
    
    // Initialize servos
    if (!init_servos()) {
        printf("Failed to initialize servos!\n");
    } else {
        printf("Servos initialized successfully\n");
    }
    
    // Clear buffers
    memset(adc_buffer, 0, sizeof(adc_buffer));
    memset(emg_buffer, 0, sizeof(emg_buffer));
    
    printf("EMG Gesture Recognition System\n");
    printf("==============================\n");
    printf("1. Data Collection Mode\n");
    printf("2. Real-time Classification\n");
    printf("Select mode (1 or 2): \n");
    
    // Wait for mode selection via serial
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
        printf("Starting real-time classification...\n");
        printf("Accuracy: 91.3%% (trained)\n");
        printf("Window: 250ms, Step: 25ms\n");
        printf("Ready...\n");
        
        // Start ADC with DMA
        if (HAL_ADC_Start_DMA(&hadc1, (uint32_t *)adc_buffer, 3 * 1) != HAL_OK) {
            Error_Handler();
        }
        
        uint32_t last_prediction_time = 0;
        
        while (1) {
            if (data_rdy_f) {
                // Get latest sample
                uint16_t ch1 = adc_buffer[0];
                uint16_t ch2 = adc_buffer[1];
                uint16_t ch3 = adc_buffer[2];
                
                // Add to buffer
                if (buffer_idx < 83) {
                    emg_buffer[0][buffer_idx] = ch1;
                    emg_buffer[1][buffer_idx] = ch2;
                    emg_buffer[2][buffer_idx] = ch3;
                    buffer_idx++;
                }
                
                // Every 25ms (8 samples at 332Hz), try to classify
                uint32_t now = HAL_GetTick();
                if (now - last_prediction_time >= 25 && buffer_idx >= 83) {
                    float features[18];
                    
                    // Extract features
                    extract_all_features(emg_buffer[0], emg_buffer[1], emg_buffer[2], 
                                        buffer_idx, features);
                    
                    // Classify
                    int gesture = classify_gesture(features);
                    
                    // Output result
                    printf("GESTURE:%d:%s\n", gesture, GESTURE_NAMES[gesture]);
                    
                    // Control servos
                    control_servo(gesture);
                    
                    // Shift buffer for next window (remove oldest 8 samples)
                    if (buffer_idx >= 8) {
                        for (uint32_t i = 0; i < buffer_idx - 8; i++) {
                            emg_buffer[0][i] = emg_buffer[0][i + 8];
                            emg_buffer[1][i] = emg_buffer[1][i + 8];
                            emg_buffer[2][i] = emg_buffer[2][i + 8];
                        }
                        buffer_idx -= 8;
                    }
                    
                    last_prediction_time = now;
                }
                
                data_rdy_f = false;
            }
            
            // Flash LED to show it's running
            static uint32_t last_blink = 0;
            if (HAL_GetTick() - last_blink > 500) {
                HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
                last_blink = HAL_GetTick();
            }
            
            HAL_Delay(1);
        }
    }
}

bool init_servos(void) {
    // Initialize PCA9685
    if (!PCA9685_Init(&pca9685, &hi2c1, 0x40, 50.0f)) {
        printf("PCA9685 initialization failed!\n");
        return false;
    }
    
    // Initialize all servos to open position
    OpenHand();
    HAL_Delay(1000);
    
    printf("Servos initialized to open position\n");
    return true;
}

void control_servo(int gesture) {
    switch(gesture) {
        case 0: // REST - open hand
            OpenHand();
            break;
            
        case 1: // ROCK
            SetServo1Angle(90);   // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(90);   // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;
            
        case 2: // SCISSORS
            SetServo1Angle(0);    // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(90);   // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;
            
        case 3: // PAPER
            OpenHand();
            break;

        case 4: // ONE
            SetServo1Angle(90);   // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(90);   // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;

        case 5: // THREE
            SetServo1Angle(90);   // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(0);    // Middle
            SetServo4Angle(0);    // Ring
            SetServo5Angle(90);   // Pinky
            break;

        case 6: // FOUR
            SetServo1Angle(90);   // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(0);    // Middle
            SetServo4Angle(0);    // Ring
            SetServo5Angle(0);    // Pinky
            break;

        case 7: // GOOD
            SetServo1Angle(0);    // Thumb
            SetServo2Angle(90);   // Index
            SetServo3Angle(90);   // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;
        
        case 8: // OKAY
            SetServo1Angle(45);   // Thumb
            SetServo2Angle(45);   // Index
            SetServo3Angle(0);    // Middle
            SetServo4Angle(0);    // Ring
            SetServo5Angle(0);    // Pinky
            break;

        case 9: // FINGER GUN
            SetServo1Angle(0);    // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(90);   // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;
            
        default:
            OpenHand();
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