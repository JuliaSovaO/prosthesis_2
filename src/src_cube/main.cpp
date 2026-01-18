#include "main.h"
#include "periph_init.h"
#include "emg_classifier.h"
#include "servo_control.h"
#include "pca9685.h"
#include "emg_model.h"
#include <string.h>
#include <stdio.h>

// Constants
#define ADC_CHANNELS 3
#define BUF_SIZE 6
#define CLASSIFICATION_INTERVAL 200  // 200ms
#define OUTPUT_RAW_DATA 1           // Set to 1 to enable raw data output

// Global buffer for ADC DMA
__attribute__((aligned(4))) uint16_t adc_buffer[BUF_SIZE] = {0};

// External handles
extern ADC_HandleTypeDef hadc1;
extern DMA_HandleTypeDef hdma_adc1;
extern UART_HandleTypeDef huart1;
extern TIM_HandleTypeDef htim3;
extern I2C_HandleTypeDef hi2c1;

// PCA9685 handle
PCA9685_HandleTypeDef pca9685;

// Global state
uint32_t last_classification_time = 0;
uint32_t last_output_time = 0;
uint32_t output_sample_count = 0;
uint8_t current_gesture = GESTURE_REST;

// Gesture names for debugging
const char* GESTURE_NAMES[10] = {
    "REST", "ROCK", "SCISSORS", "PAPER", "ONE",
    "THREE", "FOUR", "GOOD", "OKAY", "FINGER_GUN"
};

// Function prototypes
void Error_Handler(void);
bool init_servos(void);
void control_servos(uint8_t gesture);
void output_raw_data(uint16_t ch1, uint16_t ch2, uint16_t ch3);

// Initialize everything
void init_system(void) {
    // Initialize EMG classifier buffer
    init_emg_buffer(&emg_buffer);
    
    // Initialize servos
    if (!init_servos()) {
        printf("Servo initialization failed!\n");
        Error_Handler();
    }
    
    printf("EMG Gesture Recognition System Initialized\n");
    printf("Output raw data: %s\n", OUTPUT_RAW_DATA ? "ENABLED" : "DISABLED");
    printf("Sampling rate: 1500Hz, Classification interval: 200ms\n");
}

// Control servos based on gesture
void control_servos(uint8_t gesture) {
    switch(gesture) {
        case GESTURE_REST:
            OpenHand();
            break;
        case GESTURE_ROCK:
            SetServo1Angle(90);   // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(90);   // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;
        case GESTURE_SCISSORS:
            SetServo1Angle(0);    // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(90);   // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;
        case GESTURE_PAPER:
            OpenHand();
            break;
        case GESTURE_FUCK:
            SetServo1Angle(90);   // Thumb
            SetServo2Angle(90);   // Index
            SetServo3Angle(0);    // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;
        case GESTURE_THREE:
            SetServo1Angle(90);   // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(0);    // Middle
            SetServo4Angle(0);    // Ring
            SetServo5Angle(90);   // Pinky
            break;
        case GESTURE_FOUR:
            SetServo1Angle(90);   // Thumb
            SetServo2Angle(0);    // Index
            SetServo3Angle(0);    // Middle
            SetServo4Angle(0);    // Ring
            SetServo5Angle(0);    // Pinky
            break;
        case GESTURE_GOOD:
            SetServo1Angle(0);    // Thumb
            SetServo2Angle(90);   // Index
            SetServo3Angle(90);   // Middle
            SetServo4Angle(90);   // Ring
            SetServo5Angle(90);   // Pinky
            break;
        case GESTURE_OKAY:
            SetServo1Angle(45);   // Thumb
            SetServo2Angle(45);   // Index
            SetServo3Angle(0);    // Middle
            SetServo4Angle(0);    // Ring
            SetServo5Angle(0);    // Pinky
            break;
        case GESTURE_FINGER_GUN:
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

// Output raw data at 1500Hz (every ~0.667ms)
void output_raw_data(uint16_t ch1, uint16_t ch2, uint16_t ch3) {
    char buf[32];
    // Format: ch1,ch2,ch3,gesture_label
    snprintf(buf, sizeof(buf), "%04d,%04d,%04d,%d\n", 
             ch1, ch2, ch3, current_gesture);
    HAL_UART_Transmit(&huart1, (uint8_t*)buf, strlen(buf), 10);
    output_sample_count++;
}

int main(void) {
    // HAL initialization
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    MX_TIM3_Init();
    MX_I2C1_Init();
    
    // System initialization
    init_system();
    
    // Start ADC with DMA
    HAL_TIM_Base_Start(&htim3);
    HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buffer, BUF_SIZE);
    
    // Main loop
    uint32_t last_dma_pos = 0;
    uint32_t sample_count = 0;
    
    while (1) {
        uint32_t current_time = HAL_GetTick();
        
        // Get current DMA position
        uint32_t dma_pos = BUF_SIZE - hdma_adc1.Instance->NDTR;
        
        // Process new ADC data
        if (dma_pos != last_dma_pos) {
            uint32_t sample_idx = (dma_pos - ADC_CHANNELS + BUF_SIZE) % BUF_SIZE;
            
            if (sample_idx + 2 < BUF_SIZE) {
                uint16_t ch1_val = adc_buffer[sample_idx];
                uint16_t ch2_val = adc_buffer[sample_idx + 1];
                uint16_t ch3_val = adc_buffer[sample_idx + 2];
                
                // Add to EMG buffer
                add_emg_sample(&emg_buffer, ch1_val, ch2_val, ch3_val);
                
                // Output raw data at ~1500Hz
                if (OUTPUT_RAW_DATA) {
                    output_raw_data(ch1_val, ch2_val, ch3_val);
                }
                
                sample_count++;
                
                // Flash LED every 1000 samples
                if (sample_count % 1000 == 0) {
                    HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
                }
            }
            
            last_dma_pos = dma_pos;
        }
        
        // Classify gesture periodically (every 200ms)
        if (current_time - last_classification_time >= CLASSIFICATION_INTERVAL) {
            uint8_t new_gesture = classify_current_gesture();
            
            if (new_gesture != current_gesture) {
                current_gesture = new_gesture;
                control_servos(current_gesture);
                
                // Send debug info (only when gesture changes)
                if (current_gesture < 10) {
                    printf("Gesture: %s (%d)\n", 
                           GESTURE_NAMES[current_gesture], 
                           current_gesture);
                }
            }
            
            last_classification_time = current_time;
        }
        
        // Small delay to maintain 1500Hz sampling
        // At 1500Hz, we need ~0.667ms between samples
        // HAL_Delay(1) gives 1ms, which is close enough
        HAL_Delay(1);
    }
}

// Initialize servo controller
bool init_servos(void) {
    // Initialize PCA9685
    if (!PCA9685_Init(&pca9685, &hi2c1, 0x40, 50.0f)) {
        return false;
    }
    
    // Start with open hand
    OpenHand();
    HAL_Delay(500);
    
    return true;
}

void Error_Handler(void) {
    while (1) {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        HAL_Delay(100);
    }
}

// UART write function for printf
extern "C" {
    int _write(int file, char *ptr, int len) {
        HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, 100);
        return len;
    }
    
    void DMA2_Stream0_IRQHandler(void) { 
        HAL_DMA_IRQHandler(&hdma_adc1); 
    }
    
    void TIM3_IRQHandler(void) { 
        HAL_TIM_IRQHandler(&htim3); 
    }
}