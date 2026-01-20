#include "gesture.h"
#include <stdio.h>
#include "main.h"  // For HAL_UART_Transmit

extern UART_HandleTypeDef huart1;

void execute_gesture(GestureType gesture) {
    char buffer[100];
    int len = 0;
    
    switch(gesture) {
        case GESTURE_REST:
            // rest - relaxed
            len = sprintf(buffer, "Executing: REST (relaxed)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Set all servos to relaxed position
            SetServo1Angle(20);  // Thumb slightly bent
            SetServo2Angle(20);  // Index slightly bent
            SetServo3Angle(20);  // Middle slightly bent
            SetServo4Angle(20);  // Ring slightly bent
            SetServo5Angle(20);   // Pinky slightly bent
            break;
            
        case GESTURE_ROCK:
            // rock - all closed
            len = sprintf(buffer, "Executing: ROCK (all closed)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            CloseHand();  // This should close all fingers
            break;
            
        case GESTURE_SCISSORS:
            // scissors - index, middle opened, others closed
            len = sprintf(buffer, "Executing: SCISSORS (index, middle opened)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Index and middle open, others closed
            SetServo1Angle(SERVO1_CLOSED);   // Thumb closed
            SetServo2Angle(SERVO2_OPEN);     // Index open
            SetServo3Angle(SERVO3_OPEN);     // Middle open
            SetServo4Angle(SERVO4_CLOSED);   // Ring closed
            SetServo5Angle(SERVO5_CLOSED);   // Pinky closed
            break;
            
        case GESTURE_PAPER:
            // paper - all opened
            len = sprintf(buffer, "Executing: PAPER (all opened)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            OpenHand();  // This should open all fingers
            break;
            
        case GESTURE_FUCK:
            // fuck - middle finger opened, others closed
            len = sprintf(buffer, "Executing: FUCK (middle finger)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Middle open, others closed
            SetServo1Angle(SERVO1_CLOSED);   // Thumb closed
            SetServo2Angle(SERVO2_CLOSED);   // Index closed
            SetServo3Angle(SERVO3_OPEN);     // Middle open
            SetServo4Angle(SERVO4_CLOSED);   // Ring closed
            SetServo5Angle(SERVO5_CLOSED);   // Pinky closed
            break;
            
        case GESTURE_THREE:
            // three - index, middle, ring opened, others closed
            len = sprintf(buffer, "Executing: THREE (index, middle, ring opened)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Index, middle, ring open, thumb and pinky closed
            SetServo1Angle(SERVO1_CLOSED);   // Thumb closed
            SetServo2Angle(SERVO2_OPEN);     // Index open
            SetServo3Angle(SERVO3_OPEN);     // Middle open
            SetServo4Angle(SERVO4_OPEN);     // Ring open
            SetServo5Angle(SERVO5_CLOSED);   // Pinky closed
            break;
            
        case GESTURE_FOUR:
            // four - only thumb closed
            len = sprintf(buffer, "Executing: FOUR (only thumb closed)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Only thumb closed, all others open
            SetServo1Angle(SERVO1_CLOSED);   // Thumb closed
            SetServo2Angle(SERVO2_OPEN);     // Index open
            SetServo3Angle(SERVO3_OPEN);     // Middle open
            SetServo4Angle(SERVO4_OPEN);     // Ring open
            SetServo5Angle(SERVO5_OPEN);     // Pinky open
            break;
            
        case GESTURE_GOOD:
            // good - only thumb opened
            len = sprintf(buffer, "Executing: GOOD (only thumb opened)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Only thumb open, all others closed
            SetServo1Angle(SERVO1_OPEN);     // Thumb open (up)
            SetServo2Angle(SERVO2_CLOSED);   // Index closed
            SetServo3Angle(SERVO3_CLOSED);   // Middle closed
            SetServo4Angle(SERVO4_CLOSED);   // Ring closed
            SetServo5Angle(SERVO5_CLOSED);   // Pinky closed
            break;
            
        case GESTURE_OKAY:
            // okay - index and thumb make a circle, others opened
            len = sprintf(buffer, "Executing: OKAY (index & thumb circle)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Index and thumb make circle (both at ~60°), others open
            SetServo1Angle(80);              // Thumb at circle position
            SetServo2Angle(100);              // Index at circle position
            SetServo3Angle(SERVO3_OPEN);     // Middle open
            SetServo4Angle(SERVO4_OPEN);     // Ring open
            SetServo5Angle(SERVO5_OPEN);     // Pinky open
            break;
            
        case GESTURE_FINGER_GUN:
            // finger-gun - index, thumb opened, others closed
            len = sprintf(buffer, "Executing: FINGER-GUN (index & thumb)\r\n");
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Index and thumb open, others closed
            SetServo1Angle(SERVO1_OPEN);     // Thumb open
            SetServo2Angle(SERVO2_OPEN);     // Index open
            SetServo3Angle(SERVO3_CLOSED);   // Middle closed
            SetServo4Angle(SERVO4_CLOSED);   // Ring closed
            SetServo5Angle(SERVO5_CLOSED);   // Pinky closed
            break;
            
        default:
            len = sprintf(buffer, "Unknown gesture: %d\r\n", gesture);
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
            // Default to rest position
            SetServo1Angle(20);
            SetServo2Angle(20);
            SetServo3Angle(20);
            SetServo4Angle(20);
            SetServo5Angle(20);
            break;
    }
}