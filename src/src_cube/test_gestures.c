#include "gesture.h"
#include "servo_control.h"
#include "main.h"
#include <stdio.h>

extern UART_HandleTypeDef huart1;

void test_all_gestures(void) {
    char buffer[100];
    int len;
    
    len = sprintf(buffer, "\r\n=== Testing All Gestures ===\r\n");
    HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
    
    // Test each gesture with 2-second delay
    GestureType gestures[] = {
        GESTURE_REST,
        GESTURE_ROCK,
        GESTURE_SCISSORS,
        GESTURE_PAPER,
        GESTURE_FUCK,
        GESTURE_THREE,
        GESTURE_FOUR,
        GESTURE_GOOD,
        GESTURE_OKAY,
        GESTURE_FINGER_GUN
    };
    
    const char* gesture_names[] = {
        "REST",
        "ROCK",
        "SCISSORS",
        "PAPER",
        "FUCK",
        "THREE",
        "FOUR",
        "GOOD",
        "OKAY",
        "FINGER_GUN"
    };
    
    for (int i = 0; i < 10; i++) {
        len = sprintf(buffer, "\r\nTesting: %s\r\n", gesture_names[i]);
        HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
        
        execute_gesture(gestures[i]);
        HAL_Delay(2000);
    }
    
    len = sprintf(buffer, "\r\n=== Test Complete ===\r\n");
    HAL_UART_Transmit(&huart1, (uint8_t*)buffer, len, 100);
}